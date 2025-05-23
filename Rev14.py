#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %%capture --no-stderr
# This Jupyter magic command suppresses output from NLTK download.
# If not in Jupyter, remove this line.
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' tokenizer already downloaded.")
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)
    print("NLTK 'punkt' tokenizer downloaded.")

import os
import socket
import hashlib
import time
import logging
import threading
from datetime import datetime, timedelta
from functools import wraps
import json # For Groq API interaction

from flask import Flask, render_template_string, render_template, url_for, redirect, abort, request, jsonify, session, flash
from newsapi import NewsApiClient, newsapi_exception
from newspaper import Article # newspaper3k for article scraping
from dotenv import load_dotenv
from IPython.display import display, HTML # For Jupyter Notebook display (though Flask serves HTML)
from werkzeug.security import generate_password_hash, check_password_hash # For login
import groq # For Groq API

print("Cell 1: Imports executed.")


# In[2]:


from flask import Flask
from jinja2 import DictLoader
import os
from dotenv import load_dotenv
import logging
from newsapi import NewsApiClient
from langchain_groq import ChatGroq # Use LangChain’s Groq wrapper

load_dotenv()
app = Flask(__name__)
# This dictionary will be populated in Cell 5 after templates are defined
template_storage = {}
app.jinja_loader = DictLoader(template_storage)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your_very_secret_key_for_sessions_dev_only')

# --- CONFIGURATION ---
app.config['PER_PAGE'] = 10
# MODIFICATION: Renamed 'All' to 'All Articles' and replaced 'Opinions' with 'My Articles'
app.config['CATEGORIES'] = [
    'All Articles', 'My Articles'
]
# MODIFICATION: Changed the query to be more general for the "Briefly" brand
app.config['NEWS_API_QUERY'] = (
    'technology OR business OR world news OR finance OR innovation'
)
app.config['NEWS_API_DAYS_AGO'] = 3
app.config['NEWS_API_PAGE_SIZE'] = 100
app.config['SUMMARY_SENTENCES'] = 3
app.config['CACHE_EXPIRY_SECONDS'] = 300
app.config['READING_SPEED_WPM'] = 230

# Logging Configuration
if not app.debug:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
app.logger.setLevel(logging.INFO)
app.logger.info("Flask App Configuration Loaded.")

# NewsAPI Configuration
NEWSAPI_KEY = os.environ.get('NEWSAPI_KEY', '9fbbe94dd5114373a8e94cdaa1d7713f') # Replace with your key if needed

if NEWSAPI_KEY:
    newsapi = NewsApiClient(api_key= NEWSAPI_KEY)
    app.logger.info("NewsAPI client initialized.")
else:
    newsapi = None
    app.logger.error(
        "NEWSAPI_KEY is missing from .env file. News fetching will fail."
    )
    print("WARNING: NEWSAPI_KEY is missing. News fetching will fail.")

# Groq API Configuration using LangChain
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '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') # Replace with your key if needed

if GROQ_API_KEY:
    try:
        groq_client = ChatGroq(
            model="llama3-8b-8192",
            groq_api_key=GROQ_API_KEY,
            temperature=0
        )
        app.logger.info("Groq Chat client initialized with LangChain.")
    except Exception as e:
        groq_client = None
        app.logger.error(f"Failed to initialize Groq client via LangChain: {e}")
        print(f"WARNING: Failed to initialize Groq client via LangChain: {e}")
else:
    groq_client = None
    app.logger.warning(
        "GROQ_API_KEY is missing from .env file. Advanced article processing will be disabled."
    )
    print("WARNING: GROQ_API_KEY is missing. Advanced article processing will be disabled.")

print("Cell 2: Initial setup and configuration complete. Jinja DictLoader configured.")


# In[3]:


# Cell 3: Global Stores and Helper Functions
import hashlib # For generating article IDs
import json # For parsing JSON responses from Groq
import time # For caching timestamps and sleep
import socket # For find_free_port
from functools import wraps # For cache decorator
from datetime import datetime, timedelta # For date calculations in news fetching
import urllib.parse # For URL encoding text in placeholder images

# For user authentication
from werkzeug.security import generate_password_hash

from newsapi.newsapi_client import NewsAPIException # Specific exception for NewsAPI
# MODIFICATION: Import Config from newspaper
from newspaper import Article, Config
import nltk # For NLP tasks like sentence tokenization (used in summary fallback)

# LangChain specific imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import LangChainException


# --- NLTK Data Check ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    app.logger.info("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt', quiet=True)
    app.logger.info("'punkt' tokenizer downloaded.")
except Exception as e:
    app.logger.error(f"Error during NLTK 'punkt' check/download: {e}. Some NLP features might fail.")


# --- Global Stores ---
MASTER_ARTICLE_STORE = {}  # Maps article_hash_id to article_data
USER_ADDED_ARTICLES_STORE = [] # Stores user-added articles, newest first
API_CACHE = {} # Simple dictionary cache

# Basic in-memory user store
USERS = {}
if USERS:
    app.logger.info(f"Initial users loaded: {list(USERS.keys())}")
else:
    app.logger.info("User store is initially empty. Register users through the app.")

print("Cell 3: Global stores initialized.")


# --- Helper Functions ---

def generate_article_id(url_or_title):
    """Generates a unique ID for an article based on its URL or title."""
    return hashlib.md5(url_or_title.encode('utf-8')).hexdigest()

def jinja_truncate_filter(s, length=120, killwords=False, end='...'):
    """Jinja filter for truncating strings."""
    if not s: return ''
    if len(s) <= length:
        return s
    if killwords:
        return s[:length - len(end)] + end
    else:
        words = s.split()
        result_words = []
        current_length = 0
        for word in words:
            if current_length + len(word) + (1 if result_words else 0) > length - len(end):
                break
            result_words.append(word)
            current_length += len(word) + (1 if len(result_words) > 1 else 0)

        if not result_words:
            return s[:length - len(end)] + end
        return ' '.join(result_words) + end

# Register the custom filter with Jinja environment
app.jinja_env.filters['truncate'] = jinja_truncate_filter

def simple_cache(expiry_seconds_default=None):
    """A simple decorator for caching function results in memory."""
    actual_expiry_seconds = expiry_seconds_default

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal actual_expiry_seconds
            if actual_expiry_seconds is None:
                current_expiry_setting = app.config.get('CACHE_EXPIRY_SECONDS', 300)
                actual_expiry_seconds_to_use = current_expiry_setting
            else:
                actual_expiry_seconds_to_use = actual_expiry_seconds

            key_parts = [func.__name__] + list(args) + sorted(kwargs.items())
            cache_key = hashlib.md5(str(key_parts).encode('utf-8')).hexdigest()

            cached_entry = API_CACHE.get(cache_key)
            if cached_entry:
                data, timestamp = cached_entry
                if time.time() - timestamp < actual_expiry_seconds_to_use:
                    app.logger.debug(f"Cache HIT for {func.__name__} with key {cache_key[:10]}...")
                    return data
                else:
                    app.logger.debug(f"Cache EXPIRED for {func.__name__} with key {cache_key[:10]}...")

            app.logger.debug(f"Cache MISS for {func.__name__} with key {cache_key[:10]}... Calling function.")
            result = func(*args, **kwargs)
            API_CACHE[cache_key] = (result, time.time())
            return result
        return wrapper
    return decorator


def calculate_read_time(text):
    """Estimates reading time in minutes for a given text."""
    if not text or not isinstance(text, str):
        return 0
    words = text.split()
    num_words = len(words)
    reading_speed_wpm = app.config.get('READING_SPEED_WPM', 230)
    return max(1, round(num_words / reading_speed_wpm))

@simple_cache(expiry_seconds_default=3600 * 6)
def get_article_analysis_with_groq(article_text, article_title=""):
    """
    Uses Groq API to summarize and get takeaways for an article.
    """
    if not groq_client or not article_text:
        app.logger.warning("Groq client not available or no text provided for analysis.")
        return {
            "groq_summary": "Groq summary unavailable (client/text missing).",
            "groq_takeaways": ["Groq takeaways unavailable (client/text missing)."],
            "error": "Groq client not available or no text provided."
        }

    # ... (rest of get_article_analysis_with_groq is unchanged) ...
    app.logger.info(f"Requesting Groq analysis for article (title: {article_title[:50]}...).")
    max_retries = 2
    text_limit = 20000
    truncated_text = article_text[:text_limit]

    system_prompt_content = (
        "You are an expert news analyst. Analyze the following article. "
        "1. Provide a concise summary (around 500 words). "
        "2. List 7-8 key takeaways as bullet points. "
        "Format your entire response as a single JSON object with keys 'summary' (string) and 'takeaways' (a list of strings)."
    )
    user_prompt_content = f"Article Title: {article_title}\n\nArticle Text:\n{truncated_text}"

    lc_messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=user_prompt_content),
    ]

    try:
        json_model = groq_client.bind(
            response_format={"type": "json_object"},
            temperature=0.3
        )
    except Exception as e:
        app.logger.error(f"Failed to bind model with JSON format: {e}")
        return {
            "groq_summary": f"Groq summary unavailable due to a model configuration error: {e}",
            "groq_takeaways": [f"Groq takeaways unavailable due to a model configuration error: {e}"],
            "error": str(e)
        }

    for attempt in range(max_retries):
        try:
            ai_response = json_model.invoke(lc_messages)
            response_content_str = ai_response.content
            app.logger.debug(f"Groq raw JSON response string: {response_content_str[:500]}...")

            try:
                analysis = json.loads(response_content_str)
                if not all(k in analysis for k in ['summary', 'takeaways']):
                    raise ValueError("Missing 'summary' or 'takeaways' key in Groq JSON response.")
                if not isinstance(analysis.get('takeaways'), list):
                    analysis['takeaways'] = [str(analysis['takeaways'])] if analysis.get('takeaways') is not None else ["Takeaways format error."]

                return {
                    "groq_summary": analysis.get("summary", "Summary not generated."),
                    "groq_takeaways": analysis.get("takeaways", ["Takeaways not generated."]),
                    "error": None
                }
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries - 1:
                    return {"groq_summary": "Could not parse summary.", "groq_takeaways": [f"Could not parse takeaways. Error: {e}"], "error": f"JSON parsing error: {e}"}
                time.sleep(1)
        except Exception as e:
            if attempt == max_retries - 1:
                return {"groq_summary": f"Groq summary unavailable: {e}", "groq_takeaways": [f"Groq takeaways unavailable: {e}"], "error": str(e)}
            time.sleep(1)
    return {"groq_summary": "Groq summary unavailable after retries.", "groq_takeaways": ["Groq takeaways unavailable after retries."], "error": "Failed after retries."}

@simple_cache()
def fetch_news_from_api(query=None, category_keyword=None,
                              days_ago=None,
                              page_size=None, lang='en'):
    # ... (this function is unchanged) ...
    if not newsapi:
        app.logger.error("NewsAPI client not available. Cannot fetch news.")
        return []

    final_query = query or app.config['NEWS_API_QUERY']
    final_days_ago = days_ago or app.config['NEWS_API_DAYS_AGO']
    final_page_size = page_size or app.config['NEWS_API_PAGE_SIZE']
    from_date = (datetime.utcnow() - timedelta(days=final_days_ago)).strftime('%Y-%m-%d')
    search_query_parts = [f"({final_query})"]
    if category_keyword and category_keyword.lower() not in ['all', 'all articles']:
        search_query_parts.append(f"({category_keyword})")
    full_search_query = " AND ".join(search_query_parts)
    
    # ... (rest of fetch_news_from_api is unchanged) ...
    try:
        response = newsapi.get_everything(q=full_search_query, from_param=from_date, language=lang, sort_by='publishedAt', page_size=final_page_size)
        raw_articles = response.get('articles', [])
        processed_articles = []
        for art_data in raw_articles:
            if not all(art_data.get(key) for key in ['url', 'title', 'source']) or art_data.get('title') == '[Removed]' or not art_data['source'].get('name'):
                continue
            article_id = generate_article_id(art_data['url'])
            source_name = art_data['source']['name']
            placeholder_image_text = urllib.parse.quote_plus(source_name[:20] if source_name else "News")
            standardized_article = {
                'id': article_id, 'title': art_data.get('title'), 'description': art_data.get('description', ''), 'url': art_data.get('url'),
                'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/0D2C54/FFFFFF?text={placeholder_image_text}',
                'publishedAt': art_data.get('publishedAt', ''), 'source': {'name': source_name, 'id': art_data['source'].get('id')},
                'api_category_keyword': category_keyword or "All Articles",
                'full_text': None, 'newspaper_summary': None, 'read_time_minutes': 0, 'groq_analysis': None
            }
            MASTER_ARTICLE_STORE[article_id] = standardized_article
            processed_articles.append(standardized_article)
        return processed_articles
    except Exception as e:
        app.logger.error(f"NewsAPI Error: {e}")
        return []


@simple_cache(expiry_seconds_default=3600 * 2)
def fetch_process_and_analyze_article_content(article_id, url, title=""):
    """
    Downloads article content using Newspaper3k, parses it, and uses Groq for analysis.
    Updates MASTER_ARTICLE_STORE.
    """
    app.logger.info(f"Fetching, processing, and analyzing article ID: {article_id}, URL: {url[:70]}...")
    summary_sentences_count = app.config.get('SUMMARY_SENTENCES', 3)

    if article_id in MASTER_ARTICLE_STORE:
        cached_master_data = MASTER_ARTICLE_STORE[article_id]
        if cached_master_data.get('full_text') and cached_master_data.get('groq_analysis') and not cached_master_data['groq_analysis'].get('error'):
            app.logger.info(f"Using already fully processed data from MASTER_ARTICLE_STORE for {article_id}")
            return (
                cached_master_data['full_text'],
                cached_master_data.get('newspaper_summary', "Summary not found."),
                cached_master_data.get('read_time_minutes', 0),
                cached_master_data['groq_analysis']
            )

    # --- START OF MODIFICATION TO FIX 406 ERROR ---
    config = Config()
    # Set a realistic User-Agent to mimic a web browser
    config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
    config.request_timeout = 10 # Add a timeout for requests
    # --- END OF MODIFICATION ---

    full_text = "Article text could not be extracted."
    newspaper_summary = "Newspaper3k summary unavailable."
    groq_analysis_result = None
    read_time = 0

    try:
        # MODIFICATION: Pass the config to the Article object
        article_scraper = Article(url, config=config)
        article_scraper.download()
        article_scraper.parse()

        if article_scraper.text:
            full_text = article_scraper.text
            read_time = calculate_read_time(full_text)

            try:
                article_scraper.nlp()
                newspaper_summary = article_scraper.summary.strip()
                if not newspaper_summary:
                    sentences = nltk.sent_tokenize(article_scraper.text)
                    newspaper_summary = ' '.join(sentences[:summary_sentences_count])
            except Exception as nlp_e:
                app.logger.warning(f"Newspaper3k NLP failed for {url}: {nlp_e}")
                sentences = nltk.sent_tokenize(full_text)
                newspaper_summary = ' '.join(sentences[:summary_sentences_count])

        if full_text != "Article text could not be extracted." and groq_client:
            groq_analysis_result = get_article_analysis_with_groq(full_text, title)
        else:
            groq_analysis_result = {"error": "Groq analysis not performed."}
            
    except Exception as e:
        app.logger.error(f"Error processing article {url} with Newspaper3k: {e}")
        # Pass the specific error to the analysis result for better debugging
        error_message = f"Article processing error: {e} on URL {url}"
        full_text = "Article text could not be extracted."
        groq_analysis_result = {
            "groq_summary": f"Groq summary unavailable.",
            "groq_takeaways": [f"Groq takeaways unavailable."],
            "error": error_message
        }
        read_time = 0

    if article_id in MASTER_ARTICLE_STORE:
        MASTER_ARTICLE_STORE[article_id].update({
            'full_text': full_text,
            'newspaper_summary': newspaper_summary,
            'read_time_minutes': read_time,
            'groq_analysis': groq_analysis_result
        })
    else:
        # This case is a fallback
        MASTER_ARTICLE_STORE[article_id] = {
            'id': article_id, 'url': url, 'title': title,
            'full_text': full_text, 'newspaper_summary': newspaper_summary,
            'read_time_minutes': read_time, 'groq_analysis': groq_analysis_result,
            # ... other necessary fields ...
        }

    return full_text, newspaper_summary, read_time, groq_analysis_result


def find_free_port():
    """Finds an available port on the local machine."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port

print("Cell 3: Helper functions defined and updated.")


# In[4]:


# Cell 4: HTML Templates

# Using triple quotes for multiline strings
# NOTE: BASE_HTML_TEMPLATE, INDEX_HTML_TEMPLATE, LOGIN_HTML_TEMPLATE, and REGISTER_HTML_TEMPLATE
# remain the same as the previous turn. The only change is to ARTICLE_HTML_TEMPLATE.

BASE_HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}Briefly{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #0A2342; /* Deep Oxford Blue */
            --primary-light: #1E3A5E;
            --secondary-color: #B8860B; /* DarkGoldenRod - Luxurious accent */
            --secondary-light: #D4A017;
            --accent-color: #F07F2D; /* Bright Orange for specific CTAs if needed */

            --text-color: #343a40;
            --text-muted-color: #6c757d;
            --light-bg: #F8F9FA;
            --white-bg: #FFFFFF;
            --card-border-color: #E0E0E0;
            --footer-bg: #061A30;
            --footer-text: rgba(255,255,255,0.8);
            --footer-link-hover: var(--secondary-color);

            --primary-gradient: linear-gradient(135deg, var(--primary-color), var(--primary-light));

            /* RGB versions for use in rgba() */
            --primary-color-rgb: 10, 35, 66;
            --secondary-color-rgb: 184, 134, 11;
        }

        body {
            padding-top: 145px;
            font-family: 'Roboto', sans-serif; /* Modern body font */
            line-height: 1.65;
            color: var(--text-color);
            background-color: var(--light-bg);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        .main-content { flex-grow: 1; }

        /* Dark Mode Styles */
        body.dark-mode {
            --primary-color: #1E3A5E;
            --primary-light: #2A4B7C;
            --secondary-color: #D4A017;
            --secondary-light: #E7B400;
            --accent-color: #FF983E;
            --text-color: #E9ECEF;
            --text-muted-color: #ADB5BD;
            --light-bg: #121212; /* Dark background */
            --white-bg: #1E1E1E; /* Card background */
            --card-border-color: #333333;
            --footer-bg: #0A0A0A;
            --footer-text: rgba(255,255,255,0.7);
            --primary-color-rgb: 30, 58, 94;
            --secondary-color-rgb: 212, 160, 23;
        }
        body.dark-mode .navbar-main { background: linear-gradient(135deg, #0A1A2F, #10233B); border-bottom: 1px solid #2A4B7C; }
        body.dark-mode .category-nav { background: #1A1A1A; border-bottom: 1px solid #2A2A2A; }
        body.dark-mode .category-link { color: var(--text-muted-color) !important; }
        body.dark-mode .category-link.active { background: var(--primary-color) !important; color: var(--white-bg) !important; }
        body.dark-mode .category-link:hover:not(.active) { background: #2C2C2C !important; color: var(--secondary-color) !important; }
        body.dark-mode .article-card, body.dark-mode .featured-article, body.dark-mode .article-full-content-wrapper, body.dark-mode .auth-container { background-color: var(--white-bg); border-color: var(--card-border-color); }
        body.dark-mode .article-title a, body.dark-mode h1, body.dark-mode h2, body.dark-mode h3, body.dark-mode h4, body.dark-mode h5, body.dark-mode .auth-title { color: var(--text-color) !important; }
        body.dark-mode .article-description, body.dark-mode .meta-item, body.dark-mode .content-text p, body.dark-mode .article-meta-detailed { color: var(--text-muted-color) !important; }
        body.dark-mode .read-more { background: var(--secondary-color); color: #000 !important; }
        body.dark-mode .read-more:hover { background: var(--secondary-light); }
        body.dark-mode .btn-outline-primary { color: var(--secondary-color); border-color: var(--secondary-color); }
        body.dark-mode .btn-outline-primary:hover { background: var(--secondary-color); color: #000; }
        body.dark-mode .modal-content { background-color: var(--white-bg); color: var(--text-color); border-color: var(--card-border-color);}
        body.dark-mode .modal-form-control { background-color: #2C2C2C; color: var(--text-color); border-color: #444; }
        body.dark-mode .modal-form-control::placeholder { color: var(--text-muted-color); }
        body.dark-mode .close-modal { color: var(--text-muted-color); }
        body.dark-mode .close-modal:hover { background: #2C2C2C; color: var(--text-color); }
        body.dark-mode .page-link { background-color: var(--white-bg); border-color: var(--card-border-color); color: var(--secondary-color); }
        body.dark-mode .page-item.active .page-link { background-color: var(--primary-color); border-color: var(--primary-color); color: var(--white-bg); }
        body.dark-mode .page-item.disabled .page-link { background-color: var(--white-bg); color: var(--text-muted-color); }
        body.dark-mode .page-link:hover:not(.active) { background-color: #2C2C2C; }
        body.dark-mode .summary-box { background-color: rgba(var(--secondary-color-rgb), 0.05); border-color: rgba(var(--secondary-color-rgb), 0.2); }
        body.dark-mode .summary-box h5 { color: var(--secondary-light); }
        body.dark-mode .takeaways-box { background-color: rgba(var(--secondary-color-rgb), 0.05); border-left-color: var(--secondary-light); }
        body.dark-mode .takeaways-box h5 { color: var(--secondary-light); }
        body.dark-mode .content-text a {color: var(--secondary-light);}
        body.dark-mode .content-text a:hover {color: var(--accent-color);}
        body.dark-mode .loader { border-top-color: var(--secondary-color); }


        /* Navbar Styles */
        .navbar-main {
            background: var(--primary-gradient);
            padding: 0.8rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-bottom: 2px solid rgba(255,255,255,0.15);
            transition: background 0.3s ease, border-bottom 0.3s ease;
            height: 95px; /* Fixed height for the main navbar */
            display: flex;
            align-items: center;
        }
        .navbar-brand-custom {
            color: white !important; font-weight: 800; font-size: 2.2rem; letter-spacing: 0.5px;
            font-family: 'Poppins', sans-serif; /* Modern font for brand */
            margin-bottom: 0; /* Ensure no extra margin */
            display: flex; align-items: center; gap: 12px;
        }
        .navbar-brand-custom .brand-icon { color: var(--secondary-light); font-size: 2.5rem; }

        .search-form-container { flex-grow: 1; display: flex; justify-content: center; padding: 0 1rem; }
        .search-container { position: relative; width: 100%; max-width: 550px; }
        .navbar-search {
            border-radius: 25px; padding: 0.7rem 1.25rem 0.7rem 2.8rem;
            border: 1px solid rgba(255,255,255,0.2);
            font-size: 0.95rem; transition: all 0.3s ease;
            background: rgba(255,255,255,0.1); color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .navbar-search::placeholder { color: rgba(255,255,255,0.6); }
        .navbar-search:focus {
            background: rgba(255,255,255,0.2);
            box-shadow: 0 0 0 3px rgba(var(--secondary-color-rgb),0.3);
            border-color: var(--secondary-color); outline: none; color:white;
        }
        .search-icon { color: rgba(255,255,255,0.7); transition: all 0.3s ease; left: 1rem; position: absolute; top: 50%; transform: translateY(-50%); }
        .navbar-search:focus + .search-icon { color: var(--secondary-light); }

        .header-controls { display: flex; gap: 0.8rem; align-items: center; }
        .header-btn {
            background: transparent; border: 1px solid rgba(255,255,255,0.3);
            padding: 0.5rem 1rem; border-radius: 20px;
            color: white; font-weight: 500; transition: all 0.3s ease;
            display: flex; align-items: center; gap: 0.5rem; cursor: pointer; text-decoration:none; font-size: 0.9rem;
        }
        .header-btn:hover { background: var(--secondary-color); border-color: var(--secondary-color); color: var(--primary-color); transform: translateY(-1px); }
        .dark-mode-toggle { font-size: 1.1rem; width: 40px; height: 40px; justify-content: center;} /* Ensure icon is centered */

        .category-nav {
            background: var(--white-bg);
            box-shadow: 0 3px 10px rgba(0,0,0,0.03);
            position: fixed; top: 95px; /* Position right below main navbar */
            width: 100%; z-index: 1020; border-bottom: 1px solid var(--card-border-color);
            transition: background 0.3s ease, border-bottom 0.3s ease;
        }
        .categories-wrapper { display: flex; justify-content: center; align-items: center; width: 100%; overflow-x: auto; padding: 0.4rem 0; scrollbar-width: thin; scrollbar-color: var(--secondary-color) var(--light-bg); }
        .categories-wrapper::-webkit-scrollbar { height: 6px; }
        .categories-wrapper::-webkit-scrollbar-thumb { background: var(--secondary-color); border-radius: 3px; }
        .category-link {
            color: var(--primary-color) !important; font-weight: 600;
            padding: 0.6rem 1.3rem !important; border-radius: 20px;
            transition: all 0.25s ease; white-space: nowrap; text-decoration: none; margin: 0 0.3rem;
            font-size: 0.9rem; border: 1px solid transparent;
            font-family: 'Roboto', sans-serif; /* Modern font for categories */
        }
        .category-link.active {
            background: var(--primary-color) !important; color: white !important;
            box-shadow: 0 3px 10px rgba(var(--primary-color-rgb), 0.2);
            border-color: var(--primary-light);
        }
        .category-link:hover:not(.active) { background: var(--light-bg) !important; color: var(--secondary-color) !important; border-color: var(--secondary-color); }

        /* Common Card & Content Styles */
        .article-card, .featured-article, .article-full-content-wrapper, .auth-container {
            background: var(--white-bg); border-radius: 10px;
            transition: all 0.3s ease; border: 1px solid var(--card-border-color);
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        .article-card:hover, .featured-article:hover {
            transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        }
        .article-image-container { height: 200px; overflow: hidden; position: relative; border-top-left-radius: 9px; border-top-right-radius: 9px;}
        .article-image { width: 100%; height: 100%; object-fit: cover; transition: transform 0.4s ease; }
        .article-card:hover .article-image { transform: scale(1.08); }

        .category-tag { position: absolute; top: 10px; left: 10px; background: var(--secondary-color); color: var(--primary-color); font-size: 0.65rem; font-weight: 700; padding: 0.3rem 0.7rem; border-radius: 15px; z-index: 5; text-transform: uppercase; letter-spacing: 0.3px; }
        body.dark-mode .category-tag { color: var(--white-bg); background-color: var(--primary-light); } /* Dark mode tag */

        .article-body { padding: 1.25rem; flex-grow: 1; display: flex; flex-direction: column; }
        .article-title { font-weight: 700; line-height: 1.35; margin-bottom: 0.6rem; font-size:1.1rem; }
        .article-title a { color: var(--primary-color); text-decoration: none; }
        .article-card:hover .article-title a { color: var(--primary-color) !important; } /* Light mode hover */
        body.dark-mode .article-card .article-title a { color: var(--text-color) !important; } /* Dark mode default */
        body.dark-mode .article-card:hover .article-title a { color: var(--secondary-color) !important; } /* Dark mode hover */


        .article-meta { display: flex; align-items: center; margin-bottom: 0.8rem; flex-wrap: wrap; gap: 0.4rem 1rem; }
        .meta-item { display: flex; align-items: center; font-size: 0.8rem; color: var(--text-muted-color); }
        .meta-item i { font-size: 0.9rem; margin-right: 0.3rem; color: var(--secondary-color); }
        .article-description { color: var(--text-muted-color); margin-bottom: 1rem; font-size: 0.9rem; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }

        .read-more {
            margin-top: auto; background: var(--primary-color); color: white !important; border: none;
            padding: 0.5rem 0; border-radius: 6px; font-weight: 600; font-size: 0.85rem;
            transition: all 0.3s ease; width: 100%; text-align: center; text-decoration: none; display:inline-block;
        }
        .read-more:hover { background: var(--primary-light); transform: translateY(-2px); color: white !important; }
        body.dark-mode .read-more { background: var(--secondary-color); color: var(--primary-color) !important;}
        body.dark-mode .read-more:hover { background: var(--secondary-light); }


        /* Pagination */
        .pagination { margin: 2rem 0; display: flex; justify-content: center; gap: 0.3rem; }
        .page-item .page-link {
            border-radius: 50%; width: 40px; height: 40px; display:flex; align-items:center; justify-content:center;
            color: var(--primary-color); border: 1px solid var(--card-border-color);
            font-weight: 600; transition: all 0.2s ease; font-size:0.9rem;
        }
        .page-item .page-link:hover { background-color: var(--light-bg); border-color: var(--secondary-color); color: var(--secondary-color); }
        .page-item.active .page-link { background-color: var(--primary-color); border-color: var(--primary-color); color: white; box-shadow: 0 2px 8px rgba(var(--primary-color-rgb), 0.3); }
        .page-item.disabled .page-link { color: var(--text-muted-color); pointer-events: none; background-color: var(--light-bg); }
        .page-link-prev-next .page-link { width: auto; padding-left:1rem; padding-right:1rem; border-radius:20px; }

        /* Footer */
        footer { background: var(--footer-bg); color: var(--footer-text); margin-top: auto; padding: 3rem 0 1.5rem; font-size:0.9rem; }
        .footer-content { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 2rem; }
        .footer-section h5 { color: var(--secondary-color); margin-bottom: 1rem; font-weight: 700; letter-spacing: 0.3px; position: relative; padding-bottom: 0.6rem; font-size: 1.1rem; }
        .footer-section h5:after { content: ''; position: absolute; left: 0; bottom: 0; width: 35px; height: 2.5px; background: var(--secondary-light); }
        .footer-links { display: flex; flex-direction: column; gap: 0.6rem; }
        .footer-links a { color: var(--footer-text); text-decoration: none; transition: all 0.2s ease; display: flex; align-items: center; gap: 0.4rem; }
        .footer-links a:hover { color: var(--footer-link-hover); transform: translateX(3px); }
        .social-links { display: flex; gap: 0.8rem; margin-top: 0.5rem; }
        .social-links a { color: var(--footer-text); font-size: 1.1rem; transition: all 0.2s ease; width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; border-radius: 50%; background: rgba(255,255,255,0.08); }
        .social-links a:hover { color: var(--primary-color); background: var(--secondary-color); transform: translateY(-3px); }
        .footer-brand-icon { color: var(--secondary-color); font-size: 1.8rem; }
        .footer-brand-text { font-family: 'Poppins', sans-serif; font-weight: 700; color: white; }
        .copyright { text-align: center; padding-top: 1.5rem; margin-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.85rem; color: rgba(255,255,255,0.6); }

        /* Add Article Modal */
        .admin-controls { position: fixed; bottom: 25px; right: 25px; z-index: 1030; }
        .add-article-btn {
            width: 55px; height: 55px; border-radius: 50%; background: var(--secondary-color);
            color: var(--primary-color); border: none; box-shadow: 0 4px 15px rgba(var(--secondary-color-rgb),0.3);
            display: flex; align-items: center; justify-content: center; font-size: 22px; cursor: pointer;
            transition: all 0.3s ease;
        }
        .add-article-btn:hover { transform: translateY(-4px) scale(1.05); box-shadow: 0 7px 20px rgba(var(--secondary-color-rgb),0.4); background: var(--secondary-light); }
        .add-article-modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 2000; background-color: rgba(0, 0, 0, 0.6); backdrop-filter: blur(5px); align-items: center; justify-content: center; }
        .modal-content { width: 90%; max-width: 700px; background: var(--white-bg); border-radius: 10px; padding: 2rem; box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15); position: relative; animation: fadeInUp 0.3s ease-out; max-height: 90vh; overflow-y: auto;}
        .close-modal { position: absolute; top: 12px; right: 12px; font-size: 20px; color: var(--text-muted-color); background: none; border: none; cursor: pointer; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; transition: all 0.2s ease; }
        .close-modal:hover { background: var(--light-bg); color: var(--text-color); }
        .modal-form-group { margin-bottom: 1.2rem; }
        .modal-form-group label { display: block; margin-bottom: 0.4rem; font-weight: 600; color: var(--text-color); font-size:0.9rem; }
        .modal-form-control { width: 100%; padding: 0.65rem 0.9rem; border-radius: 6px; border: 1px solid var(--card-border-color); font-size: 0.95rem; transition: all 0.2s ease; background-color: var(--light-bg); }
        .modal-form-control:focus { border-color: var(--primary-color); box-shadow: 0 0 0 3px rgba(var(--primary-color-rgb),0.15); outline: none; background-color: var(--white-bg); }
        .modal-title {font-weight: 700; color: var(--primary-color); margin-bottom: 1.5rem !important;}
        .btn-primary-modal { background-color: var(--primary-color); border-color: var(--primary-color); color:white; padding: 0.6rem 1.2rem; font-weight:600; }
        .btn-primary-modal:hover { background-color: var(--primary-light); border-color: var(--primary-light); }
        .btn-outline-secondary-modal { padding: 0.6rem 1.2rem; font-weight:600; border-color: var(--text-muted-color); color: var(--text-muted-color); }
        body.dark-mode .btn-outline-secondary-modal { border-color: var(--text-muted-color); color: var(--text-muted-color); }
        body.dark-mode .btn-outline-secondary-modal:hover { background-color: #333; color: var(--text-color); border-color: #444;}

        /* Alert Styling */
        .alert-top { position: fixed; top: 85px; left: 50%; transform: translateX(-50%); z-index: 2050; min-width:320px; text-align:center; box-shadow: 0 3px 10px rgba(0,0,0,0.1);}

        /* Utility & Animation */
        .animate-fade-in { animation: fadeIn 0.5s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(25px); } to { opacity: 1; transform: translateY(0); } }
        .fade-in-delay-1 { animation-delay: 0.1s; } .fade-in-delay-2 { animation-delay: 0.2s; } .fade-in-delay-3 { animation-delay: 0.3s; }

        /* Responsive Adjustments */
        .navbar-content-wrapper { display: flex; justify-content: space-between; align-items: center; width: 100%; }
        @media (max-width: 991.98px) { /* md breakpoint */
            body { padding-top: 185px; }
            .navbar-main { padding-bottom: 0.5rem; height: auto;} /* Allow height to adjust */
            .navbar-content-wrapper { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
            .navbar-brand-custom { margin-bottom: 0.5rem; }
            .search-form-container { width: 100%; order: 3; margin-top:0.5rem; padding: 0; }
            .header-controls { position: absolute; top: 0.9rem; right: 1rem; order: 2; }
            .category-nav { top: 125px; /* Adjusted top for category nav, below expanded main nav */ }
        }
        @media (max-width: 767.98px) { /* sm breakpoint */
            body { padding-top: 175px; }
            .category-nav { top: 125px; }
            .featured-article .row { flex-direction: column; }
            .featured-image { margin-bottom: 1rem; height: 250px; }
        }
        @media (max-width: 575.98px) { /* xs breakpoint */
            .navbar-brand-custom { font-size: 1.8rem;} /* Slightly smaller on very small screens */
            .header-controls { gap: 0.3rem; }
            .header-btn { padding: 0.4rem 0.8rem; font-size: 0.8rem; }
            .dark-mode-toggle { font-size: 1rem; }
        }

        /* Styles for Login/Auth pages */
        .auth-container { max-width: 450px; margin: 3rem auto; padding: 2rem; }
        .auth-title { text-align: center; color: var(--primary-color); margin-bottom: 1.5rem; font-weight: 700;}
        body.dark-mode .auth-title { color: var(--secondary-color); }

    </style>
    {% block head_extra %}{% endblock %}
</head>
<body class="{{ request.cookies.get('darkMode', 'disabled') }}"> {# Apply theme from cookie #}
    <div id="alert-placeholder">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                <div class="alert alert-{{ category }} alert-dismissible fade show alert-top" role="alert">
                    <span>{{ message }}</span>
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
    </div>

    <nav class="navbar navbar-main navbar-expand-lg fixed-top">
        <div class="container">
            <div class="navbar-content-wrapper">
                <a class="navbar-brand-custom animate-fade-in" href="{{ url_for('index') }}">
                    <i class="fas fa-bolt-lightning brand-icon"></i>
                    <span>Briefly</span>
                </a>

                <div class="search-form-container">
                    <form action="{{ url_for('search_results', page=1) }}" method="GET" class="search-container animate-fade-in fade-in-delay-1">
                        <input type="search" name="query" class="form-control navbar-search" placeholder="Search news articles..." value="{{ request.args.get('query', '') }}">
                        <i class="fas fa-search search-icon"></i>
                        <button type="submit" class="d-none">Search</button>
                    </form>
                </div>

                <div class="header-controls animate-fade-in fade-in-delay-2">
                    <button class="header-btn dark-mode-toggle" aria-label="Toggle dark mode" title="Toggle Dark Mode">
                        <i class="fas fa-moon"></i>
                    </button>
                    {% if session.get('user_id') %}
                    <span class="text-white me-2 d-none d-md-inline">Hi, {{ session.get('user_name', 'User')|truncate(15) }}!</span>
                    <a href="{{ url_for('logout') }}" class="header-btn" title="Logout">
                        <i class="fas fa-sign-out-alt"></i> <span class="d-none d-sm-inline">Logout</span>
                    </a>
                    {% else %}
                    <a href="{{ url_for('login') }}" class="header-btn" title="Login/Register">
                        <i class="fas fa-user"></i> <span class="d-none d-sm-inline">Login</span>
                    </a>
                    {% endif %}
                </div>
            </div>
        </div>
    </nav>

    <nav class="navbar navbar-expand-lg category-nav">
        <div class="container">
            <div class="categories-wrapper">
                {% for cat_item in categories %}
                    {# Only show 'My Articles' if the user is logged in #}
                    {% if cat_item != 'My Articles' or session.get('user_id') %}
                    <a href="{{ url_for('index', category_name=cat_item, page=1) }}"
                       class="category-link {% if selected_category == cat_item %}active{% endif %}">
                        <i class="fas fa-{% if cat_item == 'All Articles' %}globe-americas{% elif cat_item == 'My Articles' %}feather-alt{% endif %} me-1 d-none d-sm-inline"></i>
                        {{ cat_item }}
                    </a>
                    {% endif %}
                {% endfor %}
            </div>
        </div>
    </nav>

    <main class="container main-content my-4">
        {% block content %}{% endblock %}
    </main>

    {% if session.get('user_id') %} {# Show Add Article button only if logged in #}
    <div class="admin-controls">
        <button class="add-article-btn" id="addArticleBtn" title="Add Custom Article">
            <i class="fas fa-plus"></i>
        </button>
    </div>

    <div class="add-article-modal" id="addArticleModal">
        <div class="modal-content">
            <button class="close-modal" id="closeModalBtn" title="Close Modal"><i class="fas fa-times"></i></button>
            <h3 class="modal-title">Add New Custom Article</h3>
            <form id="addArticleForm">
                <div class="modal-form-group">
                    <label for="articleTitle">Article Title</label>
                    <input type="text" id="articleTitle" name="title" class="modal-form-control" placeholder="Enter article title" required>
                </div>
                <div class="modal-form-group">
                    <label for="articleDescription">Short Description / Summary</label>
                    <textarea id="articleDescription" name="description" class="modal-form-control" rows="3" placeholder="Brief summary of the article" required></textarea>
                </div>
                <div class="modal-form-group">
                    <label for="articleSource">Source Name (e.g., Your Blog, Company News)</label>
                    <input type="text" id="articleSource" name="sourceName" class="modal-form-control" placeholder="Source of this article" value="My Publication" required>
                </div>

                <div class="modal-form-group">
                    <label for="articleImage">Featured Image URL (Optional)</label>
                    <input type="url" id="articleImage" name="imageUrl" class="modal-form-control" placeholder="https://example.com/image.jpg">
                </div>
                <div class="modal-form-group">
                    <label for="articleContent">Full Article Content</label>
                    <textarea id="articleContent" name="content" class="modal-form-control" rows="7" placeholder="Write the full article content here..." required></textarea>
                </div>
                <div class="d-flex justify-content-end gap-2">
                    <button type="button" class="btn btn-outline-secondary-modal" id="cancelArticleBtn">Cancel</button>
                    <button type="submit" class="btn btn-primary-modal">Save Article</button>
                </div>
            </form>
        </div>
    </div>
    {% endif %}

    <footer class="mt-auto">
        <div class="container">
            <div class="footer-content">
                <div class="footer-section">
                    <div class="d-flex align-items-center mb-2">
                        <i class="fas fa-bolt-lightning footer-brand-icon me-2"></i>
                        <span class="h5 mb-0 footer-brand-text">Briefly</span>
                    </div>
                    <p class="small">Your premier source for news, summarized.</p>
                    <div class="social-links">
                        <a href="#" title="Twitter"><i class="fab fa-twitter"></i></a>
                        <a href="#" title="Facebook"><i class="fab fa-facebook-f"></i></a>
                        <a href="#" title="LinkedIn"><i class="fab fa-linkedin-in"></i></a>
                        <a href="#" title="Instagram"><i class="fab fa-instagram"></i></a>
                    </div>
                </div>
                <div class="footer-section">
                    <h5>Quick Links</h5>
                    <div class="footer-links">
                        <a href="{{ url_for('index') }}"><i class="fas fa-angle-right"></i> Home</a>
                        <a href="#"><i class="fas fa-angle-right"></i> About Us (Example)</a>
                        <a href="#"><i class="fas fa-angle-right"></i> Contact (Example)</a>
                        <a href="#"><i class="fas fa-angle-right"></i> Privacy Policy (Example)</a>
                    </div>
                </div>
                <div class="footer-section">
                    <h5>Categories</h5>
                    <div class="footer-links">
                         {% for cat_item in categories %}
                            {% if cat_item != 'My Articles' or session.get('user_id') %}
                                <a href="{{ url_for('index', category_name=cat_item, page=1) }}"><i class="fas fa-angle-right"></i> {{ cat_item }}</a>
                            {% endif %}
                        {% endfor %}
                    </div>
                </div>
                <div class="footer-section">
                    <h5>Newsletter</h5>
                    <p class="small">Subscribe for weekly updates (Feature not implemented).</p>
                    <form class="mt-2">
                        <div class="input-group">
                            <input type="email" class="form-control form-control-sm" placeholder="Your Email" aria-label="Your Email" disabled>
                            <button class="btn btn-sm btn-primary-modal" type="submit" disabled>Subscribe</button>
                        </div>
                    </form>
                </div>
            </div>
            <div class="copyright">&copy; {{ current_year if current_year else namespace(current_year=2024).current_year }} Briefly. All rights reserved.</div>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
    document.addEventListener('DOMContentLoaded', function () {
        // Dark Mode Toggle
        const darkModeToggle = document.querySelector('.dark-mode-toggle');
        const body = document.body;

        function updateThemeIcon() {
            if(darkModeToggle) {
                darkModeToggle.innerHTML = body.classList.contains('dark-mode') ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
            }
        }

        function applyTheme(theme) {
            if (theme === 'enabled') {
                body.classList.add('dark-mode');
            } else {
                body.classList.remove('dark-mode');
            }
            updateThemeIcon();
            // Persist choice
            localStorage.setItem('darkMode', theme); // localStorage for immediate effect on next load
            document.cookie = "darkMode=" + theme + ";path=/;max-age=" + (60*60*24*365); // Cookie for server-side rendering consistency
        }

        if(darkModeToggle) {
            darkModeToggle.addEventListener('click', () => {
                applyTheme(body.classList.contains('dark-mode') ? 'disabled' : 'enabled');
            });
        }

        const storedTheme = localStorage.getItem('darkMode');
        if (storedTheme) {
            applyTheme(storedTheme);
        } else {
            updateThemeIcon();
        }


        // Add Article Modal Logic
        const addArticleBtn = document.getElementById('addArticleBtn');
        const addArticleModal = document.getElementById('addArticleModal');
        const closeModalBtn = document.getElementById('closeModalBtn');
        const cancelArticleBtn = document.getElementById('cancelArticleBtn');
        const addArticleForm = document.getElementById('addArticleForm');

        if(addArticleBtn && addArticleModal) {
            addArticleBtn.addEventListener('click', () => {
                addArticleModal.style.display = 'flex';
                body.style.overflow = 'hidden';
            });

            const closeModalFunction = () => {
                addArticleModal.style.display = 'none';
                if(addArticleForm) addArticleForm.reset();
                body.style.overflow = 'auto';
            };

            if(closeModalBtn) closeModalBtn.addEventListener('click', closeModalFunction);
            if(cancelArticleBtn) cancelArticleBtn.addEventListener('click', closeModalFunction);
            addArticleModal.addEventListener('click', (e) => {
                if (e.target === addArticleModal) closeModalFunction();
            });

            if(addArticleForm) addArticleForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                const submitButton = addArticleForm.querySelector('button[type="submit"]');
                submitButton.disabled = true;
                submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Saving...';

                const formData = new FormData(addArticleForm);
                const articleData = {};
                formData.forEach((value, key) => { articleData[key] = value; });

                articleData.imageUrl = articleData.imageUrl || 'https://via.placeholder.com/700x350/0D2C54/FFFFFF?text=Custom+Article';

                try {
                    const response = await fetch("{{ url_for('add_article') }}", {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(articleData),
                    });
                    const result = await response.json();
                    if (response.ok) {
                        showAlert(result.message || 'Article added successfully!', 'success');
                        closeModalFunction();
                        if (result.redirect_url) {
                            setTimeout(() => window.location.href = result.redirect_url, 1500);
                        } else {
                            setTimeout(() => window.location.reload(), 1500);
                        }
                    } else {
                        showAlert('Error: ' + (result.error || 'Could not add article.'), 'danger');
                    }
                } catch (error) {
                    console.error('Form submission error:', error);
                    showAlert('Client-side error: Could not submit form.', 'danger');
                } finally {
                    submitButton.disabled = false;
                    submitButton.textContent = 'Save Article';
                }
            });
        }

        // Dynamic Alert Function
        function showAlert(message, type = 'info', duration = 7000) {
            const alertPlaceholder = document.getElementById('alert-placeholder');
            if (!alertPlaceholder) {
                console.warn("Alert placeholder not found in DOM for message:", message);
                return;
            }

            const wrapper = document.createElement('div');
            wrapper.innerHTML = [
                '<div class="alert alert-' + type + ' alert-dismissible fade show alert-top" role="alert">',
                '   <span>' + message + '</span>',
                '   <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>',
                '</div>'
            ].join('');
            const alertElement = wrapper.firstChild;
            alertPlaceholder.append(alertElement);

            if (alertElement) {
                setTimeout(() => {
                    const bsAlert = bootstrap.Alert.getOrCreateInstance(alertElement);
                    if (bsAlert) bsAlert.close();
                }, duration);
            }
        }
        window.showAlert = showAlert;

        // Auto-dismiss flashed messages from server
        const flashedAlerts = document.querySelectorAll('#alert-placeholder .alert');
        flashedAlerts.forEach(function(alert) {
            setTimeout(function() {
                const bsAlert = bootstrap.Alert.getOrCreateInstance(alert);
                if (bsAlert) bsAlert.close();
            }, 7000);
        });

    });
    </script>
    {% block scripts_extra %}{% endblock %}
</body>
</html>
'''

INDEX_HTML_TEMPLATE = '''
{% extends "BASE_HTML_TEMPLATE" %}

{% block title %}
    {% if query %}Search: {{ query|truncate(30) }}{% elif selected_category %}{{selected_category}}{% else %}Home{% endif %} - Briefly
{% endblock %}

{% block content %}
    {# Featured Article Section (Only on Page 1 of 'All Articles' or 'My Articles' when not searching) #}
    {% if articles and articles[0] and featured_article_on_this_page %}
    <article class="featured-article p-md-4 p-3 mb-4 animate-fade-in">
        <div class="row g-0 g-md-4">
            <div class="col-lg-6">
                <div class="featured-image rounded overflow-hidden shadow-sm" style="height:320px;">
                    <a href="{{ url_for('article_detail', article_id=articles[0].id) }}">
                    <img src="{{ articles[0].urlToImage or 'https://via.placeholder.com/700x350/0A2342/FFFFFF?text=Featured' }}"
                             class="img-fluid w-100 h-100" style="object-fit:cover;"
                             alt="Featured: {{ articles[0].title|truncate(50) }}">
                    </a>
                </div>
            </div>
            <div class="col-lg-6 d-flex flex-column ps-lg-3 pt-3 pt-lg-0">
                <div class="article-meta mb-2">
                    <span class="badge bg-primary me-2" style="font-size:0.75rem;"> {{ articles[0].source.name | truncate(25) }}</span>
                    <span class="meta-item">
                        <i class="far fa-calendar-alt"></i>
                        {{ articles[0].publishedAt.split('T')[0] if articles[0].publishedAt else 'N/A' }}
                    </span>
                    {% if articles[0].read_time_minutes and articles[0].read_time_minutes > 0 %}
                        <span class="meta-item"><i class="far fa-clock"></i> {{ articles[0].read_time_minutes }} min read</span>
                    {% endif %}
                    {% set f_display_cat = articles[0].groq_analysis.groq_category if articles[0].groq_analysis and articles[0].groq_analysis.groq_category and articles[0].groq_analysis.groq_category not in ['General', 'General (Parsing Error)'] else articles[0].category %}
                    {% if f_display_cat and f_display_cat not in ['All Articles', 'General', 'General (Parsing Error)', 'Search Result', 'My Articles'] %}
                            <span class="meta-item"><i class="fas fa-tag"></i> {{ f_display_cat }}</span>
                    {% endif %}
                </div>

                <h2 class="mb-2 h4">
                    <a href="{{ url_for('article_detail', article_id=articles[0].id) }}" class="text-decoration-none article-title">
                        {{ articles[0].title }}
                    </a>
                </h2>

                <p class="article-description flex-grow-1 small">
                    {# Prioritize Groq summary if available and valid #}
                    {% if articles[0].groq_analysis and articles[0].groq_analysis.groq_summary and "unavailable" not in articles[0].groq_analysis.groq_summary|lower and "not generated" not in articles[0].groq_analysis.groq_summary|lower and "could not parse" not in articles[0].groq_analysis.groq_summary|lower %}
                        {{ articles[0].groq_analysis.groq_summary|truncate(220) }}
                    {% elif articles[0].description %}
                        {{ articles[0].description|truncate(220) }}
                    {% else %}
                        No summary available for this article.
                    {% endif %}
                </p>
                <a href="{{ url_for('article_detail', article_id=articles[0].id) }}" class="read-more mt-auto align-self-start py-2 px-3" style="width:auto;">
                    Read Full Article <i class="fas fa-arrow-right ms-1 small"></i>
                </a>
            </div>
        </div>
    </article>
    {% elif not articles and request.endpoint != 'search_results' and selected_category != "My Articles" %}
        <div class="alert alert-warning text-center my-4 p-3 small">No articles found for '{{selected_category}}'. Try 'All Articles' or check back later.</div>
    {% elif not articles and selected_category == "My Articles" %}
        <div class="alert alert-info text-center my-4 p-3">
            <h4><i class="fas fa-feather-alt me-2"></i>No Articles Penned Yet</h4>
            <p>You haven't added any articles. {% if session.get('user_id') %}Click the '+' button to share your insights!{% else %}Login to add your articles.{% endif %}</p>
        </div>
    {% endif %}

    {# Regular Article Grid #}
    {% set articles_to_display = (articles[1:] if featured_article_on_this_page and articles else articles) %}
    {% if articles_to_display %}
    <div class="article-grid mt-4 row gx-3 gy-4"> {# Bootstrap row for grid structure #}
        {% for art in articles_to_display %}
        <div class="col-md-6 col-lg-4 d-flex"> {# Bootstrap column classes #}
        <article class="article-card animate-fade-in w-100" style="animation-delay: {{ loop.index0 * 0.05 }}s">
            <div class="article-image-container">
                {% set display_category = art.groq_analysis.groq_category if art.groq_analysis and art.groq_analysis.groq_category and art.groq_analysis.groq_category not in ['General', 'General (Parsing Error)'] else art.category %}
                {% if display_category and display_category not in ['All Articles', 'General', 'General (Parsing Error)', 'Search Result', 'My Articles'] %}
                <span class="category-tag">
                    {{ display_category | truncate(15) }}
                </span>
                {% endif %}
                <a href="{{ url_for('article_detail', article_id=art.id) }}">
                    <img src="{{ art.urlToImage or 'https://via.placeholder.com/400x220/0A2342/FFFFFF?text=News' }}"
                         class="article-image"
                         alt="{{ art.title|truncate(50) }}">
                </a>
            </div>
            <div class="article-body">
                <h5 class="article-title mb-2">
                    <a href="{{ url_for('article_detail', article_id=art.id) }}" class="text-decoration-none">
                        {{ art.title|truncate(70) }}
                    </a>
                </h5>

                <div class="article-meta small">
                    <span class="meta-item text-muted">
                        <i class="fas fa-building"></i> {{ art.source.name | truncate(20) }}
                    </span>
                    <span class="meta-item text-muted">
                        <i class="far fa-calendar-alt"></i> {{ art.publishedAt.split('T')[0] if art.publishedAt else 'N/A' }}
                    </span>
                </div>

                <p class="article-description my-2 small">
                    {% if art.groq_analysis and art.groq_analysis.groq_summary and "unavailable" not in art.groq_analysis.groq_summary|lower and "not generated" not in art.groq_analysis.groq_summary|lower and "could not parse" not in art.groq_analysis.groq_summary|lower %}
                        {{ art.groq_analysis.groq_summary|truncate(100) }}
                    {% elif art.description %}
                        {{ art.description|truncate(100) }}
                    {% else %}
                        Click to read more.
                    {% endif %}
                </p>
                <a href="{{ url_for('article_detail', article_id=art.id) }}" class="read-more btn btn-sm">
                    Read More <i class="fas fa-chevron-right ms-1 small"></i>
                </a>
            </div>
        </article>
        </div>
        {% endfor %}
    </div>
    {% elif not articles and request.endpoint == 'search_results' and request.args.get('query') %}
        <div class="alert alert-info text-center my-5 p-4">
            <h4><i class="fas fa-search me-2"></i>No results for "{{ request.args.get('query') }}"</h4>
            <p>Try different keywords or browse categories.</p>
        </div>
    {% endif %}

    {# Pagination Controls #}
    {% if total_pages > 1 %}
    <nav aria-label="Page navigation" class="mt-4">
        <ul class="pagination justify-content-center">
            {# Previous Page Link #}
            <li class="page-item page-link-prev-next {% if current_page == 1 %}disabled{% endif %}">
                <a class="page-link" href="{{ url_for(request.endpoint, page=current_page-1, category_name=selected_category if request.endpoint != 'search_results' else None, query=request.args.get('query')) if current_page > 1 else '#' }}" aria-label="Previous">
                    <span aria-hidden="true">&laquo;</span> Prev
                </a>
            </li>

            {# Page Number Links - Smart display logic #}
            {% set page_window = 1 %} {# Number of pages to show around current page #}
            {% set show_first = 1 %}
            {% set show_last = total_pages %}

            {# First page and ellipsis if needed #}
            {% if current_page - page_window > show_first %}
                <li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=1, category_name=selected_category if request.endpoint != 'search_results' else None, query=request.args.get('query')) }}">1</a></li>
                {% if current_page - page_window > show_first + 1 %}
                    <li class="page-item disabled"><span class="page-link">...</span></li>
                {% endif %}
            {% endif %}

            {# Pages around current page #}
            {% for p in range(1, total_pages + 1) %}
                {% if p == current_page %}
                    <li class="page-item active" aria-current="page"><span class="page-link">{{ p }}</span></li>
                {% elif p >= current_page - page_window and p <= current_page + page_window %}
                    <li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=p, category_name=selected_category if request.endpoint != 'search_results' else None, query=request.args.get('query')) }}">{{ p }}</a></li>
                {% endif %}
            {% endfor %}

            {# Last page and ellipsis if needed #}
            {% if current_page + page_window < show_last %}
                {% if current_page + page_window < show_last - 1 %}
                     <li class="page-item disabled"><span class="page-link">...</span></li>
                {% endif %}
                <li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=total_pages, category_name=selected_category if request.endpoint != 'search_results' else None, query=request.args.get('query')) }}">{{ total_pages }}</a></li>
            {% endif %}

            {# Next Page Link #}
            <li class="page-item page-link-prev-next {% if current_page == total_pages %}disabled{% endif %}">
                <a class="page-link" href="{{ url_for(request.endpoint, page=current_page+1, category_name=selected_category if request.endpoint != 'search_results' else None, query=request.args.get('query')) if current_page < total_pages else '#' }}" aria-label="Next">
                    Next <span aria-hidden="true">&raquo;</span>
                </a>
            </li>
        </ul>
    </nav>
    {% endif %}
{% endblock %}
'''

ARTICLE_HTML_TEMPLATE = '''
{% extends "BASE_HTML_TEMPLATE" %}

{% block title %}{{ article.title|truncate(50) if article else "Article" }} - Briefly{% endblock %}

{% block head_extra %}
<style>
    .article-full-content-wrapper {
        background-color: var(--white-bg); padding: 2rem; border-radius: 10px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.07); margin-bottom: 2rem; margin-top: 1rem;
    }
    .article-full-content-wrapper .main-article-image {
        width: 100%; max-height: 480px; object-fit: cover; border-radius: 8px; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .article-title-main {font-weight: 700; color: var(--primary-color); line-height:1.3; font-family: 'Poppins', sans-serif;}
    body.dark-mode .article-title-main { color: var(--text-color); }

    .article-meta-detailed {
        font-size: 0.85rem; color: var(--text-muted-color); margin-bottom: 1.5rem;
        display:flex; flex-wrap:wrap; gap: 0.5rem 1.2rem; align-items:center; border-bottom: 1px solid var(--card-border-color); padding-bottom:1rem;
    }
    body.dark-mode .article-meta-detailed { color: var(--text-muted-color); border-bottom-color: var(--card-border-color); }

    .article-meta-detailed .meta-item i { color: var(--secondary-color); margin-right: 0.4rem; font-size:0.95rem; }

    .summary-box { background-color: var(--light-bg); padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; border: 1px solid var(--card-border-color); }
    .summary-box h5 { color: var(--primary-color); font-weight: 600; margin-bottom: 0.75rem; font-size:1.1rem; }
    .summary-box p {font-size:0.95rem; line-height:1.6; color: var(--text-color);}
    body.dark-mode .summary-box { background-color: rgba(var(--secondary-color-rgb, 212, 160, 23), 0.05); border-color: rgba(var(--secondary-color-rgb, 212, 160, 23), 0.2); }
    body.dark-mode .summary-box h5 { color: var(--secondary-light); }
    body.dark-mode .summary-box p { color: var(--text-muted-color); }


    .takeaways-box { margin: 1.5rem 0; padding: 1.5rem; border-left: 4px solid var(--secondary-color); background-color: var(--light-bg); border-radius: 0 8px 8px 0;}
    .takeaways-box h5 { color: var(--primary-color); font-weight: 600; margin-bottom: 0.75rem; font-size:1.1rem; }
    .takeaways-box ul { padding-left: 1.2rem; margin-bottom:0; list-style-type: disc; color: var(--text-color); }
    .takeaways-box ul li { margin-bottom: 0.5rem; font-size:0.95rem; }
    body.dark-mode .takeaways-box { background-color: rgba(var(--secondary-color-rgb, 212, 160, 23), 0.05); border-left-color: var(--secondary-light); }
    body.dark-mode .takeaways-box h5 { color: var(--secondary-light); }
    body.dark-mode .takeaways-box ul { color: var(--text-muted-color); }

    .article-source-link { display: inline-block; font-weight: 500; }

    .loader-container {
        display: flex; justify-content: center; align-items: center;
        min-height: 200px; padding: 2rem; font-size: 1rem; color: var(--text-muted-color);
    }
    .loader {
        border: 5px solid var(--light-bg); border-top: 5px solid var(--primary-color);
        border-radius: 50%; width: 50px; height: 50px;
        animation: spin 1s linear infinite; margin-right: 10px;
    }
    body.dark-mode .loader { border-top-color: var(--secondary-color); }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
{% endblock %}

{% block content %}
    {% if article %}
    <article class="article-full-content-wrapper animate-fade-in">
        <h1 class="mb-2 article-title-main display-6">{{ article.title }}</h1>
        <div class="article-meta-detailed">
            <span class="meta-item" title="Source"><i class="fas fa-building"></i> {{ article.source.name }}</span>
            <span class="meta-item" title="Published Date"><i class="far fa-calendar-alt"></i> {{ article.publishedAt.split('T')[0] if article.publishedAt else 'N/A' }}</span>
            <span class="meta-item" title="Estimated Reading Time" id="articleReadTimeMeta" {% if not (read_time_minutes and read_time_minutes > 0) %} style="display:none;" {% endif %}>
                <i class="far fa-clock"></i> <span id="articleReadTimeText">{{ read_time_minutes if read_time_minutes and read_time_minutes > 0 else '' }}</span> min read
            </span>
        </div>

        {% if article.urlToImage %}
        <img src="{{ article.urlToImage }}" alt="{{ article.title|truncate(50) }}" class="main-article-image">
        {% endif %}

        <div id="contentLoader" class="loader-container my-4">
            <div class="loader"></div> Loading analysis...
        </div>

        <div id="articleAnalysisContainer" style="display: none;">
            <div id="articleAnalysisSection"></div>

            <div id="originalLinkContainer" class="mt-4"></div>
        </div>

    </article>
    {% else %}
    <div class="alert alert-danger text-center my-5 p-4">
        <h4><i class="fas fa-exclamation-triangle me-2"></i>Article Not Found</h4>
        <p>The article you are looking for could not be found or is no longer available.</p>
        <a href="{{ url_for('index') }}" class="btn btn-primary mt-2">Go to Homepage</a>
    </div>
    {% endif %}
{% endblock %}

{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    const articleId = {{ article.id | tojson | safe if article else 'null' }};
    const articleUrlForJs = {{ article.url | tojson | safe if article and article.url != '#' and not article.url.startswith('#user-added') else 'null' }};
    const articleSourceForJs = {{ article.source.name | tojson | safe if article and article.source and article.source.name else 'null' }};

    const contentLoader = document.getElementById('contentLoader');
    const articleAnalysisContainer = document.getElementById('articleAnalysisContainer');
    const analysisSection = document.getElementById('articleAnalysisSection');
    const originalLinkContainer = document.getElementById('originalLinkContainer'); // MODIFICATION: Target new container
    const readTimeMetaSpan = document.getElementById('articleReadTimeMeta');
    const readTimeTextSpan = document.getElementById('articleReadTimeText');

    if (!articleId) {
        if(contentLoader) contentLoader.style.display = 'none';
        return;
    }

    // Function to create and append the original article link
    function appendOriginalLink(container, url, sourceName) {
        if (container && url && sourceName) {
            container.innerHTML = ''; // Clear previous content
            const link = document.createElement('a');
            link.href = url;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.className = 'btn btn-outline-primary article-source-link';
            link.innerHTML = `Read Original at ${sourceName} <i class="fas fa-external-link-alt ms-1"></i>`;
            container.appendChild(link);
        }
    }

    fetch(`{{ url_for('get_article_content_json', article_id='ARTICLE_ID_PLACEHOLDER') }}`.replace('ARTICLE_ID_PLACEHOLDER', articleId))
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { throw new Error(err.error || `HTTP error! status: ${response.status}`) });
            }
            return response.json();
        })
        .then(data => {
            if (contentLoader) contentLoader.style.display = 'none';
            if (articleAnalysisContainer) articleAnalysisContainer.style.display = 'block';

            if (data.error) {
                analysisSection.innerHTML = `<div class="alert alert-danger small p-2">Could not load article analysis: ${data.error}</div>`;
                return;
            }

            // Update read time
            if (readTimeMetaSpan && readTimeTextSpan) {
                if (data.read_time_minutes && data.read_time_minutes > 0) {
                    readTimeTextSpan.textContent = data.read_time_minutes;
                    readTimeMetaSpan.style.display = 'inline-flex';
                } else {
                    readTimeMetaSpan.style.display = 'none';
                }
            }

            // Update analysis section (Summary and Takeaways)
            if (analysisSection && data.groq_analysis) {
                let newAnalysisHtml = '';
                const groqData = data.groq_analysis;

                const hasValidGroqSummary = groqData.groq_summary && !groqData.groq_summary.toLowerCase().includes("unavailable");
                if (hasValidGroqSummary) {
                    newAnalysisHtml += `
                    <div class="summary-box my-3">
                        <h5><i class="fas fa-bookmark me-2"></i>Article Summary (AI Enhanced)</h5>
                        <p class="mb-0">${groqData.groq_summary}</p>
                    </div>`;
                }

                const hasValidGroqTakeaways = groqData.groq_takeaways && groqData.groq_takeaways.length > 0 && !groqData.groq_takeaways[0].toLowerCase().includes("unavailable");
                if (hasValidGroqTakeaways) {
                    newAnalysisHtml += `
                    <div class="takeaways-box my-3">
                        <h5><i class="fas fa-list-check me-2"></i>Key Takeaways (AI Enhanced)</h5>
                        <ul>${groqData.groq_takeaways.map(takeaway => `<li>${takeaway}</li>`).join('')}</ul>
                    </div>`;
                }

                if (!hasValidGroqSummary && !hasValidGroqTakeaways) {
                    newAnalysisHtml = `<div class="alert alert-secondary small p-2 mt-3">No AI-generated summary or takeaways are available for this article.</div>`;
                }
                analysisSection.innerHTML = newAnalysisHtml;
            }

            // MODIFICATION: Always try to add the original link if the URL exists
            if (articleUrlForJs) {
                appendOriginalLink(originalLinkContainer, articleUrlForJs, articleSourceForJs);
            }
        })
        .catch(error => {
            console.error("Error fetching article content:", error);
            if (contentLoader) contentLoader.style.display = 'none';
            if (articleAnalysisContainer) articleAnalysisContainer.style.display = 'block';
            analysisSection.innerHTML = `<div class="alert alert-danger small p-2">Failed to load analysis. Please try again later.</div>`;
        });
});
</script>
{% endblock %}
'''


LOGIN_HTML_TEMPLATE = '''
{% extends "BASE_HTML_TEMPLATE" %}

{% block title %}Login - Briefly{% endblock %}

{% block content %}
<div class="auth-container article-card animate-fade-in mx-auto">
    <h2 class="auth-title mb-4"><i class="fas fa-sign-in-alt me-2"></i>Member Login</h2>
    <form method="POST" action="{{ url_for('login') }}">
        <div class="modal-form-group">
            <label for="username" class="form-label">Username</label>
            <input type="text" class="modal-form-control" id="username" name="username" required placeholder="Enter your username">
        </div>
        <div class="modal-form-group">
            <label for="password" class="form-label">Password</label>
            <input type="password" class="modal-form-control" id="password" name="password" required placeholder="Enter your password">
        </div>
        <button type="submit" class="btn btn-primary-modal w-100 mt-3">Login</button>
    </form>
    <p class="mt-3 text-center small">
        Don't have an account? <a href="{{ url_for('register') }}" class="fw-medium">Register here</a>
    </p>
</div>
{% endblock %}
'''

REGISTER_HTML_TEMPLATE = '''
{% extends "BASE_HTML_TEMPLATE" %}

{% block title %}Register - Briefly{% endblock %}

{% block content %}
<div class="auth-container article-card animate-fade-in mx-auto">
    <h2 class="auth-title mb-4"><i class="fas fa-user-plus me-2"></i>Create Account</h2>
    <form method="POST" action="{{ url_for('register') }}">
        <div class="modal-form-group">
            <label for="name" class="form-label">Full Name</label>
            <input type="text" class="modal-form-control" id="name" name="name" required placeholder="Enter your full name">
        </div>
        <div class="modal-form-group">
            <label for="username" class="form-label">Username</label>
            <input type="text" class="modal-form-control" id="username" name="username" required placeholder="Choose a username (min 3 chars)">
        </div>
        <div class="modal-form-group">
            <label for="password" class="form-label">Password</label>
            <input type="password" class="modal-form-control" id="password" name="password" required placeholder="Create a strong password (min 6 chars)">
        </div>
        <button type="submit" class="btn btn-primary-modal w-100 mt-3">Register</button>
    </form>
    <p class="mt-3 text-center small">
        Already have an account? <a href="{{ url_for('login') }}" class="fw-medium">Login here</a>
    </p>
</div>
{% endblock %}
'''
print("Cell 4: HTML templates defined and updated.")


# In[5]:


# Cell 5: Populate Jinja2 DictLoader
# The keys here are the "filenames" Jinja will look for.
if 'template_storage' in globals():
    template_storage['BASE_HTML_TEMPLATE'] = BASE_HTML_TEMPLATE
    template_storage['INDEX_HTML_TEMPLATE'] = INDEX_HTML_TEMPLATE
    template_storage['ARTICLE_HTML_TEMPLATE'] = ARTICLE_HTML_TEMPLATE
    template_storage['LOGIN_HTML_TEMPLATE'] = LOGIN_HTML_TEMPLATE
    template_storage['REGISTER_HTML_TEMPLATE'] = REGISTER_HTML_TEMPLATE
    print("Cell 5: HTML templates loaded into Jinja DictLoader storage.")
else:
    print("Error: template_storage dictionary not found. Make sure Flask app initialization cell was run correctly.")


# In[6]:


# Cell 6: Flask Routes
@app.route('/')
@app.route('/page/<int:page>')
@app.route('/category/<category_name>')
@app.route('/category/<category_name>/page/<int:page>')
def index(page=1, category_name='All Articles'):
    app.logger.info(f"Request received for Index/Category. Category: '{category_name}', Page: {page}")
    per_page = app.config['PER_PAGE']
    query_param = request.args.get('query', None)
    display_articles = []
    total_articles = 0
    featured_article_on_this_page = (page == 1 and not query_param and category_name in ['All Articles', 'My Articles'])

    if category_name == 'My Articles':
        if not session.get('user_id'):
            flash("You need to be logged in to view 'My Articles'.", "warning")
            return redirect(url_for('login'))
        
        all_user_articles = sorted(
            [art for art in USER_ADDED_ARTICLES_STORE if art.get('user_id') == session['user_id']],
            key=lambda x: x.get('publishedAt', datetime.min.isoformat()),
            reverse=True
        )
        total_articles = len(all_user_articles)
        start_index = (page - 1) * per_page
        end_index = start_index + per_page
        display_articles = all_user_articles[start_index:end_index]
    else:
        category_keyword = category_name if category_name != 'All Articles' else None
        api_page_size_to_fetch = app.config['NEWS_API_PAGE_SIZE']
        fetched_articles_from_api = fetch_news_from_api(
            query=app.config['NEWS_API_QUERY'],
            category_keyword=category_keyword,
            page_size=api_page_size_to_fetch
        )
        unique_titles = set()
        unique_api_articles = []
        for art in fetched_articles_from_api:
            if art['title'] and art['title'].lower() not in unique_titles:
                unique_api_articles.append(art)
                unique_titles.add(art['title'].lower())
        
        unique_api_articles.sort(key=lambda x: x.get('publishedAt', datetime.min.isoformat()), reverse=True)
        total_articles = len(unique_api_articles)
        start_index = (page - 1) * per_page
        end_index = start_index + per_page
        display_articles = unique_api_articles[start_index:end_index]

    if not display_articles and page > 1:
        return redirect(url_for('index', category_name=category_name, page=1))

    for art in display_articles:
        if art['id'] not in MASTER_ARTICLE_STORE:
            MASTER_ARTICLE_STORE[art['id']] = art
        stored_art_data = MASTER_ARTICLE_STORE[art['id']]
        art.update(stored_art_data)

    total_pages = (total_articles + per_page - 1) // per_page
    if page > total_pages and total_pages > 0:
        return redirect(url_for('index', category_name=category_name, page=total_pages))

    return render_template(
        "INDEX_HTML_TEMPLATE",
        articles=display_articles,
        selected_category=category_name,
        categories=app.config['CATEGORIES'],
        current_page=page,
        total_pages=total_pages,
        query=query_param,
        featured_article_on_this_page=featured_article_on_this_page and bool(display_articles),
        current_year=datetime.utcnow().year,
        session=session
    )

@app.route('/search/')
@app.route('/search/page/<int:page>')
def search_results(page=1):
    query = request.args.get('query', '').strip()
    app.logger.info(f"Search request: Query='{query}', Page={page}")
    per_page = app.config['PER_PAGE']
    if not query:
        flash("Please enter a search term.", "warning")
        return redirect(url_for('index'))
    api_search_query = f"({app.config['NEWS_API_QUERY']}) AND ({query})"
    api_results = fetch_news_from_api(query=api_search_query, page_size=app.config['NEWS_API_PAGE_SIZE'])
    
    user_articles_results = []
    if USER_ADDED_ARTICLES_STORE:
        for art in USER_ADDED_ARTICLES_STORE:
            if query.lower() in art.get('title', '').lower() or \
               query.lower() in art.get('description', '').lower() or \
               query.lower() in art.get('full_text', '').lower() or \
               (art.get('groq_analysis') and query.lower() in art['groq_analysis'].get('groq_summary','').lower()):
                user_articles_results.append(art)

    combined_articles = api_results + user_articles_results
    unique_ids = set()
    all_search_results = []
    for art in combined_articles:
        if art['id'] not in unique_ids:
            if art['id'] not in MASTER_ARTICLE_STORE:
                 MASTER_ARTICLE_STORE[art['id']] = art
            all_search_results.append(MASTER_ARTICLE_STORE[art['id']])
            unique_ids.add(art['id'])

    all_search_results.sort(key=lambda x: x.get('publishedAt', datetime.min.isoformat()), reverse=True)
    total_articles = len(all_search_results)
    paginated_results = all_search_results[(page - 1) * per_page : page * per_page]

    total_pages = (total_articles + per_page - 1) // per_page
    if page > total_pages and total_pages > 0:
        return redirect(url_for('search_results', query=query, page=total_pages))

    return render_template(
        "INDEX_HTML_TEMPLATE",
        articles=paginated_results,
        selected_category=f"Search: {query}",
        categories=app.config['CATEGORIES'],
        current_page=page,
        total_pages=total_pages,
        query=query,
        featured_article_on_this_page=False,
        current_year=datetime.utcnow().year,
        session=session
    )

@app.route('/article/<article_id>')
def article_detail(article_id):
    app.logger.info(f"Article detail requested for ID: {article_id}")
    article_data = MASTER_ARTICLE_STORE.get(article_id)
    if not article_data:
        app.logger.warning(f"Article ID {article_id} not found.")
        flash("Article not found.", "danger")
        return redirect(url_for('index'))

    # We still need to calculate initial read_time if available,
    # but full_text itself is no longer passed to the template for display.
    read_time_minutes = article_data.get('read_time_minutes', 0)
    if read_time_minutes == 0 and article_data.get('full_text'):
        read_time_minutes = calculate_read_time(article_data.get('full_text'))

    return render_template(
        "ARTICLE_HTML_TEMPLATE",
        article=article_data,
        read_time_minutes=read_time_minutes,
        # groq_analysis is needed for server-side render if available
        groq_analysis=article_data.get('groq_analysis'),
        categories=app.config['CATEGORIES'],
        selected_category=article_data.get('category', 'All Articles'),
        current_year=datetime.utcnow().year,
        session=session
    )

@app.route('/get_article_content/<article_id>')
def get_article_content_json(article_id):
    app.logger.info(f"Async JSON request for processed content of article ID: {article_id}")
    article_data_from_master = MASTER_ARTICLE_STORE.get(article_id)
    if not article_data_from_master:
        app.logger.warning(f"Article ID {article_id} not found for async content fetch.")
        return jsonify({"error": "Article not found"}), 404

    url = article_data_from_master.get('url')
    title = article_data_from_master.get('title', '')
    
    # This function still needs the full_text to perform analysis, but won't return it.
    _, newspaper_summary, read_time, groq_analysis_data = fetch_process_and_analyze_article_content(article_id, url, title)

    # MODIFICATION: Removed 'full_text' and 'newspaper_summary' from the JSON response
    # as they are no longer displayed on the page.
    return jsonify({
        "read_time_minutes": read_time,
        "groq_analysis": groq_analysis_data,
        "error": None
    })

@app.route('/add_article', methods=['POST'])
def add_article():
    if not session.get('user_id'):
        app.logger.warning("Unauthorized attempt to add article.")
        return jsonify({"error": "Authentication required", "redirect_url": url_for('login')}), 401

    data = request.json
    app.logger.info(f"User '{session.get('user_name')}' adding article: {data.get('title')[:50]}...")

    required_fields = ['title', 'description', 'content', 'sourceName']
    missing_fields = [field for field in required_fields if not data.get(field)]
    if missing_fields:
        app.logger.warning(f"Add article failed: Missing fields: {', '.join(missing_fields)}")
        return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}."}), 400

    unique_identifier = data['title'] + session['user_id'] + str(time.time())
    article_id = generate_article_id(unique_identifier)
    current_time_iso = datetime.utcnow().isoformat() + "Z"

    new_article_data = {
        'id': article_id,
        'title': data['title'],
        'description': data['description'],
        'url': f'#user-added-{article_id}',
        'urlToImage': data.get('imageUrl') or f'https://via.placeholder.com/400x220/1E3A5E/FFFFFF?text={data["title"][:20].replace(" ", "+")}',
        'publishedAt': current_time_iso,
        'source': {'name': data.get('sourceName', 'My Publication'), 'id': None},
        'full_text': data['content'],
        'category': 'My Articles',
        'api_category_keyword': 'My Articles',
        'user_id': session['user_id'],
        'user_name': session.get('user_name', 'Unknown User'),
        'read_time_minutes': calculate_read_time(data['content']),
        'newspaper_summary': data['description']
    }

    if groq_client and new_article_data['full_text']:
        app.logger.info(f"Performing Groq analysis for new user-added article: {new_article_data['title'][:50]}")
        groq_result = get_article_analysis_with_groq(new_article_data['full_text'], new_article_data['title'])
        new_article_data['groq_analysis'] = groq_result
    else:
        new_article_data['groq_analysis'] = {
            "groq_summary": "", "groq_takeaways": [], "error": "Groq analysis not performed."
        }

    USER_ADDED_ARTICLES_STORE.insert(0, new_article_data)
    MASTER_ARTICLE_STORE[article_id] = new_article_data

    app.logger.info(f"User article '{new_article_data['title'][:50]}' (ID: {article_id}) added by {session.get('user_name')}.")
    flash("Custom article added successfully! You can find it under 'My Articles'.", "success")
    return jsonify({"message": "Article added successfully!", "redirect_url": url_for('index', category_name='My Articles')}), 201


@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('user_id'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        name = request.form.get('name', '').strip()
        if not username or not password or not name:
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))
        if len(username) < 3:
            flash('Username must be at least 3 characters long.', 'warning')
            return redirect(url_for('register'))
        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'warning')
            return redirect(url_for('register'))
        if username in USERS:
            flash('Username already exists. Please choose a different one.', 'warning')
            return redirect(url_for('register'))
        USERS[username] = {
            "password_hash": generate_password_hash(password),
            "name": name
        }
        app.logger.info(f"New user registered: {username}")
        flash(f'Registration successful, {name}! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template(
        "REGISTER_HTML_TEMPLATE",
        categories=app.config['CATEGORIES'],
        selected_category=None,
        current_year=datetime.utcnow().year,
        session=session
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('user_id'):
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user_account = USERS.get(username)
        if user_account and check_password_hash(user_account['password_hash'], password):
            session['user_id'] = username
            session['user_name'] = user_account['name']
            session.permanent = True
            app.permanent_session_lifetime = timedelta(days=30)
            app.logger.info(f"User '{username}' logged in successfully.")
            flash(f"Welcome back, {user_account['name']}!", "success")
            next_url = request.args.get('next')
            if next_url:
                return redirect(next_url)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
    return render_template(
        "LOGIN_HTML_TEMPLATE",
        categories=app.config['CATEGORIES'],
        selected_category=None,
        current_year=datetime.utcnow().year,
        session=session
    )

@app.route('/logout')
def logout():
    user_name_logged_out = session.get('user_name', 'User')
    session.pop('user_id', None)
    session.pop('user_name', None)
    app.logger.info(f"User '{user_name_logged_out}' logged out.")
    flash("You have been logged out successfully.", "info")
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    app.logger.error(f"Page not found (404): {request.url} - {e}")
    return render_template_string(
        "{% extends 'BASE_HTML_TEMPLATE' %}{% block title %}404 Not Found{% endblock %}{% block content %}<div class='container text-center my-5'><h1><i class='fas fa-exclamation-triangle text-warning me-2'></i>404 - Page Not Found</h1><p>The page you are looking for does not exist or has been moved.</p><a href='{{url_for(\"index\")}}' class='btn btn-primary'>Go to Homepage</a></div>{% endblock %}",
        session=session, categories=app.config['CATEGORIES'], selected_category='All Articles', current_year=datetime.utcnow().year
    ), 404

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"Internal server error (500): {request.url} - {e}", exc_info=True)
    return render_template_string(
        "{% extends 'BASE_HTML_TEMPLATE' %}{% block title %}500 Server Error{% endblock %}{% block content %}<div class='container text-center my-5'><h1><i class='fas fa-cogs text-danger me-2'></i>500 - Internal Server Error</h1><p>Oops! Something went wrong on our end. We are working to fix it. Please try again later.</p><a href='{{url_for(\"index\")}}' class='btn btn-primary'>Go to Homepage</a></div>{% endblock %}",
        session=session, categories=app.config['CATEGORIES'], selected_category='All Articles', current_year=datetime.utcnow().year
    ), 500

print("Cell 6: Flask routes defined and updated.")

# In[ ]:




