#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import hashlib
import time
import logging
import urllib.parse
from datetime import datetime, timedelta, timezone # Keep timezone for direct UTC usage
from functools import wraps

# Third-party imports
import nltk
import requests
from flask import (Flask, render_template, url_for, redirect, request, jsonify, session, flash)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, case
from sqlalchemy.orm import joinedload
from jinja2 import DictLoader
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from newspaper import Article, Config
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import LangChainException
import pytz

# --- Load Environment Variables ---
load_dotenv()

# ==============================================================================
# --- 1. NLTK 'punkt' Tokenizer Setup ---
# ==============================================================================
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    local_nltk_data_path = os.path.join(project_root, 'nltk_data')
    if local_nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_data_path)
    nltk.data.find('tokenizers/punkt', paths=[local_nltk_data_path])
    print("NLTK 'punkt' tokenizer found.", file=sys.stderr)
except LookupError:
    print("WARNING: NLTK 'punkt' tokenizer not found. Attempting to download...", file=sys.stderr)
    try:
        nltk.download('punkt')
        print("NLTK 'punkt' downloaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"FATAL: Failed to download 'punkt'. Error: {e}", file=sys.stderr)
        sys.exit("Exiting: Missing critical NLTK data.")

# ==============================================================================
# --- 2. Flask Application Initialization & Configuration ---
# ==============================================================================
app = Flask(__name__)

template_storage = {}
app.jinja_loader = DictLoader(template_storage)

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'YOUR_FALLBACK_FLASK_SECRET_KEY_HERE_32_CHARS')
app.config['PER_PAGE'] = 9
app.config['CATEGORIES'] = ['All Articles', 'Community Hub']

app.config['NEWS_API_QUERY'] = 'India OR "Indian politics" OR "Indian economy" OR "Bollywood"'
app.config['NEWS_API_DOMAINS'] = 'timesofindia.indiatimes.com,thehindu.com,ndtv.com,indianexpress.com,hindustantimes.com'
app.config['NEWS_API_PAGE_SIZE'] = 100
app.config['NEWS_API_SORT_BY'] = 'publishedAt'
app.config['NEWS_API_DAYS_AGO'] = 7 # Default days to fetch if no specific date (as per user reference)
app.config['CACHE_EXPIRY_SECONDS'] = 1800 # 30 minutes
app.permanent_session_lifetime = timedelta(days=30)

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

# ==============================================================================
# --- Data Persistence Configuration ---
# ==============================================================================
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.logger.info("Connecting to persistent PostgreSQL database.")
else:
    db_file_name = 'app_data.db'
    project_root_for_db = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(project_root_for_db, db_file_name)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.logger.warning(f"Using temporary SQLite database at {db_path}. THIS IS NOT FOR PRODUCTION.")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Removed Celery specific configuration as AI task is now synchronous.
# If other Celery tasks are needed, the configuration should be added back.

# ==============================================================================
# --- 3. API Client Initialization ---
# ==============================================================================
NEWSAPI_KEY = os.environ.get('NEWSAPI_KEY')
newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None
if not newsapi: app.logger.error("NEWSAPI_KEY missing. News fetching will fail.")

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = ChatGroq(model="llama3-70b-8192", groq_api_key=GROQ_API_KEY, temperature=0.1)
        app.logger.info("Groq client initialized.")
    except Exception as e:
        app.logger.error(f"Failed to initialize Groq client: {e}")
else:
    app.logger.warning("GROQ_API_KEY missing. AI analysis disabled.")

SCRAPER_API_KEY = os.environ.get('SCRAPER_API_KEY')
if not SCRAPER_API_KEY:
    app.logger.warning("SCRAPER_API_KEY missing. Article content fetching may fail.")

# ==============================================================================
# --- 4. Database Models ---
# ==============================================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    articles = db.relationship('CommunityArticle', backref='author', lazy='dynamic', cascade="all, delete-orphan")
    comments = db.relationship('Comment', backref=db.backref('author', lazy='joined'), lazy='dynamic', cascade="all, delete-orphan")
    comment_votes = db.relationship('CommentVote', backref='user', lazy='dynamic', cascade="all, delete-orphan")

class CommunityArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    article_hash_id = db.Column(db.String(32), unique=True, nullable=False, index=True)
    title = db.Column(db.String(250), nullable=False)
    description = db.Column(db.Text, nullable=False)
    full_text = db.Column(db.Text, nullable=False)
    source_name = db.Column(db.String(100), nullable=False)
    image_url = db.Column(db.String(500), nullable=True)
    published_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)) # Store as UTC
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    groq_summary = db.Column(db.Text, nullable=True)
    groq_takeaways = db.Column(db.Text, nullable=True)
    comments = db.relationship('Comment', backref=db.backref('community_article', lazy='joined'), lazy='dynamic', foreign_keys='Comment.community_article_id', cascade="all, delete-orphan")

# AiAnalysisTask model is removed as Celery is not used for this specific feature anymore.

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)) # Store as UTC
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    community_article_id = db.Column(db.Integer, db.ForeignKey('community_article.id'), nullable=True)
    api_article_hash_id = db.Column(db.String(32), nullable=True, index=True)
    parent_id = db.Column(db.Integer, db.ForeignKey('comment.id'), nullable=True)
    replies = db.relationship('Comment', backref=db.backref('parent', remote_side=[id]), lazy='selectin', cascade="all, delete-orphan")
    votes = db.relationship('CommentVote', backref='comment', lazy='dynamic', cascade="all, delete-orphan")

class CommentVote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False)
    comment_id = db.Column(db.Integer, db.ForeignKey('comment.id', ondelete="CASCADE"), nullable=False)
    vote_type = db.Column(db.SmallInteger, nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'comment_id', name='_user_comment_uc'),)

class Subscriber(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    subscribed_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)) # Store as UTC

def init_db():
    with app.app_context():
        app.logger.info("Attempting to create database tables...")
        db.create_all()
        app.logger.info("Database tables should be ready.")

# ==============================================================================
# --- 5. Helper Functions ---
# ==============================================================================
MASTER_ARTICLE_STORE, API_CACHE = {}, {}
INDIAN_TIMEZONE = pytz.timezone('Asia/Kolkata')

def generate_article_id(url_or_title): return hashlib.md5(url_or_title.encode('utf-8')).hexdigest()

def jinja_truncate_filter(s, length=120, killwords=False, end='...'):
    if not s: return ''
    if len(s) <= length: return s
    if killwords: return s[:length - len(end)] + end
    words = s.split()
    result_words = []
    current_length = 0
    for word in words:
        if current_length + len(word) + (1 if result_words else 0) > length - len(end): break
        result_words.append(word)
        current_length += len(word) + (1 if len(result_words) > 1 else 0)
    if not result_words: return s[:length - len(end)] + end
    return ' '.join(result_words) + end
app.jinja_env.filters['truncate'] = jinja_truncate_filter

def to_ist_filter(utc_dt):
    if not utc_dt: return "N/A"
    if isinstance(utc_dt, str):
        try:
            utc_dt = datetime.fromisoformat(utc_dt.replace('Z', '+00:00'))
        except ValueError:
            try: 
                utc_dt = datetime.strptime(utc_dt, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                return "Invalid date string"
    if utc_dt.tzinfo is None:
        utc_dt = pytz.utc.localize(utc_dt) 
    else:
        utc_dt = utc_dt.astimezone(pytz.utc) 

    ist_dt = utc_dt.astimezone(INDIAN_TIMEZONE)
    return ist_dt.strftime('%b %d, %Y at %I:%M %p %Z')
app.jinja_env.filters['to_ist'] = to_ist_filter


def simple_cache(expiry_seconds_default=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            expiry = expiry_seconds_default or app.config['CACHE_EXPIRY_SECONDS']
            key_parts = [func.__name__] + list(map(str, args)) + sorted(kwargs.items())
            if 'target_date_str' in kwargs:
                 key_parts.append(kwargs['target_date_str'])
            elif request and request.args and 'date' in request.args : 
                 key_parts.append(request.args.get('date'))

            cache_key = hashlib.md5(str(key_parts).encode('utf-8')).hexdigest()
            cached_entry = API_CACHE.get(cache_key)
            if cached_entry and (time.time() - cached_entry[1] < expiry):
                app.logger.debug(f"Cache HIT for {func.__name__} with key {cache_key}")
                return cached_entry[0]
            app.logger.debug(f"Cache MISS for {func.__name__} with key {cache_key}. Calling function.")
            result = func(*args, **kwargs)
            API_CACHE[cache_key] = (result, time.time())
            return result
        return wrapper
    return decorator

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("You must be logged in to access this page.", "warning")
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# MODIFIED: get_article_analysis_with_groq is now a synchronous, cached function
@simple_cache(expiry_seconds_default=3600 * 12) # Cache AI analysis for 12 hours
def get_article_analysis_with_groq(article_text, article_title=""):
    if not groq_client:
        app.logger.warning("Groq client not available for AI analysis.")
        return {"error": "AI analysis service not available."}
    if not article_text or not article_text.strip():
        return {"error": "No text provided for AI analysis."}
    
    app.logger.info(f"Requesting SYNC Groq analysis for: {article_title[:50]}...")
    system_prompt = ("You are an expert news analyst. Analyze the following article. "
                     "1. Provide a concise, neutral summary (3-4 paragraphs). "
                     "2. List 5-7 key takeaways as bullet points. Each takeaway must be a complete sentence. "
                     "Format your entire response as a single JSON object with keys 'summary' (string) and 'takeaways' (a list of strings).")
    human_prompt = f"Article Title: {article_title}\n\nArticle Text:\n{article_text[:20000]}" # Limiting input size
    
    try:
        json_model = groq_client.bind(response_format={"type": "json_object"})
        ai_response = json_model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        analysis = json.loads(ai_response.content)
        
        if 'summary' in analysis and 'takeaways' in analysis:
            app.logger.info(f"SYNC Groq analysis successful for: {article_title[:50]}.")
            return {"groq_summary": analysis.get("summary"), "groq_takeaways": analysis.get("takeaways"), "error": None}
        
        app.logger.error(f"Groq JSON missing 'summary' or 'takeaways' for '{article_title[:50]}'. Response: {ai_response.content}")
        # Do not raise error here, return it in the dict so it can be handled
        return {"error": "AI response missing 'summary' or 'takeaways'.", "groq_summary": None, "groq_takeaways": None}
        
    except json.JSONDecodeError as e:
        app.logger.error(f"SYNC Groq analysis JSONDecodeError for '{article_title[:50]}': {e}. Response: {ai_response.content if 'ai_response' in locals() else 'N/A'}")
        return {"error": f"AI analysis response was not valid JSON: {str(e)}", "groq_summary": None, "groq_takeaways": None}
    except LangChainException as e: # More specific LangChain errors
        app.logger.error(f"SYNC Groq LangChainException for '{article_title[:50]}': {e}")
        return {"error": f"AI analysis processing error: {str(e)}", "groq_summary": None, "groq_takeaways": None}
    except Exception as e: # Catch-all for other unexpected errors
        app.logger.error(f"SYNC Unexpected error during Groq analysis for '{article_title[:50]}': {e}", exc_info=True)
        return {"error": "An unexpected error occurred during AI analysis.", "groq_summary": None, "groq_takeaways": None}

# ==============================================================================
# --- NEWS FETCHING: MODIFIED FOR DAY-WISE FILTERING & CORRECT DATE FORMAT ---
# ==============================================================================
@simple_cache()
def fetch_news_from_api(target_date_str=None): # target_date_str is like 'YYYY-MM-DD' in IST
    if not newsapi:
        app.logger.error("NewsAPI client not initialized. Cannot fetch news.")
        return []

    from_date_utc_api_str = None
    to_date_utc_api_str = None
    days_ago_config = app.config.get('NEWS_API_DAYS_AGO', 7) # Use configured value or default to 7

    try:
        if target_date_str:
            app.logger.info(f"Fetching news for specific IST date: {target_date_str}")
            selected_day_naive = datetime.strptime(target_date_str, '%Y-%m-%d')
            day_start_ist = INDIAN_TIMEZONE.localize(selected_day_naive)
            day_end_ist = day_start_ist + timedelta(days=1) - timedelta(seconds=1)

            from_date_utc = day_start_ist.astimezone(pytz.utc)
            to_date_utc = day_end_ist.astimezone(pytz.utc)

            from_date_utc_api_str = from_date_utc.strftime('%Y-%m-%dT%H:%M:%S') # No 'Z'
            to_date_utc_api_str = to_date_utc.strftime('%Y-%m-%dT%H:%M:%S')   # No 'Z'
            app.logger.info(f"Querying NewsAPI from UTC: {from_date_utc_api_str} to UTC: {to_date_utc_api_str}")
        else:
            from_date_utc = datetime.now(timezone.utc) - timedelta(days=days_ago_config)
            to_date_utc = datetime.now(timezone.utc)
            
            from_date_utc_api_str = from_date_utc.strftime('%Y-%m-%dT%H:%M:%S') # No 'Z'
            to_date_utc_api_str = to_date_utc.strftime('%Y-%m-%dT%H:%M:%S')   # No 'Z'
            app.logger.info(f"Fetching news for default period (last {days_ago_config} days). Querying NewsAPI from UTC: {from_date_utc_api_str} to UTC: {to_date_utc_api_str}")

    except (ValueError, TypeError) as e:
        app.logger.warning(f"Invalid date format or processing error for '{target_date_str}': {e}. Falling back to default period.")
        from_date_utc = datetime.now(timezone.utc) - timedelta(days=days_ago_config)
        to_date_utc = datetime.now(timezone.utc)
        from_date_utc_api_str = from_date_utc.strftime('%Y-%m-%dT%H:%M:%S') # No 'Z'
        to_date_utc_api_str = to_date_utc.strftime('%Y-%m-%dT%H:%M:%S')   # No 'Z'

    all_raw_articles = []
    try:
        app.logger.info(f"Attempting to fetch 'everything' with query: {app.config['NEWS_API_QUERY']} from {from_date_utc_api_str} to {to_date_utc_api_str}")
        everything_response = newsapi.get_everything(
            q=app.config['NEWS_API_QUERY'],
            from_param=from_date_utc_api_str,
            to=to_date_utc_api_str,
            language='en',
            sort_by=app.config['NEWS_API_SORT_BY'],
            page_size=app.config['NEWS_API_PAGE_SIZE']
        )
        status = everything_response.get('status')
        total_results = everything_response.get('totalResults', 0)
        app.logger.info(f"Everything API Response -> Status: {status}, TotalResults: {total_results}")
        if status == 'ok' and total_results > 0:
            all_raw_articles.extend(everything_response['articles'])
        elif status == 'error':
            app.logger.error(f"NewsAPI Error (Everything): {everything_response.get('message')}")
    except NewsAPIException as e:
        app.logger.error(f"NewsAPIException (Everything): {e}", exc_info=True)
    except ValueError as e: 
        app.logger.error(f"ValueError during NewsAPI call setup (Everything): {e}", exc_info=True)
    except Exception as e: 
        app.logger.error(f"General Exception during NewsAPI call (Everything): {e}", exc_info=True)

    processed_articles, unique_urls = [], set()
    app.logger.info(f"Total raw articles fetched before deduplication: {len(all_raw_articles)}")
    for art_data in all_raw_articles:
        url = art_data.get('url')
        if not url or url in unique_urls: continue
        title = art_data.get('title')
        if not all([title, art_data.get('source'), art_data.get('description')]) or title == '[Removed]' or not title.strip():
            continue
        unique_urls.add(url)
        article_id = generate_article_id(url)
        source_name = art_data['source'].get('name', 'Unknown Source')
        placeholder_text = urllib.parse.quote_plus(source_name[:20])
        standardized_article = {
            'id': article_id, 'title': title, 'description': art_data.get('description', ''),
            'url': url, 'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/0D2C54/FFFFFF?text={placeholder_text}',
            'publishedAt': art_data.get('publishedAt'), 
            'source': {'name': source_name}, 'is_community_article': False
        }
        MASTER_ARTICLE_STORE[article_id] = standardized_article
        processed_articles.append(standardized_article)

    processed_articles.sort(key=lambda x: x.get('publishedAt', '') or '', reverse=True)
    app.logger.info(f"Total unique articles processed and ready to serve: {len(processed_articles)}.")
    return processed_articles

# MODIFIED: fetch_and_parse_article_content now calls AI analysis synchronously
@simple_cache(expiry_seconds_default=3600 * 6) # Cache for 6 hours
def fetch_and_parse_article_content(article_hash_id, url): # url is the article's original URL
    app.logger.info(f"Fetching content for API article ID: {article_hash_id}, URL: {url}")
    if not SCRAPER_API_KEY:
        app.logger.warning("SCRAPER_API_KEY not found, cannot fetch article content.")
        return {"error": "Content fetching service unavailable.", "full_text": None, "groq_analysis": {"error": "Content fetching service unavailable."}}
    
    params = {'api_key': SCRAPER_API_KEY, 'url': url}
    article_text = None
    # Get title from MASTER_ARTICLE_STORE as a fallback or primary if scraper doesn't provide one
    article_title_for_ai = MASTER_ARTICLE_STORE.get(article_hash_id, {}).get('title', 'Article')

    try:
        response = requests.get('http://api.scraperapi.com', params=params, timeout=45)
        response.raise_for_status()

        config = Config()
        config.fetch_images = False
        config.memoize_articles = False
        config.request_timeout = 30

        article_scraper = Article(url, config=config)
        article_scraper.download(input_html=response.text)
        article_scraper.parse()
        
        article_text = article_scraper.text
        if article_scraper.title: # Prefer title from scraped content if available
            article_title_for_ai = article_scraper.title

        if not article_text or not article_text.strip():
            app.logger.warning(f"Could not extract text from article URL: {url}")
            return {"error": "Could not extract text from the article.", "full_text": None, "groq_analysis": {"error": "Could not extract text from the article."}}
        
        app.logger.info(f"Content parsed for {article_hash_id}. Requesting AI analysis for title: '{article_title_for_ai[:50]}'")
        groq_analysis_result = get_article_analysis_with_groq(article_text, article_title_for_ai)
        
        return {
            "full_text": article_text, 
            "groq_analysis": groq_analysis_result,
            "error": None 
        }

    except requests.exceptions.Timeout:
        app.logger.error(f"Timeout while fetching article content via proxy for {url}")
        return {"error": "Timeout while fetching article content.", "full_text": None, "groq_analysis": {"error": "Timeout fetching content."}}
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Failed to fetch article content via proxy for {url}: {e}")
        return {"error": f"Failed to fetch article content: {str(e)}", "full_text": None, "groq_analysis": {"error": f"Failed to fetch content: {str(e)}"}}
    except Exception as e: 
        app.logger.error(f"Failed to parse article content for {url}: {e}", exc_info=True)
        return {"error": f"Failed to parse article content: {str(e)}", "full_text": article_text, "groq_analysis": {"error": f"Failed to parse content: {str(e)}"}}


# ==============================================================================
# --- 6. Flask Routes ---
# ==============================================================================
@app.context_processor
def inject_global_vars():
    selected_date_val = request.args.get('date') or \
                        (request.view_args.get('date_str') if request.view_args else None) or \
                        datetime.now(INDIAN_TIMEZONE).strftime('%Y-%m-%d')
    return {
        'categories': app.config['CATEGORIES'],
        'current_year': datetime.utcnow().year,
        'session': session,
        'request': request, 
        'selected_date': selected_date_val,
        'GROQ_API_KEY': GROQ_API_KEY # Make available to template for conditional JS logic
    }

def get_paginated_articles(articles, page, per_page):
    total_articles = len(articles)
    max_pages = 11 
    total_displayable_articles = max_pages * per_page
    effective_total_articles = min(total_articles, total_displayable_articles)
    total_pages = (effective_total_articles + per_page - 1) // per_page if per_page > 0 else 0
    start = (page - 1) * per_page
    if page > total_pages and total_pages > 0 : 
         return [], total_pages 
    end = start + per_page
    paginated_items = articles[start:end]
    return paginated_items, total_pages


def get_sort_key(article):
    date_val = None
    if isinstance(article, dict): date_val = article.get('publishedAt') 
    elif hasattr(article, 'published_at'): date_val = article.published_at 
    
    if not date_val: return datetime.min.replace(tzinfo=pytz.utc) 

    if isinstance(date_val, str):
        try:
            dt_obj = datetime.fromisoformat(date_val.replace('Z', '+00:00'))
            return dt_obj 
        except ValueError:
            app.logger.warning(f"Could not parse date string for sorting: {date_val}")
            return datetime.min.replace(tzinfo=pytz.utc)
            
    if isinstance(date_val, datetime):
        return date_val.astimezone(pytz.utc) if date_val.tzinfo else pytz.utc.localize(date_val)
        
    return datetime.min.replace(tzinfo=pytz.utc)


@app.route('/')
@app.route('/page/<int:page>')
@app.route('/date/<string:date_str>') 
@app.route('/date/<string:date_str>/page/<int:page>')
@app.route('/category/<category_name>')
@app.route('/category/<category_name>/page/<int:page>')
@app.route('/category/<category_name>/date/<string:date_str>') 
@app.route('/category/<category_name>/date/<string:date_str>/page/<int:page>')
def index(page=1, category_name='All Articles', date_str=None):
    session['previous_list_page'] = request.full_path 
    per_page = app.config['PER_PAGE']
    all_display_articles = []

    if category_name == 'Community Hub':
        db_articles = CommunityArticle.query.options(joinedload(CommunityArticle.author)).order_by(CommunityArticle.published_at.desc()).all()
        for art in db_articles:
            art.is_community_article = True 
            all_display_articles.append(art)
    else: 
        api_articles = fetch_news_from_api(target_date_str=date_str)
        for art_dict in api_articles:
            art_dict['is_community_article'] = False 
            all_display_articles.append(art_dict)

    all_display_articles.sort(key=get_sort_key, reverse=True)
    paginated_display_articles, total_pages = get_paginated_articles(all_display_articles, page, per_page)
    
    featured_article_on_this_page = (page == 1 and
                                     category_name == 'All Articles' and
                                     not request.args.get('query') and 
                                     not date_str and 
                                     paginated_display_articles)

    return render_template("INDEX_HTML_TEMPLATE",
                           articles=paginated_display_articles,
                           selected_category=category_name,
                           current_page=page,
                           total_pages=total_pages,
                           featured_article_on_this_page=featured_article_on_this_page,
                           )


@app.route('/search')
@app.route('/search/page/<int:page>')
def search_results(page=1):
    session['previous_list_page'] = request.full_path
    query_str = request.args.get('query', '').strip()
    per_page = app.config['PER_PAGE']
    if not query_str: return redirect(url_for('index'))
    app.logger.info(f"Search query: '{query_str}'")

    api_articles = fetch_news_from_api() 
    api_results = []
    for art_data in api_articles:
        if query_str.lower() in art_data.get('title', '').lower() or \
           query_str.lower() in art_data.get('description', '').lower():
            art_copy = art_data.copy()
            art_copy['is_community_article'] = False
            api_results.append(art_copy)

    community_db_articles_query = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter(
        db.or_(CommunityArticle.title.ilike(f'%{query_str}%'), CommunityArticle.description.ilike(f'%{query_str}%'))
    ).order_by(CommunityArticle.published_at.desc())
    community_db_articles = []
    for art in community_db_articles_query.all():
        art.is_community_article = True
        community_db_articles.append(art)
    
    all_search_results = api_results + community_db_articles
    all_search_results.sort(key=get_sort_key, reverse=True)
    paginated_search_articles, total_pages = get_paginated_articles(all_search_results, page, per_page)
    
    return render_template("INDEX_HTML_TEMPLATE",
                           articles=paginated_search_articles,
                           selected_category=f"Search: {query_str}",
                           current_page=page,
                           total_pages=total_pages,
                           featured_article_on_this_page=False,
                           query=query_str)


@app.route('/article/<article_hash_id>')
def article_detail(article_hash_id):
    article_data, is_community_article, comments_for_template, all_article_comments_list, comment_data = None, False, [], [], {}
    previous_list_page = session.get('previous_list_page', url_for('index'))

    article_db = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter_by(article_hash_id=article_hash_id).first()

    if article_db:
        article_data = article_db
        is_community_article = True
        article_data.is_community_article = True 
        try:
            article_data.parsed_takeaways = json.loads(article_data.groq_takeaways) if article_data.groq_takeaways else []
        except json.JSONDecodeError:
            app.logger.error(f"JSONDecodeError for groq_takeaways on community article {article_data.article_hash_id}")
            article_data.parsed_takeaways = []
        
        all_article_comments_list = Comment.query.options(
            joinedload(Comment.author),
            joinedload(Comment.replies).options(joinedload(Comment.author)) 
        ).filter_by(community_article_id=article_db.id).order_by(Comment.timestamp.asc()).all()
        comments_for_template = [c for c in all_article_comments_list if c.parent_id is None]
    else:
        article_api_dict = MASTER_ARTICLE_STORE.get(article_hash_id)
        if article_api_dict:
            article_data = article_api_dict.copy() 
            is_community_article = False
            article_data['is_community_article'] = False 
            
            all_article_comments_list = Comment.query.options(
                joinedload(Comment.author),
                joinedload(Comment.replies).options(joinedload(Comment.author))
            ).filter_by(api_article_hash_id=article_hash_id).order_by(Comment.timestamp.asc()).all()
            comments_for_template = [c for c in all_article_comments_list if c.parent_id is None]
        else:
            flash("Article not found.", "danger")
            return redirect(previous_list_page) 

    if all_article_comments_list:
        comment_ids = []
        for c_outer in all_article_comments_list: # Renamed to avoid conflict with inner 'c'
            comment_ids.append(c_outer.id)
            for r_outer in c_outer.replies: 
                comment_ids.append(r_outer.id)
        
        for c_id_vote in comment_ids: comment_data[c_id_vote] = {'likes': 0, 'dislikes': 0, 'user_vote': 0}
        
        vote_counts_query = db.session.query(
            CommentVote.comment_id,
            func.sum(case((CommentVote.vote_type == 1, 1), else_=0)).label('likes'),
            func.sum(case((CommentVote.vote_type == -1, 1), else_=0)).label('dislikes')
        ).filter(CommentVote.comment_id.in_(comment_ids)).group_by(CommentVote.comment_id).all()
        
        for c_id_vote_count, likes, dislikes in vote_counts_query:
            if c_id_vote_count in comment_data:
                comment_data[c_id_vote_count]['likes'] = likes
                comment_data[c_id_vote_count]['dislikes'] = dislikes
        
        if 'user_id' in session:
            user_votes = CommentVote.query.filter(CommentVote.comment_id.in_(comment_ids), CommentVote.user_id == session['user_id']).all()
            for vote in user_votes:
                if vote.comment_id in comment_data:
                    comment_data[vote.comment_id]['user_vote'] = vote.vote_type
    
    return render_template("ARTICLE_HTML_TEMPLATE",
                           article=article_data,
                           is_community_article=is_community_article,
                           comments=comments_for_template,
                           comment_data=comment_data,
                           previous_list_page=previous_list_page)

# NEW Endpoint for fetching API article content and AI analysis synchronously
@app.route('/api_article_data/<article_hash_id>')
def api_article_data_json(article_hash_id):
    app.logger.info(f"Request received for /api_article_data/{article_hash_id}")
    article_api_dict = MASTER_ARTICLE_STORE.get(article_hash_id)

    if not article_api_dict:
        app.logger.warning(f"Article {article_hash_id} not found in MASTER_ARTICLE_STORE for API data fetch.")
        return jsonify({"error": "Article not found in local cache.", "groq_analysis": None}), 404

    article_url = article_api_dict.get('url')
    if not article_url:
        app.logger.error(f"URL missing for article {article_hash_id} in MASTER_ARTICLE_STORE.")
        return jsonify({"error": "Article URL not found.", "groq_analysis": None}), 404

    # fetch_and_parse_article_content is cached, and get_article_analysis_with_groq within it is also cached.
    processed_data = fetch_and_parse_article_content(article_hash_id, article_url)
    
    # The 'error' key in processed_data is for fetch/parse errors.
    # The AI specific error is within processed_data['groq_analysis']['error']
    if processed_data.get("error") and not processed_data.get("groq_analysis"):
        app.logger.error(f"Error processing article {article_hash_id} for API data: {processed_data.get('error')}")
        return jsonify(processed_data), 500 

    # Update MASTER_ARTICLE_STORE with the fetched analysis (for very short-term in-memory cache)
    # This is mostly for consistency if the same article is requested again quickly before outer cache expires.
    if 'groq_analysis' in processed_data:
         MASTER_ARTICLE_STORE[article_hash_id]['groq_analysis'] = processed_data['groq_analysis']
    
    app.logger.info(f"Successfully fetched/generated data for {article_hash_id} for API endpoint.")
    return jsonify({
        "groq_analysis": processed_data.get('groq_analysis'),
        "error": processed_data.get('error') 
    })


@app.route('/add_comment/<article_hash_id>', methods=['POST'])
@login_required
def add_comment(article_hash_id):
    content = request.json.get('content', '').strip()
    parent_id = request.json.get('parent_id') 
    if not content: return jsonify({"error": "Comment cannot be empty."}), 400
    
    user = User.query.get(session['user_id'])
    if not user:
        app.logger.error(f"User not found in add_comment for user_id {session.get('user_id')}")
        return jsonify({"error": "User not found."}), 401

    new_comment_instance = None
    community_article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
    
    if community_article:
        new_comment_instance = Comment(content=content, user_id=user.id, community_article_id=community_article.id, parent_id=parent_id)
    elif article_hash_id in MASTER_ARTICLE_STORE: 
        new_comment_instance = Comment(content=content, user_id=user.id, api_article_hash_id=article_hash_id, parent_id=parent_id)
    else:
        return jsonify({"error": "Article not found."}), 404
        
    try:
        db.session.add(new_comment_instance)
        db.session.commit()
        author_name = user.name 
        comment_timestamp_ist_str = to_ist_filter(new_comment_instance.timestamp)

        return jsonify({
            "success": True,
            "comment": {
                "id": new_comment_instance.id,
                "content": new_comment_instance.content,
                "timestamp_iso": new_comment_instance.timestamp.isoformat(), 
                "timestamp_display": comment_timestamp_ist_str, 
                "author": {"name": author_name, "initial": author_name[0].upper() if author_name else "U"},
                "parent_id": new_comment_instance.parent_id,
                "likes": 0, 
                "dislikes": 0,
                "user_vote": 0 
            }
        }), 201
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error adding comment: {e}", exc_info=True)
        return jsonify({"error": "Could not post comment due to a server error."}), 500


@app.route('/vote_comment/<int:comment_id>', methods=['POST'])
@login_required
def vote_comment(comment_id):
    comment = Comment.query.get_or_404(comment_id)
    vote_type = request.json.get('vote_type') 
    if vote_type not in [1, -1]:
        return jsonify({"error": "Invalid vote type."}), 400

    user_id = session['user_id']
    existing_vote = CommentVote.query.filter_by(user_id=user_id, comment_id=comment_id).first()
    
    new_user_vote_status = 0 

    if existing_vote:
        if existing_vote.vote_type == vote_type: 
            db.session.delete(existing_vote)
            new_user_vote_status = 0 
        else: 
            existing_vote.vote_type = vote_type
            new_user_vote_status = vote_type
    else: 
        db.session.add(CommentVote(user_id=user_id, comment_id=comment_id, vote_type=vote_type))
        new_user_vote_status = vote_type
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error voting on comment {comment_id}: {e}", exc_info=True)
        return jsonify({"error": "Could not process vote due to server error."}), 500

    likes = CommentVote.query.filter_by(comment_id=comment_id, vote_type=1).count()
    dislikes = CommentVote.query.filter_by(comment_id=comment_id, vote_type=-1).count()
    
    return jsonify({
        "success": True,
        "likes": likes,
        "dislikes": dislikes,
        "user_vote_status": new_user_vote_status 
    }), 200

@app.route('/post_article', methods=['POST'])
@login_required
def post_article():
    title = request.form.get('title', '').strip()
    description = request.form.get('description', '').strip()
    content = request.form.get('content', '').strip()
    source_name = request.form.get('sourceName', '').strip() or 'Community Post' 
    image_url = request.form.get('imageUrl', '').strip()

    if not all([title, description, content]): 
        flash("Title, Description, and Full Content are required.", "danger")
        return redirect(request.referrer or url_for('index'))

    article_hash_id = generate_article_id(title + str(session['user_id']) + str(time.time())) 

    groq_summary_text, groq_takeaways_json_str = None, None
    if groq_client and content: 
        try:
            app.logger.info(f"Requesting SYNC Groq analysis for new community article: {title[:50]}...")
            analysis_result = get_article_analysis_with_groq(content, title) 

            if analysis_result and not analysis_result.get("error"):
                groq_summary_text = analysis_result.get('groq_summary')
                takeaways_list = analysis_result.get('groq_takeaways')
                if takeaways_list and isinstance(takeaways_list, list):
                    groq_takeaways_json_str = json.dumps(takeaways_list)
                app.logger.info(f"SYNC Groq analysis successful for community post: {title[:50]}.")
            else:
                app.logger.warning(f"SYNC Groq analysis failed or no result for {title[:50]}: {analysis_result.get('error') if analysis_result else 'No result'}")
        except Exception as e:
            app.logger.error(f"Exception during SYNC Groq analysis for community post {title[:50]}: {e}", exc_info=True)


    new_article = CommunityArticle(
        article_hash_id=article_hash_id,
        title=title,
        description=description,
        full_text=content,
        source_name=source_name,
        image_url=image_url or f'https://via.placeholder.com/400x220/1E3A5E/FFFFFF?text={urllib.parse.quote_plus(title[:20])}',
        user_id=session['user_id'],
        published_at=datetime.now(timezone.utc), 
        groq_summary=groq_summary_text,
        groq_takeaways=groq_takeaways_json_str
    )
    
    try:
        db.session.add(new_article)
        db.session.commit()
        flash("Your article has been posted!", "success")
        return redirect(url_for('article_detail', article_hash_id=new_article.article_hash_id))
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error posting article: {e}", exc_info=True)
        flash("Could not post article due to a server error.", "danger")
        return redirect(request.referrer or url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        
        if not all([name, username, password]):
            flash('All fields are required.', 'danger')
        elif len(username) < 3:
            flash('Username must be at least 3 characters.', 'warning')
        elif len(password) < 6:
            flash('Password must be at least 6 characters.', 'warning')
        elif User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different one.', 'warning')
        else:
            try:
                new_user = User(name=name, username=username, password_hash=generate_password_hash(password))
                db.session.add(new_user)
                db.session.commit()
                flash(f'Registration successful, {name}! Please log in.', 'success')
                return redirect(url_for('login'))
            except Exception as e:
                db.session.rollback()
                app.logger.error(f"Error during registration for {username}: {e}", exc_info=True)
                flash('Registration failed due to a server error. Please try again.', 'danger')
    return render_template("REGISTER_HTML_TEMPLATE")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session.permanent = True 
            session['user_id'] = user.id
            session['user_name'] = user.name
            flash(f"Welcome back, {user.name}!", "success")
            next_url = request.args.get('next')
            session.pop('previous_list_page', None) 
            return redirect(next_url or url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template("LOGIN_HTML_TEMPLATE")

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been successfully logged out.", "info")
    return redirect(url_for('index'))

@app.route('/about')
def about(): return render_template("ABOUT_US_HTML_TEMPLATE")
@app.route('/contact')
def contact(): return render_template("CONTACT_HTML_TEMPLATE")
@app.route('/privacy')
def privacy(): return render_template("PRIVACY_POLICY_HTML_TEMPLATE")

@app.route('/subscribe', methods=['POST'])
def subscribe():
    email = request.form.get('email', '').strip().lower()
    if not email: 
        flash('Email is required to subscribe.', 'warning')
    elif Subscriber.query.filter_by(email=email).first():
        flash('You are already subscribed with this email address.', 'info')
    else:
        try:
            db.session.add(Subscriber(email=email, subscribed_at=datetime.now(timezone.utc)))
            db.session.commit()
            flash('Thank you for subscribing!', 'success')
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error subscribing email {email}: {e}", exc_info=True)
            flash('Could not subscribe due to a server error. Please try again.', 'danger')
    return redirect(request.referrer or url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    app.logger.warning(f"404 error at {request.url}: {e}", exc_info=False) # No need for full traceback for 404
    return render_template("404_TEMPLATE"), 404

@app.errorhandler(500)
def internal_server_error(e):
    db.session.rollback()
    app.logger.error(f"500 error at {request.url}: {e}", exc_info=True)
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify(error="Internal Server Error", message=str(e)), 500
    return render_template("500_TEMPLATE"), 500

# Rev14.py (Continued)

# ==============================================================================
# --- 7. HTML Templates (Full, Unabbreviated Version) ---
# ==============================================================================
BASE_HTML_TEMPLATE = """
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
            --primary-color: #0A2342; --primary-light: #1E3A5E;
            --secondary-color: #B8860B; --secondary-light: #D4A017;
            --accent-color: #F07F2D; --text-color: #343a40;
            --text-muted-color: #6c757d; --light-bg: #F8F9FA;
            --white-bg: #FFFFFF; --card-border-color: #E0E0E0;
            --footer-bg: #061A30; --footer-text: rgba(255,255,255,0.8);
            --footer-link-hover: var(--secondary-color);
            --primary-gradient: linear-gradient(135deg, var(--primary-color), var(--primary-light));
            --primary-color-rgb: 10, 35, 66; --secondary-color-rgb: 184, 134, 11;
        }
        body { padding-top: 145px; font-family: 'Roboto', sans-serif; line-height: 1.65; color: var(--text-color); background-color: var(--light-bg); display: flex; flex-direction: column; min-height: 100vh; transition: background-color 0.3s ease, color 0.3s ease; }
        .main-content { flex-grow: 1; }
        body.dark-mode { --primary-color: #1E3A5E; --primary-light: #2A4B7C; --secondary-color: #D4A017; --secondary-light: #E7B400; --accent-color: #FF983E; --text-color: #E9ECEF; --text-muted-color: #ADB5BD; --light-bg: #121212; --white-bg: #1E1E1E; --card-border-color: #333333; --footer-bg: #0A0A0A; --footer-text: rgba(255,255,255,0.7); --primary-color-rgb: 30, 58, 94; --secondary-color-rgb: 212, 160, 23; }
        body.dark-mode .navbar-main { background: linear-gradient(135deg, #0A1A2F, #10233B); border-bottom: 1px solid #2A4B7C; }
        body.dark-mode .category-nav { background: #1A1A1A; border-bottom: 1px solid #2A2A2A; }
        body.dark-mode .category-link { color: var(--text-muted-color) !important; }
        body.dark-mode .category-link.active { background: var(--primary-color) !important; color: var(--white-bg) !important; }
        body.dark-mode .category-link:hover:not(.active) { background: #2C2C2C !important; color: var(--secondary-color) !important; }
        body.dark-mode .article-card, body.dark-mode .featured-article, body.dark-mode .article-full-content-wrapper, body.dark-mode .auth-container, body.dark-mode .static-content-wrapper { background-color: var(--white-bg); border-color: var(--card-border-color); }
        body.dark-mode .article-title a, body.dark-mode h1, body.dark-mode h2, body.dark-mode h3, body.dark-mode h4, body.dark-mode h5, body.dark-mode .auth-title { color: var(--text-color) !important; }
        body.dark-mode .article-description, body.dark-mode .meta-item, body.dark-mode .content-text p, body.dark-mode .article-meta-detailed, body.dark-mode .comment-content, body.dark-mode .comment-date { color: var(--text-muted-color) !important; }
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
        .navbar-main { background: var(--primary-gradient); padding: 0.8rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-bottom: 2px solid rgba(255,255,255,0.15); transition: background 0.3s ease, border-bottom 0.3s ease; height: 95px; display: flex; align-items: center; }
        .navbar-brand-custom { color: white !important; font-weight: 800; font-size: 2.2rem; letter-spacing: 0.5px; font-family: 'Poppins', sans-serif; margin-bottom: 0; display: flex; align-items: center; gap: 12px; text-decoration: none !important; }
        .navbar-brand-custom .brand-icon { color: var(--secondary-light); font-size: 2.5rem; }
        .navbar-brand-custom:hover { text-decoration: none !important; }
        .search-form-container { flex-grow: 1; display: flex; justify-content: center; padding: 0 1rem; }
        .search-container { position: relative; width: 100%; max-width: 550px; }
        .navbar-search { border-radius: 25px; padding: 0.7rem 1.25rem 0.7rem 2.8rem; border: 1px solid rgba(255,255,255,0.2); font-size: 0.95rem; transition: all 0.3s ease; background: rgba(255,255,255,0.1); color: white; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        .navbar-search::placeholder { color: rgba(255,255,255,0.6); }
        .navbar-search:focus { background: rgba(255,255,255,0.2); box-shadow: 0 0 0 3px rgba(var(--secondary-color-rgb),0.3); border-color: var(--secondary-color); outline: none; color:white; }
        .search-icon { color: rgba(255,255,255,0.7); transition: all 0.3s ease; left: 1rem; position: absolute; top: 50%; transform: translateY(-50%); }
        .navbar-search:focus + .search-icon { color: var(--secondary-light); }
        .header-controls { display: flex; gap: 0.8rem; align-items: center; }
        .header-btn { background: transparent; border: 1px solid rgba(255,255,255,0.3); padding: 0.5rem 1rem; border-radius: 20px; color: white; font-weight: 500; transition: all 0.3s ease; display: flex; align-items: center; gap: 0.5rem; cursor: pointer; text-decoration:none; font-size: 0.9rem; }
        .header-btn:hover { background: var(--secondary-color); border-color: var(--secondary-color); color: var(--primary-color); transform: translateY(-1px); }
        .dark-mode-toggle { font-size: 1.1rem; width: 40px; height: 40px; justify-content: center;}
        .category-nav { background: var(--white-bg); box-shadow: 0 3px 10px rgba(0,0,0,0.03); position: fixed; top: 95px; width: 100%; z-index: 1020; border-bottom: 1px solid var(--card-border-color); transition: background 0.3s ease, border-bottom 0.3s ease; }
        .categories-wrapper { display: flex; justify-content: center; align-items: center; width: 100%; overflow-x: auto; padding: 0.4rem 0; scrollbar-width: thin; scrollbar-color: var(--secondary-color) var(--light-bg); }
        .categories-wrapper::-webkit-scrollbar { height: 6px; }
        .categories-wrapper::-webkit-scrollbar-thumb { background: var(--secondary-color); border-radius: 3px; }
        .category-link { color: var(--primary-color) !important; font-weight: 600; padding: 0.6rem 1.3rem !important; border-radius: 20px; transition: all 0.25s ease; white-space: nowrap; text-decoration: none; margin: 0 0.3rem; font-size: 0.9rem; border: 1px solid transparent; font-family: 'Roboto', sans-serif; }
        .category-link.active { background: var(--primary-color) !important; color: white !important; box-shadow: 0 3px 10px rgba(var(--primary-color-rgb), 0.2); border-color: var(--primary-light); }
        .category-link:hover:not(.active) { background: var(--light-bg) !important; color: var(--secondary-color) !important; border-color: var(--secondary-color); }
        .article-card, .featured-article, .article-full-content-wrapper, .auth-container, .static-content-wrapper { background: var(--white-bg); border-radius: 10px; transition: all 0.3s ease; border: 1px solid var(--card-border-color); box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
        .article-card:hover, .featured-article:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.08); }
        .article-image-container { height: 200px; overflow: hidden; position: relative; border-top-left-radius: 9px; border-top-right-radius: 9px;}
        .article-image { width: 100%; height: 100%; object-fit: cover; transition: transform 0.4s ease; }
        .article-card:hover .article-image { transform: scale(1.08); }
        .category-tag { position: absolute; top: 10px; left: 10px; background: var(--secondary-color); color: var(--primary-color); font-size: 0.65rem; font-weight: 700; padding: 0.3rem 0.7rem; border-radius: 15px; z-index: 5; text-transform: uppercase; letter-spacing: 0.3px; }
        body.dark-mode .category-tag { color: var(--white-bg); background-color: var(--primary-light); }
        .article-body { padding: 1.25rem; flex-grow: 1; display: flex; flex-direction: column; }
        .article-title { font-weight: 700; line-height: 1.35; margin-bottom: 0.6rem; font-size:1.1rem; }
        .article-title a { color: var(--primary-color); text-decoration: none; }
        .article-card:hover .article-title a { color: var(--primary-color) !important; } 
        body.dark-mode .article-card .article-title a { color: var(--text-color) !important; }
        body.dark-mode .article-card:hover .article-title a { color: var(--secondary-color) !important; }
        .article-meta { display: flex; align-items: center; margin-bottom: 0.8rem; flex-wrap: wrap; gap: 0.4rem 1rem; }
        .meta-item { display: flex; align-items: center; font-size: 0.8rem; color: var(--text-muted-color); }
        .meta-item i { font-size: 0.9rem; margin-right: 0.3rem; color: var(--secondary-color); }
        .article-description { color: var(--text-muted-color); margin-bottom: 1rem; font-size: 0.9rem; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
        .read-more { margin-top: auto; background: var(--primary-color); color: white !important; border: none; padding: 0.5rem 0; border-radius: 6px; font-weight: 600; font-size: 0.85rem; transition: all 0.3s ease; width: 100%; text-align: center; text-decoration: none; display:inline-block; }
        .read-more:hover { background: var(--primary-light); transform: translateY(-2px); color: white !important; }
        body.dark-mode .read-more { background: var(--secondary-color); color: var(--primary-color) !important;}
        body.dark-mode .read-more:hover { background: var(--secondary-light); }
        .pagination { margin: 2rem 0; display: flex; justify-content: center; gap: 0.3rem; }
        .page-item .page-link { border-radius: 50%; width: 40px; height: 40px; display:flex; align-items:center; justify-content:center; color: var(--primary-color); border: 1px solid var(--card-border-color); font-weight: 600; transition: all 0.2s ease; font-size:0.9rem; }
        .page-item .page-link:hover { background-color: var(--light-bg); border-color: var(--secondary-color); color: var(--secondary-color); }
        .page-item.active .page-link { background-color: var(--primary-color); border-color: var(--primary-color); color: white; box-shadow: 0 2px 8px rgba(var(--primary-color-rgb), 0.3); }
        .page-item.disabled .page-link { color: var(--text-muted-color); pointer-events: none; background-color: var(--light-bg); }
        .page-link-prev-next .page-link { width: auto; padding-left:1rem; padding-right:1rem; border-radius:20px; }
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
        .admin-controls { position: fixed; bottom: 25px; right: 25px; z-index: 1030; }
        .add-article-btn { width: 55px; height: 55px; border-radius: 50%; background: var(--secondary-color); color: var(--primary-color); border: none; box-shadow: 0 4px 15px rgba(var(--secondary-color-rgb),0.3); display: flex; align-items: center; justify-content: center; font-size: 22px; cursor: pointer; transition: all 0.3s ease; }
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
        body.dark-mode .modal-title { color: var(--secondary-color); }
        .btn-primary-modal { background-color: var(--primary-color); border-color: var(--primary-color); color:white; padding: 0.6rem 1.2rem; font-weight:600; }
        .btn-primary-modal:hover { background-color: var(--primary-light); border-color: var(--primary-light); }
        body.dark-mode .btn-primary-modal { background-color: var(--secondary-color); border-color: var(--secondary-color); color: var(--primary-color); }
        body.dark-mode .btn-primary-modal:hover { background-color: var(--secondary-light); border-color: var(--secondary-light); }
        .btn-outline-secondary-modal { padding: 0.6rem 1.2rem; font-weight:600; border-color: var(--text-muted-color); color: var(--text-muted-color); }
        body.dark-mode .btn-outline-secondary-modal { border-color: var(--text-muted-color); color: var(--text-muted-color); }
        body.dark-mode .btn-outline-secondary-modal:hover { background-color: #333; color: var(--text-color); border-color: #444;}
        .alert-top { position: fixed; top: 105px; left: 50%; transform: translateX(-50%); z-index: 2050; min-width:320px; text-align:center; box-shadow: 0 3px 10px rgba(0,0,0,0.1);}
        .animate-fade-in { animation: fadeIn 0.5s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(25px); } to { opacity: 1; transform: translateY(0); } }
        .fade-in-delay-1 { animation-delay: 0.1s; } .fade-in-delay-2 { animation-delay: 0.2s; } .fade-in-delay-3 { animation-delay: 0.3s; }
        .navbar-content-wrapper { display: flex; justify-content: space-between; align-items: center; width: 100%; }
        .static-content-wrapper { padding: 2rem; margin-top: 1rem; }
        .static-content-wrapper h1, .static-content-wrapper h2 { color: var(--primary-color); font-family: 'Poppins', sans-serif; }
        body.dark-mode .static-content-wrapper h1, body.dark-mode .static-content-wrapper h2 { color: var(--secondary-color); }
        .auth-container { max-width: 450px; margin: 3rem auto; padding: 2rem; }
        .auth-title { text-align: center; color: var(--primary-color); margin-bottom: 1.5rem; font-weight: 700;}
        body.dark-mode .auth-title { color: var(--secondary-color); }
        .comment-section { margin-top: 3rem; }
        .comment-container { margin-bottom: 1.5rem; }
        .comment-card { display: flex; gap: 1rem; }
        .comment-avatar { width: 40px; height: 40px; border-radius: 50%; background: var(--primary-light); color: white; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0; }
        body.dark-mode .comment-avatar { background: var(--primary-color); }
        .comment-body { flex-grow: 1; border-bottom: 1px solid var(--card-border-color); padding-bottom: 1rem; }
        .comment-container:last-child > .comment-card > .comment-body { border-bottom: none; }
        .comment-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem; }
        .comment-author { font-weight: 600; color: var(--primary-color); }
        body.dark-mode .comment-author { color: var(--secondary-light); }
        .comment-date { font-size: 0.8rem; color: var(--text-muted-color); }
        .comment-content { font-size: 0.95rem; color: var(--text-color); margin-bottom: 0.5rem; white-space: pre-wrap; } 
        .comment-actions { display: flex; align-items: center; gap: 0.75rem; font-size: 0.85rem; }
        .comment-actions button { background: none; border: none; padding: 0.2rem 0.4rem; color: var(--text-muted-color); cursor: pointer; display: flex; align-items: center; gap: 0.3rem; transition: color 0.2s ease, background-color 0.2s ease; border-radius: 4px; }
        .comment-actions button:hover { color: var(--primary-color); background-color: rgba(var(--primary-color-rgb), 0.1); }
        body.dark-mode .comment-actions button:hover { color: var(--secondary-light); background-color: rgba(var(--secondary-color-rgb),0.2); }
        .comment-actions button.active.vote-up { color: var(--primary-color); } 
        .comment-actions button.active.vote-down { color: var(--accent-color); } 
        body.dark-mode .comment-actions button.active.vote-up { color: var(--secondary-color); }
        body.dark-mode .comment-actions button.active.vote-down { color: var(--accent-color); }
        .comment-actions .vote-count { font-weight: 500; min-width: 12px; text-align: center;}
        .comment-replies { margin-left: 30px; padding-left: 1.25rem; border-left: 2px solid var(--card-border-color); margin-top: 1rem; }
        .reply-form-container { display: none; margin-top: 0.75rem; padding: 0.75rem; background-color: rgba(var(--primary-color-rgb), 0.03); border-radius: 6px;}
        body.dark-mode .reply-form-container { background-color: rgba(var(--secondary-color-rgb), 0.05); }
        .add-comment-form textarea { min-height: 100px; }
        .date-picker-container {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0 0.5rem 0 1rem; 
        }
        .date-picker-input {
            background-color: #fff;
            color: var(--primary-color);
            border: 1px solid var(--card-border-color);
            border-radius: 20px;
            padding: 0.4rem 0.8rem;
            font-size: 0.85rem;
            font-weight: 500;
            max-width: 150px; 
        }
        body.dark-mode .date-picker-input {
            background-color: #2C2C2C; color: var(--text-color); border-color: #444;
        }
        .date-picker-container .btn {
            padding: 0.4rem 0.8rem;
            font-size: 0.85rem;
            border-radius: 20px;
        }
        @media (max-width: 991.98px) { 
            body { padding-top: 180px; } 
            .navbar-main { padding-bottom: 0.5rem; height: auto;}
            .navbar-content-wrapper { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
            .navbar-brand-custom { margin-bottom: 0.5rem; }
            .search-form-container { width: 100%; order: 3; margin-top:0.5rem; padding: 0; }
            .header-controls { position: absolute; top: 0.9rem; right: 1rem; order: 2; }
            .category-nav { top: 130px; } 
        }
        @media (max-width: 767.98px) { 
            body { padding-top: 170px; }
            .category-nav { top: 120px; }
            .featured-article .row { flex-direction: column; }
            .featured-image { margin-bottom: 1rem; height: 250px; }
            .categories-wrapper { justify-content: flex-start; } 
            .date-picker-container { margin-left: 0.5rem; } 
        }
        @media (max-width: 575.98px) { 
            .navbar-brand-custom { font-size: 1.8rem;}
            .header-controls { gap: 0.3rem; }
            .header-btn { padding: 0.4rem 0.8rem; font-size: 0.8rem; }
            .dark-mode-toggle { font-size: 1rem; }
            .article-title-main {font-size: 1.8rem;} 
        }
    </style>
    {% block head_extra %}{% endblock %}
</head>
<body class="{{ request.cookies.get('darkMode', 'disabled') if request and request.cookies else 'disabled' }}">
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
                    <form action="{{ url_for('search_results') }}" method="GET" class="search-container animate-fade-in fade-in-delay-1">
                        <input type="search" name="query" class="form-control navbar-search" placeholder="Search news articles..." value="{{ request.args.get('query', '') if request and request.args else '' }}">
                        <i class="fas fa-search search-icon"></i>
                        <button type="submit" class="d-none">Search</button>
                    </form>
                </div>
                <div class="header-controls animate-fade-in fade-in-delay-2">
                    <button class="header-btn dark-mode-toggle" aria-label="Toggle dark mode" title="Toggle Dark Mode">
                        <i class="fas fa-moon"></i>
                    </button>
                    {% if session.user_id %}
                    <span class="text-white me-2 d-none d-md-inline">Hi, {{ session.user_name|truncate(15) }}!</span>
                    <a href="{{ url_for('logout') }}" class="header-btn" title="Logout">
                        <i class="fas fa-sign-out-alt"></i> <span class="d-none d-sm-inline">Logout</span>
                    </a>
                    {% else %}
                    <a href="{{ url_for('login', next=(request.url if request else '')) }}" class="header-btn" title="Login/Register">
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
                {% set _view_args = request.view_args if request and request.view_args is defined else {} %}
                {% set _query_args_date = request.args.get('date') if request and request.args else none %}
                {% set _current_category_from_view = _view_args.get('category_name') %}
                {% set _current_date_from_view = _view_args.get('date_str') %}

                {% for cat_item in categories if categories is defined %}
                    {% set is_active_category = (selected_category is defined and selected_category == cat_item) or \
                                                (_current_category_from_view is defined and _current_category_from_view == cat_item) or \
                                                (selected_category is not defined and _current_category_from_view is none and cat_item == 'All Articles' and not (_current_date_from_view or _query_args_date) and not (request.args.get('query') if request and request.args else '')) %}
                    {% set link_date_str = selected_date if selected_date is defined and cat_item != 'Community Hub' and (_current_date_from_view or _query_args_date) else none %}
                    <a href="{{ url_for('index', category_name=cat_item, date_str=link_date_str) }}"
                       class="category-link {% if is_active_category %}active{% endif %}">
                        <i class="fas fa-{% if cat_item == 'All Articles' %}globe-americas{% elif cat_item == 'Community Hub' %}users{% else %}tag{% endif %} me-1 d-none d-sm-inline"></i>
                        {{ cat_item }}
                    </a>
                {% endfor %}
                
                {% set _selected_category_for_picker = selected_category if selected_category is defined else _current_category_from_view %}
                {% set show_date_picker = (_selected_category_for_picker is defined and _selected_category_for_picker != 'Community Hub') or \
                                          (_selected_category_for_picker is none and not (request.args.get('query') if request and request.args else '')) %}

                {% if show_date_picker %}
                <div class="date-picker-container">
                    <form id="dateFilterForm" class="d-flex align-items-center">
                        <label for="news_date" class="form-label me-2 mb-0 text-muted small d-none d-sm-inline">Date:</label>
                        <input type="date" id="news_date" name="date" class="form-control form-control-sm date-picker-input" value="{{ selected_date if selected_date is defined else '' }}">
                        <button type="submit" class="btn btn-sm btn-secondary ms-2">Go</button>
                    </form>
                </div>
                {% endif %}
            </div>
        </div>
    </nav>

    <main class="container main-content my-4">
        {% block content %}{% endblock %}
    </main>

    {% if session.user_id %}
    <div class="admin-controls">
        <button class="add-article-btn" id="addArticleBtn" title="Post a New Article">
            <i class="fas fa-plus"></i>
        </button>
    </div>
    <div class="add-article-modal" id="addArticleModal">
        <div class="modal-content">
            <button class="close-modal" id="closeModalBtn" title="Close Modal"><i class="fas fa-times"></i></button>
            <h3 class="modal-title">Post New Article to Community Hub</h3>
            <form id="addArticleForm" action="{{ url_for('post_article') }}" method="POST">
                <div class="modal-form-group"><label for="articleTitle">Article Title</label><input type="text" id="articleTitle" name="title" class="modal-form-control" placeholder="Enter article title" required></div>
                <div class="modal-form-group"><label for="articleDescription">Short Description / Summary</label><textarea id="articleDescription" name="description" class="modal-form-control" rows="3" placeholder="Brief summary of the article" required></textarea></div>
                <div class="modal-form-group"><label for="articleSource">Source Name (e.g., Your Blog, Personal Research)</label><input type="text" id="articleSource" name="sourceName" class="modal-form-control" placeholder="Source of this article" value="Community Post" required></div>
                <div class="modal-form-group"><label for="articleImage">Featured Image URL (Optional)</label><input type="url" id="articleImage" name="imageUrl" class="modal-form-control" placeholder="https://example.com/image.jpg"></div>
                <div class="modal-form-group"><label for="articleContent">Full Article Content</label><textarea id="articleContent" name="content" class="modal-form-control" rows="7" placeholder="Write the full article content here..." required></textarea></div>
                <div class="d-flex justify-content-end gap-2"><button type="button" class="btn btn-outline-secondary-modal" id="cancelArticleBtn">Cancel</button><button type="submit" class="btn btn-primary-modal">Post Article</button></div>
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
                    <p class="small">Your premier source for AI summarized, India-centric news.</p>
                    <div class="social-links">
                        <a href="#" title="Twitter"><i class="fab fa-twitter"></i></a><a href="#" title="Facebook"><i class="fab fa-facebook-f"></i></a><a href="#" title="LinkedIn"><i class="fab fa-linkedin-in"></i></a><a href="#" title="Instagram"><i class="fab fa-instagram"></i></a>
                    </div>
                </div>
                <div class="footer-section">
                    <h5>Quick Links</h5>
                    <div class="footer-links">
                        <a href="{{ url_for('index') }}"><i class="fas fa-angle-right"></i> Home</a>
                        <a href="{{ url_for('about') }}"><i class="fas fa-angle-right"></i> About Us</a>
                        <a href="{{ url_for('contact') }}"><i class="fas fa-angle-right"></i> Contact</a>
                        <a href="{{ url_for('privacy') }}"><i class="fas fa-angle-right"></i> Privacy Policy</a>
                    </div>
                </div>
                <div class="footer-section">
                    <h5>Categories</h5>
                    <div class="footer-links">
                        {% if categories is defined %}
                           {% for cat_item in categories %}<a href="{{ url_for('index', category_name=cat_item) }}"><i class="fas fa-angle-right"></i> {{ cat_item }}</a>{% endfor %}
                        {% endif %}
                    </div>
                </div>
                <div class="footer-section">
                    <h5>Newsletter</h5>
                    <p class="small">Subscribe for weekly updates on the latest news!</p>
                    <form action="{{ url_for('subscribe') }}" method="POST" class="mt-2">
                        <div class="input-group">
                            <input type="email" name="email" class="form-control form-control-sm" placeholder="Your Email" aria-label="Your Email" required>
                            <button class="btn btn-sm btn-primary-modal" type="submit">Subscribe</button>
                        </div>
                    </form>
                </div>
            </div>
            <div class="copyright">&copy; {{ current_year if current_year is defined else '' }} Briefly. All rights reserved. Made with <i class="fas fa-heart text-danger"></i> in India.</div>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
    document.addEventListener('DOMContentLoaded', function () {
        const darkModeToggle = document.querySelector('.dark-mode-toggle');
        const body = document.body;
        function updateThemeIcon() { if(darkModeToggle) { darkModeToggle.innerHTML = body.classList.contains('dark-mode') ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>'; } }
        function applyTheme(theme) {
            if (theme === 'enabled') { body.classList.add('dark-mode'); } else { body.classList.remove('dark-mode'); }
            updateThemeIcon();
            localStorage.setItem('darkMode', theme); 
            document.cookie = "darkMode=" + theme + ";path=/;max-age=" + (60*60*24*365*5) + ";SameSite=Lax"; 
        }
        if(darkModeToggle) { darkModeToggle.addEventListener('click', () => { applyTheme(body.classList.contains('dark-mode') ? 'disabled' : 'enabled'); }); }

        let storedTheme = localStorage.getItem('darkMode');
        if (!storedTheme) { 
            const cookieTheme = document.cookie.split('; ').find(row => row.startsWith('darkMode='))?.split('=')[1];
            if (cookieTheme) storedTheme = cookieTheme;
        }
        if (storedTheme) { applyTheme(storedTheme); } else { updateThemeIcon(); } 

        const addArticleBtn = document.getElementById('addArticleBtn');
        const addArticleModal = document.getElementById('addArticleModal');
        const closeModalBtn = document.getElementById('closeModalBtn');
        const cancelArticleBtn = document.getElementById('cancelArticleBtn');
        if(addArticleBtn && addArticleModal) {
            addArticleBtn.addEventListener('click', () => { addArticleModal.style.display = 'flex'; body.style.overflow = 'hidden'; });
            const closeModalFunction = () => { addArticleModal.style.display = 'none'; body.style.overflow = 'auto'; if(document.getElementById('addArticleForm')) {document.getElementById('addArticleForm').reset();} };
            if(closeModalBtn) closeModalBtn.addEventListener('click', closeModalFunction);
            if(cancelArticleBtn) cancelArticleBtn.addEventListener('click', closeModalFunction);
            addArticleModal.addEventListener('click', (e) => { if (e.target === addArticleModal) closeModalFunction(); });
        }

        const flashedAlerts = document.querySelectorAll('#alert-placeholder .alert');
        flashedAlerts.forEach(function(alert) { setTimeout(function() { const bsAlert = bootstrap.Alert.getOrCreateInstance(alert); if (bsAlert) bsAlert.close(); }, 7000); });

        const dateFilterForm = document.getElementById('dateFilterForm');
        if(dateFilterForm) {
            dateFilterForm.addEventListener('submit', function(e) {
                e.preventDefault();
                const selectedDateInput = document.getElementById('news_date');
                if (!selectedDateInput) return;
                const selectedDateValue = selectedDateInput.value;

                if(selectedDateValue) {
                    let currentPath = window.location.pathname;
                    let pathSegments = currentPath.split('/').filter(Boolean); 
                    let categoryName = 'All Articles'; // Default

                    // Determine current category from URL segments
                    if (pathSegments.includes('category')) {
                        let catIndex = pathSegments.indexOf('category');
                        if (catIndex + 1 < pathSegments.length && pathSegments[catIndex + 1] !== 'date') {
                            categoryName = decodeURIComponent(pathSegments[catIndex + 1]);
                        }
                    }
                    
                    let newUrl = "";
                    if (categoryName !== 'All Articles') { // Specific category selected
                        newUrl = `/category/${encodeURIComponent(categoryName)}/date/${selectedDateValue}`;
                    } else { // 'All Articles' or no category in path (e.g. home, /date/...)
                        newUrl = `/date/${selectedDateValue}`;
                    }
                    window.location.href = newUrl;
                }
            });
        }
    });
    </script>
    {% block scripts_extra %}{% endblock %}
</body>
</html>
"""

INDEX_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}
    {% if query %}Search: {{ query|truncate(30) }}{% elif request.view_args.get('date_str') if request and request.view_args %}{{ selected_category if selected_category is defined else 'News' }} for {{ request.view_args.get('date_str') }}{% elif selected_category is defined %}{{selected_category}}{% else %}Home{% endif %} - Briefly
{% endblock %}
{% block content %}
    {% if articles and articles[0] and featured_article_on_this_page %}
    <article class="featured-article p-md-4 p-3 mb-4 animate-fade-in">
        <div class="row g-0 g-md-4">
            {% set art0 = articles[0] %}
            {% set article_url = url_for('article_detail', article_hash_id=(art0.article_hash_id if art0.is_community_article else art0.id)) %}
            <div class="col-lg-6">
                <div class="featured-image rounded overflow-hidden shadow-sm" style="height:320px;">
                    <a href="{{ article_url }}">
                    <img src="{{ art0.image_url if art0.is_community_article else art0.urlToImage }}" class="img-fluid w-100 h-100" style="object-fit:cover;" alt="Featured: {{ art0.title|truncate(50) }}">
                    </a>
                </div>
            </div>
            <div class="col-lg-6 d-flex flex-column ps-lg-3 pt-3 pt-lg-0">
                <div class="article-meta mb-2">
                    <span class="badge bg-primary me-2" style="font-size:0.75rem;">{{ (art0.author.name if art0.is_community_article and art0.author else art0.source.name)|truncate(25) }}</span>
                    <span class="meta-item"><i class="far fa-calendar-alt"></i> {{ (art0.published_at | to_ist if art0.is_community_article else (art0.publishedAt | to_ist if art0.publishedAt else 'N/A')) }}</span>
                </div>
                <h2 class="mb-2 h4"><a href="{{ article_url }}" class="text-decoration-none article-title">{{ art0.title }}</a></h2>
                <p class="article-description flex-grow-1 small">{{ art0.description|truncate(220) }}</p>
                <a href="{{ article_url }}" class="read-more mt-auto align-self-start py-2 px-3" style="width:auto;">Read Full Article <i class="fas fa-arrow-right ms-1 small"></i></a>
            </div>
        </div>
    </article>
    {% elif not articles and selected_category is defined and selected_category != 'Community Hub' and ((request.view_args.get('date_str') if request and request.view_args else none) or (request.args.get('date') if request and request.args else none)) %}
        <div class="alert alert-warning text-center my-4 p-3 small">No news found for the selected date ({{ selected_date if selected_date is defined else 'this date' }}). Please try another day or check back later.</div>
    {% elif not articles and selected_category is defined and selected_category == 'Community Hub' %}
        <div class="alert alert-info text-center my-4 p-3"><h4><i class="fas fa-feather-alt me-2"></i>No Articles Penned Yet</h4><p>No articles in the Community Hub. {% if session.user_id %}Click the '+' button to share your insights!{% else %}Login to add articles.{% endif %}</p></div>
    {% elif not articles and query is defined and query %}
        <div class="alert alert-info text-center my-5 p-4"><h4><i class="fas fa-search me-2"></i>No results for "{{ query }}"</h4><p>Try different keywords or browse categories.</p></div>
    {% elif not articles %}
         <div class="alert alert-info text-center my-4 p-3 small">No articles available at the moment. Please check back later.</div>
    {% endif %}

    <div class="row g-4">
        {% set articles_to_display = (articles[1:] if featured_article_on_this_page and articles and articles|length > 1 else articles) %}
        {% for art in articles_to_display %}
        <div class="col-md-6 col-lg-4 d-flex">
        <article class="article-card animate-fade-in d-flex flex-column w-100" style="animation-delay: {{ loop.index0 * 0.05 }}s">
            {% set article_url = url_for('article_detail', article_hash_id=(art.article_hash_id if art.is_community_article else art.id)) %}
            <div class="article-image-container">
                <a href="{{ article_url }}">
                <img src="{{ art.image_url if art.is_community_article else art.urlToImage }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a>
            </div>
            <div class="article-body d-flex flex-column">
                <h5 class="article-title mb-2"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                <div class="article-meta small mb-2">
                    <span class="meta-item text-muted"><i class="fas fa-{{ 'user-edit' if art.is_community_article else 'building' }}"></i> {{ (art.author.name if art.is_community_article and art.author else art.source.name)|truncate(20) }}</span>
                    <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ (art.published_at | to_ist if art.is_community_article else (art.publishedAt | to_ist if art.publishedAt else 'N/A')) }}</span>
                </div>
                <p class="article-description small">{{ art.description|truncate(100) }}</p>
                <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
            </div>
        </article>
        </div>
        {% endfor %}
    </div>

    {% if total_pages and total_pages > 1 %}
    <nav aria-label="Page navigation" class="mt-5"><ul class="pagination justify-content-center">
        {% set nav_category = request.view_args.get('category_name') if request.view_args else (selected_category if selected_category is defined else 'All Articles') %}
        {% set nav_date_str = request.view_args.get('date_str') if request.view_args else (request.args.get('date') if request.args else none) %}
        {% set nav_query = query if query is defined else none %}
        <li class="page-item page-link-prev-next {% if current_page == 1 %}disabled{% endif %}"><a class="page-link" href="{{ url_for(request.endpoint, page=current_page-1, category_name=nav_category, query=nav_query, date_str=nav_date_str) if current_page > 1 else '#' }}">&laquo; Prev</a></li>
        {% set page_window = 1 %}{% set show_first = 1 %}{% set show_last = total_pages %}
        {% if current_page - page_window > show_first %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=1, category_name=nav_category, query=nav_query, date_str=nav_date_str) }}">1</a></li>{% if current_page - page_window > show_first + 1 %}<li class="page-item disabled"><span class="page-link">...</span></li>{% endif %}{% endif %}
        {% for p in range(1, total_pages + 1) %}{% if p == current_page %}<li class="page-item active" aria-current="page"><span class="page-link">{{ p }}</span></li>{% elif p >= current_page - page_window and p <= current_page + page_window %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=p, category_name=nav_category, query=nav_query, date_str=nav_date_str) }}">{{ p }}</a></li>{% endif %}{% endfor %}
        {% if current_page + page_window < show_last %}{% if current_page + page_window < show_last - 1 %}<li class="page-item disabled"><span class="page-link">...</span></li>{% endif %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=total_pages, category_name=nav_category, query=nav_query, date_str=nav_date_str) }}">{{ total_pages }}</a></li>{% endif %}
        <li class="page-item page-link-prev-next {% if current_page == total_pages %}disabled{% endif %}"><a class="page-link" href="{{ url_for(request.endpoint, page=current_page+1, category_name=nav_category, query=nav_query, date_str=nav_date_str) if current_page < total_pages else '#' }}">Next &raquo;</a></li>
    </ul></nav>
    {% endif %}
{% endblock %}
"""

ARTICLE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ article.title|truncate(50) if article else "Article" }} - Briefly{% endblock %}
{% block head_extra %}
<style>
    .article-full-content-wrapper { background-color: var(--white-bg); padding: 1.5rem; border-radius: 10px; box-shadow: 0 5px 20px rgba(0,0,0,0.07); margin-bottom: 2rem; margin-top: 1rem; }
    body.dark-mode .article-full-content-wrapper { background-color: var(--white-bg); border-color: var(--card-border-color); }
    .article-full-content-wrapper .main-article-image { width: 100%; max-height: 480px; object-fit: cover; border-radius: 8px; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .article-title-main {font-weight: 700; color: var(--primary-color); line-height:1.3; font-family: 'Poppins', sans-serif;}
    body.dark-mode .article-title-main { color: var(--text-color); }
    .article-meta-detailed { font-size: 0.85rem; color: var(--text-muted-color); margin-bottom: 1.5rem; display:flex; flex-wrap:wrap; gap: 0.5rem 1.2rem; align-items:center; border-bottom: 1px solid var(--card-border-color); padding-bottom:1rem; }
    body.dark-mode .article-meta-detailed {color: var(--text-muted-color); border-bottom-color: var(--card-border-color);}
    .article-meta-detailed .meta-item i { color: var(--secondary-color); margin-right: 0.4rem; font-size:0.95rem; }
    .summary-box { background-color: rgba(var(--primary-color-rgb), 0.04); padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; border: 1px solid rgba(var(--primary-color-rgb), 0.1); }
    body.dark-mode .summary-box { background-color: rgba(var(--secondary-color-rgb), 0.05); border-color: rgba(var(--secondary-color-rgb),0.2); }
    .summary-box h5 { color: var(--primary-color); font-weight: 600; margin-bottom: 0.75rem; font-size:1.1rem; }
    body.dark-mode .summary-box h5 { color: var(--secondary-light); }
    .summary-box p {font-size:0.95rem; line-height:1.7; color: var(--text-color);}
    body.dark-mode .summary-box p { color: var(--text-muted-color); }
    .takeaways-box { margin: 1.5rem 0; padding: 1.5rem 1.5rem 1.5rem 1.8rem; border-left: 4px solid var(--secondary-color); background-color: rgba(var(--primary-color-rgb), 0.04); border-radius: 0 8px 8px 0;}
    body.dark-mode .takeaways-box { background-color: rgba(var(--secondary-color-rgb), 0.05); border-left-color: var(--secondary-light); }
    .takeaways-box h5 { color: var(--primary-color); font-weight: 600; margin-bottom: 0.75rem; font-size:1.1rem; }
    body.dark-mode .takeaways-box h5 { color: var(--secondary-light); }
    .takeaways-box ul { padding-left: 1.2rem; margin-bottom:0; color: var(--text-color); }
    body.dark-mode .takeaways-box ul { color: var(--text-muted-color); }
    .takeaways-box ul li { margin-bottom: 0.6rem; font-size:0.95rem; line-height:1.6; }
    .loader-container { display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 200px; padding: 2rem; font-size: 1rem; color: var(--text-muted-color); text-align: center; }
    .loader { border: 5px solid var(--light-bg); border-top: 5px solid var(--primary-color); border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin-bottom: 1rem; }
    body.dark-mode .loader { border-top-color: var(--secondary-color); }
    .content-text { white-space: pre-wrap; line-height: 1.8; font-size: 1.05rem; color: var(--text-color); }
    body.dark-mode .content-text { color: var(--text-muted-color); }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
{% endblock %}
{% block content %}
{% if not article %}
    <div class="alert alert-danger text-center my-5 p-4"><h4><i class="fas fa-exclamation-triangle me-2"></i>Article Not Found</h4><p>The article you are looking for could not be found.</p><a href="{{ previous_list_page or url_for('index') }}" class="btn btn-primary mt-2">Go Back</a></div>
{% else %}
<article class="article-full-content-wrapper animate-fade-in">
    <div class="mb-3">
        <a href="{{ previous_list_page or url_for('index') }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-arrow-left me-2"></i>Back to List</a>
    </div>

    <h1 class="mb-2 article-title-main display-6">{{ article.title }}</h1>
    <div class="article-meta-detailed">
        <span class="meta-item" title="Source"><i class="fas fa-{{ 'user-edit' if article.is_community_article else 'building' }}"></i> {{ article.author.name if article.is_community_article and article.author else (article.source.name if article.source else 'Unknown Source') }}</span>
        <span class="meta-item" title="Published Date"><i class="far fa-calendar-alt"></i> {{ (article.published_at | to_ist if article.is_community_article else (article.publishedAt | to_ist if article.publishedAt else 'N/A')) }}</span>
    </div>
    {% set image_to_display = article.image_url if article.is_community_article else article.urlToImage %}
    {% if image_to_display %}<img src="{{ image_to_display }}" alt="{{ article.title|truncate(50) }}" class="main-article-image">{% endif %}

    <div id="articleAnalysisContainer">
    {% if article.is_community_article %}
        {% if article.groq_summary %}
            <div class="summary-box my-3"><h5><i class="fas fa-bookmark me-2"></i>Article Summary (AI Enhanced)</h5><p class="mb-0">{{ article.groq_summary|replace('\\n', '<br>')|safe }}</p></div>
        {% elif article.groq_summary is none and GROQ_API_KEY %}
             <div class="alert alert-secondary small p-3 mt-3">AI Summary was not generated for this community article.</div>
        {% elif not GROQ_API_KEY %}
            <div class="alert alert-info small p-3 mt-3">AI analysis service is currently unavailable.</div>
        {% endif %}

        {% if article.parsed_takeaways and article.parsed_takeaways|length > 0 %}
            <div class="takeaways-box my-3"><h5><i class="fas fa-list-check me-2"></i>Key Takeaways (AI Enhanced)</h5>
                <ul>{% for takeaway in article.parsed_takeaways %}<li>{{ takeaway }}</li>{% endfor %}</ul>
            </div>
        {% elif article.groq_takeaways is none and GROQ_API_KEY %}
             <div class="alert alert-secondary small p-3 mt-3">AI Takeaways were not generated for this community article.</div>
        {% endif %}
        <hr class="my-4">
        <h4 class="mb-3">Full Article Content</h4>
        <div class="content-text">{{ article.full_text }}</div>
    {% else %} {# API Article: Content and AI summary will be loaded by JavaScript #}
        <div id="contentLoader" class="loader-container my-4">
            <div class="loader"></div>
            <div>Analyzing article and generating summary... <br><small>This may take a moment.</small></div>
        </div>
        <div id="apiArticleContent"></div> {# Content (summary, takeaways, link) will be injected here by JS #}
    {% endif %}
    </div>

    <section class="comment-section" id="comment-section">
        <h3 class="mb-4">Community Discussion (<span id="comment-count">{{ comments|sum(attribute='replies')|sum(attribute='id', start=comments|length) if comments else 0 }}</span>)</h3>

        {% macro render_comment_with_replies(comment, comment_data, is_logged_in, article_hash_id_for_js, current_user_id) %}
            <div class="comment-container" id="comment-{{ comment.id }}">
                <div class="comment-card">
                    <div class="comment-avatar" title="{{ comment.author.name if comment.author else 'Unknown' }}">
                        {{ (comment.author.name[0]|upper if comment.author and comment.author.name else 'U') }}
                    </div>
                    <div class="comment-body">
                        <div class="comment-header">
                            <span class="comment-author">{{ comment.author.name if comment.author else 'Anonymous' }}</span>
                            <span class="comment-date">{{ comment.timestamp | to_ist }}</span>
                        </div>
                        <p class="comment-content mb-2">{{ comment.content }}</p>
                        {% if is_logged_in %}
                        <div class="comment-actions">
                            <button class="vote-btn {% if comment_data.get(comment.id, {}).get('user_vote') == 1 %}active vote-up{% endif %}" data-comment-id="{{ comment.id }}" data-vote-type="1" title="Like">
                                <i class="fas fa-thumbs-up"></i>
                                <span class="vote-count" id="likes-count-{{ comment.id }}">{{ comment_data.get(comment.id, {}).get('likes', 0) }}</span>
                            </button>
                            <button class="vote-btn {% if comment_data.get(comment.id, {}).get('user_vote') == -1 %}active vote-down{% endif %}" data-comment-id="{{ comment.id }}" data-vote-type="-1" title="Dislike">
                                <i class="fas fa-thumbs-down"></i>
                                <span class="vote-count" id="dislikes-count-{{ comment.id }}">{{ comment_data.get(comment.id, {}).get('dislikes', 0) }}</span>
                            </button>
                            <button class="reply-btn" data-comment-id="{{ comment.id }}" title="Reply">
                                <i class="fas fa-reply"></i> Reply
                            </button>
                        </div>
                        <div class="reply-form-container" id="reply-form-container-{{ comment.id }}">
                            <form class="reply-form mt-2">
                                <input type="hidden" name="article_hash_id" value="{{ article_hash_id_for_js }}">
                                <input type="hidden" name="parent_id" value="{{ comment.id }}">
                                <div class="mb-2">
                                    <textarea class="form-control form-control-sm" name="content" rows="2" placeholder="Write a reply..." required></textarea>
                                </div>
                                <button type="submit" class="btn btn-sm btn-primary-modal">Post Reply</button>
                                <button type="button" class="btn btn-sm btn-outline-secondary-modal cancel-reply-btn">Cancel</button>
                            </form>
                        </div>
                        {% endif %}
                    </div>
                </div>
                <div class="comment-replies" id="replies-of-{{ comment.id }}">
                    {% for reply in comment.replies|sort(attribute='timestamp') %}
                        {{ render_comment_with_replies(reply, comment_data, is_logged_in, article_hash_id_for_js, current_user_id) }}
                    {% endfor %}
                </div>
            </div>
        {% endmacro %}

        <div id="comments-list">
            {% for comment in comments %}
                {{ render_comment_with_replies(comment, comment_data, session.user_id, (article.article_hash_id if article.is_community_article else article.id), session.user_id) }}
            {% else %}
                <p id="no-comments-msg" class="text-muted">No comments yet. Be the first to share your thoughts!</p>
            {% endfor %}
        </div>

        {% if session.user_id %}
            <div class="add-comment-form mt-4 pt-4 border-top">
                <h5 class="mb-3">Leave a Comment</h5>
                <form id="comment-form">
                    <input type="hidden" name="article_hash_id" value="{{ article.article_hash_id if article.is_community_article else article.id }}">
                    <div class="mb-3">
                        <textarea class="form-control" id="comment-content" name="content" rows="4" placeholder="Share your insights..." required></textarea>
                    </div>
                    <button type="submit" class="btn btn-primary-modal">Post Comment</button>
                </form>
            </div>
        {% else %}
            <div class="alert alert-light mt-4 text-center">Please <a href="{{ url_for('login', next=request.url) }}">log in</a> to join the discussion.</div>
        {% endif %}
    </section>
</article>
{% endif %}
{% endblock %}
{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    const articleData = {{ article|tojson if article else 'null' }};
    const isCommunityArticle = {{ article.is_community_article|tojson if article else 'false' }};
    const GROQ_API_IS_CONFIGURED = {{ GROQ_API_KEY is defined and GROQ_API_KEY|tojson != 'null' }};


    if (articleData && !isCommunityArticle) {
        const articleHashIdGlobal = articleData.id; 
        const contentLoader = document.getElementById('contentLoader');
        const apiArticleContentContainer = document.getElementById('apiArticleContent');
        const originalArticleUrl = articleData.url; 
        const originalArticleSourceName = articleData.source ? articleData.source.name : 'the source';

        function displayAnalysis(data) {
            if (contentLoader) contentLoader.style.display = 'none';
            if (!apiArticleContentContainer) return;

            let htmlContent = '';
            const analysisResult = data.groq_analysis;

            if (analysisResult && !analysisResult.error) {
                if (analysisResult.groq_summary) {
                    htmlContent += `<div class="summary-box my-3 animate-fade-in"><h5><i class="fas fa-bookmark me-2"></i>Article Summary (AI Enhanced)</h5><p class="mb-0">${analysisResult.groq_summary.replace(/\\n/g, '<br>')}</p></div>`;
                }
                if (analysisResult.groq_takeaways && analysisResult.groq_takeaways.length > 0) {
                    htmlContent += `<div class="takeaways-box my-3 animate-fade-in"><h5><i class="fas fa-list-check me-2"></i>Key Takeaways (AI Enhanced)</h5><ul>${analysisResult.groq_takeaways.map(t => `<li>${t}</li>`).join('')}</ul></div>`;
                }
            }
            
            // Handle errors from AI analysis or content fetching
            let overallError = data.error; // Error from fetch_and_parse_article_content itself
            let aiError = analysisResult ? analysisResult.error : "AI analysis data not found.";

            if (htmlContent.trim() === '' || (analysisResult && analysisResult.error)) { // If no content OR if there was an AI error
                 let errorToShow = aiError || overallError || "AI-generated summary is not available for this article.";
                 if (overallError && overallError.includes("extract text")) { // More specific error for parsing
                    errorToShow = "Could not extract text from the article for analysis.";
                 }
                 htmlContent = `<div class="alert alert-warning small p-3 mt-3">${errorToShow}</div>` + htmlContent;
            } else if (htmlContent.trim() === '' && !overallError) { // No AI content, no error reported from AI, no fetch error
                 htmlContent = `<div class="alert alert-secondary small p-3 mt-3">AI analysis did not return specific content for this article.</div>`;
            }


            if (originalArticleUrl) {
                htmlContent += `<hr class="my-4"><p class="text-center"><a href="${originalArticleUrl}" class="btn btn-outline-primary mt-3 mb-3" target="_blank" rel="noopener noreferrer">Read Original Article at ${originalArticleSourceName} <i class="fas fa-external-link-alt ms-1"></i></a></p>`;
            }
            apiArticleContentContainer.innerHTML = htmlContent;
        }
        
        if (GROQ_API_IS_CONFIGURED) {
            if (contentLoader) contentLoader.style.display = 'flex';

            fetch(`/api_article_data/${articleHashIdGlobal}`)
                .then(response => {
                    if (!response.ok) {
                        return response.json().then(errData => { // Try to get JSON error from server
                            throw new Error(errData.error || `Server error: ${response.status} ${response.statusText}`);
                        }).catch(() => { // Fallback if error response isn't JSON
                            throw new Error(`Network error: ${response.status} ${response.statusText}`);
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    displayAnalysis(data);
                })
                .catch(error => {
                    console.error("Error fetching article data with AI analysis:", error);
                    if (contentLoader) contentLoader.style.display = 'none';
                    if (apiArticleContentContainer) {
                        let errorHtml = `<div class="alert alert-danger">Failed to load article summary: ${error.message}.</div>`;
                         if (originalArticleUrl) {
                            errorHtml += `<hr class="my-4"><p class="text-center"><a href="${originalArticleUrl}" class="btn btn-outline-primary mt-3 mb-3" target="_blank" rel="noopener noreferrer">You can still read the original article at ${originalArticleSourceName} <i class="fas fa-external-link-alt ms-1"></i></a></p>`;
                        }
                        apiArticleContentContainer.innerHTML = errorHtml;
                    }
                });
        } else { // GROQ_API_KEY not configured
            if (contentLoader) contentLoader.style.display = 'none';
            if (apiArticleContentContainer) {
                 let noAIHtml = `<div class="alert alert-info small p-3 mt-3">AI analysis service is not configured.</div>`;
                 if (originalArticleUrl) {
                    noAIHtml += `<hr class="my-4"><p class="text-center"><a href="${originalArticleUrl}" class="btn btn-outline-primary mt-3 mb-3" target="_blank" rel="noopener noreferrer">Read Original Article at ${originalArticleSourceName} <i class="fas fa-external-link-alt ms-1"></i></a></p>`;
                }
                apiArticleContentContainer.innerHTML = noAIHtml;
            }
        }
    }

    const commentSection = document.getElementById('comment-section');
    if (commentSection && articleData) {
        const isUserLoggedIn = {{ 'true' if session.user_id else 'false' }};
        const currentArticleHashId = isCommunityArticle ? articleData.article_hash_id : articleData.id;

        function convertUTCToISTForComment(utcIsoString) {
            if (!utcIsoString) return "N/A";
            const date = new Date(utcIsoString);
            return new Intl.DateTimeFormat('en-IN', {
                year: 'numeric', month: 'short', day: 'numeric',
                hour: 'numeric', minute: '2-digit', hour12: true,
                timeZone: 'Asia/Kolkata', timeZoneName: 'shortOffset' // Using shortOffset like +05:30
            }).format(date);
        }
        
        function createCommentHTML(commentDataFromServer) {
            const comment = commentDataFromServer.comment; // Assuming server sends { success: true, comment: {...} }
            const authorName = comment.author && comment.author.name ? comment.author.name : 'Anonymous';
            const userInitial = comment.author && comment.author.initial ? comment.author.initial : (authorName[0] ? authorName[0].toUpperCase() : 'A');
            const commentDate = comment.timestamp_display || convertUTCToISTForComment(comment.timestamp_iso);

            let actionsHTML = '';
            if (isUserLoggedIn) {
                actionsHTML = `
                <div class="comment-actions">
                    <button class="vote-btn ${comment.user_vote === 1 ? 'active vote-up' : ''}" data-comment-id="${comment.id}" data-vote-type="1" title="Like"><i class="fas fa-thumbs-up"></i> <span class="vote-count" id="likes-count-${comment.id}">${comment.likes || 0}</span></button>
                    <button class="vote-btn ${comment.user_vote === -1 ? 'active vote-down' : ''}" data-comment-id="${comment.id}" data-vote-type="-1" title="Dislike"><i class="fas fa-thumbs-down"></i> <span class="vote-count" id="dislikes-count-${comment.id}">${comment.dislikes || 0}</span></button>
                    <button class="reply-btn" data-comment-id="${comment.id}" title="Reply"><i class="fas fa-reply"></i> Reply</button>
                </div>
                <div class="reply-form-container" id="reply-form-container-${comment.id}"><form class="reply-form mt-2"><input type="hidden" name="article_hash_id" value="${currentArticleHashId}"><input type="hidden" name="parent_id" value="${comment.id}"><div class="mb-2"><textarea class="form-control form-control-sm" name="content" rows="2" placeholder="Write a reply..." required></textarea></div><button type="submit" class="btn btn-sm btn-primary-modal">Post Reply</button><button type="button" class="btn btn-sm btn-outline-secondary-modal cancel-reply-btn">Cancel</button></form></div>`;
            }
            return `<div class="comment-container animate-fade-in" id="comment-${comment.id}"><div class="comment-card"><div class="comment-avatar" title="${authorName}">${userInitial}</div><div class="comment-body"><div class="comment-header"><span class="comment-author">${authorName}</span><span class="comment-date">${commentDate}</span></div><p class="comment-content mb-2">${comment.content.replace(/\\n/g, '<br>')}</p>${actionsHTML}</div></div><div class="comment-replies" id="replies-of-${comment.id}"></div></div>`;
        }

        function handleCommentSubmit(form, articleHashId, parentId = null) {
            const content = form.querySelector('textarea[name="content"]').value;
            if (!content.trim()) { alert("Comment cannot be empty."); return; }

            const submitButton = form.querySelector('button[type="submit"]');
            const originalButtonText = submitButton.innerHTML;
            submitButton.disabled = true;
            submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Posting...';

            fetch(`{{ url_for('add_comment', article_hash_id='PLACEHOLDER') }}`.replace('PLACEHOLDER', articleHashId), {
                method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' }, body: JSON.stringify({ content: content, parent_id: parentId })
            })
            .then(res => res.json().then(data => ({ status: res.status, body: data })))
            .then(({ status, body }) => {
                if (status === 201 && body.success) {
                    const newCommentHTML = createCommentHTML(body); // Pass the whole body which contains the comment object
                    const tempDiv = document.createElement('div');
                    tempDiv.innerHTML = newCommentHTML.trim();
                    const newCommentNode = tempDiv.firstChild;

                    if (parentId) {
                        document.getElementById(`replies-of-${parentId}`).appendChild(newCommentNode);
                        form.closest('.reply-form-container').style.display = 'none';
                    } else {
                        const list = document.getElementById('comments-list');
                        const noCommentsMsg = document.getElementById('no-comments-msg');
                        if (noCommentsMsg) noCommentsMsg.remove();
                        list.appendChild(newCommentNode);
                        const countEl = document.getElementById('comment-count');
                        if(countEl) countEl.textContent = parseInt(countEl.textContent || 0) + 1;
                    }
                    form.reset();
                } else {
                    alert('Error: ' + (body.error || 'Unknown error posting comment. Please try again.'));
                }
            }).catch(err => {
                console.error("Comment submission error:", err);
                alert("Could not submit comment: " + err.message);
            }).finally(() => {
                submitButton.disabled = false;
                submitButton.innerHTML = originalButtonText;
            });
        }

        commentSection.addEventListener('click', function(e) {
            const voteBtn = e.target.closest('.vote-btn');
            if (voteBtn && isUserLoggedIn) {
                e.preventDefault();
                const commentId = voteBtn.dataset.commentId;
                const voteType = parseInt(voteBtn.dataset.voteType);
                
                fetch(`{{ url_for('vote_comment', comment_id=0) }}`.replace('0', commentId), {
                    method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' }, body: JSON.stringify({ vote_type: voteType })
                })
                .then(res => res.json().then(data => ({ status: res.status, body: data })))
                .then(({ status, body }) => {
                    if (status === 200 && body.success) {
                        document.getElementById(`likes-count-${commentId}`).textContent = body.likes;
                        document.getElementById(`dislikes-count-${commentId}`).textContent = body.dislikes;
                        
                        const allVoteBtnsOnComment = document.querySelectorAll(`.vote-btn[data-comment-id="${commentId}"]`);
                        allVoteBtnsOnComment.forEach(btn => {
                            btn.classList.remove('active', 'vote-up', 'vote-down');
                        });

                        if (body.user_vote_status === 1) {
                            voteBtn.closest('.comment-actions').querySelector(`.vote-btn[data-vote-type="1"]`).classList.add('active', 'vote-up');
                        } else if (body.user_vote_status === -1) {
                             voteBtn.closest('.comment-actions').querySelector(`.vote-btn[data-vote-type="-1"]`).classList.add('active', 'vote-down');
                        }
                    } else {
                        alert('Error voting: ' + (body.error || 'Unknown error.'));
                    }
                }).catch(err => {
                    console.error("Vote error:", err);
                    alert("Could not process vote: " + err.message);
                });
            }

            const replyBtn = e.target.closest('.reply-btn');
            if (replyBtn && isUserLoggedIn) {
                e.preventDefault();
                const commentId = replyBtn.dataset.commentId;
                const formContainer = document.getElementById(`reply-form-container-${commentId}`);
                if (formContainer) {
                    const isDisplayed = formContainer.style.display === 'block';
                    document.querySelectorAll('.reply-form-container').forEach(fc => { fc.style.display = 'none'; }); 
                    formContainer.style.display = isDisplayed ? 'none' : 'block';
                    if(formContainer.style.display === 'block') { formContainer.querySelector('textarea').focus(); }
                }
            }
            const cancelReplyBtn = e.target.closest('.cancel-reply-btn');
            if (cancelReplyBtn) { cancelReplyBtn.closest('.reply-form-container').style.display = 'none'; }
        });

        commentSection.addEventListener('submit', function(e) {
            const mainCommentForm = e.target.closest('#comment-form');
            if (mainCommentForm) {
                e.preventDefault();
                handleCommentSubmit(mainCommentForm, currentArticleHashId);
            }
            const replyForm = e.target.closest('.reply-form');
            if (replyForm) {
                e.preventDefault();
                handleCommentSubmit(replyForm, currentArticleHashId, replyForm.querySelector('input[name="parent_id"]').value);
            }
        });
    }
});
</script>
{% endblock %}
"""

LOGIN_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Login - Briefly{% endblock %}
{% block content %}
<div class="auth-container article-card animate-fade-in mx-auto">
    <h2 class="auth-title mb-4"><i class="fas fa-sign-in-alt me-2"></i>Member Login</h2>
    <form method="POST" action="{{ url_for('login', next=request.args.get('next') if request and request.args else '') }}">
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
    <p class="mt-3 text-center small">Don't have an account? <a href="{{ url_for('register', next=request.args.get('next') if request and request.args else '') }}" class="fw-medium">Register here</a></p>
</div>
{% endblock %}
"""

REGISTER_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Register - Briefly{% endblock %}
{% block content %}
<div class="auth-container article-card animate-fade-in mx-auto">
    <h2 class="auth-title mb-4"><i class="fas fa-user-plus me-2"></i>Create Account</h2>
    <form method="POST" action="{{ url_for('register') }}">
        <div class="modal-form-group">
            <label for="name" class="form-label">Full Name</label>
            <input type="text" class="modal-form-control" id="name" name="name" required placeholder="Enter your full name" value="{{ request.form.name if request and request.form else '' }}">
        </div>
        <div class="modal-form-group">
            <label for="username" class="form-label">Username</label>
            <input type="text" class="modal-form-control" id="username" name="username" required placeholder="Choose a username (min 3 chars)" value="{{ request.form.username if request and request.form else '' }}">
        </div>
        <div class="modal-form-group">
            <label for="password" class="form-label">Password</label>
            <input type="password" class="modal-form-control" id="password" name="password" required placeholder="Create a strong password (min 6 chars)">
        </div>
        <button type="submit" class="btn btn-primary-modal w-100 mt-3">Register</button>
    </form>
    <p class="mt-3 text-center small">Already have an account? <a href="{{ url_for('login') }}" class="fw-medium">Login here</a></p>
</div>
{% endblock %}
"""

ABOUT_US_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}About Us - Briefly{% endblock %}
{% block content %}
<div class="static-content-wrapper article-card animate-fade-in">
    <h1 class="mb-4">About Briefly</h1>
    <p class="lead">Briefly is your premier destination for the latest news from India and around the world, delivered in a concise and easy-to-digest format. We leverage the power of cutting-edge AI to summarize complex news articles into key takeaways, saving you time while keeping you informed.</p>
    <h2 class="mt-5 mb-3">Our Mission</h2>
    <p>In a world of information overload, our mission is to provide clarity and efficiency. We believe that everyone deserves access to accurate, unbiased news without spending hours sifting through lengthy articles. Briefly cuts through the noise, offering insightful summaries that matter.</p>
    <h2 class="mt-5 mb-3">Community Hub</h2>
    <p>Beyond AI-driven news, Briefly is a platform for discussion and community engagement. Our Community Hub allows users to post their own articles, share perspectives, and engage in meaningful conversations about the topics that shape our world. We are committed to fostering a respectful and intelligent environment for all our members.</p>
    <h2 class="mt-5 mb-3">Our Technology</h2>
    <p>We use state-of-the-art Natural Language Processing (NLP) models to analyze and summarize news content from trusted sources. Our system is designed to identify the most crucial points of an article, presenting them as a quick summary and a list of key takeaways. This service is active when AI analysis is enabled via API keys.</p>
</div>
{% endblock %}
"""

CONTACT_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Contact Us - Briefly{% endblock %}
{% block content %}
<div class="static-content-wrapper article-card animate-fade-in">
    <h1 class="mb-4">Contact Us</h1>
    <p class="lead">We'd love to hear from you! Whether you have a question, feedback, or a news tip, feel free to reach out.</p>
    <div class="row mt-5">
        <div class="col-md-6">
            <h2 class="h4">General Inquiries</h2>
            <p>For general questions, feedback, or support, please email us at:</p>
            <p><i class="fas fa-envelope me-2"></i><a href="mailto:contact@example.com">contact@example.com</a></p>
        </div>
        <div class="col-md-6">
            <h2 class="h4">Partnerships & Media</h2>
            <p>For partnership opportunities or media inquiries, please contact:</p>
            <p><i class="fas fa-envelope me-2"></i><a href="mailto:partners@example.com">partners@example.com</a></p>
        </div>
    </div>
    <div class="mt-5">
        <h2 class="h4">Follow Us</h2>
        <p>Stay connected with us on social media:</p>
        <div class="social-links fs-4 d-flex gap-3">
            <a href="#" title="Twitter" class="text-decoration-none"><i class="fab fa-twitter"></i></a>
            <a href="#" title="Facebook" class="text-decoration-none"><i class="fab fa-facebook-f"></i></a>
            <a href="#" title="LinkedIn" class="text-decoration-none"><i class="fab fa-linkedin-in"></i></a>
            <a href="#" title="Instagram" class="text-decoration-none"><i class="fab fa-instagram"></i></a>
        </div>
    </div>
</div>
{% endblock %}
"""

PRIVACY_POLICY_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Privacy Policy - Briefly{% endblock %}
{% block content %}
<div class="static-content-wrapper article-card animate-fade-in">
    <h1 class="mb-4">Privacy Policy</h1>
    <p class="text-muted">Last updated: May 30, 2025</p>
    <p>Briefly ("we," "our," or "us") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you visit our website.</p>
    <h2 class="mt-5 mb-3">1. Information We Collect</h2>
    <p>We may collect personal information that you voluntarily provide to us when you register on the website, post articles or comments, or subscribe to our newsletter. This information may include your name, username, and email address. We also collect information automatically, such as your IP address and Browse behavior, through cookies and similar technologies.</p>
    <h2 class="mt-5 mb-3">2. How We Use Your Information</h2>
    <p>We use the information we collect to:</p>
    <ul>
        <li>Create and manage your account.</li>
        <li>Operate and maintain the website.</li>
        <li>Personalize your experience.</li>
        <li>Send you newsletters or promotional materials, if you have opted in.</li>
        <li>Respond to your comments and inquiries.</li>
        <li>Improve our website and services.</li>
        <li>Monitor and analyze usage and trends.</li>
    </ul>
    <h2 class="mt-5 mb-3">3. Disclosure of Your Information</h2>
    <p>We do not sell, trade, or otherwise transfer your personally identifiable information to outside parties without your consent, except as described in this policy. This does not include trusted third parties who assist us in operating our website (e.g., hosting providers, analytics services), so long as those parties agree to keep this information confidential. We may also release information when its release is appropriate to comply with the law, enforce our site policies, or protect ours or others' rights, property, or safety.</p>
    <h2 class="mt-5 mb-3">4. Cookies and Tracking Technologies</h2>
    <p>We use cookies to enhance your experience, remember your preferences (like dark mode), and gather anonymous usage data. You can control the use of cookies at the individual browser level.</p>
    <h2 class="mt-5 mb-3">5. Security of Your Information</h2>
    <p>We use administrative, technical, and physical security measures to help protect your personal information. While we have taken reasonable steps to secure the personal information you provide to us, please be aware that despite our efforts, no security measures are perfect or impenetrable, and no method of data transmission can be guaranteed against any interception or other type of misuse.</p>
    <h2 class="mt-5 mb-3">6. Your Data Rights</h2>
    <p>Depending on your jurisdiction, you may have rights to access, correct, delete, or restrict the use of your personal information. Please contact us to make such requests.</p>
    <h2 class="mt-5 mb-3">7. Changes to This Privacy Policy</h2>
    <p>We may update this Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page and updating the "Last updated" date. You are advised to review this Privacy Policy periodically for any changes.</p>
    <h2 class="mt-5 mb-3">8. Contact Us</h2>
    <p>If you have any questions about this Privacy Policy, please contact us at <a href="mailto:privacy@example.com">privacy@example.com</a>.</p>
</div>
{% endblock %}
"""

ERROR_404_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}404 Not Found - Briefly{% endblock %}{% block content %}<div class='text-center my-5 p-4 article-card animate-fade-in mx-auto' style='max-width: 600px;'><h1><i class='fas fa-ghost text-warning me-2'></i>404 - Page Not Found</h1><p class='lead'>Oops! The page you are looking for does not exist or has been moved.</p><p>Perhaps you were looking for one of these?</p><ul class="list-unstyled mt-3"><li class="mb-1"><a href='{{url_for("index")}}' class='btn btn-sm btn-outline-primary'>Go to Homepage</a></li>{% if categories is defined %}{% for cat_item in categories %}<li class="mb-1"><a href="{{ url_for('index', category_name=cat_item) }}" class="btn btn-sm btn-outline-secondary">{{cat_item}}</a></li>{% endfor %}{% endif %}</ul></div>{% endblock %}"""
ERROR_500_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}500 Server Error - Briefly{% endblock %}{% block content %}<div class='text-center my-5 p-4 article-card animate-fade-in mx-auto' style='max-width: 600px;'><h1><i class='fas fa-cogs text-danger me-2'></i>500 - Internal Server Error</h1><p class='lead'>Yikes! Something went wrong on our end. We've been notified and are looking into it.</p><p>Please try again in a few moments, or <a href='{{url_for("index")}}' class=''>return to the homepage</a>.</p></div>{% endblock %}"""


# ==============================================================================
# --- 8. Add all templates to the template_storage dictionary ---
# ==============================================================================
template_storage['BASE_HTML_TEMPLATE'] = BASE_HTML_TEMPLATE
template_storage['INDEX_HTML_TEMPLATE'] = INDEX_HTML_TEMPLATE
template_storage['ARTICLE_HTML_TEMPLATE'] = ARTICLE_HTML_TEMPLATE
template_storage['LOGIN_HTML_TEMPLATE'] = LOGIN_HTML_TEMPLATE
template_storage['REGISTER_HTML_TEMPLATE'] = REGISTER_HTML_TEMPLATE
template_storage['ABOUT_US_HTML_TEMPLATE'] = ABOUT_US_HTML_TEMPLATE
template_storage['CONTACT_HTML_TEMPLATE'] = CONTACT_HTML_TEMPLATE
template_storage['PRIVACY_POLICY_HTML_TEMPLATE'] = PRIVACY_POLICY_HTML_TEMPLATE
template_storage['404_TEMPLATE'] = ERROR_404_TEMPLATE
template_storage['500_TEMPLATE'] = ERROR_500_TEMPLATE

# ==============================================================================
# --- 9. App Context & Main Execution Block ---
# ==============================================================================
with app.app_context():
    init_db()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080)) 
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    app.logger.info(f"Starting Flask app in {'debug' if debug_mode else 'production'} mode on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
