#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import hashlib
import time
import logging
import urllib.parse
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import Response

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
app.config['CATEGORIES'] = ['All Articles', 'Popular Stories', "Yesterday's Headlines", 'Community Hub']

app.config['NEWS_API_QUERY'] = 'India OR "Indian politics" OR "Indian economy" OR "Bollywood"'
app.config['NEWS_API_DOMAINS'] = 'timesofindia.indiatimes.com,thehindu.com,ndtv.com,indianexpress.com,hindustantimes.com'
app.config['NEWS_API_DAYS_AGO'] = 7 
app.config['NEWS_API_PAGE_SIZE'] = 100
app.config['NEWS_API_SORT_BY'] = 'publishedAt' 
app.config['CACHE_EXPIRY_SECONDS'] = 1800 
app.permanent_session_lifetime = timedelta(days=30)

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

using_postgres_flag = False
database_url = os.environ.get('DATABASE_URL')

if database_url and (database_url.startswith("postgres://") or database_url.startswith("postgresql://")):
    if database_url.startswith("postgres://"):
        configured_db_uri = database_url.replace("postgres://", "postgresql://", 1)
    else: 
        configured_db_uri = database_url
    app.config['SQLALCHEMY_DATABASE_URI'] = configured_db_uri
    using_postgres_flag = True 
else:
    db_file_name = 'app_data.db'
    try:
        project_root_for_db = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(project_root_for_db, db_file_name)
    except NameError: 
        db_path = db_file_name 
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app) 

# ==============================================================================
# --- 3. API Client Initialization ---
# ==============================================================================
NEWSAPI_KEY = os.environ.get('NEWSAPI_KEY')
newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = ChatGroq(model="qwen/qwen3.8-27b", groq_api_key=GROQ_API_KEY, temperature=0.1)
    except Exception as e:
        app.logger.error(f"Failed to initialize Groq client: {e}")

SCRAPER_API_KEY = os.environ.get('SCRAPER_API_KEY')

# ==============================================================================
# --- 4. Database Models ---
# ==============================================================================
class ReportedArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    article_id = db.Column(db.Integer, db.ForeignKey('community_article.id', ondelete="CASCADE"), nullable=False)
    reporter_user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False)
    reason = db.Column(db.String(250), nullable=True)
    status = db.Column(db.String(20), nullable=False, default='pending') 
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    __table_args__ = (db.UniqueConstraint('article_id', 'reporter_user_id', name='_article_reporter_uc'),)
    article = db.relationship('CommunityArticle', backref=db.backref('reports', lazy='dynamic', cascade="all, delete-orphan"))
    
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    articles = db.relationship('CommunityArticle', backref='author', lazy='dynamic', cascade="all, delete-orphan")
    comments = db.relationship('Comment', backref=db.backref('author', lazy='joined'), lazy='dynamic', cascade="all, delete-orphan")
    comment_votes = db.relationship('CommentVote', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    bookmarks = db.relationship('BookmarkedArticle', backref='user', lazy='dynamic', cascade="all, delete-orphan")

class CommunityArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    article_hash_id = db.Column(db.String(32), unique=True, nullable=False, index=True)
    title = db.Column(db.String(250), nullable=False)
    description = db.Column(db.Text, nullable=False)
    full_text = db.Column(db.Text, nullable=False)
    source_name = db.Column(db.String(100), nullable=False)
    image_url = db.Column(db.String(500), nullable=True)
    published_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    groq_summary = db.Column(db.Text, nullable=True)
    groq_takeaways = db.Column(db.Text, nullable=True) 
    comments = db.relationship('Comment', backref=db.backref('community_article', lazy='joined'), lazy='dynamic', foreign_keys='Comment.community_article_id', cascade="all, delete-orphan")

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
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
    vote_emoji = db.Column(db.String(10), nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'comment_id', name='_user_comment_uc'),)

class Subscriber(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    subscribed_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

class BookmarkedArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False)
    article_hash_id = db.Column(db.String(32), nullable=False, index=True)
    is_community_article = db.Column(db.Boolean, default=False, nullable=False)
    title_cache = db.Column(db.String(250), nullable=True)
    source_name_cache = db.Column(db.String(100), nullable=True)
    image_url_cache = db.Column(db.String(500), nullable=True)
    description_cache = db.Column(db.Text, nullable=True)
    published_at_cache = db.Column(db.DateTime, nullable=True) 
    bookmarked_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    __table_args__ = (db.UniqueConstraint('user_id', 'article_hash_id', name='_user_article_bookmark_uc'),)

def init_db():
    global using_postgres_flag
    with app.app_context():
        try:
            db.create_all()
        except Exception as e:
            app.logger.error(f"Error during db.create_all(): {e}", exc_info=True)

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
            if utc_dt.endswith('Z'):
                 utc_dt = datetime.fromisoformat(utc_dt[:-1] + '+00:00')
            else:
                 utc_dt = datetime.fromisoformat(utc_dt)
        except ValueError: return "Invalid date string"
    if not isinstance(utc_dt, datetime): return "Invalid date object"
    if utc_dt.tzinfo is None: utc_dt = pytz.utc.localize(utc_dt)
    else: utc_dt = utc_dt.astimezone(pytz.utc)
    ist_dt = utc_dt.astimezone(INDIAN_TIMEZONE)
    return ist_dt.strftime('%b %d, %Y at %I:%M %p %Z')
app.jinja_env.filters['to_ist'] = to_ist_filter

def simple_cache(expiry_seconds_default=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            expiry = expiry_seconds_default or app.config['CACHE_EXPIRY_SECONDS']
            key_parts = [func.__name__] + list(map(str, args)) + sorted(kwargs.items())
            cache_key = hashlib.md5(str(key_parts).encode('utf-8')).hexdigest()
            cached_entry = API_CACHE.get(cache_key)
            if cached_entry and (time.time() - cached_entry[1] < expiry):
                return cached_entry[0]
            result = func(*args, **kwargs)
            API_CACHE[cache_key] = (result, time.time())
            return result
        return wrapper
    return decorator

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.headers.get('Accept') == 'application/json':
                return jsonify({"success": False, "error": "Authentication required. Please log in."}), 401
            else:
                flash("You must be logged in to access this page.", "warning")
                return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@simple_cache(expiry_seconds_default=3600 * 12)
def get_article_analysis_with_groq(article_text, article_title=""):
    if not groq_client: return {"error": "AI analysis service not available."}
    if not article_text or not article_text.strip(): return {"error": "No text provided for AI analysis."}
    system_prompt = ("You are an expert news analyst. Analyze the following article. "
        "1. Provide a concise, neutral summary (3-4 paragraphs). "
        "2. List 5-7 key takeaways as bullet points. Each takeaway must be a complete sentence. "
        "Format your entire response as a single JSON object with keys 'summary' (string) and 'takeaways' (a list of strings).")
    human_prompt = f"Article Title: {article_title}\n\nArticle Text:\n{article_text[:20000]}"
    try:
        json_model = groq_client.bind(response_format={"type": "json_object"})
        ai_response = json_model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        analysis = json.loads(ai_response.content) 
        return {"groq_summary": analysis.get("summary"), "groq_takeaways": analysis.get("takeaways"), "error": None}
    except Exception as e:
        return {"error": f"AI analysis failed: {str(e)}"}

@simple_cache(expiry_seconds_default=14400)
def get_daily_synthesis():
    if not groq_client: return {"synthesis_text": None, "keywords": []}
    articles_to_synthesize = fetch_popular_news()
    if not articles_to_synthesize: return {"synthesis_text": None, "keywords": []}
    content_for_ai = "".join([f"Title: {art.get('title', '')}\\nDescription: {art.get('description', '')}\\n\\n" for art in articles_to_synthesize[:15]])
    system_prompt = (
        "You are a top-tier news editor for an Indian audience. Your task is to provide a 'big picture' summary of the day's news based on a collection of article titles and descriptions. "
        "Analyze the provided text and generate a JSON object with two keys: 'synthesis_text' and 'keywords'. "
        "1. For 'synthesis_text': Write a single, insightful, and cohesive paragraph (3-4 sentences) synthesizing important themes. "
        "2. For 'keywords': Extract the 4-5 most significant keywords."
    )
    human_prompt = f"Here is the collection of today's news articles:\n\n{content_for_ai}"
    try:
        json_model = groq_client.bind(response_format={"type": "json_object"})
        ai_response = json_model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        analysis = json.loads(ai_response.content)
        return {"synthesis_text": analysis.get("synthesis_text"), "keywords": analysis.get("keywords")}
    except Exception as e:
        return {"synthesis_text": "The AI summary could not be generated at this time.", "keywords": []}

@simple_cache()
def fetch_news_from_api(target_date_str=None):
    if not newsapi: return []
    api_call_from_date_str, api_call_to_date_str, is_specific_date_fetch = None, None, False
    if target_date_str:
        try:
            local_day_start_naive = datetime.strptime(target_date_str, '%Y-%m-%d')
            local_day_start_aware_ist = INDIAN_TIMEZONE.localize(local_day_start_naive) 
            local_day_end_aware_ist = local_day_start_aware_ist.replace(hour=23, minute=59, second=59, microsecond=999999)
            api_call_from_utc_dt = local_day_start_aware_ist.astimezone(timezone.utc)
            api_call_to_utc_dt = local_day_end_aware_ist.astimezone(timezone.utc)
            api_call_from_date_str = api_call_from_utc_dt.strftime('%Y-%m-%dT%H:%M:%S')
            api_call_to_date_str = api_call_to_utc_dt.strftime('%Y-%m-%dT%H:%M:%S')
            is_specific_date_fetch = True
        except Exception:
            target_date_str = None
            is_specific_date_fetch = False
    if not is_specific_date_fetch: 
        from_date_utc_default = datetime.now(timezone.utc) - timedelta(days=app.config['NEWS_API_DAYS_AGO'])
        api_call_from_date_str = from_date_utc_default.strftime('%Y-%m-%dT%H:%M:%S')
        current_day_utc_end_default = datetime.now(timezone.utc).replace(hour=23, minute=59, second=59, microsecond=0)
        api_call_to_date_str = current_day_utc_end_default.strftime('%Y-%m-%dT%H:%M:%S')

    all_raw_articles = []
    if not is_specific_date_fetch:
        try:
            top_headlines_response = newsapi.get_top_headlines(country='in', language='en', page_size=app.config['NEWS_API_PAGE_SIZE'])
            if top_headlines_response.get('status') == 'ok': all_raw_articles.extend(top_headlines_response.get('articles', []))
        except Exception: pass

    try:
        everything_response = newsapi.get_everything(q=app.config['NEWS_API_QUERY'], from_param=api_call_from_date_str, to=api_call_to_date_str, language='en', sort_by=app.config['NEWS_API_SORT_BY'], page_size=app.config['NEWS_API_PAGE_SIZE'])
        if everything_response.get('status') == 'ok': all_raw_articles.extend(everything_response.get('articles', []))
    except Exception: pass

    if not all_raw_articles or is_specific_date_fetch:
        try:
            fallback_response = newsapi.get_everything(domains=app.config['NEWS_API_DOMAINS'], from_param=api_call_from_date_str, to=api_call_to_date_str, language='en', sort_by=app.config['NEWS_API_SORT_BY'], page_size=app.config['NEWS_API_PAGE_SIZE'])
            if fallback_response.get('status') == 'ok': all_raw_articles.extend(fallback_response.get('articles', []))
        except Exception: pass

    processed_articles, unique_urls = [], set()
    for art_data in all_raw_articles:
        url = art_data.get('url')
        if not url or url in unique_urls: continue
        title, description = art_data.get('title'), art_data.get('description')
        if not all([title, art_data.get('source'), description]) or title == '[Removed]' or not title.strip() or not description.strip(): continue
        unique_urls.add(url)
        article_id = generate_article_id(url)
        source_name = art_data['source'].get('name', 'Unknown Source')
        placeholder_text = urllib.parse.quote_plus(source_name[:20])
        
        try: published_at_dt = datetime.fromisoformat(art_data.get('publishedAt', '').replace('Z', '+00:00'))
        except (ValueError, TypeError): published_at_dt = datetime.now(timezone.utc)
        
        standardized_article = {
            'id': article_id, 'title': title, 'description': description, 'url': url,
            'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/1a1a2e/FFFFFF?text={placeholder_text}',
            'publishedAt': published_at_dt.isoformat(), 'source': {'name': source_name}, 'is_community_article': False,
            'groq_summary': None, 'groq_takeaways': None
        }
        MASTER_ARTICLE_STORE[article_id] = standardized_article
        processed_articles.append(standardized_article)
    
    processed_articles.sort(key=lambda x: x.get('publishedAt', datetime.min.replace(tzinfo=timezone.utc).isoformat()), reverse=True)
    return processed_articles

@simple_cache(expiry_seconds_default=21600) 
def fetch_popular_news():
    if not newsapi: return []
    to_date = datetime.now(timezone.utc) - timedelta(days=1)
    from_date = to_date - timedelta(days=5)
    try:
        response = newsapi.get_everything(q=app.config['NEWS_API_QUERY'], language='en', from_param=from_date.strftime('%Y-%m-%d'), to=to_date.strftime('%Y-%m-%d'), sort_by='popularity', page_size=30)
        if response.get('status') == 'ok':
            processed_articles, unique_urls = [], set()
            for art_data in response.get('articles', []):
                url = art_data.get('url')
                if not url or url in unique_urls: continue
                title, description = art_data.get('title'), art_data.get('description')
                if not all([title, art_data.get('source'), description]) or title == '[Removed]' or not title.strip() or not description.strip(): continue
                unique_urls.add(url)
                article_id = generate_article_id(url)
                source_name = art_data['source'].get('name', 'Unknown Source')
                try: published_at_dt = datetime.fromisoformat(art_data.get('publishedAt', '').replace('Z', '+00:00'))
                except (ValueError, TypeError): published_at_dt = datetime.now(timezone.utc)
                standardized_article = {
                    'id': article_id, 'title': title, 'description': description, 'url': url,
                    'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/1a1a2e/FFFFFF?text={urllib.parse.quote_plus(source_name[:20])}',
                    'publishedAt': published_at_dt.isoformat(), 'source': {'name': source_name}, 'is_community_article': False
                }
                MASTER_ARTICLE_STORE[article_id] = standardized_article
                processed_articles.append(standardized_article)
            return processed_articles
    except Exception: pass
    return []

@simple_cache(expiry_seconds_default=14400) 
def fetch_yesterdays_latest_news():
    if not newsapi: return []
    now_in_ist = datetime.now(INDIAN_TIMEZONE)
    yesterday_in_ist = now_in_ist - timedelta(days=1)
    start_utc = yesterday_in_ist.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(pytz.utc)
    end_utc = yesterday_in_ist.replace(hour=23, minute=59, second=59, microsecond=999999).astimezone(pytz.utc)
    try:
        response = newsapi.get_everything(q=app.config['NEWS_API_QUERY'], language='en', from_param=start_utc.strftime('%Y-%m-%dT%H:%M:%S'), to=end_utc.strftime('%Y-%m-%dT%H:%M:%S'), sort_by='publishedAt', page_size=100)
        if response.get('status') == 'ok':
            processed_articles, unique_urls = [], set()
            for art_data in response.get('articles', []):
                url = art_data.get('url')
                if not url or url in unique_urls: continue
                title, description = art_data.get('title'), art_data.get('description')
                if not all([title, art_data.get('source'), description]) or title == '[Removed]' or not title.strip(): continue
                unique_urls.add(url)
                article_id = generate_article_id(url)
                try: published_at_dt = datetime.fromisoformat(art_data.get('publishedAt', '').replace('Z', '+00:00'))
                except (ValueError, TypeError): published_at_dt = datetime.now(timezone.utc)
                standardized_article = {
                    'id': article_id, 'title': title, 'description': description, 'url': url,
                    'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/1a1a2e/FFFFFF?text=News',
                    'publishedAt': published_at_dt.isoformat(), 'source': {'name': art_data['source'].get('name', 'Unknown Source')}, 'is_community_article': False
                }
                MASTER_ARTICLE_STORE[article_id] = standardized_article
                processed_articles.append(standardized_article)
            return processed_articles
    except Exception: pass
    return []
        
@simple_cache(expiry_seconds_default=3600 * 6)
def fetch_and_parse_article_content(article_hash_id, url):
    if not SCRAPER_API_KEY: return {"full_text": None, "groq_analysis": None, "error": "Content fetching service unavailable."}
    try:
        response = requests.get('http://api.scraperapi.com', params={'api_key': SCRAPER_API_KEY, 'url': url}, timeout=45)
        response.raise_for_status() 
        article_scraper = Article(url, config=Config())
        article_scraper.download(input_html=response.text)
        article_scraper.parse()
        if not article_scraper.text: return {"full_text": None, "groq_analysis": None, "error": "Could not extract text."}
        article_title_for_groq = article_scraper.title or MASTER_ARTICLE_STORE.get(article_hash_id, {}).get('title', 'Unknown Title')
        
        groq_analysis_result = None
        if article_hash_id in MASTER_ARTICLE_STORE and MASTER_ARTICLE_STORE[article_hash_id].get('groq_summary'):
            groq_analysis_result = {"groq_summary": MASTER_ARTICLE_STORE[article_hash_id]['groq_summary'], "groq_takeaways": MASTER_ARTICLE_STORE[article_hash_id]['groq_takeaways'], "error": None}
        else:
            groq_analysis_result = get_article_analysis_with_groq(article_scraper.text, article_title_for_groq)
            if article_hash_id in MASTER_ARTICLE_STORE and groq_analysis_result and not groq_analysis_result.get("error"):
                MASTER_ARTICLE_STORE[article_hash_id]['groq_summary'] = groq_analysis_result.get("groq_summary")
                MASTER_ARTICLE_STORE[article_hash_id]['groq_takeaways'] = groq_analysis_result.get("groq_takeaways")
        return {"full_text": article_scraper.text, "groq_analysis": groq_analysis_result, "error": None}
    except Exception as e:
        return {"full_text": None, "groq_analysis": None, "error": f"Failed to parse content: {str(e)}"}
    
# ==============================================================================
# --- 6. Flask Routes ---
# ==============================================================================
@app.context_processor
def inject_global_vars():
    return {'categories': app.config['CATEGORIES'], 'current_year': datetime.utcnow().year, 'session': session, 'request': request, 'groq_client': groq_client is not None}

def get_paginated_articles(articles, page, per_page):
    total = len(articles)
    start = (page - 1) * per_page
    return articles[start:start + per_page], ((total + per_page - 1) // per_page if per_page > 0 else 0)

def get_sort_key(article):
    date_val = article.get('publishedAt') if isinstance(article, dict) else getattr(article, 'published_at', None)
    if not date_val: return datetime.min.replace(tzinfo=timezone.utc)
    if isinstance(date_val, str):
        try:
            return datetime.fromisoformat(date_val.replace('Z', '+00:00')) if not date_val.endswith('Z') else datetime.fromisoformat(date_val[:-1] + '+00:00')
        except ValueError: return datetime.min.replace(tzinfo=timezone.utc)
    return date_val if date_val.tzinfo else pytz.utc.localize(date_val)

@app.route('/report_article/<article_hash_id>', methods=['POST'])
@login_required
def report_article(article_hash_id):
    article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first_or_404()
    if ReportedArticle.query.filter_by(article_id=article.id, reporter_user_id=session['user_id']).first():
        return jsonify({"success": False, "error": "You have already reported this article."}), 409 
    try:
        db.session.add(ReportedArticle(article_id=article.id, reporter_user_id=session['user_id']))
        db.session.commit()
        return jsonify({"success": True, "message": "Article reported for review."})
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "error": "Database error."}), 500

@app.route('/')
@app.route('/page/<int:page>')
@app.route('/category/<category_name>')
@app.route('/category/<category_name>/page/<int:page>')
def index(page=1, category_name='All Articles'):
    session['previous_list_page'] = request.full_path
    per_page, query_str, filter_date_str = app.config['PER_PAGE'], request.args.get('query'), request.args.get('filter_date')

    if page == 1 and category_name == 'All Articles' and not query_str and not filter_date_str:
        synthesis_data = get_daily_synthesis()
        all_popular_articles = fetch_popular_news()
        featured_article = all_popular_articles[0] if all_popular_articles else None
        popular_articles = all_popular_articles[1:7] if all_popular_articles else [] 
        latest_yesterday_articles = fetch_yesterdays_latest_news()[:6]
        
        user_bookmarks_hashes = {b.article_hash_id for b in BookmarkedArticle.query.filter_by(user_id=session['user_id']).all()} if 'user_id' in session else set()

        for art in ([featured_article] + popular_articles + latest_yesterday_articles):
            if art: art['is_bookmarked'] = art.get('id') in user_bookmarks_hashes

        return render_template("INDEX_HTML_TEMPLATE", synthesis=synthesis_data.get('synthesis_text'), keywords=synthesis_data.get('keywords', []), featured_article=featured_article, popular_articles=popular_articles, latest_yesterday_articles=latest_yesterday_articles, selected_category=category_name, is_main_homepage=True, current_page=1, total_pages=1, query=None, current_filter_date=None)

    else:
        all_display_articles_raw = []
        if category_name == 'Popular Stories': all_display_articles_raw = fetch_popular_news()
        elif category_name == "Yesterday's Headlines": all_display_articles_raw = fetch_yesterdays_latest_news()
        elif category_name == 'Community Hub':
            db_articles = CommunityArticle.query.options(joinedload(CommunityArticle.author)).order_by(CommunityArticle.published_at.desc()).all()
            for art in db_articles: art.is_community_article = True
            all_display_articles_raw.extend(db_articles)
        else:
            api_articles = fetch_news_from_api(target_date_str=filter_date_str)
            all_display_articles_raw.extend(api_articles)
            
        all_display_articles_raw.sort(key=get_sort_key, reverse=True)
        paginated_display_articles_raw, total_pages = get_paginated_articles(all_display_articles_raw, page, per_page)
        
        user_bookmarks_hashes = {b.article_hash_id for b in BookmarkedArticle.query.filter_by(user_id=session['user_id']).all()} if 'user_id' in session else set()
        paginated_display_articles_with_bookmark_status = []
        for art_item in paginated_display_articles_raw:
            if hasattr(art_item, 'is_community_article') and art_item.is_community_article:
                art_item.is_bookmarked = art_item.article_hash_id in user_bookmarks_hashes
                paginated_display_articles_with_bookmark_status.append(art_item)
            elif isinstance(art_item, dict):
                art_item_copy = art_item.copy()
                art_item_copy['is_bookmarked'] = art_item_copy.get('id') in user_bookmarks_hashes
                paginated_display_articles_with_bookmark_status.append(art_item_copy)
                
        return render_template("INDEX_HTML_TEMPLATE", articles=paginated_display_articles_with_bookmark_status, selected_category=category_name, is_main_homepage=False, current_page=page, total_pages=total_pages, featured_article_on_this_page=False, current_filter_date=filter_date_str, query=query_str)

@app.route('/user/<username>')
def public_profile(username):
    user = User.query.filter_by(username=username).first_or_404()
    posted_articles = CommunityArticle.query.filter_by(user_id=user.id).order_by(CommunityArticle.published_at.desc()).all()
    return render_template("PUBLIC_PROFILE_HTML_TEMPLATE", user=user, posted_articles=posted_articles)
    
@app.route('/search')
@app.route('/search/page/<int:page>')
def search_results(page=1):
    session['previous_list_page'] = request.full_path
    query_str = request.args.get('query', '').strip()
    if not query_str: return redirect(url_for('index'))
    
    api_articles = []
    if newsapi:
        try:
            search_response = newsapi.get_everything(q=query_str, language='en', sort_by='relevancy', page_size=100)
            if search_response.get('status') == 'ok':
                unique_urls = set()
                for art_data in search_response.get('articles', []):
                    url = art_data.get('url')
                    title, description = art_data.get('title'), art_data.get('description')
                    if not url or url in unique_urls or not all([title, description, art_data.get('source')]) or title == '[Removed]': continue
                    unique_urls.add(url)
                    article_id = generate_article_id(url)
                    try: published_at_dt = datetime.fromisoformat(art_data.get('publishedAt', '').replace('Z', '+00:00'))
                    except (ValueError, TypeError): published_at_dt = datetime.now(timezone.utc)
                    standardized_article = {
                        'id': article_id, 'title': title, 'description': description, 'url': url,
                        'urlToImage': art_data.get('urlToImage') or 'https://via.placeholder.com/400x220/1a1a2e/FFFFFF?text=Search+Result',
                        'publishedAt': published_at_dt.isoformat(), 'source': {'name': art_data['source'].get('name', 'Unknown')}, 'is_community_article': False
                    }
                    MASTER_ARTICLE_STORE[article_id] = standardized_article 
                    api_articles.append(standardized_article)
        except Exception: flash("Error communicating with news API.", "danger")
    
    community_db_articles = []
    for art in CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter(db.or_(CommunityArticle.title.ilike(f'%{query_str}%'), CommunityArticle.description.ilike(f'%{query_str}%'))).order_by(CommunityArticle.published_at.desc()).all():
        art.is_community_article = True
        community_db_articles.append(art)

    all_search_results_raw = api_articles + community_db_articles
    all_search_results_raw.sort(key=get_sort_key, reverse=True)
    paginated_search, total_pages = get_paginated_articles(all_search_results_raw, page, app.config['PER_PAGE'])

    user_bookmarks_hashes = {b.article_hash_id for b in BookmarkedArticle.query.filter_by(user_id=session['user_id']).all()} if 'user_id' in session else set()
    for art_item in paginated_search:
        if isinstance(art_item, dict): art_item['is_bookmarked'] = art_item.get('id') in user_bookmarks_hashes
        else: art_item.is_bookmarked = art_item.article_hash_id in user_bookmarks_hashes

    return render_template("INDEX_HTML_TEMPLATE", articles=paginated_search, selected_category=f"Search: {query_str}", current_page=page, total_pages=total_pages, is_main_homepage=False, query=query_str, current_filter_date=None)

@app.route('/article/<article_hash_id>')
def article_detail(article_hash_id):
    article_data, is_community_article, is_bookmarked = None, False, False
    previous_list_page = session.get('previous_list_page', url_for('index'))

    article_db = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter_by(article_hash_id=article_hash_id).first()
    if article_db:
        article_data, is_community_article = article_db, True
        article_data.parsed_takeaways = json.loads(article_data.groq_takeaways) if article_data.groq_takeaways else []
    else:
        if not MASTER_ARTICLE_STORE: fetch_news_from_api()
        if article_hash_id in MASTER_ARTICLE_STORE:
            article_data, is_community_article = MASTER_ARTICLE_STORE[article_hash_id].copy(), False
        else:
            flash("Article not found.", "danger"); return redirect(previous_list_page)

    if 'user_id' in session:
        is_bookmarked = bool(BookmarkedArticle.query.filter_by(user_id=session['user_id'], article_hash_id=article_hash_id).first())

    comment_data, total_comment_count = {}, 0
    base_comments_query = Comment.query.options(joinedload(Comment.author)).filter_by(community_article_id=article_data.id) if is_community_article else Comment.query.options(joinedload(Comment.author)).filter_by(api_article_hash_id=article_hash_id)

    all_comments_in_thread = base_comments_query.all()
    total_comment_count = len(all_comments_in_thread)
    all_comment_ids = {c.id for c in all_comments_in_thread}

    if all_comment_ids:
        for c_id in all_comment_ids: comment_data[c_id] = {'reactions': {}, 'user_reaction': None}
        for c_id, emoji, count in db.session.query(CommentVote.comment_id, CommentVote.vote_emoji, func.count(CommentVote.vote_emoji)).filter(CommentVote.comment_id.in_(all_comment_ids)).group_by(CommentVote.comment_id, CommentVote.vote_emoji).all():
            if c_id in comment_data: comment_data[c_id]['reactions'][emoji] = count
        if 'user_id' in session:
            for vote in CommentVote.query.filter(CommentVote.comment_id.in_(all_comment_ids), CommentVote.user_id==session['user_id']).all():
                if vote.comment_id in comment_data: comment_data[vote.comment_id]['user_reaction'] = vote.vote_emoji

    comments_for_template = base_comments_query.filter(Comment.parent_id.is_(None)).order_by(Comment.timestamp.asc()).all()

    if isinstance(article_data, dict): article_data['is_community_article'] = False
    elif article_data: article_data.is_community_article = True
            
    return render_template("ARTICLE_HTML_TEMPLATE", article=article_data, is_community_article=is_community_article, comments=comments_for_template, comment_data=comment_data, total_comment_count=total_comment_count, previous_list_page=previous_list_page, is_bookmarked=is_bookmarked)

@app.route('/get_article_content/<article_hash_id>')
def get_article_content_json(article_hash_id):
    if not MASTER_ARTICLE_STORE and not CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first(): fetch_news_from_api()
    article_data = MASTER_ARTICLE_STORE.get(article_hash_id)
    if not article_data or 'url' not in article_data: return jsonify({"error": "Article data or URL not found"}), 404
    if article_data.get('groq_summary') is not None and article_data.get('groq_takeaways') is not None:
        return jsonify({"groq_analysis": {"groq_summary": article_data['groq_summary'], "groq_takeaways": article_data['groq_takeaways'], "error": None}, "error": None})
    return jsonify(fetch_and_parse_article_content(article_hash_id, article_data['url']))

@app.route('/add_comment/<article_hash_id>', methods=['POST'])
@login_required
def add_comment(article_hash_id):
    content = request.json.get('content', '').strip()
    parent_id = request.json.get('parent_id')
    if not content: return jsonify({"success": False, "error": "Comment cannot be empty."}), 400
    user = db.session.get(User, session['user_id'])
    
    new_comment = Comment(content=content, user_id=user.id, parent_id=parent_id)
    community_article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
    
    if community_article: new_comment.community_article_id = community_article.id
    else: new_comment.api_article_hash_id = article_hash_id

    try:
        db.session.add(new_comment); db.session.commit(); db.session.refresh(new_comment)
    except Exception:
        db.session.rollback()
        return jsonify({"success": False, "error": "Database error."}), 500

    return jsonify({"success": True, "html": render_template("_COMMENT_TEMPLATE", comment=new_comment, session=session), "parent_id": new_comment.parent_id}), 201
    
@app.route('/vote_comment/<int:comment_id>', methods=['POST'])
@login_required
def vote_comment(comment_id):
    db.get_or_404(Comment, comment_id)
    emoji = request.json.get('emoji')
    if not emoji or emoji not in ['👍', '❤️', '😂', '😮', '😢', '😠']: return jsonify({"error": "Invalid reaction."}), 400

    existing_vote = CommentVote.query.filter_by(user_id=session['user_id'], comment_id=comment_id).first()
    user_reaction_after_vote = None
    
    if existing_vote:
        if existing_vote.vote_emoji == emoji: db.session.delete(existing_vote)
        else: existing_vote.vote_emoji = emoji; user_reaction_after_vote = emoji
    else:
        db.session.add(CommentVote(user_id=session['user_id'], comment_id=comment_id, vote_emoji=emoji))
        user_reaction_after_vote = emoji
        
    db.session.commit()
    reactions = {emo: count for emo, count in db.session.query(CommentVote.vote_emoji, func.count(CommentVote.vote_emoji)).filter(CommentVote.comment_id == comment_id).group_by(CommentVote.vote_emoji).all()}
    return jsonify({"success": True, "reactions": reactions, "user_reaction": user_reaction_after_vote}), 200

@app.route('/delete_comment/<int:comment_id>', methods=['POST'])
@login_required
def delete_comment(comment_id):
    comment = db.get_or_404(Comment, comment_id)
    if comment.user_id != session['user_id']: return jsonify({"success": False, "error": "Unauthorized."}), 403
    try:
        db.session.delete(comment); db.session.commit()
        return jsonify({"success": True})
    except Exception:
        db.session.rollback(); return jsonify({"success": False, "error": "Database error."}), 500

@app.route('/edit_comment/<int:comment_id>', methods=['POST'])
@login_required
def edit_comment(comment_id):
    comment = db.get_or_404(Comment, comment_id)
    if comment.user_id != session['user_id']: return jsonify({"success": False, "error": "Unauthorized."}), 403
    new_content = request.json.get('content', '').strip()
    if not new_content: return jsonify({"success": False, "error": "Cannot be empty."}), 400
    try:
        comment.content = new_content; db.session.commit()
        return jsonify({"success": True, "new_content": comment.content})
    except Exception:
        db.session.rollback(); return jsonify({"success": False, "error": "Database error."}), 500

@app.route('/post_article', methods=['POST'])
@login_required
def post_article():
    title, description, content, source_name, image_url = map(lambda x: request.form.get(x, '').strip(), ['title', 'description', 'content', 'sourceName', 'imageUrl'])
    source_name = source_name or 'Community Post'
    if not all([title, description, content, source_name]):
        flash("Title, Description, Full Content, and Source Name are required.", "danger")
        return redirect(request.referrer or url_for('index'))
    article_hash_id = generate_article_id(title + str(session['user_id']) + str(time.time()))
    groq_analysis_result = get_article_analysis_with_groq(content, title)
    groq_summary_text, groq_takeaways_json_str = None, None
    if groq_analysis_result and not groq_analysis_result.get("error"):
        groq_summary_text = groq_analysis_result.get('groq_summary')
        takeaways_list = groq_analysis_result.get('groq_takeaways')
        if takeaways_list and isinstance(takeaways_list, list): groq_takeaways_json_str = json.dumps(takeaways_list)
    new_article = CommunityArticle(article_hash_id=article_hash_id, title=title, description=description, full_text=content, source_name=source_name, image_url=image_url or f'https://via.placeholder.com/400x220/3B82F6/FFFFFF?text={urllib.parse.quote_plus(title[:20])}', user_id=session['user_id'], published_at=datetime.now(timezone.utc), groq_summary=groq_summary_text, groq_takeaways=groq_takeaways_json_str)
    db.session.add(new_article); db.session.commit()
    flash("Your article has been posted!", "success")
    return redirect(url_for('article_detail', article_hash_id=new_article.article_hash_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        name, username, password = request.form.get('name', '').strip(), request.form.get('username', '').strip().lower(), request.form.get('password', '')
        if not all([name, username, password]): flash('All fields are required.', 'danger')
        elif len(username) < 3: flash('Username must be at least 3 characters.', 'warning')
        elif len(password) < 6: flash('Password must be at least 6 characters.', 'warning')
        elif User.query.filter_by(username=username).first(): flash('Username already exists.', 'warning')
        else:
            new_user = User(name=name, username=username, password_hash=generate_password_hash(password))
            db.session.add(new_user); db.session.commit()
            flash(f'Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        return redirect(url_for('register'))
    return render_template("REGISTER_HTML_TEMPLATE")

@app.route('/delete_community_article/<article_hash_id>', methods=['POST'])
@login_required
def delete_community_article(article_hash_id):
    if not session.get('is_admin') or session.get('username') != 'vbdevil': return jsonify({"success": False, "error": "Admin required."}), 403
    article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
    if not article: return jsonify({"success": False, "error": "Article not found."}), 404
    try:
        db.session.delete(article); db.session.commit()
        flash("Article deleted by administrator.", "success")
        return jsonify({"success": True, "redirect_url": url_for('index', category_name='Community Hub')})
    except Exception:
        db.session.rollback(); return jsonify({"success": False, "error": "Database error."}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        username, password = request.form.get('username', '').strip().lower(), request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session.permanent = True
            session.update({'user_id': user.id, 'user_name': user.name, 'username': user.username, 'is_admin': user.username == "vbdevil"})
            flash(f"Welcome back, {user.name}!", "success")
            return redirect(request.args.get('next') or url_for('index'))
        else: flash('Invalid credentials.', 'danger')
    return render_template("LOGIN_HTML_TEMPLATE")

@app.route('/logout')
def logout(): session.clear(); flash("Successfully logged out.", "info"); return redirect(url_for('index'))
@app.route('/about')
def about(): return render_template("ABOUT_US_HTML_TEMPLATE")
@app.route('/contact')
def contact(): return render_template("CONTACT_HTML_TEMPLATE")
@app.route('/privacy')
def privacy(): return render_template("PRIVACY_POLICY_HTML_TEMPLATE")

@app.route('/subscribe', methods=['POST'])
def subscribe():
    email = request.form.get('email', '').strip().lower()
    if not email: flash('Email required.', 'warning')
    elif Subscriber.query.filter_by(email=email).first(): flash('Already subscribed.', 'info')
    else:
        try: db.session.add(Subscriber(email=email)); db.session.commit(); flash('Subscribed!', 'success')
        except Exception: db.session.rollback(); flash('Error subscribing.', 'danger')
    return redirect(request.referrer or url_for('index'))

@app.route('/toggle_bookmark/<article_hash_id>', methods=['POST'])
@login_required
def toggle_bookmark(article_hash_id):
    user_id = session['user_id']
    is_community = request.json.get('is_community_article', 'false').lower() == 'true'
    existing_bookmark = BookmarkedArticle.query.filter_by(user_id=user_id, article_hash_id=article_hash_id).first()
    if existing_bookmark:
        db.session.delete(existing_bookmark); db.session.commit()
        return jsonify({"success": True, "status": "removed", "message": "Bookmark removed."})
    else:
        if is_community and not CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first():
            return jsonify({"success": False, "error": "Not found."}), 404
        elif not is_community and article_hash_id not in MASTER_ARTICLE_STORE:
            fetch_news_from_api() 
            if article_hash_id not in MASTER_ARTICLE_STORE: return jsonify({"success": False, "error": "Not found."}), 404
            
        published_at_cache = None
        if dt_str := request.json.get('published_at'):
            try: published_at_cache = pytz.utc.localize(datetime.fromisoformat(dt_str.replace('Z', '+00:00'))) if not datetime.fromisoformat(dt_str.replace('Z', '+00:00')).tzinfo else datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            except ValueError: pass

        db.session.add(BookmarkedArticle(user_id=user_id, article_hash_id=article_hash_id, is_community_article=is_community, title_cache=request.json.get('title'), source_name_cache=request.json.get('source_name'), image_url_cache=request.json.get('image_url'), description_cache=request.json.get('description'), published_at_cache=published_at_cache))
        db.session.commit()
        return jsonify({"success": True, "status": "added", "message": "Article bookmarked!"})

@app.route('/profile')
@login_required
def profile():
    user = db.session.get(User, session['user_id'])
    page, per_page = request.args.get('page', 1, type=int), app.config['PER_PAGE']
    user_posted_articles = CommunityArticle.query.filter_by(user_id=user.id).order_by(CommunityArticle.published_at.desc()).all()
    user_bookmarks_paginated_query = BookmarkedArticle.query.filter_by(user_id=user.id).order_by(BookmarkedArticle.bookmarked_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    user_bookmarked_articles_data = []
    
    for bookmark in user_bookmarks_paginated_query.items:
        art_data = None
        if bookmark.is_community_article:
            comm_art = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter_by(article_hash_id=bookmark.article_hash_id).first()
            if comm_art: art_data = {'id': comm_art.article_hash_id, 'title': comm_art.title, 'description': comm_art.description, 'urlToImage': comm_art.image_url, 'publishedAt': comm_art.published_at.isoformat() if comm_art.published_at else None, 'source': {'name': comm_art.author.name if comm_art.author else comm_art.source_name}, 'is_community_article': True, 'article_url': url_for('article_detail', article_hash_id=comm_art.article_hash_id)}
        else:
            api_art = MASTER_ARTICLE_STORE.get(bookmark.article_hash_id)
            if api_art: art_data = {'id': api_art['id'], 'title': api_art['title'], 'description': api_art['description'], 'urlToImage': api_art['urlToImage'], 'publishedAt': api_art['publishedAt'], 'source': {'name': api_art['source']['name']}, 'is_community_article': False, 'article_url': url_for('article_detail', article_hash_id=api_art['id'])}
            else: art_data = {'id': bookmark.article_hash_id, 'title': bookmark.title_cache or "Details N/A", 'description': bookmark.description_cache or "N/A", 'urlToImage': bookmark.image_url_cache, 'publishedAt': bookmark.published_at_cache.isoformat() if bookmark.published_at_cache else None, 'source': {'name': bookmark.source_name_cache}, 'is_community_article': False, 'article_url': url_for('article_detail', article_hash_id=bookmark.article_hash_id), 'is_stale_bookmark': True}
        if art_data: user_bookmarked_articles_data.append(art_data)
    return render_template("PROFILE_HTML_TEMPLATE", user=user, posted_articles=user_posted_articles, bookmarked_articles=user_bookmarked_articles_data, bookmarks_pagination=user_bookmarks_paginated_query, current_page=page)

@app.errorhandler(404)
def page_not_found(e): return render_template("404_TEMPLATE"), 404
@app.errorhandler(500)
def internal_server_error(e): db.session.rollback(); return render_template("500_TEMPLATE"), 500

# ==============================================================================
# --- 7. HTML Templates (Stored in memory) ---
# ==============================================================================
BASE_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}BrieflyAI{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #000000; --primary-light: #333333; --primary-dark: #000000; --secondary-color: #3B82F6; --secondary-light: #60A5FA; --accent-color: #10B981; 
            --text-color: #111827; --text-muted-color: #6B7280; --light-bg: #FAFAFA; --card-bg: #FFFFFF; --card-border-color: #E5E7EB; --footer-bg: #111827; --footer-text: #D1D5DB; 
            --bookmark-active-color: var(--secondary-color);
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05); --shadow-md: 0 10px 30px rgba(0,0,0,0.04); --shadow-lg: 0 20px 40px rgba(0,0,0,0.08);
            --border-radius-sm: 8px; --border-radius-md: 12px; --border-radius-lg: 20px;
        }
        body.dark-mode {
            --primary-color: #FFFFFF; --primary-light: #E5E5E5; --primary-dark: #FFFFFF; --secondary-color: #60A5FA; --secondary-light: #93C5FD; --accent-color: #34D399; 
            --text-color: #F9FAFB; --text-muted-color: #9CA3AF; --light-bg: #0A0A0A; --card-bg: #141414; --card-border-color: #262626; --footer-bg: #000000;
            --shadow-md: 0 10px 30px rgba(0,0,0,0.4); --shadow-lg: 0 20px 40px rgba(0,0,0,0.6);
        }
        body { padding-top: 90px; font-family: 'Inter', sans-serif; background-color: var(--light-bg); color: var(--text-color); transition: background-color 0.3s, color 0.3s; display: flex; flex-direction: column; min-height: 100vh; }
        h1, h2, h3, h4, h5, .navbar-brand-custom, .article-title-main { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; letter-spacing: -0.02em; }
        
        /* Modern Glassmorphism Navbar */
        .glass-nav { background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); border-bottom: 1px solid var(--card-border-color); position: fixed; top: 0; width: 100%; z-index: 1040; transition: all 0.3s ease; }
        body.dark-mode .glass-nav { background: rgba(10, 10, 10, 0.7); border-bottom: 1px solid rgba(255,255,255,0.05); }
        .navbar-brand-custom { color: var(--text-color) !important; font-size: 1.5rem; text-decoration: none; display: flex; align-items: center; gap: 8px; }
        .navbar-brand-custom i { color: var(--secondary-color); }
        .header-btn { background: var(--card-bg); border: 1px solid var(--card-border-color); padding: 0.5rem 1rem; border-radius: 50px; color: var(--text-color); font-weight: 600; cursor: pointer; transition: 0.2s; }
        .header-btn:hover { background: var(--primary-color); color: var(--card-bg); }
        .search-container { position: relative; max-width: 400px; width: 100%; margin: 0 auto; }
        .navbar-search { border-radius: 50px; padding: 0.5rem 1.25rem 0.5rem 2.5rem; border: 1px solid var(--card-border-color); background: var(--light-bg); color: var(--text-color); }
        .navbar-search:focus { outline: none; border-color: var(--secondary-color); box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1); }
        .search-icon { position: absolute; left: 1rem; top: 50%; transform: translateY(-50%); color: var(--text-muted-color); }
        
        /* Soft Bento UI Cards */
        .article-card, .ai-synthesis-card, .profile-header-card, .auth-card { background: var(--card-bg); border-radius: var(--border-radius-lg); border: 1px solid var(--card-border-color); box-shadow: var(--shadow-md); transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s; overflow: hidden; }
        .article-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-lg); }
        .article-image-container { height: 200px; overflow: hidden; }
        .article-image { width: 100%; height: 100%; object-fit: cover; transition: transform 0.5s ease; }
        .article-card:hover .article-image { transform: scale(1.05); }
        .article-body { padding: 1.5rem; flex-grow: 1; display: flex; flex-direction: column; }
        .article-title { font-weight: 700; line-height: 1.3; font-size: 1.1rem; }
        .article-title a { color: var(--text-color); text-decoration: none; }
        .article-title a:hover { color: var(--secondary-color); }
        .article-description { color: var(--text-muted-color); font-size: 0.95rem; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
        
        /* Modern Buttons */
        .btn-primary { background: var(--primary-color); color: var(--card-bg); border: none; border-radius: 50px; font-weight: 600; padding: 0.6rem 1.5rem; transition: 0.2s; }
        .btn-primary:hover { background: var(--primary-light); color: var(--card-bg); transform: scale(1.02); }
        body.dark-mode .btn-primary:hover { background: #CCCCCC; color: #000; }
        .btn-outline-primary { border: 2px solid var(--primary-color); color: var(--primary-color); border-radius: 50px; font-weight: 600; transition: 0.2s; }
        .btn-outline-primary:hover { background: var(--primary-color); color: var(--card-bg); }
        .read-more { margin-top: auto; width: 100%; background: var(--light-bg); color: var(--text-color) !important; border-radius: 12px; font-weight: 600; border: 1px solid var(--card-border-color); padding: 0.6rem; text-align: center; text-decoration: none; transition: 0.2s; }
        .read-more:hover { background: var(--primary-color); color: var(--card-bg) !important; border-color: var(--primary-color); }
        
        /* Toast Notifications Container */
        #toast-container { position: fixed; bottom: 20px; right: 20px; z-index: 1055; display: flex; flex-direction: column; gap: 10px; }
        .custom-toast { background: var(--card-bg); color: var(--text-color); border-radius: 12px; box-shadow: var(--shadow-lg); border: 1px solid var(--card-border-color); padding: 1rem 1.5rem; display: flex; align-items: center; justify-content: space-between; min-width: 250px; animation: slideInRight 0.3s ease forwards; }
        .custom-toast.hide { animation: fadeOutRight 0.3s ease forwards; }
        @keyframes slideInRight { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
        @keyframes fadeOutRight { from { transform: translateX(0); opacity: 1; } to { transform: translateX(100%); opacity: 0; } }

        /* Comments / Reactions */
        .comment-thread { border-bottom: 1px solid var(--card-border-color); padding-bottom: 1rem; margin-bottom: 1rem; }
        .comment-avatar { width: 40px; height: 40px; border-radius: 50%; background: var(--primary-light); color: var(--card-bg); font-weight: 700; display: flex; align-items: center; justify-content: center; }
        .reaction-pill { background: var(--light-bg); border: 1px solid var(--card-border-color); padding: 2px 10px; border-radius: 20px; font-size: 0.85rem; cursor: pointer; transition: 0.2s; }
        .reaction-pill.user-reacted { background: rgba(59, 130, 246, 0.1); border-color: var(--secondary-color); color: var(--secondary-color); }
        .reaction-box { position: absolute; background: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: 30px; padding: 5px 10px; display: none; box-shadow: var(--shadow-md); z-index: 10; gap: 8px; }
        .reaction-box.show { display: flex; animation: fadeInUp 0.2s; }

        footer { background: var(--footer-bg); color: var(--footer-text); padding: 3rem 0; font-size: 0.9rem; margin-top: auto; }
        footer a { color: var(--footer-text); text-decoration: none; transition: 0.2s; }
        footer a:hover { color: var(--secondary-light); }
        .add-article-btn { position: fixed; bottom: 30px; right: 30px; width: 60px; height: 60px; border-radius: 50%; background: var(--primary-color); color: var(--card-bg); border: none; font-size: 24px; display: flex; align-items: center; justify-content: center; box-shadow: var(--shadow-lg); transition: 0.3s; z-index: 1030;}
        .add-article-btn:hover { transform: scale(1.1) rotate(5deg); }
        .bookmark-btn { background: none; border: none; color: var(--text-muted-color); font-size: 1.2rem; transition: 0.2s; }
        .bookmark-btn.active { color: var(--bookmark-active-color); }
        .bookmark-btn:hover { color: var(--secondary-color); transform: scale(1.1); }
    </style>
    {% block head_extra %}{% endblock %}
</head>
<body class="{{ request.cookies.get('darkMode', 'disabled') }}">

    <nav class="glass-nav py-3">
        <div class="container d-flex align-items-center justify-content-between">
            <a class="navbar-brand-custom" href="{{ url_for('index') }}"><i class="fas fa-bolt-lightning"></i> BrieflyAI</a>
            <div class="d-none d-md-block flex-grow-1 mx-4">
                <form action="{{ url_for('search_results') }}" method="GET" class="search-container">
                    <input type="search" name="query" class="form-control navbar-search" placeholder="Search insights..." value="{{ request.args.get('query', '') }}">
                    <i class="fas fa-search search-icon"></i>
                </form>
            </div>
            <button class="header-btn" type="button" data-bs-toggle="offcanvas" data-bs-target="#mainOffcanvas"><i class="fas fa-bars"></i></button>
        </div>
    </nav>

    <!-- Offcanvas Menu -->
    <div class="offcanvas offcanvas-end" tabindex="-1" id="mainOffcanvas" style="background: var(--card-bg); color: var(--text-color);">
        <div class="offcanvas-header border-bottom border-secondary">
            <h5 class="offcanvas-title font-weight-bold"><i class="fas fa-compass me-2 text-primary"></i>Menu</h5>
            <button type="button" class="btn-close" style="filter: var(--close-btn-filter);" data-bs-dismiss="offcanvas"></button>
        </div>
        <div class="offcanvas-body">
            {% if session.user_id %}
                <div class="mb-4 p-3 rounded" style="background: var(--light-bg);">
                    <strong><i class="fas fa-user-circle me-2"></i>{{ session.user_name }}</strong>
                    <div class="mt-2">
                        <a href="{{ url_for('profile') }}" class="btn btn-sm btn-outline-primary rounded-pill w-100 mb-2">My Profile</a>
                        <a href="{{ url_for('logout') }}" class="btn btn-sm btn-outline-danger rounded-pill w-100">Logout</a>
                    </div>
                </div>
            {% else %}
                <a href="{{ url_for('login') }}" class="btn btn-primary w-100 mb-4 rounded-pill">Sign In / Register</a>
            {% endif %}
            <button class="btn btn-outline-secondary w-100 mb-4 rounded-pill dark-mode-toggle"><i class="fas fa-moon me-2"></i><span class="theme-text">Dark Mode</span></button>
            <h6 class="text-uppercase text-muted small fw-bold mb-3">Categories</h6>
            <div class="d-flex flex-column gap-2">
                {% for cat_item in categories %}
                    <a href="{{ url_for('index', category_name=cat_item) }}" class="btn btn-light text-start rounded-pill" style="background: var(--light-bg); border: 1px solid var(--card-border-color); color: var(--text-color);">{{ cat_item }}</a>
                {% endfor %}
            </div>
        </div>
    </div>

    <!-- Toast Notifications -->
    <div id="toast-container"></div>
    <div id="flash-data" style="display:none;">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {{ messages | tojson }}
            {% endif %}
        {% endwith %}
    </div>

    <main class="container my-4 flex-grow-1">
        {% block content %}{% endblock %}
    </main>
    
    {% if session.user_id %}
    <button class="add-article-btn" data-bs-toggle="modal" data-bs-target="#addArticleModal" title="Post a New Article"><i class="fas fa-plus"></i></button>
    <div class="modal fade" id="addArticleModal" tabindex="-1" style="z-index: 1060;">
        <div class="modal-dialog modal-dialog-centered modal-lg">
            <div class="modal-content" style="background: var(--card-bg); color: var(--text-color); border: 1px solid var(--card-border-color); border-radius: var(--border-radius-lg);">
                <div class="modal-header border-0"><h4 class="modal-title font-weight-bold">Post New Article</h4><button type="button" class="btn-close" data-bs-dismiss="modal"></button></div>
                <div class="modal-body">
                    <form action="{{ url_for('post_article') }}" method="POST">
                        <input type="text" name="title" class="form-control mb-3" placeholder="Article Title" required style="background: var(--light-bg); color: var(--text-color); border: 1px solid var(--card-border-color);">
                        <textarea name="description" class="form-control mb-3" rows="2" placeholder="Short Description" required style="background: var(--light-bg); color: var(--text-color); border: 1px solid var(--card-border-color);"></textarea>
                        <input type="text" name="sourceName" class="form-control mb-3" value="Community Post" required style="background: var(--light-bg); color: var(--text-color); border: 1px solid var(--card-border-color);">
                        <textarea name="content" class="form-control mb-4" rows="6" placeholder="Full Content" required style="background: var(--light-bg); color: var(--text-color); border: 1px solid var(--card-border-color);"></textarea>
                        <button type="submit" class="btn btn-primary w-100 rounded-pill">Publish Article</button>
                    </form>
                </div>
            </div>
        </div>
    </div>
    {% endif %}

    <footer>
        <div class="container text-center">
            <h4 class="mb-3 font-weight-bold"><i class="fas fa-bolt-lightning text-primary"></i> BrieflyAI</h4>
            <p class="small text-muted mb-4">Your modern source for AI-summarized insights.</p>
            <div class="d-flex justify-content-center gap-4 mb-4">
                <a href="{{ url_for('about') }}">About</a> <a href="{{ url_for('privacy') }}">Privacy</a> <a href="{{ url_for('contact') }}">Contact</a>
            </div>
            <p class="small text-muted m-0">&copy; 2026 BrieflyAI. All rights reserved.</p>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        function showToast(message, type="info") {
            const container = document.getElementById('toast-container');
            const icon = type === 'success' ? 'check-circle text-success' : type === 'danger' ? 'exclamation-circle text-danger' : 'info-circle text-primary';
            const toastHTML = `<div class="custom-toast"><div class="d-flex align-items-center gap-3"><i class="fas fa-${icon} fs-5"></i><span>${message}</span></div><button class="btn-close btn-close-sm" onclick="this.parentElement.classList.add('hide'); setTimeout(()=>this.parentElement.remove(), 300)"></button></div>`;
            container.insertAdjacentHTML('beforeend', toastHTML);
            const newToast = container.lastElementChild;
            setTimeout(() => { if(newToast) { newToast.classList.add('hide'); setTimeout(()=>newToast.remove(), 300); } }, 5000);
        }

        document.addEventListener('DOMContentLoaded', () => {
            const flashData = document.getElementById('flash-data');
            if (flashData && flashData.textContent.trim()) {
                const messages = JSON.parse(flashData.textContent);
                messages.forEach(msg => showToast(msg[1], msg[0]));
            }

            const darkModeToggle = document.querySelector('.dark-mode-toggle');
            if (darkModeToggle) {
                const body = document.body;
                const updateThemeUI = () => {
                    const isDark = body.classList.contains('dark-mode');
                    darkModeToggle.innerHTML = isDark ? '<i class="fas fa-sun me-2"></i> Light Mode' : '<i class="fas fa-moon me-2"></i> Dark Mode';
                    document.documentElement.style.setProperty('--close-btn-filter', isDark ? 'invert(1) grayscale(100%) brightness(200%)' : 'none');
                };
                darkModeToggle.addEventListener('click', () => {
                    body.classList.toggle('dark-mode');
                    document.cookie = "darkMode=" + (body.classList.contains('dark-mode') ? 'enabled' : 'disabled') + ";path=/;max-age=31536000;SameSite=Lax";
                    updateThemeUI();
                });
                updateThemeUI();
            }
        });
    </script>
    {% block scripts_extra %}{% endblock %}
</body>
</html>
"""

INDEX_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Home - BrieflyAI{% endblock %}
{% block content %}
{% if is_main_homepage %}
    {% if synthesis %}
    <div class="ai-synthesis-card mb-5 p-4 text-center animate-fade-in" style="background: linear-gradient(145deg, rgba(59,130,246,0.05), transparent); border: 1px solid var(--secondary-light);">
        <h4 class="mb-3"><i class="fas fa-sparkles text-primary me-2"></i> Today's Big Picture</h4>
        <p class="lead mb-4" style="font-weight: 500;">"{{ synthesis }}"</p>
        <div class="d-flex flex-wrap justify-content-center gap-2">
            {% for keyword in keywords %}<a href="{{ url_for('search_results', query=keyword) }}" class="badge rounded-pill bg-light text-dark border p-2 text-decoration-none px-3" style="background: var(--card-bg)!important; color: var(--text-color)!important;">{{ keyword }}</a>{% endfor %}
        </div>
    </div>
    {% endif %}

    <div class="d-flex justify-content-between align-items-end border-bottom pb-2 mb-4">
        <h2 class="m-0"><i class="fas fa-fire text-danger me-2"></i> Top Stories</h2>
        <a href="{{ url_for('index', category_name='Popular Stories') }}" class="text-decoration-none small fw-bold">View All</a>
    </div>

    <div class="row g-4 mb-5">
        {% for art in popular_articles %}
        <div class="col-md-6 col-lg-4 d-flex">
            <article class="article-card d-flex flex-column w-100 animate-fade-in">
                {% set article_url = url_for('article_detail', article_hash_id=art.id) %}
                <div class="article-image-container"><a href="{{ article_url }}"><img src="{{ art.urlToImage }}" class="article-image"></a></div>
                <div class="article-body">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <h5 class="article-title m-0"><a href="{{ article_url }}">{{ art.title|truncate(65) }}</a></h5>
                        {% if session.user_id %}<button class="bookmark-btn {% if art.is_bookmarked %}active{% endif %}" onclick="toggleBookmark(this, '{{ art.id }}', false)"><i class="fa-solid fa-bookmark"></i></button>{% endif %}
                    </div>
                    <p class="small text-muted mb-3"><i class="fas fa-building me-1"></i> {{ art.source.name }} &bull; {{ art.publishedAt | to_ist if art.publishedAt else 'Recent' }}</p>
                    <p class="article-description mb-4">{{ art.description|truncate(100) }}</p>
                    <a href="{{ article_url }}" class="read-more mt-auto">Read Insights</a>
                </div>
            </article>
        </div>
        {% endfor %}
    </div>
{% else %}
    <h2 class="border-bottom pb-2 mb-4">{{ selected_category }}</h2>
    <div class="row g-4">
        {% for art in articles %}
        <div class="col-md-6 col-lg-4 d-flex">
            <article class="article-card d-flex flex-column w-100">
                {% set is_comm = 'true' if art.is_community_article else 'false' %}
                {% set art_id = art.article_hash_id if art.is_community_article else art.id %}
                {% set url = url_for('article_detail', article_hash_id=art_id) %}
                <div class="article-image-container"><a href="{{ url }}"><img src="{{ art.image_url if art.is_community_article else art.urlToImage }}" class="article-image"></a></div>
                <div class="article-body">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <h5 class="article-title m-0"><a href="{{ url }}">{{ art.title|truncate(65) }}</a></h5>
                        {% if session.user_id %}<button class="bookmark-btn {% if art.is_bookmarked %}active{% endif %}" onclick="toggleBookmark(this, '{{ art_id }}', {{ is_comm }})"><i class="fa-solid fa-bookmark"></i></button>{% endif %}
                    </div>
                    <p class="article-description mb-4">{{ art.description|truncate(100) }}</p>
                    <a href="{{ url }}" class="read-more mt-auto">Read Insights</a>
                </div>
            </article>
        </div>
        {% endfor %}
    </div>
{% endif %}
{% endblock %}
{% block scripts_extra %}
<script>
    function toggleBookmark(btn, hashId, isComm) {
        fetch(`/toggle_bookmark/${hashId}`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ is_community_article: isComm })
        }).then(res=>res.json()).then(data=>{
            if(data.success) { btn.classList.toggle('active', data.status==='added'); showToast(data.message, 'success'); }
        });
    }
</script>
{% endblock %}
"""

ARTICLE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ article.title|truncate(50) }}{% endblock %}
{% block content %}
<div class="article-card p-4 p-md-5 mb-5 mx-auto" style="max-width: 900px; border: none; box-shadow: var(--shadow-lg);">
    <a href="{{ previous_list_page }}" class="text-decoration-none text-muted small fw-bold mb-4 d-inline-block"><i class="fas fa-arrow-left me-2"></i>Back</a>
    <h1 class="article-title-main display-5 mb-3">{{ article.title }}</h1>
    <div class="d-flex align-items-center justify-content-between flex-wrap gap-3 mb-4 border-bottom pb-4">
        <div class="text-muted font-weight-bold">
            <i class="fas fa-{{ 'user' if is_community_article else 'building' }} me-1"></i> {{ article.author.name if is_community_article else article.source.name }}
        </div>
        {% if session.user_id %}<button class="btn btn-outline-primary rounded-pill btn-sm" onclick="toggleBookmark(this, '{{ article.article_hash_id if is_community_article else article.id }}', {{ 'true' if is_community_article else 'false' }})"><i class="fas fa-bookmark me-1"></i> Bookmark</button>{% endif %}
    </div>
    
    {% set img = article.image_url if is_community_article else article.urlToImage %}
    {% if img %}<img src="{{ img }}" class="w-100 rounded mb-4" style="max-height: 400px; object-fit: cover;">{% endif %}

    <div id="ai-content-area" class="mb-5">
        {% if is_community_article %}
            {% if article.groq_summary %}<div class="p-4 rounded mb-4" style="background: rgba(59,130,246,0.05); border-left: 4px solid var(--secondary-color);"><h5><i class="fas fa-robot text-primary me-2"></i> AI Summary</h5><p class="m-0">{{ article.groq_summary }}</p></div>{% endif %}
            <div style="font-size: 1.1rem; line-height: 1.8;">{{ article.full_text }}</div>
        {% else %}
            <div id="loader" class="text-center py-5"><div class="spinner-border text-primary" role="status"></div><p class="mt-3 text-muted fw-bold">AI Summarizing...</p></div>
        {% endif %}
    </div>

    <!-- Modern Comments Section -->
    <div class="border-top pt-5">
        <h3 class="mb-4">Discussion (${{ total_comment_count }})</h3>
        {% if session.user_id %}
            <form id="main-comment-form" class="mb-5">
                <textarea class="form-control mb-3 rounded-3" style="background: var(--light-bg); border: 1px solid var(--card-border-color); color: var(--text-color);" id="comment-content" rows="3" placeholder="Add to the conversation..." required></textarea>
                <button type="submit" class="btn btn-primary rounded-pill">Post Comment</button>
            </form>
        {% else %}
            <div class="p-3 mb-4 rounded text-center" style="background: var(--light-bg); border: 1px dashed var(--card-border-color);"><a href="{{ url_for('login') }}" class="fw-bold">Log in</a> to join the discussion.</div>
        {% endif %}
        <div id="comments-list">
            {% for comment in comments %}{% include '_COMMENT_TEMPLATE' %}{% endfor %}
        </div>
    </div>
</div>
{% endblock %}
{% block scripts_extra %}
<script>
    {% if not is_community_article %}
    fetch(`/get_article_content/{{ article.id }}`).then(r=>r.json()).then(data=>{
        document.getElementById('loader').style.display='none';
        let html = '';
        if(data.groq_analysis && data.groq_analysis.groq_summary) {
            html += `<div class="p-4 rounded mb-4" style="background: rgba(59,130,246,0.05); border-left: 4px solid var(--secondary-color);"><h5><i class="fas fa-robot text-primary me-2"></i> AI Summary</h5><p class="m-0">${data.groq_analysis.groq_summary}</p></div>`;
            if(data.groq_analysis.groq_takeaways.length) {
                html += `<ul>${data.groq_analysis.groq_takeaways.map(t=>`<li>${t}</li>`).join('')}</ul>`;
            }
        }
        html += `<a href="${'{{ article.url }}'}" target="_blank" class="btn btn-outline-primary rounded-pill mt-3">Read Original</a>`;
        document.getElementById('ai-content-area').innerHTML = html;
    });
    {% endif %}

    function toggleBookmark(btn, hashId, isComm) {
        fetch(`/toggle_bookmark/${hashId}`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ is_community_article: isComm }) })
        .then(r=>r.json()).then(d=>{ if(d.success) showToast(d.message, 'success'); });
    }

    document.getElementById('main-comment-form')?.addEventListener('submit', e=>{
        e.preventDefault();
        const content = document.getElementById('comment-content').value;
        fetch(`/add_comment/{{ article.article_hash_id if is_community_article else article.id }}`, {
            method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({content})
        }).then(r=>r.json()).then(d=>{
            if(d.success) {
                document.getElementById('comments-list').insertAdjacentHTML('beforeend', d.html);
                document.getElementById('comment-content').value = '';
                showToast("Comment added!", "success");
            }
        });
    });
</script>
{% endblock %}
"""

_COMMENT_TEMPLATE = """
<div class="comment-thread d-flex gap-3 mb-4" id="comment-{{ comment.id }}">
    <div class="comment-avatar">{{ comment.author.name[0]|upper }}</div>
    <div class="flex-grow-1">
        <div class="d-flex justify-content-between"><strong class="small">{{ comment.author.name }}</strong> <span class="small text-muted">{{ comment.timestamp|to_ist }}</span></div>
        <p class="my-1 text-sm">{{ comment.content }}</p>
        {% if session.user_id %}
            <div class="d-flex gap-3 mt-1 position-relative">
                <button class="btn btn-link p-0 text-muted small text-decoration-none" onclick="document.getElementById('rb-{{comment.id}}').classList.toggle('show')"><i class="far fa-smile"></i> React</button>
                <div class="reaction-box" id="rb-{{comment.id}}">
                    {% for emoji in ['👍', '❤️', '😂', '😮', '😢'] %}<span onclick="react({{comment.id}}, '{{emoji}}')" style="cursor:pointer">{{emoji}}</span>{% endfor %}
                </div>
            </div>
        {% endif %}
    </div>
</div>
<script>
    function react(cid, emoji) {
        fetch(`/vote_comment/${cid}`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({emoji})})
        .then(r=>r.json()).then(d=>{ if(d.success) document.getElementById(`rb-${cid}`).classList.remove('show'); });
    }
</script>
"""

LOGIN_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Login - BrieflyAI{% endblock %}
{% block content %}
<div class="auth-card mx-auto p-4 p-md-5 my-5" style="max-width: 400px;">
    <div class="text-center mb-4"><i class="fas fa-bolt-lightning text-primary fs-1 mb-2"></i><h3>Welcome Back</h3></div>
    <form method="POST">
        <input type="text" name="username" class="form-control rounded-pill mb-3 py-2 px-4" placeholder="Username" required style="background: var(--light-bg); border-color: var(--card-border-color); color: var(--text-color);">
        <input type="password" name="password" class="form-control rounded-pill mb-4 py-2 px-4" placeholder="Password" required style="background: var(--light-bg); border-color: var(--card-border-color); color: var(--text-color);">
        <button type="submit" class="btn btn-primary w-100 rounded-pill py-2">Sign In</button>
    </form>
    <div class="text-center mt-4 small"><span class="text-muted">Don't have an account?</span> <a href="{{ url_for('register') }}" class="fw-bold text-decoration-none">Create one</a></div>
</div>
{% endblock %}
"""

REGISTER_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Register - BrieflyAI{% endblock %}
{% block content %}
<div class="auth-card mx-auto p-4 p-md-5 my-5" style="max-width: 400px;">
    <div class="text-center mb-4"><i class="fas fa-user-plus text-primary fs-1 mb-2"></i><h3>Join BrieflyAI</h3></div>
    <form method="POST">
        <input type="text" name="name" class="form-control rounded-pill mb-3 py-2 px-4" placeholder="Full Name" required style="background: var(--light-bg); border-color: var(--card-border-color); color: var(--text-color);">
        <input type="text" name="username" class="form-control rounded-pill mb-3 py-2 px-4" placeholder="Username" required style="background: var(--light-bg); border-color: var(--card-border-color); color: var(--text-color);">
        <input type="password" name="password" class="form-control rounded-pill mb-4 py-2 px-4" placeholder="Password (min 6 chars)" required style="background: var(--light-bg); border-color: var(--card-border-color); color: var(--text-color);">
        <button type="submit" class="btn btn-primary w-100 rounded-pill py-2">Create Account</button>
    </form>
    <div class="text-center mt-4 small"><span class="text-muted">Already have an account?</span> <a href="{{ url_for('login') }}" class="fw-bold text-decoration-none">Sign in</a></div>
</div>
{% endblock %}
"""

PROFILE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Profile{% endblock %}
{% block content %}
<div class="profile-header-card p-5 mb-5 text-center mx-auto" style="max-width: 800px;">
    <div class="d-inline-flex justify-content-center align-items-center rounded-circle bg-primary text-white mb-3" style="width: 80px; height: 80px; font-size: 2rem; font-weight: bold;">{{ user.name[0]|upper }}</div>
    <h2 class="mb-1">{{ user.name }}</h2><p class="text-muted">@{{ user.username }}</p>
</div>
<h3 class="mb-4">My Bookmarks</h3>
<div class="row g-4">
    {% for art in bookmarked_articles %}
    <div class="col-md-6 col-lg-4 d-flex">
        <article class="article-card d-flex flex-column w-100">
            <div class="article-image-container"><a href="{{ art.article_url }}"><img src="{{ art.urlToImage }}" class="article-image"></a></div>
            <div class="article-body">
                <h5 class="article-title mb-2"><a href="{{ art.article_url }}">{{ art.title|truncate(65) }}</a></h5>
                <a href="{{ art.article_url }}" class="read-more mt-auto">Read Insights</a>
            </div>
        </article>
    </div>
    {% endfor %}
</div>
{% endblock %}
"""

template_storage['BASE_HTML_TEMPLATE'] = BASE_HTML_TEMPLATE
template_storage['INDEX_HTML_TEMPLATE'] = INDEX_HTML_TEMPLATE
template_storage['ARTICLE_HTML_TEMPLATE'] = ARTICLE_HTML_TEMPLATE
template_storage['LOGIN_HTML_TEMPLATE'] = LOGIN_HTML_TEMPLATE
template_storage['REGISTER_HTML_TEMPLATE'] = REGISTER_HTML_TEMPLATE
template_storage['PROFILE_HTML_TEMPLATE'] = PROFILE_HTML_TEMPLATE
template_storage['_COMMENT_TEMPLATE'] = _COMMENT_TEMPLATE
template_storage['404_TEMPLATE'] = """{% extends "BASE_HTML_TEMPLATE" %}{% block content %}<div class="text-center py-5"><h2>404</h2><p>Not found</p></div>{% endblock %}"""
template_storage['500_TEMPLATE'] = """{% extends "BASE_HTML_TEMPLATE" %}{% block content %}<div class="text-center py-5"><h2>500</h2><p>Server Error</p></div>{% endblock %}"""
template_storage['ABOUT_US_HTML_TEMPLATE'] = """{% extends "BASE_HTML_TEMPLATE" %}{% block content %}<div class="text-center py-5"><h2>About BrieflyAI</h2><p>AI summarization for India.</p></div>{% endblock %}"""
template_storage['CONTACT_HTML_TEMPLATE'] = """{% extends "BASE_HTML_TEMPLATE" %}{% block content %}<div class="text-center py-5"><h2>Contact</h2><p>vbansal639@gmail.com</p></div>{% endblock %}"""
template_storage['PRIVACY_POLICY_HTML_TEMPLATE'] = """{% extends "BASE_HTML_TEMPLATE" %}{% block content %}<div class="text-center py-5"><h2>Privacy</h2><p>Your data is secure.</p></div>{% endblock %}"""
template_storage['PUBLIC_PROFILE_HTML_TEMPLATE'] = """{% extends "BASE_HTML_TEMPLATE" %}{% block content %}<div class="text-center py-5"><h2>{{user.name}}</h2></div>{% endblock %}"""

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
