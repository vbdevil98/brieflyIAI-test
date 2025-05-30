# Rev14.py - MODIFIED FOR MORE ROBUST AND DIAGNOSTIC NEWS FETCHING, NEW COMMENT FEATURES,
# BOOKMARKING, USER PROFILES, AI TAGS, AND CELERY BACKGROUND TASKS

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

# Third-party imports
import nltk
import requests
from flask import (Flask, render_template, url_for, redirect, request, jsonify, session, flash)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func, case
from sqlalchemy.orm import joinedload
from sqlalchemy.exc import IntegrityError # Added for graceful handling
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
from celery import Celery # Added for background tasks

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
app.config['CATEGORIES'] = ['All Articles', 'Community Hub'] # My Bookmarks will be a separate link

app.config['NEWS_API_QUERY'] = 'India OR "Indian politics" OR "Indian economy" OR "Bollywood"'
app.config['NEWS_API_DOMAINS'] = 'timesofindia.indiatimes.com,thehindu.com,ndtv.com,indianexpress.com,hindustantimes.com'
app.config['NEWS_API_DAYS_AGO'] = 7
app.config['NEWS_API_PAGE_SIZE'] = 100
app.config['NEWS_API_SORT_BY'] = 'publishedAt'
app.config['CACHE_EXPIRY_SECONDS'] = 3600 # 1 hour
app.permanent_session_lifetime = timedelta(days=30)

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

# Data Persistence
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
    app.logger.info(f"Using local SQLite database for development at {db_path}.")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ==============================================================================
# --- Celery Configuration ---
# ==============================================================================
def make_celery(flask_app):
    broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    result_backend_url = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    
    # Check if Redis is accessible (basic check)
    try:
        if broker_url.startswith('redis://'):
            from redis import Redis
            from redis.exceptions import ConnectionError as RedisConnectionError
            redis_client_test = Redis.from_url(broker_url)
            redis_client_test.ping()
            app.logger.info(f"Successfully connected to Redis broker for Celery: {broker_url}")
    except (ImportError, RedisConnectionError) as e:
        app.logger.warning(f"Could not connect to Redis broker at {broker_url}. Celery background tasks may not work. Error: {e}")
        app.logger.warning("Falling back to EAGER mode for Celery tasks if Redis is unavailable.")
        # Fallback to running tasks eagerly if Redis isn't available or redis-py not installed
        flask_app.config['CELERY_TASK_ALWAYS_EAGER'] = True
        flask_app.config['CELERY_TASK_EAGER_PROPAGATES'] = True # Propagates exceptions like normal functions
        # No need to set broker/backend if always eager
        broker_url = None # Effectively disables broker for eager mode
        result_backend_url = None


    celery_instance = Celery(
        flask_app.import_name,
        backend=result_backend_url,
        broker=broker_url
    )
    celery_instance.conf.update(flask_app.config)
    
    # Ensure tasks run within Flask application context
    class ContextTask(celery_instance.Task):
        def __call__(self, *args, **kwargs):
            with flask_app.app_context():
                return self.run(*args, **kwargs)
    celery_instance.Task = ContextTask
    return celery_instance

# Update Flask app config for Celery (can be overridden by make_celery if Redis fails)
app.config.update(
    CELERY_BROKER_URL=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    CELERY_RESULT_BACKEND=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    CELERY_TASK_SERIALIZER='json',
    CELERY_RESULT_SERIALIZER='json',
    CELERY_ACCEPT_CONTENT=['json'],
    CELERY_TIMEZONE='UTC',
    CELERY_ENABLE_UTC=True,
)
celery = make_celery(app)
app.logger.info(f"Celery configured. Broker: {celery.conf.broker_url}, Backend: {celery.conf.result_backend}, Always Eager: {app.config.get('CELERY_TASK_ALWAYS_EAGER', False)}")


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
        groq_client = ChatGroq(model="llama3-70b-8192", groq_api_key=GROQ_API_KEY, temperature=0.2) # Slightly higher temp for tag creativity
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
    created_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)) # NEW
    articles = db.relationship('CommunityArticle', backref='author', lazy='dynamic', cascade="all, delete-orphan")
    comments = db.relationship('Comment', backref=db.backref('author', lazy='joined'), lazy='dynamic', cascade="all, delete-orphan")
    comment_votes = db.relationship('CommentVote', backref='user', lazy='dynamic', cascade="all, delete-orphan")
    bookmarks = db.relationship('Bookmark', backref=db.backref('user', lazy='joined'), lazy='dynamic', cascade="all, delete-orphan") # NEW

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
    groq_takeaways = db.Column(db.Text, nullable=True) # Stored as JSON string
    tags = db.Column(db.Text, nullable=True) # NEW: Stored as JSON string list of tags
    comments = db.relationship('Comment', backref=db.backref('community_article', lazy='joined'), lazy='dynamic', foreign_keys='Comment.community_article_id', cascade="all, delete-orphan")
    bookmarks = db.relationship('Bookmark', foreign_keys='Bookmark.community_article_id', backref='community_article_ref', lazy='dynamic', cascade="all, delete-orphan") # NEW

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
    vote_type = db.Column(db.SmallInteger, nullable=False)
    __table_args__ = (db.UniqueConstraint('user_id', 'comment_id', name='_user_comment_uc'),)

class Subscriber(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    subscribed_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

# NEW Bookmark Model
class Bookmark(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False)
    api_article_hash_id = db.Column(db.String(32), nullable=True, index=True)
    community_article_id = db.Column(db.Integer, db.ForeignKey('community_article.id', ondelete="CASCADE"), nullable=True)
    timestamp = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    # Ensure a user can only bookmark an article once
    __table_args__ = (
        db.UniqueConstraint('user_id', 'api_article_hash_id', name='_user_api_article_bookmark_uc'),
        db.UniqueConstraint('user_id', 'community_article_id', name='_user_community_article_bookmark_uc'),
        db.CheckConstraint(
            "(api_article_hash_id IS NOT NULL AND community_article_id IS NULL) OR "
            "(api_article_hash_id IS NULL AND community_article_id IS NOT NULL)",
            name="chk_bookmark_type_exclusive" # Ensures one of the article foreign keys is set, but not both
        )
    )


def init_db():
    with app.app_context():
        app.logger.info("Attempting to create/update database tables...")
        # db.drop_all() # Use with caution during development if schema changes frequently
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
        try: utc_dt = datetime.fromisoformat(utc_dt.replace('Z', '+00:00'))
        except ValueError: return "Invalid date"
    if utc_dt.tzinfo is None: utc_dt = pytz.utc.localize(utc_dt)
    else: utc_dt = utc_dt.astimezone(pytz.utc)
    ist_dt = utc_dt.astimezone(INDIAN_TIMEZONE)
    return ist_dt.strftime('%b %d, %Y at %I:%M %p %Z')
app.jinja_env.filters['to_ist'] = to_ist_filter

def json_loads_safe(s): # NEW Jinja filter for parsing JSON in templates
    try:
        return json.loads(s) if s else []
    except json.JSONDecodeError:
        return []
app.jinja_env.filters['json_loads_safe'] = json_loads_safe


def simple_cache(expiry_seconds_default=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            expiry = expiry_seconds_default or app.config['CACHE_EXPIRY_SECONDS']
            key_parts = [func.__name__] + list(map(str, args)) + sorted(kwargs.items())
            cache_key = hashlib.md5(str(key_parts).encode('utf-8')).hexdigest()
            cached_entry = API_CACHE.get(cache_key)
            if cached_entry and (time.time() - cached_entry[1] < expiry):
                app.logger.debug(f"Cache HIT for {func.__name__}")
                return cached_entry[0]
            app.logger.debug(f"Cache MISS for {func.__name__}. Calling function.")
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

@simple_cache(expiry_seconds_default=3600 * 12) # Cache Groq responses for 12 hours
def get_article_analysis_with_groq(article_text, article_title=""):
    if not groq_client: return {"error": "AI analysis service not available."}
    if not article_text or not article_text.strip(): return {"error": "No text provided for AI analysis."}
    app.logger.info(f"Requesting Groq analysis for: {article_title[:50]}...")
    system_prompt = (
        "You are an expert news analyst. Analyze the following article. "
        "1. Provide a concise, neutral summary (3-4 paragraphs). "
        "2. List 5-7 key takeaways as bullet points. Each takeaway must be a complete sentence. "
        "3. Provide 3-5 relevant keywords or tags for this article as a list of strings. " # NEW requirement for tags
        "Format your entire response as a single JSON object with keys 'summary' (string), 'takeaways' (a list of strings), and 'tags' (a list of strings)."
    )
    human_prompt = f"Article Title: {article_title}\n\nArticle Text:\n{article_text[:20000]}" # Max length for Groq
    try:
        json_model = groq_client.bind(response_format={"type": "json_object"})
        ai_response = json_model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        analysis = json.loads(ai_response.content)
        if 'summary' in analysis and 'takeaways' in analysis and 'tags' in analysis: # Check for tags
            return {
                "groq_summary": analysis.get("summary"),
                "groq_takeaways": analysis.get("takeaways"),
                "tags": analysis.get("tags"), # Return tags
                "error": None
            }
        missing_keys = [key for key in ['summary', 'takeaways', 'tags'] if key not in analysis]
        raise ValueError(f"Missing keys in Groq JSON: {', '.join(missing_keys)}")
    except (json.JSONDecodeError, ValueError, LangChainException) as e:
        app.logger.error(f"Groq analysis failed for '{article_title[:50]}': {e}")
        return {"error": f"AI analysis failed: {str(e)}"}
    except Exception as e:
        app.logger.error(f"Unexpected error during Groq analysis for '{article_title[:50]}': {e}", exc_info=True)
        return {"error": "An unexpected error occurred during AI analysis."}

# ==============================================================================
# --- Celery Tasks ---
# ==============================================================================
@celery.task(name='Rev14.analyze_community_article_content_task', bind=True)
def analyze_community_article_content_task(self, article_id, article_text, article_title):
    app.logger.info(f"Celery task started: analyze_community_article_content_task for article_id {article_id}")
    try:
        # No need to import db, CommunityArticle if ContextTask is working correctly and app_context is pushed.
        # However, explicit imports inside task can be safer depending on Celery setup.
        # from Rev14 import db, CommunityArticle, get_article_analysis_with_groq # Potentially needed
        
        analysis_result = get_article_analysis_with_groq(article_text, article_title) # This function is defined above
        
        article = CommunityArticle.query.get(article_id)
        if article:
            if analysis_result and not analysis_result.get("error"):
                article.groq_summary = analysis_result.get('groq_summary')
                takeaways = analysis_result.get('groq_takeaways')
                if takeaways and isinstance(takeaways, list):
                    article.groq_takeaways = json.dumps(takeaways)
                
                tags = analysis_result.get('tags')
                if tags and isinstance(tags, list):
                    article.tags = json.dumps(tags) # Store tags as JSON string
                else:
                    article.tags = json.dumps([]) # Ensure it's a JSON list even if empty/error

                db.session.commit()
                app.logger.info(f"Celery task: Successfully updated Groq analysis and tags for article {article_id}")
                return {"status": "success", "article_id": article_id, "summary_present": bool(article.groq_summary), "tags_present": bool(article.tags)}
            else:
                error_msg = analysis_result.get('error') if analysis_result else 'No result from Groq'
                app.logger.error(f"Celery task: Groq analysis failed for article {article_id}: {error_msg}")
                return {"status": "error", "article_id": article_id, "error": error_msg}
        else:
            app.logger.error(f"Celery task: CommunityArticle with id {article_id} not found.")
            return {"status": "error", "article_id": article_id, "error": "Article not found in DB"}
    except Exception as e:
        app.logger.error(f"Celery task unhandled exception for article_id {article_id}: {e}", exc_info=True)
        # self.retry(exc=e, countdown=60) # Optional: retry task after 1 minute
        return {"status": "exception", "article_id": article_id, "error": str(e)}

# ==============================================================================
# --- NEWS FETCHING ---
# ==============================================================================
@simple_cache()
def fetch_news_from_api():
    if not newsapi:
        app.logger.error("NewsAPI client not initialized. Cannot fetch news.")
        return []

    from_date_utc = datetime.now(timezone.utc) - timedelta(days=app.config['NEWS_API_DAYS_AGO'])
    from_date_str = from_date_utc.strftime('%Y-%m-%dT%H:%M:%S')
    to_date_utc = datetime.now(timezone.utc)
    to_date_str = to_date_utc.strftime('%Y-%m-%dT%H:%M:%S')

    all_raw_articles = []
    try:
        app.logger.info("Attempt 1: Fetching top headlines from country: 'in'")
        top_headlines_response = newsapi.get_top_headlines(
            country='in', language='en', page_size=app.config['NEWS_API_PAGE_SIZE']
        )
        status = top_headlines_response.get('status')
        total_results = top_headlines_response.get('totalResults', 0)
        app.logger.info(f"Top-Headlines API Response -> Status: {status}, TotalResults: {total_results}")
        if status == 'ok' and total_results > 0: all_raw_articles.extend(top_headlines_response['articles'])
        elif status == 'error': app.logger.error(f"NewsAPI Error (Top-Headlines): {top_headlines_response.get('message')}")
    except NewsAPIException as e: app.logger.error(f"NewsAPIException (Top-Headlines): {e}", exc_info=True)
    except Exception as e: app.logger.error(f"Generic Exception (Top-Headlines): {e}", exc_info=True)


    try:
        app.logger.info(f"Attempt 2: Fetching 'everything' with query: {app.config['NEWS_API_QUERY']} from {from_date_str} to {to_date_str}")
        everything_response = newsapi.get_everything(
            q=app.config['NEWS_API_QUERY'], from_param=from_date_str, to=to_date_str,
            language='en', sort_by=app.config['NEWS_API_SORT_BY'], page_size=app.config['NEWS_API_PAGE_SIZE']
        )
        status = everything_response.get('status')
        total_results = everything_response.get('totalResults', 0)
        app.logger.info(f"Everything API Response -> Status: {status}, TotalResults: {total_results}")
        if status == 'ok' and total_results > 0: all_raw_articles.extend(everything_response['articles'])
        elif status == 'error': app.logger.error(f"NewsAPI Error (Everything): {everything_response.get('message')}")
    except NewsAPIException as e: app.logger.error(f"NewsAPIException (Everything): {e}", exc_info=True)
    except Exception as e: app.logger.error(f"Generic Exception (Everything): {e}", exc_info=True)


    if not all_raw_articles:
        try:
            app.logger.warning("No articles from primary calls. Trying Fallback with domains.")
            domains_to_check = app.config['NEWS_API_DOMAINS']
            app.logger.info(f"Attempt 3 (Fallback): Fetching from domains: {domains_to_check} from {from_date_str} to {to_date_str}")
            fallback_response = newsapi.get_everything(
                domains=domains_to_check, from_param=from_date_str, to=to_date_str,
                language='en', sort_by=app.config['NEWS_API_SORT_BY'], page_size=app.config['NEWS_API_PAGE_SIZE']
            )
            status = fallback_response.get('status')
            total_results = fallback_response.get('totalResults', 0)
            app.logger.info(f"Fallback API Response -> Status: {status}, TotalResults: {total_results}")
            if status == 'ok' and total_results > 0: all_raw_articles.extend(fallback_response['articles'])
            elif status == 'error': app.logger.error(f"NewsAPI Error (Fallback): {fallback_response.get('message')}")
        except NewsAPIException as e: app.logger.error(f"NewsAPIException (Fallback): {e}", exc_info=True)
        except Exception as e: app.logger.error(f"Generic Exception (Fallback): {e}", exc_info=True)

    processed_articles, unique_urls = [], set()
    app.logger.info(f"Total raw articles fetched before deduplication: {len(all_raw_articles)}")
    for art_data in all_raw_articles:
        url = art_data.get('url')
        if not url or url in unique_urls: continue
        title = art_data.get('title')
        if not all([title, art_data.get('source'), art_data.get('description')]) or title == '[Removed]' or not title.strip(): continue
        unique_urls.add(url)
        article_id = generate_article_id(url)
        source_name = art_data['source'].get('name', 'Unknown Source')
        placeholder_text = urllib.parse.quote_plus(source_name[:20])
        standardized_article = {
            'id': article_id, 'title': title, 'description': art_data.get('description', ''),
            'url': url, 'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/0D2C54/FFFFFF?text={placeholder_text}',
            'publishedAt': art_data.get('publishedAt'), 'source': {'name': source_name}, 'is_community_article': False,
            'article_hash_id': article_id # For consistency with community article field name in templates/JS
        }
        MASTER_ARTICLE_STORE[article_id] = standardized_article
        processed_articles.append(standardized_article)
    
    processed_articles.sort(key=lambda x: x.get('publishedAt', '') or '' if isinstance(x.get('publishedAt'), str) else (x.get('publishedAt') or datetime.min.replace(tzinfo=timezone.utc)).isoformat(), reverse=True)
    app.logger.info(f"Total unique articles processed and ready to serve: {len(processed_articles)}.")
    return processed_articles

@simple_cache(expiry_seconds_default=3600 * 6)
def fetch_and_parse_article_content(article_hash_id, url):
    app.logger.info(f"Fetching content for API article ID: {article_hash_id}, URL: {url}")
    if not SCRAPER_API_KEY: return {"error": "Content fetching service unavailable."}
    params = {'api_key': SCRAPER_API_KEY, 'url': url}
    try:
        response = requests.get('http://api.scraperapi.com', params=params, timeout=45)
        response.raise_for_status()
        config = Config()
        config.fetch_images = False
        config.memoize_articles = False
        article_scraper = Article(url, config=config)
        article_scraper.download(input_html=response.text)
        article_scraper.parse()
        if not article_scraper.text: return {"error": "Could not extract text from the article."}
        article_title = article_scraper.title or MASTER_ARTICLE_STORE.get(article_hash_id, {}).get('title', 'Unknown Title')
        
        # For API articles, we only need summary and takeaways. Tags are for community articles.
        # If you want tags for API articles too, modify Groq call or its processing here.
        groq_analysis_result = get_article_analysis_with_groq(article_scraper.text, article_title)
        
        return_data = {"full_text": article_scraper.text, "groq_analysis": None, "error": None}

        if groq_analysis_result and not groq_analysis_result.get("error"):
            return_data["groq_analysis"] = {
                "groq_summary": groq_analysis_result.get("groq_summary"),
                "groq_takeaways": groq_analysis_result.get("groq_takeaways")
                # Not including 'tags' for API articles from this function by default
            }
        elif groq_analysis_result: # There was an error in analysis
             return_data["error"] = groq_analysis_result.get("error")
        else: # No analysis result at all
            return_data["error"] = "AI analysis unavailable."
            
        return return_data

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Failed to fetch article content via proxy for {url}: {e}")
        return {"error": f"Failed to fetch article content via proxy: {str(e)}"}
    except Exception as e:
        app.logger.error(f"Failed to parse article content for {url}: {e}", exc_info=True)
        return {"error": f"Failed to parse article content: {str(e)}"}

# ==============================================================================
# --- 6. Flask Routes ---
# ==============================================================================
@app.context_processor
def inject_global_vars():
    user_is_logged_in = 'user_id' in session
    return {
        'categories': app.config['CATEGORIES'], 
        'current_year': datetime.utcnow().year, 
        'session': session, 
        'request': request,
        'user_is_logged_in': user_is_logged_in # Add this for convenience in templates
    }

def get_paginated_articles(articles, page, per_page):
    total = len(articles)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_items = articles[start:end]
    total_pages = (total + per_page - 1) // per_page if per_page > 0 else 0
    return paginated_items, total_pages

def get_sort_key(article):
    date_val = None
    # For sorting bookmarks page by bookmarked_at if available
    if isinstance(article, dict) and 'bookmarked_at' in article:
        date_val = article.get('bookmarked_at')
    elif hasattr(article, 'bookmarked_at') and article.bookmarked_at:
         date_val = article.bookmarked_at
    elif isinstance(article, dict): date_val = article.get('publishedAt')
    elif hasattr(article, 'published_at'): date_val = article.published_at
    
    if not date_val: return datetime.min.replace(tzinfo=timezone.utc)
    if isinstance(date_val, str):
        try: return datetime.fromisoformat(date_val.replace('Z', '+00:00'))
        except ValueError:
            app.logger.warning(f"Could not parse date string: {date_val}")
            return datetime.min.replace(tzinfo=timezone.utc)
    if isinstance(date_val, datetime): return date_val if date_val.tzinfo else pytz.utc.localize(date_val)
    return datetime.min.replace(tzinfo=timezone.utc)

@app.route('/')
@app.route('/page/<int:page>')
@app.route('/category/<category_name>')
@app.route('/category/<category_name>/page/<int:page>')
def index(page=1, category_name='All Articles'):
    session['previous_list_page'] = request.full_path # Save for back button on article detail
    per_page = app.config['PER_PAGE']
    all_display_articles = []

    if category_name == 'Community Hub':
        db_articles_query = CommunityArticle.query.options(joinedload(CommunityArticle.author)).order_by(CommunityArticle.published_at.desc())
        db_articles_paginated = db_articles_query.paginate(page=page, per_page=per_page, error_out=False)
        
        for art in db_articles_paginated.items:
            art.is_community_article = True
            all_display_articles.append(art)
        total_pages = db_articles_paginated.pages
    else: # All Articles or any other (future) NewsAPI-based category
        api_articles = fetch_news_from_api() # This is cached
        for art_dict in api_articles:
            art_dict_copy = art_dict.copy() # Avoid modifying MASTER_ARTICLE_STORE items directly
            art_dict_copy['is_community_article'] = False
            all_display_articles.append(art_dict_copy)
        # Manual pagination for combined list if other sources were merged here.
        # For now, only API articles are in 'All Articles'
        all_display_articles.sort(key=get_sort_key, reverse=True)
        paginated_articles_list, total_pages = get_paginated_articles(all_display_articles, page, per_page)
        all_display_articles = paginated_articles_list # Replace with paginated list


    # featured_article_on_this_page logic might need adjustment if pagination source changes
    featured_article_on_this_page = (page == 1 and category_name == 'All Articles' and not request.args.get('query') and all_display_articles)
    
    return render_template("INDEX_HTML_TEMPLATE", articles=all_display_articles, selected_category=category_name, current_page=page, total_pages=total_pages, featured_article_on_this_page=featured_article_on_this_page)

@app.route('/search')
@app.route('/search/page/<int:page>')
def search_results(page=1):
    session['previous_list_page'] = request.full_path
    query_str = request.args.get('query', '').strip()
    per_page = app.config['PER_PAGE']
    if not query_str: return redirect(url_for('index'))
    app.logger.info(f"Search query: '{query_str}'")
    
    api_results = []
    # Search MASTER_ARTICLE_STORE (cached API articles)
    for art_id, art_data in MASTER_ARTICLE_STORE.items():
        if query_str.lower() in art_data.get('title', '').lower() or \
           query_str.lower() in art_data.get('description', '').lower():
            art_copy = art_data.copy()
            art_copy['is_community_article'] = False
            api_results.append(art_copy)

    # Search Community Articles
    community_db_articles_query = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter(
        db.or_(
            CommunityArticle.title.ilike(f'%{query_str}%'), 
            CommunityArticle.description.ilike(f'%{query_str}%'),
            CommunityArticle.full_text.ilike(f'%{query_str}%'), # Search full text
            CommunityArticle.tags.ilike(f'%"{query_str}"%') # Search in JSON tags string
        )
    )
    community_db_articles = []
    for art in community_db_articles_query.all(): # Fetch all then sort/paginate
        art.is_community_article = True
        community_db_articles.append(art)
    
    all_search_results = api_results + community_db_articles
    all_search_results.sort(key=get_sort_key, reverse=True)
    
    paginated_search_articles, total_pages = get_paginated_articles(all_search_results, page, per_page)
    
    return render_template("INDEX_HTML_TEMPLATE", articles=paginated_search_articles, selected_category=f"Search: {query_str}", current_page=page, total_pages=total_pages, featured_article_on_this_page=False, query=query_str)


@app.route('/article/<article_hash_id>')
def article_detail(article_hash_id):
    article_data, is_community_article, comments_for_template, all_article_comments_list, comment_data = None, False, [], [], {}
    previous_list_page = session.get('previous_list_page', url_for('index'))
    is_bookmarked = False # NEW

    article_db = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter_by(article_hash_id=article_hash_id).first()

    if article_db:
        article_data = article_db
        is_community_article = True
        try: article_data.parsed_takeaways = json.loads(article_data.groq_takeaways) if article_data.groq_takeaways else []
        except json.JSONDecodeError:
            app.logger.error(f"JSONDecodeError for groq_takeaways on community article {article_data.article_hash_id}")
            article_data.parsed_takeaways = []
        try: article_data.parsed_tags = json.loads(article_data.tags) if article_data.tags else [] # NEW: Parse tags
        except json.JSONDecodeError:
            app.logger.error(f"JSONDecodeError for tags on community article {article_data.article_hash_id}")
            article_data.parsed_tags = []
        
        all_article_comments_list = Comment.query.options(
            joinedload(Comment.author), 
            joinedload(Comment.replies).options(joinedload(Comment.author))
        ).filter_by(community_article_id=article_db.id).order_by(Comment.timestamp.asc()).all()
        if 'user_id' in session: # Check bookmark status
            is_bookmarked = Bookmark.query.filter_by(user_id=session['user_id'], community_article_id=article_db.id).first() is not None
    else:
        article_api_dict = MASTER_ARTICLE_STORE.get(article_hash_id)
        if article_api_dict:
            article_data = article_api_dict.copy()
            is_community_article = False
            # Add article_hash_id for consistency if not present (it should be from fetch_news_from_api)
            article_data.setdefault('article_hash_id', article_hash_id)

            all_article_comments_list = Comment.query.options(
                joinedload(Comment.author),
                joinedload(Comment.replies).options(joinedload(Comment.author))
            ).filter_by(api_article_hash_id=article_hash_id).order_by(Comment.timestamp.asc()).all()
            if 'user_id' in session: # Check bookmark status
                is_bookmarked = Bookmark.query.filter_by(user_id=session['user_id'], api_article_hash_id=article_hash_id).first() is not None
        else:
            flash("Article not found.", "danger"); return redirect(previous_list_page)

    comments_for_template = [c for c in all_article_comments_list if c.parent_id is None]
    
    if all_article_comments_list:
        comment_ids = [c.id for c in all_article_comments_list for c_reply in [c] + (c.replies or [])] # Get all comment IDs including replies
        for c_id in comment_ids: comment_data[c_id] = {'likes': 0, 'dislikes': 0, 'user_vote': 0}
        
        vote_counts_query = db.session.query(
            CommentVote.comment_id,
            func.sum(case((CommentVote.vote_type == 1, 1), else_=0)).label('likes'),
            func.sum(case((CommentVote.vote_type == -1, 1), else_=0)).label('dislikes')
        ).filter(CommentVote.comment_id.in_(comment_ids)).group_by(CommentVote.comment_id).all()
        for c_id, likes, dislikes in vote_counts_query:
            if c_id in comment_data: comment_data[c_id]['likes'] = likes; comment_data[c_id]['dislikes'] = dislikes
        
        if 'user_id' in session:
            user_votes = CommentVote.query.filter(CommentVote.comment_id.in_(comment_ids), CommentVote.user_id == session['user_id']).all()
            for vote in user_votes:
                if vote.comment_id in comment_data: comment_data[vote.comment_id]['user_vote'] = vote.vote_type

    # Ensure consistent 'is_community_article' attribute for the template
    if isinstance(article_data, dict): article_data['is_community_article'] = False
    elif article_data: article_data.is_community_article = True
    
    # For API articles, check if Groq analysis is already in MASTER_ARTICLE_STORE
    if not is_community_article and article_data and 'groq_analysis' not in article_data:
        article_data['groq_analysis_pending'] = True # Flag for template to show loader or fetch
    elif not is_community_article and article_data and article_data.get('groq_analysis') and article_data['groq_analysis'].get("error"):
        article_data['groq_analysis_error'] = article_data['groq_analysis'].get("error")


    return render_template("ARTICLE_HTML_TEMPLATE", 
                           article=article_data, 
                           is_community_article=is_community_article, 
                           comments=comments_for_template, 
                           comment_data=comment_data, 
                           previous_list_page=previous_list_page,
                           is_bookmarked=is_bookmarked) # Pass bookmark status

@app.route('/get_article_content/<article_hash_id>')
def get_article_content_json(article_hash_id):
    article_data = MASTER_ARTICLE_STORE.get(article_hash_id)
    if not article_data or 'url' not in article_data: return jsonify({"error": "Article data or URL not found"}), 404
    
    # Check if analysis is already cached in MASTER_ARTICLE_STORE
    if 'groq_analysis' in article_data and article_data['groq_analysis'] is not None:
        app.logger.info(f"Returning cached Groq analysis for API article ID: {article_hash_id}")
        # Ensure the structure matches what fetch_and_parse_article_content would return for groq_analysis part
        return jsonify({
            "groq_analysis": article_data['groq_analysis'], 
            "error": article_data['groq_analysis'].get("error") if isinstance(article_data['groq_analysis'], dict) else None
        })
        
    processed_content = fetch_and_parse_article_content(article_hash_id, article_data['url']) # This is cached
    
    if processed_content and not processed_content.get("error"):
        # Store only the 'groq_analysis' part in MASTER_ARTICLE_STORE to avoid storing full_text repeatedly
        MASTER_ARTICLE_STORE[article_hash_id]['groq_analysis'] = processed_content.get('groq_analysis')
        return jsonify({"groq_analysis": processed_content.get('groq_analysis'), "error": None})
    elif processed_content: # Error occurred during processing
        return jsonify({"groq_analysis": None, "error": processed_content.get("error")})
    else: # Should not happen if fetch_and_parse_article_content always returns a dict
        return jsonify({"groq_analysis": None, "error": "Unknown error processing article content."})


@app.route('/add_comment/<article_hash_id>', methods=['POST'])
@login_required
def add_comment(article_hash_id):
    content = request.json.get('content', '').strip()
    parent_id = request.json.get('parent_id') 
    if not content: return jsonify({"error": "Comment cannot be empty."}), 400
    
    user = User.query.get(session['user_id']) # Already joined with author for eager loading
    if not user: 
        app.logger.error(f"User not found in add_comment for user_id {session.get('user_id')}")
        return jsonify({"error": "User not found."}), 401
    
    new_comment = None
    community_article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
    if community_article: 
        new_comment = Comment(content=content, user_id=user.id, community_article_id=community_article.id, parent_id=parent_id)
    elif article_hash_id in MASTER_ARTICLE_STORE: 
        new_comment = Comment(content=content, user_id=user.id, api_article_hash_id=article_hash_id, parent_id=parent_id)
    else: 
        return jsonify({"error": "Article not found."}), 404
        
    db.session.add(new_comment)
    db.session.commit()
    # Author is already eager loaded via user.comments backref or direct query if needed
    author_name = new_comment.author.name if new_comment.author else "Anonymous"
    
    return jsonify({
        "success": True, 
        "comment": {
            "id": new_comment.id, 
            "content": new_comment.content, 
            "timestamp": new_comment.timestamp.isoformat(), 
            "author": {"name": author_name, "username": new_comment.author.username if new_comment.author else None}, # Add username for profile link
            "parent_id": new_comment.parent_id,
            "replies": [] # New comments won't have replies yet
        }
    }), 201

@app.route('/vote_comment/<int:comment_id>', methods=['POST'])
@login_required
def vote_comment(comment_id):
    comment = Comment.query.get_or_404(comment_id)
    vote_type = request.json.get('vote_type')
    if vote_type not in [1, -1]: return jsonify({"error": "Invalid vote type."}), 400
    
    existing_vote = CommentVote.query.filter_by(user_id=session['user_id'], comment_id=comment_id).first()
    if existing_vote:
        if existing_vote.vote_type == vote_type: 
            db.session.delete(existing_vote) # User is toggling off their vote
        else: 
            existing_vote.vote_type = vote_type # User is changing their vote
    else: 
        db.session.add(CommentVote(user_id=session['user_id'], comment_id=comment_id, vote_type=vote_type))
    
    try:
        db.session.commit()
    except IntegrityError: # Should not happen with the above logic but as a safeguard
        db.session.rollback()
        return jsonify({"success": False, "error": "Could not record vote due to a conflict."}), 409

    likes = CommentVote.query.filter_by(comment_id=comment_id, vote_type=1).count()
    dislikes = CommentVote.query.filter_by(comment_id=comment_id, vote_type=-1).count()
    
    # Determine the user's current vote status after the transaction
    current_user_vote = 0
    final_vote_check = CommentVote.query.filter_by(user_id=session['user_id'], comment_id=comment_id).first()
    if final_vote_check:
        current_user_vote = final_vote_check.vote_type

    return jsonify({"success": True, "likes": likes, "dislikes": dislikes, "user_vote": current_user_vote}), 200


@app.route('/post_article', methods=['POST'])
@login_required
def post_article():
    title, description, content, source_name, image_url = map(
        lambda x: request.form.get(x, '').strip(), 
        ['title', 'description', 'content', 'sourceName', 'imageUrl']
    )
    source_name = source_name or 'Community Post'
    if not all([title, description, content, source_name]):
        flash("Title, Description, Full Content, and Source Name are required.", "danger")
        return redirect(request.referrer or url_for('index'))

    article_hash_id = generate_article_id(title + str(session['user_id']) + str(time.time()))
    
    new_article = CommunityArticle(
        article_hash_id=article_hash_id, 
        title=title, 
        description=description, 
        full_text=content, 
        source_name=source_name, 
        image_url=image_url or f'https://via.placeholder.com/400x220/1E3A5E/FFFFFF?text={urllib.parse.quote_plus(title[:20])}', 
        user_id=session['user_id'], 
        published_at=datetime.now(timezone.utc),
        # AI fields will be populated by Celery task, initialize as None or empty JSON
        groq_summary=None, 
        groq_takeaways=json.dumps([]),
        tags=json.dumps([]) 
    )
    db.session.add(new_article)
    db.session.commit() # Commit to get the new_article.id

    # Offload AI analysis to Celery
    # Ensure Celery is running and configured, otherwise this might run eagerly or fail
    # if not app.config.get('CELERY_TASK_ALWAYS_EAGER'): # Only log if truly async
    app.logger.info(f"Queuing Celery task for AI analysis of new community article ID: {new_article.id}")
    
    analyze_community_article_content_task.delay(new_article.id, content, title)
    
    flash("Your article has been posted! AI analysis is in progress.", "success")
    return redirect(url_for('article_detail', article_hash_id=new_article.article_hash_id))

# --- NEW User Profile Route ---
@app.route('/profile/<username>')
@app.route('/profile/<username>/page/<int:page>')
def user_profile(username, page=1):
    profile_user = User.query.filter_by(username=username).first_or_404()
    per_page = app.config['PER_PAGE'] - 3 # Show slightly fewer articles on profile

    user_articles_query = CommunityArticle.query.filter_by(user_id=profile_user.id)\
                                             .order_by(CommunityArticle.published_at.desc())
    
    user_articles_pagination = user_articles_query.paginate(page=page, per_page=per_page, error_out=False)
    
    articles_for_template = []
    for art in user_articles_pagination.items:
        art.is_community_article = True # Ensure flag is set
        # No need to parse tags here if template uses json_loads_safe filter
        articles_for_template.append(art)

    return render_template("PROFILE_HTML_TEMPLATE", 
                           profile_user=profile_user, 
                           articles=articles_for_template, 
                           pagination=user_articles_pagination,
                           selected_category=f"{profile_user.name}'s Profile")

# --- NEW Bookmark Routes ---
@app.route('/bookmark_article/<article_hash_id>', methods=['POST'])
@login_required
def bookmark_article(article_hash_id):
    is_api_article = article_hash_id in MASTER_ARTICLE_STORE
    community_article_db = None
    if not is_api_article:
        community_article_db = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()

    if not is_api_article and not community_article_db:
        return jsonify({"success": False, "error": "Article not found"}), 404

    existing_bookmark = None
    bookmark_args = {'user_id': session['user_id']}
    if is_api_article:
        bookmark_args['api_article_hash_id'] = article_hash_id
    elif community_article_db : # Must be community article
        bookmark_args['community_article_id'] = community_article_db.id
    else: # Should not be reached if previous checks are correct
         return jsonify({"success": False, "error": "Invalid article type for bookmarking"}), 400


    existing_bookmark = Bookmark.query.filter_by(**bookmark_args).first()

    if existing_bookmark:
        db.session.delete(existing_bookmark)
        bookmarked_status = False
        message = "Bookmark removed."
    else:
        new_bookmark = Bookmark(**bookmark_args)
        db.session.add(new_bookmark)
        bookmarked_status = True
        message = "Article bookmarked!"
    
    try:
        db.session.commit()
    except IntegrityError as e:
        db.session.rollback()
        app.logger.error(f"Bookmark integrity error for article {article_hash_id}, user {session['user_id']}: {e}")
        return jsonify({"success": False, "error": "Could not update bookmark due to a conflict. Please try again."}), 409
        
    return jsonify({"success": True, "bookmarked": bookmarked_status, "message": message})


@app.route('/bookmarks')
@app.route('/bookmarks/page/<int:page>')
@login_required
def bookmarks_page(page=1):
    session['previous_list_page'] = request.full_path
    user_id = session['user_id']
    per_page = app.config['PER_PAGE']

    # Fetch all bookmark entries for the user first
    all_user_bookmarks = Bookmark.query.filter_by(user_id=user_id).order_by(Bookmark.timestamp.desc()).all()

    bookmarked_articles_combined = []
    for bm in all_user_bookmarks:
        if bm.api_article_hash_id:
            article_data = MASTER_ARTICLE_STORE.get(bm.api_article_hash_id)
            if article_data:
                art_copy = article_data.copy()
                art_copy['is_community_article'] = False
                art_copy['bookmarked_at'] = bm.timestamp
                art_copy['article_hash_id'] = bm.api_article_hash_id # Ensure consistent key
                bookmarked_articles_combined.append(art_copy)
        elif bm.community_article_id:
            # Eager load author when fetching community article for bookmark
            community_art = CommunityArticle.query.options(joinedload(CommunityArticle.author))\
                                              .filter_by(id=bm.community_article_id).first()
            if community_art:
                community_art.is_community_article = True
                community_art.bookmarked_at = bm.timestamp
                # Tags parsing can be done here or in template with json_loads_safe
                bookmarked_articles_combined.append(community_art)
    
    # Sort by bookmark timestamp (already done by query for initial list, but good if merging from multiple sources)
    # bookmarked_articles_combined.sort(key=get_sort_key, reverse=True) # get_sort_key needs to handle bookmarked_at

    paginated_bookmarks, total_pages = get_paginated_articles(bookmarked_articles_combined, page, per_page)

    return render_template("BOOKMARKS_HTML_TEMPLATE", 
                           articles=paginated_bookmarks, 
                           current_page=page, 
                           total_pages=total_pages, 
                           selected_category="My Bookmarks")

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
            flash(f'Registration successful, {name}! Please log in.', 'success')
            return redirect(url_for('login'))
        # Fall through to re-render register form with flashed messages
        # No need for explicit redirect(url_for('register')) here if we just return the template render
    return render_template("REGISTER_HTML_TEMPLATE")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        username, password = request.form.get('username', '').strip().lower(), request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session.permanent = True; session['user_id'] = user.id; session['user_name'] = user.name
            session['user_username'] = user.username # Store username for profile links
            flash(f"Welcome back, {user.name}!", "success")
            next_url = request.args.get('next')
            # session.pop('previous_list_page', None) # Keep previous_list_page for back button navigation
            return redirect(next_url or url_for('index'))
        else: flash('Invalid username or password.', 'danger')
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
    if not email: flash('Email is required to subscribe.', 'warning')
    elif Subscriber.query.filter_by(email=email).first(): flash('You are already subscribed.', 'info')
    else:
        try: 
            db.session.add(Subscriber(email=email))
            db.session.commit()
            flash('Thank you for subscribing!', 'success')
        except IntegrityError: # Handles race conditions or other DB unique constraint violations
            db.session.rollback()
            flash('This email is already subscribed or an error occurred.', 'warning')
        except Exception as e: 
            db.session.rollback()
            app.logger.error(f"Error subscribing email {email}: {e}")
            flash('Could not subscribe. Please try again.', 'danger')
    return redirect(request.referrer or url_for('index'))

@app.errorhandler(404)
def page_not_found(e): return render_template("404_TEMPLATE"), 404
@app.errorhandler(500)
def internal_server_error(e): 
    db.session.rollback() # Ensure session is clean after an error
    app.logger.error(f"500 error at {request.url}: {e}", exc_info=True)
    return render_template("500_TEMPLATE"), 500

# Rev14.py - CONTINUED

# ==============================================================================
# --- 7. HTML Templates (Stored in memory) ---
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
            --tag-bg-light: #e9ecef; --tag-text-light: var(--primary-color);
            --tag-bg-dark: #343a40; --tag-text-dark: #ced4da;
        }
        body { padding-top: 145px; font-family: 'Roboto', sans-serif; line-height: 1.65; color: var(--text-color); background-color: var(--light-bg); display: flex; flex-direction: column; min-height: 100vh; transition: background-color 0.3s ease, color 0.3s ease; }
        .main-content { flex-grow: 1; }
        body.dark-mode { 
            --primary-color: #1E3A5E; --primary-light: #2A4B7C; --secondary-color: #D4A017; --secondary-light: #E7B400; --accent-color: #FF983E; --text-color: #E9ECEF; --text-muted-color: #ADB5BD; --light-bg: #121212; --white-bg: #1E1E1E; --card-border-color: #333333; --footer-bg: #0A0A0A; --footer-text: rgba(255,255,255,0.7); --primary-color-rgb: 30, 58, 94; --secondary-color-rgb: 212, 160, 23;
            --tag-bg-light: var(--tag-bg-dark); --tag-text-light: var(--tag-text-dark);
        }
        body.dark-mode .navbar-main { background: linear-gradient(135deg, #0A1A2F, #10233B); border-bottom: 1px solid #2A4B7C; }
        body.dark-mode .category-nav { background: #1A1A1A; border-bottom: 1px solid #2A2A2A; }
        body.dark-mode .category-link { color: var(--text-muted-color) !important; }
        body.dark-mode .category-link.active { background: var(--primary-color) !important; color: var(--white-bg) !important; }
        body.dark-mode .category-link:hover:not(.active) { background: #2C2C2C !important; color: var(--secondary-color) !important; }
        body.dark-mode .article-card, body.dark-mode .featured-article, body.dark-mode .article-full-content-wrapper, body.dark-mode .auth-container, body.dark-mode .static-content-wrapper, body.dark-mode .profile-header { background-color: var(--white-bg); border-color: var(--card-border-color); }
        body.dark-mode .article-title a, body.dark-mode h1, body.dark-mode h2, body.dark-mode h3, body.dark-mode h4, body.dark-mode h5, body.dark-mode .auth-title, body.dark-mode .profile-username { color: var(--text-color) !important; }
        body.dark-mode .article-description, body.dark-mode .meta-item, body.dark-mode .content-text p, body.dark-mode .article-meta-detailed, body.dark-mode .comment-content, body.dark-mode .comment-date, body.dark-mode .profile-bio { color: var(--text-muted-color) !important; }
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
        .article-card, .featured-article, .article-full-content-wrapper, .auth-container, .static-content-wrapper, .profile-header { background: var(--white-bg); border-radius: 10px; transition: all 0.3s ease; border: 1px solid var(--card-border-color); box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
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
        .btn-primary-modal { background-color: var(--primary-color); border-color: var(--primary-color); color:white; padding: 0.6rem 1.2rem; font-weight:600; }
        .btn-primary-modal:hover { background-color: var(--primary-light); border-color: var(--primary-light); }
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
        @media (max-width: 991.98px) { /* Adjusted breakpoint for navbar changes */
            body { padding-top: 180px; } /* Increased padding for taller mobile navbar */
            .navbar-main { padding-bottom: 0.5rem; height: auto;}
            .navbar-content-wrapper { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
            .navbar-brand-custom { margin-bottom: 0.5rem; }
            .search-form-container { width: 100%; order: 3; margin-top:0.5rem; padding: 0; }
            .header-controls { position: absolute; top: 0.9rem; right: 1rem; order: 2; }
            .category-nav { top: 130px; } /* Adjusted based on navbar height */
            .header-controls .dropdown-menu {transform: translateX(-80%) !important;}
        }
         @media (max-width: 767.98px) {
            body { padding-top: 170px; }
            .category-nav { top: 120px; }
            .featured-article .row { flex-direction: column; }
            .featured-image { margin-bottom: 1rem; height: 250px; }
        }
        @media (max-width: 575.98px) {
            .navbar-brand-custom { font-size: 1.8rem;}
            .header-controls { gap: 0.3rem; }
            .header-btn { padding: 0.4rem 0.8rem; font-size: 0.8rem; }
            .dark-mode-toggle { font-size: 1rem; }
            .header-controls .dropdown-menu {min-width: 200px; transform: translateX(-60%) !important;}
        }
        .auth-container { max-width: 450px; margin: 3rem auto; padding: 2rem; }
        .auth-title { text-align: center; color: var(--primary-color); margin-bottom: 1.5rem; font-weight: 700;}
        body.dark-mode .auth-title { color: var(--secondary-color); }

        /* Comment Section Styles */
        .comment-section { margin-top: 3rem; }
        .comment-container { margin-bottom: 1.5rem; } 
        .comment-card { display: flex; gap: 1rem; }
        .comment-avatar { width: 40px; height: 40px; border-radius: 50%; background: var(--primary-light); color: white; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0; }
        body.dark-mode .comment-avatar { background: var(--primary-color); }
        .comment-body { flex-grow: 1; border-bottom: 1px solid var(--card-border-color); padding-bottom: 1rem; }
        .comment-container:last-child > .comment-card > .comment-body { border-bottom: none; } 
        .comment-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem; }
        .comment-author { font-weight: 600; color: var(--primary-color); }
        .comment-author a { color: var(--primary-color); text-decoration: none;}
        .comment-author a:hover { color: var(--secondary-color); }
        body.dark-mode .comment-author, body.dark-mode .comment-author a { color: var(--secondary-light); }
        body.dark-mode .comment-author a:hover { color: var(--accent-color); }
        .comment-date { font-size: 0.8rem; color: var(--text-muted-color); }
        .comment-content { font-size: 0.95rem; color: var(--text-color); margin-bottom: 0.5rem; white-space: pre-wrap; } 
        .comment-actions { display: flex; align-items: center; gap: 0.75rem; font-size: 0.85rem; }
        .comment-actions button { background: none; border: none; padding: 0.2rem 0.4rem; color: var(--text-muted-color); cursor: pointer; display: flex; align-items: center; gap: 0.3rem; transition: color 0.2s ease, background-color 0.2s ease; border-radius: 4px; }
        .comment-actions button:hover { color: var(--primary-color); background-color: rgba(var(--primary-color-rgb), 0.1); }
        body.dark-mode .comment-actions button:hover { color: var(--secondary-light); background-color: rgba(var(--secondary-color-rgb),0.2); }
        .comment-actions button.active { color: var(--primary-color); font-weight: 600; } /* For reply button, if needed */
        body.dark-mode .comment-actions button.active { color: var(--secondary-color); }
        .comment-actions .vote-btn.user-liked .fa-thumbs-up { color: var(--primary-color) !important; } 
        .comment-actions .vote-btn.user-disliked .fa-thumbs-down { color: var(--accent-color) !important; } 
        body.dark-mode .comment-actions .vote-btn.user-liked .fa-thumbs-up { color: var(--secondary-color) !important; }
        body.dark-mode .comment-actions .vote-btn.user-disliked .fa-thumbs-down { color: var(--accent-color) !important; } /* Keep accent for dislike in dark */
        .comment-actions .vote-count { font-weight: 500; min-width: 12px; text-align: center;}
        .comment-replies { margin-left: 30px; padding-left: 1.25rem; border-left: 2px solid var(--card-border-color); margin-top: 1rem; } 
        .reply-form-container { display: none; margin-top: 0.75rem; padding: 0.75rem; background-color: rgba(var(--primary-color-rgb), 0.03); border-radius: 6px;}
        body.dark-mode .reply-form-container { background-color: rgba(var(--secondary-color-rgb), 0.05); }
        .add-comment-form textarea { min-height: 100px; }

        /* Tag Styles */
        .article-tags { margin-top: 0.5rem; margin-bottom: 0.5rem; display: flex; flex-wrap: wrap; gap: 0.4rem; }
        .tag-badge { font-size: 0.7rem; padding: 0.25rem 0.6rem; border-radius: 15px; background-color: var(--tag-bg-light); color: var(--tag-text-light); text-decoration: none; border: 1px solid transparent; transition: all 0.2s ease; }
        .tag-badge:hover { background-color: var(--primary-color); color: white; border-color: var(--primary-light); }
        body.dark-mode .tag-badge { background-color: var(--tag-bg-dark); color: var(--tag-text-dark); }
        body.dark-mode .tag-badge:hover { background-color: var(--secondary-color); color: var(--primary-color); border-color: var(--secondary-light); }
        
        /* Bookmark Icon */
        .bookmark-btn { background: none; border: none; color: var(--text-muted-color); cursor: pointer; padding: 0.3rem; font-size: 1.1rem; transition: color 0.2s ease; }
        .bookmark-btn:hover { color: var(--secondary-color); }
        .bookmark-btn.bookmarked { color: var(--secondary-color); } /* Gold when bookmarked */
        body.dark-mode .bookmark-btn:hover { color: var(--secondary-light); }
        body.dark-mode .bookmark-btn.bookmarked { color: var(--secondary-light); }
        .article-card .bookmark-btn { position: absolute; top: 8px; right: 8px; z-index: 6; background-color: rgba(255,255,255,0.7); border-radius:50%; width:30px; height:30px; display:flex; align-items:center; justify-content:center;}
        body.dark-mode .article-card .bookmark-btn { background-color: rgba(30,30,30,0.7);}

        /* Profile Header */
        .profile-header { padding: 2rem; margin-bottom: 2rem; text-align: center; }
        .profile-avatar { width: 120px; height: 120px; border-radius: 50%; background: var(--primary-color); color: white; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 3rem; margin: 0 auto 1rem; }
        body.dark-mode .profile-avatar { background: var(--secondary-color); color: var(--primary-color); }
        .profile-username { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
        .profile-name { font-size: 1.1rem; color: var(--text-muted-color); margin-bottom: 0.5rem; }
        .profile-joined-date { font-size: 0.9rem; color: var(--text-muted-color); }
        .profile-no-articles {text-align: center; margin-top: 2rem; color: var(--text-muted-color);}
    </style>
    {% block head_extra %}{% endblock %}
</head>
<body class="{{ request.cookies.get('darkMode', 'disabled') }}">
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
                        <input type="search" name="query" class="form-control navbar-search" placeholder="Search news articles..." value="{{ request.args.get('query', '') }}">
                        <i class="fas fa-search search-icon"></i>
                        <button type="submit" class="d-none">Search</button>
                    </form>
                </div>
                <div class="header-controls animate-fade-in fade-in-delay-2">
                    <button class="header-btn dark-mode-toggle" aria-label="Toggle dark mode" title="Toggle Dark Mode">
                        <i class="fas fa-moon"></i>
                    </button>
                    {% if user_is_logged_in %}
                    <div class="dropdown">
                        <a href="#" class="header-btn dropdown-toggle" id="userDropdown" role="button" data-bs-toggle="dropdown" aria-expanded="false" title="User Menu">
                            <i class="fas fa-user me-1"></i> Hi, {{ session.user_name|truncate(10) }}
                        </a>
                        <ul class="dropdown-menu dropdown-menu-end" aria-labelledby="userDropdown">
                            <li><a class="dropdown-item" href="{{ url_for('user_profile', username=session.user_username) }}"><i class="fas fa-user-circle me-2"></i>My Profile</a></li>
                            <li><a class="dropdown-item" href="{{ url_for('bookmarks_page') }}"><i class="fas fa-bookmark me-2"></i>My Bookmarks</a></li>
                            <li><hr class="dropdown-divider"></li>
                            <li><a class="dropdown-item" href="{{ url_for('logout') }}"><i class="fas fa-sign-out-alt me-2"></i>Logout</a></li>
                        </ul>
                    </div>
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
                    <a href="{{ url_for('index', category_name=cat_item, page=1) }}" class="category-link {% if selected_category == cat_item %}active{% endif %}">
                        <i class="fas fa-{% if cat_item == 'All Articles' %}globe-americas{% elif cat_item == 'Community Hub' %}users{% endif %} me-1 d-none d-sm-inline"></i>
                        {{ cat_item }}
                    </a>
                {% endfor %}
            </div>
        </div>
    </nav>

    <main class="container main-content my-4">
        {% block content %}{% endblock %}
    </main>

    {% if user_is_logged_in %}
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
                        {% for cat_item in categories %}<a href="{{ url_for('index', category_name=cat_item, page=1) }}"><i class="fas fa-angle-right"></i> {{ cat_item }}</a>{% endfor %}
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
            <div class="copyright">&copy; {{ current_year }} Briefly. All rights reserved. Made with <i class="fas fa-heart text-danger"></i> in India.</div>
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
            document.cookie = "darkMode=" + theme + ";path=/;max-age=" + (60*60*24*365) + ";SameSite=Lax";
        }
        if(darkModeToggle) { darkModeToggle.addEventListener('click', () => { applyTheme(body.classList.contains('dark-mode') ? 'disabled' : 'enabled'); }); }

        let storedTheme = localStorage.getItem('darkMode');
        if (!storedTheme) {
            const cookieTheme = document.cookie.split('; ').find(row => row.startsWith('darkMode='))?.split('=')[1];
            if (cookieTheme) storedTheme = cookieTheme;
        }
        if (storedTheme) { applyTheme(storedTheme); } else { updateThemeIcon(); } // Apply if found or just update icon

        const addArticleBtn = document.getElementById('addArticleBtn');
        const addArticleModal = document.getElementById('addArticleModal');
        const closeModalBtn = document.getElementById('closeModalBtn');
        const cancelArticleBtn = document.getElementById('cancelArticleBtn');
        if(addArticleBtn && addArticleModal) {
            addArticleBtn.addEventListener('click', () => { addArticleModal.style.display = 'flex'; body.style.overflow = 'hidden'; });
            const closeModalFunction = () => { addArticleModal.style.display = 'none'; body.style.overflow = 'auto'; document.getElementById('addArticleForm').reset(); };
            if(closeModalBtn) closeModalBtn.addEventListener('click', closeModalFunction);
            if(cancelArticleBtn) cancelArticleBtn.addEventListener('click', closeModalFunction);
            addArticleModal.addEventListener('click', (e) => { if (e.target === addArticleModal) closeModalFunction(); });
        }

        const flashedAlerts = document.querySelectorAll('#alert-placeholder .alert');
        flashedAlerts.forEach(function(alert) { setTimeout(function() { const bsAlert = bootstrap.Alert.getOrCreateInstance(alert); if (bsAlert) bsAlert.close(); }, 7000); });
        
        // --- NEW: Bookmark handling ---
        document.querySelectorAll('.bookmark-btn-dynamic').forEach(button => {
            button.addEventListener('click', function(event) {
                event.preventDefault();
                const articleHashId = this.dataset.articleHashId;
                const icon = this.querySelector('i');
                const isCurrentlyBookmarked = icon.classList.contains('fas'); // 'fas' for solid (bookmarked), 'far' for regular

                fetch(`/bookmark_article/${articleHashId}`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        if (data.bookmarked) {
                            icon.classList.remove('far');
                            icon.classList.add('fas');
                            this.classList.add('bookmarked');
                            this.title = 'Remove Bookmark';
                        } else {
                            icon.classList.remove('fas');
                            icon.classList.add('far');
                            this.classList.remove('bookmarked');
                            this.title = 'Add Bookmark';
                        }
                        // Optionally, show a small toast/alert for success
                        // console.log(data.message);
                    } else {
                        // console.error('Failed to update bookmark:', data.error);
                        // Optionally, show error to user
                         const alertPlaceholder = document.getElementById('alert-placeholder');
                         if(alertPlaceholder) {
                            const alertDiv = document.createElement('div');
                            alertDiv.className = 'alert alert-danger alert-dismissible fade show alert-top';
                            alertDiv.role = 'alert';
                            alertDiv.innerHTML = `<span>${data.error || 'Could not update bookmark.'}</span><button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>`;
                            alertPlaceholder.appendChild(alertDiv);
                            setTimeout(() => { bootstrap.Alert.getOrCreateInstance(alertDiv)?.close(); }, 5000);
                         }
                    }
                })
                .catch(error => {
                    console.error('Error bookmarking:', error);
                    // Show generic error
                });
            });
        });
    });
    </script>
    {% block scripts_extra %}{% endblock %}
</body>
</html>
"""

INDEX_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}
    {% if query %}Search: {{ query|truncate(30) }}{% elif selected_category %}{{selected_category}}{% else %}Home{% endif %} - Briefly (India News)
{% endblock %}
{% block content %}
    {# Featured Article Section #}
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
                <div class="d-flex justify-content-between align-items-start">
                    <div class="article-meta mb-2">
                        <span class="badge bg-primary me-2" style="font-size:0.75rem;">
                            {% if art0.is_community_article and art0.author %}
                                <a href="{{ url_for('user_profile', username=art0.author.username) }}" class="text-white text-decoration-none">{{ art0.author.name|truncate(25) }}</a>
                            {% else %}
                                {{ art0.source.name|truncate(25) }}
                            {% endif %}
                        </span>
                        <span class="meta-item"><i class="far fa-calendar-alt"></i> {{ (art0.published_at | to_ist if art0.is_community_article else (art0.publishedAt | to_ist if art0.publishedAt else 'N/A')) }}</span>
                    </div>
                    {% if user_is_logged_in %}
                    <button class="bookmark-btn bookmark-btn-dynamic" data-article-hash-id="{{ art0.article_hash_id if art0.is_community_article else art0.id }}" title="Add Bookmark">
                        <i class="far fa-bookmark"></i> {# JS will fill this based on actual status later if needed #}
                    </button>
                    {% endif %}
                </div>
                <h2 class="mb-2 h4"><a href="{{ article_url }}" class="text-decoration-none article-title">{{ art0.title }}</a></h2>
                {% if art0.is_community_article and art0.tags %}
                    <div class="article-tags mb-2">
                        {% for tag in art0.tags | json_loads_safe | slice(0, 3) %} {# Show max 3 tags on card #}
                            <a href="{{ url_for('search_results', query=tag) }}" class="tag-badge">{{ tag }}</a>
                        {% endfor %}
                    </div>
                {% endif %}
                <p class="article-description flex-grow-1 small">{{ art0.description|truncate(220) }}</p>
                <a href="{{ article_url }}" class="read-more mt-auto align-self-start py-2 px-3" style="width:auto;">Read Full Article <i class="fas fa-arrow-right ms-1 small"></i></a>
            </div>
        </div>
    </article>
    {% elif not articles and selected_category != 'Community Hub' and not query %}
        <div class="alert alert-warning text-center my-4 p-3 small">No recent Indian news found. Please check back later.</div>
    {% elif not articles and selected_category == 'Community Hub' %}
        <div class="alert alert-info text-center my-4 p-3"><h4><i class="fas fa-feather-alt me-2"></i>No Articles Penned Yet</h4><p>No articles in the Community Hub. {% if user_is_logged_in %}Click the '+' button to share your insights!{% else %}Login to add articles.{% endif %}</p></div>
    {% elif not articles and query %}
        <div class="alert alert-info text-center my-5 p-4"><h4><i class="fas fa-search me-2"></i>No results for "{{ query }}"</h4><p>Try different keywords or browse categories.</p></div>
    {% endif %}

    <div class="row g-4">
        {% set articles_to_display = (articles[1:] if featured_article_on_this_page and articles else articles) %}
        {% for art in articles_to_display %}
        <div class="col-md-6 col-lg-4 d-flex">
        <article class="article-card animate-fade-in d-flex flex-column w-100" style="animation-delay: {{ loop.index0 * 0.05 }}s; position:relative;">
            {% set article_url = url_for('article_detail', article_hash_id=(art.article_hash_id if art.is_community_article else art.id)) %}
            <div class="article-image-container">
                <a href="{{ article_url }}">
                <img src="{{ art.image_url if art.is_community_article else art.urlToImage }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a>
                 {% if user_is_logged_in %}
                <button class="bookmark-btn bookmark-btn-dynamic" data-article-hash-id="{{ art.article_hash_id if art.is_community_article else art.id }}" title="Add Bookmark">
                    <i class="far fa-bookmark"></i>
                </button>
                {% endif %}
            </div>
            <div class="article-body d-flex flex-column">
                <h5 class="article-title mb-2"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                <div class="article-meta small mb-2">
                    <span class="meta-item text-muted">
                        <i class="fas fa-{{ 'user-edit' if art.is_community_article else 'building' }}"></i> 
                        {% if art.is_community_article and art.author %}
                            <a href="{{ url_for('user_profile', username=art.author.username) }}" class="text-muted text-decoration-none">{{ art.author.name|truncate(20) }}</a>
                        {% else %}
                            {{ art.source.name|truncate(20) }}
                        {% endif %}
                    </span>
                    <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ (art.published_at | to_ist if art.is_community_article else (art.publishedAt | to_ist if art.publishedAt else 'N/A')) }}</span>
                </div>
                {% if art.is_community_article and art.tags %}
                    <div class="article-tags">
                        {% for tag in art.tags | json_loads_safe | slice(0, 3) %} {# Show max 3 tags on card #}
                            <a href="{{ url_for('search_results', query=tag) }}" class="tag-badge">{{ tag }}</a>
                        {% endfor %}
                    </div>
                {% endif %}
                <p class="article-description small">{{ art.description|truncate(100) }}</p>
                <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
            </div>
        </article>
        </div>
        {% endfor %}
    </div>

    {% if total_pages and total_pages > 1 %}
    <nav aria-label="Page navigation" class="mt-5"><ul class="pagination justify-content-center">
        <li class="page-item page-link-prev-next {% if current_page == 1 %}disabled{% endif %}"><a class="page-link" href="{{ url_for(request.endpoint, page=current_page-1, category_name=selected_category if request.endpoint != 'search_results' and request.endpoint != 'bookmarks_page' else None, query=query if request.endpoint == 'search_results' else None, username=profile_user.username if request.endpoint == 'user_profile' else None ) if current_page > 1 else '#' }}">&laquo; Prev</a></li>
        {% set page_window = 1 %}{% set show_first = 1 %}{% set show_last = total_pages %}
        {% if current_page - page_window > show_first %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=1, category_name=selected_category if request.endpoint != 'search_results' and request.endpoint != 'bookmarks_page' else None, query=query if request.endpoint == 'search_results' else None, username=profile_user.username if request.endpoint == 'user_profile' else None) }}">1</a></li>{% if current_page - page_window > show_first + 1 %}<li class="page-item disabled"><span class="page-link">...</span></li>{% endif %}{% endif %}
        {% for p in range(1, total_pages + 1) %}{% if p == current_page %}<li class="page-item active" aria-current="page"><span class="page-link">{{ p }}</span></li>{% elif p >= current_page - page_window and p <= current_page + page_window %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=p, category_name=selected_category if request.endpoint != 'search_results' and request.endpoint != 'bookmarks_page' else None, query=query if request.endpoint == 'search_results' else None, username=profile_user.username if request.endpoint == 'user_profile' else None) }}">{{ p }}</a></li>{% endif %}{% endfor %}
        {% if current_page + page_window < show_last %}{% if current_page + page_window < show_last - 1 %}<li class="page-item disabled"><span class="page-link">...</span></li>{% endif %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=total_pages, category_name=selected_category if request.endpoint != 'search_results' and request.endpoint != 'bookmarks_page' else None, query=query if request.endpoint == 'search_results' else None, username=profile_user.username if request.endpoint == 'user_profile' else None) }}">{{ total_pages }}</a></li>{% endif %}
        <li class="page-item page-link-prev-next {% if current_page == total_pages %}disabled{% endif %}"><a class="page-link" href="{{ url_for(request.endpoint, page=current_page+1, category_name=selected_category if request.endpoint != 'search_results' and request.endpoint != 'bookmarks_page' else None, query=query if request.endpoint == 'search_results' else None, username=profile_user.username if request.endpoint == 'user_profile' else None) if current_page < total_pages else '#' }}">Next &raquo;</a></li>
    </ul></nav>
    {% endif %}
{% endblock %}
"""

ARTICLE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ article.title|truncate(50) if article else "Article" }} - Briefly{% endblock %}
{% block head_extra %}
<style>
    .article-full-content-wrapper { background-color: var(--white-bg); padding: 2rem; border-radius: 10px; box-shadow: 0 5px 20px rgba(0,0,0,0.07); margin-bottom: 2rem; margin-top: 1rem; }
    .article-full-content-wrapper .main-article-image { width: 100%; max-height: 480px; object-fit: cover; border-radius: 8px; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .article-title-main {font-weight: 700; color: var(--primary-color); line-height:1.3; font-family: 'Poppins', sans-serif;}
    body.dark-mode .article-title-main { color: var(--text-color); }
    .article-meta-detailed { font-size: 0.85rem; color: var(--text-muted-color); margin-bottom: 1.5rem; display:flex; flex-wrap:wrap; gap: 0.5rem 1.2rem; align-items:center; border-bottom: 1px solid var(--card-border-color); padding-bottom:1rem; }
    .article-meta-detailed .meta-item i { color: var(--secondary-color); margin-right: 0.4rem; font-size:0.95rem; }
    .summary-box { background-color: rgba(var(--primary-color-rgb), 0.04); padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; border: 1px solid rgba(var(--primary-color-rgb), 0.1); }
    .summary-box h5 { color: var(--primary-color); font-weight: 600; margin-bottom: 0.75rem; font-size:1.1rem; }
    body.dark-mode .summary-box h5 { color: var(--secondary-light); }
    .summary-box p {font-size:0.95rem; line-height:1.7; color: var(--text-color);}
    body.dark-mode .summary-box p { color: var(--text-muted-color); }
    .takeaways-box { margin: 1.5rem 0; padding: 1.5rem 1.5rem 1.5rem 1.8rem; border-left: 4px solid var(--secondary-color); background-color: rgba(var(--primary-color-rgb), 0.04); border-radius: 0 8px 8px 0;}
    .takeaways-box h5 { color: var(--primary-color); font-weight: 600; margin-bottom: 0.75rem; font-size:1.1rem; }
    body.dark-mode .takeaways-box h5 { color: var(--secondary-light); }
    .takeaways-box ul { padding-left: 1.2rem; margin-bottom:0; color: var(--text-color); }
    body.dark-mode .takeaways-box ul { color: var(--text-muted-color); }
    .takeaways-box ul li { margin-bottom: 0.6rem; font-size:0.95rem; line-height:1.6; }
    .loader-container { display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 200px; padding: 2rem; font-size: 1rem; color: var(--text-muted-color); }
    .loader { border: 5px solid var(--light-bg); border-top: 5px solid var(--primary-color); border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin-bottom: 1rem; }
    .content-text { white-space: pre-wrap; line-height: 1.8; font-size: 1.05rem; color: var(--text-color); } 
    body.dark-mode .content-text { color: var(--text-muted-color); } 
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .article-actions-bar { display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;}
</style>
{% endblock %}
{% block content %}
{% if not article %}
    <div class="alert alert-danger text-center my-5 p-4"><h4><i class="fas fa-exclamation-triangle me-2"></i>Article Not Found</h4><p>The article you are looking for could not be found.</p><a href="{{ url_for('index') }}" class="btn btn-primary mt-2">Go to Homepage</a></div>
{% else %}
<article class="article-full-content-wrapper animate-fade-in">
    <div class="article-actions-bar">
        <a href="{{ previous_list_page }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-arrow-left me-2"></i>Back to List</a>
        {% if user_is_logged_in %}
        <button class="bookmark-btn bookmark-btn-dynamic {% if is_bookmarked %}bookmarked{% endif %}" 
                data-article-hash-id="{{ article.article_hash_id }}" 
                title="{{ 'Remove Bookmark' if is_bookmarked else 'Add Bookmark' }}">
            <i class="{% if is_bookmarked %}fas{% else %}far{% endif %} fa-bookmark"></i>
        </button>
        {% endif %}
    </div>

    <h1 class="mb-2 article-title-main display-6">{{ article.title }}</h1>
    <div class="article-meta-detailed">
        <span class="meta-item" title="Source">
            <i class="fas fa-{{ 'user-edit' if is_community_article else 'building' }}"></i> 
            {% if is_community_article and article.author %}
                <a href="{{ url_for('user_profile', username=article.author.username) }}" class="text-decoration-none" style="color: inherit;">{{ article.author.name }}</a>
            {% else %}
                {{ article.source.name }}
            {% endif %}
        </span>
        <span class="meta-item" title="Published Date"><i class="far fa-calendar-alt"></i> {{ (article.published_at | to_ist if is_community_article else (article.publishedAt | to_ist if article.publishedAt else 'N/A')) }}</span>
    </div>

    {% if is_community_article and article.tags %}
        {% set parsed_tags = article.tags | json_loads_safe %}
        {% if parsed_tags %}
        <div class="article-tags mb-3">
            <strong>Tags:</strong> 
            {% for tag in parsed_tags %}
                <a href="{{ url_for('search_results', query=tag) }}" class="tag-badge">{{ tag }}</a>
            {% endfor %}
        </div>
        {% endif %}
    {% endif %}


    {% set image_to_display = article.image_url if is_community_article else article.urlToImage %}
    {% if image_to_display %}<img src="{{ image_to_display }}" alt="{{ article.title|truncate(50) }}" class="main-article-image">{% endif %}

    <div id="contentLoader" class="loader-container my-4 {% if is_community_article or (not is_community_article and article.groq_analysis) %}d-none{% endif %}"><div class="loader"></div><div>Analyzing article and generating summary...</div></div>
    
    <div id="articleAnalysisContainer">
    {% if is_community_article %}
        {% if article.groq_summary %}
            <div class="summary-box my-3"><h5><i class="fas fa-bookmark me-2"></i>Article Summary (AI Enhanced)</h5><p class="mb-0">{{ article.groq_summary|replace('\\n', '<br>')|safe }}</p></div>
        {% elif article.groq_summary is none and not (article.groq_takeaways | json_loads_safe) %} {# Show pending only if both are missing #}
             <div class="alert alert-info small p-3 mt-3"><i class="fas fa-hourglass-half me-2"></i>AI analysis for this community article is currently in progress. Please check back shortly.</div>
        {% elif not article.groq_summary %}  {# Analysis attempted but summary is empty #}
             <div class="alert alert-secondary small p-3 mt-3">AI Summary not available for this community article.</div>
        {% endif %}

        {% set parsed_takeaways = article.groq_takeaways | json_loads_safe %}
        {% if parsed_takeaways %}
            <div class="takeaways-box my-3"><h5><i class="fas fa-list-check me-2"></i>Key Takeaways (AI Enhanced)</h5>
                <ul>{% for takeaway in parsed_takeaways %}<li>{{ takeaway }}</li>{% endfor %}</ul>
            </div>
        {% elif article.groq_summary is not none and not parsed_takeaways %} {# Summary exists but no takeaways #}
            <div class="alert alert-secondary small p-3 mt-3">AI Takeaways not available for this community article.</div>
        {% endif %}
        <hr class="my-4">
        <h4 class="mb-3">Full Article Content</h4>
        <div class="content-text">{{ article.full_text }}</div>
    {% else %} {# API Article #}
        {% if article.groq_analysis_error %}
             <div class="alert alert-warning small p-3 mt-3">Could not load full analysis: {{ article.groq_analysis_error }}</div>
        {% endif %}
        <div id="apiArticleContent">
            {% if article.groq_analysis and not article.groq_analysis.error %}
                {% if article.groq_analysis.groq_summary %}
                <div class="summary-box my-3"><h5><i class="fas fa-bookmark me-2"></i>Article Summary (AI Enhanced)</h5><p class="mb-0">{{ article.groq_analysis.groq_summary|replace('\\n', '<br>')|safe }}</p></div>
                {% endif %}
                {% if article.groq_analysis.groq_takeaways %}
                <div class="takeaways-box my-3"><h5><i class="fas fa-list-check me-2"></i>Key Takeaways (AI Enhanced)</h5>
                    <ul>{% for takeaway in article.groq_analysis.groq_takeaways %}<li>{{ takeaway }}</li>{% endfor %}</ul>
                </div>
                {% endif %}
            {% endif %}
        </div>
        {% if article.url %}
             <hr class="my-4"><a href="{{ article.url }}" class="btn btn-outline-primary mt-3 mb-3" target="_blank" rel="noopener noreferrer">Read Original at {{ article.source.name }} <i class="fas fa-external-link-alt ms-1"></i></a>
        {% endif %}
    {% endif %}
    </div>

    <section class="comment-section" id="comment-section">
        <h3 class="mb-4">Community Discussion (<span id="comment-count">{{ comments|length }}</span>)</h3>

        {% macro render_comment_with_replies(comment, comment_data, user_is_logged_in, article_hash_id_for_js) %}
            <div class="comment-container" id="comment-{{ comment.id }}">
                <div class="comment-card">
                    <div class="comment-avatar" title="{{ comment.author.name if comment.author else 'Unknown' }}">
                        {{ (comment.author.name[0]|upper if comment.author and comment.author.name else 'U') }}
                    </div>
                    <div class="comment-body">
                        <div class="comment-header">
                            <span class="comment-author">
                                {% if comment.author and comment.author.username %}
                                <a href="{{ url_for('user_profile', username=comment.author.username) }}">{{ comment.author.name }}</a>
                                {% else %}
                                {{ comment.author.name if comment.author else 'Anonymous' }}
                                {% endif %}
                            </span>
                            <span class="comment-date">{{ comment.timestamp | to_ist }}</span>
                        </div>
                        <p class="comment-content mb-2">{{ comment.content }}</p>
                        {% if user_is_logged_in %}
                        <div class="comment-actions">
                            <button class="vote-btn {% if comment_data.get(comment.id, {}).get('user_vote') == 1 %}user-liked{% endif %}" data-comment-id="{{ comment.id }}" data-vote-type="1" title="Like">
                                <i class="fas fa-thumbs-up"></i>
                                <span class="vote-count" id="likes-count-{{ comment.id }}">{{ comment_data.get(comment.id, {}).get('likes', 0) }}</span>
                            </button>
                            <button class="vote-btn {% if comment_data.get(comment.id, {}).get('user_vote') == -1 %}user-disliked{% endif %}" data-comment-id="{{ comment.id }}" data-vote-type="-1" title="Dislike">
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
                        {{ render_comment_with_replies(reply, comment_data, user_is_logged_in, article_hash_id_for_js) }}
                    {% endfor %}
                </div>
            </div>
        {% endmacro %}

        <div id="comments-list">
            {% for comment in comments %} 
                {{ render_comment_with_replies(comment, comment_data, user_is_logged_in, article.article_hash_id) }}
            {% else %}
                <p id="no-comments-msg">No comments yet. Be the first to share your thoughts!</p>
            {% endfor %}
        </div>

        {% if user_is_logged_in %}
            <div class="add-comment-form mt-4 pt-4 border-top">
                <h5 class="mb-3">Leave a Comment</h5>
                <form id="comment-form">
                    <input type="hidden" name="article_hash_id" value="{{ article.article_hash_id }}">
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
    {% if article %} 
    const isCommunityArticlePage = {{ is_community_article | tojson }};
    const articleHashIdGlobalPage = {{ article.article_hash_id | tojson }};
    const userIsLoggedInPage = {{ user_is_logged_in | tojson }};

    function convertUTCToIST(utcIsoString) {
        if (!utcIsoString) return "N/A";
        const date = new Date(utcIsoString);
        return new Intl.DateTimeFormat('en-IN', {
            year: 'numeric', month: 'short', day: 'numeric',
            hour: 'numeric', minute: '2-digit', hour12: true,
            timeZone: 'Asia/Kolkata', timeZoneName: 'short'
        }).format(date);
    }

    if (!isCommunityArticlePage && articleHashIdGlobalPage && document.getElementById('contentLoader')) {
        const contentLoaderEl = document.getElementById('contentLoader');
        const apiArticleContentEl = document.getElementById('apiArticleContent');
        
        // Only fetch if content isn't already rendered by Jinja (e.g. from MASTER_ARTICLE_STORE cache)
        // and if the loader is visible (meaning content is pending)
        if (contentLoaderEl && getComputedStyle(contentLoaderEl).display !== 'none' && apiArticleContentEl && !apiArticleContentEl.innerHTML.trim()) {
            fetch(`{{ url_for('get_article_content_json', article_hash_id='PLACEHOLDER') }}`.replace('PLACEHOLDER', articleHashIdGlobalPage))
            .then(response => {
                if (!response.ok) { throw new Error('Network response error: ' + response.statusText); }
                return response.json();
            })
            .then(data => {
                if(contentLoaderEl) contentLoaderEl.style.display = 'none';
                if (!apiArticleContentEl) return;

                if (data.error) {
                    apiArticleContentEl.innerHTML = `<div class="alert alert-warning small p-3 mt-3">Could not load full analysis: ${data.error}</div>`;
                    return;
                }

                let html = '';
                const analysis = data.groq_analysis;

                if (analysis && analysis.groq_summary) { // Removed !analysis.error here as error is top-level
                    html += `<div class="summary-box my-3"><h5><i class="fas fa-bookmark me-2"></i>Article Summary (AI Enhanced)</h5><p class="mb-0">${analysis.groq_summary.replace(/\\n/g, '<br>')}</p></div>`;
                }
                if (analysis && analysis.groq_takeaways && analysis.groq_takeaways.length > 0) {
                    html += `<div class="takeaways-box my-3"><h5><i class="fas fa-list-check me-2"></i>Key Takeaways (AI Enhanced)</h5><ul>${analysis.groq_takeaways.map(t => `<li>${t}</li>`).join('')}</ul></div>`;
                }
                
                if (html === '' && !data.error) { // No summary/takeaways but also no explicit error from analysis itself
                     html = `<div class="alert alert-secondary small p-3 mt-3">AI analysis is not available for this article. You can read the original article via the link below.</div>`;
                } else if (html === '' && data.error) { // No content and an error was returned
                     html = `<div class="alert alert-warning small p-3 mt-3">Could not load full analysis: ${data.error}</div>`;
                }
                apiArticleContentEl.innerHTML = html;
            })
            .catch(error => {
                if(contentLoaderEl) contentLoaderEl.innerHTML = '<div class="alert alert-danger">Failed to load article analysis. The source may be blocking requests or an unexpected error occurred.</div>';
                console.error("Error fetching article content:", error);
            });
        } else if (contentLoaderEl) {
             contentLoaderEl.style.display = 'none'; // Hide if already loaded by Jinja
        }
    }

    const commentSection = document.getElementById('comment-section');

    function createCommentHTML(comment, articleHashIdForJs) {
        const commentDate = convertUTCToIST(comment.timestamp);
        const authorName = comment.author && comment.author.name ? comment.author.name : 'Anonymous';
        const authorUsername = comment.author && comment.author.username ? comment.author.username : null;
        const userInitial = authorName[0].toUpperCase();

        let authorLinkHTML = authorName;
        if (authorUsername) {
            authorLinkHTML = `<a href="/profile/${authorUsername}">${authorName}</a>`;
        }

        let actionsHTML = '';
        if (userIsLoggedInPage) {
            actionsHTML = `
            <div class="comment-actions">
                <button class="vote-btn" data-comment-id="${comment.id}" data-vote-type="1" title="Like">
                    <i class="fas fa-thumbs-up"></i> <span class="vote-count" id="likes-count-${comment.id}">0</span>
                </button>
                <button class="vote-btn" data-comment-id="${comment.id}" data-vote-type="-1" title="Dislike">
                    <i class="fas fa-thumbs-down"></i> <span class="vote-count" id="dislikes-count-${comment.id}">0</span>
                </button>
                <button class="reply-btn" data-comment-id="${comment.id}" title="Reply">
                    <i class="fas fa-reply"></i> Reply
                </button>
            </div>
            <div class="reply-form-container" id="reply-form-container-${comment.id}">
                <form class="reply-form mt-2">
                    <input type="hidden" name="article_hash_id" value="${articleHashIdForJs}">
                    <input type="hidden" name="parent_id" value="${comment.id}">
                    <div class="mb-2"><textarea class="form-control form-control-sm" name="content" rows="2" placeholder="Write a reply..." required></textarea></div>
                    <button type="submit" class="btn btn-sm btn-primary-modal">Post Reply</button>
                    <button type="button" class="btn btn-sm btn-outline-secondary-modal cancel-reply-btn">Cancel</button>
                </form>
            </div>`;
        }

        return \`
        <div class="comment-container" id="comment-${comment.id}">
            <div class="comment-card">
                <div class="comment-avatar" title="${authorName}">${userInitial}</div>
                <div class="comment-body">
                    <div class="comment-header">
                        <span class="comment-author">${authorLinkHTML}</span>
                        <span class="comment-date">${commentDate}</span>
                    </div>
                    <p class="comment-content mb-2">${comment.content}</p>
                    ${actionsHTML}
                </div>
            </div>
            <div class="comment-replies" id="replies-of-${comment.id}"></div>
        </div>\`;
    }

    function handleCommentSubmit(form, articleHashId, parentId = null) {
        const content = form.querySelector('textarea[name="content"]').value;
        if (!content.trim()) return;

        const submitButton = form.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.innerHTML;
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Posting...';

        fetch(\`{{ url_for('add_comment', article_hash_id='PLACEHOLDER') }}\`.replace('PLACEHOLDER', articleHashId), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ content: content, parent_id: parentId })
        })
        .then(res => {
            if (!res.ok) {
                return res.json().then(err => { throw new Error(err.error || \`HTTP error! status: \${res.status}\`); });
            }
            return res.json();
        })
        .then(data => {
            if (data.success) {
                const newCommentHTML = createCommentHTML(data.comment, articleHashId);
                const tempDiv = document.createElement('div');
                tempDiv.innerHTML = newCommentHTML.trim();
                const newCommentNode = tempDiv.firstChild;

                if (parentId) {
                    document.getElementById(\`replies-of-\${parentId}\`).appendChild(newCommentNode);
                    form.closest('.reply-form-container').style.display = 'none';
                } else {
                    const list = document.getElementById('comments-list');
                    const noCommentsMsg = document.getElementById('no-comments-msg');
                    if (noCommentsMsg) noCommentsMsg.remove();
                    list.appendChild(newCommentNode); 

                    const countEl = document.getElementById('comment-count');
                    countEl.textContent = parseInt(countEl.textContent) + 1;
                }
                form.reset();
            } else {
                // alert('Error: ' + (data.error || 'Unknown error posting comment.'));
                const alertPlaceholder = document.getElementById('alert-placeholder');
                 if(alertPlaceholder) {
                    const alertDiv = document.createElement('div');
                    alertDiv.className = 'alert alert-danger alert-dismissible fade show alert-top';
                    alertDiv.role = 'alert';
                    alertDiv.innerHTML = \`<span>\${data.error || 'Could not post comment.'}</span><button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>\`;
                    alertPlaceholder.appendChild(alertDiv);
                    setTimeout(() => { bootstrap.Alert.getOrCreateInstance(alertDiv)?.close(); }, 5000);
                 }
            }
        })
        .catch(err => {
            console.error("Comment submission error:", err);
            // alert("Could not submit comment: " + err.message);
             const alertPlaceholder = document.getElementById('alert-placeholder');
             if(alertPlaceholder) {
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-danger alert-dismissible fade show alert-top';
                alertDiv.role = 'alert';
                alertDiv.innerHTML = \`<span>Could not submit comment: \${err.message}</span><button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>\`;
                alertPlaceholder.appendChild(alertDiv);
                setTimeout(() => { bootstrap.Alert.getOrCreateInstance(alertDiv)?.close(); }, 5000);
             }
        })
        .finally(() => {
            submitButton.disabled = false;
            submitButton.innerHTML = originalButtonText;
        });
    }

    const mainCommentForm = document.getElementById('comment-form');
    if (mainCommentForm) {
        mainCommentForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const articleHashIdFromForm = this.querySelector('input[name="article_hash_id"]').value;
            handleCommentSubmit(this, articleHashIdFromForm);
        });
    }

    if (commentSection) {
        commentSection.addEventListener('click', function(e) {
            const voteBtn = e.target.closest('.vote-btn');
            if (voteBtn && userIsLoggedInPage) {
                const commentId = voteBtn.dataset.commentId;
                const voteType = parseInt(voteBtn.dataset.voteType);

                fetch(\`{{ url_for('vote_comment', comment_id=0) }}\`.replace('0', commentId), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ vote_type: voteType })
                })
                .then(res => {
                    if (!res.ok) {
                        return res.json().then(err => { throw new Error(err.error || \`HTTP error! status: \${res.status}\`); });
                    }
                    return res.json();
                })
                .then(data => {
                    if(data.success) {
                        document.getElementById(\`likes-count-\${commentId}\`).textContent = data.likes;
                        document.getElementById(\`dislikes-count-\${commentId}\`).textContent = data.dislikes;

                        const currentLikeBtn = document.querySelector(\`.vote-btn[data-comment-id="\${commentId}"][data-vote-type="1"]\`);
                        const currentDislikeBtn = document.querySelector(\`.vote-btn[data-comment-id="\${commentId}"][data-vote-type="-1"]\`);
                        
                        currentLikeBtn.classList.remove('user-liked');
                        currentDislikeBtn.classList.remove('user-disliked');

                        if (data.user_vote === 1) {
                            currentLikeBtn.classList.add('user-liked');
                        } else if (data.user_vote === -1) {
                            currentDislikeBtn.classList.add('user-disliked');
                        }
                    } else {
                         // alert('Error voting: ' + (data.error || 'Unknown error.'));
                        const alertPlaceholder = document.getElementById('alert-placeholder');
                         if(alertPlaceholder) {
                            const alertDiv = document.createElement('div');
                            alertDiv.className = 'alert alert-danger alert-dismissible fade show alert-top';
                            alertDiv.role = 'alert';
                            alertDiv.innerHTML = \`<span>\${data.error || 'Could not process vote.'}</span><button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>\`;
                            alertPlaceholder.appendChild(alertDiv);
                            setTimeout(() => { bootstrap.Alert.getOrCreateInstance(alertDiv)?.close(); }, 5000);
                         }
                    }
                }).catch(err => {
                    console.error("Vote error:", err);
                    // alert("Could not process vote: " + err.message);
                });
            }

            const replyBtn = e.target.closest('.reply-btn');
            if (replyBtn && userIsLoggedInPage) {
                const commentId = replyBtn.dataset.commentId;
                const formContainer = document.getElementById(\`reply-form-container-\${commentId}\`);
                if (formContainer) {
                    const isDisplayed = formContainer.style.display === 'block';
                    document.querySelectorAll('.reply-form-container').forEach(fc => { // Close other open reply forms
                        if (fc.id !== \`reply-form-container-\${commentId}\`) {
                           fc.style.display = 'none'; 
                        }
                    });
                    formContainer.style.display = isDisplayed ? 'none' : 'block'; 
                    if(formContainer.style.display === 'block') {
                        formContainer.querySelector('textarea').focus();
                    }
                }
            }

            const cancelReplyBtn = e.target.closest('.cancel-reply-btn');
            if (cancelReplyBtn) {
                const formContainer = cancelReplyBtn.closest('.reply-form-container');
                formContainer.style.display = 'none';
                formContainer.querySelector('form').reset();
            }
        });

        commentSection.addEventListener('submit', function(e) {
            const replyForm = e.target.closest('.reply-form');
            if (replyForm) {
                e.preventDefault();
                const articleHashIdFromForm = replyForm.querySelector('input[name="article_hash_id"]').value;
                const parentId = replyForm.querySelector('input[name="parent_id"]').value;
                handleCommentSubmit(replyForm, articleHashIdFromForm, parentId);
            }
        });
    }
    {% endif %} 
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
    <form method="POST" action="{{ url_for('login', next=request.args.get('next')) }}">
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
    <p class="mt-3 text-center small">Don't have an account? <a href="{{ url_for('register', next=request.args.get('next')) }}" class="fw-medium">Register here</a></p>
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
            <input type="text" class="modal-form-control" id="name" name="name" value="{{ request.form.name }}" required placeholder="Enter your full name">
        </div>
        <div class="modal-form-group">
            <label for="username" class="form-label">Username</label>
            <input type="text" class="modal-form-control" id="username" name="username" value="{{ request.form.username }}" required placeholder="Choose a username (min 3 chars)">
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
<div class="static-content-wrapper animate-fade-in">
    <h1 class="mb-4">About Briefly</h1>
    <p class="lead">Briefly is your premier destination for the latest news from India and around the world, delivered in a concise and easy-to-digest format. We leverage the power of cutting-edge AI to summarize complex news articles into key takeaways, saving you time while keeping you informed.</p>

    <h2 class="mt-5 mb-3">Our Mission</h2>
    <p>In a world of information overload, our mission is to provide clarity and efficiency. We believe that everyone deserves access to accurate, unbiased news without spending hours sifting through lengthy articles. Briefly cuts through the noise, offering insightful summaries that matter.</p>

    <h2 class="mt-5 mb-3">Community Hub</h2>
    <p>Beyond AI-driven news, Briefly is a platform for discussion and community engagement. Our Community Hub allows users to post their own articles, share perspectives, and engage in meaningful conversations about the topics that shape our world. We are committed to fostering a respectful and intelligent environment for all our members. Articles posted to the Community Hub are also enhanced with AI-generated summaries, takeaways, and tags to enrich the content and improve discoverability.</p>

    <h2 class="mt-5 mb-3">Our Technology</h2>
    <p>We use state-of-the-art Natural Language Processing (NLP) models via Groq to analyze and summarize news content from various sources, as well as to generate tags for community-contributed articles. Our system is designed to identify the most crucial points of an article, presenting them as a quick summary and a list of key takeaways.</p>
    
    <h2 class="mt-5 mb-3">Features</h2>
    <ul>
        <li>AI-powered summaries and key takeaways for news articles.</li>
        <li>Community Hub for user-generated content, also featuring AI summaries, takeaways, and tags.</li>
        <li>User commenting system with replies and voting.</li>
        <li>Article bookmarking to save interesting reads for later.</li>
        <li>User profiles to showcase community contributions.</li>
        <li>Dark mode for comfortable reading.</li>
        <li>Search functionality across all news and community articles.</li>
    </ul>
</div>
{% endblock %}
"""

CONTACT_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Contact Us - Briefly{% endblock %}
{% block content %}
<div class="static-content-wrapper animate-fade-in">
    <h1 class="mb-4">Contact Us</h1>
    <p class="lead">We'd love to hear from you! Whether you have a question, feedback, or a news tip, feel free to reach out.</p>

    <div class="row mt-5">
        <div class="col-md-6">
            <h2 class="h4">General Inquiries</h2>
            <p>For general questions, feedback, or support, please email us at:</p>
            <p><i class="fas fa-envelope me-2"></i><a href="mailto:contact@brieflynews.example">contact@brieflynews.example</a></p> {# Placeholder email #}
        </div>
        <div class="col-md-6">
            <h2 class="h4">Partnerships & Media</h2>
            <p>For partnership opportunities or media inquiries, please contact:</p>
            <p><i class="fas fa-envelope me-2"></i><a href="mailto:partners@brieflynews.example">partners@brieflynews.example</a></p> {# Placeholder email #}
        </div>
    </div>

    <div class="mt-5">
        <h2 class="h4">Follow Us</h2>
        <p>Stay connected with us on social media:</p>
        <div class="social-links fs-4">
            <a href="#" title="Twitter"><i class="fab fa-twitter"></i></a>
            <a href="#" title="Facebook"><i class="fab fa-facebook-f"></i></a>
            <a href="#" title="LinkedIn"><i class="fab fa-linkedin-in"></i></a>
            <a href="#" title="Instagram"><i class="fab fa-instagram"></i></a>
        </div>
    </div>
</div>
{% endblock %}
"""

PRIVACY_POLICY_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Privacy Policy - Briefly{% endblock %}
{% block content %}
<div class="static-content-wrapper animate-fade-in">
    <h1 class="mb-4">Privacy Policy</h1>
    <p class="text-muted">Last updated: May 30, 2024</p>

    <p>Briefly ("we," "our," or "us") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you visit our website.</p>

    <h2 class="mt-5 mb-3">1. Information We Collect</h2>
    <p>We may collect personal information that you voluntarily provide to us when you register on the website, post articles or comments, subscribe to our newsletter, or bookmark articles. This information may include your name, username, email address, and content you generate (articles, comments).</p>
    <p>We also collect non-personal information such as browser type, operating system, and website usage data through cookies and similar technologies to improve our services. Our dark mode preference is stored using cookies and local storage.</p>


    <h2 class="mt-5 mb-3">2. How We Use Your Information</h2>
    <p>We use the information we collect to:</p>
    <ul>
        <li>Create and manage your account.</li>
        <li>Operate and maintain the website, including displaying your contributions.</li>
        <li>Personalize your experience (e.g., saved bookmarks).</li>
        <li>Send you newsletters or promotional materials, if you have opted in.</li>
        <li>Respond to your comments and inquiries.</li>
        <li>Improve our website and services.</li>
        <li>Monitor site usage and prevent abuse.</li>
    </ul>

    <h2 class="mt-5 mb-3">3. Disclosure of Your Information</h2>
    <p>We do not sell, trade, or otherwise transfer your personally identifiable information to outside parties without your consent, except as described herein. This does not include trusted third parties who assist us in operating our website (e.g., hosting providers, API services for AI analysis), so long as those parties agree to keep this information confidential and use it only for the purposes we specify.</p>
    <p>Your username and any articles or comments you post will be publicly visible.</p>
    <p>We may also release your information when we believe release is appropriate to comply with the law, enforce our site policies, or protect ours or others' rights, property, or safety.</p>

    <h2 class="mt-5 mb-3">4. Third-Party Services</h2>
     <p>We use NewsAPI for fetching news articles and Groq for AI-powered analysis. These services have their own privacy policies, and we encourage you to review them.</p>
     <p>We use ScraperAPI for fetching full article content for analysis. This service also has its own privacy policy.</p>


    <h2 class="mt-5 mb-3">5. Security of Your Information</h2>
    <p>We use administrative, technical, and physical security measures to help protect your personal information. While we have taken reasonable steps to secure the personal information you provide to us, please be aware that despite our efforts, no security measures are perfect or impenetrable, and no method of data transmission can be guaranteed against any interception or other type of misuse.</p>

    <h2 class="mt-5 mb-3">6. Your Data Rights</h2>
    <p>Depending on your location, you may have rights regarding your personal data, such as the right to access, correct, or delete your personal information. Please contact us to make such requests.</p>
    
    <h2 class="mt-5 mb-3">7. Cookies and Local Storage</h2>
    <p>We use cookies for session management, remembering preferences like dark mode, and analytics. You can control the use of cookies at the individual browser level.</p>


    <h2 class="mt-5 mb-3">8. Changes to This Privacy Policy</h2>
    <p>We may update this Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page and updating the "Last updated" date. You are advised to review this Privacy Policy periodically for any changes.</p>
</div>
{% endblock %}
"""

# --- NEW BOOKMARKS TEMPLATE ---
BOOKMARKS_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}My Bookmarks - Briefly{% endblock %}
{% block content %}
<div class="container mt-4">
    <h1 class="mb-4">My Bookmarks</h1>
    {% if not articles %}
        <div class="alert alert-info text-center p-4">
            <h4><i class="far fa-bookmark me-2"></i>No Bookmarks Yet</h4>
            <p>You haven't bookmarked any articles. Start exploring and save your favorite reads!</p>
            <a href="{{ url_for('index') }}" class="btn btn-primary-modal mt-2">Browse Articles</a>
        </div>
    {% else %}
        <div class="row g-4">
            {% for art in articles %}
            <div class="col-md-6 col-lg-4 d-flex">
            <article class="article-card animate-fade-in d-flex flex-column w-100" style="position:relative;">
                {% set article_url = url_for('article_detail', article_hash_id=(art.article_hash_id if art.is_community_article else art.id)) %}
                <div class="article-image-container">
                    <a href="{{ article_url }}">
                    <img src="{{ art.image_url if art.is_community_article else art.urlToImage }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a>
                    {% if user_is_logged_in %} {# Always true on this page, but good practice #}
                    <button class="bookmark-btn bookmark-btn-dynamic bookmarked" data-article-hash-id="{{ art.article_hash_id if art.is_community_article else art.id }}" title="Remove Bookmark">
                        <i class="fas fa-bookmark"></i> {# Bookmarked by default on this page #}
                    </button>
                    {% endif %}
                </div>
                <div class="article-body d-flex flex-column">
                    <h5 class="article-title mb-2"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                    <div class="article-meta small mb-2">
                         <span class="meta-item text-muted">
                            <i class="fas fa-{{ 'user-edit' if art.is_community_article else 'building' }}"></i> 
                            {% if art.is_community_article and art.author %}
                                <a href="{{ url_for('user_profile', username=art.author.username) }}" class="text-muted text-decoration-none">{{ art.author.name|truncate(20) }}</a>
                            {% else %}
                                {{ art.source.name|truncate(20) }}
                            {% endif %}
                        </span>
                        <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ (art.published_at | to_ist if art.is_community_article else (art.publishedAt | to_ist if art.publishedAt else 'N/A')) }}</span>
                    </div>
                     {% if art.is_community_article and art.tags %}
                        <div class="article-tags">
                             {% for tag in art.tags | json_loads_safe | slice(0,3) %}
                                <a href="{{ url_for('search_results', query=tag) }}" class="tag-badge">{{ tag }}</a>
                             {% endfor %}
                        </div>
                    {% endif %}
                    <p class="article-description small text-muted">Bookmarked: {{ art.bookmarked_at | to_ist if art.bookmarked_at else 'N/A' }}</p>
                    <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
                </div>
            </article>
            </div>
            {% endfor %}
        </div>

        {% if total_pages and total_pages > 1 %}
        <nav aria-label="Page navigation" class="mt-5"><ul class="pagination justify-content-center">
            <li class="page-item page-link-prev-next {% if current_page == 1 %}disabled{% endif %}"><a class="page-link" href="{{ url_for('bookmarks_page', page=current_page-1) if current_page > 1 else '#' }}">&laquo; Prev</a></li>
            {% for p in range(1, total_pages + 1) %}<li class="page-item {% if p == current_page %}active{% endif %}"><a class="page-link" href="{{ url_for('bookmarks_page', page=p) }}">{{ p }}</a></li>{% endfor %}
            <li class="page-item page-link-prev-next {% if current_page == total_pages %}disabled{% endif %}"><a class="page-link" href="{{ url_for('bookmarks_page', page=current_page+1) if current_page < total_pages else '#' }}">Next &raquo;</a></li>
        </ul></nav>
        {% endif %}
    {% endif %}
</div>
{% endblock %}
"""

# --- NEW USER PROFILE TEMPLATE ---
PROFILE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ profile_user.name }}'s Profile - Briefly{% endblock %}
{% block content %}
<div class="container mt-4">
    <header class="profile-header article-card mb-4 animate-fade-in">
        <div class="profile-avatar">{{ profile_user.name[0]|upper }}</div>
        <h1 class="profile-username">{{ profile_user.name }}</h1>
        <p class="profile-name">@{{ profile_user.username }}</p>
        <p class="profile-joined-date">Joined: {{ profile_user.created_at | to_ist if profile_user.created_at else "Not available" }}</p>
        {# Add more profile info here if desired, e.g., bio, number of posts #}
    </header>

    <h2 class="mb-4">Articles by {{ profile_user.name }}</h2>
    {% if not articles and pagination.page == 1 %}
         <div class="alert alert-light profile-no-articles">
            <p><i class="fas fa-feather-alt me-2"></i>{{ profile_user.name }} hasn't posted any articles yet.</p>
            {% if session.user_id == profile_user.id %}
            <p>Why not share your first story? Click the '+' button to get started!</p>
            {% endif %}
        </div>
    {% elif not articles and pagination.page > 1 %}
        <div class="alert alert-light profile-no-articles">
            <p>No more articles to display.</p>
        </div>
    {% else %}
        <div class="row g-4">
            {% for art in articles %}
            <div class="col-md-6 col-lg-4 d-flex">
            <article class="article-card animate-fade-in d-flex flex-column w-100" style="position:relative;">
                {% set article_url = url_for('article_detail', article_hash_id=art.article_hash_id) %}
                <div class="article-image-container">
                    <a href="{{ article_url }}">
                    <img src="{{ art.image_url }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a>
                    {% if user_is_logged_in %}
                    <button class="bookmark-btn bookmark-btn-dynamic" data-article-hash-id="{{ art.article_hash_id }}" title="Add Bookmark">
                        <i class="far fa-bookmark"></i>
                    </button>
                    {% endif %}
                </div>
                <div class="article-body d-flex flex-column">
                    <h5 class="article-title mb-2"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                    <div class="article-meta small mb-2">
                        <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ art.published_at | to_ist }}</span>
                    </div>
                    {% if art.tags %}
                        <div class="article-tags">
                             {% for tag in art.tags | json_loads_safe | slice(0,3) %}
                                <a href="{{ url_for('search_results', query=tag) }}" class="tag-badge">{{ tag }}</a>
                             {% endfor %}
                        </div>
                    {% endif %}
                    <p class="article-description small">{{ art.description|truncate(100) }}</p>
                    <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
                </div>
            </article>
            </div>
            {% endfor %}
        </div>

        {% if pagination and pagination.pages > 1 %}
        <nav aria-label="Page navigation" class="mt-5"><ul class="pagination justify-content-center">
            <li class="page-item page-link-prev-next {% if not pagination.has_prev %}disabled{% endif %}"><a class="page-link" href="{{ url_for('user_profile', username=profile_user.username, page=pagination.prev_num) if pagination.has_prev else '#' }}">&laquo; Prev</a></li>
            {% for p in pagination.iter_pages(left_edge=1, right_edge=1, left_current=1, right_current=2) %}
                {% if p %}
                    {% if p == pagination.page %}<li class="page-item active" aria-current="page"><span class="page-link">{{ p }}</span></li>
                    {% else %}<li class="page-item"><a class="page-link" href="{{ url_for('user_profile', username=profile_user.username, page=p) }}">{{ p }}</a></li>{% endif %}
                {% else %}<li class="page-item disabled"><span class="page-link">...</span></li>{% endif %}
            {% endfor %}
            <li class="page-item page-link-prev-next {% if not pagination.has_next %}disabled{% endif %}"><a class="page-link" href="{{ url_for('user_profile', username=profile_user.username, page=pagination.next_num) if pagination.has_next else '#' }}">Next &raquo;</a></li>
        </ul></nav>
        {% endif %}
    {% endif %}
</div>
{% endblock %}
"""


ERROR_404_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}404 Not Found{% endblock %}{% block content %}<div class='text-center my-5 p-4 article-card animate-fade-in mx-auto' style='max-width: 600px;'><h1><i class='fas fa-exclamation-triangle text-warning me-2'></i>404 - Page Not Found</h1><p class='lead'>Sorry, the page you are looking for does not exist or has been moved.</p><a href='{{url_for("index")}}' class='btn btn-primary-modal mt-2'>Go to Homepage</a></div>{% endblock %}"""
ERROR_500_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}500 Server Error{% endblock %}{% block content %}<div class='text-center my-5 p-4 article-card animate-fade-in mx-auto' style='max-width: 600px;'><h1><i class='fas fa-cogs text-danger me-2'></i>500 - Internal Server Error</h1><p class='lead'>Something went wrong on our end. We've been notified and are looking into it.</p><a href='{{url_for("index")}}' class='btn btn-primary-modal mt-2'>Go to Homepage</a></div>{% endblock %}"""


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
template_storage['BOOKMARKS_HTML_TEMPLATE'] = BOOKMARKS_HTML_TEMPLATE
template_storage['PROFILE_HTML_TEMPLATE'] = PROFILE_HTML_TEMPLATE
template_storage['404_TEMPLATE'] = ERROR_404_TEMPLATE
template_storage['500_TEMPLATE'] = ERROR_500_TEMPLATE

# ==============================================================================
# --- 9. App Context & Main Execution Block ---
# ==============================================================================
# IMPORTANT: Database Initialization for Render
# This block ensures that init_db() (which calls db.create_all()) is executed
# when the application module is imported by Gunicorn or any other WSGI server.
# This is crucial for creating your database tables if they don't exist.

with app.app_context():
    app.logger.info("Application context pushed for module-level initialization.")
    init_db() # This will attempt to create all defined tables.

if __name__ == '__main__':
    # init_db() is already called above when the module loads,
    # so it's not strictly necessary to call it again here for __main__ execution.
    # However, having it here for direct script execution doesn't harm if it's idempotent.
    # with app.app_context():
    #     init_db()

    port = int(os.environ.get("PORT", 8080))
    # When running locally with `python Rev14.py`, FLASK_DEBUG is more relevant.
    # Gunicorn will typically manage debug mode through its own flags or environment variables.
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')

    app.logger.info(f"Starting Flask app directly (not via Gunicorn) in {'debug' if debug_mode else 'production-like'} mode on port {port}")
    app.logger.info("For production on Render, Gunicorn is typically used as the WSGI server.")
    app.logger.info("Ensure Celery worker is running separately for background tasks: celery -A Rev14.celery worker -l info")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
