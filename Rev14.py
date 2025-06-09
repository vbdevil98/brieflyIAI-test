#Perfectly Working Code

# Rev14.py - MODIFIED FOR ROBUST NEWS, COMMENTS, PROFILE, BOOKMARKS, AND AI DISPLAY FIXES

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
app.config['CATEGORIES'] = ['All Articles', 'Community Hub']

app.config['NEWS_API_QUERY'] = 'India OR "Indian politics" OR "Indian economy" OR "Bollywood"'
app.config['NEWS_API_DOMAINS'] = 'timesofindia.indiatimes.com,thehindu.com,ndtv.com,indianexpress.com,hindustantimes.com'
app.config['NEWS_API_DAYS_AGO'] = 7 # Fetch news from the last 7 days
app.config['NEWS_API_PAGE_SIZE'] = 100
app.config['NEWS_API_SORT_BY'] = 'publishedAt' #relevance, popularity, publishedAt
app.config['CACHE_EXPIRY_SECONDS'] = 7200 # 1 hour
app.permanent_session_lifetime = timedelta(days=30)

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

# Data Persistence
using_postgres_flag = False

# --- MODIFIED Data Persistence ---
app.logger.info(f"--- Database Configuration ---")
app.logger.info(f"Attempting to read DATABASE_URL environment variable...")
database_url = os.environ.get('DATABASE_URL') # Using 'database_url' for the value from environment

if database_url: 
    app.logger.info(f"DATABASE_URL found. Raw value (prefix): '{database_url[:30]}...'") # Log prefix for security
else:
    app.logger.info("DATABASE_URL environment variable NOT FOUND or is empty.")

configured_db_uri = None
# using_postgres_flag is already initialized to False above

if database_url and (database_url.startswith("postgres://") or database_url.startswith("postgresql://")):
    app.logger.info(f"DATABASE_URL indicates a PostgreSQL connection.")
    if database_url.startswith("postgres://"):
        configured_db_uri = database_url.replace("postgres://", "postgresql://", 1)
        app.logger.info(f"Converted DATABASE_URL from 'postgres://' to 'postgresql://'.")
    else: # Already starts with "postgresql://"
        configured_db_uri = database_url
        app.logger.info(f"DATABASE_URL already uses 'postgresql://' scheme.")
    
    app.config['SQLALCHEMY_DATABASE_URI'] = configured_db_uri
    uri_to_log = configured_db_uri
    try:
        parsed_uri = urllib.parse.urlparse(configured_db_uri)
        if parsed_uri.username or parsed_uri.password: # Mask credentials
            # Ensure port is handled correctly if present
            host_port = parsed_uri.hostname
            if parsed_uri.port:
                host_port += f":{parsed_uri.port}"
            uri_to_log = f"{parsed_uri.scheme}://********:********@{host_port}{parsed_uri.path}"
    except Exception:
        pass # Keep original if parsing fails
    app.logger.info(f"SQLAlchemy URI configured for PostgreSQL: {uri_to_log}")
    using_postgres_flag = True # Set flag to True
else:
    # This block is entered if database_url (from env) is None, empty, or not a valid Postgres URL format
    if database_url: # It existed but wasn't a postgres URL
        app.logger.warning(f"DATABASE_URL found ('{database_url[:30]}...') but it does not seem to be a PostgreSQL URL (expected 'postgres://' or 'postgresql://').")
    
    app.logger.info("Falling back to local SQLite database.")
    
    db_file_name = 'app_data.db'
    # Ensure project_root_for_db is correctly determined
    project_root_for_db = ""
    try:
        project_root_for_db = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(project_root_for_db, db_file_name)
    except NameError: # __file__ might not be defined in some execution contexts (e.g. interactive)
        app.logger.warning("__file__ not defined when setting SQLite path, using relative path for DB.")
        db_path = db_file_name # Fallback to relative path if absolute path fails
        
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.logger.info(f"SQLAlchemy URI configured for SQLite: {app.config['SQLALCHEMY_DATABASE_URI']}")
    # using_postgres_flag remains False (as initialized)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app) 
app.logger.info(f"SQLAlchemy instance created.")
app.logger.info(f"--- End of Database Configuration ---")

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

ALLOWED_REACTIONS = {'useful': '👍', 'insightful': '💡', 'thinking': '🤔', 'outrage': '😠'}

class CommentVote(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete="CASCADE"), nullable=False)
    comment_id = db.Column(db.Integer, db.ForeignKey('comment.id', ondelete="CASCADE"), nullable=False)
    reaction_type = db.Column(db.String(20), nullable=False)
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
    with app.app_context():
        app.logger.info("Initializing database and creating tables...")
        try:
            db.create_all()
            app.logger.info("db.create_all() executed successfully.")
        except Exception as e:
            app.logger.error(f"Error during db.create_all(): {e}", exc_info=True)

# ==============================================================================
# --- 5. Helper Functions ---
# ==============================================================================
MASTER_ARTICLE_STORE, API_CACHE = {}, {}
INDIAN_TIMEZONE = pytz.timezone('Asia/Kolkata')

def generate_article_id(url_or_title): return hashlib.md5(url_or_title.encode('utf-8')).hexdigest()

def jinja_truncate_filter(s, length=120, killwords=False, end='...'):
    if not s or len(s) <= length: return s
    if killwords: return s[:length - len(end)] + end
    words = s.split()
    result_words, current_length = [], 0
    for word in words:
        if current_length + len(word) + (1 if result_words else 0) > length - len(end): break
        result_words.append(word)
        current_length += len(word) + (1 if len(result_words) > 1 else 0)
    return ' '.join(result_words) + end if result_words else s[:length - len(end)] + end
app.jinja_env.filters['truncate'] = jinja_truncate_filter

def to_ist_filter(utc_dt):
    if not utc_dt: return "N/A"
    try:
        if isinstance(utc_dt, str):
            utc_dt = datetime.fromisoformat(utc_dt.replace('Z', '+00:00'))
        if not isinstance(utc_dt, datetime): return "Invalid date object"
        if utc_dt.tzinfo is None: utc_dt = pytz.utc.localize(utc_dt)
        return utc_dt.astimezone(INDIAN_TIMEZONE).strftime('%b %d, %Y at %I:%M %p %Z')
    except (ValueError, TypeError): return "Invalid date"
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
            flash("You must be logged in to access this page.", "warning")
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@simple_cache(expiry_seconds_default=3600 * 12)
def get_article_analysis_with_groq(article_text, article_title=""):
    if not groq_client or not article_text or not article_text.strip():
        return {"error": "AI analysis service not available or no text provided."}
    app.logger.info(f"Requesting Groq analysis for: {article_title[:50]}...")
    system_prompt = "You are an expert news analyst. Analyze the article. Provide a concise, neutral summary (3-4 paragraphs) and list 5-7 key takeaways as complete sentences. Format your entire response as a single JSON object with keys 'summary' (string) and 'takeaways' (a list of strings)."
    human_prompt = f"Article Title: {article_title}\n\nArticle Text:\n{article_text[:20000]}"
    try:
        json_model = groq_client.bind(response_format={"type": "json_object"})
        ai_response = json_model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        analysis = json.loads(ai_response.content)
        return {"groq_summary": analysis.get("summary"), "groq_takeaways": analysis.get("takeaways"), "error": None}
    except Exception as e:
        app.logger.error(f"Error during Groq analysis for '{article_title[:50]}': {e}", exc_info=True)
        return {"error": f"An unexpected error occurred during AI analysis: {e}"}

# ==============================================================================
# --- NEWS FETCHING ---
# ==============================================================================
@simple_cache()
def fetch_news_from_api(target_date_str=None):
    if not newsapi: return []
    from_date, to_date = None, None
    if target_date_str:
        try:
            target_dt_naive = datetime.strptime(target_date_str, '%Y-%m-%d')
            ist_start = INDIAN_TIMEZONE.localize(target_dt_naive)
            ist_end = ist_start.replace(hour=23, minute=59, second=59)
            from_date = ist_start.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
            to_date = ist_end.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
            app.logger.info(f"Fetching news for IST date {target_date_str} (UTC range: {from_date} to {to_date})")
        except ValueError:
            target_date_str = None
    if not target_date_str:
        now_utc = datetime.now(timezone.utc)
        from_date = (now_utc - timedelta(days=app.config['NEWS_API_DAYS_AGO'])).strftime('%Y-%m-%dT%H:%M:%S')
        to_date = now_utc.strftime('%Y-%m-%dT%H:%M:%S')
        app.logger.info(f"Fetching news for last {app.config['NEWS_API_DAYS_AGO']} days (UTC range: {from_date} to {to_date})")

    all_raw_articles = []
    try:
        everything_response = newsapi.get_everything(
            q=app.config['NEWS_API_QUERY'], from_param=from_date, to=to_date, language='en',
            sort_by=app.config['NEWS_API_SORT_BY'], page_size=app.config['NEWS_API_PAGE_SIZE'])
        if everything_response['status'] == 'ok': all_raw_articles.extend(everything_response['articles'])
    except NewsAPIException as e: app.logger.error(f"NewsAPI 'everything' error: {e}")
    
    # Fallback/augment with domains if needed
    if not all_raw_articles:
        try:
            domains_response = newsapi.get_everything(
                domains=app.config['NEWS_API_DOMAINS'], from_param=from_date, to=to_date, language='en',
                sort_by=app.config['NEWS_API_SORT_BY'], page_size=app.config['NEWS_API_PAGE_SIZE'])
            if domains_response['status'] == 'ok': all_raw_articles.extend(domains_response['articles'])
        except NewsAPIException as e: app.logger.error(f"NewsAPI 'domains' error: {e}")

    processed_articles, unique_urls = [], set()
    for art_data in all_raw_articles:
        url = art_data.get('url')
        title = art_data.get('title')
        if not all([url, title]) or url in unique_urls or title == '[Removed]' or not art_data.get('description'): continue
        unique_urls.add(url)
        article_id = generate_article_id(url)
        published_at_dt_iso = art_data.get('publishedAt', datetime.now(timezone.utc).isoformat())
        standardized_article = {
            'id': article_id, 'title': title, 'description': art_data['description'],
            'url': url, 'urlToImage': art_data.get('urlToImage') or f"https://via.placeholder.com/400x220/0D2C54/FFFFFF?text={urllib.parse.quote_plus(art_data['source'].get('name',''))}",
            'publishedAt': published_at_dt_iso, 'source': {'name': art_data['source'].get('name', 'Unknown')}, 'is_community_article': False
        }
        MASTER_ARTICLE_STORE[article_id] = standardized_article
        processed_articles.append(standardized_article)
    
    processed_articles.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
    app.logger.info(f"Fetched and processed {len(processed_articles)} unique articles.")
    return processed_articles

@simple_cache(expiry_seconds_default=3600 * 6)
def fetch_and_parse_article_content(article_hash_id, url):
    if not SCRAPER_API_KEY: return {"full_text": None, "groq_analysis": None, "error": "Content fetching service unavailable."}
    app.logger.info(f"Fetching content for API article ID: {article_hash_id}, URL: {url}")
    try:
        response = requests.get('http://api.scraperapi.com', params={'api_key': SCRAPER_API_KEY, 'url': url}, timeout=45)
        response.raise_for_status()
        article_scraper = Article(url, config=Config())
        article_scraper.download(input_html=response.text)
        article_scraper.parse()
        if not article_scraper.text: return {"full_text": None, "groq_analysis": None, "error": "Could not extract text."}
        
        article_title_for_groq = article_scraper.title or MASTER_ARTICLE_STORE.get(article_hash_id, {}).get('title', '')
        groq_analysis_result = get_article_analysis_with_groq(article_scraper.text, article_title_for_groq)
        
        if article_hash_id in MASTER_ARTICLE_STORE and groq_analysis_result and not groq_analysis_result.get("error"):
            MASTER_ARTICLE_STORE[article_hash_id]['groq_summary'] = groq_analysis_result.get("groq_summary")
            MASTER_ARTICLE_STORE[article_hash_id]['groq_takeaways'] = groq_analysis_result.get("groq_takeaways")
        
        return {"full_text": article_scraper.text, "groq_analysis": groq_analysis_result, "error": None}
    except Exception as e:
        app.logger.error(f"Failed to fetch/parse article content for {url}: {e}", exc_info=True)
        return {"full_text": None, "groq_analysis": None, "error": f"Failed to process article: {e}"}

# ==============================================================================
# --- 6. Flask Routes ---
# ==============================================================================
@app.context_processor
def inject_global_vars():
    return {
        'categories': app.config['CATEGORIES'], 'current_year': datetime.utcnow().year,
        'session': session, 'request': request, 'groq_client': groq_client is not None,
        'ALLOWED_REACTIONS': ALLOWED_REACTIONS
    }

def get_paginated_articles(articles, page, per_page):
    total = len(articles)
    start = (page - 1) * per_page
    paginated_items = articles[start:start + per_page]
    total_pages = (total + per_page - 1) // per_page
    return paginated_items, total_pages

def get_sort_key(article):
    date_val = getattr(article, 'published_at', None) if not isinstance(article, dict) else article.get('publishedAt')
    if not date_val: return datetime.min.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(date_val).replace('Z', '+00:00'))
        return dt if dt.tzinfo else pytz.utc.localize(dt)
    except (ValueError, TypeError): return datetime.min.replace(tzinfo=timezone.utc)

@app.route('/')
@app.route('/page/<int:page>')
@app.route('/category/<category_name>')
@app.route('/category/<category_name>/page/<int:page>')
def index(page=1, category_name='All Articles'):
    # ... (This route is unchanged) ...
    session['previous_list_page'] = request.full_path
    per_page = app.config['PER_PAGE']
    all_display_articles_raw = []
    query_str = request.args.get('query')
    filter_date_str = request.args.get('filter_date')

    if filter_date_str:
        try:
            datetime.strptime(filter_date_str, '%Y-%m-%d')
        except ValueError:
            flash("Invalid date format for filter. Showing all latest articles instead.", "warning")
            filter_date_str = None

    if category_name == 'Community Hub':
        db_articles = CommunityArticle.query.options(joinedload(CommunityArticle.author)).order_by(CommunityArticle.published_at.desc()).all()
        for art in db_articles:
            art.is_community_article = True
            all_display_articles_raw.append(art)
    else:
        api_articles = fetch_news_from_api(target_date_str=filter_date_str)
        all_display_articles_raw.extend(api_articles)

    all_display_articles_raw.sort(key=get_sort_key, reverse=True)
    paginated_display_articles_raw, total_pages = get_paginated_articles(all_display_articles_raw, page, per_page)
    
    paginated_display_articles_with_bookmark_status = []
    user_bookmarks_hashes = set()
    if 'user_id' in session:
        user_bookmarks_hashes = {b.article_hash_id for b in BookmarkedArticle.query.filter_by(user_id=session['user_id']).all()}

    for art_item in paginated_display_articles_raw:
        art_copy = art_item if hasattr(art_item, 'is_community_article') else art_item.copy()
        hash_id = art_copy.article_hash_id if hasattr(art_copy, 'article_hash_id') else art_copy.get('id')
        art_copy.is_bookmarked = hash_id in user_bookmarks_hashes
        paginated_display_articles_with_bookmark_status.append(art_copy)
    
    featured_article_on_this_page = (page == 1 and category_name == 'All Articles' and not query_str and not filter_date_str and paginated_display_articles_with_bookmark_status)
    
    return render_template("INDEX_HTML_TEMPLATE",
                           articles=paginated_display_articles_with_bookmark_status,
                           selected_category=category_name,
                           current_page=page, total_pages=total_pages,
                           featured_article_on_this_page=featured_article_on_this_page,
                           current_filter_date=filter_date_str, query=query_str)


@app.route('/search')
@app.route('/search/page/<int:page>')
def search_results(page=1):
    # ... (This route is unchanged) ...
    session['previous_list_page'] = request.full_path
    query_str = request.args.get('query', '').strip()
    if not query_str: return redirect(url_for('index'))
    
    per_page = app.config['PER_PAGE']
    if not MASTER_ARTICLE_STORE: fetch_news_from_api()
    
    api_results = [art for art in MASTER_ARTICLE_STORE.values() if query_str.lower() in art.get('title', '').lower() or query_str.lower() in art.get('description', '').lower()]
    community_results_query = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter(db.or_(CommunityArticle.title.ilike(f'%{query_str}%'), CommunityArticle.description.ilike(f'%{query_str}%')))
    community_results = []
    for art in community_results_query.all():
        art.is_community_article = True
        community_results.append(art)
    
    all_search_results_raw = api_results + community_results
    all_search_results_raw.sort(key=get_sort_key, reverse=True)
    
    paginated_search_articles_raw, total_pages = get_paginated_articles(all_search_results_raw, page, per_page)
    
    paginated_search_articles_with_bookmark_status = []
    user_bookmarks_hashes = set()
    if 'user_id' in session:
        user_bookmarks_hashes = {b.article_hash_id for b in BookmarkedArticle.query.filter_by(user_id=session['user_id']).all()}
    
    for art_item in paginated_search_articles_raw:
        art_copy = art_item if hasattr(art_item, 'is_community_article') else art_item.copy()
        hash_id = art_copy.article_hash_id if hasattr(art_copy, 'article_hash_id') else art_copy.get('id')
        art_copy.is_bookmarked = hash_id in user_bookmarks_hashes
        paginated_search_articles_with_bookmark_status.append(art_copy)
        
    return render_template("INDEX_HTML_TEMPLATE",
                           articles=paginated_search_articles_with_bookmark_status,
                           selected_category=f"Search: {query_str}",
                           current_page=page, total_pages=total_pages,
                           featured_article_on_this_page=False,
                           query=query_str, current_filter_date=None)


@app.route('/article/<article_hash_id>')
def article_detail(article_hash_id):
    # --- MODIFIED FOR BUG FIXES & REACTION FEATURE ---
    session['previous_list_page'] = session.get('previous_list_page', url_for('index'))
    article_data, is_community_article = None, False
    
    article_db = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter_by(article_hash_id=article_hash_id).first()
    if article_db:
        article_data, is_community_article = article_db, True
        try:
            article_data.parsed_takeaways = json.loads(article_data.groq_takeaways) if article_data.groq_takeaways else []
        except json.JSONDecodeError:
            article_data.parsed_takeaways = []
    else:
        if not MASTER_ARTICLE_STORE: fetch_news_from_api()
        api_dict = MASTER_ARTICLE_STORE.get(article_hash_id)
        if api_dict:
            article_data, is_community_article = api_dict, False
        else:
            flash("Article not found.", "danger")
            return redirect(session['previous_list_page'])

    # Fetch comments
    if is_community_article:
        all_article_comments_list = Comment.query.options(joinedload(Comment.author), joinedload(Comment.replies).options(joinedload(Comment.author))).filter_by(community_article_id=article_data.id).order_by(Comment.timestamp.asc()).all()
    else:
        all_article_comments_list = Comment.query.options(joinedload(Comment.author), joinedload(Comment.replies).options(joinedload(Comment.author))).filter_by(api_article_hash_id=article_hash_id).order_by(Comment.timestamp.asc()).all()
    
    comments_for_template = [c for c in all_article_comments_list if c.parent_id is None]
    
    comment_data = {}
    try:
        if all_article_comments_list:
            all_comment_ids = [c.id for c in all_article_comments_list for c in [c] + (c.replies or [])]
            for c_id in all_comment_ids:
                comment_data[c_id] = {'reactions': {r: 0 for r in ALLOWED_REACTIONS.keys()}, 'user_reaction': None}

            reaction_counts_query = db.session.query(CommentVote.comment_id, CommentVote.reaction_type, func.count(CommentVote.id)).filter(CommentVote.comment_id.in_(all_comment_ids)).group_by(CommentVote.comment_id, CommentVote.reaction_type).all()
            for c_id, r_type, count in reaction_counts_query:
                if c_id in comment_data and r_type in comment_data[c_id]['reactions']:
                    comment_data[c_id]['reactions'][r_type] = count

            if 'user_id' in session:
                user_reactions_query = CommentVote.query.filter(CommentVote.comment_id.in_(all_comment_ids), CommentVote.user_id == session['user_id']).all()
                for reaction in user_reactions_query:
                    if reaction.comment_id in comment_data:
                        comment_data[reaction.comment_id]['user_reaction'] = reaction.reaction_type

    except ProgrammingError as e:
        db.session.rollback()
        app.logger.error(f"DATABASE SCHEMA ERROR: {e}. This likely means the 'comment_vote' table needs to be updated. Please reset your database or run migrations.", exc_info=True)
        flash("A database error occurred while loading comments. The database schema might be out of date.", "danger")
        # Allow page to render without comment data
        comment_data = {}
        
    is_bookmarked = False
    if 'user_id' in session:
        is_bookmarked = BookmarkedArticle.query.filter_by(user_id=session['user_id'], article_hash_id=article_hash_id).first() is not None

    if isinstance(article_data, dict): article_data['is_community_article'] = False
    elif article_data: article_data.is_community_article = True

    return render_template("ARTICLE_HTML_TEMPLATE", article=article_data, is_community_article=is_community_article, comments=comments_for_template, comment_data=comment_data, previous_list_page=session['previous_list_page'], is_bookmarked=is_bookmarked)


@app.route('/get_article_content/<article_hash_id>')
def get_article_content_json(article_hash_id):
    # ... (This route is unchanged) ...
    if not MASTER_ARTICLE_STORE: fetch_news_from_api()
    article_data = MASTER_ARTICLE_STORE.get(article_hash_id)
    if not article_data or 'url' not in article_data:
        return jsonify({"error": "Article data or URL not found"}), 404
    
    if article_data.get('groq_summary') is not None:
        return jsonify({"groq_analysis": {"groq_summary": article_data['groq_summary'], "groq_takeaways": article_data.get('groq_takeaways'), "error": None}, "error": None})
    
    processed_content = fetch_and_parse_article_content(article_hash_id, article_data['url'])
    return jsonify(processed_content)


@app.route('/add_comment/<article_hash_id>', methods=['POST'])
@login_required
def add_comment(article_hash_id):
    # --- MODIFIED FOR BUG FIXES ---
    content = request.json.get('content', '').strip()
    parent_id = request.json.get('parent_id')
    if not content: return jsonify({"error": "Comment cannot be empty."}), 400
    
    community_article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
    if community_article:
        new_comment = Comment(content=content, user_id=session['user_id'], community_article_id=community_article.id, parent_id=parent_id)
    elif article_hash_id in MASTER_ARTICLE_STORE:
        new_comment = Comment(content=content, user_id=session['user_id'], api_article_hash_id=article_hash_id, parent_id=parent_id)
    else:
        return jsonify({"error": "Article not found to comment on."}), 404
        
    try:
        db.session.add(new_comment)
        db.session.commit()
        db.session.refresh(new_comment)
        author_name = new_comment.author.name if new_comment.author else "Anonymous"
        return jsonify({
            "success": True, 
            "comment": {
                "id": new_comment.id, "content": new_comment.content, "timestamp": new_comment.timestamp.isoformat() + 'Z',
                "author": {"name": author_name}, "parent_id": new_comment.parent_id,
                "reactions": {reaction: 0 for reaction in ALLOWED_REACTIONS.keys()}, "user_reaction": None
            }
        }), 201
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error adding comment: {e}", exc_info=True)
        return jsonify({"error": "Could not post comment due to a server error."}), 500


@app.route('/vote_comment/<int:comment_id>', methods=['POST'])
@login_required
def vote_comment(comment_id):
    # --- This route is correct for the reaction feature ---
    comment = Comment.query.get_or_404(comment_id)
    reaction_type = request.json.get('reaction_type')
    if reaction_type not in ALLOWED_REACTIONS.keys():
        return jsonify({"error": "Invalid reaction type."}), 400

    existing_reaction = CommentVote.query.filter_by(user_id=session['user_id'], comment_id=comment_id).first()
    new_user_reaction_status = None

    if existing_reaction:
        if existing_reaction.reaction_type == reaction_type:
            db.session.delete(existing_reaction)
        else:
            existing_reaction.reaction_type = reaction_type
            new_user_reaction_status = reaction_type
    else:
        new_reaction = CommentVote(user_id=session['user_id'], comment_id=comment_id, reaction_type=reaction_type)
        db.session.add(new_reaction)
        new_user_reaction_status = reaction_type
    
    try:
        db.session.commit()
        counts_query = db.session.query(CommentVote.reaction_type, func.count(CommentVote.id)).filter_by(comment_id=comment_id).group_by(CommentVote.reaction_type).all()
        reaction_counts = {r: 0 for r in ALLOWED_REACTIONS.keys()}
        reaction_counts.update(dict(counts_query))
        return jsonify({"success": True, "reactions": reaction_counts, "user_reaction": new_user_reaction_status}), 200
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error voting on comment: {e}", exc_info=True)
        return jsonify({"error": "Could not process vote due to a server error."}), 500


@app.route('/post_article', methods=['POST'])
@login_required
def post_article():
    # ... (This route is unchanged) ...
    title, description, content, source_name, image_url = (request.form.get(k, '').strip() for k in ['title', 'description', 'content', 'sourceName', 'imageUrl'])
    if not all([title, description, content]):
        flash("Title, Description, and Full Content are required.", "danger")
        return redirect(request.referrer or url_for('index'))
    source_name = source_name or 'Community Post'
    article_hash_id = generate_article_id(title + str(session['user_id']) + str(time.time()))
    groq_analysis = get_article_analysis_with_groq(content, title)
    new_article = CommunityArticle(
        article_hash_id=article_hash_id, title=title, description=description, full_text=content,
        source_name=source_name, image_url=image_url or None, user_id=session['user_id'],
        groq_summary=groq_analysis.get('groq_summary'),
        groq_takeaways=json.dumps(groq_analysis.get('groq_takeaways')) if groq_analysis.get('groq_takeaways') else None
    )
    db.session.add(new_article)
    db.session.commit()
    flash("Your article has been posted!", "success")
    return redirect(url_for('article_detail', article_hash_id=new_article.article_hash_id))

@app.route('/register', methods=['GET', 'POST'])
def register():
    # ... (This route is unchanged) ...
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        name, username, password = request.form.get('name', '').strip(), request.form.get('username', '').strip().lower(), request.form.get('password', '')
        if not all([name, username, password]) or len(username) < 3 or len(password) < 6:
            flash('All fields are required. Username must be at least 3 characters, and password at least 6 characters.', 'danger')
        elif User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose another.', 'warning')
        else:
            new_user = User(name=name, username=username, password_hash=generate_password_hash(password))
            db.session.add(new_user); db.session.commit()
            flash(f'Registration successful, {name}! Please log in.', 'success')
            return redirect(url_for('login'))
    return render_template("REGISTER_HTML_TEMPLATE")


@app.route('/login', methods=['GET', 'POST'])
def login():
    # ... (This route is unchanged) ...
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        username, password = request.form.get('username', '').strip().lower(), request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session.permanent = True; session['user_id'] = user.id; session['user_name'] = user.name
            flash(f"Welcome back, {user.name}!", "success")
            return redirect(request.args.get('next') or url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template("LOGIN_HTML_TEMPLATE")

@app.route('/logout')
def logout(): session.clear(); flash("You have been successfully logged out.", "info"); return redirect(url_for('index'))
@app.route('/about')
def about(): return render_template("ABOUT_US_HTML_TEMPLATE")
@app.route('/contact')
def contact(): return render_template("CONTACT_HTML_TEMPLATE")
@app.route('/privacy')
def privacy(): return render_template("PRIVACY_POLICY_HTML_TEMPLATE")
@app.route('/ads.txt')
def ads_txt(): return Response("google.com, pub-6975904325280886, DIRECT, f08c47fec0942fa0", mimetype='text/plain')

@app.route('/subscribe', methods=['POST'])
def subscribe():
    # ... (This route is unchanged) ...
    email = request.form.get('email', '').strip().lower()
    if not email: flash('Email is required to subscribe.', 'warning')
    elif Subscriber.query.filter_by(email=email).first(): flash('You are already subscribed.', 'info')
    else:
        try:
            db.session.add(Subscriber(email=email)); db.session.commit()
            flash('Thank you for subscribing!', 'success')
        except Exception as e:
            db.session.rollback()
            app.logger.error(f"Error subscribing email {email}: {e}")
            flash('Could not subscribe at this time.', 'danger')
    return redirect(request.referrer or url_for('index'))

@app.route('/toggle_bookmark/<article_hash_id>', methods=['POST'])
@login_required
def toggle_bookmark(article_hash_id):
    # ... (This route is unchanged) ...
    user_id = session['user_id']
    existing_bookmark = BookmarkedArticle.query.filter_by(user_id=user_id, article_hash_id=article_hash_id).first()
    if existing_bookmark:
        db.session.delete(existing_bookmark)
        db.session.commit()
        return jsonify({"success": True, "status": "removed", "message": "Bookmark removed."})
    else:
        is_community = request.json.get('is_community_article', 'false').lower() == 'true'
        published_at_str = request.json.get('published_at')
        published_at_dt = None
        if published_at_str:
            try: published_at_dt = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
            except ValueError: pass
        new_bookmark = BookmarkedArticle(
            user_id=user_id, article_hash_id=article_hash_id, is_community_article=is_community,
            title_cache=request.json.get('title'), source_name_cache=request.json.get('source_name'),
            image_url_cache=request.json.get('image_url'), description_cache=request.json.get('description'),
            published_at_cache=published_at_dt
        )
        db.session.add(new_bookmark)
        db.session.commit()
        return jsonify({"success": True, "status": "added", "message": "Article bookmarked!"})

@app.route('/profile')
@login_required
def profile():
    # ... (This route is unchanged) ...
    user = User.query.get_or_404(session['user_id'])
    page = request.args.get('page', 1, type=int)
    per_page = app.config['PER_PAGE']
    user_posted_articles = CommunityArticle.query.filter_by(user_id=user.id).order_by(CommunityArticle.published_at.desc()).all()
    bookmarks_pagination = BookmarkedArticle.query.filter_by(user_id=user.id).order_by(BookmarkedArticle.bookmarked_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    
    user_bookmarked_articles_data = []
    for bookmark in bookmarks_pagination.items:
        article_detail_data = None
        if bookmark.is_community_article:
            comm_art = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter_by(article_hash_id=bookmark.article_hash_id).first()
            if comm_art:
                article_detail_data = {'id': comm_art.article_hash_id, 'title': comm_art.title, 'description': comm_art.description, 'urlToImage': comm_art.image_url, 'publishedAt': comm_art.published_at.isoformat(), 'source': {'name': comm_art.author.name if comm_art.author else 'Community'}, 'is_community_article': True, 'article_url': url_for('article_detail', article_hash_id=comm_art.article_hash_id)}
        else:
            api_art = MASTER_ARTICLE_STORE.get(bookmark.article_hash_id)
            if api_art:
                article_detail_data = {'id': api_art['id'], 'title': api_art['title'], 'description': api_art['description'], 'urlToImage': api_art['urlToImage'], 'publishedAt': api_art['publishedAt'], 'source': {'name': api_art['source']['name']}, 'is_community_article': False, 'article_url': url_for('article_detail', article_hash_id=api_art['id'])}
            else: # Stale bookmark
                article_detail_data = {'id': bookmark.article_hash_id, 'title': bookmark.title_cache, 'description': bookmark.description_cache, 'urlToImage': bookmark.image_url_cache, 'publishedAt': bookmark.published_at_cache.isoformat() if bookmark.published_at_cache else None, 'source': {'name': bookmark.source_name_cache}, 'is_community_article': False, 'article_url': url_for('article_detail', article_hash_id=bookmark.article_hash_id), 'is_stale_bookmark': True}
        if article_detail_data: user_bookmarked_articles_data.append(article_detail_data)
        
    return render_template("PROFILE_HTML_TEMPLATE", user=user, posted_articles=user_posted_articles, bookmarked_articles=user_bookmarked_articles_data, bookmarks_pagination=bookmarks_pagination)


@app.errorhandler(404)
def page_not_found(e): return render_template("404_TEMPLATE"), 404
@app.errorhandler(500)
def internal_server_error(e):
    db.session.rollback()
    app.logger.error(f"500 error at {request.url}: {e}", exc_info=True)
    return render_template("500_TEMPLATE"), 500

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
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #4F46E5; --primary-light: #6366F1; --primary-dark: #4338CA; --secondary-color: #14B8A6; --secondary-light: #2DD4BF; --accent-color: #F97316; --text-color: #1F2937; --text-muted-color: #6B7280; --light-bg: #F9FAFB; --card-bg: #FFFFFF; --card-border-color: #E5E7EB; --footer-bg: #111827; --footer-text: #D1D5DB; --footer-link-hover: var(--primary-light);
            --primary-color-rgb: 79, 70, 229; --secondary-color-rgb: 20, 184, 166;
            --bookmark-active-color: var(--secondary-color);
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05); --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1); --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
            --border-radius-sm: 0.375rem; --border-radius-md: 0.5rem; --border-radius-lg: 0.75rem;
        }
        body { padding-top: 145px; font-family: 'Inter', sans-serif; line-height: 1.65; color: var(--text-color); background-color: var(--light-bg); display: flex; flex-direction: column; min-height: 100vh; transition: background-color 0.3s ease, color 0.3s ease; }
        body.dark-mode {
            --primary-color: #6366F1; --primary-light: #818CF8; --primary-dark: #4F46E5; --secondary-color: #2DD4BF; --secondary-light: #5EEAD4; --accent-color: #FB923C; --text-color: #F9FAFB; --text-muted-color: #9CA3AF; --light-bg: #111827; --card-bg: #1F2937; --card-border-color: #374151; --footer-bg: #000000; --footer-text: #9CA3AF;
            --primary-color-rgb: 99, 102, 241; --secondary-color-rgb: 45, 212, 191;
            --bookmark-active-color: var(--secondary-light);
        }
        .navbar-main { background-color: var(--primary-color); padding: 0; box-shadow: var(--shadow-md); height: 95px; }
        .navbar-content-wrapper { position: relative; display: flex; justify-content: space-between; align-items: center; width: 100%; height: 100%; }
        .navbar-brand-custom { color: white !important; font-weight: 700; font-size: 2rem; font-family: 'Poppins', sans-serif; display: flex; align-items: center; gap: 10px; text-decoration: none !important; }
        .search-form-container { position: absolute; left: 50%; transform: translateX(-50%); width: 45%; max-width: 550px; }
        .navbar-search { border-radius: 50px; padding: 0.7rem 1.25rem 0.7rem 2.8rem; border: 1px solid transparent; font-size: 0.95rem; background: rgba(255,255,255,0.15); color: white; }
        .navbar-search::placeholder { color: rgba(255,255,255,0.7); }
        .navbar-search:focus { background: rgba(255,255,255,0.25); box-shadow: 0 0 0 4px rgba(255,255,255,0.2); border-color: var(--secondary-light); outline: none; }
        .header-btn { background: transparent; border: 1px solid rgba(255,255,255,0.4); padding: 0.5rem 1rem; border-radius: 50px; color: white; font-weight: 500; }
        .header-btn:hover { background: rgba(255,255,255,0.9); border-color: transparent; color: var(--primary-dark); }
        .category-nav { background: var(--card-bg); box-shadow: var(--shadow-sm); position: fixed; top: 95px; width: 100%; z-index: 1020; border-bottom: 1px solid var(--card-border-color); }
        .category-link { color: var(--text-muted-color) !important; font-weight: 600; padding: 0.6rem 1.3rem !important; border-radius: 50px; margin: 0 0.3rem; }
        .category-link.active { background: var(--primary-color) !important; color: white !important; }
        body.dark-mode .category-link.active { color: var(--card-bg) !important; }
        .article-card, .featured-article, .article-full-content-wrapper, .auth-container, .static-content-wrapper, .profile-card { background: var(--card-bg); border-radius: var(--border-radius-lg); border: 1px solid var(--card-border-color); box-shadow: var(--shadow-md); }
        .article-card:hover, .featured-article:hover { transform: translateY(-5px); box-shadow: var(--shadow-lg); }
        .comment-actions { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.75rem; }
        .reaction-btn { background-color: var(--light-bg); border: 1px solid var(--card-border-color); padding: 0.25rem 0.6rem; color: var(--text-muted-color); cursor: pointer; display: flex; align-items: center; gap: 0.4rem; transition: all 0.2s ease; border-radius: 20px; font-size: 0.9rem; }
        .reaction-btn .emoji { font-size: 1.2em; transition: transform 0.2s ease; }
        .reaction-btn:hover .emoji { transform: scale(1.2); }
        .reaction-btn .reaction-count { font-weight: 500; min-width: 12px; text-align: left; }
        .reaction-btn.active { border-color: var(--primary-color); background-color: rgba(var(--primary-color-rgb), 0.1); color: var(--primary-color); font-weight: 600; }
        .reaction-btn.active .emoji { transform: scale(1.1); }
        body.dark-mode .reaction-btn { background-color: var(--card-bg); border-color: #444; }
        body.dark-mode .reaction-btn:hover { border-color: var(--primary-light); color: var(--primary-light); }
        body.dark-mode .reaction-btn.active { border-color: var(--primary-light); background-color: rgba(var(--primary-color-rgb), 0.2); color: var(--primary-light); }
        .reply-btn { background: none; border: none; padding: 0.2rem 0.4rem; color: var(--text-muted-color); cursor: pointer; display: flex; align-items: center; gap: 0.3rem; transition: color 0.2s ease; border-radius: 4px; font-size: 0.9rem; margin-left: auto; }
        .reply-btn:hover { color: var(--primary-color); }
        body.dark-mode .reply-btn:hover { color: var(--primary-light); }
        @media (max-width: 991.98px) {
            body { padding-top: 180px; } .navbar-main { padding: 1rem 0 0.5rem; height: auto; } .navbar-content-wrapper { position: static; flex-direction: column; align-items: flex-start; gap: 0.75rem; height: auto; } .navbar-brand-custom { margin-bottom: 0.5rem; } .search-form-container { position: static; transform: none; width: 100%; order: 3; } .header-controls { position: absolute; top: 1.2rem; right: 1rem; } .category-nav { top: 130px; } .categories-wrapper { justify-content: flex-start; } #dateFilterForm { width: 100%; margin-left: 0 !important; margin-top: 0.5rem; }
        }
    </style>
    </head>
<body>
    </body>
</html>
"""

INDEX_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}
    {% if query %}Search: {{ query|truncate(30) }}
    {% elif selected_category == 'All Articles' and current_filter_date %}Articles for {{ current_filter_date }}
    {% elif selected_category %}{{selected_category}}
    {% else %}Home{% endif %} - Briefly (India News)
{% endblock %}
{% block content %}
    {% if selected_category == 'All Articles' and current_filter_date %}
        <h4 class="mb-3 fst-italic">Showing articles for: {{ current_filter_date }}</h4>
    {% endif %}
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
                    <div>
                        <div class="article-meta mb-2">
                            <span class="badge bg-primary me-2" style="font-size:0.75rem;">{{ (art0.author.name if art0.is_community_article and art0.author else art0.source.name)|truncate(25) }}</span>
                            <span class="meta-item"><i class="far fa-calendar-alt"></i> {{ (art0.published_at | to_ist if art0.is_community_article else (art0.publishedAt | to_ist if art0.publishedAt else 'N/A')) }}</span>
                        </div>
                    </div>
                    {% if session.user_id %}
                    <button class="bookmark-btn homepage-bookmark-btn {% if art0.is_bookmarked %}active{% endif %}"
                            title="{% if art0.is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}"
                            data-article-hash-id="{{ art0.article_hash_id if art0.is_community_article else art0.id }}"
                            data-is-community="{{ 'true' if art0.is_community_article else 'false' }}"
                            data-title="{{ art0.title|e }}"
                            data-source-name="{{ (art0.author.name if art0.is_community_article and art0.author else art0.source.name)|e }}"
                            data-image-url="{{ (art0.image_url if art0.is_community_article else art0.urlToImage)|e }}"
                            data-description="{{ (art0.description if art0.description else '')|e }}"
                            data-published-at="{{ (art0.published_at.isoformat() if art0.is_community_article and art0.published_at else (art0.publishedAt if not art0.is_community_article and art0.publishedAt else ''))|e }}">
                        <i class="fa-solid fa-bookmark"></i>
                    </button>
                    {% endif %}
                </div>
                <h2 class="mb-2 h4"><a href="{{ article_url }}" class="text-decoration-none article-title">{{ art0.title }}</a></h2>
                <p class="article-description flex-grow-1 small">{{ art0.description|truncate(220) }}</p>
                <a href="{{ article_url }}" class="read-more mt-auto align-self-start py-2 px-3" style="width:auto;">Read Full Article <i class="fas fa-arrow-right ms-1 small"></i></a>
            </div>
        </div>
    </article>
    {% elif not articles and (selected_category != 'Community Hub' or query) %}
        <div class="alert alert-info text-center my-5 p-4"><h4><i class="fas fa-search me-2"></i>
        {% if query %}No results for "{{ query }}"{% elif current_filter_date %}No articles found for {{ current_filter_date }}{% else %}No recent news found{% endif %}
        </h4><p>Try different keywords or browse categories.</p></div>
    {% elif not articles and selected_category == 'Community Hub' %}
        <div class="alert alert-info text-center my-4 p-3"><h4><i class="fas fa-feather-alt me-2"></i>No Articles Here Yet</h4><p>Be the first to contribute to the Community Hub!</p></div>
    {% endif %}
    <div class="row g-4">
        {% set articles_to_display = (articles[1:] if featured_article_on_this_page and articles else articles) %}
        {% for art in articles_to_display %}
        <div class="col-md-6 col-lg-4 d-flex">
        <article class="article-card animate-fade-in d-flex flex-column w-100" style="animation-delay: {{ loop.index0 * 0.05 }}s">
            {% set article_url = url_for('article_detail', article_hash_id=(art.article_hash_id if art.is_community_article else art.id)) %}
            <div class="article-image-container">
                <a href="{{ article_url }}">
                <img src="{{ art.image_url if art.is_community_article else art.urlToImage }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a>
            </div>
            <div class="article-body d-flex flex-column">
                <div class="d-flex justify-content-between align-items-start">
                    <h5 class="article-title mb-2 flex-grow-1"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                    {% if session.user_id %}
                    <button class="bookmark-btn homepage-bookmark-btn {% if art.is_bookmarked %}active{% endif %}" style="margin-left: 10px; padding-top:0;"
                            data-article-hash-id="{{ art.article_hash_id if art.is_community_article else art.id }}"
                            data-is-community="{{ 'true' if art.is_community_article else 'false' }}"
                            data-title="{{ art.title|e }}"
                            data-source-name="{{ (art.author.name if art.is_community_article and art.author else art.source.name)|e }}"
                            data-image-url="{{ (art.image_url if art.is_community_article else art.urlToImage)|e }}"
                            data-description="{{ (art.description if art.description else '')|e }}"
                            data-published-at="{{ (art.published_at.isoformat() if art.is_community_article and art.published_at else (art.publishedAt if not art.is_community_article and art.publishedAt else ''))|e }}">
                        <i class="fa-solid fa-bookmark"></i>
                    </button>
                    {% endif %}
                </div>
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
        {% set nav_url_params = {'category_name': selected_category if request.endpoint != 'search_results' else None, 'query': query if request.endpoint == 'search_results' else None, 'filter_date': request.args.get('filter_date')} %}
        <li class="page-item page-link-prev-next {% if current_page == 1 %}disabled{% endif %}"><a class="page-link" href="{{ url_for(request.endpoint, page=current_page-1, **nav_url_params) if current_page > 1 else '#' }}">&laquo; Prev</a></li>
        {% set page_window = 1 %}
        {% for p in range(1, total_pages + 1) %}{% if p == 1 or p == total_pages or (p >= current_page - page_window and p <= current_page + page_window) %}
            {% if loop.previtem is defined and p > loop.previtem + 1 %}<li class="page-item disabled"><span class="page-link">...</span></li>{% endif %}
            <li class="page-item {% if p == current_page %}active{% endif %}"><a class="page-link" href="{{ url_for(request.endpoint, page=p, **nav_url_params) }}">{{ p }}</a></li>
        {% endif %}{% endfor %}
        <li class="page-item page-link-prev-next {% if current_page == total_pages %}disabled{% endif %}"><a class="page-link" href="{{ url_for(request.endpoint, page=current_page+1, **nav_url_params) if current_page < total_pages else '#' }}">Next &raquo;</a></li>
    </ul></nav>
    {% endif %}
{% endblock %}
{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    const isUserLoggedInForHomepage = {{ 'true' if session.user_id else 'false' }};
    document.querySelectorAll('.homepage-bookmark-btn').forEach(button => {
        if (!isUserLoggedInForHomepage) return;
        button.addEventListener('click', function(event) {
            event.preventDefault(); event.stopPropagation();
            const data = this.dataset;
            fetch(`{{ url_for('toggle_bookmark', article_hash_id='_') }}`.replace('_', data.articleHashId), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_community_article: data.isCommunity, title: data.title, source_name: data.sourceName, image_url: data.imageUrl, description: data.description, published_at: data.publishedAt })
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    this.classList.toggle('active', data.status === 'added');
                    // ... (Alert logic can be added here if desired)
                }
            }).catch(err => console.error("Bookmark error:", err));
        });
    });
});
</script>
{% endblock %}
"""

ARTICLE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ article.title|truncate(50) if article else "Article" }} - Briefly{% endblock %}
{% block head_extra %}
<style>
    .article-full-content-wrapper { background-color: var(--card-bg); padding: 2rem; border-radius: var(--border-radius-lg); box-shadow: var(--shadow-md); margin-bottom: 2rem; margin-top: 1rem; }
    .article-full-content-wrapper .main-article-image { width: 100%; max-height: 480px; object-fit: cover; border-radius: var(--border-radius-md); margin-bottom: 1.5rem; box-shadow: var(--shadow-md); }
    .article-title-main {font-weight: 700; color: var(--text-color); line-height:1.3; font-family: 'Poppins', sans-serif;}
    .article-meta-detailed { font-size: 0.85rem; color: var(--text-muted-color); margin-bottom: 1.5rem; display:flex; flex-wrap:wrap; gap: 0.5rem 1.2rem; align-items:center; border-bottom: 1px solid var(--card-border-color); padding-bottom:1rem; }
    .article-meta-detailed .meta-item i { color: var(--secondary-color); margin-right: 0.4rem; font-size:0.95rem; }
    .summary-box { background-color: rgba(var(--primary-color-rgb), 0.05); padding: 1.5rem; border-radius: var(--border-radius-md); margin: 1.5rem 0; border: 1px solid rgba(var(--primary-color-rgb), 0.1); }
    .summary-box h5 { color: var(--primary-color); font-weight: 600; margin-bottom: 0.75rem; font-size:1.1rem; }
    .takeaways-box { margin: 1.5rem 0; padding: 1.5rem 1.5rem 1.5rem 1.8rem; border-left: 4px solid var(--secondary-color); background-color: rgba(var(--secondary-color-rgb), 0.05); border-radius: 0 var(--border-radius-md) var(--border-radius-md) 0;}
    .takeaways-box h5 { color: var(--primary-color); font-weight: 600; margin-bottom: 0.75rem; font-size:1.1rem; }
    .loader-container { display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 200px; padding: 2rem; font-size: 1rem; color: var(--text-muted-color); }
    .loader { border: 5px solid var(--light-bg); border-top: 5px solid var(--primary-color); border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin-bottom: 1rem; }
    .content-text { white-space: pre-wrap; line-height: 1.8; font-size: 1.05rem; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    body.dark-mode .summary-box { background-color: rgba(var(--primary-color-rgb), 0.1); border-color: rgba(var(--primary-color-rgb), 0.2); }
    body.dark-mode .takeaways-box { background-color: rgba(var(--secondary-color-rgb), 0.1); border-left-color: var(--secondary-light); }
</style>
{% endblock %}
{% block content %}
{% if not article %}
    <div class="alert alert-danger text-center my-5 p-4"><h4><i class="fas fa-exclamation-triangle me-2"></i>Article Not Found</h4><p>The article you are looking for could not be found.</p><a href="{{ url_for('index') }}" class="btn btn-primary mt-2">Go to Homepage</a></div>
{% else %}
<article class="article-full-content-wrapper animate-fade-in">
    <div class="mb-3 d-flex justify-content-between align-items-center">
        <a href="{{ previous_list_page }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-arrow-left me-2"></i>Back to List</a>
        {% if session.user_id %}
        <button id="bookmarkBtn" class="bookmark-btn {% if is_bookmarked %}active{% endif %}" 
                title="{% if is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}"
                data-article-hash-id="{{ article.article_hash_id if is_community_article else article.id }}"
                data-is-community="{{ 'true' if is_community_article else 'false' }}"
                data-title="{{ article.title|e }}"
                data-source-name="{{ (article.author.name if is_community_article and article.author else article.source.name)|e if article.source else '' }}"
                data-image-url="{{ (article.image_url if is_community_article else article.urlToImage)|e }}"
                data-description="{{ (article.description if article.description else '')|e }}"
                data-published-at="{{ (article.published_at.isoformat() if is_community_article and article.published_at else (article.publishedAt if not is_community_article and article.publishedAt else ''))|e }}">
            <i class="fa-solid fa-bookmark"></i>
        </button>
        {% endif %}
    </div>

    <h1 class="mb-2 article-title-main display-6">{{ article.title }}</h1>
    <div class="article-meta-detailed">
        <span class="meta-item" title="Source"><i class="fas fa-{{ 'user-edit' if is_community_article else 'building' }}"></i> {{ article.author.name if is_community_article and article.author else (article.source.name if article.source else 'Unknown Source') }}</span>
        <span class="meta-item" title="Published Date"><i class="far fa-calendar-alt"></i> {{ (article.published_at | to_ist if is_community_article else (article.publishedAt | to_ist if article.publishedAt else 'N/A')) }}</span>
    </div>
    {% set image_to_display = article.image_url if is_community_article else article.urlToImage %}
    {% if image_to_display %}<img src="{{ image_to_display }}" alt="{{ article.title|truncate(50) }}" class="main-article-image">{% endif %}

    <div id="contentLoader" class="loader-container my-4 {% if is_community_article %}d-none{% endif %}"><div class="loader"></div><div>Analyzing article and generating summary...</div></div>

    <div id="articleAnalysisContainer">
        {% if is_community_article %}
            {% if article.groq_summary %}
            <div class="summary-box my-3"><h5><i class="fas fa-book-open me-2"></i>Article Summary (AI Enhanced)</h5><p class="mb-0">{{ article.groq_summary|replace('\\n', '<br>')|safe }}</p></div>
            {% elif not article.groq_summary and groq_client %} 
            <div class="alert alert-secondary small p-3 mt-3">AI Summary not available for this community article.</div>
            {% endif %}
            {% if article.parsed_takeaways %}
            <div class="takeaways-box my-3"><h5><i class="fas fa-list-check me-2"></i>Key Takeaways (AI Enhanced)</h5>
                <ul>{% for takeaway in article.parsed_takeaways %}<li>{{ takeaway }}</li>{% endfor %}</ul>
            </div>
            {% elif not article.groq_takeaways and groq_client %} 
            <div class="alert alert-secondary small p-3 mt-3">AI Takeaways not available for this community article.</div>
            {% endif %}
            <hr class="my-4">
            <h4 class="mb-3">Full Article Content</h4>
            <div class="content-text">{{ article.full_text }}</div>
        {% else %}
            <div id="apiArticleContent"></div>
        {% endif %}
    </div>

    <section class="comment-section" id="comment-section">
        <h3 class="mb-4">Community Discussion (<span id="comment-count">{{ comments|length }}</span>)</h3>
        {% macro render_comment_with_replies(comment, comment_data, is_logged_in) %}
            <div class="comment-container" id="comment-{{ comment.id }}">
                <div class="comment-card"><div class="comment-avatar" title="{{ comment.author.name if comment.author else 'Unknown' }}">{{ (comment.author.name[0]|upper if comment.author and comment.author.name else 'U') }}</div>
                    <div class="comment-body">
                        <div class="comment-header"><span class="comment-author">{{ comment.author.name if comment.author else 'Anonymous' }}</span><span class="comment-date">{{ comment.timestamp | to_ist }}</span></div>
                        <p class="comment-content mb-2">{{ comment.content }}</p>
                        {% if is_logged_in %}{% set current_comment_data = comment_data.get(comment.id, {}) %}
                        <div class="comment-actions">
                            {% for type, emoji in ALLOWED_REACTIONS.items() %}
                            <button class="reaction-btn {% if current_comment_data.get('user_reaction') == type %}active{% endif %}" data-comment-id="{{ comment.id }}" data-reaction-type="{{ type }}" title="{{ type|capitalize }}"><span class="emoji">{{ emoji }}</span><span class="reaction-count">{{ current_comment_data.get('reactions', {}).get(type, 0) }}</span></button>
                            {% endfor %}
                            <button class="reply-btn" data-comment-id="{{ comment.id }}" title="Reply"><i class="fas fa-reply"></i> Reply</button>
                        </div>
                        <div class="reply-form-container" id="reply-form-container-{{ comment.id }}">
                            <form class="reply-form mt-2"><input type="hidden" name="parent_id" value="{{ comment.id }}"><div class="mb-2"><textarea class="form-control form-control-sm" name="content" rows="2" placeholder="Write a reply..." required></textarea></div><button type="submit" class="btn btn-sm btn-primary-modal">Post Reply</button><button type="button" class="btn btn-sm btn-outline-secondary-modal cancel-reply-btn">Cancel</button></form>
                        </div>{% endif %}
                    </div>
                </div>
                <div class="comment-replies" id="replies-of-{{ comment.id }}">
                    {% for reply in comment.replies|sort(attribute='timestamp') %}{{ render_comment_with_replies(reply, comment_data, is_logged_in) }}{% endfor %}
                </div>
            </div>
        {% endmacro %}
        <div id="comments-list">
            {% for comment in comments %}{{ render_comment_with_replies(comment, comment_data, session.user_id) }}{% else %}<p id="no-comments-msg">No comments yet. Be the first to share your thoughts!</p>{% endfor %}
        </div>
        {% if session.user_id %}
            <div class="add-comment-form mt-4 pt-4 border-top">
                <h5 class="mb-3">Leave a Comment</h5>
                <form id="comment-form"><div class="mb-3"><textarea class="form-control" id="comment-content" name="content" rows="4" placeholder="Share your insights..." required></textarea></div><button type="submit" class="btn btn-primary-modal">Post Comment</button></form>
            </div>
        {% else %}<div class="alert alert-light mt-4 text-center">Please <a href="{{ url_for('login', next=request.url) }}">log in</a> to join the discussion.</div>{% endif %}
    </section>
</article>
{% endif %}
{% endblock %}
{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    {% if article %} 
    const isCommunityArticle = {{ is_community_article | tojson }};
    const articleHashIdGlobal = {{ (article.article_hash_id if is_community_article else article.id) | tojson }};
    const isUserLoggedIn = {{ 'true' if session.user_id else 'false' }};
    const ALLOWED_REACTIONS_JS = {{ ALLOWED_REACTIONS | tojson }};

    function convertUTCToIST(utcIsoString) {
        if (!utcIsoString) return "N/A";
        const date = new Date(utcIsoString);
        return new Intl.DateTimeFormat('en-IN', { year: 'numeric', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'Asia/Kolkata', timeZoneName: 'short' }).format(date);
    }

    if (!isCommunityArticle && articleHashIdGlobal) {
        const contentLoader = document.getElementById('contentLoader');
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            controller.abort();
            if (contentLoader) contentLoader.innerHTML = `<div class="alert alert-warning small p-3">The request timed out. The server or external APIs might be slow. Please try again later.</div>`;
        }, 60000);

        fetch(`{{ url_for('get_article_content_json', article_hash_id='_') }}`.replace('_', articleHashIdGlobal), { signal: controller.signal })
            .then(response => {
                clearTimeout(timeoutId);
                if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
                return response.json();
            })
            .then(data => {
                if (contentLoader) contentLoader.style.display = 'none';
                const apiArticleContent = document.getElementById('apiArticleContent');
                if (!apiArticleContent) return;
                let html = '';
                if (data.error) {
                    html = `<div class="alert alert-warning small p-3 mt-3">Could not load article content: ${data.error}</div>`;
                } else if (data.groq_analysis) {
                    const analysis = data.groq_analysis;
                    if (analysis.error) { html += `<div class="alert alert-secondary small p-3 mt-3">AI analysis failed: ${analysis.error}</div>`; }
                    else {
                        if (analysis.groq_summary) html += `<div class="summary-box my-3"><h5><i class="fas fa-book-open me-2"></i>Article Summary (AI Enhanced)</h5><p class="mb-0">${analysis.groq_summary.replace(/\\n/g, '<br>')}</p></div>`;
                        if (analysis.groq_takeaways && analysis.groq_takeaways.length) html += `<div class="takeaways-box my-3"><h5><i class="fas fa-list-check me-2"></i>Key Takeaways (AI Enhanced)</h5><ul>${analysis.groq_takeaways.map(t => `<li>${String(t)}</li>`).join('')}</ul></div>`;
                    }
                }
                const articleUrl = {{ article.url | tojson if article and not is_community_article else 'null' }};
                if (articleUrl) html += `<hr class="my-4"><a href="${articleUrl}" class="btn btn-outline-primary mt-3 mb-3" target="_blank" rel="noopener noreferrer">Read Original Article</a>`;
                apiArticleContent.innerHTML = html;
            })
            .catch(error => {
                clearTimeout(timeoutId);
                if (error.name !== 'AbortError' && contentLoader) {
                    contentLoader.innerHTML = `<div class="alert alert-danger small p-3">Failed to load article analysis: ${error.message}</div>`;
                }
                console.error("Error fetching article content:", error);
            });
    }

    const commentSection = document.getElementById('comment-section');
    function createCommentHTML(comment) {
        const commentDate = convertUTCToIST(comment.timestamp);
        const authorName = comment.author ? comment.author.name : 'Anonymous';
        let actionsHTML = '';
        if (isUserLoggedIn) {
            let reactionButtons = Object.entries(ALLOWED_REACTIONS_JS).map(([type, emoji]) => 
                `<button class="reaction-btn" data-comment-id="${comment.id}" data-reaction-type="${type}"><span class="emoji">${emoji}</span><span class="reaction-count">0</span></button>`
            ).join('');
            actionsHTML = `<div class="comment-actions">${reactionButtons}<button class="reply-btn" data-comment-id="${comment.id}"><i class="fas fa-reply"></i> Reply</button></div><div class="reply-form-container" id="reply-form-container-${comment.id}"><form class="reply-form mt-2"><input type="hidden" name="parent_id" value="${comment.id}"><div class="mb-2"><textarea class="form-control form-control-sm" name="content" rows="2" required></textarea></div><button type="submit" class="btn btn-sm btn-primary-modal">Post</button><button type="button" class="btn btn-sm btn-outline-secondary-modal cancel-reply-btn">Cancel</button></form></div>`;
        }
        return `<div class="comment-container" id="comment-${comment.id}"><div class="comment-card"><div class="comment-avatar">${authorName[0]}</div><div class="comment-body"><div class="comment-header"><span class="comment-author">${authorName}</span><span class="comment-date">${commentDate}</span></div><p class="comment-content mb-2">${comment.content}</p>${actionsHTML}</div></div><div class="comment-replies" id="replies-of-${comment.id}"></div></div>`;
    }

    function handleCommentSubmit(form, parentId = null) {
        const content = form.querySelector('textarea[name="content"]').value.trim(); if (!content) return;
        fetch(`{{ url_for('add_comment', article_hash_id='_') }}`.replace('_', articleHashIdGlobal), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content, parent_id: parentId }) })
        .then(res => res.json()).then(data => {
            if (data.success) {
                const newCommentNode = document.createElement('div'); newCommentNode.innerHTML = createCommentHTML(data.comment);
                if (parentId) { document.getElementById(`replies-of-${parentId}`).appendChild(newCommentNode.firstChild); form.closest('.reply-form-container').style.display = 'none'; }
                else { const list = document.getElementById('comments-list'); document.getElementById('no-comments-msg')?.remove(); list.appendChild(newCommentNode.firstChild); document.getElementById('comment-count').textContent = parseInt(document.getElementById('comment-count').textContent) + 1; }
                form.reset();
            } else { alert(`Error: ${data.error}`); }
        });
    }

    if (commentSection) {
        commentSection.addEventListener('click', function(e) {
            const target = e.target;
            const reactionBtn = target.closest('.reaction-btn');
            if (reactionBtn && isUserLoggedIn) {
                fetch(`{{ url_for('vote_comment', comment_id=0) }}`.replace('0', reactionBtn.dataset.commentId), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ reaction_type: reactionBtn.dataset.reactionType }) })
                .then(res => res.json()).then(data => {
                    if (data.success) {
                        document.querySelectorAll(`.reaction-btn[data-comment-id="${reactionBtn.dataset.commentId}"]`).forEach(btn => {
                            btn.querySelector('.reaction-count').textContent = data.reactions[btn.dataset.reactionType] || 0;
                            btn.classList.toggle('active', btn.dataset.reactionType === data.user_reaction);
                        });
                    }
                });
            }
            const replyBtn = target.closest('.reply-btn');
            if (replyBtn) document.getElementById(`reply-form-container-${replyBtn.dataset.commentId}`).style.display = 'block';
            const cancelReplyBtn = target.closest('.cancel-reply-btn');
            if (cancelReplyBtn) cancelReplyBtn.closest('.reply-form-container').style.display = 'none';
        });
        commentSection.addEventListener('submit', function(e) {
            e.preventDefault();
            const replyForm = e.target.closest('.reply-form');
            if (replyForm) handleCommentSubmit(replyForm, replyForm.querySelector('input[name="parent_id"]').value);
            const mainForm = e.target.closest('#comment-form');
            if (mainForm) handleCommentSubmit(mainForm);
        });
    }

    const bookmarkBtn = document.getElementById('bookmarkBtn');
    if (bookmarkBtn) { bookmarkBtn.addEventListener('click', function() { /* ... unchanged bookmark logic ... */ }); }
    {% endif %} 
});
</script>
{% endblock %}
"""
LOGIN_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Login - BrieflyAI{% endblock %}
{% block content %}
<div class="auth-container article-card animate-fade-in mx-auto">
    <h2 class="auth-title mb-4"><i class="fas fa-sign-in-alt me-2"></i>Member Login</h2>
    <form method="POST" action="{{ url_for('login', next=request.args.get('next')) }}">
        <div class="modal-form-group"><label for="username" class="form-label">Username</label><input type="text" class="modal-form-control" id="username" name="username" required placeholder="Enter your username"></div>
        <div class="modal-form-group"><label for="password" class="form-label">Password</label><input type="password" class="modal-form-control" id="password" name="password" required placeholder="Enter your password"></div>
        <button type="submit" class="btn btn-primary-modal w-100 mt-3">Login</button>
    </form>
    <p class="mt-3 text-center small">Don't have an account? <a href="{{ url_for('register', next=request.args.get('next')) }}" class="fw-medium">Register here</a></p>
</div>
{% endblock %}
"""

REGISTER_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Register - BrieflyAI{% endblock %}
{% block content %}
<div class="auth-container article-card animate-fade-in mx-auto">
    <h2 class="auth-title mb-4"><i class="fas fa-user-plus me-2"></i>Create Account</h2>
    <form method="POST" action="{{ url_for('register') }}">
        <div class="modal-form-group"><label for="name" class="form-label">Full Name</label><input type="text" class="modal-form-control" id="name" name="name" required placeholder="Enter your full name"></div>
        <div class="modal-form-group"><label for="username" class="form-label">Username</label><input type="text" class="modal-form-control" id="username" name="username" required placeholder="Choose a username (min 3 chars)"></div>
        <div class="modal-form-group"><label for="password" class="form-label">Password</label><input type="password" class="modal-form-control" id="password" name="password" required placeholder="Create a strong password (min 6 chars)"></div>
        <button type="submit" class="btn btn-primary-modal w-100 mt-3">Register</button>
    </form>
    <p class="mt-3 text-center small">Already have an account? <a href="{{ url_for('login') }}" class="fw-medium">Login here</a></p>
</div>
{% endblock %}
"""

PROFILE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ user.name }}'s Profile - {% endblock %}
{% block content %}
<div class="profile-card animate-fade-in">
    <div class="profile-avatar">{{ user.name[0]|upper }}</div>
    <h2 class="mb-1">{{ user.name }}</h2>
    <p class="text-muted">@{{ user.username }}</p>
    <p class="small text-muted">Joined: {{ user.created_at | to_ist }}</p>
</div>
<div class="mt-4 animate-fade-in">
    <ul class="nav nav-tabs profile-tabs nav-fill mb-4" id="profileTab" role="tablist">
        <li class="nav-item" role="presentation">
            <button class="nav-link active" id="bookmarks-tab" data-bs-toggle="tab" data-bs-target="#bookmarks-content" type="button" role="tab" aria-controls="bookmarks-content" aria-selected="true"><i class="fas fa-bookmark me-1"></i>My Bookmarks ({{ bookmarks_pagination.total if bookmarks_pagination else 0 }})</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="posted-tab" data-bs-toggle="tab" data-bs-target="#posted-content" type="button" role="tab" aria-controls="posted-content" aria-selected="false"><i class="fas fa-feather-alt me-1"></i>My Posted Articles ({{ posted_articles|length }})</button>
        </li>
    </ul>
    <div class="tab-content" id="profileTabContent">
        <div class="tab-pane fade show active" id="bookmarks-content" role="tabpanel" aria-labelledby="bookmarks-tab">
            {% if bookmarked_articles %}
            <div class="row g-4">
                {% for art in bookmarked_articles %}
                <div class="col-md-6 col-lg-4 d-flex">
                    <article class="article-card d-flex flex-column w-100">
                        <div class="article-image-container">
                            <a href="{{ art.article_url }}"><img src="{{ art.urlToImage if art.urlToImage else 'https://via.placeholder.com/400x220/EEEEEE/AAAAAA?text=No+Image' }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a>
                            {% if art.is_stale_bookmark %}<span class="category-tag" style="background-color: #777; color:white;">Cached Bookmark</span>{% endif %}
                        </div>
                        <div class="article-body d-flex flex-column">
                            <h5 class="article-title mb-2"><a href="{{ art.article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                            <div class="article-meta small mb-2">
                                <span class="meta-item text-muted"><i class="fas fa-{{ 'user-edit' if art.is_community_article else 'building' }}"></i> {{ art.source.name|truncate(20) }}</span>
                                <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ (art.publishedAt | to_ist if art.publishedAt else 'N/A') }}</span>
                            </div>
                            <p class="article-description small">{{ art.description|truncate(100) }}</p>
                            <a href="{{ art.article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
                        </div>
                    </article>
                </div>
                {% endfor %}
            </div>
            {% else %}<div class="alert alert-light text-center p-4">You haven't bookmarked any articles yet.</div>{% endif %}
            {% if bookmarks_pagination and bookmarks_pagination.pages > 1 %}
            <nav aria-label="Bookmarks navigation" class="mt-5">
                <ul class="pagination justify-content-center">
                    <li class="page-item page-link-prev-next {% if not bookmarks_pagination.has_prev %}disabled{% endif %}"><a class="page-link" href="{{ url_for('profile', page=bookmarks_pagination.prev_num) if bookmarks_pagination.has_prev else '#' }}">&laquo; Prev</a></li>
                    {% for p in bookmarks_pagination.iter_pages(left_edge=1, right_edge=1, left_current=1, right_current=2) %}{% if p %}{% if p == bookmarks_pagination.page %}<li class="page-item active" aria-current="page"><span class="page-link">{{ p }}</span></li>{% else %}<li class="page-item"><a class="page-link" href="{{ url_for('profile', page=p) }}">{{ p }}</a></li>{% endif %}{% else %}<li class="page-item disabled"><span class="page-link">...</span></li>{% endif %}{% endfor %}
                    <li class="page-item page-link-prev-next {% if not bookmarks_pagination.has_next %}disabled{% endif %}"><a class="page-link" href="{{ url_for('profile', page=bookmarks_pagination.next_num) if bookmarks_pagination.has_next else '#' }}">Next &raquo;</a></li>
                </ul>
            </nav>
            {% endif %}
        </div>
        <div class="tab-pane fade" id="posted-content" role="tabpanel" aria-labelledby="posted-tab">
            {% if posted_articles %}
            <div class="row g-4">
                {% for art in posted_articles %}
                <div class="col-md-6 col-lg-4 d-flex">
                     <article class="article-card d-flex flex-column w-100">
                        {% set article_url = url_for('article_detail', article_hash_id=art.article_hash_id) %}
                        <div class="article-image-container"><a href="{{ article_url }}"><img src="{{ art.image_url if art.image_url else 'https://via.placeholder.com/400x220/EEEEEE/AAAAAA?text=No+Image' }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a></div>
                        <div class="article-body d-flex flex-column">
                            <h5 class="article-title mb-2"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                            <div class="article-meta small mb-2">
                                <span class="meta-item text-muted"><i class="fas fa-user-edit"></i> {{ art.author.name|truncate(20) }}</span>
                                <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ art.published_at | to_ist }}</span>
                            </div>
                            <p class="article-description small">{{ art.description|truncate(100) }}</p>
                            <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
                        </div>
                    </article>
                </div>
                {% endfor %}
            </div>
            {% else %}<div class="alert alert-light text-center p-4">You haven't posted any articles yet. Click the '+' button to share your insights!</div>{% endif %}
        </div>
    </div>
</div>
{% endblock %}
"""

ABOUT_US_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}About Us - {% endblock %}
{% block content %}<div class="static-content-wrapper animate-fade-in"><h1 class="mb-4">About </h1><p class="lead"> is your premier destination for the latest news from India and around the world, delivered in a concise and easy-to-digest format. We leverage the power of cutting-edge AI to summarize complex news articles into key takeaways, saving you time while keeping you informed.</p><h2 class="mt-5 mb-3">Our Mission</h2><p>In a world of information overload, our mission is to provide clarity and efficiency. We believe that everyone deserves access to accurate, unbiased news without spending hours sifting through lengthy articles.  cuts through the noise, offering insightful summaries that matter.</p><h2 class="mt-5 mb-3">Community Hub</h2><p>Beyond AI-driven news,  is a platform for discussion and community engagement. Our Community Hub allows users to post their own articles, share perspectives, and engage in meaningful conversations about the topics that shape our world. We are committed to fostering a respectful and intelligent environment for all our members.</p><h2 class="mt-5 mb-3">Our Technology</h2><p>We use state-of-the-art Natural Language Processing (NLP) models to analyze and summarize news content from trusted sources. Our system is designed to identify the most crucial points of an article, presenting them as a quick summary and a list of key takeaways.</p></div>{% endblock %}
"""

CONTACT_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Contact Us - {% endblock %}
{% block content %}<div class="static-content-wrapper animate-fade-in"><h1 class="mb-4">Contact Us</h1><p class="lead">We'd love to hear from you! Whether you have a question, feedback, or a news tip, feel free to reach out.</p><div class="row mt-5"><div class="col-md-6"><h2 class="h4">General Inquiries</h2><p>For general questions, feedback, or support, please email us at:</p><p><i class="fas fa-envelope me-2"></i><a href="mailto:vbansal639@gmail.com">vbansal639@gmail.com</a></p></div><div class="col-md-6"><h2 class="h4">Partnerships & Media</h2><p>For partnership opportunities or media inquiries, please contact:</p><p><i class="fas fa-envelope me-2"></i><a href="mailto:vbansal639@gmail.com">vbansal639@gmail.com</a></p></div></div><div class="mt-5"><h2 class="h4">Follow Us</h2><p>Stay connected with us on social media:</p><div class="social-links fs-4"><a href="#" title="Twitter"><i class="fab fa-twitter"></i></a><a href="#" title="Facebook"><i class="fab fa-facebook-f"></i></a><a href="#" title="LinkedIn"><i class="fab fa-linkedin-in"></i></a><a href="#" title="Instagram"><i class="fab fa-instagram"></i></a></div></div></div>{% endblock %}
"""

PRIVACY_POLICY_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Privacy Policy - {% endblock %}
{% block content %}<div class="static-content-wrapper animate-fade-in"><h1 class="mb-4">Privacy Policy</h1><p class="text-muted">Last updated: May 31, 2025</p><p>BrieflyAI ("we," "our," or "us") is committed to protecting your privacy. This Privacy Policy explains how we collect, use, disclose, and safeguard your information when you visit our website.</p><h2 class="mt-5 mb-3">1. Information We Collect</h2><p>We may collect personal information that you voluntarily provide to us when you register on the website, post articles or comments, bookmark articles, or subscribe to our newsletter. This information may include your name, username, email address, and your activities on our platform such as articles posted and bookmarked.</p><h2 class="mt-5 mb-3">2. How We Use Your Information</h2><p>We use the information we collect to:</p><ul><li>Create and manage your account.</li><li>Operate and maintain the website, including your profile page.</li><li>Display your posted and bookmarked articles as part of your profile.</li><li>Send you newsletters or promotional materials, if you have opted in.</li><li>Respond to your comments and inquiries.</li><li>Improve our website and services.</li></ul><h2 class="mt-5 mb-3">3. Disclosure of Your Information</h2><p>Your username and posted articles are publicly visible. Your bookmarked articles are visible on your profile page to you when logged in. We do not sell, trade, or otherwise transfer your personally identifiable information like your email address to outside parties without your consent, except to trusted third parties who assist us in operating our website, so long as those parties agree to keep this information confidential.</p><h2 class="mt-5 mb-3">4. Security of Your Information</h2><p>We use administrative, technical, and physical security measures to help protect your personal information. While we have taken reasonable steps to secure the personal information you provide to us, please be aware that despite our efforts, no security measures are perfect or impenetrable.</p><h2 class="mt-5 mb-3">5. Your Choices</h2><p>You can review and change your profile information by logging into your account. You may also request deletion of your account and associated data by contacting us.</p><h2 class="mt-5 mb-3">6. Changes to This Privacy Policy</h2><p>We may update this Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page. You are advised to review this Privacy Policy periodically for any changes.</p></div>{% endblock %}
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
template_storage['PROFILE_HTML_TEMPLATE'] = PROFILE_HTML_TEMPLATE
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
