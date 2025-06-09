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
app.config['CACHE_EXPIRY_SECONDS'] = 1800 
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
    groq_takeaways = db.Column(db.Text, nullable=True) # Stored as JSON string
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
    # MODIFIED: Changed to store the specific emoji character for the reaction.
    vote_emoji = db.Column(db.String(10), nullable=False)
    # The unique constraint ensures a user can only have one reaction per comment.
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
    published_at_cache = db.Column(db.DateTime, nullable=True) # Store as datetime for API articles
    bookmarked_at = db.Column(db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    __table_args__ = (db.UniqueConstraint('user_id', 'article_hash_id', name='_user_article_bookmark_uc'),)

def init_db():
    # To access the global 'using_postgres_flag' set earlier
    global using_postgres_flag
    with app.app_context():
        app.logger.info("--- Database Initialization (init_db) ---")
        try:
            engine = db.get_engine()
            engine_url_str = str(engine.url) 
            dialect_name = engine.dialect.name
            app.logger.info(f"SQLAlchemy engine is configured with URL (actual): {engine_url_str}")
            app.logger.info(f"SQLAlchemy dialect in use: {dialect_name}")

            # Check the flag set during initial config
            if using_postgres_flag: 
                if dialect_name == "postgresql":
                    app.logger.info("CONFIRMED: SQLAlchemy is using PostgreSQL dialect as intended.")
                else:
                    app.logger.warning(f"WARNING: Intended to use PostgreSQL (using_postgres_flag=True), but SQLAlchemy dialect is '{dialect_name}'. Check DATABASE_URL and configuration.")
            else: # Intended to use SQLite (fallback)
                if dialect_name == "sqlite":
                    app.logger.info("CONFIRMED: SQLAlchemy is using SQLite dialect as intended (fallback or no valid DATABASE_URL).")
                else:
                    app.logger.warning(f"WARNING: Intended to use SQLite (using_postgres_flag=False), but SQLAlchemy dialect is '{dialect_name}'.")
        except Exception as e:
            app.logger.error(f"Error during SQLAlchemy engine/dialect logging: {e}", exc_info=True)

        app.logger.info("Attempting to create database tables (db.create_all()). This is non-destructive to existing tables.")
        try:
            db.create_all()
            app.logger.info("db.create_all() executed successfully. Tables should be ready or already exist.")
        except Exception as e:
            app.logger.error(f"Error during db.create_all(): {e}", exc_info=True)
        app.logger.info("--- End of Database Initialization (init_db) ---")

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

@simple_cache(expiry_seconds_default=3600 * 12)
def get_article_analysis_with_groq(article_text, article_title=""):
    if not groq_client: return {"error": "AI analysis service not available."}
    if not article_text or not article_text.strip(): return {"error": "No text provided for AI analysis."}
    app.logger.info(f"Requesting Groq analysis for: {article_title[:50]}...")
    system_prompt = ("You are an expert news analyst. Analyze the following article. "
        "1. Provide a concise, neutral summary (3-4 paragraphs). "
        "2. List 5-7 key takeaways as bullet points. Each takeaway must be a complete sentence. "
        "Format your entire response as a single JSON object with keys 'summary' (string) and 'takeaways' (a list of strings).")
    human_prompt = f"Article Title: {article_title}\n\nArticle Text:\n{article_text[:20000]}"
    try:
        json_model = groq_client.bind(response_format={"type": "json_object"})
        ai_response = json_model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        analysis = json.loads(ai_response.content) # Groq should return JSON string in ai_response.content
        
        # Ensure keys exist, default to null or empty if not explicitly present
        groq_summary = analysis.get("summary") 
        groq_takeaways = analysis.get("takeaways")

        # Even if keys exist, they might be null from Groq if it couldn't generate them
        # No specific error from Groq if it returns nulls in valid JSON format
        return {"groq_summary": groq_summary, "groq_takeaways": groq_takeaways, "error": None}

    except json.JSONDecodeError as e:
        app.logger.error(f"Groq analysis - JSONDecodeError for '{article_title[:50]}': {e}. Response content: {ai_response.content if 'ai_response' in locals() else 'N/A'}")
        return {"error": f"AI analysis failed to decode JSON: {str(e)}"}
    except LangChainException as e: # More specific Langchain errors
        app.logger.error(f"Groq analysis - LangChainException for '{article_title[:50]}': {e}")
        return {"error": f"AI analysis failed (LangChain): {str(e)}"}
    except Exception as e:
        app.logger.error(f"Unexpected error during Groq analysis for '{article_title[:50]}': {e}", exc_info=True)
        return {"error": "An unexpected error occurred during AI analysis."}

# ==============================================================================
# --- NEWS FETCHING ---
# ==============================================================================
@simple_cache()
def fetch_news_from_api(target_date_str=None):
    if not newsapi:
        app.logger.error("NewsAPI client not initialized. Cannot fetch news.")
        return []

    api_call_from_date_str = None
    api_call_to_date_str = None
    is_specific_date_fetch = False

    # INDIAN_TIMEZONE should be globally defined in your script, e.g.,
    # INDIAN_TIMEZONE = pytz.timezone('Asia/Kolkata')

    if target_date_str:
        try:
            # Primary: Interpret target_date_str as user's local day in INDIAN_TIMEZONE
            local_day_start_naive = datetime.strptime(target_date_str, '%Y-%m-%d')
            local_day_start_aware_ist = INDIAN_TIMEZONE.localize(local_day_start_naive) # Assign IST timezone
            # Define end of the local day in IST
            local_day_end_aware_ist = local_day_start_aware_ist.replace(hour=23, minute=59, second=59, microsecond=999999)

            # Convert these IST times to UTC for the NewsAPI query
            api_call_from_utc_dt = local_day_start_aware_ist.astimezone(timezone.utc)
            api_call_to_utc_dt = local_day_end_aware_ist.astimezone(timezone.utc)

            api_call_from_date_str = api_call_from_utc_dt.strftime('%Y-%m-%dT%H:%M:%S')
            api_call_to_date_str = api_call_to_utc_dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            app.logger.info(f"Date filter active for '{target_date_str}' (interpreted as IST).")
            app.logger.info(f"Querying NewsAPI with UTC range: FROM {api_call_from_date_str} TO {api_call_to_date_str}")
            is_specific_date_fetch = True
        except Exception as e:
            app.logger.error(f"Error processing target_date_str '{target_date_str}' with IST conversion: {e}. Reverting to simpler UTC day or default fetch.", exc_info=True)
            # Fallback 1: Try interpreting target_date_str as a simple UTC date (original behavior for specific date)
            try:
                utc_target_dt = datetime.strptime(target_date_str, '%Y-%m-%d')
                api_call_from_date_str = utc_target_dt.strftime('%Y-%m-%dT00:00:00')
                api_call_to_date_str = utc_target_dt.strftime('%Y-%m-%dT23:59:59')
                app.logger.info(f"Fallback: Fetching news specifically for UTC date: {target_date_str} (from {api_call_from_date_str} to {api_call_to_date_str})")
                is_specific_date_fetch = True
            except ValueError: # If target_date_str is malformed even for simple parsing
                 app.logger.warning(f"Invalid target_date_str '{target_date_str}' for both IST and UTC interpretation. Clearing date filter.")
                 target_date_str = None # This will ensure it uses the default N-day range logic in the next block
                 is_specific_date_fetch = False # Ensure default logic runs if date string is unusable

    if not is_specific_date_fetch: 
        # Default fetch logic (no specific date selected, or date was invalid)
        # This fetches news from the last N days up to the end of the current UTC day.
        from_date_utc_default = datetime.now(timezone.utc) - timedelta(days=app.config['NEWS_API_DAYS_AGO'])
        api_call_from_date_str = from_date_utc_default.strftime('%Y-%m-%dT%H:%M:%S')
        
        current_day_utc_end_default = datetime.now(timezone.utc).replace(hour=23, minute=59, second=59, microsecond=0)
        api_call_to_date_str = current_day_utc_end_default.strftime('%Y-%m-%dT%H:%M:%S')
        app.logger.info(f"Fetching news with default date range (last {app.config['NEWS_API_DAYS_AGO']} days up to current UTC day end): from {api_call_from_date_str} to {api_call_to_date_str}")

    # --- API Call Attempts (The rest of the function remains the same as my previous detailed response) ---
    all_raw_articles = []

    # Attempt 1: Top Headlines (only if not fetching for a specific historical date, or for default range)
    # Note: is_specific_date_fetch is true if a date was successfully parsed (either IST-based or fallback UTC-based)
    if not is_specific_date_fetch: # Only run top_headlines if no specific date was successfully set for the fetch
        try:
            app.logger.info("Attempt 1: Fetching top headlines from country: 'in' (default range).")
            top_headlines_response = newsapi.get_top_headlines(
                country='in',
                language='en',
                page_size=app.config['NEWS_API_PAGE_SIZE']
            )
            status = top_headlines_response.get('status')
            total_results = top_headlines_response.get('totalResults', 0)
            app.logger.info(f"Top-Headlines API Response -> Status: {status}, TotalResults: {total_results}")
            if status == 'ok' and total_results > 0:
                all_raw_articles.extend(top_headlines_response['articles'])
            elif status == 'error':
                app.logger.error(f"NewsAPI Error (Top-Headlines): Code: {top_headlines_response.get('code')}, Message: {top_headlines_response.get('message')}. Full response: {top_headlines_response}")
            elif total_results == 0:
                app.logger.info(f"NewsAPI (Top-Headlines) returned 0 results. Response: {top_headlines_response}")
        except NewsAPIException as e:
            app.logger.error(f"NewsAPIException (Top-Headlines): {e.get_message() if hasattr(e, 'get_message') else str(e)}", exc_info=False)
        except Exception as e:
            app.logger.error(f"Generic Exception (Top-Headlines): {e}", exc_info=True)

    # Attempt 2: Everything query (This will always run, using the determined date range)
    try:
        current_query = app.config['NEWS_API_QUERY']
        current_sort_by = app.config['NEWS_API_SORT_BY']
        app.logger.info(f"Attempt 2: Fetching 'everything' with query: \"{current_query}\" for period: {api_call_from_date_str} to {api_call_to_date_str}, sort_by: {current_sort_by}")
        everything_response = newsapi.get_everything(
            q=current_query,
            from_param=api_call_from_date_str,
            to=api_call_to_date_str,
            language='en',
            sort_by=current_sort_by,
            page_size=app.config['NEWS_API_PAGE_SIZE']
        )
        status = everything_response.get('status')
        total_results = everything_response.get('totalResults', 0)
        app.logger.info(f"Everything API Response -> Status: {status}, TotalResults: {total_results}")
        if status == 'ok' and total_results > 0:
            all_raw_articles.extend(everything_response['articles'])
        elif status == 'error':
            app.logger.error(f"NewsAPI Error (Everything Query): Code: {everything_response.get('code')}, Message: {everything_response.get('message')}. Parameters used: q='{current_query}', from='{api_call_from_date_str}', to='{api_call_to_date_str}', sort_by='{current_sort_by}'. Full response: {everything_response}")
        elif total_results == 0:
            app.logger.info(f"NewsAPI (Everything Query) returned 0 results for q='{current_query}' from '{api_call_from_date_str}' to '{api_call_to_date_str}', sort_by='{current_sort_by}'. Response: {everything_response}")
    except NewsAPIException as e:
        app.logger.error(f"NewsAPIException (Everything Query): {e.get_message() if hasattr(e, 'get_message') else str(e)}. Parameters used: q='{current_query}', from='{api_call_from_date_str}', to='{api_call_to_date_str}', sort_by='{current_sort_by}'.", exc_info=False)
    except Exception as e:
        app.logger.error(f"Generic Exception (Everything Query): {e}", exc_info=True)

    # Attempt 3: Fallback with domains
    # Condition: Run if no articles yet, OR if it was a specific date fetch (to maximize chances for that day)
    # The `is_specific_date_fetch` flag is true if the target_date_str was successfully parsed into a date range (either IST-based or UTC-based fallback).
    if not all_raw_articles or is_specific_date_fetch:
        log_prefix_attempt3 = "Fallback/Augment"
        if not all_raw_articles and not is_specific_date_fetch:
             app.logger.warning("No articles from primary calls. Trying Fallback with domains for default range.")
        elif not all_raw_articles and is_specific_date_fetch:
            app.logger.warning(f"No articles from query for specific date '{target_date_str}'. Trying with domains.")
        elif all_raw_articles and is_specific_date_fetch:
            app.logger.info(f"Augmenting results for specific date '{target_date_str}' with domain-specific search.")


        try:
            domains_to_check = app.config['NEWS_API_DOMAINS']
            current_sort_by_fallback = app.config['NEWS_API_SORT_BY']
            app.logger.info(f"Attempt 3 ({log_prefix_attempt3}): Fetching from domains: {domains_to_check} for period: {api_call_from_date_str} to {api_call_to_date_str}, sort_by: {current_sort_by_fallback}")
            fallback_response = newsapi.get_everything(
                domains=domains_to_check,
                from_param=api_call_from_date_str,
                to=api_call_to_date_str,
                language='en',
                sort_by=current_sort_by_fallback,
                page_size=app.config['NEWS_API_PAGE_SIZE']
            )
            status = fallback_response.get('status')
            total_results = fallback_response.get('totalResults', 0)
            app.logger.info(f"{log_prefix_attempt3} API Response -> Status: {status}, TotalResults: {total_results}")
            if status == 'ok' and total_results > 0:
                all_raw_articles.extend(fallback_response['articles'])
            elif status == 'error':
                app.logger.error(f"NewsAPI Error ({log_prefix_attempt3}): Code: {fallback_response.get('code')}, Message: {fallback_response.get('message')}. Parameters used: domains='{domains_to_check}', from='{api_call_from_date_str}', to='{api_call_to_date_str}', sort_by='{current_sort_by_fallback}'. Full response: {fallback_response}")
            elif total_results == 0:
                app.logger.info(f"NewsAPI ({log_prefix_attempt3}) returned 0 results for domains='{domains_to_check}' from '{api_call_from_date_str}' to '{api_call_to_date_str}', sort_by='{current_sort_by_fallback}'. Response: {fallback_response}")
        except NewsAPIException as e:
            app.logger.error(f"NewsAPIException ({log_prefix_attempt3}): {e.get_message() if hasattr(e, 'get_message') else str(e)}. Parameters used: domains='{domains_to_check}', from='{api_call_from_date_str}', to='{api_call_to_date_str}', sort_by='{current_sort_by_fallback}'.", exc_info=False)
        except Exception as e:
            app.logger.error(f"Generic Exception ({log_prefix_attempt3}): {e}", exc_info=True)

    processed_articles, unique_urls = [], set()
    app.logger.info(f"Total raw articles fetched before deduplication: {len(all_raw_articles)}")
    for art_data in all_raw_articles:
        url = art_data.get('url')
        if not url or url in unique_urls: continue
        title = art_data.get('title')
        description = art_data.get('description')

        if not all([title, art_data.get('source'), description]) or title == '[Removed]' or not title.strip() or not description.strip():
            continue
        unique_urls.add(url)
        article_id = generate_article_id(url)
        source_name = art_data['source'].get('name', 'Unknown Source')
        placeholder_text = urllib.parse.quote_plus(source_name[:20])
        published_at_dt = None
        if art_data.get('publishedAt'):
            try:
                published_at_dt = datetime.fromisoformat(art_data.get('publishedAt').replace('Z', '+00:00'))
            except ValueError:
                app.logger.warning(f"Could not parse publishedAt date for article: {title} - Date: {art_data.get('publishedAt')}")
                published_at_dt = datetime.now(timezone.utc) # Fallback if parsing fails
        else:
            # If publishedAt is missing, which can happen, use current time as a fallback.
            # Or decide if such articles should be skipped. For now, using current time.
            app.logger.warning(f"Missing 'publishedAt' for article: {title}. Using current UTC time.")
            published_at_dt = datetime.now(timezone.utc)


        standardized_article = {
            'id': article_id, 'title': title, 'description': description,
            'url': url, 'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/0D2C54/FFFFFF?text={placeholder_text}',
            'publishedAt': published_at_dt.isoformat(), # Store as ISO string
            'source': {'name': source_name}, 'is_community_article': False,
            'groq_summary': None, 'groq_takeaways': None
        }
        MASTER_ARTICLE_STORE[article_id] = standardized_article
        processed_articles.append(standardized_article)
    
    processed_articles.sort(key=lambda x: x.get('publishedAt', datetime.min.replace(tzinfo=timezone.utc).isoformat()), reverse=True)
    app.logger.info(f"Total unique articles processed and returned by fetch_news_from_api: {len(processed_articles)} for period from {api_call_from_date_str} to {api_call_to_date_str}.")
    return processed_articles

@simple_cache(expiry_seconds_default=21600) # Cache popular news for 6 hours
def fetch_popular_news():
    """
    Fetches articles from the last few days and sorts them by popularity
    to find the most impactful stories.
    """
    app.logger.info("Fetching POPULAR news from API.")
    if not newsapi:
        return []

    # Query a window of the last 5 days to get a good measure of popularity
    to_date = datetime.now(timezone.utc) - timedelta(days=1)
    from_date = to_date - timedelta(days=5)

    try:
        response = newsapi.get_everything(
            q=app.config['NEWS_API_QUERY'],
            language='en',
            from_param=from_date.strftime('%Y-%m-%d'),
            to=to_date.strftime('%Y-%m-%d'),
            sort_by='popularity', # This is the key for this section
            page_size=30
        )
        
        if response.get('status') == 'ok':
            raw_articles = response.get('articles', [])
            processed_articles, unique_urls = [], set()
            for art_data in raw_articles:
                url = art_data.get('url')
                if not url or url in unique_urls: continue
                title = art_data.get('title')
                description = art_data.get('description')

                if not all([title, art_data.get('source'), description]) or title == '[Removed]' or not title.strip() or not description.strip():
                    continue
                
                unique_urls.add(url)
                article_id = generate_article_id(url)
                source_name = art_data['source'].get('name', 'Unknown Source')
                placeholder_text = urllib.parse.quote_plus(source_name[:20])
                
                published_at_dt = None
                if art_data.get('publishedAt'):
                    try:
                        published_at_dt = datetime.fromisoformat(art_data.get('publishedAt').replace('Z', '+00:00'))
                    except ValueError:
                        published_at_dt = datetime.now(timezone.utc)
                else:
                    published_at_dt = datetime.now(timezone.utc)

                standardized_article = {
                    'id': article_id, 'title': title, 'description': description, 'url': url,
                    'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/0D2C54/FFFFFF?text={placeholder_text}',
                    'publishedAt': published_at_dt.isoformat(), 'source': {'name': source_name}, 
                    'is_community_article': False, 'groq_summary': None, 'groq_takeaways': None
                }
                MASTER_ARTICLE_STORE[article_id] = standardized_article
                processed_articles.append(standardized_article)
            
            app.logger.info(f"Returning {len(processed_articles)} unique POPULAR articles.")
            return processed_articles
        return []
    except Exception as e:
        app.logger.error(f"Exception in fetch_popular_news: {e}", exc_info=True)
        return []

@simple_cache(expiry_seconds_default=14400) # Cache yesterday's news for 4 hours
def fetch_yesterdays_latest_news():
    """
    Fetches all articles specifically from yesterday and sorts them by time.
    """
    app.logger.info("Fetching LATEST news from YESTERDAY from API.")
    if not newsapi:
        return []
    
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    try:
        response = newsapi.get_everything(
            q=app.config['NEWS_API_QUERY'],
            language='en',
            from_param=yesterday_str,
            to=yesterday_str,
            sort_by='publishedAt', # This is the key for this section
            page_size=40
        )
        
        if response.get('status') == 'ok':
            raw_articles = response.get('articles', [])
            processed_articles, unique_urls = [], set()
            for art_data in raw_articles:
                url = art_data.get('url')
                if not url or url in unique_urls: continue
                title = art_data.get('title')
                description = art_data.get('description')

                if not all([title, art_data.get('source'), description]) or title == '[Removed]' or not title.strip() or not description.strip():
                    continue
                
                unique_urls.add(url)
                article_id = generate_article_id(url)
                source_name = art_data['source'].get('name', 'Unknown Source')
                placeholder_text = urllib.parse.quote_plus(source_name[:20])
                
                published_at_dt = None
                if art_data.get('publishedAt'):
                    try:
                        published_at_dt = datetime.fromisoformat(art_data.get('publishedAt').replace('Z', '+00:00'))
                    except ValueError:
                        published_at_dt = datetime.now(timezone.utc)
                else:
                    published_at_dt = datetime.now(timezone.utc)

                standardized_article = {
                    'id': article_id, 'title': title, 'description': description, 'url': url,
                    'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/0D2C54/FFFFFF?text={placeholder_text}',
                    'publishedAt': published_at_dt.isoformat(), 'source': {'name': source_name}, 
                    'is_community_article': False, 'groq_summary': None, 'groq_takeaways': None
                }
                MASTER_ARTICLE_STORE[article_id] = standardized_article
                processed_articles.append(standardized_article)
            
            app.logger.info(f"Returning {len(processed_articles)} unique LATEST articles from YESTERDAY.")
            return processed_articles
        return []
    except Exception as e:
        app.logger.error(f"Exception in fetch_yesterdays_latest_news: {e}", exc_info=True)
        return []
        
@simple_cache(expiry_seconds_default=3600 * 6)
def fetch_and_parse_article_content(article_hash_id, url):
    app.logger.info(f"Fetching content for API article ID: {article_hash_id}, URL: {url}")
    if not SCRAPER_API_KEY:
        return {"full_text": None, "groq_analysis": None, "error": "Content fetching service unavailable."}
    
    params = {'api_key': SCRAPER_API_KEY, 'url': url}
    try:
        response = requests.get('http://api.scraperapi.com', params=params, timeout=45)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

        config = Config()
        config.fetch_images = False # To speed up, images are not needed for text analysis
        config.memoize_articles = False # Avoid disk caching by newspaper itself
        article_scraper = Article(url, config=config)
        article_scraper.download(input_html=response.text)
        article_scraper.parse()

        if not article_scraper.text:
            app.logger.warning(f"Could not extract text from article URL: {url}")
            return {"full_text": None, "groq_analysis": None, "error": "Could not extract text from the article."}
        
        article_title_for_groq = article_scraper.title or MASTER_ARTICLE_STORE.get(article_hash_id, {}).get('title', 'Unknown Title')
        
        groq_analysis_result = None # Initialize

        # Check if Groq analysis already exists in MASTER_ARTICLE_STORE for this API article
        # This is a redundancy check; the route get_article_content_json already does this.
        # However, keeping it ensures consistency if this function were called from elsewhere.
        if article_hash_id in MASTER_ARTICLE_STORE and \
           MASTER_ARTICLE_STORE[article_hash_id].get('groq_summary') is not None and \
           MASTER_ARTICLE_STORE[article_hash_id].get('groq_takeaways') is not None:
            app.logger.info(f"Re-confirming pre-cached Groq analysis from MASTER_ARTICLE_STORE for {article_hash_id} within fetch_and_parse.")
            groq_analysis_result = {
                "groq_summary": MASTER_ARTICLE_STORE[article_hash_id]['groq_summary'],
                "groq_takeaways": MASTER_ARTICLE_STORE[article_hash_id]['groq_takeaways'],
                "error": None 
            }
        else:
            # If not in MASTER_ARTICLE_STORE or incomplete, generate it
            groq_analysis_result = get_article_analysis_with_groq(article_scraper.text, article_title_for_groq)
            # And cache it in MASTER_ARTICLE_STORE if successfully generated
            if article_hash_id in MASTER_ARTICLE_STORE and groq_analysis_result and not groq_analysis_result.get("error"):
                MASTER_ARTICLE_STORE[article_hash_id]['groq_summary'] = groq_analysis_result.get("groq_summary")
                MASTER_ARTICLE_STORE[article_hash_id]['groq_takeaways'] = groq_analysis_result.get("groq_takeaways")
                app.logger.info(f"Groq analysis generated and cached in MASTER_ARTICLE_STORE for API article ID: {article_hash_id}")
            elif groq_analysis_result and groq_analysis_result.get("error"):
                 app.logger.warning(f"Groq analysis for {article_hash_id} resulted in error: {groq_analysis_result.get('error')}")


        return {
            "full_text": article_scraper.text,
            "groq_analysis": groq_analysis_result, 
            "error": None # Overall error for this function; specific errors are in groq_analysis_result if they occurred there
        }
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Failed to fetch article content via proxy for {url}: {e}")
        return {"full_text": None, "groq_analysis": None, "error": f"Failed to fetch article content: {str(e)}"}
    except Exception as e: # Catches ArticleException from newspaper, and any other general errors during parsing/processing
        app.logger.error(f"Failed to parse or process article content for {url}: {e}", exc_info=True)
        return {"full_text": None, "groq_analysis": None, "error": f"Failed to parse or process article content: {str(e)}"}
    
# ==============================================================================
# --- 6. Flask Routes ---
# ==============================================================================
# Rev14.py - MODIFIED FOR ROBUST NEWS, COMMENTS, PROFILE, BOOKMARKS, AI DISPLAY FIXES, AND DATE FILTER (v2)

# ... (context_processor, get_paginated_articles, get_sort_key remain unchanged from previous response) ...
@app.context_processor
def inject_global_vars():
    return {'categories': app.config['CATEGORIES'],
            'current_year': datetime.utcnow().year,
            'session': session,
            'request': request,
            'groq_client': groq_client is not None}

def get_paginated_articles(articles, page, per_page):
    total = len(articles)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_items = articles[start:end]
    total_pages = (total + per_page - 1) // per_page if per_page > 0 else 0
    return paginated_items, total_pages

def get_sort_key(article):
    date_val = None
    if isinstance(article, dict): date_val = article.get('publishedAt')
    elif hasattr(article, 'published_at'): date_val = article.published_at
    if not date_val: return datetime.min.replace(tzinfo=timezone.utc)
    if isinstance(date_val, str):
        try:
            if date_val.endswith('Z'): date_val_dt = datetime.fromisoformat(date_val[:-1] + '+00:00')
            elif '+' in date_val or '-' in date_val[10:]: date_val_dt = datetime.fromisoformat(date_val)
            else: date_val_dt = datetime.fromisoformat(date_val).replace(tzinfo=timezone.utc)
            return date_val_dt
        except ValueError:
            app.logger.warning(f"Could not parse date string: {date_val}")
            return datetime.min.replace(tzinfo=timezone.utc)
    elif isinstance(date_val, datetime): return date_val if date_val.tzinfo else pytz.utc.localize(date_val)
    return datetime.min.replace(tzinfo=timezone.utc)


@app.route('/')
@app.route('/page/<int:page>')
@app.route('/category/<category_name>')
@app.route('/category/<category_name>/page/<int:page>')
def index(page=1, category_name='All Articles'):
    session['previous_list_page'] = request.full_path
    per_page = app.config['PER_PAGE']
    query_str = request.args.get('query')
    filter_date_str = request.args.get('filter_date')

    # This condition renders the special two-section homepage.
    if page == 1 and category_name == 'All Articles' and not query_str and not filter_date_str:
        app.logger.info("Rendering main homepage with Popular and Yesterday's Latest sections.")
        
        popular_articles = fetch_popular_news()
        latest_yesterday_articles = fetch_yesterdays_latest_news()

        POPULAR_NEWS_COUNT = 6
        LATEST_NEWS_COUNT = 9
        
        user_bookmarks_hashes = set()
        if 'user_id' in session:
            bookmarks = BookmarkedArticle.query.filter_by(user_id=session['user_id']).all()
            user_bookmarks_hashes = {b.article_hash_id for b in bookmarks}

        for art_list in [popular_articles, latest_yesterday_articles]:
            for art in art_list:
                art['is_bookmarked'] = art.get('id') in user_bookmarks_hashes

        return render_template("INDEX_HTML_TEMPLATE",
                               popular_articles=popular_articles[:POPULAR_NEWS_COUNT],
                               latest_yesterday_articles=latest_yesterday_articles[:LATEST_NEWS_COUNT],
                               selected_category=category_name,
                               is_main_homepage=True,
                               current_page=1,
                               total_pages=1,
                               query=None,
                               current_filter_date=None)

    # This 'else' block handles all other cases: Community Hub, pagination, date filters, etc.
    else:
        app.logger.info(f"Rendering standard list view for: category='{category_name}', page='{page}', filter='{filter_date_str}'")
        all_display_articles_raw = []

        if filter_date_str:
            try:
                datetime.strptime(filter_date_str, '%Y-%m-%d')
            except ValueError:
                flash("Invalid date format for filter.", "warning")
                filter_date_str = None

        if category_name == 'Community Hub':
            db_articles = CommunityArticle.query.options(joinedload(CommunityArticle.author)).order_by(CommunityArticle.published_at.desc()).all()
            for art in db_articles:
                art.is_community_article = True
                all_display_articles_raw.append(art)
        else:
            api_articles = fetch_news_from_api(target_date_str=filter_date_str)
            for art_dict in api_articles:
                art_dict['is_community_article'] = False
                all_display_articles_raw.append(art_dict)
        
        all_display_articles_raw.sort(key=get_sort_key, reverse=True)
        paginated_display_articles_raw, total_pages = get_paginated_articles(all_display_articles_raw, page, per_page)
        
        paginated_display_articles_with_bookmark_status = []
        user_bookmarks_hashes = set()
        if 'user_id' in session:
            bookmarks = BookmarkedArticle.query.filter_by(user_id=session['user_id']).all()
            user_bookmarks_hashes = {b.article_hash_id for b in bookmarks}

        for art_item in paginated_display_articles_raw:
            current_article_hash_id = None
            if hasattr(art_item, 'is_community_article') and art_item.is_community_article:
                current_article_hash_id = art_item.article_hash_id
                art_item.is_bookmarked = current_article_hash_id in user_bookmarks_hashes
                paginated_display_articles_with_bookmark_status.append(art_item)
            elif isinstance(art_item, dict):
                current_article_hash_id = art_item.get('id')
                art_item_copy = art_item.copy()
                art_item_copy['is_bookmarked'] = current_article_hash_id in user_bookmarks_hashes
                paginated_display_articles_with_bookmark_status.append(art_item_copy)
            else:
                paginated_display_articles_with_bookmark_status.append(art_item)
        
        featured_article_on_this_page = (page == 1 and category_name == 'All Articles' and not query_str and not filter_date_str and paginated_display_articles_with_bookmark_status)
        
        return render_template("INDEX_HTML_TEMPLATE",
                               articles=paginated_display_articles_with_bookmark_status,
                               selected_category=category_name,
                               is_main_homepage=False,
                               current_page=page,
                               total_pages=total_pages,
                               featured_article_on_this_page=featured_article_on_this_page,
                               current_filter_date=filter_date_str,
                               query=query_str)

# search_results route remains largely the same, ensuring current_filter_date=None is passed
@app.route('/search')
@app.route('/search/page/<int:page>')
def search_results(page=1):
    session['previous_list_page'] = request.full_path
    query_str = request.args.get('query', '').strip()
    per_page = app.config['PER_PAGE']

    if not query_str: return redirect(url_for('index'))
    app.logger.info(f"Search query: '{query_str}'")
    
    # Search should ideally search through all available articles, not just date-filtered ones.
    # For simplicity with MASTER_ARTICLE_STORE, we ensure it's populated with general news.
    # If MASTER_ARTICLE_STORE could be filtered by a date from a previous call, search might be limited.
    # To ensure search is broad, we can call fetch_news_from_api() without a date to populate MASTER_ARTICLE_STORE.
    if not MASTER_ARTICLE_STORE: # Or if you want to ensure it has the latest general set for searching
        fetch_news_from_api() # Fetches default 7 days

    api_results = []
    for art_id, art_data in MASTER_ARTICLE_STORE.items(): # Search all in-memory API articles
        if query_str.lower() in art_data.get('title', '').lower() or \
           query_str.lower() in art_data.get('description', '').lower():
            art_copy = art_data.copy()
            art_copy['is_community_article'] = False
            api_results.append(art_copy)

    community_db_articles_query = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter(
        db.or_(CommunityArticle.title.ilike(f'%{query_str}%'), CommunityArticle.description.ilike(f'%{query_str}%'))).order_by(CommunityArticle.published_at.desc())
    community_db_articles = []
    for art in community_db_articles_query.all():
        art.is_community_article = True
        community_db_articles.append(art)

    all_search_results_raw = api_results + community_db_articles
    all_search_results_raw.sort(key=get_sort_key, reverse=True)
    paginated_search_articles_raw, total_pages = get_paginated_articles(all_search_results_raw, page, per_page)

    paginated_search_articles_with_bookmark_status = []
    user_bookmarks_hashes = set()
    if 'user_id' in session:
        bookmarks = BookmarkedArticle.query.filter_by(user_id=session['user_id']).all()
        user_bookmarks_hashes = {b.article_hash_id for b in bookmarks}

    for art_item in paginated_search_articles_raw:
        current_article_hash_id = None
        if hasattr(art_item, 'is_community_article') and art_item.is_community_article:
            current_article_hash_id = art_item.article_hash_id
            art_item.is_bookmarked = current_article_hash_id in user_bookmarks_hashes
            paginated_search_articles_with_bookmark_status.append(art_item)
        elif isinstance(art_item, dict):
            current_article_hash_id = art_item.get('id')
            art_item_copy = art_item.copy()
            art_item_copy['is_bookmarked'] = current_article_hash_id in user_bookmarks_hashes
            paginated_search_articles_with_bookmark_status.append(art_item_copy)
        else:
            paginated_search_articles_with_bookmark_status.append(art_item)
            
    return render_template("INDEX_HTML_TEMPLATE",
                           articles=paginated_search_articles_with_bookmark_status,
                           selected_category=f"Search: {query_str}",
                           current_page=page,
                           total_pages=total_pages,
                           featured_article_on_this_page=False,
                           query=query_str,
                           current_filter_date=None) # Search results don't use date filter

@app.route('/article/<article_hash_id>')
def article_detail(article_hash_id):
    article_data, is_community_article, is_bookmarked = None, False, False
    previous_list_page = session.get('previous_list_page', url_for('index'))

    article_db = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter_by(article_hash_id=article_hash_id).first()
    if article_db:
        article_data, is_community_article = article_db, True
        if article_data.groq_takeaways:
            try: article_data.parsed_takeaways = json.loads(article_data.groq_takeaways)
            except json.JSONDecodeError: article_data.parsed_takeaways = []
    else:
        if not MASTER_ARTICLE_STORE: fetch_news_from_api()
        article_api_dict = MASTER_ARTICLE_STORE.get(article_hash_id)
        if article_api_dict:
            article_data, is_community_article = article_api_dict.copy(), False
        else:
            flash("Article not found.", "danger"); return redirect(previous_list_page)

    if 'user_id' in session:
        is_bookmarked = bool(BookmarkedArticle.query.filter_by(user_id=session['user_id'], article_hash_id=article_hash_id).first())

    # --- SIMPLIFIED & STABLE COMMENT HANDLING ---
    comment_data, total_comment_count = {}, 0
    
    # Define a base query for all comments on this article, loading authors efficiently.
    base_comments_query = None
    if is_community_article:
        base_comments_query = Comment.query.options(joinedload(Comment.author)).filter_by(community_article_id=article_data.id)
    else:
        base_comments_query = Comment.query.options(joinedload(Comment.author)).filter_by(api_article_hash_id=article_hash_id)

    # 1. Get ALL comments (including replies) to process reactions.
    all_comments_in_thread = base_comments_query.all()
    total_comment_count = len(all_comments_in_thread)
    all_comment_ids = {c.id for c in all_comments_in_thread}

    # 2. Fetch reaction data for all comments in a single batch.
    if all_comment_ids:
        for c_id in all_comment_ids: comment_data[c_id] = {'reactions': {}, 'user_reaction': None}
        
        reaction_counts = db.session.query(
            CommentVote.comment_id, CommentVote.vote_emoji, func.count(CommentVote.vote_emoji)
        ).filter(CommentVote.comment_id.in_(all_comment_ids)).group_by(CommentVote.comment_id, CommentVote.vote_emoji).all()
        for c_id, emoji, count in reaction_counts:
            if c_id in comment_data: comment_data[c_id]['reactions'][emoji] = count

        if 'user_id' in session:
            user_reactions = CommentVote.query.filter(CommentVote.comment_id.in_(all_comment_ids), CommentVote.user_id==session['user_id']).all()
            for vote in user_reactions:
                if vote.comment_id in comment_data: comment_data[vote.comment_id]['user_reaction'] = vote.vote_emoji

    # 3. Fetch only the TOP-LEVEL comments to start the template loop.
    # The 'replies' relationship in the model will handle fetching children.
    comments_for_template = base_comments_query.filter(Comment.parent_id.is_(None)).order_by(Comment.timestamp.asc()).all()

    if isinstance(article_data, dict): article_data['is_community_article'] = False
    elif article_data: article_data.is_community_article = True
            
    return render_template("ARTICLE_HTML_TEMPLATE", 
                           article=article_data, 
                           is_community_article=is_community_article, 
                           comments=comments_for_template, 
                           comment_data=comment_data,
                           total_comment_count=total_comment_count,
                           previous_list_page=previous_list_page, 
                           is_bookmarked=is_bookmarked)

@app.route('/get_article_content/<article_hash_id>')
def get_article_content_json(article_hash_id):
    if not MASTER_ARTICLE_STORE and not CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first(): fetch_news_from_api()
    article_data = MASTER_ARTICLE_STORE.get(article_hash_id)
    if not article_data or 'url' not in article_data: return jsonify({"error": "Article data or URL not found in API cache"}), 404
    if article_data.get('groq_summary') is not None and article_data.get('groq_takeaways') is not None:
        app.logger.info(f"Returning cached Groq analysis from MASTER_ARTICLE_STORE for API article ID: {article_hash_id}")
        return jsonify({"groq_analysis": {"groq_summary": article_data['groq_summary'], "groq_takeaways": article_data['groq_takeaways'], "error": None}, "error": None})
    processed_content = fetch_and_parse_article_content(article_hash_id, article_data['url'])
    return jsonify(processed_content)

@app.route('/add_comment/<article_hash_id>', methods=['POST'])
@login_required
def add_comment(article_hash_id):
    content = request.json.get('content', '').strip()
    parent_id = request.json.get('parent_id')
    if not content: return jsonify({"error": "Comment cannot be empty."}), 400
    user = User.query.get(session['user_id'])
    if not user: app.logger.error(f"User not found in add_comment for user_id {session.get('user_id')}"); return jsonify({"error": "User not found."}), 401
    new_comment = None
    community_article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
    if community_article: new_comment = Comment(content=content, user_id=user.id, community_article_id=community_article.id, parent_id=parent_id)
    elif article_hash_id in MASTER_ARTICLE_STORE: new_comment = Comment(content=content, user_id=user.id, api_article_hash_id=article_hash_id, parent_id=parent_id)
    else:
        fetch_news_from_api()
        if article_hash_id in MASTER_ARTICLE_STORE: new_comment = Comment(content=content, user_id=user.id, api_article_hash_id=article_hash_id, parent_id=parent_id)
        else: return jsonify({"error": "Article not found to comment on."}), 404
    db.session.add(new_comment); db.session.commit(); db.session.refresh(new_comment)
    author_name = new_comment.author.name if new_comment.author else "Anonymous"
    return jsonify({"success": True, "comment": {"id": new_comment.id, "content": new_comment.content, "timestamp": new_comment.timestamp.isoformat() + 'Z', "author": {"name": author_name }, "parent_id": new_comment.parent_id, "likes": 0, "dislikes": 0, "user_vote": 0}}), 201

@app.route('/vote_comment/<int:comment_id>', methods=['POST'])
@login_required
def vote_comment(comment_id):
    comment = Comment.query.get_or_404(comment_id)
    emoji = request.json.get('emoji')
    
    # Define the set of allowed emojis for reactions.
    allowed_emojis = ['👍', '❤️', '😂', '😮', '😢', '😠']
    if not emoji or emoji not in allowed_emojis:
        return jsonify({"error": "Invalid reaction."}), 400

    existing_vote = CommentVote.query.filter_by(user_id=session['user_id'], comment_id=comment_id).first()
    
    user_reaction_after_vote = None
    
    if existing_vote:
        # If the user clicks the same emoji again, it's an "un-react", so we delete the vote.
        if existing_vote.vote_emoji == emoji:
            db.session.delete(existing_vote)
            user_reaction_after_vote = None
        # If they click a different emoji, we update their existing vote.
        else:
            existing_vote.vote_emoji = emoji
            user_reaction_after_vote = emoji
    # If no vote exists from this user, create a new one.
    else:
        new_vote = CommentVote(user_id=session['user_id'], comment_id=comment_id, vote_emoji=emoji)
        db.session.add(new_vote)
        user_reaction_after_vote = emoji
        
    db.session.commit()

    # After any change, recalculate all reaction counts for this specific comment.
    reaction_counts_query = db.session.query(
        CommentVote.vote_emoji,
        func.count(CommentVote.vote_emoji)
    ).filter(CommentVote.comment_id == comment_id).group_by(CommentVote.vote_emoji).all()
    
    # Format the counts into a dictionary for the frontend.
    reactions = {emo: count for emo, count in reaction_counts_query}

    return jsonify({
        "success": True, 
        "reactions": reactions, 
        "user_reaction": user_reaction_after_vote
    }), 200

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
    new_article = CommunityArticle(article_hash_id=article_hash_id, title=title, description=description, full_text=content, source_name=source_name, image_url=image_url or f'https://via.placeholder.com/400x220/1E3A5E/FFFFFF?text={urllib.parse.quote_plus(title[:20])}', user_id=session['user_id'], published_at=datetime.now(timezone.utc), groq_summary=groq_summary_text, groq_takeaways=groq_takeaways_json_str)
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
        elif User.query.filter_by(username=username).first(): flash('Username already exists. Please choose another.', 'warning')
        else:
            new_user = User(name=name, username=username, password_hash=generate_password_hash(password))
            db.session.add(new_user); db.session.commit()
            flash(f'Registration successful, {name}! Please log in.', 'success')
            return redirect(url_for('login'))
        return redirect(url_for('register'))
    return render_template("REGISTER_HTML_TEMPLATE")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        username, password = request.form.get('username', '').strip().lower(), request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session.permanent = True; session['user_id'] = user.id; session['user_name'] = user.name
            flash(f"Welcome back, {user.name}!", "success")
            next_url = request.args.get('next')
            session.pop('previous_list_page', None) 
            return redirect(next_url or url_for('index'))
        else: flash('Invalid username or password.', 'danger')
    return render_template("LOGIN_HTML_TEMPLATE")

@app.route('/logout')
def logout(): session.clear(); flash("You have been successfully logged out.", "info"); return redirect(url_for('index'))
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
    elif Subscriber.query.filter_by(email=email).first(): flash('You are already subscribed to our newsletter.', 'info')
    else:
        try: db.session.add(Subscriber(email=email)); db.session.commit(); flash('Thank you for subscribing!', 'success')
        except Exception as e: db.session.rollback(); app.logger.error(f"Error subscribing email {email}: {e}"); flash('Could not subscribe at this time. Please try again later.', 'danger')
    return redirect(request.referrer or url_for('index'))

@app.route('/toggle_bookmark/<article_hash_id>', methods=['POST'])
@login_required
def toggle_bookmark(article_hash_id):
    user_id = session['user_id']
    is_community_str = request.json.get('is_community_article', 'false').lower()
    is_community = True if is_community_str == 'true' else False
    article_title_cache = request.json.get('title', 'Bookmarked Article')
    article_source_cache = request.json.get('source_name', 'Unknown Source')
    article_image_cache = request.json.get('image_url', None)
    article_desc_cache = request.json.get('description', None)
    article_published_at_cache_str = request.json.get('published_at', None)
    article_published_at_dt = None
    if article_published_at_cache_str:
        try:
            if article_published_at_cache_str.endswith('Z'): article_published_at_dt = datetime.fromisoformat(article_published_at_cache_str[:-1] + '+00:00')
            else: article_published_at_dt = datetime.fromisoformat(article_published_at_cache_str)
            if article_published_at_dt.tzinfo is None: article_published_at_dt = pytz.utc.localize(article_published_at_dt)
        except ValueError: app.logger.warning(f"Could not parse published_at_cache_str for bookmark: {article_published_at_cache_str}"); article_published_at_dt = None
    existing_bookmark = BookmarkedArticle.query.filter_by(user_id=user_id, article_hash_id=article_hash_id).first()
    if existing_bookmark:
        db.session.delete(existing_bookmark); db.session.commit()
        return jsonify({"success": True, "status": "removed", "message": "Bookmark removed."})
    else:
        if is_community:
            if not CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first(): return jsonify({"success": False, "error": "Community article not found."}), 404
        else:
            if article_hash_id not in MASTER_ARTICLE_STORE:
                fetch_news_from_api() 
                if article_hash_id not in MASTER_ARTICLE_STORE: return jsonify({"success": False, "error": "API article not found."}), 404
        new_bookmark = BookmarkedArticle(user_id=user_id, article_hash_id=article_hash_id, is_community_article=is_community, title_cache=article_title_cache, source_name_cache=article_source_cache, image_url_cache=article_image_cache, description_cache=article_desc_cache, published_at_cache=article_published_at_dt)
        db.session.add(new_bookmark); db.session.commit()
        return jsonify({"success": True, "status": "added", "message": "Article bookmarked!"})

@app.route('/profile')
@login_required
def profile():
    user = User.query.get_or_404(session['user_id'])
    page = request.args.get('page', 1, type=int)
    per_page = app.config['PER_PAGE']
    user_posted_articles = CommunityArticle.query.filter_by(user_id=user.id).order_by(CommunityArticle.published_at.desc()).all()
    bookmarks_query = BookmarkedArticle.query.filter_by(user_id=user.id).order_by(BookmarkedArticle.bookmarked_at.desc())
    user_bookmarks_paginated_query = bookmarks_query.paginate(page=page, per_page=per_page, error_out=False)
    user_bookmarked_articles_data = []
    for bookmark in user_bookmarks_paginated_query.items:
        article_detail_data = None
        if bookmark.is_community_article:
            comm_art = CommunityArticle.query.options(joinedload(CommunityArticle.author)).filter_by(article_hash_id=bookmark.article_hash_id).first()
            if comm_art: article_detail_data = {'id': comm_art.article_hash_id, 'title': comm_art.title, 'description': comm_art.description, 'urlToImage': comm_art.image_url, 'publishedAt': comm_art.published_at.isoformat() if comm_art.published_at else None, 'source': {'name': comm_art.author.name if comm_art.author else comm_art.source_name}, 'is_community_article': True, 'article_url': url_for('article_detail', article_hash_id=comm_art.article_hash_id)}
        else:
            api_art = MASTER_ARTICLE_STORE.get(bookmark.article_hash_id)
            if api_art: article_detail_data = {'id': api_art['id'], 'title': api_art['title'], 'description': api_art['description'], 'urlToImage': api_art['urlToImage'], 'publishedAt': api_art['publishedAt'], 'source': {'name': api_art['source']['name']}, 'is_community_article': False, 'article_url': url_for('article_detail', article_hash_id=api_art['id'])}
            else: article_detail_data = {'id': bookmark.article_hash_id, 'title': bookmark.title_cache or "Bookmarked Article (Details N/A)", 'description': bookmark.description_cache or "Description not available.", 'urlToImage': bookmark.image_url_cache or f'https://via.placeholder.com/400x220/CCCCCC/000000?text=Preview+N/A', 'publishedAt': bookmark.published_at_cache.isoformat() if bookmark.published_at_cache else None, 'source': {'name': bookmark.source_name_cache or "Unknown Source"}, 'is_community_article': False, 'article_url': url_for('article_detail', article_hash_id=bookmark.article_hash_id), 'is_stale_bookmark': True}
        if article_detail_data: user_bookmarked_articles_data.append(article_detail_data)
    return render_template("PROFILE_HTML_TEMPLATE", user=user, posted_articles=user_posted_articles, bookmarked_articles=user_bookmarked_articles_data, bookmarks_pagination=user_bookmarks_paginated_query, current_page=page)

@app.errorhandler(404)
def page_not_found(e): return render_template("404_TEMPLATE"), 404
@app.errorhandler(500)
def internal_server_error(e): db.session.rollback(); app.logger.error(f"500 error at {request.url}: {e}", exc_info=True); return render_template("500_TEMPLATE"), 500

@app.route('/ads.txt')
def ads_txt():
    # Ensure this is your correct AdSense Publisher ID
    ads_content = "google.com, pub-6975904325280886, DIRECT, f08c47fec0942fa0"
    # If you have other ad partners, add their lines here, each on a new line.
    # e.g., ads_content += "\notheradsystem.com, theirPubId, DIRECT, theirTagId"
    return Response(ads_content, mimetype='text/plain')
# ==============================================================================
# --- 7. HTML Templates (Stored in memory) ---
# ==============================================================================
# --- File: Rev14.py ---

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
        .main-content { flex-grow: 1; }
        body.dark-mode {
            --primary-color: #6366F1; --primary-light: #818CF8; --primary-dark: #4F46E5; --secondary-color: #2DD4BF; --secondary-light: #5EEAD4; --accent-color: #FB923C; --text-color: #F9FAFB; --text-muted-color: #9CA3AF; --light-bg: #111827; --card-bg: #1F2937; --card-border-color: #374151; --footer-bg: #000000; --footer-text: #9CA3AF;
            --primary-color-rgb: 99, 102, 241; --secondary-color-rgb: 45, 212, 191;
            --bookmark-active-color: var(--secondary-light);
        }
        h1, h2, h3, h4, h5, .auth-title, .profile-card h2, .article-title-main, .modal-title { font-family: 'Poppins', sans-serif; font-weight: 700; }
        .alert-top { position: fixed; top: 110px; left: 50%; transform: translateX(-50%); z-index: 2050; min-width:320px; text-align:center; box-shadow: var(--shadow-lg); border-radius: var(--border-radius-md); }
        
        /* --- HEADER LAYOUT FIX --- */
        .navbar-main { background-color: var(--primary-color); padding: 0; box-shadow: var(--shadow-md); transition: background-color 0.3s ease; height: 95px; }
        .navbar-content-wrapper { position: relative; display: flex; justify-content: space-between; align-items: center; width: 100%; height: 100%; }
        .navbar-brand-custom { color: white !important; font-weight: 700; font-size: 2rem; font-family: 'Poppins', sans-serif; display: flex; align-items: center; gap: 10px; text-decoration: none !important; }
        .navbar-brand-custom .brand-icon { color: var(--secondary-light); font-size: 2.2rem; }
        .search-form-container { position: absolute; left: 50%; transform: translateX(-50%); width: 45%; max-width: 550px; }
        .search-container { position: relative; width: 100%; }
        .navbar-search { border-radius: 50px; padding: 0.7rem 1.25rem 0.7rem 2.8rem; border: 1px solid transparent; font-size: 0.95rem; transition: all 0.3s ease; background: rgba(255,255,255,0.15); color: white; }
        .navbar-search::placeholder { color: rgba(255,255,255,0.7); }
        .navbar-search:focus { background: rgba(255,255,255,0.25); box-shadow: 0 0 0 4px rgba(255,255,255,0.2); border-color: var(--secondary-light); outline: none; color:white; }
        .search-icon { color: rgba(255,255,255,0.8); transition: all 0.3s ease; left: 1.1rem; position: absolute; top: 50%; transform: translateY(-50%); }
        .header-controls { display: flex; gap: 0.8rem; align-items: center; }
        .header-btn { background: transparent; border: 1px solid rgba(255,255,255,0.4); padding: 0.5rem 1rem; border-radius: 50px; color: white; font-weight: 500; transition: all 0.3s ease; display: flex; align-items: center; gap: 0.5rem; cursor: pointer; text-decoration:none; font-size: 0.9rem; }
        .header-btn:hover { background: rgba(255,255,255,0.9); border-color: transparent; color: var(--primary-dark); }
        .dark-mode-toggle { font-size: 1.1rem; width: 42px; height: 42px; justify-content: center;}
        
        /* Category Nav */
        .category-nav { background: var(--card-bg); box-shadow: var(--shadow-sm); position: fixed; top: 95px; width: 100%; z-index: 1020; border-bottom: 1px solid var(--card-border-color); transition: background-color 0.3s ease, border-bottom-color 0.3s ease; }
        .categories-wrapper { display: flex; justify-content: center; align-items: center; width: 100%; overflow-x: auto; padding: 0.4rem 0.5rem; scrollbar-width: none; }
        .categories-wrapper::-webkit-scrollbar { display: none; }
        .category-links-container { display: flex; flex-shrink: 0; }
        .category-link { color: var(--text-muted-color) !important; font-weight: 600; padding: 0.6rem 1.3rem !important; border-radius: 50px; transition: all 0.25s ease; white-space: nowrap; text-decoration: none; margin: 0 0.3rem; font-size: 0.9rem; border: 1px solid transparent; }
        .category-link.active { background: var(--primary-color) !important; color: white !important; box-shadow: var(--shadow-sm); }
        .category-link:hover:not(.active) { background: var(--light-bg) !important; color: var(--primary-color) !important; }
        body.dark-mode .category-link.active { color: var(--card-bg) !important; }
        body.dark-mode .category-link:hover:not(.active) { background: var(--light-bg) !important; color: var(--primary-light) !important; }

        /* Cards */
        .article-card, .featured-article, .article-full-content-wrapper, .auth-container, .static-content-wrapper, .profile-card { background: var(--card-bg); border-radius: var(--border-radius-lg); transition: all 0.3s ease; border: 1px solid var(--card-border-color); box-shadow: var(--shadow-md); }
        .article-card:hover, .featured-article:hover { transform: translateY(-5px); box-shadow: var(--shadow-lg); }
        .article-image-container { height: 220px; overflow: hidden; position: relative; border-top-left-radius: var(--border-radius-lg); border-top-right-radius: var(--border-radius-lg);}
        .article-image { width: 100%; height: 100%; object-fit: cover; transition: transform 0.4s ease; }
        .article-card:hover .article-image { transform: scale(1.05); }
        .article-body { padding: 1.5rem; flex-grow: 1; display: flex; flex-direction: column; }
        .article-title { font-weight: 600; line-height: 1.4; margin-bottom: 0.6rem; font-size:1.15rem; }
        .article-title a { color: var(--text-color); text-decoration: none; transition: color 0.2s ease; }
        .article-card:hover .article-title a { color: var(--primary-color) !important; }
        .article-meta { display: flex; align-items: center; margin-bottom: 0.8rem; flex-wrap: wrap; gap: 0.4rem 1rem; }
        .meta-item { display: flex; align-items: center; font-size: 0.8rem; color: var(--text-muted-color); }
        .meta-item i { font-size: 0.9rem; margin-right: 0.4rem; color: var(--secondary-color); }
        .article-description { color: var(--text-muted-color); margin-bottom: 1.25rem; font-size: 0.95rem; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
        .read-more { margin-top: auto; background: var(--primary-color); color: white !important; border: none; padding: 0.6rem 0; border-radius: var(--border-radius-md); font-weight: 600; font-size: 0.9rem; transition: all 0.3s ease; width: 100%; text-align: center; text-decoration: none; display:inline-block; }
        .read-more:hover { background: var(--primary-dark); transform: translateY(-2px); color: white !important; box-shadow: var(--shadow-md); }
        body.dark-mode .read-more { color: var(--card-bg) !important; }
        
        /* Pagination */
        .page-item .page-link { border-radius: 50%; width: 40px; height: 40px; display:flex; align-items:center; justify-content:center; color: var(--text-muted-color); background-color: var(--card-bg); border: 1px solid var(--card-border-color); font-weight: 600; transition: all 0.2s ease; font-size:0.9rem; margin: 0 0.2rem;}
        .page-item .page-link:hover { border-color: var(--primary-light); color: var(--primary-color); }
        .page-item.active .page-link { background-color: var(--primary-color); border-color: var(--primary-color); color: white; box-shadow: 0 2px 8px rgba(var(--primary-color-rgb), 0.4); }
        .page-item.disabled .page-link { color: var(--text-muted-color); pointer-events: none; background-color: var(--light-bg); }
        body.dark-mode .page-item.disabled .page-link { background-color: var(--card-bg); }
        .page-link-prev-next .page-link { width: auto; padding-left:1.2rem; padding-right:1.2rem; border-radius:50px; }
        
        /* Footer */
        footer { background: var(--footer-bg); color: var(--footer-text); margin-top: auto; padding: 3.5rem 0 1.5rem; font-size:0.9rem; }
        .footer-section h5 { color: white; margin-bottom: 1.2rem; font-weight: 600; letter-spacing: 0.3px; font-size: 1.1rem; }
        .footer-links { display: flex; flex-direction: column; gap: 0.8rem; }
        .footer-links a { color: var(--footer-text); text-decoration: none; transition: all 0.2s ease; }
        .footer-links a:hover { color: var(--footer-link-hover); padding-left: 5px; }
        .social-links { display: flex; gap: 1rem; margin-top: 0.5rem; }
        .social-links a { color: var(--footer-text); font-size: 1.2rem; transition: all 0.2s ease; }
        .social-links a:hover { color: var(--secondary-light); transform: translateY(-2px); }
        .copyright { text-align: center; padding-top: 2rem; margin-top: 2rem; border-top: 1px solid #374151; font-size: 0.85rem; color: var(--text-muted-color); }
        body.dark-mode .copyright { border-top-color: var(--card-border-color); color: var(--footer-text); }
        
        /* Modal & FAB */
        .admin-controls { position: fixed; bottom: 25px; right: 25px; z-index: 1030; }
        .add-article-btn { width: 60px; height: 60px; border-radius: 50%; background: var(--primary-color); color: white; border: none; box-shadow: var(--shadow-lg); display: flex; align-items: center; justify-content: center; font-size: 26px; cursor: pointer; transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); }
        .add-article-btn:hover { transform: translateY(-4px) scale(1.05); background: var(--primary-light); }
        .add-article-modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 2000; background-color: rgba(0, 0, 0, 0.6); backdrop-filter: blur(5px); align-items: center; justify-content: center; }
        .modal-content { border-radius: var(--border-radius-lg); border: none; }
        .close-modal { position: absolute; top: 12px; right: 12px; font-size: 20px; color: var(--text-muted-color); background: none; border: none; cursor: pointer; width: 32px; height: 32px; border-radius: 50%; display: flex; align-items: center; justify-content: center; transition: all 0.2s ease; }
        .close-modal:hover { background: var(--light-bg); color: var(--text-color); }
        .modal-form-control { border-radius: var(--border-radius-md); border: 1px solid var(--card-border-color); font-size: 0.95rem; transition: all 0.2s ease; background-color: var(--light-bg); }
        .modal-form-control:focus { border-color: var(--primary-color); box-shadow: 0 0 0 3px rgba(var(--primary-color-rgb),0.2); outline: none; background-color: var(--card-bg); }
        .btn-primary-modal { background-color: var(--primary-color); border-color: var(--primary-color); color:white; padding: 0.7rem 1.4rem; font-weight:600; border-radius: var(--border-radius-md); }
        .btn-primary-modal:hover { background-color: var(--primary-dark); border-color: var(--primary-dark); }
        
        /* Auth pages */
        .auth-container { max-width: 450px; margin: 3rem auto; padding: 2.5rem; }
        
        /* --- NEW Comment Section Styles (Replaces old comment CSS) --- */
        .comment-thread { position: relative; animation: fadeIn 0.4s ease; }
        .comment-container { display: flex; gap: 1rem; align-items: flex-start; }
        #comments-list > .comment-thread + .comment-thread { margin-top: 1.75rem; padding-top: 1.75rem; border-top: 1px solid var(--card-border-color); }
        .comment-replies { margin-top: 1rem; margin-left: calc(45px + 1rem); padding-left: 1.25rem; border-left: 2px solid var(--card-border-color); }
        .comment-replies > .comment-thread + .comment-thread { margin-top: 1.25rem; padding-top: 1.25rem; border-top: 1px dashed var(--card-border-color); }
        .comment-avatar { width: 45px; height: 45px; border-radius: 50%; background: var(--primary-light); color: white; display: flex; align-items: center; justify-content: center; font-weight: 600; flex-shrink: 0; }
        .comment-replies .comment-avatar { width: 40px; height: 40px; }
        body.dark-mode .comment-avatar { background: var(--primary-dark); }
        .comment-body { flex-grow: 1; }
        .comment-author { font-weight: 600; color: var(--text-color); }
        .comment-date { font-size: 0.8rem; color: var(--text-muted-color); }
        .comment-header { display: flex; align-items: baseline; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.25rem; }
        .comment-content { word-wrap: break-word; }
        .comment-actions { position: relative; display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem; }
        .comment-actions button { background: none; border: none; color: var(--text-muted-color); padding: 0.25rem 0.5rem; border-radius: var(--border-radius-md); font-size: 0.85rem; font-weight: 500; display: flex; align-items: center; gap: 0.3rem; transition: all 0.2s ease; }
        .comment-actions button:hover { color: var(--primary-color); background-color: rgba(var(--primary-color-rgb), 0.1); }
        body.dark-mode .comment-actions button:hover { color: var(--primary-light); }
        .react-btn { position: relative; }
        .reaction-box { display: none; position: absolute; bottom: 100%; left: 0; margin-bottom: 8px; background-color: var(--card-bg); border: 1px solid var(--card-border-color); border-radius: 50px; padding: 4px 8px; box-shadow: var(--shadow-md); z-index: 10; white-space: nowrap; animation: fadeInUp 0.2s ease-out; }
        .reaction-box.show { display: flex; gap: 5px; }
        .reaction-emoji { font-size: 1.4rem; cursor: pointer; transition: transform 0.15s cubic-bezier(0.215, 0.610, 0.355, 1); padding: 2px; }
        .reaction-emoji:hover { transform: scale(1.25); }
        .reaction-summary { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 12px; }
        .reaction-pill { display: flex; align-items: center; background-color: rgba(var(--primary-color-rgb), 0.08); border: 1px solid transparent; border-radius: 20px; padding: 2px 8px; font-size: 0.8rem; font-weight: 500; cursor: default; transition: all 0.2s ease; }
        .reaction-pill.user-reacted { background-color: var(--primary-color); color: white; border-color: var(--primary-dark); box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        body.dark-mode .reaction-pill.user-reacted { background-color: var(--primary-light); color: var(--footer-bg); border-color: var(--primary-color); }
        .reaction-pill .emoji { font-size: 0.9rem; margin-right: 4px; }
        .reply-form-container { display: none; padding: 1rem; border-radius: var(--border-radius-md); margin-top: 0.75rem; background-color: var(--light-bg); border: 1px solid var(--card-border-color); }
        body.dark-mode .reply-form-container { background-color: var(--footer-bg); }
        /* --- END NEW Comment Section Styles --- */

        /* Profile Page */
        .profile-card .profile-avatar { background-color: var(--primary-color); }
        body.dark-mode .profile-card .profile-avatar { background-color: var(--primary-dark); }
        .profile-tabs .nav-link { color: var(--text-muted-color); font-weight: 600; }
        .profile-tabs .nav-link.active { color: var(--primary-color); border-bottom: 3px solid var(--primary-color); background: transparent; }
        body.dark-mode .profile-tabs .nav-link.active { color: var(--primary-light); border-bottom-color: var(--primary-light); }
        
        /* Bookmark Button */
        .bookmark-btn { background: none; border: none; font-size: 1.6rem; color: var(--text-muted-color); cursor: pointer; padding: 0.25rem 0.5rem; transition: all 0.2s ease; vertical-align: middle; }
        .bookmark-btn.active { color: var(--bookmark-active-color); transform: scale(1.1); }
        .bookmark-btn:hover { color: var(--secondary-light); }
        .article-card .bookmark-btn { font-size: 1.3rem; }
        
        /* --- RESPONSIVE HEADER LAYOUT --- */
        @media (max-width: 991.98px) {
            body { padding-top: 180px; }
            .navbar-main { padding: 1rem 0 0.5rem; height: auto; }
            .navbar-content-wrapper { position: static; flex-direction: column; align-items: flex-start; gap: 0.75rem; height: auto; }
            .navbar-brand-custom { margin-bottom: 0.5rem; }
            .search-form-container { position: static; transform: none; width: 100%; order: 3; }
            .header-controls { position: absolute; top: 1.2rem; right: 1rem; }
            .category-nav { top: 130px; }
            .categories-wrapper { justify-content: flex-start; }
            #dateFilterForm { width: 100%; margin-left: 0 !important; margin-top: 0.5rem; }
        }
        @media (max-width: 767.98px) { body { padding-top: 170px; } .category-nav { top: 120px; } .featured-article .row { flex-direction: column; } .featured-image { margin-bottom: 1rem; height: 250px; } }
        @media (max-width: 575.98px) { .navbar-brand-custom { font-size: 1.8rem;} .header-controls { gap: 0.3rem; } .header-btn { padding: 0.4rem 0.8rem; font-size: 0.8rem; } .dark-mode-toggle { font-size: 1rem; } .comment-replies { margin-left: 1rem; padding-left: 1rem; } }
        
        /* Animations */
        .animate-fade-in { animation: fadeIn 0.5s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(25px); } to { opacity: 1; transform: translateY(0); } }
    </style>
    {% block head_extra %}{% endblock %}    
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-6975904325280886" crossorigin="anonymous"></script>
    <script async src="https://www.googletagmanager.com/gtag/js?id=G-CV5LWJ7NQ7"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){dataLayer.push(arguments);}
      gtag('js', new Date());
      gtag('config', 'G-CV5LWJ7NQ7');
    </script>
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
                    <form action="{{ url_for('search_results') }}" method="GET" class="search-container">
                        <input type="search" name="query" class="form-control navbar-search" placeholder="Search news articles..." value="{{ request.args.get('query', '') }}">
                        <i class="fas fa-search search-icon"></i>
                        <button type="submit" class="d-none">Search</button>
                    </form>
                </div>
                <div class="header-controls">
                    <button class="header-btn dark-mode-toggle" aria-label="Toggle dark mode" title="Toggle Dark Mode">
                        <i class="fas fa-moon"></i>
                    </button>
                    {% if session.user_id %}
                    <div class="dropdown">
                        <button class="header-btn dropdown-toggle" type="button" id="userDropdown" data-bs-toggle="dropdown" aria-expanded="false" title="User Menu">
                            <i class="fas fa-user-circle"></i> <span class="d-none d-md-inline">Hi, {{ session.user_name|truncate(15) }}!</span>
                        </button>
                        <ul class="dropdown-menu dropdown-menu-end" aria-labelledby="userDropdown">
                            <li><a class="dropdown-item" href="{{ url_for('profile') }}"><i class="fas fa-id-card me-2"></i>Profile</a></li>
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
                <div class="category-links-container">
                {% for cat_item in categories %}
                    {% set cat_url_params = {'category_name': cat_item, 'page': 1} %}
                    {% if cat_item == 'All Articles' and selected_category == 'All Articles' and request.args.get('filter_date') %}
                        {% set _ = cat_url_params.update({'filter_date': request.args.get('filter_date')}) %}
                    {% endif %}
                    <a href="{{ url_for('index', **cat_url_params) }}" class="category-link {% if selected_category == cat_item %}active{% endif %}">
                        <i class="fas fa-{% if cat_item == 'All Articles' %}globe-americas{% elif cat_item == 'Community Hub' %}users{% endif %} me-1 d-none d-sm-inline"></i>
                        {{ cat_item }}
                    </a>
                {% endfor %}
                </div>
                
                <form id="dateFilterForm" class="ms-2 ms-md-3" style="min-width: 180px;">
                    <label for="articleDateFilter" class="visually-hidden">Filter articles by date</label>
                    <div class="input-group input-group-sm">
                        <input type="date" id="articleDateFilter" class="form-control form-control-sm" 
                               value="{{ current_filter_date | default('', true) }}" 
                               aria-label="Filter by date for All Articles"
                               title="Filter 'All Articles' by date">
                        <button class="btn btn-outline-secondary btn-sm" type="submit" title="Apply Date Filter" style="padding-left: 0.5rem; padding-right: 0.5rem;">Go</button>
                        {% if current_filter_date %}
                        <button class="btn btn-outline-danger btn-sm" type="button" id="clearDateFilter" title="Clear Date Filter"><i class="fas fa-times"></i></button>
                        {% endif %}
                    </div>
                </form>
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
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content p-4">
                <div class="modal-header border-0 pb-0">
                    <h4 class="modal-title">Post New Article</h4>
                    <button type="button" class="btn-close" id="closeModalBtn" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <form id="addArticleForm" action="{{ url_for('post_article') }}" method="POST">
                        <div class="mb-3"><label for="articleTitle" class="form-label">Article Title</label><input type="text" id="articleTitle" name="title" class="form-control modal-form-control" required></div>
                        <div class="mb-3"><label for="articleDescription" class="form-label">Short Description</label><textarea id="articleDescription" name="description" class="form-control modal-form-control" rows="3" required></textarea></div>
                        <div class="mb-3"><label for="articleSource" class="form-label">Source Name</label><input type="text" id="articleSource" name="sourceName" class="form-control modal-form-control" value="Community Post" required></div>
                        <div class="mb-3"><label for="articleImage" class="form-label">Image URL (Optional)</label><input type="url" id="articleImage" name="imageUrl" class="form-control modal-form-control"></div>
                        <div class="mb-3"><label for="articleContent" class="form-label">Full Article Content</label><textarea id="articleContent" name="content" class="form-control modal-form-control" rows="7" required></textarea></div>
                        <div class="d-flex justify-content-end gap-2 mt-4"><button type="button" class="btn btn-outline-secondary" id="cancelArticleBtn">Cancel</button><button type="submit" class="btn btn-primary-modal">Post Article</button></div>
                    </form>
                </div>
            </div>
        </div>
    </div>
    {% endif %}

    <footer class="mt-auto">
        <div class="container">
            <div class="footer-content row">
                <div class="footer-section col-lg-4 col-md-6 mb-4">
                    <div class="d-flex align-items-center mb-2">
                        <i class="fas fa-bolt-lightning footer-brand-icon me-2" style="color:var(--secondary-light);"></i>
                        <span class="h5 mb-0" style="color:white; font-family: 'Poppins', sans-serif;">Briefly</span>
                    </div>
                    <p class="small text-muted">Your premier source for AI summarized, India-centric news.</p>
                    <div class="social-links">
                        <a href="#" title="Twitter"><i class="fab fa-twitter"></i></a><a href="#" title="Facebook"><i class="fab fa-facebook-f"></i></a><a href="#" title="LinkedIn"><i class="fab fa-linkedin-in"></i></a><a href="#" title="Instagram"><i class="fab fa-instagram"></i></a>
                    </div>
                </div>
                <div class="footer-section col-lg-2 col-md-6 mb-4">
                    <h5>Quick Links</h5>
                    <div class="footer-links">
                        <a href="{{ url_for('index') }}">Home</a>
                        <a href="{{ url_for('about') }}">About Us</a>
                        <a href="{{ url_for('contact') }}">Contact</a>
                        <a href="{{ url_for('privacy') }}">Privacy Policy</a>
                        {% if session.user_id %}<a href="{{ url_for('profile') }}">My Profile</a>{% endif %}
                    </div>
                </div>
                <div class="footer-section col-lg-2 col-md-6 mb-4">
                    <h5>Categories</h5>
                    <div class="footer-links">
                        {% for cat_item in categories %}<a href="{{ url_for('index', category_name=cat_item, page=1) }}">{{ cat_item }}</a>{% endfor %}
                    </div>
                </div>
                <div class="footer-section col-lg-4 col-md-6 mb-4">
                    <h5>Newsletter</h5>
                    <p class="small text-muted">Subscribe for weekly updates!</p>
                    <form action="{{ url_for('subscribe') }}" method="POST" class="mt-3">
                        <div class="input-group">
                            <input type="email" name="email" class="form-control form-control-sm" placeholder="Your Email" aria-label="Your Email" required style="background: #374151; border-color: #4B5563; color: white;">
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
        if (!storedTheme) { const cookieTheme = document.cookie.split('; ').find(row => row.startsWith('darkMode='))?.split('=')[1]; if (cookieTheme) storedTheme = cookieTheme; }
        if (storedTheme) { applyTheme(storedTheme); } else { updateThemeIcon(); }
        
        const addArticleBtn = document.getElementById('addArticleBtn');
        const addArticleModalEl = document.getElementById('addArticleModal');
        if (addArticleBtn && addArticleModalEl) {
            const addArticleModal = new bootstrap.Modal(addArticleModalEl);
            addArticleBtn.addEventListener('click', () => addArticleModal.show());
            document.getElementById('closeModalBtn').addEventListener('click', () => addArticleModal.hide());
            document.getElementById('cancelArticleBtn').addEventListener('click', () => addArticleModal.hide());
            addArticleModalEl.addEventListener('hidden.bs.modal', () => {
                const form = document.getElementById('addArticleForm');
                if (form) form.reset();
            });
        }
        
        const flashedAlerts = document.querySelectorAll('#alert-placeholder .alert');
        flashedAlerts.forEach(function(alert) { setTimeout(function() { const bsAlert = bootstrap.Alert.getOrCreateInstance(alert); if (bsAlert) bsAlert.close(); }, 7000); });

        const dateFilterForm = document.getElementById('dateFilterForm');
        if (dateFilterForm) {
            dateFilterForm.addEventListener('submit', function(event) {
                event.preventDefault();
                const dateInput = document.getElementById('articleDateFilter');
                const selectedDate = dateInput.value;
                let targetUrl = new URL("{{ url_for('index', category_name='All Articles') }}", window.location.origin);
                if (selectedDate) {
                    targetUrl.searchParams.set('filter_date', selectedDate);
                }
                window.location.href = targetUrl.toString();
            });

            const clearDateFilterBtn = document.getElementById('clearDateFilter');
            if (clearDateFilterBtn) {
                clearDateFilterBtn.addEventListener('click', function() {
                    window.location.href = "{{ url_for('index', category_name='All Articles') }}";
                });
            }
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
    {% if query %}Search: {{ query|truncate(30) }}
    {% elif selected_category == 'All Articles' and current_filter_date %}Articles for {{ current_filter_date }}
    {% elif selected_category and not is_main_homepage %}{{selected_category}}
    {% else %}Popular & Latest News from India{% endif %} - BrieflyAI
{% endblock %}

{% block content %}

{# Use the flag from the backend to decide which layout to show #}
{% if is_main_homepage %}

    {# ============== LAYOUT 1: MAIN HOMEPAGE (TWO SECTIONS) ============== #}

    <div class="mb-5 animate-fade-in">
        <h2 class="pb-2 mb-4" style="font-weight: 700; border-bottom: 2px solid var(--card-border-color);">
            <i class="fas fa-fire-alt text-danger me-2"></i>Popular Stories
        </h2>
        <div class="row g-4">
            {% if popular_articles %}
                {% for art in popular_articles %}
                    <div class="col-md-6 col-lg-4 d-flex">
                        <article class="article-card d-flex flex-column w-100">
                            {% set article_url = url_for('article_detail', article_hash_id=art.id) %}
                            <div class="article-image-container">
                                <a href="{{ article_url }}">
                                <img src="{{ art.urlToImage }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a>
                            </div>
                            <div class="article-body d-flex flex-column">
                                <div class="d-flex justify-content-between align-items-start">
                                    <h5 class="article-title mb-2 flex-grow-1"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                                    {% if session.user_id %}
                                    <button class="bookmark-btn homepage-bookmark-btn {% if art.is_bookmarked %}active{% endif %}" style="margin-left: 10px; padding-top:0;"
                                            title="{% if art.is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}" data-article-hash-id="{{ art.id }}"
                                            data-is-community="false" data-title="{{ art.title|e }}" data-source-name="{{ art.source.name|e }}"
                                            data-image-url="{{ art.urlToImage|e }}" data-description="{{ (art.description if art.description else '')|e }}"
                                            data-published-at="{{ (art.publishedAt if art.publishedAt else '')|e }}">
                                        <i class="fa-solid fa-bookmark"></i>
                                    </button>
                                    {% endif %}
                                </div>
                                <div class="article-meta small mb-2">
                                    <span class="meta-item text-muted"><i class="fas fa-building"></i> {{ art.source.name|truncate(20) }}</span>
                                    <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ (art.publishedAt | to_ist if art.publishedAt else 'N/A') }}</span>
                                </div>
                                <p class="article-description small">{{ art.description|truncate(100) }}</p>
                                <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
                            </div>
                        </article>
                    </div>
                {% endfor %}
            {% else %}
                <div class="col-12"><div class="alert alert-light text-center">Popular stories are currently unavailable. Please check back shortly.</div></div>
            {% endif %}
        </div>
    </div>

    <div class="mb-4 animate-fade-in" style="animation-delay: 0.2s;">
        <h2 class="pb-2 mb-4" style="font-weight: 700; border-bottom: 2px solid var(--card-border-color);">
             <i class="fas fa-history me-2"></i>Yesterday's Headlines
        </h2>
        <div class="row g-4">
             {% if latest_yesterday_articles %}
                {% for art in latest_yesterday_articles %}
                    <div class="col-md-6 col-lg-4 d-flex">
                        <article class="article-card d-flex flex-column w-100">
                             {% set article_url = url_for('article_detail', article_hash_id=art.id) %}
                            <div class="article-image-container">
                                <a href="{{ article_url }}">
                                <img src="{{ art.urlToImage }}" class="article-image" alt="{{ art.title|truncate(50) }}"></a>
                            </div>
                            <div class="article-body d-flex flex-column">
                                <div class="d-flex justify-content-between align-items-start">
                                    <h5 class="article-title mb-2 flex-grow-1"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                                    {% if session.user_id %}
                                    <button class="bookmark-btn homepage-bookmark-btn {% if art.is_bookmarked %}active{% endif %}" style="margin-left: 10px; padding-top:0;"
                                            title="{% if art.is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}" data-article-hash-id="{{ art.id }}"
                                            data-is-community="false" data-title="{{ art.title|e }}" data-source-name="{{ art.source.name|e }}"
                                            data-image-url="{{ art.urlToImage|e }}" data-description="{{ (art.description if art.description else '')|e }}"
                                            data-published-at="{{ (art.publishedAt if art.publishedAt else '')|e }}">
                                        <i class="fa-solid fa-bookmark"></i>
                                    </button>
                                    {% endif %}
                                </div>
                                <div class="article-meta small mb-2">
                                    <span class="meta-item text-muted"><i class="fas fa-building"></i> {{ art.source.name|truncate(20) }}</span>
                                    <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ (art.publishedAt | to_ist if art.publishedAt else 'N/A') }}</span>
                                </div>
                                <p class="article-description small">{{ art.description|truncate(100) }}</p>
                                <a href="{{ article_url }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
                            </div>
                        </article>
                    </div>
                {% endfor %}
            {% else %}
                <div class="col-12"><div class="alert alert-light text-center">Could not load yesterday's articles.</div></div>
            {% endif %}
        </div>
    </div>
    
{% else %}

    {# ============ LAYOUT 2: STANDARD PAGINATED LIST (OLD BEHAVIOR) ============ #}
    
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
    {% elif not articles and selected_category == 'All Articles' and current_filter_date %}
        <div class="alert alert-info text-center my-4 p-3 small">No articles found for <strong>{{ current_filter_date }}</strong>. Please try a different date or clear the date filter.</div>
    {% elif not articles and selected_category != 'Community Hub' and not query %}
        <div class="alert alert-warning text-center my-4 p-3 small">No recent Indian news found. Please check back later.</div>
    {% elif not articles and selected_category == 'Community Hub' %}
        <div class="alert alert-info text-center my-4 p-3"><h4><i class="fas fa-feather-alt me-2"></i>No Articles Penned Yet</h4><p>No articles in the Community Hub. {% if session.user_id %}Click the '+' button to share your insights!{% else %}Login to add articles.{% endif %}</p></div>
    {% elif not articles and query %}
        <div class="alert alert-info text-center my-5 p-4"><h4><i class="fas fa-search me-2"></i>No results for "{{ query }}"</h4><p>Try different keywords or browse categories.</p></div>
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
                            title="{% if art.is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}"
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
        {% set filter_date_for_url = request.args.get('filter_date') if selected_category == 'All Articles' and request.args.get('filter_date') else None %}
        <li class="page-item page-link-prev-next {% if current_page == 1 %}disabled{% endif %}">
            <a class="page-link" href="{{ url_for(request.endpoint, page=current_page-1, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) if current_page > 1 else '#' }}">&laquo; Prev</a>
        </li>
        {% set page_window = 1 %}{% set show_first = 1 %}{% set show_last = total_pages %}
        {% if current_page - page_window > show_first %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=1, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) }}">1</a></li>{% if current_page - page_window > show_first + 1 %}<li class="page-item disabled"><span class="page-link">...</span></li>{% endif %}{% endif %}
        {% for p in range(1, total_pages + 1) %}{% if p == current_page %}<li class="page-item active" aria-current="page"><span class="page-link">{{ p }}</span></li>{% elif p >= current_page - page_window and p <= current_page + page_window %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=p, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) }}">{{ p }}</a></li>{% endif %}{% endfor %}
        {% if current_page + page_window < show_last %}{% if current_page + page_window < show_last - 1 %}<li class="page-item disabled"><span class="page-link">...</span></li>{% endif %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=total_pages, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) }}">{{ total_pages }}</a></li>{% endif %}
        <li class="page-item page-link-prev-next {% if current_page == total_pages %}disabled{% endif %}">
            <a class="page-link" href="{{ url_for(request.endpoint, page=current_page+1, category_name=selected_category if request.endpoint != 'search_results' else None, query=query if request.endpoint == 'search_results' else None, filter_date=filter_date_for_url) if current_page < total_pages else '#' }}">Next &raquo;</a>
        </li>
    </ul></nav>
    {% endif %}
{% endif %}
{% endblock %}


{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    const isUserLoggedInForHomepage = {{ 'true' if session.user_id else 'false' }};
    document.querySelectorAll('.homepage-bookmark-btn').forEach(button => {
        if (isUserLoggedInForHomepage) {
            button.addEventListener('click', function(event) {
                event.preventDefault(); event.stopPropagation();
                const articleHashId = this.dataset.articleHashId;
                const isCommunity = this.dataset.isCommunity;
                const title = this.dataset.title;
                const sourceName = this.dataset.sourceName;
                const imageUrl = this.dataset.imageUrl;
                const description = this.dataset.description;
                const publishedAt = this.dataset.publishedAt;
                fetch(`{{ url_for('toggle_bookmark', article_hash_id='PLACEHOLDER') }}`.replace('PLACEHOLDER', articleHashId), {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ is_community_article: isCommunity, title: title, source_name: sourceName, image_url: imageUrl, description: description, published_at: publishedAt })
                })
                .then(res => { if (!res.ok) { return res.json().then(err => { throw new Error(err.error || `HTTP error! status: ${res.status}`); }); } return res.json(); })
                .then(data => {
                    if (data.success) {
                        this.classList.toggle('active', data.status === 'added');
                        this.title = data.status === 'added' ? 'Remove Bookmark' : 'Add Bookmark';
                        const alertPlaceholder = document.getElementById('alert-placeholder');
                        if(alertPlaceholder) {
                            const existingAlerts = alertPlaceholder.querySelectorAll('.bookmark-alert');
                            existingAlerts.forEach(al => bootstrap.Alert.getOrCreateInstance(al)?.close());
                            const alertDiv = `<div class="alert alert-info alert-dismissible fade show alert-top bookmark-alert" role="alert" style="z-index: 2060;">${data.message}<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>`;
                            alertPlaceholder.insertAdjacentHTML('beforeend', alertDiv);
                            const newAlert = alertPlaceholder.lastChild;
                            setTimeout(() => { bootstrap.Alert.getOrCreateInstance(newAlert)?.close(); }, 3000);
                        }
                    } else { alert('Error: ' + (data.error || 'Could not update bookmark.')); }
                })
                .catch(err => { console.error("Bookmark error on homepage:", err); alert("Could not update bookmark: " + err.message); });
            });
        }
    });
});
</script>
{% endblock %}
"""

ARTICLE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ article.title|truncate(50) if article else "Article" }} - BrieflyAI{% endblock %}
{% block head_extra %}
<style>
    /* Key styles for this page for context. Assumes extended styles are in BASE_HTML_TEMPLATE. */
    .article-full-content-wrapper { background-color: var(--card-bg); padding: clamp(1rem, 4vw, 2rem); border-radius: var(--border-radius-lg); box-shadow: var(--shadow-md); margin-bottom: 2rem; margin-top: 1rem; }
    .article-title-main {font-weight: 700; color: var(--text-color); line-height:1.3; font-family: 'Poppins', sans-serif;}
    .summary-box, .takeaways-box { background-color: rgba(var(--primary-color-rgb), 0.04); border: 1px solid rgba(var(--primary-color-rgb), 0.1); border-radius: var(--border-radius-md); margin: 1.5rem 0; padding: 1.5rem; }
    body.dark-mode .summary-box, body.dark-mode .takeaways-box { background-color: #2a30422e; }
    .takeaways-box { border-left: 4px solid var(--secondary-color); }
    .loader-container { display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 200px; padding: 2rem; color: var(--text-muted-color); }
    .loader { border: 5px solid var(--light-bg); border-top: 5px solid var(--primary-color); border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin-bottom: 1rem; }
    .content-text { white-space: pre-wrap; line-height: 1.8; font-size: 1.05rem; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>
{% endblock %}
{% block content %}
{% if not article %}
    <div class="alert alert-danger text-center my-5 p-4"><h4><i class="fas fa-exclamation-triangle me-2"></i>Article Not Found</h4><p>The article you are looking for could not be found.</p><a href="{{ url_for('index') }}" class="btn btn-primary mt-2">Go to Homepage</a></div>
{% else %}
<article class="article-full-content-wrapper animate-fade-in">
    {# The entire article content section (title, image, summary, etc.) remains unchanged #}
    <div class="mb-3 d-flex justify-content-between align-items-center">
        <a href="{{ previous_list_page }}" class="btn btn-sm btn-outline-secondary"><i class="fas fa-arrow-left me-2"></i>Back to List</a>
        {% if session.user_id %}
        <button id="bookmarkBtn" class="bookmark-btn {% if is_bookmarked %}active{% endif %}" title="{% if is_bookmarked %}Remove Bookmark{% else %}Add Bookmark{% endif %}" data-article-hash-id="{{ article.article_hash_id if is_community_article else article.id }}" data-is-community="{{ 'true' if is_community_article else 'false' }}" data-title="{{ article.title|e }}" data-source-name="{{ (article.author.name if is_community_article and article.author else article.source.name)|e }}" data-image-url="{{ (article.image_url if is_community_article else article.urlToImage)|e }}" data-description="{{ (article.description if article.description else '')|e }}" data-published-at="{{ (article.published_at.isoformat() if is_community_article and article.published_at else (article.publishedAt if not is_community_article and article.publishedAt else ''))|e }}"><i class="fa-solid fa-bookmark"></i></button>
        {% endif %}
    </div>
    <h1 class="mb-2 article-title-main display-6">{{ article.title }}</h1>
    <div class="article-meta-detailed"><span class="meta-item" title="Source"><i class="fas fa-{{ 'user-edit' if is_community_article else 'building' }}"></i> {{ article.author.name if is_community_article and article.author else article.source.name }}</span><span class="meta-item" title="Published Date"><i class="far fa-calendar-alt"></i> {{ (article.published_at | to_ist if is_community_article else (article.publishedAt | to_ist if article.publishedAt else 'N/A')) }}</span></div>
    {% set image_to_display = article.image_url if is_community_article else article.urlToImage %}
    {% if image_to_display %}<img src="{{ image_to_display }}" alt="{{ article.title|truncate(50) }}" class="img-fluid rounded my-3 shadow-sm">{% endif %}
    <div id="contentLoader" class="loader-container my-4 {% if is_community_article %}d-none{% endif %}"><div class="loader"></div><div>Analyzing article and generating summary...</div></div>
    <div id="articleAnalysisContainer">
    {% if is_community_article %}
        {% if article.groq_summary %}<div class="summary-box my-3"><h5><i class="fas fa-book-open me-2"></i>AI Summary</h5><p class="mb-0">{{ article.groq_summary|replace('\\n', '<br>')|safe }}</p></div>{% endif %}
        {% if article.parsed_takeaways %}<div class="takeaways-box my-3"><h5><i class="fas fa-list-check me-2"></i>AI Key Takeaways</h5><ul>{% for takeaway in article.parsed_takeaways %}<li>{{ takeaway }}</li>{% endfor %}</ul></div>{% endif %}
        <hr class="my-4"><h4 class="mb-3">Full Article Content</h4><div class="content-text">{{ article.full_text }}</div>
    {% else %}<div id="apiArticleContent"></div>{% endif %}
    </div>

    <section class="comment-section mt-5" id="comment-section">
        <h3 class="mb-4">Community Discussion (<span id="comment-count">{{ total_comment_count }}</span>)</h3>
        
        {# --- MACRO WITH IMPROVED HTML STRUCTURE FOR BETTER STYLING --- #}
        {% macro render_comment_with_replies(comment, comment_data, is_logged_in) %}
            <div class="comment-thread" id="comment-{{ comment.id }}">
                <div class="comment-container">
                    <div class="comment-avatar" title="{{ comment.author.name if comment.author else 'Unknown' }}">{{ (comment.author.name[0]|upper if comment.author and comment.author.name else 'U') }}</div>
                    <div class="comment-body">
                        <div class="comment-header">
                            <span class="comment-author">{{ comment.author.name if comment.author else 'Anonymous' }}</span>
                            <span class="comment-date">{{ comment.timestamp | to_ist }}</span>
                        </div>
                        <p class="comment-content mb-2">{{ comment.content }}</p>
                        
                        {% if is_logged_in %}
                        <div class="comment-actions">
                            <div class="reaction-box" id="reaction-box-{{ comment.id }}">{% for emoji in ['👍', '❤️', '😂', '😮', '😢', '😠'] %}<span class="reaction-emoji" data-emoji="{{ emoji }}" data-comment-id="{{ comment.id }}" title="{{ emoji }}">{{ emoji }}</span>{% endfor %}</div>
                            <button class="react-btn" data-comment-id="{{ comment.id }}" title="React"><i class="far fa-smile"></i> React</button>
                            <button class="reply-btn" data-comment-id="{{ comment.id }}" title="Reply"><i class="fas fa-reply"></i> Reply</button>
                        </div>
                        <div class="reply-form-container" id="reply-form-container-{{ comment.id }}">
                            <form class="reply-form"><input type="hidden" name="parent_id" value="{{ comment.id }}"><div class="mb-2"><textarea class="form-control form-control-sm" name="content" rows="2" placeholder="Write a reply..." required></textarea></div><div class="d-flex justify-content-end gap-2"><button type="button" class="btn btn-sm btn-outline-secondary cancel-reply-btn">Cancel</button><button type="submit" class="btn btn-sm btn-primary-modal">Post Reply</button></div></form>
                        </div>
                        {% endif %}

                        <div class="reaction-summary" id="reaction-summary-{{ comment.id }}">
                            {% set reactions = comment_data.get(comment.id, {}).get('reactions', {}) %}
                            {% if reactions %}{% for emoji, count in reactions.items() %}{% set user_reacted_class = 'user-reacted' if is_logged_in and comment_data.get(comment.id, {}).get('user_reaction') == emoji else '' %}<div class="reaction-pill {{ user_reacted_class }}" data-emoji="{{ emoji }}"><span class="emoji">{{ emoji }}</span><span class="count">{{ count }}</span></div>{% endfor %}{% endif %}
                        </div>
                    </div>
                </div>
                <div class="comment-replies" id="replies-of-{{ comment.id }}">
                    {% for reply in comment.replies %}
                        {{ render_comment_with_replies(reply, comment_data, is_logged_in) }}
                    {% endfor %}
                </div>
            </div>
        {% endmacro %}

        <div id="comments-list">
            {% for comment in comments %}
                {{ render_comment_with_replies(comment, comment_data, session.user_id) }}
            {% else %}
                <p id="no-comments-msg" class="text-muted mt-3">No comments yet. Be the first to share your thoughts!</p>
            {% endfor %}
        </div>

        {% if session.user_id %}
            <div class="add-comment-form mt-4 pt-4 border-top">
                <h5 class="mb-3">Leave a Comment</h5>
                <form id="comment-form"><div class="mb-3"><textarea class="form-control" id="comment-content" name="content" rows="4" placeholder="Share your insights..." required></textarea></div><button type="submit" class="btn btn-primary-modal">Post Comment</button></form>
            </div>
        {% else %}
            <div class="alert alert-light mt-4 text-center">Please <a href="{{ url_for('login', next=request.url) }}" class="fw-bold">log in</a> to join the discussion.</div>
        {% endif %}
    </section>
</article>
{% endif %}
{% endblock %}
{% block scripts_extra %}
<script>
// The entire <script> block remains the same as the previous version.
// No changes are needed in the JavaScript logic.
document.addEventListener('DOMContentLoaded', function () {
    {% if article %}
    const isCommunityArticle = {{ is_community_article | tojson }};
    const articleHashIdGlobal = {{ (article.article_hash_id if is_community_article else article.id) | tojson }};
    const isUserLoggedIn = {{ 'true' if session.user_id else 'false' }};
    const groqConfiguredGlobal = {{ groq_client | tojson }};

    function convertUTCToIST(utcIsoString) {
        if (!utcIsoString) return "N/A";
        const date = new Date(utcIsoString);
        return new Intl.DateTimeFormat('en-IN', { year: 'numeric', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', hour12: true, timeZone: 'Asia/Kolkata', timeZoneName: 'short' }).format(date);
    }
    
    if (!isCommunityArticle && articleHashIdGlobal) {
        const contentLoader = document.getElementById('contentLoader');
        const apiArticleContent = document.getElementById('apiArticleContent');
        fetch(`{{ url_for('get_article_content_json', article_hash_id='PLACEHOLDER') }}`.replace('PLACEHOLDER', articleHashIdGlobal))
            .then(response => { if (!response.ok) { throw new Error('Network response error: ' + response.statusText); } return response.json(); })
            .then(data => {
                if(contentLoader) contentLoader.style.display = 'none';
                if (!apiArticleContent) return;
                let html = '';
                const articleUrl = {{ article.url | tojson if article and not is_community_article else 'null' }};
                const articleSourceName = {{ article.source.name | tojson if article and not is_community_article and article.source else 'Source'|tojson }};
                if (data.error && !data.groq_analysis) { html = `<div class="alert alert-warning small p-3 mt-3">Could not load article content: ${data.error}</div>`; }
                else {
                    const analysis = data.groq_analysis;
                    if (analysis) {
                        if (analysis.error) { html += `<div class="alert alert-secondary small p-3 mt-3">AI analysis could not be performed: ${analysis.error}</div>`; }
                        else {
                            if (analysis.groq_summary) { html += `<div class="summary-box my-3"><h5><i class="fas fa-book-open me-2"></i>AI Summary</h5><p class="mb-0">${analysis.groq_summary.replace(/\\n/g, '<br>')}</p></div>`; }
                            if (analysis.groq_takeaways && analysis.groq_takeaways.length > 0) { html += `<div class="takeaways-box my-3"><h5><i class="fas fa-list-check me-2"></i>AI Key Takeaways</h5><ul>${analysis.groq_takeaways.map(t => `<li>${String(t)}</li>`).join('')}</ul></div>`; }
                        }
                    } else if (groqConfiguredGlobal) { html = `<div class="alert alert-warning small p-3 mt-3">AI analysis data is missing.</div>`; }
                }
                if (articleUrl) { html += `<hr class="my-4"><a href="${articleUrl}" class="btn btn-outline-primary mt-3 mb-3" target="_blank" rel="noopener noreferrer">Read Original Article at ${articleSourceName} <i class="fas fa-external-link-alt ms-1"></i></a>`; }
                apiArticleContent.innerHTML = html;
            })
            .catch(error => { if(contentLoader) { contentLoader.innerHTML = `<div class="alert alert-danger small p-3">Failed to load article analysis. Details: ${error.message}</div>`; }});
    }

    const commentSection = document.getElementById('comment-section');

    function updateReactionUI(commentId, reactions, userReaction) {
        const summaryContainer = document.getElementById(`reaction-summary-${commentId}`);
        if (!summaryContainer) return;
        let summaryHTML = '';
        if (reactions) {
            for (const [emoji, count] of Object.entries(reactions)) {
                const userReactedClass = (userReaction === emoji) ? 'user-reacted' : '';
                summaryHTML += `<div class="reaction-pill ${userReactedClass}" data-emoji="${emoji}"><span class="emoji">${emoji}</span> <span class="count">${count}</span></div>`;
            }
        }
        summaryContainer.innerHTML = summaryHTML;
    }

    function createCommentHTML(comment) {
        const commentDate = convertUTCToIST(comment.timestamp);
        const authorName = comment.author && comment.author.name ? comment.author.name : 'Anonymous';
        const userInitial = authorName[0].toUpperCase();
        let actionsHTML = '';
        if (isUserLoggedIn) {
            actionsHTML = `
            <div class="comment-actions">
                <div class="reaction-box" id="reaction-box-${comment.id}">${['👍', '❤️', '😂', '😮', '😢', '😠'].map(emoji => `<span class="reaction-emoji" data-emoji="${emoji}" data-comment-id="${comment.id}" title="${emoji}">${emoji}</span>`).join('')}</div>
                <button class="react-btn" data-comment-id="${comment.id}" title="React"><i class="far fa-smile"></i> React</button>
                <button class="reply-btn" data-comment-id="${comment.id}" title="Reply"><i class="fas fa-reply"></i> Reply</button>
            </div>
            <div class="reply-form-container" id="reply-form-container-${comment.id}">
                <form class="reply-form"><input type="hidden" name="parent_id" value="${comment.id}"><div class="mb-2"><textarea class="form-control form-control-sm" name="content" rows="2" placeholder="Write a reply..." required></textarea></div><div class="d-flex justify-content-end gap-2"><button type="button" class="btn btn-sm btn-outline-secondary cancel-reply-btn">Cancel</button><button type="submit" class="btn btn-sm btn-primary-modal">Post Reply</button></div></form>
            </div>`;
        }
        const sanitizedContent = comment.content.replace(/</g, "&lt;").replace(/>/g, "&gt;");
        return `<div class="comment-thread" id="comment-${comment.id}"><div class="comment-container"><div class="comment-avatar" title="${authorName}">${userInitial}</div><div class="comment-body"><div class="comment-header"><span class="comment-author">${authorName}</span><span class="comment-date">${commentDate}</span></div><p class="comment-content mb-2">${sanitizedContent}</p>${actionsHTML}<div class="reaction-summary" id="reaction-summary-${comment.id}"></div></div></div><div class="comment-replies" id="replies-of-${comment.id}"></div></div>`;
    }

    function handleCommentSubmit(form, parentId = null) {
        const content = form.querySelector('textarea[name="content"]').value;
        if (!content.trim()) return;
        const submitButton = form.querySelector('button[type="submit"]');
        const originalButtonText = submitButton.innerHTML;
        submitButton.disabled = true; submitButton.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Posting...';

        fetch(`{{ url_for('add_comment', article_hash_id='PLACEHOLDER') }}`.replace('PLACEHOLDER', articleHashIdGlobal), {
            method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content: content, parent_id: parentId })
        })
        .then(res => { if (!res.ok) { return res.json().then(err => { throw new Error(err.error || 'HTTP Error ' + res.status); }); } return res.json(); })
        .then(data => {
            if (data.success) {
                const newCommentHTML = createCommentHTML(data.comment);
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
                }
                const countEl = document.getElementById('comment-count');
                countEl.textContent = parseInt(countEl.textContent) + 1;
                form.reset();
            } else { alert('Error: ' + (data.error || 'Unknown error.')); }
        })
        .catch(err => { console.error("Comment error:", err); alert("Could not submit comment: " + err.message); })
        .finally(() => { submitButton.disabled = false; submitButton.innerHTML = originalButtonText; });
    }

    if (commentSection) {
        commentSection.addEventListener('click', function(e) {
            const reactBtn = e.target.closest('.react-btn');
            const reactionEmoji = e.target.closest('.reaction-emoji');
            const replyBtn = e.target.closest('.reply-btn');
            const cancelReplyBtn = e.target.closest('.cancel-reply-btn');

            if (reactBtn) {
                const commentId = reactBtn.dataset.commentId;
                const reactionBox = document.getElementById(`reaction-box-${commentId}`);
                if (reactionBox) {
                    const isShown = reactionBox.classList.contains('show');
                    document.querySelectorAll('.reaction-box').forEach(box => box.classList.remove('show'));
                    if (!isShown) reactionBox.classList.add('show');
                }
            } else if (reactionEmoji) {
                const commentId = reactionEmoji.dataset.commentId;
                const emoji = reactionEmoji.dataset.emoji;
                reactionEmoji.closest('.reaction-box').classList.remove('show');
                fetch(`{{ url_for('vote_comment', comment_id=0) }}`.replace('0', commentId), {
                    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ emoji: emoji })
                })
                .then(res => res.json())
                .then(data => { if (data.success) { updateReactionUI(commentId, data.reactions, data.user_reaction); } })
                .catch(err => console.error("Reaction error:", err));
            } else if (replyBtn) {
                const commentId = replyBtn.dataset.commentId;
                const formContainer = document.getElementById(`reply-form-container-${commentId}`);
                if (formContainer) {
                    const isDisplayed = formContainer.style.display === 'block';
                    document.querySelectorAll('.reply-form-container').forEach(fc => fc.style.display = 'none');
                    formContainer.style.display = isDisplayed ? 'none' : 'block';
                    if (!isDisplayed) formContainer.querySelector('textarea').focus();
                }
            } else if (cancelReplyBtn) {
                const formContainer = cancelReplyBtn.closest('.reply-form-container');
                formContainer.style.display = 'none';
                formContainer.querySelector('form').reset();
            } else {
                if (!e.target.closest('.reaction-box')) {
                    document.querySelectorAll('.reaction-box.show').forEach(box => box.classList.remove('show'));
                }
            }
        });

        commentSection.addEventListener('submit', function(e) {
            if (e.target.matches('.reply-form') || e.target.matches('#comment-form')) {
                e.preventDefault();
                const parentId = e.target.querySelector('input[name="parent_id"]')?.value || null;
                handleCommentSubmit(e.target, parentId);
            }
        });
    }

    const bookmarkBtn = document.getElementById('bookmarkBtn');
    if (bookmarkBtn && isUserLoggedIn) {
        bookmarkBtn.addEventListener('click', function() {
            const articleHashId = this.dataset.articleHashId; const isCommunity = this.dataset.isCommunity; const title = this.dataset.title; const sourceName = this.dataset.sourceName; const imageUrl = this.dataset.imageUrl; const description = this.dataset.description; const publishedAt = this.dataset.publishedAt;
            fetch(`{{ url_for('toggle_bookmark', article_hash_id='PLACEHOLDER') }}`.replace('PLACEHOLDER', articleHashId), { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ is_community_article: isCommunity, title: title, source_name: sourceName, image_url: imageUrl, description: description, published_at: publishedAt }) })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    this.classList.toggle('active', data.status === 'added'); this.title = data.status === 'added' ? 'Remove Bookmark' : 'Add Bookmark';
                    const alertPlaceholder = document.getElementById('alert-placeholder'); if(alertPlaceholder) { const existingAlerts = alertPlaceholder.querySelectorAll('.bookmark-alert'); existingAlerts.forEach(al => bootstrap.Alert.getOrCreateInstance(al)?.close()); const alertDiv = `<div class="alert alert-info alert-dismissible fade show alert-top bookmark-alert" role="alert" style="z-index: 2060;">${data.message}<button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button></div>`; alertPlaceholder.insertAdjacentHTML('beforeend', alertDiv); const newAlert = alertPlaceholder.lastChild; setTimeout(() => { bootstrap.Alert.getOrCreateInstance(newAlert)?.close(); }, 3000); }
                } else { alert('Error: ' + (data.error || 'Could not update bookmark.')); }
            })
            .catch(err => { console.error("Bookmark error:", err); alert("Could not update bookmark: " + err.message); });
        });
    }
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
