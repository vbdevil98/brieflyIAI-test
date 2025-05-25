# Rev14.py - CORRECTED CODE FOR ImportError

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import hashlib
import time
import logging
import urllib.parse
from datetime import datetime, timedelta
from functools import wraps

# Third-party imports
import nltk
import requests
from flask import (Flask, render_template, url_for, redirect, request, jsonify, session, flash)
from flask_sqlalchemy import SQLAlchemy
# ## FIX ##: Removed 'from_json' as it does not exist in Jinja2
from jinja2 import DictLoader
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from newspaper import Article, Config
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.exceptions import LangChainException

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
    print("FATAL: NLTK 'punkt' tokenizer not found. Downloading...", file=sys.stderr)
    try:
        nltk.download('punkt')
        print("NLTK 'punkt' downloaded successfully.", file=sys.stderr)
    except Exception as e:
        print(f"FATAL: Failed to download 'punkt'. Please ensure the 'nltk_data' folder is in your repository. Error: {e}", file=sys.stderr)
        sys.exit("Exiting: Missing critical NLTK data.")


# ==============================================================================
# --- 2. Flask Application Initialization & Configuration ---
# ==============================================================================
app = Flask(__name__)

template_storage = {}
app.jinja_loader = DictLoader(template_storage)

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_strong_fallback_secret_key_for_dev_32_chars') # IMPORTANT: Set in Render
app.config['PER_PAGE'] = 9
app.config['CATEGORIES'] = ['All Articles', 'Community Hub']
app.config['NEWS_API_SOURCES'] = os.environ.get('NEWS_API_SOURCES', '')
app.config['NEWS_API_QUERY'] = (
    '("latest tech trends" OR "AI breakthroughs" OR "market analysis" OR "business innovation" OR "startup funding") NOT "press release"'
)
app.config['NEWS_API_DAYS_AGO'] = 7
app.config['NEWS_API_PAGE_SIZE'] = 100
app.config['NEWS_API_SORT_BY'] = 'popularity'
app.config['CACHE_EXPIRY_SECONDS'] = 3600
app.config['READING_SPEED_WPM'] = 230
app.permanent_session_lifetime = timedelta(days=30)

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.logger.info("Connecting to PostgreSQL database.")
else:
    db_file_name = 'app_data.db'
    project_root_for_db = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(project_root_for_db, db_file_name)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.logger.info(f"Using local SQLite database at {db_path}.")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ==============================================================================
# --- 3. API Client Initialization ---
# ==============================================================================
NEWSAPI_KEY = os.environ.get('NEWSAPI_KEY')
newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None
if not newsapi: app.logger.error("CRITICAL: NEWSAPI_KEY is not set. News fetching will fail.")

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = ChatGroq(model="llama3-70b-8192", groq_api_key=GROQ_API_KEY, temperature=0.2)
        app.logger.info("Groq client initialized successfully with llama3-70b-8192.")
    except Exception as e:
        app.logger.error(f"Failed to initialize Groq client: {e}")
else:
    app.logger.warning("WARNING: GROQ_API_KEY is not set. AI analysis features will be disabled.")

SCRAPER_API_KEY = os.environ.get('SCRAPER_API_KEY')
if not SCRAPER_API_KEY:
    app.logger.warning("WARNING: SCRAPER_API_KEY is not set. Fetching external article content may fail.")

# ==============================================================================
# --- 4. Database Models ---
# ==============================================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    articles = db.relationship('CommunityArticle', backref='author', lazy='dynamic', cascade="all, delete-orphan")
    comments = db.relationship('Comment', backref='author', lazy='dynamic', cascade="all, delete-orphan")

class CommunityArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    article_hash_id = db.Column(db.String(32), unique=True, nullable=False, index=True)
    title = db.Column(db.String(250), nullable=False)
    description = db.Column(db.Text, nullable=False)
    full_text = db.Column(db.Text, nullable=False)
    source_name = db.Column(db.String(100), nullable=False)
    image_url = db.Column(db.String(500), nullable=True)
    published_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    groq_summary = db.Column(db.Text, nullable=True)
    groq_takeaways = db.Column(db.Text, nullable=True) # Stored as JSON string
    comments = db.relationship('Comment', backref='community_article', lazy='dynamic', foreign_keys='Comment.community_article_id', cascade="all, delete-orphan")

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    community_article_id = db.Column(db.Integer, db.ForeignKey('community_article.id'), nullable=True)
    api_article_hash_id = db.Column(db.String(32), nullable=True, index=True)

def init_db():
    with app.app_context():
        app.logger.info("Initializing database and creating tables if they don't exist...")
        db.create_all()
        app.logger.info("Database tables are ready.")

init_db()

# ==============================================================================
# --- 5. Helper Functions ---
# ==============================================================================
MASTER_ARTICLE_STORE, API_CACHE = {}, {}

def generate_article_id(url_or_title): return hashlib.md5(url_or_title.encode('utf-8')).hexdigest()

def simple_cache(expiry_seconds_default=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            expiry = expiry_seconds_default or app.config['CACHE_EXPIRY_SECONDS']
            key_parts = [func.__name__] + list(args) + sorted(kwargs.items())
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

def calculate_read_time(text):
    if not text: return 0
    word_count = len(text.split())
    return max(1, round(word_count / app.config['READING_SPEED_WPM']))

@simple_cache(expiry_seconds_default=3600 * 24)
def get_article_analysis_with_groq(article_text, article_title=""):
    if not groq_client:
        app.logger.warning("Groq client not initialized. Skipping AI analysis.")
        return {"error": "AI analysis service is not configured."}
    if not article_text or not article_text.strip():
        return {"error": "Cannot analyze an empty article."}

    app.logger.info(f"Requesting Groq analysis for: {article_title[:50]}...")
    system_prompt = (
        "You are an expert news analyst. Your task is to analyze the provided article. "
        "From the text, you must extract two specific things: "
        "1. A concise, professional, and neutral summary of the article, about 3-4 paragraphs long. "
        "2. A list of 5 to 7 key takeaways. Each takeaway must be a complete, impactful sentence. "
        "You MUST format your entire response as a single, valid JSON object with two keys: "
        "'summary' (a string) and 'takeaways' (a list of strings). "
        "Do not include any text or formatting outside of this JSON object."
    )
    human_prompt = f"Here is the article to analyze.\n\nTitle: {article_title}\n\nText:\n{article_text[:15000]}"

    try:
        json_model = groq_client.bind(response_format={"type": "json_object"})
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        ai_response = json_model.invoke(messages)
        analysis = json.loads(ai_response.content)

        if 'summary' in analysis and 'takeaways' in analysis and isinstance(analysis['takeaways'], list):
            app.logger.info(f"Groq analysis successful for: {article_title[:50]}")
            return {"groq_summary": analysis.get("summary"), "groq_takeaways": analysis.get("takeaways"), "error": None}
        else:
            app.logger.error(f"Groq response for '{article_title[:50]}' was malformed. Content: {ai_response.content}")
            raise ValueError("Response from AI is missing 'summary' or 'takeaways' key, or 'takeaways' is not a list.")

    except (json.JSONDecodeError, ValueError, LangChainException) as e:
        app.logger.error(f"Groq JSON analysis failed for '{article_title[:50]}'. Error: {e}")
        return {"error": f"The AI failed to generate a valid analysis. Details: {str(e)}"}
    except Exception as e:
        app.logger.error(f"An unexpected error occurred during Groq analysis for '{article_title[:50]}': {e}", exc_info=True)
        return {"error": "An unexpected error occurred while communicating with the AI service."}


@simple_cache()
def fetch_news_from_api():
    if not newsapi: return []
    from_date = (datetime.utcnow() - timedelta(days=app.config['NEWS_API_DAYS_AGO'])).strftime('%Y-%m-%d')
    preferred_sources = app.config.get('NEWS_API_SOURCES', '').strip()
    
    try:
        if preferred_sources:
            app.logger.info(f"Fetching top headlines from preferred sources: {preferred_sources}")
            response = newsapi.get_top_headlines(sources=preferred_sources, language='en', page_size=app.config['NEWS_API_PAGE_SIZE'])
        else:
            app.logger.info(f"Fetching general news with query: {app.config['NEWS_API_QUERY']}")
            response = newsapi.get_everything(q=app.config['NEWS_API_QUERY'], from_param=from_date, language='en', sort_by=app.config['NEWS_API_SORT_BY'], page_size=app.config['NEWS_API_PAGE_SIZE'])

        processed_articles, unique_titles = [], set()
        for art_data in response.get('articles', []):
            title, url = art_data.get('title'), art_data.get('url')
            if not all([url, title, art_data.get('source'), art_data.get('description')]) or title == '[Removed]' or title.lower() in unique_titles:
                continue
            
            unique_titles.add(title.lower())
            article_hash_id = generate_article_id(url)
            source_name = art_data['source'].get('name', 'Unknown Source')
            
            standardized_article = {
                'id': article_hash_id,
                'title': title, 'description': art_data.get('description', ''), 'url': url,
                'urlToImage': art_data.get('urlToImage'), 'publishedAt': art_data.get('publishedAt', ''),
                'source': {'name': source_name}, 'is_community_article': False
            }
            MASTER_ARTICLE_STORE[article_hash_id] = standardized_article
            processed_articles.append(standardized_article)
            
        app.logger.info(f"Fetched and processed {len(processed_articles)} articles.")
        return processed_articles
    except NewsAPIException as e:
        app.logger.error(f"NewsAPI error: {e}")
        return []
    except Exception as e:
        app.logger.error(f"Generic error fetching news: {e}", exc_info=True)
        return []


@simple_cache(expiry_seconds_default=3600 * 6)
def fetch_and_parse_article_content(article_hash_id, url):
    app.logger.info(f"Fetching content for API article ID: {article_hash_id} from URL: {url}")
    if not SCRAPER_API_KEY:
        app.logger.error("SCRAPER_API_KEY not set. Cannot fetch external article content.")
        return {"error": "Content fetching service is not available."}

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

        if not article_scraper.text:
            app.logger.warning(f"Newspaper3k could not extract text from {url}")
            return {"error": "The tool could not extract readable text from the source page."}
        
        article_title = article_scraper.title or MASTER_ARTICLE_STORE.get(article_hash_id, {}).get('title', 'Unknown Title')
        groq_analysis = get_article_analysis_with_groq(article_scraper.text, article_title)
        
        return {
            "full_text": article_scraper.text,
            "read_time_minutes": calculate_read_time(article_scraper.text),
            "groq_analysis": groq_analysis,
            "error": groq_analysis.get("error") if groq_analysis else "AI analysis was not performed."
        }
    except requests.exceptions.RequestException as e:
        app.logger.error(f"ScraperAPI request failed for {url}: {e}")
        return {"error": f"Failed to fetch the article content. The source website might be down or blocking requests."}
    except Exception as e:
        app.logger.error(f"Error parsing article {url}: {e}", exc_info=True)
        return {"error": "An unexpected error occurred while parsing the article."}

# ==============================================================================
# --- 6. Flask Routes ---
# ==============================================================================
@app.context_processor
def inject_global_vars():
    user_context = {}
    if 'user_id' in session:
        user_context['user_id'] = session['user_id']
        user_context['user_name'] = session.get('user_name', 'User')

    return {
        'categories': app.config['CATEGORIES'], 'current_year': datetime.utcnow().year,
        'user': user_context, 'request': request
    }

def get_paginated_articles(articles, page, per_page):
    total = len(articles)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_items = articles[start:end]
    total_pages = (total + per_page - 1) // per_page if total > 0 else 0
    return paginated_items, total_pages

@app.route('/')
@app.route('/page/<int:page>')
@app.route('/category/<category_name>')
@app.route('/category/<category_name>/page/<int:page>')
def index(page=1, category_name='All Articles'):
    per_page = app.config['PER_PAGE']
    all_display_articles = []
    
    api_articles = fetch_news_from_api() if category_name != 'Community Hub' else []
    db_articles = CommunityArticle.query.order_by(CommunityArticle.published_at.desc()).all() if category_name != 'All Articles' else []

    if category_name == 'All Articles':
        all_display_articles.extend(api_articles)
        all_display_articles.extend(db_articles)
        all_display_articles.sort(key=lambda x: getattr(x, 'published_at', datetime.fromisoformat(x.get('publishedAt', '1970-01-01T00:00:00Z').replace('Z', '+00:00'))) if isinstance(x, dict) and x.get('publishedAt') else (getattr(x, 'published_at', datetime.min)), reverse=True)
    elif category_name == 'Community Hub':
        all_display_articles = db_articles
    else: # Should not happen with current categories, but for future-proofing
        all_display_articles = api_articles

    # Add flags after sorting
    for art in all_display_articles:
        if isinstance(art, CommunityArticle):
            art.is_community_article = True
        else:
            art['is_community_article'] = False

    paginated_articles, total_pages = get_paginated_articles(all_display_articles, page, per_page)
    is_featured_page = (page == 1 and category_name == 'All Articles' and not request.args.get('query') and paginated_articles)

    return render_template("INDEX_HTML_TEMPLATE",
                           articles=paginated_articles, selected_category=category_name,
                           current_page=page, total_pages=total_pages, is_featured_page=is_featured_page)


@app.route('/search')
@app.route('/search/page/<int:page>')
def search_results(page=1):
    query_str = request.args.get('query', '').strip()
    if not query_str: return redirect(url_for('index'))

    app.logger.info(f"Executing search for query: '{query_str}'")
    per_page = app.config['PER_PAGE']
    
    api_results = [art for art_id, art in MASTER_ARTICLE_STORE.items() if query_str.lower() in art.get('title', '').lower() or query_str.lower() in art.get('description', '').lower()]
    community_results = CommunityArticle.query.filter(db.or_(CommunityArticle.title.ilike(f'%{query_str}%'), CommunityArticle.description.ilike(f'%{query_str}%'))).order_by(CommunityArticle.published_at.desc()).all()

    all_search_results = api_results + community_results
    all_search_results.sort(key=lambda x: getattr(x, 'published_at', datetime.fromisoformat(x.get('publishedAt', '1970-01-01T00:00:00Z').replace('Z', '+00:00'))) if isinstance(x, dict) and x.get('publishedAt') else (getattr(x, 'published_at', datetime.min)), reverse=True)
    
    for art in all_search_results:
        if isinstance(art, CommunityArticle): art.is_community_article = True
        else: art['is_community_article'] = False

    paginated_results, total_pages = get_paginated_articles(all_search_results, page, per_page)

    return render_template("INDEX_HTML_TEMPLATE",
                           articles=paginated_results, selected_category=f"Search: {query_str}",
                           current_page=page, total_pages=total_pages, is_featured_page=False, query=query_str)


@app.route('/article/<article_hash_id>')
def article_detail(article_hash_id):
    article = None
    is_community = False

    article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
    if article:
        is_community = True
        # ## FIX ##: Parse the JSON string into a Python list in the view function
        if article.groq_takeaways:
            try:
                article.parsed_takeaways = json.loads(article.groq_takeaways)
            except json.JSONDecodeError:
                article.parsed_takeaways = []
        else:
            article.parsed_takeaways = []
    else:
        article = MASTER_ARTICLE_STORE.get(article_hash_id)
        if not article:
            flash("The article you are looking for could not be found.", "danger")
            return redirect(url_for('index'))

    if is_community:
        comments = article.comments.order_by(Comment.timestamp.asc()).all()
    else:
        comments = Comment.query.filter_by(api_article_hash_id=article_hash_id).order_by(Comment.timestamp.asc()).all()

    return render_template("ARTICLE_HTML_TEMPLATE",
                           article=article, is_community_article=is_community, comments=comments)


@app.route('/get_article_content/<article_hash_id>')
def get_article_content_json(article_hash_id):
    article_data = MASTER_ARTICLE_STORE.get(article_hash_id)
    if not article_data or 'url' not in article_data:
        app.logger.warning(f"API Article or URL not found for hash_id: {article_hash_id}")
        return jsonify({"error": "Article data could not be located."}), 404
    
    if 'full_text' in article_data:
        app.logger.info(f"Serving cached content for article {article_hash_id}")
        return jsonify(article_data)

    processed_content = fetch_and_parse_article_content(article_hash_id, article_data['url'])
    if processed_content and not processed_content.get("error"):
        MASTER_ARTICLE_STORE[article_hash_id].update(processed_content)
    else:
        app.logger.error(f"Failed to process content for {article_hash_id}: {processed_content.get('error')}")
    return jsonify(processed_content)


@app.route('/add_comment/<article_hash_id>', methods=['POST'])
@login_required
def add_comment(article_hash_id):
    content = request.json.get('content', '').strip()
    if not content: return jsonify({"error": "Comment cannot be empty."}), 400

    user = User.query.get(session['user_id'])
    if not user: return jsonify({"error": "User not found."}), 401

    community_article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
    if community_article:
        new_comment = Comment(content=content, user_id=user.id, community_article_id=community_article.id)
    elif article_hash_id in MASTER_ARTICLE_STORE:
        new_comment = Comment(content=content, user_id=user.id, api_article_hash_id=article_hash_id)
    else:
        return jsonify({"error": "Cannot comment on an unknown article."}), 404

    db.session.add(new_comment)
    db.session.commit()
    app.logger.info(f"User '{user.username}' added comment to article '{article_hash_id}'")
    
    return jsonify({"success": True, "comment": {"content": new_comment.content, "timestamp": new_comment.timestamp.isoformat(), "author": {"name": user.name}}}), 201


@app.route('/post_article', methods=['POST'])
@login_required
def post_article():
    title = request.form.get('title', '').strip()
    description = request.form.get('description', '').strip()
    content = request.form.get('content', '').strip()
    source_name = request.form.get('sourceName', 'Community Post').strip()
    image_url = request.form.get('imageUrl', '').strip()

    if not all([title, description, content]):
        flash("Title, Description, and Full Content are required fields.", "danger")
        return redirect(request.referrer or url_for('index'))

    article_hash_id = generate_article_id(title + str(session['user_id']) + str(time.time()))
    groq_analysis = get_article_analysis_with_groq(content, title)
    groq_summary, groq_takeaways_json = None, None
    if groq_analysis and not groq_analysis.get("error"):
        groq_summary = groq_analysis.get('groq_summary')
        takeaways = groq_analysis.get('groq_takeaways')
        if takeaways: groq_takeaways_json = json.dumps(takeaways)
    
    new_article = CommunityArticle(
        article_hash_id=article_hash_id, title=title, description=description, full_text=content,
        source_name=source_name,
        image_url=image_url or f'https://via.placeholder.com/400x220/1E3A5E/FFFFFF?text={urllib.parse.quote_plus(title[:20])}',
        user_id=session['user_id'], groq_summary=groq_summary, groq_takeaways=groq_takeaways_json
    )
    db.session.add(new_article)
    db.session.commit()
    
    flash("Your article has been successfully posted to the Community Hub!", "success")
    return redirect(url_for('article_detail', article_hash_id=new_article.article_hash_id))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not all([name, username, password]): flash('All fields are required.', 'danger')
        elif len(username) < 3: flash('Username must be at least 3 characters long.', 'warning')
        elif len(password) < 6: flash('Password must be at least 6 characters long.', 'warning')
        elif User.query.filter(db.func.lower(User.username) == username.lower()).first(): flash('That username is already taken. Please choose another.', 'warning')
        else:
            new_user = User(name=name, username=username, password_hash=generate_password_hash(password))
            db.session.add(new_user)
            db.session.commit()
            app.logger.info(f"New user registered: {username}")
            flash(f'Welcome, {name}! Your account has been created. Please log in.', 'success')
            return redirect(url_for('login'))
        return redirect(url_for('register'))

    return render_template("REGISTER_HTML_TEMPLATE")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter(db.func.lower(User.username) == username.lower()).first()

        if user and check_password_hash(user.password_hash, password):
            session.permanent = True
            session['user_id'] = user.id
            session['user_name'] = user.name
            app.logger.info(f"User '{username}' logged in successfully.")
            flash(f"Welcome back, {user.name}!", "success")
            next_url = request.args.get('next')
            return redirect(next_url or url_for('index'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
            
    return render_template("LOGIN_HTML_TEMPLATE")

@app.route('/logout')
@login_required
def logout():
    user_name_logged_out = session.get('user_name', 'User')
    session.clear()
    app.logger.info(f"User '{user_name_logged_out}' logged out.")
    flash("You have been successfully logged out.", "info")
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    app.logger.warning(f"404 error triggered for URL: {request.url}")
    return render_template("404_TEMPLATE"), 404

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"500 internal server error at {request.url}: {e}", exc_info=True)
    return render_template("500_TEMPLATE"), 500

# ==============================================================================
# --- 7. HTML Templates (Inline for simplicity) ---
# ==============================================================================
# The HTML templates now correctly link to articles and render takeaways for community posts.

BASE_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}Briefly{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root { --primary-color: #0A2342; --secondary-color: #B8860B; --white-bg: #FFFFFF; }
        body { padding-top: 95px; font-family: 'Roboto', sans-serif; }
        .main-content { flex-grow: 1; }
        .navbar-main { background: linear-gradient(135deg, var(--primary-color), #1E3A5E); height: 95px; }
        .navbar-brand-custom { color: white !important; font-weight: 800; font-size: 2.2rem; font-family: 'Poppins', sans-serif;}
        .brand-icon { color: #D4A017; }
        .header-btn { border: 1px solid rgba(255,255,255,0.3); color: white; }
        .header-btn:hover { background: var(--secondary-color); border-color: var(--secondary-color); color: var(--primary-color); }
        .category-nav { background: var(--white-bg); position: fixed; top: 95px; width: 100%; z-index: 1020; }
        .category-link.active { background: var(--primary-color) !important; color: white !important; }
        .article-card { border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
        footer { background: #061A30; color: rgba(255,255,255,0.8); }
        .alert-top { position: fixed; top: 105px; left: 50%; transform: translateX(-50%); z-index: 2050; }
        .add-article-btn { position: fixed; bottom: 25px; right: 25px; z-index: 1030; width: 55px; height: 55px; border-radius: 50%; background: var(--secondary-color); color: var(--primary-color); border: none; font-size: 22px; }
        .add-article-modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 2000; background-color: rgba(0,0,0,0.6); align-items: center; justify-content: center; }
        .modal-content { max-width: 700px; background: var(--white-bg); padding: 2rem; }
    </style>
    {% block head_extra %}{% endblock %}
</head>
<body class="d-flex flex-column min-vh-100">
    <div id="alert-placeholder">
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% for category, message in messages %}
        <div class="alert alert-{{ category }} alert-dismissible fade show alert-top" role="alert">
            <span>{{ message }}</span><button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
        {% endfor %}
    {% endwith %}
    </div>
    <nav class="navbar navbar-main navbar-expand-lg fixed-top">
        <div class="container">
            <a class="navbar-brand-custom" href="{{ url_for('index') }}"><i class="fas fa-bolt-lightning brand-icon"></i> Briefly</a>
            <div class="ms-auto d-flex align-items-center">
            {% if user.user_id %}
                <span class="text-white me-3">Hi, {{ user.user_name.split(' ')[0] }}!</span>
                <a href="{{ url_for('logout') }}" class="header-btn btn btn-sm">Logout</a>
            {% else %}
                <a href="{{ url_for('login') }}" class="header-btn btn btn-sm">Login</a>
            {% endif %}
            </div>
        </div>
    </nav>
    <nav class="navbar navbar-expand-lg category-nav py-0">
        <div class="container justify-content-center">
        {% for cat_item in categories %}
            <a href="{{ url_for('index', category_name=cat_item) }}" class="nav-link category-link px-3 py-2 {% if selected_category == cat_item %}active{% endif %}">{{ cat_item }}</a>
        {% endfor %}
        </div>
    </nav>
    <main class="container main-content my-4">
        {% block content %}{% endblock %}
    </main>
    {% if user.user_id %}
    <button class="add-article-btn" id="addArticleBtn" title="Post Article"><i class="fas fa-plus"></i></button>
    <div class="add-article-modal" id="addArticleModal">
        <div class="modal-content position-relative">
            <button class="btn-close position-absolute top-0 end-0 m-3" id="closeModalBtn"></button>
            <h3 class="mb-4">Post to Community Hub</h3>
            <form id="addArticleForm" action="{{ url_for('post_article') }}" method="POST">
                <div class="mb-3"><label class="form-label">Title</label><input type="text" name="title" class="form-control" required></div>
                <div class="mb-3"><label class="form-label">Description</label><textarea name="description" class="form-control" rows="3" required></textarea></div>
                <div class="mb-3"><label class="form-label">Source Name</label><input type="text" name="sourceName" class="form-control" value="Community Post" required></div>
                <div class="mb-3"><label class="form-label">Image URL (Optional)</label><input type="url" name="imageUrl" class="form-control"></div>
                <div class="mb-3"><label class="form-label">Full Content</label><textarea name="content" class="form-control" rows="7" required></textarea></div>
                <div class="d-flex justify-content-end gap-2"><button type="button" class="btn btn-secondary" id="cancelArticleBtn">Cancel</button><button type="submit" class="btn btn-primary">Post</button></div>
            </form>
        </div>
    </div>
    {% endif %}
    <footer class="mt-auto py-4 text-center">
        <div class="container">&copy; {{ current_year }} Briefly. All rights reserved.</div>
    </footer>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
    document.addEventListener('DOMContentLoaded', function () {
        const addArticleBtn = document.getElementById('addArticleBtn');
        const addArticleModal = document.getElementById('addArticleModal');
        if (addArticleBtn) {
            const closeModal = () => addArticleModal.style.display = 'none';
            addArticleBtn.addEventListener('click', () => addArticleModal.style.display = 'flex');
            document.getElementById('closeModalBtn').addEventListener('click', closeModal);
            document.getElementById('cancelArticleBtn').addEventListener('click', closeModal);
        }
        document.querySelectorAll('.alert').forEach(a => setTimeout(() => bootstrap.Alert.getOrCreateInstance(a).close(), 5000));
    });
    </script>
    {% block scripts_extra %}{% endblock %}
</body>
</html>
"""

INDEX_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{% if query %}Search: {{ query|truncate(30) }}{% else %}{{selected_category}}{% endif %} - Briefly{% endblock %}
{% block content %}
    <div class="row g-4">
        {% for art in articles %}
        <div class="col-md-6 col-lg-4 d-flex">
            <div class="article-card card w-100">
                {% set article_url = url_for('article_detail', article_hash_id=(art.article_hash_id if art.is_community_article else art.id)) %}
                <a href="{{ article_url }}"><img src="{{ art.image_url if art.is_community_article else art.urlToImage }}" class="card-img-top" style="height: 200px; object-fit: cover;" alt=""></a>
                <div class="card-body d-flex flex-column">
                    <h5 class="card-title h6"><a href="{{ article_url }}" class="text-decoration-none text-dark">{{ art.title|truncate(70) }}</a></h5>
                    <p class="card-text small text-muted flex-grow-1">{{ art.description|truncate(100) }}</p>
                    <a href="{{ article_url }}" class="btn btn-sm btn-primary mt-auto align-self-start">Read More</a>
                </div>
            </div>
        </div>
        {% else %}
        <div class="col-12"><div class="alert alert-info text-center">No articles found.</div></div>
        {% endfor %}
    </div>

    {% if total_pages and total_pages > 1 %}
    <nav class="mt-5"><ul class="pagination justify-content-center">
        <li class="page-item {% if current_page == 1 %}disabled{% endif %}"><a class="page-link" href="{{ url_for(request.endpoint, page=current_page-1, category_name=selected_category, query=query) }}">&laquo;</a></li>
        {% for p in range(1, total_pages + 1) %}
        <li class="page-item {% if p == current_page %}active{% endif %}"><a class="page-link" href="{{ url_for(request.endpoint, page=p, category_name=selected_category, query=query) }}">{{ p }}</a></li>
        {% endfor %}
        <li class="page-item {% if current_page == total_pages %}disabled{% endif %}"><a class="page-link" href="{{ url_for(request.endpoint, page=current_page+1, category_name=selected_category, query=query) }}">&raquo;</a></li>
    </ul></nav>
    {% endif %}
{% endblock %}
"""

ARTICLE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ article.title|truncate(50) if article else "Article" }} - Briefly{% endblock %}
{% block head_extra %}
<style>
    .loader { border: 5px solid #f3f3f3; border-top: 5px solid var(--primary-color); border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin: auto;}
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .summary-box, .takeaways-box { background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; }
</style>
{% endblock %}
{% block content %}
<div class="bg-white p-4 rounded-3">
{% if not article %}
    <div class="alert alert-danger text-center">Article not found.</div>
{% else %}
<article>
    <h1 class="mb-3 display-6">{{ article.title }}</h1>
    <p class="text-muted mb-3">By: {{ article.author.name if is_community_article else article.source.name }} | Published: {{ (article.published_at.strftime('%b %d, %Y') if is_community_article else (article.publishedAt.split('T')[0] if article.publishedAt else 'N/A')) }}</p>
    {% if article.image_url or article.urlToImage %}<img src="{{ article.image_url if is_community_article else article.urlToImage }}" alt="{{ article.title }}" class="img-fluid rounded mb-4">{% endif %}

    <div id="aiContentContainer">
        {% if is_community_article %}
            {% if article.groq_summary %}
                <div class="summary-box"><h5>AI Summary</h5><div>{{ article.groq_summary|replace('\\n', '<br>')|safe }}</div></div>
            {% endif %}
            {# ## FIX ##: Loop over 'article.parsed_takeaways' directly, without the broken filter #}
            {% if article.parsed_takeaways %}
                <div class="takeaways-box"><h5>AI Key Takeaways</h5>
                    <ul>{% for takeaway in article.parsed_takeaways %}<li>{{ takeaway }}</li>{% endfor %}</ul>
                </div>
            {% endif %}
        {% else %}
            <div id="contentLoader" class="text-center p-4"><div class="loader"></div><p class="mt-2">Generating AI summary...</p></div>
        {% endif %}
    </div>
    
    <hr>
    <div class="mt-4">
        <h5>{% if is_community_article %}Full Article{% else %}Original Article{% endif %}</h5>
        {% if is_community_article %}
            <p style="white-space: pre-wrap;">{{ article.full_text }}</p>
        {% else %}
            <a href="{{ article.url }}" class="btn btn-outline-primary" target="_blank" rel="noopener noreferrer">Read at {{ article.source.name }} <i class="fas fa-external-link-alt ms-1"></i></a>
        {% endif %}
    </div>

    <section class="comment-section mt-5" id="comment-section">
        <h3 class="mb-4">Discussion ({{ comments|length }})</h3>
        <div id="comments-list">
        {% for comment in comments %}<div class="card mb-2"><div class="card-body small"><p class="mb-1">{{ comment.content }}</p><footer class="blockquote-footer small mb-0">{{ comment.author.name }}</footer></div></div>{% else %}<p id="no-comments-msg">No comments yet.</p>{% endfor %}
        </div>
        {% if user.user_id %}
            <form id="comment-form" class="mt-4"><div class="mb-2"><textarea class="form-control" id="comment-content" rows="3" required></textarea></div><button type="submit" class="btn btn-primary">Post</button></form>
        {% else %}
            <div class="alert alert-light mt-4">Please <a href="{{ url_for('login', next=request.url) }}">log in</a> to comment.</div>
        {% endif %}
    </section>
</article>
{% endif %}
</div>
{% endblock %}
{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    const isCommunity = {{ is_community_article | tojson }};
    const articleHashId = {{ (article.article_hash_id if is_community_article else article.id) | tojson }};

    if (!isCommunity) {
        const aiContainer = document.getElementById('aiContentContainer');
        fetch(`{{ url_for('get_article_content_json', article_hash_id='_') }}`.replace('_', articleHashId))
            .then(res => res.json()).then(data => {
                aiContainer.innerHTML = ''; // Clear loader
                if (data.error) { aiContainer.innerHTML = `<div class="alert alert-warning">${data.error}</div>`; return; }
                let html = '';
                const analysis = data.groq_analysis;
                if (analysis && analysis.groq_summary) { html += `<div class="summary-box"><h5>AI Summary</h5><p>${analysis.groq_summary.replace(/\\n/g, '<br>')}</p></div>`; }
                if (analysis && analysis.groq_takeaways && analysis.groq_takeaways.length > 0) { html += `<div class="takeaways-box"><h5>AI Key Takeaways</h5><ul>${analysis.groq_takeaways.map(t => `<li>${t}</li>`).join('')}</ul></div>`; }
                aiContainer.innerHTML = html;
            }).catch(err => {
                aiContainer.innerHTML = '<div class="alert alert-danger">Failed to load article analysis.</div>';
            });
    }

    const commentForm = document.getElementById('comment-form');
    if (commentForm) {
        commentForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const content = document.getElementById('comment-content').value.trim();
            if (!content) return;
            fetch(`{{ url_for('add_comment', article_hash_id='_') }}`.replace('_', articleHashId), {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content: content })
            }).then(res => res.json()).then(data => {
                if (data.success) { window.location.reload(); } else { alert('Error: ' + data.error); }
            });
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
<div style="max-width: 450px; margin: 3rem auto; padding: 2rem; background: var(--white-bg); border-radius: 10px;">
    <h2 class="text-center mb-4">Login</h2>
    <form method="POST" action="{{ url_for('login', next=request.args.get('next')) }}">
        <div class="mb-3"><label class="form-label">Username</label><input type="text" class="form-control" name="username" required></div>
        <div class="mb-3"><label class="form-label">Password</label><input type="password" class="form-control" name="password" required></div>
        <button type="submit" class="btn btn-primary w-100 mt-3">Login</button>
    </form>
    <p class="mt-3 text-center small">No account? <a href="{{ url_for('register') }}">Register here</a></p>
</div>
{% endblock %}
"""

REGISTER_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Register - Briefly{% endblock %}
{% block content %}
<div style="max-width: 450px; margin: 3rem auto; padding: 2rem; background: var(--white-bg); border-radius: 10px;">
    <h2 class="text-center mb-4">Create Account</h2>
    <form method="POST" action="{{ url_for('register') }}">
        <div class="mb-3"><label class="form-label">Full Name</label><input type="text" class="form-control" name="name" required></div>
        <div class="mb-3"><label class="form-label">Username</label><input type="text" class="form-control" name="username" required></div>
        <div class="mb-3"><label class="form-label">Password</label><input type="password" class="form-control" name="password" required></div>
        <button type="submit" class="btn btn-primary w-100 mt-3">Register</button>
    </form>
    <p class="mt-3 text-center small">Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
</div>
{% endblock %}
"""

ERROR_404_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}404 Not Found{% endblock %}{% block content %}<div class='text-center my-5'><h1>404 - Not Found</h1><p>The page you are looking for does not exist.</p><a href='{{url_for("index")}}' class='btn btn-primary'>Go Home</a></div>{% endblock %}"""
ERROR_500_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}500 Error{% endblock %}{% block content %}<div class='text-center my-5'><h1>500 - Server Error</h1><p>Something went wrong. We are looking into it.</p><a href='{{url_for("index")}}' class='btn btn-primary'>Go Home</a></div>{% endblock %}"""


# ==============================================================================
# --- 8. Load templates into the in-memory dictionary loader ---
# ==============================================================================
template_storage['BASE_HTML_TEMPLATE'] = BASE_HTML_TEMPLATE
template_storage['INDEX_HTML_TEMPLATE'] = INDEX_HTML_TEMPLATE
template_storage['ARTICLE_HTML_TEMPLATE'] = ARTICLE_HTML_TEMPLATE
template_storage['LOGIN_HTML_TEMPLATE'] = LOGIN_HTML_TEMPLATE
template_storage['REGISTER_HTML_TEMPLATE'] = REGISTER_HTML_TEMPLATE
template_storage['404_TEMPLATE'] = ERROR_404_TEMPLATE
template_storage['500_TEMPLATE'] = ERROR_500_TEMPLATE

# ==============================================================================
# --- 9. Main Execution Block ---
# ==============================================================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
