# COPY AND PASTE THIS ENTIRE BLOCK INTO Rev14.py ON GITHUB

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
    print("FATAL: NLTK 'punkt' tokenizer not found. Please ensure the 'nltk_data' folder is in your repository.", file=sys.stderr)
    sys.exit("Exiting: Missing critical NLTK data.")

# ==============================================================================
# --- 2. Flask Application Initialization & Configuration ---
# ==============================================================================
app = Flask(__name__)

# --- Restore the original and correct DictLoader for templates ---
template_storage = {}
app.jinja_loader = DictLoader(template_storage)

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a-very-strong-dev-secret-key-that-is-long')
app.config['PER_PAGE'] = 9 # Better for a 3-column grid
app.config['CATEGORIES'] = ['All Articles', 'Community Hub']
app.config['NEWS_API_QUERY'] = (
    '("latest tech trends" OR "AI breakthroughs" OR "market analysis" OR "business innovation" OR "startup funding")'
)
app.config['NEWS_API_DAYS_AGO'] = 7
app.config['NEWS_API_PAGE_SIZE'] = 99
app.config['NEWS_API_SORT_BY'] = 'popularity' # 'relevancy', 'publishedAt'
app.config['CACHE_EXPIRY_SECONDS'] = 3600
app.config['READING_SPEED_WPM'] = 230
app.permanent_session_lifetime = timedelta(days=30)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

# --- Database Configuration ---
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app_data.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ==============================================================================
# --- 3. API Client Initialization ---
# ==============================================================================
NEWSAPI_KEY = os.environ.get('NEWSAPI_KEY')
newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None
if not newsapi: app.logger.error("NEWSAPI_KEY is missing. News fetching will fail.")

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = ChatGroq(model="llama3-8b-8192", groq_api_key=GROQ_API_KEY, temperature=0.1)
        app.logger.info("Groq client initialized.")
    except Exception as e:
        app.logger.error(f"Failed to initialize Groq client: {e}")
else:
    app.logger.warning("GROQ_API_KEY is missing. AI analysis features will be disabled.")

# ==============================================================================
# --- 4. Database Models ---
# ==============================================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    articles = db.relationship('CommunityArticle', backref='author', lazy=True)
    comments = db.relationship('Comment', backref='author', lazy=True)

class CommunityArticle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    article_hash_id = db.Column(db.String(32), unique=True, nullable=False)
    title = db.Column(db.String(250), nullable=False)
    description = db.Column(db.Text, nullable=False)
    full_text = db.Column(db.Text, nullable=False)
    source_name = db.Column(db.String(100), nullable=False)
    image_url = db.Column(db.String(500), nullable=True)
    published_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    groq_summary = db.Column(db.Text, nullable=True)
    groq_takeaways = db.Column(db.Text, nullable=True)
    comments = db.relationship('Comment',
                               foreign_keys='Comment.article_id_str', # Specify the FK column in Comment model
                               backref='community_article_ref', # Changed backref name for clarity
                               lazy=True,
                               cascade="all, delete-orphan")
class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    # This will now store the hash_id for BOTH community articles AND API articles.
    # For community articles, it's linked via ForeignKey. For API articles, it's just stored.
    article_id_str = db.Column(db.String(32), db.ForeignKey('community_article.article_hash_id', name='fk_comment_community_article_hash'), nullable=True)
    # We can remove api_article_hash_id if article_id_str serves both purposes
    # api_article_hash_id = db.Column(db.String(32), nullable=True) # Consider removing

with app.app_context():
    db.create_all()
    app.logger.info("Database tables checked and created.")

# ==============================================================================
# --- 5. Helper Functions ---
# ==============================================================================
MASTER_ARTICLE_STORE, API_CACHE = {}, {}

def generate_article_id(url_or_title): return hashlib.md5(url_or_title.encode('utf-8')).hexdigest()

def jinja_truncate_filter(s, length=120, killwords=False, end='...'):
    # This is the full truncate filter from your original code
    if not s: return ''
    if len(s) <= length: return s
    if killwords: return s[:length - len(end)] + end
    words, result_words, current_length = s.split(), [], 0
    for word in words:
        if current_length + len(word) + (1 if result_words else 0) > length - len(end): break
        result_words.append(word)
        current_length += len(word) + (1 if len(result_words) > 1 else 0)
    if not result_words: return s[:length - len(end)] + end
    return ' '.join(result_words) + end
app.jinja_env.filters['truncate'] = jinja_truncate_filter

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
    return max(1, round(len(text.split()) / app.config['READING_SPEED_WPM']))

@simple_cache(expiry_seconds_default=3600 * 12)
def get_article_analysis_with_groq(article_text, article_title=""):
    if not groq_client or not article_text:
        return {"error": "AI analysis is unavailable."}
    app.logger.info(f"Requesting Groq analysis for: {article_title[:50]}...")
    system_prompt = (
        "You are an expert news analyst. Analyze the following article. "
        "1. Provide a concise, neutral summary (3-4 paragraphs). "
        "2. List 5-7 key takeaways as bullet points. Each takeaway must be a complete sentence. "
        "Format your entire response as a single JSON object with keys 'summary' (string) and 'takeaways' (a list of strings)."
    )
    human_prompt = f"Article Title: {article_title}\n\nArticle Text:\n{article_text[:20000]}"
    try:
        json_model = groq_client.bind(response_format={"type": "json_object"})
        ai_response = json_model.invoke([SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)])
        analysis = json.loads(ai_response.content)
        if 'summary' in analysis and 'takeaways' in analysis:
            return { "groq_summary": analysis.get("summary"), "groq_takeaways": analysis.get("takeaways"), "error": None }
        raise ValueError("Missing 'summary' or 'takeaways' key in Groq JSON.")
    except (json.JSONDecodeError, ValueError, LangChainException) as e:
        app.logger.error(f"Groq analysis failed for '{article_title[:50]}': {e}")
        return {"error": f"AI analysis failed: {e}"}

@simple_cache()
def fetch_news_from_api():
    if not newsapi: return []
    from_date = (datetime.utcnow() - timedelta(days=app.config['NEWS_API_DAYS_AGO'])).strftime('%Y-%m-%d')
    try:
        app.logger.info("Fetching news from NewsAPI.")
        response = newsapi.get_everything(
            q=app.config['NEWS_API_QUERY'], from_param=from_date, language='en',
            sort_by=app.config['NEWS_API_SORT_BY'], page_size=app.config['NEWS_API_PAGE_SIZE']
        )
        processed_articles, unique_titles = [], set()
        for art_data in response.get('articles', []):
            title, url = art_data.get('title'), art_data.get('url')
            if not all([url, title, art_data.get('source')]) or title == '[Removed]' or title.lower() in unique_titles:
                continue
            unique_titles.add(title.lower())
            article_id = generate_article_id(url)
            source_name = art_data['source'].get('name', 'Unknown Source')
            placeholder_text = urllib.parse.quote_plus(source_name[:20])
            standardized_article = {
                'id': article_id, 'title': title, 'description': art_data.get('description', ''),
                'url': url, 'urlToImage': art_data.get('urlToImage') or f'https://via.placeholder.com/400x220/0D2C54/FFFFFF?text={placeholder_text}',
                'publishedAt': art_data.get('publishedAt', ''), 'source': {'name': source_name}, 'is_user_added': False, 'is_community_article': False
            }
            MASTER_ARTICLE_STORE[article_id] = standardized_article
            processed_articles.append(standardized_article)
        return processed_articles
    except NewsAPIException as e:
        app.logger.error(f"NewsAPI error: {e}")
        return []

@simple_cache(expiry_seconds_default=3600 * 6)
def fetch_and_parse_article_content(article_id, url):
    app.logger.info(f"Fetching content for article ID: {article_id}")
    SCRAPER_API_KEY = os.environ.get('SCRAPER_API_KEY')
    if not SCRAPER_API_KEY:
        app.logger.error("SCRAPER_API_KEY is not set. Cannot fetch article content.")
        return {"error": "Content fetching service is unavailable."}
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
            return {"error": "Could not extract text from the article."}
        groq_analysis = get_article_analysis_with_groq(article_scraper.text, article_scraper.title)
        return {
            "full_text": article_scraper.text,
            "read_time_minutes": calculate_read_time(article_scraper.text),
            "groq_analysis": groq_analysis, "error": groq_analysis.get("error")
        }
    except requests.RequestException as e:
        app.logger.error(f"ScraperAPI error for {url}: {e}")
        return {"error": f"Failed to fetch article content."}
    except Exception as e:
        app.logger.error(f"Parsing error for {url}: {e}")
        return {"error": f"Failed to parse article content."}

# ==============================================================================
# --- 6. Flask Routes ---
# ==============================================================================
@app.context_processor
def inject_global_vars():
    return { 'categories': app.config['CATEGORIES'], 'current_year': datetime.utcnow().year, 'session': session }

def get_paginated_articles(articles, page, per_page):
    total = len(articles)
    start = (page - 1) * per_page
    end = start + per_page
    paginated = articles[start:end]
    total_pages = (total + per_page - 1) // per_page
    return paginated, total_pages

@app.route('/')
@app.route('/page/<int:page>')
@app.route('/category/<category_name>')
@app.route('/category/<category_name>/page/<int:page>')
def index(page=1, category_name='All Articles'):
    per_page = app.config['PER_PAGE']
    all_articles = []
    
    if category_name == 'Community Hub':
        community_articles = CommunityArticle.query.order_by(CommunityArticle.published_at.desc()).all()
        # Adapt DB objects to look like the dictionary structure the template expects
        for art in community_articles:
            art.is_community_article = True # Add flag for template
        all_articles = community_articles
    else: # 'All Articles'
        all_articles = fetch_news_from_api()

    display_articles, total_pages = get_paginated_articles(all_articles, page, per_page)
    
    # Logic for featured article on the first page of the main category
    featured_article_on_this_page = (page == 1 and category_name == 'All Articles' and not request.args.get('query'))

    return render_template(
        "INDEX_HTML_TEMPLATE", articles=display_articles, selected_category=category_name,
        current_page=page, total_pages=total_pages, featured_article_on_this_page=featured_article_on_this_page
    )

@app.route('/search')
@app.route('/search/page/<int:page>')
def search_results(page=1):
    query_str = request.args.get('query', '').strip()
    if not query_str:
        return redirect(url_for('index'))

    # Search API articles
    api_results = [art for art_id, art in MASTER_ARTICLE_STORE.items() if query_str.lower() in art.get('title', '').lower()]
    
    # Search community articles
    community_results = CommunityArticle.query.filter(CommunityArticle.title.ilike(f'%{query_str}%')).all()
    for art in community_results:
        art.is_community_article = True
    
    all_results = sorted(api_results + community_results, key=lambda x: x.published_at if hasattr(x, 'published_at') else x.get('publishedAt', ''), reverse=True)
    
    display_articles, total_pages = get_paginated_articles(all_results, page, app.config['PER_PAGE'])
    
    return render_template(
        "INDEX_HTML_TEMPLATE", articles=display_articles, selected_category=f"Search: {query_str}",
        current_page=page, total_pages=total_pages, featured_article_on_this_page=False, query=query_str
    )

@app.route('/article/<article_id>')
def article_detail(article_id):
    is_community_article = article_id.isdigit()
    comments = []
    article_data = None

    if is_community_article:
        article_db_id = int(article_id)
        article_data = CommunityArticle.query.get_or_404(article_db_id)
        # SQLAlchemy handles this through the relationship if lazy='dynamic'
        # comments = article_data.comments.order_by(Comment.timestamp.asc()).all() 
        # Or if lazy='select' (default) or 'joined'
        comments = sorted(article_data.comments, key=lambda c: c.timestamp)
    else:
        article_data = MASTER_ARTICLE_STORE.get(article_id)
        if not article_data: 
            flash("Article not found.", "danger")
            return redirect(url_for('index'))
        # Query comments for API articles using the hash_id
        comments = Comment.query.filter_by(api_article_hash_id=article_id).order_by(Comment.timestamp.asc()).all()

    return render_template("ARTICLE_HTML_TEMPLATE", article=article_data, is_community_article=is_community_article, comments=comments)
@app.route('/get_article_content/<article_id>')
def get_article_content_json(article_id):
    article_data = MASTER_ARTICLE_STORE.get(article_id)
    if not article_data or not article_data.get('url'):
        return jsonify({"error": "Article data or URL not found"}), 404
    processed_content = fetch_and_parse_article_content(article_id, article_data.get('url'))
    MASTER_ARTICLE_STORE[article_id].update(processed_content)
    return jsonify(processed_content)

@app.route('/add_comment/<article_id>', methods=['POST'])
@login_required
def add_comment(article_id):
    content = request.json.get('content', '').strip()
    if not content: return jsonify({"error": "Comment cannot be empty."}), 400
    user = User.query.get(session['user_id'])
    
    is_community_article = article_id.isdigit()
    new_comment_attrs = {
        "content": content,
        "user_id": user.id,
    }

    if is_community_article:
        comm_article_db_id = int(article_id)
        # Verify community article exists
        comm_article = CommunityArticle.query.get(comm_article_db_id)
        if not comm_article:
            return jsonify({"error": "Community article not found."}), 404
        new_comment_attrs["community_article_db_id"] = comm_article_db_id
    else:
        # For API articles, article_id is the hash
        new_comment_attrs["api_article_hash_id"] = article_id
        # You'd also want to check if the API article exists in MASTER_ARTICLE_STORE
        if article_id not in MASTER_ARTICLE_STORE:
             return jsonify({"error": "API article not found."}), 404


    new_comment = Comment(**new_comment_attrs)
    db.session.add(new_comment)
    db.session.commit()
    
    return jsonify({"success": True, "comment": {"content": new_comment.content, "timestamp": new_comment.timestamp.isoformat(), "author": {"name": user.name}}}), 201

@app.route('/post_article', methods=['POST'])
@login_required
def post_article():
    title, description, content, source, image_url = (request.form.get('title'), request.form.get('description'), request.form.get('content'), request.form.get('sourceName'), request.form.get('imageUrl'))
    if not all([title, description, content, source]):
        flash("All fields except Image URL are required.", "danger")
        return redirect(url_for('index')) # Redirecting to a dedicated post page would be better
    
    unique_identifier = title + str(session['user_id']) + str(time.time())
    article_hash = generate_article_id(unique_identifier)
    groq_analysis = get_article_analysis_with_groq(content, title)
    
    new_article = CommunityArticle(
        article_hash_id=article_hash, title=title, description=description, full_text=content,
        source_name=source, image_url=image_url, user_id=session['user_id'],
        groq_summary=groq_analysis.get('groq_summary'),
        groq_takeaways=json.dumps(groq_analysis.get('takeaways')) if groq_analysis.get('takeaways') else None
    )
    db.session.add(new_article)
    db.session.commit()
    flash("Your article has been posted successfully!", "success")
    return redirect(url_for('index', category_name='Community Hub'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        name, username, password = request.form.get('name'), request.form.get('username'), request.form.get('password')
        if not all([name, username, password]) or len(password) < 6:
            flash('All fields are required, and password must be at least 6 characters.', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'warning')
            return redirect(url_for('register'))
        new_user = User(name=name, username=username, password_hash=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        flash(f'Registration successful, {name}! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template("REGISTER_HTML_TEMPLATE")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        username, password = request.form.get('username'), request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session.permanent = True
            session['user_id'], session['user_name'] = user.id, user.name
            flash(f"Welcome back, {user.name}!", "success")
            return redirect(request.args.get('next') or url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template("LOGIN_HTML_TEMPLATE")

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template("404_TEMPLATE"), 404

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"Internal server error (500): {request.url} - {e}", exc_info=True)
    return render_template("500_TEMPLATE"), 500

# ==============================================================================
# --- 7. HTML Templates (Restored to original, full versions) ---
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
        .navbar-main { background: var(--primary-gradient); padding: 0.8rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-bottom: 2px solid rgba(255,255,255,0.15); transition: background 0.3s ease, border-bottom 0.3s ease; height: 95px; display: flex; align-items: center; }
        .navbar-brand-custom { color: white !important; font-weight: 800; font-size: 2.2rem; letter-spacing: 0.5px; font-family: 'Poppins', sans-serif; margin-bottom: 0; display: flex; align-items: center; gap: 12px; }
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
        .article-card, .featured-article, .article-full-content-wrapper, .auth-container { background: var(--white-bg); border-radius: 10px; transition: all 0.3s ease; border: 1px solid var(--card-border-color); box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
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
        .alert-top { position: fixed; top: 85px; left: 50%; transform: translateX(-50%); z-index: 2050; min-width:320px; text-align:center; box-shadow: 0 3px 10px rgba(0,0,0,0.1);}
        .animate-fade-in { animation: fadeIn 0.5s ease-in-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(15px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeInUp { from { opacity: 0; transform: translateY(25px); } to { opacity: 1; transform: translateY(0); } }
        .fade-in-delay-1 { animation-delay: 0.1s; } .fade-in-delay-2 { animation-delay: 0.2s; } .fade-in-delay-3 { animation-delay: 0.3s; }
        .navbar-content-wrapper { display: flex; justify-content: space-between; align-items: center; width: 100%; }
        @media (max-width: 991.98px) { body { padding-top: 185px; } .navbar-main { padding-bottom: 0.5rem; height: auto;} .navbar-content-wrapper { flex-direction: column; align-items: flex-start; gap: 0.5rem; } .navbar-brand-custom { margin-bottom: 0.5rem; } .search-form-container { width: 100%; order: 3; margin-top:0.5rem; padding: 0; } .header-controls { position: absolute; top: 0.9rem; right: 1rem; order: 2; } .category-nav { top: 125px; } }
        @media (max-width: 767.98px) { body { padding-top: 175px; } .category-nav { top: 125px; } .featured-article .row { flex-direction: column; } .featured-image { margin-bottom: 1rem; height: 250px; } }
        @media (max-width: 575.98px) { .navbar-brand-custom { font-size: 1.8rem;} .header-controls { gap: 0.3rem; } .header-btn { padding: 0.4rem 0.8rem; font-size: 0.8rem; } .dark-mode-toggle { font-size: 1rem; } }
        .auth-container { max-width: 450px; margin: 3rem auto; padding: 2rem; }
        .auth-title { text-align: center; color: var(--primary-color); margin-bottom: 1.5rem; font-weight: 700;}
        body.dark-mode .auth-title { color: var(--secondary-color); }
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

    {% if session.get('user_id') %}
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
                <div class="modal-form-group"><label for="articleSource">Source Name (e.g., Your Blog, Company News)</label><input type="text" id="articleSource" name="sourceName" class="modal-form-control" placeholder="Source of this article" value="My Publication" required></div>
                <div class="modal-form-group"><label for="articleImage">Featured Image URL (Optional)</label><input type="url" id="articleImage" name="imageUrl" class="modal-form-control" placeholder="https://example.com/image.jpg"></div>
                <div class="modal-form-group"><label for="articleContent">Full Article Content</label><textarea id="articleContent" name="content" class="modal-form-control" rows="7" placeholder="Write the full article content here..." required></textarea></div>
                <div class="d-flex justify-content-end gap-2"><button type="button" class="btn btn-outline-secondary-modal" id="cancelArticleBtn">Cancel</button><button type="submit" class="btn btn-primary-modal">Save Article</button></div>
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
                        <a href="#" title="Twitter"><i class="fab fa-twitter"></i></a><a href="#" title="Facebook"><i class="fab fa-facebook-f"></i></a><a href="#" title="LinkedIn"><i class="fab fa-linkedin-in"></i></a><a href="#" title="Instagram"><i class="fab fa-instagram"></i></a>
                    </div>
                </div>
                <div class="footer-section">
                    <h5>Quick Links</h5>
                    <div class="footer-links"><a href="{{ url_for('index') }}"><i class="fas fa-angle-right"></i> Home</a><a href="#"><i class="fas fa-angle-right"></i> About Us</a><a href="#"><i class="fas fa-angle-right"></i> Contact</a><a href="#"><i class="fas fa-angle-right"></i> Privacy Policy</a></div>
                </div>
                <div class="footer-section">
                    <h5>Categories</h5>
                    <div class="footer-links">
                        {% for cat_item in categories %}<a href="{{ url_for('index', category_name=cat_item, page=1) }}"><i class="fas fa-angle-right"></i> {{ cat_item }}</a>{% endfor %}
                    </div>
                </div>
                <div class="footer-section">
                    <h5>Newsletter</h5>
                    <p class="small">Subscribe for weekly updates (Feature not implemented).</p>
                    <form class="mt-2"><div class="input-group"><input type="email" class="form-control form-control-sm" placeholder="Your Email" aria-label="Your Email" disabled><button class="btn btn-sm btn-primary-modal" type="submit" disabled>Subscribe</button></div></form>
                </div>
            </div>
            <div class="copyright">&copy; {{ current_year }} Briefly. All rights reserved.</div>
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
            document.cookie = "darkMode=" + theme + ";path=/;max-age=" + (60*60*24*365);
        }
        if(darkModeToggle) { darkModeToggle.addEventListener('click', () => { applyTheme(body.classList.contains('dark-mode') ? 'disabled' : 'enabled'); }); }
        const storedTheme = localStorage.getItem('darkMode');
        if (storedTheme) { applyTheme(storedTheme); } else { updateThemeIcon(); }

        const addArticleBtn = document.getElementById('addArticleBtn');
        const addArticleModal = document.getElementById('addArticleModal');
        const closeModalBtn = document.getElementById('closeModalBtn');
        const cancelArticleBtn = document.getElementById('cancelArticleBtn');
        if(addArticleBtn && addArticleModal) {
            addArticleBtn.addEventListener('click', () => { addArticleModal.style.display = 'flex'; body.style.overflow = 'hidden'; });
            const closeModalFunction = () => { addArticleModal.style.display = 'none'; body.style.overflow = 'auto'; };
            if(closeModalBtn) closeModalBtn.addEventListener('click', closeModalFunction);
            if(cancelArticleBtn) cancelArticleBtn.addEventListener('click', closeModalFunction);
            addArticleModal.addEventListener('click', (e) => { if (e.target === addArticleModal) closeModalFunction(); });
        }
        
        const flashedAlerts = document.querySelectorAll('#alert-placeholder .alert');
        flashedAlerts.forEach(function(alert) { setTimeout(function() { const bsAlert = bootstrap.Alert.getOrCreateInstance(alert); if (bsAlert) bsAlert.close(); }, 7000); });
    });
    </script>
    {% block scripts_extra %}{% endblock %}
</body>
</html>
"""

INDEX_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}
    {% if query %}Search: {{ query|truncate(30) }}{% elif selected_category %}{{selected_category}}{% else %}Home{% endif %} - Briefly
{% endblock %}
{% block content %}
    {% if articles and featured_article_on_this_page %}
    <article class="featured-article p-md-4 p-3 mb-4 animate-fade-in">
        <div class="row g-0 g-md-4">
            <div class="col-lg-6">
                <div class="featured-image rounded overflow-hidden shadow-sm" style="height:320px;">
                    <a href="{{ url_for('article_detail', article_id=articles[0].id if articles[0].is_community_article else articles[0]['id']) }}">
                        <img src="{{ articles[0].image_url if articles[0].is_community_article else articles[0].urlToImage }}" class="img-fluid w-100 h-100" style="object-fit:cover;" alt="Featured: {{ (articles[0].title if articles[0].is_community_article else articles[0].title)|truncate(50) }}">
                    </a>
                </div>
            </div>
            <div class="col-lg-6 d-flex flex-column ps-lg-3 pt-3 pt-lg-0">
                <div class="article-meta mb-2">
                    <span class="badge bg-primary me-2">{{ (articles[0].author.name if articles[0].is_community_article else articles[0].source.name)|truncate(25) }}</span>
                    <span class="meta-item"><i class="far fa-calendar-alt"></i> {{ (articles[0].published_at.strftime('%Y-%m-%d') if articles[0].is_community_article else articles[0].publishedAt.split('T')[0]) if (articles[0].published_at or articles[0].publishedAt) else 'N/A' }}</span>
                </div>
                <h2 class="mb-2 h4"><a href="{{ url_for('article_detail', article_id=articles[0].id if articles[0].is_community_article else articles[0].id) }}" class="text-decoration-none article-title">{{ articles[0].title }}</a></h2>
                <p class="article-description flex-grow-1 small">{{ articles[0].description|truncate(220) }}</p>
                <a href="{{ url_for('article_detail', article_id=articles[0].id if articles[0].is_community_article else articles[0].id) }}" class="read-more mt-auto align-self-start py-2 px-3" style="width:auto;">Read Full Article <i class="fas fa-arrow-right ms-1 small"></i></a>
            </div>
        </div>
    </article>
    {% elif not articles %}
        <div class="alert alert-info text-center my-4 p-4"><h4>No Articles Found</h4><p>No articles were found for this category. Please try another or check back later.</p></div>
    {% endif %}

    <div class="row g-4">
        {% set articles_to_display = (articles[1:] if featured_article_on_this_page and articles else articles) %}
        {% for art in articles_to_display %}
        <div class="col-md-6 col-lg-4 d-flex">
        <article class="article-card animate-fade-in d-flex flex-column w-100" style="animation-delay: {{ loop.index0 * 0.05 }}s">
            <div class="article-image-container">
                <a href="{{ url_for('article_detail', article_id=(art.id if art.is_community_article else art.id)) }}">
                    <img src="{{ art.image_url if art.is_community_article else art.urlToImage }}" class="article-image" alt="{{ art.title|truncate(50) }}">
                </a>
            </div>
            <div class="article-body d-flex flex-column">
                <h5 class="article-title mb-2"><a href="{{ url_for('article_detail', article_id=(art.id if art.is_community_article else art.id)) }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                <div class="article-meta small mb-2">
                    <span class="meta-item text-muted"><i class="fas fa-{{ 'user-edit' if art.is_community_article else 'building' }}"></i> {{ (art.author.name if art.is_community_article else art.source.name)|truncate(20) }}</span>
                    <span class="meta-item text-muted"><i class="far fa-calendar-alt"></i> {{ (art.published_at.strftime('%Y-%m-%d') if art.is_community_article else art.publishedAt.split('T')[0]) if (art.published_at or art.publishedAt) else 'N/A' }}</span>
                </div>
                <p class="article-description small">{{ art.description|truncate(100) }}</p>
                <a href="{{ url_for('article_detail', article_id=(art.id if art.is_community_article else art.id)) }}" class="read-more btn btn-sm mt-auto">Read More <i class="fas fa-chevron-right ms-1 small"></i></a>
            </div>
        </article>
        </div>
        {% endfor %}
    </div>

    {% if total_pages > 1 %}
    <nav aria-label="Page navigation" class="mt-5"><ul class="pagination justify-content-center">
        <li class="page-item page-link-prev-next {% if current_page == 1 %}disabled{% endif %}"><a class="page-link" href="{{ url_for(request.endpoint, page=current_page-1, category_name=selected_category, query=request.args.get('query')) if current_page > 1 else '#' }}">&laquo; Prev</a></li>
        {% for p in range(1, total_pages + 1) %}{% if p == current_page %}<li class="page-item active" aria-current="page"><span class="page-link">{{ p }}</span></li>{% elif p >= current_page - 1 and p <= current_page + 1 %}<li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=p, category_name=selected_category, query=request.args.get('query')) }}">{{ p }}</a></li>{% endif %}{% endfor %}
        <li class="page-item page-link-prev-next {% if current_page == total_pages %}disabled{% endif %}"><a class="page-link" href="{{ url_for(request.endpoint, page=current_page+1, category_name=selected_category, query=request.args.get('query')) if current_page < total_pages else '#' }}">Next &raquo;</a></li>
    </ul></nav>
    {% endif %}
{% endblock %}
"""

ARTICLE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ article.title|truncate(50) }} - Briefly{% endblock %}
{% block head_extra %}
<style>
    .article-full-content-wrapper { background-color: var(--white-bg); padding: 2rem; border-radius: 10px; box-shadow: 0 5px 20px rgba(0,0,0,0.07); margin-bottom: 2rem; margin-top: 1rem; }
    .article-full-content-wrapper .main-article-image { width: 100%; max-height: 480px; object-fit: cover; border-radius: 8px; margin-bottom: 1.5rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .article-title-main {font-weight: 700; color: var(--primary-color); line-height:1.3; font-family: 'Poppins', sans-serif;}
    .article-meta-detailed { font-size: 0.85rem; color: var(--text-muted-color); margin-bottom: 1.5rem; display:flex; flex-wrap:wrap; gap: 0.5rem 1.2rem; align-items:center; border-bottom: 1px solid var(--card-border-color); padding-bottom:1rem; }
    .article-meta-detailed .meta-item i { color: var(--secondary-color); margin-right: 0.4rem; font-size:0.95rem; }
    .summary-box, .takeaways-box { background-color: rgba(var(--primary-color-rgb), 0.04); padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; border: 1px solid rgba(var(--primary-color-rgb), 0.1); }
    .summary-box h5, .takeaways-box h5 { color: var(--primary-color); font-weight: 600; margin-bottom: 0.75rem; font-size:1.1rem; }
    .takeaways-box { border-left: 4px solid var(--secondary-color); }
    .loader-container { display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 200px; padding: 2rem; font-size: 1rem; color: var(--text-muted-color); }
    .loader { border: 5px solid var(--light-bg); border-top: 5px solid var(--primary-color); border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; margin-bottom: 1rem; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .comment-section { margin-top: 3rem; }
    .comment-card { border-bottom: 1px solid var(--card-border-color); padding-bottom: 1rem; margin-bottom: 1rem; }
    .comment-author { font-weight: 600; }
    .comment-date { font-size: 0.8rem; color: var(--text-muted-color); }
</style>
{% endblock %}
{% block content %}
<article class="article-full-content-wrapper animate-fade-in">
    <h1 class="mb-2 article-title-main display-6">{{ article.title }}</h1>
    <div class="article-meta-detailed">
        <span class="meta-item" title="Source"><i class="fas fa-{{ 'user-edit' if is_community_article else 'building' }}"></i> {{ article.author.name if is_community_article else article.source.name }}</span>
        <span class="meta-item" title="Published Date"><i class="far fa-calendar-alt"></i> {{ (article.published_at.strftime('%Y-%m-%d') if is_community_article else article.publishedAt.split('T')[0]) if (article.published_at or article.publishedAt) else 'N/A' }}</span>
        <span class="meta-item" title="Estimated Reading Time" id="articleReadTimeMeta"><i class="far fa-clock"></i> <span id="articleReadTimeText">--</span> min read</span>
    </div>
    {% set image_url = article.image_url if is_community_article else article.urlToImage %}
    {% if image_url %}<img src="{{ image_url }}" alt="{{ article.title|truncate(50) }}" class="main-article-image">{% endif %}

    <div id="contentLoader" class="loader-container my-4 {% if is_community_article %}d-none{% endif %}"><div class="loader"></div><div>Analyzing article and generating summary...</div></div>
    <div id="articleAnalysisContainer">
    {% if is_community_article %}
        {% if article.groq_summary %}
        <div class="summary-box my-3"><h5><i class="fas fa-bookmark me-2"></i>Article Summary (AI Enhanced)</h5><p class="mb-0">{{ article.groq_summary|replace('\\n', '<br>')|safe }}</p></div>
        {% endif %}
        {% if article.groq_takeaways and article.groq_takeaways != 'null' %}
        <div class="takeaways-box my-3"><h5><i class="fas fa-list-check me-2"></i>Key Takeaways (AI Enhanced)</h5>
            <ul>{% for takeaway in article.groq_takeaways|fromjson %}<li>{{ takeaway }}</li>{% endfor %}</ul>
        </div>
        {% endif %}
        <hr><h5 class="mt-4">Full Content</h5><p style="white-space: pre-wrap;">{{ article.full_text }}</p>
    {% else %}
        <div id="apiArticleContent"></div>
    {% endif %}
    </div>

    <section class="comment-section" id="comment-section">
        <h3 class="mb-4">Community Discussion ({{ comments|length }})</h3>
        <div id="comments-list">
            {% for comment in comments %}<div class="comment-card" id="comment-{{comment.id}}"><div class="d-flex justify-content-between"><span class="comment-author">{{ comment.author.name }}</span><span class="comment-date">{{ comment.timestamp.strftime('%b %d, %Y at %I:%M %p') }}</span></div><p class="mt-2 mb-0">{{ comment.content }}</p></div>{% else %}<p id="no-comments-msg">No comments yet. Be the first to share your thoughts!</p>{% endfor %}
        </div>
        {% if session.user_id %}
        <div class="add-comment-form mt-4 pt-3 border-top"><h5 class="mb-3">Leave a Comment</h5><form id="comment-form"><div class="mb-3"><textarea class="form-control" id="comment-content" name="content" rows="4" placeholder="Share your insights..." required></textarea></div><button type="submit" class="btn btn-primary">Post Comment</button></form></div>
        {% else %}
        <div class="alert alert-light mt-4 text-center">Please <a href="{{ url_for('login', next=request.url) }}">log in</a> to join the discussion.</div>
        {% endif %}
    </section>
</article>
{% endblock %}
{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    const isCommunityArticle = {{ is_community_article | tojson }};
    const articleId = {{ (article.id|string) | tojson }};
    const readTimeText = document.getElementById('articleReadTimeText');

    if (isCommunityArticle) {
        const wpm = 230;
        const text = {{ article.full_text|tojson }};
        const wordCount = text ? text.split(/\\s+/).length : 0;
        readTimeText.textContent = Math.max(1, Math.round(wordCount / wpm));
    } else {
        const contentLoader = document.getElementById('contentLoader');
        const apiArticleContent = document.getElementById('apiArticleContent');
        fetch(`{{ url_for('get_article_content_json', article_id=article.id) }}`)
            .then(response => { if (!response.ok) { throw new Error('Network response was not ok'); } return response.json(); })
            .then(data => {
                contentLoader.style.display = 'none';
                if (data.error) { apiArticleContent.innerHTML = `<div class="alert alert-warning">${data.error}</div>`; return; }
                let html = '';
                const analysis = data.groq_analysis;
                if (analysis && analysis.groq_summary) { html += `<div class="summary-box my-3"><h5><i class="fas fa-bookmark me-2"></i>Article Summary (AI Enhanced)</h5><p class="mb-0">${analysis.groq_summary.replace(/\\n/g, '<br>')}</p></div>`; }
                if (analysis && analysis.groq_takeaways && analysis.groq_takeaways.length > 0) { html += `<div class="takeaways-box my-3"><h5><i class="fas fa-list-check me-2"></i>Key Takeaways (AI Enhanced)</h5><ul>${analysis.groq_takeaways.map(t => `<li>${t}</li>`).join('')}</ul></div>`; }
                html += `<a href="{{ article.url }}" class="btn btn-outline-primary" target="_blank" rel="noopener noreferrer">Read Original at {{ article.source.name }} <i class="fas fa-external-link-alt ms-1"></i></a>`;
                apiArticleContent.innerHTML = html;
                readTimeText.textContent = data.read_time_minutes || '--';
            })
            .catch(error => { contentLoader.innerHTML = '<div class="alert alert-danger">Failed to load article analysis. The source may be blocking automated requests.</div>'; console.error("Fetch error:", error); });
    }

    const commentForm = document.getElementById('comment-form');
    if (commentForm) {
        commentForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const content = document.getElementById('comment-content').value;
            const submitButton = this.querySelector('button[type="submit"]');
            submitButton.disabled = true;
            fetch(`{{ url_for('add_comment', article_id=article.id if is_community_article else article.id) }}`, {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ content: content })
            })
            .then(res => res.json())
            .then(data => {
                if(data.success) {
                    const newList = document.getElementById('comments-list');
                    const noComments = document.getElementById('no-comments-msg');
                    if(noComments) noComments.remove();
                    const newCommentEl = document.createElement('div');
                    newCommentEl.className = 'comment-card';
                    newCommentEl.innerHTML = `<div class="d-flex justify-content-between"><span class="comment-author">${data.comment.author.name}</span><span class="comment-date">${new Date(data.comment.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' })}</span></div><p class="mt-2 mb-0">${data.comment.content}</p>`;
                    newList.prepend(newCommentEl);
                    document.getElementById('comment-content').value = '';
                } else { alert('Error: ' + data.error); }
            })
            .finally(() => { submitButton.disabled = false; });
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
    <form method="POST" action="{{ url_for('login', next=request.args.get('next')) }}">
        <div class="modal-form-group"><label for="username" class="form-label">Username</label><input type="text" class="modal-form-control" id="username" name="username" required placeholder="Enter your username"></div>
        <div class="modal-form-group"><label for="password" class="form-label">Password</label><input type="password" class="modal-form-control" id="password" name="password" required placeholder="Enter your password"></div>
        <button type="submit" class="btn btn-primary-modal w-100 mt-3">Login</button>
    </form>
    <p class="mt-3 text-center small">Don't have an account? <a href="{{ url_for('register') }}" class="fw-medium">Register here</a></p>
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
        <div class="modal-form-group"><label for="name" class="form-label">Full Name</label><input type="text" class="modal-form-control" id="name" name="name" required placeholder="Enter your full name"></div>
        <div class="modal-form-group"><label for="username" class="form-label">Username</label><input type="text" class="modal-form-control" id="username" name="username" required placeholder="Choose a username (min 3 chars)"></div>
        <div class="modal-form-group"><label for="password" class="form-label">Password</label><input type="password" class="modal-form-control" id="password" name="password" required placeholder="Create a strong password (min 6 chars)"></div>
        <button type="submit" class="btn btn-primary-modal w-100 mt-3">Register</button>
    </form>
    <p class="mt-3 text-center small">Already have an account? <a href="{{ url_for('login') }}" class="fw-medium">Login here</a></p>
</div>
{% endblock %}
"""

ERROR_404_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}404 Not Found{% endblock %}{% block content %}<div class='container text-center my-5'><h1><i class='fas fa-exclamation-triangle text-warning me-2'></i>404 - Page Not Found</h1><p>The page you are looking for does not exist.</p><a href='{{url_for("index")}}' class='btn btn-primary'>Go to Homepage</a></div>{% endblock %}"""
ERROR_500_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}500 Server Error{% endblock %}{% block content %}<div class='container text-center my-5'><h1><i class='fas fa-cogs text-danger me-2'></i>500 - Internal Server Error</h1><p>Something went wrong on our end. We're looking into it.</p><a href='{{url_for("index")}}' class='btn btn-primary'>Go to Homepage</a></div>{% endblock %}"""


# ==============================================================================
# --- 8. Add all templates to the template_storage dictionary ---
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
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)
