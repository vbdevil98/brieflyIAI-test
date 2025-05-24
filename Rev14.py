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
from flask import (Flask, render_template_string, render_template, url_for,
                   redirect, request, jsonify, session, flash)
from flask_sqlalchemy import SQLAlchemy
from jinja2 import DictLoader # <-- Re-importing the correct loader
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
# --- 1. NLTK 'punkt' Tokenizer Setup (Robust Version) ---
# ==============================================================================
try:
    project_root = os.path.dirname(os.path.abspath(__file__))
    local_nltk_data_path = os.path.join(project_root, 'nltk_data')
    if local_nltk_data_path not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_data_path)
    nltk.data.find('tokenizers/punkt', paths=[local_nltk_data_path])
    print("NLTK 'punkt' tokenizer found in project's local nltk_data directory.", file=sys.stderr)
except LookupError:
    print("FATAL: NLTK 'punkt' tokenizer not found.", file=sys.stderr)
    print("Please download it to your local project by running this command:", file=sys.stderr)
    print("python -m nltk.downloader punkt -d ./nltk_data", file=sys.stderr)
    print("Then, commit the 'nltk_data' folder to your GitHub repository.", file=sys.stderr)
    sys.exit("Exiting: Missing critical NLTK data.")


# ==============================================================================
# --- 2. Flask Application Initialization & Configuration ---
# ==============================================================================
app = Flask(__name__)

# --- FIX: Restoring the original DictLoader for templates ---
template_storage = {}
app.jinja_loader = DictLoader(template_storage)
# --- End of Fix ---

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a-strong-dev-secret-key')
app.config['PER_PAGE'] = 10
app.config['CATEGORIES'] = ['All Articles', 'Community Hub']
app.config['NEWS_API_QUERY'] = (
    '("latest tech trends" OR "AI breakthroughs" OR "market analysis" OR "business innovation") '
    'AND NOT (celebrity OR gossip OR sports)'
)
app.config['NEWS_API_DAYS_AGO'] = 7
app.config['NEWS_API_PAGE_SIZE'] = 100
app.config['NEWS_API_SORT_BY'] = 'popularity'
app.config['SUMMARY_SENTENCES'] = 3
app.config['CACHE_EXPIRY_SECONDS'] = 3600
app.config['READING_SPEED_WPM'] = 230
app.permanent_session_lifetime = timedelta(days=30)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app_data.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ==============================================================================
# --- 3. API Client Initialization (NewsAPI, Groq) ---
# ==============================================================================
NEWSAPI_KEY = os.environ.get('NEWSAPI_KEY')
newsapi = NewsApiClient(api_key=NEWSAPI_KEY) if NEWSAPI_KEY else None
if not newsapi:
    app.logger.error("CRITICAL: NEWSAPI_KEY is missing. News fetching will fail.")

GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
groq_client = None
if GROQ_API_KEY:
    try:
        groq_client = ChatGroq(model="llama3-8b-8192", groq_api_key=GROQ_API_KEY, temperature=0.1)
        app.logger.info("Groq Chat client initialized successfully.")
    except Exception as e:
        app.logger.error(f"Failed to initialize Groq client: {e}")
else:
    app.logger.warning("GROQ_API_KEY is missing. AI analysis features will be disabled.")

# ==============================================================================
# --- 4. Database Models (Users, Community Articles, Comments) ---
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
    comments = db.relationship('Comment', backref='article', lazy=True, cascade="all, delete-orphan")

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    article_id_str = db.Column(db.String(32), db.ForeignKey('community_article.article_hash_id'), nullable=True)
    api_article_hash_id = db.Column(db.String(32), nullable=True)

with app.app_context():
    db.create_all()
    app.logger.info("Database tables checked and created if necessary.")

# ==============================================================================
# --- 5. Global Stores & Helper Functions ---
# ==============================================================================
MASTER_ARTICLE_STORE = {}
API_CACHE = {}

def generate_article_id(url_or_title):
    return hashlib.md5(url_or_title.encode('utf-8')).hexdigest()

def jinja_truncate_filter(s, length=120):
    if not s or len(s) <= length:
        return s
    return s[:length-3] + '...'
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
    if not text or not isinstance(text, str): return 0
    num_words = len(text.split())
    return max(1, round(num_words / app.config['READING_SPEED_WPM']))

@simple_cache(expiry_seconds_default=3600 * 12)
def get_article_analysis_with_groq(article_text, article_title=""):
    if not groq_client or not article_text:
        return {"error": "Groq client not available or no text provided."}
    app.logger.info(f"Requesting Groq analysis for: {article_title[:50]}...")
    truncated_text = article_text[:20000]
    system_prompt = (
        "You are an expert news analyst. Analyze the following article. "
        "1. Provide a concise, neutral summary (3-4 paragraphs). "
        "2. List 5-7 key takeaways as bullet points. Each takeaway must be a complete sentence. "
        "Format your entire response as a single JSON object with keys 'summary' (string) and 'takeaways' (a list of strings)."
    )
    human_prompt = f"Article Title: {article_title}\n\nArticle Text:\n{truncated_text}"
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
        app.logger.info(f"Fetching news from NewsAPI.")
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
                'publishedAt': art_data.get('publishedAt', ''), 'source': {'name': source_name}, 'is_user_added': False
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
        return {"error": "ScraperAPI key not configured."}
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
            return {"error": "Newspaper3k could not extract text."}
        groq_analysis = get_article_analysis_with_groq(article_scraper.text, article_scraper.title)
        return {
            "full_text": article_scraper.text,
            "read_time_minutes": calculate_read_time(article_scraper.text),
            "groq_analysis": groq_analysis, "error": groq_analysis.get("error")
        }
    except requests.RequestException as e:
        return {"error": f"Failed to fetch article content: {e}"}
    except Exception as e:
        return {"error": f"Failed to parse article content: {e}"}

# ==============================================================================
# --- 6. Flask Routes ---
# ==============================================================================
@app.context_processor
def inject_global_vars():
    return { 'categories': app.config['CATEGORIES'], 'current_year': datetime.utcnow().year, 'session': session }

@app.route('/')
@app.route('/page/<int:page>')
@app.route('/category/<category_name>')
@app.route('/category/<category_name>/page/<int:page>')
def index(page=1, category_name='All Articles'):
    per_page = app.config['PER_PAGE']
    if category_name == 'Community Hub':
        query = CommunityArticle.query.order_by(CommunityArticle.published_at.desc())
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        display_articles, total_pages = pagination.items, pagination.pages
    else:
        all_api_articles = fetch_news_from_api()
        total_articles = len(all_api_articles)
        start_index = (page - 1) * per_page
        display_articles = all_api_articles[start_index:start_index + per_page]
        total_pages = (total_articles + per_page - 1) // per_page
    if page > total_pages and total_pages > 0:
        return redirect(url_for('index', category_name=category_name, page=total_pages))
    # --- FIX: Using render_template instead of render_template_string ---
    return render_template(
        "INDEX_HTML_TEMPLATE", articles=display_articles,
        selected_category=category_name, current_page=page, total_pages=total_pages
    )

@app.route('/article/<article_id>')
def article_detail(article_id):
    is_community_article = article_id.isdigit()
    if is_community_article:
        article_data = CommunityArticle.query.get_or_404(int(article_id))
        comments = Comment.query.filter_by(article_id_str=article_data.article_hash_id).order_by(Comment.timestamp.asc()).all()
    else:
        article_data = MASTER_ARTICLE_STORE.get(article_id)
        if not article_data:
            flash("Article not found.", "danger")
            return redirect(url_for('index'))
        comments = Comment.query.filter_by(api_article_hash_id=article_id).order_by(Comment.timestamp.asc()).all()
    # --- FIX: Using render_template instead of render_template_string ---
    return render_template(
        "ARTICLE_HTML_TEMPLATE", article=article_data,
        is_community_article=is_community_article, comments=comments
    )

@app.route('/get_article_content/<article_id>')
def get_article_content_json(article_id):
    article_data = MASTER_ARTICLE_STORE.get(article_id)
    if not article_data:
        return jsonify({"error": "Article not found"}), 404
    processed_content = fetch_and_parse_article_content(article_id, article_data.get('url'))
    MASTER_ARTICLE_STORE[article_id].update(processed_content)
    return jsonify(processed_content)

@app.route('/add_comment/<article_id>', methods=['POST'])
@login_required
def add_comment(article_id):
    content = request.json.get('content', '').strip()
    if not content:
        return jsonify({"error": "Comment cannot be empty."}), 400
    user = User.query.get(session['user_id'])
    if not user: return jsonify({"error": "User not found."}), 401
    is_community_article = article_id.isdigit()
    new_comment = Comment(
        content=content, user_id=user.id,
        api_article_hash_id=None if is_community_article else article_id,
        article_id_str=CommunityArticle.query.get(int(article_id)).article_hash_id if is_community_article else None
    )
    db.session.add(new_comment)
    db.session.commit()
    app.logger.info(f"User '{user.username}' added comment to article '{article_id}'")
    return jsonify({
        "success": True, "comment": {
            "content": new_comment.content, "timestamp": new_comment.timestamp.isoformat(),
            "author": {"name": user.name}
        }
    }), 201

@app.route('/post_article', methods=['GET', 'POST'])
@login_required
def post_article():
    if request.method == 'POST':
        title, description, content, source, image_url = (request.form.get('title'), request.form.get('description'), request.form.get('content'), request.form.get('sourceName'), request.form.get('imageUrl'))
        if not all([title, description, content, source]):
            flash("All fields except Image URL are required.", "danger")
            return redirect(url_for('post_article'))
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
    # --- FIX: Using render_template instead of render_template_string ---
    return render_template("POST_ARTICLE_TEMPLATE")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session: return redirect(url_for('index'))
    if request.method == 'POST':
        name, username, password = request.form.get('name'), request.form.get('username'), request.form.get('password')
        if not all([name, username, password]):
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'warning')
            return redirect(url_for('register'))
        new_user = User(name=name, username=username, password_hash=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        app.logger.info(f"New user registered: {username}")
        flash(f'Registration successful, {name}! Please log in.', 'success')
        return redirect(url_for('login'))
    # --- FIX: Using render_template instead of render_template_string ---
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
            app.logger.info(f"User '{username}' logged in successfully.")
            flash(f"Welcome back, {user.name}!", "success")
            return redirect(request.args.get('next') or url_for('index'))
        else:
            flash('Invalid username or password.', 'danger')
    # --- FIX: Using render_template instead of render_template_string ---
    return render_template("LOGIN_HTML_TEMPLATE")

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    # --- FIX: Using render_template instead of render_template_string ---
    return render_template("404_TEMPLATE"), 404

@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"Internal server error (500): {request.url} - {e}", exc_info=True)
    # --- FIX: Using render_template instead of render_template_string ---
    return render_template("500_TEMPLATE"), 500
# ==============================================================================
# --- 7. HTML Templates ---
# ==============================================================================

BASE_HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{% block title %}Briefly{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary-color: #0A2342;
            --primary-light: #1E3A5E;
            --secondary-color: #B8860B;
            --secondary-light: #D4A017;
            --text-color: #343a40;
            --text-muted-color: #6c757d;
            --light-bg: #F8F9FA;
            --white-bg: #FFFFFF;
            --card-border-color: #E0E0E0;
            --footer-bg: #061A30;
            --primary-gradient: linear-gradient(135deg, var(--primary-color), var(--primary-light));
            --primary-color-rgb: 10, 35, 66;
            --secondary-color-rgb: 184, 134, 11;
        }
        body {
            padding-top: 95px;
            font-family: 'Roboto', sans-serif;
            background-color: var(--light-bg);
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        .main-content { flex-grow: 1; }
        .navbar-main {
            background: var(--primary-gradient);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-bottom: 2px solid rgba(255,255,255,0.1);
            height: 95px;
            display: flex;
            align-items: center;
        }
        .navbar-brand-custom {
            color: white !important; font-weight: 800; font-size: 2.2rem;
            font-family: 'Poppins', sans-serif;
            display: flex; align-items: center; gap: 12px;
        }
        .navbar-brand-custom .brand-icon { color: var(--secondary-light); font-size: 2.5rem; }
        .header-controls { display: flex; gap: 0.8rem; align-items: center; }
        .header-btn {
            background: transparent; border: 1px solid rgba(255,255,255,0.3);
            padding: 0.5rem 1rem; border-radius: 20px;
            color: white; font-weight: 500; transition: all 0.3s ease;
            display: flex; align-items: center; gap: 0.5rem; text-decoration:none; font-size: 0.9rem;
        }
        .header-btn:hover { background: var(--secondary-color); border-color: var(--secondary-color); color: var(--primary-color); }
        .header-btn-filled { background: var(--secondary-color); border-color: var(--secondary-color); color: var(--primary-color); }
        .header-btn-filled:hover { background: var(--secondary-light); }

        .category-nav {
            background: var(--white-bg);
            box-shadow: 0 3px 10px rgba(0,0,0,0.03);
            position: sticky; top: 95px;
            width: 100%; z-index: 1020; border-bottom: 1px solid var(--card-border-color);
        }
        .categories-wrapper { display: flex; justify-content: center; align-items: center; width: 100%; overflow-x: auto; padding: 0.4rem 0;}
        .category-link {
            color: var(--primary-color) !important; font-weight: 600;
            padding: 0.6rem 1.3rem !important; border-radius: 20px;
            transition: all 0.25s ease; white-space: nowrap; text-decoration: none; margin: 0 0.3rem;
            font-size: 0.9rem; border: 1px solid transparent;
        }
        .category-link.active {
            background: var(--primary-color) !important; color: white !important;
            box-shadow: 0 3px 10px rgba(var(--primary-color-rgb), 0.2);
        }
        .category-link:hover:not(.active) { background: var(--light-bg) !important; color: var(--secondary-color) !important; border-color: var(--secondary-color); }

        .article-card, .article-full-content-wrapper, .auth-container {
            background: var(--white-bg); border-radius: 10px;
            transition: all 0.3s ease; border: 1px solid var(--card-border-color);
            box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        }
        .article-card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.08); }
        .article-image-container { height: 200px; overflow: hidden; border-top-left-radius: 9px; border-top-right-radius: 9px;}
        .article-image { width: 100%; height: 100%; object-fit: cover; transition: transform 0.4s ease; }
        .article-card:hover .article-image { transform: scale(1.08); }
        .article-title { font-weight: 700; line-height: 1.35; margin-bottom: 0.6rem; font-size:1.1rem; }
        .article-title a { color: var(--primary-color); text-decoration: none; }
        .article-card:hover .article-title a { color: var(--secondary-color) !important; }
        .article-meta { display: flex; align-items: center; margin-bottom: 0.8rem; flex-wrap: wrap; gap: 0.4rem 1rem; }
        .meta-item { display: flex; align-items: center; font-size: 0.8rem; color: var(--text-muted-color); }
        .meta-item i { font-size: 0.9rem; margin-right: 0.3rem; color: var(--secondary-color); }
        .alert-top { position: fixed; top: 110px; left: 50%; transform: translateX(-50%); z-index: 2050; min-width:320px; text-align:center; }
        footer { background: var(--footer-bg); color: rgba(255,255,255,0.8); margin-top: auto; padding: 3rem 0 1.5rem; font-size:0.9rem; }
        .copyright { text-align: center; padding-top: 1.5rem; margin-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.85rem; color: rgba(255,255,255,0.6); }

        @media (max-width: 991.98px) {
            body { padding-top: 80px; }
            .navbar-main { height: 80px; }
            .category-nav { top: 80px; }
            .navbar-brand-custom { font-size: 1.8rem; }
            .header-btn { padding: 0.4rem 0.8rem; font-size: 0.8rem; }
        }
    </style>
    {% block head_extra %}{% endblock %}
</head>
<body>
    <nav class="navbar navbar-main navbar-expand-lg fixed-top">
        <div class="container">
            <a class="navbar-brand-custom" href="{{ url_for('index') }}">
                <i class="fas fa-bolt-lightning brand-icon"></i>
                <span>Briefly</span>
            </a>
            <div class="header-controls">
                {% if session.get('user_id') %}
                    <a href="{{ url_for('post_article') }}" class="header-btn header-btn-filled d-none d-md-flex" title="Post a new article">
                        <i class="fas fa-plus"></i> <span class="d-none d-lg-inline">Post Article</span>
                    </a>
                    <span class="text-white me-2 d-none d-md-inline">Hi, {{ session.get('user_name', 'User') }}!</span>
                    <a href="{{ url_for('logout') }}" class="header-btn" title="Logout">
                        <i class="fas fa-sign-out-alt"></i> <span class="d-none d-sm-inline">Logout</span>
                    </a>
                {% else %}
                    <a href="{{ url_for('login') }}" class="header-btn" title="Login">
                        <i class="fas fa-user"></i> <span class="d-none d-sm-inline">Login</span>
                    </a>
                    <a href="{{ url_for('register') }}" class="header-btn header-btn-filled" title="Register">
                        <i class="fas fa-user-plus"></i> <span class="d-none d-sm-inline">Register</span>
                    </a>
                {% endif %}
            </div>
        </div>
    </nav>

    <nav class="category-nav">
        <div class="container">
            <div class="categories-wrapper">
                {% for cat_item in categories %}
                    <a href="{{ url_for('index', category_name=cat_item, page=1) }}"
                       class="category-link {% if selected_category == cat_item %}active{% endif %}">
                       <i class="fas fa-{% if cat_item == 'All Articles' %}globe-americas{% elif cat_item == 'Community Hub' %}users{% endif %} me-1 d-none d-sm-inline"></i>
                       {{ cat_item }}
                    </a>
                {% endfor %}
            </div>
        </div>
    </nav>
    
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

    <main class="container main-content my-4">
        {% block content %}{% endblock %}
    </main>

    <footer class="mt-auto">
        <div class="container">
            <div class="copyright">&copy; {{ current_year }} Briefly. All rights reserved.</div>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts_extra %}{% endblock %}
</body>
</html>
"""

INDEX_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}
    {% if selected_category %}{{selected_category}}{% else %}Home{% endif %} - Briefly
{% endblock %}

{% block content %}
    {% if not articles %}
        <div class="text-center p-5 rounded bg-light">
            <h4>
                {% if selected_category == 'Community Hub' %}
                    <i class="fas fa-users me-2"></i>The Community Hub is Quiet
                {% else %}
                    <i class="fas fa-newspaper me-2"></i>No Articles Found
                {% endif %}
            </h4>
            <p class="text-muted">
                {% if selected_category == 'Community Hub' %}
                    No articles have been posted by the community yet. Be the first!
                {% else %}
                     We couldn't find any news articles right now. Please check back later.
                {% endif %}
            </p>
            {% if selected_category == 'Community Hub' and session.user_id %}
                <a href="{{ url_for('post_article') }}" class="btn btn-primary mt-2">Post an Article</a>
            {% endif %}
        </div>
    {% else %}
    <div class="row g-4">
        {% for art in articles %}
        <div class="col-md-6 col-lg-4 d-flex">
        <article class="article-card d-flex flex-column w-100">
            {% set is_community = art.__class__.__name__ == 'CommunityArticle' %}
            <div class="article-image-container">
                <a href="{{ url_for('article_detail', article_id=art.id) }}">
                    <img src="{{ art.image_url if is_community else art.urlToImage }}"
                         class="article-image"
                         alt="{{ art.title | truncate(50) }}">
                </a>
            </div>
            <div class="article-body d-flex flex-column flex-grow-1 p-3">
                <h5 class="article-title mb-2">
                    <a href="{{ url_for('article_detail', article_id=art.id) }}" class="text-decoration-none">
                        {{ art.title | truncate(70) }}
                    </a>
                </h5>

                <div class="article-meta small mb-2">
                    <span class="meta-item text-muted">
                        {% if is_community %}
                            <i class="fas fa-user-edit"></i> {{ art.author.name | truncate(20) }}
                        {% else %}
                            <i class="fas fa-building"></i> {{ art.source.name | truncate(20) }}
                        {% endif %}
                    </span>
                    <span class="meta-item text-muted">
                        <i class="far fa-calendar-alt"></i> 
                        {{ art.published_at.strftime('%Y-%m-%d') if is_community else art.publishedAt.split('T')[0] if art.publishedAt else 'N/A' }}
                    </span>
                </div>

                <p class="small text-muted flex-grow-1">
                    {{ art.description | truncate(120) }}
                </p>
                <a href="{{ url_for('article_detail', article_id=art.id) }}" class="btn btn-sm btn-outline-primary mt-auto">
                    Read More <i class="fas fa-chevron-right ms-1 small"></i>
                </a>
            </div>
        </article>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    {# Pagination Controls #}
    {% if total_pages and total_pages > 1 %}
    <nav aria-label="Page navigation" class="mt-5">
        <ul class="pagination justify-content-center">
            <li class="page-item {% if current_page == 1 %}disabled{% endif %}">
                <a class="page-link" href="{{ url_for('index', page=current_page-1, category_name=selected_category) }}">Previous</a>
            </li>
            {% for p in range(1, total_pages + 1) %}
                <li class="page-item {% if p == current_page %}active{% endif %}">
                    <a class="page-link" href="{{ url_for('index', page=p, category_name=selected_category) }}">{{ p }}</a>
                </li>
            {% endfor %}
            <li class="page-item {% if current_page == total_pages %}disabled{% endif %}">
                <a class="page-link" href="{{ url_for('index', page=current_page+1, category_name=selected_category) }}">Next</a>
            </li>
        </ul>
    </nav>
    {% endif %}
{% endblock %}
"""

ARTICLE_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}{{ article.title|truncate(50) if article else "Article" }} - Briefly{% endblock %}

{% block head_extra %}
<style>
    .article-full-content-wrapper { padding: 2rem; margin-top: 1rem; }
    .main-article-image { width: 100%; max-height: 480px; object-fit: cover; border-radius: 8px; margin-bottom: 1.5rem; }
    .article-title-main { font-weight: 700; color: var(--primary-color); line-height: 1.3; font-family: 'Poppins', sans-serif; }
    .article-meta-detailed { font-size: 0.85rem; color: var(--text-muted-color); margin-bottom: 1.5rem; display:flex; flex-wrap:wrap; gap: 0.5rem 1.2rem; align-items:center; border-bottom: 1px solid var(--card-border-color); padding-bottom:1rem; }
    .summary-box, .takeaways-box { background-color: rgba(var(--primary-color-rgb), 0.04); padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; }
    .takeaways-box { border-left: 4px solid var(--secondary-color); }
    h5.analysis-title { color: var(--primary-color); font-weight: 600; }
    .loader-container { display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 200px; }
    .loader { border: 5px solid var(--light-bg); border-top: 5px solid var(--primary-color); border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .comment-section { margin-top: 3rem; }
    .comment-card { background-color: var(--light-bg); border: 1px solid var(--card-border-color); padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
    .comment-author { font-weight: 600; }
    .comment-date { font-size: 0.8rem; color: var(--text-muted-color); }
</style>
{% endblock %}

{% block content %}
<article class="article-full-content-wrapper">
    <h1 class="mb-3 article-title-main display-6">{{ article.title }}</h1>
    <div class="article-meta-detailed">
        <span class="meta-item">
            {% if is_community_article %}
                <i class="fas fa-user-edit"></i> {{ article.author.name }}
            {% else %}
                <i class="fas fa-building"></i> {{ article.source.name }}
            {% endif %}
        </span>
        <span class="meta-item"><i class="far fa-calendar-alt"></i> {{ (article.published_at.strftime('%Y-%m-%d') if is_community_article else article.publishedAt.split('T')[0]) if (article.published_at or article.publishedAt) else 'N/A' }}</span>
        <span class="meta-item" id="readTimeMeta"><i class="far fa-clock"></i> <span id="readTimeText">--</span> min read</span>
    </div>

    {% set image_src = article.image_url if is_community_article else article.urlToImage %}
    {% if image_src %}
        <img src="{{ image_src }}" alt="{{ article.title|truncate(50) }}" class="main-article-image">
    {% endif %}

    <div id="loader" class="loader-container my-4 {% if is_community_article %}d-none{% endif %}">
        <div class="loader"></div>
        <p class="mt-2 text-muted">Analyzing article and generating summary...</p>
    </div>

    <div id="analysisContent">
        {% if is_community_article %}
            {% if article.groq_summary %}
            <div class="summary-box">
                <h5 class="analysis-title"><i class="fas fa-bookmark me-2"></i>AI Summary</h5>
                <p>{{ article.groq_summary | safe }}</p>
            </div>
            {% endif %}
            {% if article.groq_takeaways and article.groq_takeaways != 'null' %}
            <div class="takeaways-box">
                <h5 class="analysis-title"><i class="fas fa-list-check me-2"></i>AI Key Takeaways</h5>
                <ul>
                    {% for takeaway in article.groq_takeaways | fromjson %}
                        <li>{{ takeaway }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
            <hr class="my-4">
            <h5><i class="fas fa-file-alt me-2"></i>Original Content</h5>
            <div style="white-space: pre-wrap;">{{ article.full_text }}</div>
        {% endif %}
    </div>
    
    {% if not is_community_article %}
    <a href="{{ article.url }}" class="btn btn-primary" target="_blank" rel="noopener noreferrer">
        Read Original Article <i class="fas fa-external-link-alt ms-1"></i>
    </a>
    {% endif %}

    <section class="comment-section" id="comment-section">
        <h3 class="mb-3">Community Discussion ({{ comments|length }})</h3>
        <div id="comments-list">
            {% for comment in comments %}
                <div class="comment-card">
                    <div class="d-flex justify-content-between">
                        <span class="comment-author">{{ comment.author.name }}</span>
                        <span class="comment-date">{{ comment.timestamp.strftime('%b %d, %Y %I:%M %p') }}</span>
                    </div>
                    <p class="mt-2 mb-0">{{ comment.content }}</p>
                </div>
            {% else %}
                <p id="no-comments-msg">No comments yet. Be the first to share your thoughts!</p>
            {% endfor %}
        </div>
        
        {% if session.user_id %}
        <div class="add-comment-form mt-4">
            <h5>Leave a Comment</h5>
            <form id="comment-form">
                <div class="mb-3">
                    <textarea class="form-control" id="comment-content" name="content" rows="4" placeholder="Share your insights..." required></textarea>
                </div>
                <button type="submit" class="btn btn-success">Post Comment</button>
            </form>
        </div>
        {% else %}
        <div class="alert alert-info mt-4">
            <a href="{{ url_for('login', next=request.url) }}">Log in</a> to join the discussion.
        </div>
        {% endif %}
    </section>

</article>
{% endblock %}

{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    const isCommunityArticle = {{ is_community_article | tojson }};
    const articleId = {{ (article.id | string) | tojson }};
    
    // --- Article Content Loading for API Articles ---
    if (!isCommunityArticle) {
        const loader = document.getElementById('loader');
        const analysisContainer = document.getElementById('analysisContent');
        const readTimeText = document.getElementById('readTimeText');
        
        fetch(`/get_article_content/${articleId}`)
            .then(response => response.json())
            .then(data => {
                loader.style.display = 'none';
                if (data.error) {
                    analysisContainer.innerHTML = `<div class="alert alert-warning">${data.error}</div>`;
                    return;
                }
                
                let html = '';
                const analysis = data.groq_analysis;
                if (analysis && analysis.groq_summary) {
                    html += `
                    <div class="summary-box">
                        <h5 class="analysis-title"><i class="fas fa-bookmark me-2"></i>AI Summary</h5>
                        <p>${analysis.groq_summary.replace(/\\n/g, '<br>')}</p>
                    </div>`;
                }
                if (analysis && analysis.groq_takeaways && analysis.groq_takeaways.length > 0) {
                    html += `
                    <div class="takeaways-box">
                        <h5 class="analysis-title"><i class="fas fa-list-check me-2"></i>AI Key Takeaways</h5>
                        <ul>${analysis.groq_takeaways.map(t => `<li>${t}</li>`).join('')}</ul>
                    </div>`;
                }
                analysisContainer.innerHTML = html;
                readTimeText.textContent = data.read_time_minutes || '--';
            })
            .catch(error => {
                loader.style.display = 'none';
                analysisContainer.innerHTML = `<div class="alert alert-danger">Could not load article analysis.</div>`;
                console.error("Error fetching article content:", error);
            });
    } else {
         // For community articles, just calculate and display read time
         const fullText = {{ article.full_text | tojson }};
         const wpm = {{ config.READING_SPEED_WPM }};
         const wordCount = fullText.split(/\\s+/).length;
         const readTime = Math.max(1, Math.round(wordCount / wpm));
         document.getElementById('readTimeText').textContent = readTime;
    }

    // --- Comment Form Submission ---
    const commentForm = document.getElementById('comment-form');
    if (commentForm) {
        commentForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const content = document.getElementById('comment-content').value;
            const submitBtn = this.querySelector('button[type="submit"]');
            submitBtn.disabled = true;

            fetch(`/add_comment/${articleId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: content })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    const commentsList = document.getElementById('comments-list');
                    const noCommentsMsg = document.getElementById('no-comments-msg');
                    if (noCommentsMsg) noCommentsMsg.remove();
                    
                    const newComment = document.createElement('div');
                    newComment.className = 'comment-card';
                    const commentDate = new Date(data.comment.timestamp).toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: '2-digit', minute: '2-digit' });

                    newComment.innerHTML = `
                        <div class="d-flex justify-content-between">
                            <span class="comment-author">${data.comment.author.name}</span>
                            <span class="comment-date">${commentDate}</span>
                        </div>
                        <p class="mt-2 mb-0">${data.comment.content}</p>
                    `;
                    commentsList.prepend(newComment); // Add to top
                    document.getElementById('comment-content').value = '';
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => console.error('Error posting comment:', error))
            .finally(() => {
                submitBtn.disabled = false;
            });
        });
    }
});
</script>
{% endblock %}
"""

POST_ARTICLE_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Post an Article - Briefly{% endblock %}
{% block content %}
<div class="article-full-content-wrapper mx-auto" style="max-width: 800px;">
    <h2 class="mb-4">Post to the Community Hub</h2>
    <form method="POST" action="{{ url_for('post_article') }}">
        <div class="mb-3">
            <label for="title" class="form-label">Article Title</label>
            <input type="text" class="form-control" id="title" name="title" required>
        </div>
        <div class="mb-3">
            <label for="description" class="form-label">Short Description (for homepage card)</label>
            <textarea class="form-control" id="description" name="description" rows="3" required></textarea>
        </div>
        <div class="mb-3">
            <label for="sourceName" class="form-label">Source Name (e.g., Your Blog, Personal Research)</label>
            <input type="text" class="form-control" id="sourceName" name="sourceName" required>
        </div>
        <div class="mb-3">
            <label for="imageUrl" class="form-label">Image URL (Optional)</label>
            <input type="url" class="form-control" id="imageUrl" name="imageUrl" placeholder="https://example.com/image.jpg">
        </div>
        <div class="mb-3">
            <label for="content" class="form-label">Full Article Content</label>
            <textarea class="form-control" id="content" name="content" rows="15" required></textarea>
        </div>
        <button type="submit" class="btn btn-primary">Publish Article</button>
    </form>
</div>
{% endblock %}
"""

LOGIN_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Login - Briefly{% endblock %}
{% block content %}
<div class="auth-container article-card mx-auto mt-5">
    <h2 class="text-center mb-4">Member Login</h2>
    <form method="POST" action="{{ url_for('login', next=request.args.get('next')) }}">
        <div class="mb-3">
            <label for="username" class="form-label">Username</label>
            <input type="text" class="form-control" id="username" name="username" required>
        </div>
        <div class="mb-3">
            <label for="password" class="form-label">Password</label>
            <input type="password" class="form-control" id="password" name="password" required>
        </div>
        <button type="submit" class="btn btn-primary w-100">Login</button>
    </form>
    <p class="mt-3 text-center small">
        Don't have an account? <a href="{{ url_for('register') }}">Register here</a>
    </p>
</div>
{% endblock %}
"""

REGISTER_HTML_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}Register - Briefly{% endblock %}
{% block content %}
<div class="auth-container article-card mx-auto mt-5">
    <h2 class="text-center mb-4">Create Account</h2>
    <form method="POST" action="{{ url_for('register') }}">
        <div class="mb-3">
            <label for="name" class="form-label">Full Name</label>
            <input type="text" class="form-control" id="name" name="name" required>
        </div>
        <div class="mb-3">
            <label for="username" class="form-label">Username</label>
            <input type="text" class="form-control" id="username" name="username" required>
        </div>
        <div class="mb-3">
            <label for="password" class="form-label">Password</label>
            <input type="password" class="form-control" id="password" name="password" required>
        </div>
        <button type="submit" class="btn btn-primary w-100">Register</button>
    </form>
    <p class="mt-3 text-center small">
        Already have an account? <a href="{{ url_for('login') }}">Login here</a>
    </p>
</div>
{% endblock %}
"""

ERROR_404_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}404 Not Found{% endblock %}
{% block content %}<div class='container text-center my-5'><h1><i class='fas fa-exclamation-triangle text-warning me-2'></i>404 - Page Not Found</h1><p>The page you are looking for does not exist.</p><a href='{{url_for("index")}}' class='btn btn-primary'>Go to Homepage</a></div>{% endblock %}
"""
ERROR_500_TEMPLATE = """
{% extends "BASE_HTML_TEMPLATE" %}
{% block title %}500 Server Error{% endblock %}
{% block content %}<div class='container text-center my-5'><h1><i class='fas fa-cogs text-danger me-2'></i>500 - Internal Server Error</h1><p>Something went wrong on our end. We're looking into it.</p><a href='{{url_for("index")}}' class='btn btn-primary'>Go to Homepage</a></div>{% endblock %}
"""

# ==============================================================================
# --- FIX: Add all templates to the template_storage dictionary ---
# ==============================================================================
template_storage['BASE_HTML_TEMPLATE'] = BASE_HTML_TEMPLATE
template_storage['INDEX_HTML_TEMPLATE'] = INDEX_HTML_TEMPLATE
template_storage['ARTICLE_HTML_TEMPLATE'] = ARTICLE_HTML_TEMPLATE
template_storage['POST_ARTICLE_TEMPLATE'] = POST_ARTICLE_TEMPLATE
template_storage['LOGIN_HTML_TEMPLATE'] = LOGIN_HTML_TEMPLATE
template_storage['REGISTER_HTML_TEMPLATE'] = REGISTER_HTML_TEMPLATE
template_storage['404_TEMPLATE'] = ERROR_404_TEMPLATE
template_storage['500_TEMPLATE'] = ERROR_500_TEMPLATE

# ==============================================================================
# --- 8. Main Execution Block ---
# ==============================================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), debug=True)
