# Rev14.py - FULLY REVISED CODE

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
from jinja2 import DictLoader, from_json
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
    # ## IMPROVEMENT ##: Added fallback to download if not found, useful for some environments.
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

# ## IMPROVEMENT ##: Enhanced News Quality Configuration
# Prioritize specific high-quality sources. Add a comma-separated list to your Render Env Vars.
# Example: 'techcrunch,the-verge,wired,ars-technica,reuters,bloomberg,business-insider'
app.config['NEWS_API_SOURCES'] = os.environ.get('NEWS_API_SOURCES', '')
app.config['NEWS_API_QUERY'] = (
    '("latest tech trends" OR "AI breakthroughs" OR "market analysis" OR "business innovation" OR "startup funding") NOT "press release"'
)
app.config['NEWS_API_DAYS_AGO'] = 7
app.config['NEWS_API_PAGE_SIZE'] = 100
app.config['NEWS_API_SORT_BY'] = 'popularity' # 'relevancy' or 'publishedAt' are also options
app.config['CACHE_EXPIRY_SECONDS'] = 3600  # 1 hour
app.config['READING_SPEED_WPM'] = 230
app.permanent_session_lifetime = timedelta(days=30)

logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app.logger.setLevel(logging.INFO)

# ## FIX 4 ##: Data Persistence on Render (Database Configuration)
# Use Render's PostgreSQL DATABASE_URL if available, otherwise fall back to a local SQLite file.
# This ensures data is not lost on deployment or restart.
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1) # SQLAlchemy standard
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
        # ## IMPROVEMENT ##: Using a slightly more advanced model for better instruction following.
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
    article_hash_id = db.Column(db.String(32), unique=True, nullable=False, index=True) # ## IMPROVEMENT ##: Indexed for faster lookups
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
    # ## FIX ##: Simplified comment linking. A comment belongs to a community article OR an API article.
    community_article_id = db.Column(db.Integer, db.ForeignKey('community_article.id'), nullable=True)
    api_article_hash_id = db.Column(db.String(32), nullable=True, index=True) # ## IMPROVEMENT ## Indexed for faster lookups

def init_db():
    with app.app_context():
        app.logger.info("Initializing database and creating tables if they don't exist...")
        db.create_all()
        app.logger.info("Database tables are ready.")

# Call it here to ensure it runs when the app module is loaded by Gunicorn/Flask
init_db()

# ==============================================================================
# --- 5. Helper Functions ---
# ==============================================================================
MASTER_ARTICLE_STORE, API_CACHE = {}, {}

def generate_article_id(url_or_title): return hashlib.md5(url_or_title.encode('utf-8')).hexdigest()

# ## FIX ##: Add Jinja filter 'fromjson' which was missing.
app.jinja_env.filters['fromjson'] = from_json

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

# ## FIX 1 & 2 ##: AI Summary/Takeaways Generation
@simple_cache(expiry_seconds_default=3600 * 24) # Cache AI analysis for 24 hours
def get_article_analysis_with_groq(article_text, article_title=""):
    """
    Analyzes article text using Groq to generate a summary and key takeaways.
    Returns a dictionary with 'groq_summary', 'groq_takeaways', or an 'error'.
    """
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
    # Truncate text to fit within the model's context window, leaving space for the prompt and response.
    human_prompt = f"Here is the article to analyze.\n\nTitle: {article_title}\n\nText:\n{article_text[:15000]}"

    try:
        # Use the dedicated JSON mode for more reliable output
        json_model = groq_client.bind(response_format={"type": "json_object"})
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
        ai_response = json_model.invoke(messages)
        
        # The response content should be a valid JSON string
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
        # This catches other potential errors like network issues or API-side problems
        app.logger.error(f"An unexpected error occurred during Groq analysis for '{article_title[:50]}': {e}", exc_info=True)
        return {"error": "An unexpected error occurred while communicating with the AI service."}


# ## FIX 5 ##: Improving News Quality
@simple_cache()
def fetch_news_from_api():
    if not newsapi:
        return []

    from_date = (datetime.utcnow() - timedelta(days=app.config['NEWS_API_DAYS_AGO'])).strftime('%Y-%m-%d')
    preferred_sources = app.config.get('NEWS_API_SOURCES', '').strip()
    
    try:
        # ## IMPROVEMENT ##: Prioritize fetching from a curated list of sources if provided
        if preferred_sources:
            app.logger.info(f"Fetching top headlines from preferred sources: {preferred_sources}")
            response = newsapi.get_top_headlines(
                sources=preferred_sources,
                language='en',
                page_size=app.config['NEWS_API_PAGE_SIZE']
            )
        else:
            app.logger.info(f"Fetching general news with query: {app.config['NEWS_API_QUERY']}")
            response = newsapi.get_everything(
                q=app.config['NEWS_API_QUERY'],
                from_param=from_date,
                language='en',
                sort_by=app.config['NEWS_API_SORT_BY'],
                page_size=app.config['NEWS_API_PAGE_SIZE']
            )

        processed_articles, unique_titles = [], set()
        for art_data in response.get('articles', []):
            title, url = art_data.get('title'), art_data.get('url')
            # Stricter filtering for higher quality
            if not all([url, title, art_data.get('source'), art_data.get('description')]) or title == '[Removed]' or title.lower() in unique_titles:
                continue
            
            unique_titles.add(title.lower())
            article_hash_id = generate_article_id(url)
            source_name = art_data['source'].get('name', 'Unknown Source')
            
            standardized_article = {
                'id': article_hash_id, # This is the hash_id for API articles
                'title': title,
                'description': art_data.get('description', ''),
                'url': url,
                'urlToImage': art_data.get('urlToImage'),
                'publishedAt': art_data.get('publishedAt', ''),
                'source': {'name': source_name},
                'is_community_article': False
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
        
        # Get analysis from Groq
        article_title = article_scraper.title or MASTER_ARTICLE_STORE.get(article_hash_id, {}).get('title', 'Unknown Title')
        groq_analysis = get_article_analysis_with_groq(article_scraper.text, article_title)
        
        return {
            "full_text": article_scraper.text,
            "read_time_minutes": calculate_read_time(article_scraper.text),
            "groq_analysis": groq_analysis, # This dictionary contains summary, takeaways, or an error
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
    # Use a dictionary to avoid modifying the session object directly in the context processor
    user_context = {}
    if 'user_id' in session:
        user_context['user_id'] = session['user_id']
        user_context['user_name'] = session.get('user_name', 'User')

    return {
        'categories': app.config['CATEGORIES'],
        'current_year': datetime.utcnow().year,
        'user': user_context, # Pass user info in a dedicated 'user' dictionary
        'request': request
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
    
    # Fetch all articles first to allow sorting if needed in the future.
    api_articles = fetch_news_from_api() if category_name == 'All Articles' else []
    db_articles = CommunityArticle.query.order_by(CommunityArticle.published_at.desc()).all() if category_name == 'Community Hub' else []

    # Prepare API articles for display
    for art_dict in api_articles:
        art_dict['is_community_article'] = False
        all_display_articles.append(art_dict)

    # Prepare Community articles for display
    for art in db_articles:
        art.is_community_article = True
        all_display_articles.append(art)
        
    # If 'All Articles', combine and sort
    if category_name == 'All Articles':
        all_display_articles.extend(db_articles)
        all_display_articles.sort(
            key=lambda x: getattr(x, 'published_at', datetime.fromisoformat(x.get('publishedAt', '1970-01-01T00:00:00Z').replace('Z', '+00:00'))) if isinstance(x, CommunityArticle) or x.get('publishedAt') else datetime.min,
            reverse=True
        )

    paginated_articles, total_pages = get_paginated_articles(all_display_articles, page, per_page)
    
    is_featured_page = (page == 1 and category_name == 'All Articles' and not request.args.get('query') and paginated_articles)

    return render_template("INDEX_HTML_TEMPLATE",
                           articles=paginated_articles,
                           selected_category=category_name,
                           current_page=page,
                           total_pages=total_pages,
                           is_featured_page=is_featured_page)


@app.route('/search')
@app.route('/search/page/<int:page>')
def search_results(page=1):
    query_str = request.args.get('query', '').strip()
    if not query_str:
        return redirect(url_for('index'))

    app.logger.info(f"Executing search for query: '{query_str}'")
    per_page = app.config['PER_PAGE']
    
    # Search API articles
    api_results = [art for art_id, art in MASTER_ARTICLE_STORE.items() if query_str.lower() in art.get('title', '').lower() or query_str.lower() in art.get('description', '').lower()]

    # Search community articles
    community_results = CommunityArticle.query.filter(
        db.or_(CommunityArticle.title.ilike(f'%{query_str}%'), CommunityArticle.description.ilike(f'%{query_str}%'))
    ).order_by(CommunityArticle.published_at.desc()).all()

    # Combine and sort results by date
    all_search_results = api_results + community_results
    all_search_results.sort(
        key=lambda x: getattr(x, 'published_at', datetime.fromisoformat(x.get('publishedAt', '1970-01-01T00:00:00Z').replace('Z', '+00:00'))) if isinstance(x, CommunityArticle) or x.get('publishedAt') else datetime.min,
        reverse=True
    )

    paginated_results, total_pages = get_paginated_articles(all_search_results, page, per_page)

    return render_template("INDEX_HTML_TEMPLATE",
                           articles=paginated_results,
                           selected_category=f"Search: {query_str}",
                           current_page=page,
                           total_pages=total_pages,
                           is_featured_page=False,
                           query=query_str)


# ## FIX 3 ##: Simplified and reliable routing for all articles.
@app.route('/article/<article_hash_id>')
def article_detail(article_hash_id):
    article = None
    is_community = False

    # Check if it's a community article by its unique hash ID
    article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
    if article:
        is_community = True
    else:
        # If not, check if it's an API article from the in-memory store
        article = MASTER_ARTICLE_STORE.get(article_hash_id)
        if not article:
            flash("The article you are looking for could not be found.", "danger")
            return redirect(url_for('index'))

    # Load comments for the article
    if is_community:
        comments = article.comments.order_by(Comment.timestamp.asc()).all()
    else:
        comments = Comment.query.filter_by(api_article_hash_id=article_hash_id).order_by(Comment.timestamp.asc()).all()

    return render_template("ARTICLE_HTML_TEMPLATE",
                           article=article,
                           is_community_article=is_community,
                           comments=comments)


@app.route('/get_article_content/<article_hash_id>')
def get_article_content_json(article_hash_id):
    article_data = MASTER_ARTICLE_STORE.get(article_hash_id)
    if not article_data or 'url' not in article_data:
        app.logger.warning(f"API Article or URL not found for hash_id: {article_hash_id}")
        return jsonify({"error": "Article data could not be located."}), 404
    
    # Check if content is already fetched and stored to avoid repeated processing
    if 'full_text' in article_data:
        app.logger.info(f"Serving cached content for article {article_hash_id}")
        return jsonify(article_data)

    processed_content = fetch_and_parse_article_content(article_hash_id, article_data['url'])
    
    if processed_content and not processed_content.get("error"):
        # Update the master store with the new content so it's cached for the session
        MASTER_ARTICLE_STORE[article_hash_id].update(processed_content)
    else:
        app.logger.error(f"Failed to process content for {article_hash_id}: {processed_content.get('error')}")

    return jsonify(processed_content)


@app.route('/add_comment/<article_hash_id>', methods=['POST'])
@login_required
def add_comment(article_hash_id):
    content = request.json.get('content', '').strip()
    if not content:
        return jsonify({"error": "Comment cannot be empty."}), 400

    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({"error": "User not found."}), 401

    # Check if the comment is for a community article
    community_article = CommunityArticle.query.filter_by(article_hash_id=article_hash_id).first()
    if community_article:
        new_comment = Comment(content=content, user_id=user.id, community_article_id=community_article.id)
    # Check if the comment is for an API article
    elif article_hash_id in MASTER_ARTICLE_STORE:
        new_comment = Comment(content=content, user_id=user.id, api_article_hash_id=article_hash_id)
    else:
        return jsonify({"error": "Cannot comment on an unknown article."}), 404

    db.session.add(new_comment)
    db.session.commit()
    app.logger.info(f"User '{user.username}' added comment to article '{article_hash_id}'")
    
    return jsonify({
        "success": True,
        "comment": {
            "content": new_comment.content,
            "timestamp": new_comment.timestamp.isoformat(),
            "author": {"name": user.name}
        }
    }), 201


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

    # Generate a unique and non-sequential ID for the URL
    article_hash_id = generate_article_id(title + str(session['user_id']) + str(time.time()))
    
    # Asynchronously generate AI summary and takeaways
    groq_analysis = get_article_analysis_with_groq(content, title)
    groq_summary = None
    groq_takeaways_json = None
    if groq_analysis and not groq_analysis.get("error"):
        groq_summary = groq_analysis.get('groq_summary')
        takeaways = groq_analysis.get('groq_takeaways')
        if takeaways:
            groq_takeaways_json = json.dumps(takeaways)
    
    new_article = CommunityArticle(
        article_hash_id=article_hash_id,
        title=title,
        description=description,
        full_text=content,
        source_name=source_name,
        image_url=image_url or f'https://via.placeholder.com/400x220/1E3A5E/FFFFFF?text={urllib.parse.quote_plus(title[:20])}',
        user_id=session['user_id'],
        groq_summary=groq_summary,
        groq_takeaways=groq_takeaways_json
    )
    db.session.add(new_article)
    db.session.commit()
    
    flash("Your article has been successfully posted to the Community Hub!", "success")
    # Redirect to the new article's page
    return redirect(url_for('article_detail', article_hash_id=new_article.article_hash_id))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('index'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        # Validation checks
        if not all([name, username, password]):
            flash('All fields are required.', 'danger')
        elif len(username) < 3:
            flash('Username must be at least 3 characters long.', 'warning')
        elif len(password) < 6:
            flash('Password must be at least 6 characters long.', 'warning')
        elif User.query.filter(db.func.lower(User.username) == username.lower()).first():
            flash('That username is already taken. Please choose another.', 'warning')
        else:
            new_user = User(name=name, username=username, password_hash=generate_password_hash(password))
            db.session.add(new_user)
            db.session.commit()
            app.logger.info(f"New user registered: {username}")
            flash(f'Welcome, {name}! Your account has been created. Please log in.', 'success')
            return redirect(url_for('login'))
        return redirect(url_for('register')) # Redirect back to show flashed messages

    return render_template("REGISTER_HTML_TEMPLATE")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
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
# --- 7. HTML Templates (Inline for simplicity, as per original structure) ---
# ==============================================================================
# NOTE: The python code above contains all necessary fixes. The HTML templates below
# are provided for completeness and include fixes for routing links and displaying AI content.
# You should ensure your templates match these, especially the `url_for` calls and JS logic.

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
        .navbar-main { background: var(--primary-gradient); padding: 0.8rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-bottom: 2px solid rgba(255,255,255,0.15); transition: background 0.3s ease, border-bottom 0.3s ease; height: 95px; display: flex; align-items: center; }
        .navbar-brand-custom { color: white !important; font-weight: 800; font-size: 2.2rem; letter-spacing: 0.5px; font-family: 'Poppins', sans-serif; margin-bottom: 0; display: flex; align-items: center; gap: 12px; }
        .navbar-brand-custom .brand-icon { color: var(--secondary-light); font-size: 2.5rem; }
        .search-form-container { flex-grow: 1; display: flex; justify-content: center; padding: 0 1rem; }
        .search-container { position: relative; width: 100%; max-width: 550px; }
        .navbar-search { border-radius: 25px; padding: 0.7rem 1.25rem 0.7rem 2.8rem; border: 1px solid rgba(255,255,255,0.2); font-size: 0.95rem; transition: all 0.3s ease; background: rgba(255,255,255,0.1); color: white; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
        .navbar-search::placeholder { color: rgba(255,255,255,0.6); }
        .navbar-search:focus { background: rgba(255,255,255,0.2); box-shadow: 0 0 0 3px rgba(var(--secondary-color-rgb),0.3); border-color: var(--secondary-color); outline: none; color:white; }
        .search-icon { color: rgba(255,255,255,0.7); transition: all 0.3s ease; left: 1rem; position: absolute; top: 50%; transform: translateY(-50%); }
        .header-controls { display: flex; gap: 0.8rem; align-items: center; }
        .header-btn { background: transparent; border: 1px solid rgba(255,255,255,0.3); padding: 0.5rem 1rem; border-radius: 20px; color: white; font-weight: 500; transition: all 0.3s ease; display: flex; align-items: center; gap: 0.5rem; cursor: pointer; text-decoration:none; font-size: 0.9rem; }
        .header-btn:hover { background: var(--secondary-color); border-color: var(--secondary-color); color: var(--primary-color); transform: translateY(-1px); }
        .category-nav { background: var(--white-bg); box-shadow: 0 3px 10px rgba(0,0,0,0.03); position: fixed; top: 95px; width: 100%; z-index: 1020; border-bottom: 1px solid var(--card-border-color); }
        .category-link { color: var(--primary-color) !important; font-weight: 600; padding: 0.6rem 1.3rem !important; border-radius: 20px; transition: all 0.25s ease; white-space: nowrap; text-decoration: none; margin: 0 0.3rem; font-size: 0.9rem; border: 1px solid transparent; }
        .category-link.active { background: var(--primary-color) !important; color: white !important; box-shadow: 0 3px 10px rgba(var(--primary-color-rgb), 0.2); }
        .article-card { background: var(--white-bg); border-radius: 10px; transition: all 0.3s ease; border: 1px solid var(--card-border-color); box-shadow: 0 5px 15px rgba(0,0,0,0.05); }
        .article-card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.08); }
        footer { background: var(--footer-bg); color: var(--footer-text); margin-top: auto; padding: 3rem 0 1.5rem; font-size:0.9rem; }
        .footer-section h5 { color: var(--secondary-color); }
        .copyright { text-align: center; padding-top: 1.5rem; margin-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.1); font-size: 0.85rem; }
        .alert-top { position: fixed; top: 105px; left: 50%; transform: translateX(-50%); z-index: 2050; min-width:320px; text-align:center; }
        .add-article-btn { position: fixed; bottom: 25px; right: 25px; z-index: 1030; width: 55px; height: 55px; border-radius: 50%; background: var(--secondary-color); color: var(--primary-color); border: none; box-shadow: 0 4px 15px rgba(var(--secondary-color-rgb),0.3); display: flex; align-items: center; justify-content: center; font-size: 22px; cursor: pointer; transition: all 0.3s ease; }
        .add-article-btn:hover { transform: translateY(-4px) scale(1.05); }
        .add-article-modal { display: none; /* Controlled by JS */ position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: 2000; background-color: rgba(0, 0, 0, 0.6); backdrop-filter: blur(5px); align-items: center; justify-content: center; }
        .modal-content { width: 90%; max-width: 700px; background: var(--white-bg); border-radius: 10px; padding: 2rem; position: relative; max-height: 90vh; overflow-y: auto;}
        .close-modal { position: absolute; top: 12px; right: 12px; font-size: 20px; color: var(--text-muted-color); background: none; border: none; cursor: pointer; }
        .btn-primary-modal { background-color: var(--primary-color); border-color: var(--primary-color); color:white; }
        @media (max-width: 991.98px) {
            body { padding-top: 180px; }
            .navbar-main { height: auto;}
            .navbar-content-wrapper { flex-direction: column; align-items: flex-start; gap: 0.5rem; }
            .search-form-container { width: 100%; order: 3; margin-top:0.5rem; }
            .header-controls { position: absolute; top: 0.9rem; right: 1rem; }
            .category-nav { top: 130px; }
        }
    </style>
    {% block head_extra %}{% endblock %}
</head>
<body>
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
            <div class="navbar-content-wrapper w-100">
                <a class="navbar-brand-custom" href="{{ url_for('index') }}"><i class="fas fa-bolt-lightning brand-icon"></i><span>Briefly</span></a>
                <div class="search-form-container">
                    <form action="{{ url_for('search_results') }}" method="GET" class="search-container">
                        <input type="search" name="query" class="form-control navbar-search" placeholder="Search articles..." value="{{ request.args.get('query', '') }}">
                        <i class="fas fa-search search-icon"></i>
                    </form>
                </div>
                <div class="header-controls">
                    {% if user.user_id %}
                        <span class="text-white me-2 d-none d-md-inline">Hi, {{ user.user_name.split(' ')[0] }}!</span>
                        <a href="{{ url_for('logout') }}" class="header-btn" title="Logout"><i class="fas fa-sign-out-alt"></i> <span class="d-none d-sm-inline">Logout</span></a>
                    {% else %}
                        <a href="{{ url_for('login') }}" class="header-btn" title="Login/Register"><i class="fas fa-user"></i> <span class="d-none d-sm-inline">Login</span></a>
                    {% endif %}
                </div>
            </div>
        </div>
    </nav>

    <nav class="navbar navbar-expand-lg category-nav">
        <div class="container justify-content-center">
            {% for cat_item in categories %}
                <a href="{{ url_for('index', category_name=cat_item) }}" class="category-link {% if selected_category == cat_item %}active{% endif %}">{{ cat_item }}</a>
            {% endfor %}
        </div>
    </nav>

    <main class="container main-content my-4">
        {% block content %}{% endblock %}
    </main>

    {% if user.user_id %}
    <button class="add-article-btn" id="addArticleBtn" title="Post a New Article"><i class="fas fa-plus"></i></button>
    <div class="add-article-modal" id="addArticleModal">
        <div class="modal-content">
            <button class="close-modal" id="closeModalBtn" title="Close"><i class="fas fa-times"></i></button>
            <h3 class="modal-title">Post Article to Community Hub</h3>
            <form id="addArticleForm" action="{{ url_for('post_article') }}" method="POST">
                <div class="mb-3"><label for="articleTitle">Title</label><input type="text" id="articleTitle" name="title" class="form-control" required></div>
                <div class="mb-3"><label for="articleDescription">Short Description</label><textarea id="articleDescription" name="description" class="form-control" rows="3" required></textarea></div>
                <div class="mb-3"><label for="articleSource">Source Name</label><input type="text" id="articleSource" name="sourceName" class="form-control" value="Community Post" required></div>
                <div class="mb-3"><label for="articleImage">Image URL (Optional)</label><input type="url" id="articleImage" name="imageUrl" class="form-control"></div>
                <div class="mb-3"><label for="articleContent">Full Content</label><textarea id="articleContent" name="content" class="form-control" rows="7" required></textarea></div>
                <div class="d-flex justify-content-end gap-2"><button type="button" class="btn btn-secondary" id="cancelArticleBtn">Cancel</button><button type="submit" class="btn btn-primary-modal">Post Article</button></div>
            </form>
        </div>
    </div>
    {% endif %}

    <footer class="mt-auto">
        <div class="container">
            <div class="copyright">&copy; {{ current_year }} Briefly. All rights reserved.</div>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
    document.addEventListener('DOMContentLoaded', function () {
        const addArticleBtn = document.getElementById('addArticleBtn');
        const addArticleModal = document.getElementById('addArticleModal');
        if (addArticleBtn && addArticleModal) {
            const closeModal = () => { addArticleModal.style.display = 'none'; };
            addArticleBtn.addEventListener('click', () => { addArticleModal.style.display = 'flex'; });
            document.getElementById('closeModalBtn').addEventListener('click', closeModal);
            document.getElementById('cancelArticleBtn').addEventListener('click', closeModal);
            addArticleModal.addEventListener('click', (e) => { if (e.target === addArticleModal) closeModal(); });
        }
        document.querySelectorAll('#alert-placeholder .alert').forEach(a => setTimeout(() => bootstrap.Alert.getOrCreateInstance(a).close(), 5000));
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
    {% if articles and is_featured_page %}
    <article class="featured-article p-md-4 p-3 mb-4">
        {% set art = articles[0] %}
        {# ## FIX 3 ##: Use article_hash_id for community articles and id (which is a hash) for API articles. #}
        {% set article_url = url_for('article_detail', article_hash_id=(art.article_hash_id if art.is_community_article else art.id)) %}
        <div class="row g-0 g-md-4">
            <div class="col-lg-6"><a href="{{ article_url }}"><img src="{{ art.image_url if art.is_community_article else art.urlToImage }}" class="img-fluid rounded" style="height:320px; width:100%; object-fit:cover;" alt="{{ art.title }}"></a></div>
            <div class="col-lg-6 ps-lg-3"><h2 class="h4"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title }}</a></h2><p>{{ art.description|truncate(220) }}</p></div>
        </div>
    </article>
    {% elif not articles %}
        <div class="alert alert-info text-center">No articles found.</div>
    {% endif %}

    <div class="row g-4">
        {% for art in (articles[1:] if is_featured_page else articles) %}
        <div class="col-md-6 col-lg-4 d-flex">
            <div class="article-card d-flex flex-column w-100">
                {# ## FIX 3 ##: Consistent URL generation for all article types. #}
                {% set article_url = url_for('article_detail', article_hash_id=(art.article_hash_id if art.is_community_article else art.id)) %}
                <a href="{{ article_url }}"><img src="{{ art.image_url if art.is_community_article else art.urlToImage }}" class="card-img-top" style="height: 200px; object-fit: cover;" alt="{{ art.title|truncate(50) }}"></a>
                <div class="card-body d-flex flex-column">
                    <h5 class="card-title h6"><a href="{{ article_url }}" class="text-decoration-none">{{ art.title|truncate(70) }}</a></h5>
                    <p class="card-text small text-muted flex-grow-1">{{ art.description|truncate(100) }}</p>
                    <a href="{{ article_url }}" class="btn btn-sm btn-primary mt-auto">Read More</a>
                </div>
            </div>
        </div>
        {% endfor %}
    </div>

    {% if total_pages and total_pages > 1 %}
    <nav class="mt-5"><ul class="pagination justify-content-center">
        <li class="page-item {% if current_page == 1 %}disabled{% endif %}"><a class="page-link" href="{{ url_for(request.endpoint, page=current_page-1, category_name=selected_category, query=query) }}">&laquo;</a></li>
        {% for p in range(1, total_pages + 1) %}
            {% if p == current_page %}
            <li class="page-item active"><span class="page-link">{{ p }}</span></li>
            {% else %}
            <li class="page-item"><a class="page-link" href="{{ url_for(request.endpoint, page=p, category_name=selected_category, query=query) }}">{{ p }}</a></li>
            {% endif %}
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
    .article-full-content-wrapper { background-color: var(--white-bg); padding: 2rem; border-radius: 10px; }
    .loader-container { display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 200px; }
    .loader { border: 5px solid #f3f3f3; border-top: 5px solid var(--primary-color); border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .summary-box, .takeaways-box { background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 1.5rem 0; border: 1px solid #e9ecef; }
    .takeaways-box ul { padding-left: 1.2rem; }
</style>
{% endblock %}
{% block content %}
{% if not article %}
    <div class="alert alert-danger text-center">Article not found.</div>
{% else %}
<article class="article-full-content-wrapper">
    <h1 class="mb-3 display-6">{{ article.title }}</h1>
    <div class="text-muted mb-3 small">
        <span>By: {{ article.author.name if is_community_article else article.source.name }}</span> |
        <span>Published: {{ (article.published_at.strftime('%b %d, %Y') if is_community_article else (article.publishedAt.split('T')[0] if article.publishedAt else 'N/A')) }}</span> |
        <span id="articleReadTimeMeta">Est. Read: <span id="articleReadTimeText">--</span> min</span>
    </div>
    <img src="{{ article.image_url if is_community_article else article.urlToImage }}" alt="{{ article.title }}" class="img-fluid rounded mb-4">

    {# AI Generated Content Section #}
    <div id="aiContentContainer">
        {% if is_community_article %}
            {% if article.groq_summary %}
                <div class="summary-box"><h5>AI Summary</h5><p>{{ article.groq_summary|replace('\\n', '<br>')|safe }}</p></div>
            {% endif %}
            {% if article.groq_takeaways and article.groq_takeaways|fromjson|length > 0 %}
                <div class="takeaways-box"><h5>AI Key Takeaways</h5><ul>{% for takeaway in article.groq_takeaways|fromjson %}<li>{{ takeaway }}</li>{% endfor %}</ul></div>
            {% endif %}
        {% else %}
            {# Loader for API articles, content will be injected by JS #}
            <div id="contentLoader" class="loader-container"><div class="loader"></div><p class="mt-2">Generating AI summary...</p></div>
        {% endif %}
    </div>
    
    <hr>
    {# Full Content Section #}
    <div class="mt-4">
        <h5>{% if is_community_article %}Full Article{% else %}Original Article Link{% endif %}</h5>
        {% if is_community_article %}
            <p style="white-space: pre-wrap;">{{ article.full_text }}</p>
        {% else %}
            <p>Read the full original article at the source.</p>
            <a href="{{ article.url }}" class="btn btn-outline-primary" target="_blank" rel="noopener noreferrer">Read at {{ article.source.name }} <i class="fas fa-external-link-alt ms-1"></i></a>
        {% endif %}
    </div>

    {# Comments Section #}
    <section class="comment-section mt-5" id="comment-section">
        <h3 class="mb-4">Discussion ({{ comments|length }})</h3>
        <div id="comments-list">
            {% for comment in comments %}
            <div class="card mb-3"><div class="card-body">
                <p class="card-text">{{ comment.content }}</p>
                <footer class="blockquote-footer">{{ comment.author.name }} on <cite>{{ comment.timestamp.strftime('%b %d, %Y') }}</cite></footer>
            </div></div>
            {% else %}
            <p id="no-comments-msg">No comments yet. Be the first to share your thoughts!</p>
            {% endfor %}
        </div>
        {% if user.user_id %}
            <form id="comment-form" class="mt-4"><div class="mb-3"><textarea class="form-control" id="comment-content" rows="3" placeholder="Join the discussion..." required></textarea></div><button type="submit" class="btn btn-primary">Post Comment</button></form>
        {% else %}
            <div class="alert alert-light mt-4">Please <a href="{{ url_for('login', next=request.url) }}">log in</a> to comment.</div>
        {% endif %}
    </section>
</article>
{% endif %}
{% endblock %}
{% block scripts_extra %}
<script>
document.addEventListener('DOMContentLoaded', function () {
    const isCommunity = {{ is_community_article | tojson }};
    const articleHashId = {{ (article.article_hash_id if is_community_article else article.id) | tojson }};

    function calculateAndSetReadTime(text) {
        if (!text) return;
        const wpm = 230;
        const wordCount = text.split(/\s+/).length;
        const readTime = Math.max(1, Math.round(wordCount / wpm));
        document.getElementById('articleReadTimeText').textContent = readTime;
    }

    if (isCommunity) {
        const fullText = {{ article.full_text|tojson if is_community_article else 'null' }};
        calculateAndSetReadTime(fullText);
    } else {
        const contentLoader = document.getElementById('contentLoader');
        const aiContainer = document.getElementById('aiContentContainer');
        
        fetch(`{{ url_for('get_article_content_json', article_hash_id='_') }}`.replace('_', articleHashId))
            .then(res => {
                if (!res.ok) throw new Error(`Server error: ${res.status}`);
                return res.json();
            })
            .then(data => {
                contentLoader.style.display = 'none';
                if (data.error) {
                    aiContainer.innerHTML = `<div class="alert alert-warning">${data.error}</div>`;
                    return;
                }
                
                let html = '';
                const analysis = data.groq_analysis;
                if (analysis && analysis.groq_summary) {
                    html += `<div class="summary-box"><h5>AI Summary</h5><p>${analysis.groq_summary.replace(/\\n/g, '<br>')}</p></div>`;
                }
                if (analysis && analysis.groq_takeaways && analysis.groq_takeaways.length > 0) {
                    html += `<div class="takeaways-box"><h5>AI Key Takeaways</h5><ul>${analysis.groq_takeaways.map(t => `<li>${t}</li>`).join('')}</ul></div>`;
                }
                aiContainer.innerHTML = html;
                calculateAndSetReadTime(data.full_text);
            })
            .catch(err => {
                console.error("Error fetching article content:", err);
                if(contentLoader) contentLoader.innerHTML = '<div class="alert alert-danger">Failed to load article analysis.</div>';
            });
    }

    const commentForm = document.getElementById('comment-form');
    if (commentForm) {
        commentForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const content = document.getElementById('comment-content').value.trim();
            if (!content) return;

            fetch(`{{ url_for('add_comment', article_hash_id='_') }}`.replace('_', articleHashId), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ content: content })
            })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    window.location.reload(); // Simple reload to show the new comment
                } else {
                    alert('Error: ' + data.error);
                }
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
<div class="auth-container" style="max-width: 450px; margin: 3rem auto; padding: 2rem; background: var(--white-bg); border-radius: 10px;">
    <h2 class="text-center mb-4">Login</h2>
    <form method="POST" action="{{ url_for('login', next=request.args.get('next')) }}">
        <div class="mb-3"><label for="username">Username</label><input type="text" class="form-control" id="username" name="username" required></div>
        <div class="mb-3"><label for="password">Password</label><input type="password" class="form-control" id="password" name="password" required></div>
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
<div class="auth-container" style="max-width: 450px; margin: 3rem auto; padding: 2rem; background: var(--white-bg); border-radius: 10px;">
    <h2 class="text-center mb-4">Create Account</h2>
    <form method="POST" action="{{ url_for('register') }}">
        <div class="mb-3"><label for="name">Full Name</label><input type="text" class="form-control" id="name" name="name" required></div>
        <div class="mb-3"><label for="username">Username</label><input type="text" class="form-control" id="username" name="username" required></div>
        <div class="mb-3"><label for="password">Password</label><input type="password" class="form-control" id="password" name="password" required></div>
        <button type="submit" class="btn btn-primary w-100 mt-3">Register</button>
    </form>
    <p class="mt-3 text-center small">Already have an account? <a href="{{ url_for('login') }}">Login here</a></p>
</div>
{% endblock %}
"""

ERROR_404_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}404 Not Found{% endblock %}{% block content %}<div class='container text-center my-5'><h1>404 - Not Found</h1><p>The page you are looking for does not exist.</p><a href='{{url_for("index")}}' class='btn btn-primary'>Go Home</a></div>{% endblock %}"""
ERROR_500_TEMPLATE = """{% extends "BASE_HTML_TEMPLATE" %}{% block title %}500 Error{% endblock %}{% block content %}<div class='container text-center my-5'><h1>500 - Server Error</h1><p>Something went wrong. We are looking into it.</p><a href='{{url_for("index")}}' class='btn btn-primary'>Go Home</a></div>{% endblock %}"""


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
    # The port is set by Gunicorn on Render, but this is for local testing.
    port = int(os.environ.get("PORT", 8080))
    # debug=False is safer for production, but True is fine for Render's dev environment
    app.run(host='0.0.0.0', port=port, debug=True)
