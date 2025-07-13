import os
import feedparser
import logging
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv
import re


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraper.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('rss_scraper')

# Load environment variables
load_dotenv()

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not all([SUPABASE_URL, SUPABASE_KEY]):
    logger.error("Supabase credentials not found in environment variables")
    raise ValueError("Missing Supabase credentials")

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Successfully connected to Supabase")
except Exception as e:
    logger.error(f"Failed to connect to Supabase: {str(e)}")
    raise

# 키워드 목록
KEYWORDS = ["GPT", "AI", "LLM", "ChatGPT", "Claude", "perplexity"]


def generate_rss_url(keyword):
    """Generate Google News RSS URL for a given keyword"""
    base = "https://news.google.com/rss/search?q="
    query = keyword.replace(" ", "+")
    return f"{base}{query}&hl=ko&gl=KR&ceid=KR:ko"


def fetch_and_store_news():
    """Fetch news for each keyword and store them in Supabase.
    Logs the process and any errors encountered.
    """
    logger.info("Starting news fetch process...")
    total_processed = 0

    for keyword in KEYWORDS:
        logger.info(f"Processing keyword: {keyword}")
        url = generate_rss_url(keyword)

        try:
            feed = feedparser.parse(url)
            if hasattr(feed, 'bozo_exception'):
                logger.warning(f"Feed parsing warning for {url}: {feed.bozo_exception}")
                continue

            logger.info(f"Found {len(feed.entries)} entries for keyword: {keyword}")

            keyword_processed = 0
            for entry in feed.entries:
                try:
                    # Extract title and clean it
                    title = entry.get('title', '').strip()
                    if not title:
                        logger.debug("Skipping entry with empty title")
                        continue

                    # Extract and clean description
                    description = entry.get('description', '')
                    description = re.sub(r'<[^>]+>', '', description)  # Remove HTML tags
                    description = re.sub(r'\s+', ' ', description).strip()

                    # Extract source from the entry
                    source = 'Unknown'
                    if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
                        source = entry.source.title
                    elif hasattr(entry, 'author'):
                        source = entry.author

                    # Parse publication date
                    pub_date = datetime.now()  # Default to current time
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])

                    # Prepare data for Supabase
                    data = {
                        'title': title[:500],  # Limit title length
                        'description': description[:2000],  # Limit description length
                        'source': source[:200],  # Limit source length
                        'url': entry.link,
                        'keyword': keyword,
                        'pub_date': pub_date.isoformat()
                    }

                    # Insert or update in Supabase
                    try:
                        result = supabase.table('ai_news').upsert(data, on_conflict='url').execute()
                        if hasattr(result, 'data') and result.data:
                            keyword_processed += 1
                            logger.debug(f"Processed: {title[:50]}...")
                        else:
                            logger.warning(f"No data returned for: {title[:50]}...")
                    except Exception as e:
                        logger.error(f"Database error for '{title[:50]}...': {str(e)}")

                except Exception as e:
                    logger.error(f"Error processing entry: {str(e)}", exc_info=True)
                    continue

            logger.info(f"Processed {keyword_processed} entries for keyword: {keyword}")
            total_processed += keyword_processed

        except Exception as e:
            logger.error(f"Error processing keyword {keyword}: {str(e)}", exc_info=True)
            continue

    logger.info(f"News fetch process completed. Total entries processed: {total_processed}")
    return total_processed


if __name__ == "__main__":
    fetch_and_store_news()
