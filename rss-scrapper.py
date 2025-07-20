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
CATEGORIES = {
    "LLM": ["GPT", "ChatGPT", "Claude", "LLaMA", "Mistral", "Mixtral", "Yi", "Command R"],
    "검색엔진AI": ["Perplexity", "You.com", "Poe", "Phind"],
    "오픈소스AI": ["Hugging Face", "Ollama", "GPT4All", "OpenRouter"],
    "AI플랫폼": ["OpenAI", "Anthropic", "Google Gemini", "Meta AI", "Cohere"],
    "생성형AI": ["Stable Diffusion", "MidJourney", "DALL·E", "Sora", "Pika", "Luma AI"],
    "AI에이전트": ["AutoGPT", "AgentGPT", "BabyAGI", "AI Agents"],
    "개발도구": ["LangChain", "LangGraph", "LlamaIndex", "vLLM", "TGI", "ComfyUI", "MLC LLM", "AI Copilot", "GitHub Copilot", "Cursor", "Windsurf"],
    "기업용AI": ["Enterprise AI", "Edge AI", "TinyML", "On-Device AI"],
}


def generate_rss_url(keyword):
    """Generate Google News RSS URL for a given keyword"""
    base = "https://news.google.com/rss/search?q="
    query = keyword.replace(" ", "+")
    return f"{base}{query}&hl=ko&gl=KR&ceid=KR:ko"


def fetch_and_store_news():
    """Fetch news for each keyword and store them in Supabase in batches.
    Logs the process and any errors encountered.
    """
    logger.info("Starting news fetch process...")
    total_processed = 0
    batch_size = 50  # 배치 크기 설정
    batch = []  # 배치 데이터를 저장할 리스트

    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            logger.info(f"Processing keyword: {keyword}")
            
            # 한국어 뉴스 검색
            kr_url = f"https://news.google.com/rss/search?q={keyword.replace(' ', '+')}&hl=ko&gl=KR&ceid=KR:ko"
            # 영어 뉴스 검색
            us_url = f"https://news.google.com/rss/search?q={keyword.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            
            all_entries = []
            
            # 한국어 뉴스 가져오기
            try:
                kr_feed = feedparser.parse(kr_url)
                if hasattr(kr_feed, 'entries'):
                    all_entries.extend(kr_feed.entries)
                    logger.info(f"Found {len(kr_feed.entries)} Korean entries for keyword: {keyword}")
            except Exception as e:
                logger.warning(f"Error fetching Korean news for {keyword}: {str(e)}")
            
            # 영어 뉴스 가져오기
            try:
                us_feed = feedparser.parse(us_url)
                if hasattr(us_feed, 'entries'):
                    all_entries.extend(us_feed.entries)
                    logger.info(f"Found {len(us_feed.entries)} English entries for keyword: {keyword}")
            except Exception as e:
                logger.warning(f"Error fetching US news for {keyword}: {str(e)}")
            
            if not all_entries:
                logger.info(f"No entries found for keyword: {keyword}")
                continue
                
            logger.info(f"Total {len(all_entries)} entries found for keyword: {keyword}")
            keyword_processed = 0
            
            for entry in all_entries:
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
                        'pub_date': pub_date.isoformat(),
                        'category': category,
                        'subcategory': keyword
                    }

                    # 배치에 데이터 추가
                    batch.append(data)
                    keyword_processed += 1

                    # 배치 크기에 도달하면 일괄 삽입
                    if len(batch) >= batch_size:
                        try:
                            result = supabase.table('ai_news').upsert(batch, on_conflict='url').execute()
                            if hasattr(result, 'data') and result.data:
                                logger.info(f"Inserted/Updated {len(batch)} records in batch")
                            else:
                                logger.warning("No data returned for batch insert")
                            batch = []  # 배치 초기화
                        except Exception as e:
                            logger.error(f"Batch insert error: {str(e)}")
                            batch = []  # 에러 발생 시 배치 초기화

                except Exception as e:
                    logger.error(f"Error processing entry: {str(e)}", exc_info=True)
                    continue

            # 남은 배치 데이터 처리
            if batch:
                try:
                    result = supabase.table('ai_news').upsert(batch, on_conflict='url').execute()
                    if hasattr(result, 'data') and result.data:
                        logger.info(f"Inserted/Updated final batch of {len(batch)} records")
                    else:
                        logger.warning("No data returned for final batch insert")
                except Exception as e:
                    logger.error(f"Final batch insert error: {str(e)}")
                batch = []  # 마지막 배치 초기화

            total_processed += keyword_processed
            logger.info(f"Processed {keyword_processed} entries for keyword: {keyword}")

    # 마지막으로 남은 배치가 있는지 확인
    if batch:
        try:
            result = supabase.table('ai_news').upsert(batch, on_conflict='url').execute()
            if hasattr(result, 'data') and result.data:
                logger.info(f"Inserted/Updated final batch of {len(batch)} records")
            else:
                logger.warning("No data returned for final batch insert")
        except Exception as e:
            logger.error(f"Final batch insert error: {str(e)}")

    logger.info(f"Total {total_processed} entries processed.")
    logger.info(f"News fetch process completed. Total entries processed: {total_processed}")
    return total_processed


if __name__ == "__main__":
    fetch_and_store_news()
