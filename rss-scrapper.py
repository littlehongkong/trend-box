import feedparser
from datetime import datetime
from supabase import create_client
import os
import re

# ✅ Supabase 설정
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ 키워드 목록
KEYWORDS = ["GPT", "AI", "LLM", "ChatGPT", "Claude"]

# ✅ Google News RSS URL 생성
def generate_rss_url(keyword):
    base = "https://news.google.com/rss/search?q="
    query = keyword.replace(" ", "+")
    return f"{base}{query}&hl=ko&gl=KR&ceid=KR:ko"

# ✅ 뉴스 수집 및 저장
def fetch_and_store_news():

    # 모든 키워드에 대한 기사 수집 및 저장
    for keyword in KEYWORDS:
        url = generate_rss_url(keyword)
        feed = feedparser.parse(url)
        print(f"Processing news for keyword: {keyword}")
        title: str = None

        for entry in feed.entries:
            try:
                title = entry.title
                pub_date = datetime(*entry.published_parsed[:6])
                
                # Extract and clean description
                description = ""
                if hasattr(entry, "description"):
                    description = re.sub(r'<[^>]+>', '', entry.description)
                    description = re.sub(r'https?:\/\/\S+', '', description)
                    description = description.strip()
                
                # Extract source
                source = ""
                if hasattr(entry, "source") and hasattr(entry.source, "title"):
                    source = entry.source.title
                
                # Supabase에 UPSERT로 저장 (URL이 중복되면 업데이트 안 함)
                supabase.table('ai_news').upsert({
                    'title': title,
                    'description': description,
                    'source': source,
                    'url': entry.link,  # UNIQUE 제약조건이 있는 컬럼
                    'keyword': keyword,
                    'pub_date': pub_date.isoformat()
                }, on_conflict='url').execute()
                
                print(f"[+] 처리됨: {title}")
                
            except Exception as e:
                print(f"[!] 오류 - {title}: {str(e)}")


if __name__ == "__main__":
    fetch_and_store_news()
