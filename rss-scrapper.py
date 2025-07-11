import feedparser
from datetime import datetime
from supabase import create_client
import os


# ✅ Supabase 설정
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ✅ 키워드 목록
KEYWORDS = ["GPT", "AI", "LLM", "ChatGPT", "Claude"]

# ✅ 중복 URL 체크를 위한 해시 세트
existing_urls = set()

# ✅ 현재 DB에 있는 URL 목록 불러오기
def load_existing_urls():
    res = supabase.table("ai_news").select("url").execute()
    if res.data:
        for item in res.data:
            existing_urls.add(item["url"])

# ✅ Google News RSS URL 생성
def generate_rss_url(keyword):
    base = "https://news.google.com/rss/search?q="
    query = keyword.replace(" ", "+")
    return f"{base}{query}&hl=ko&gl=KR&ceid=KR:ko"

# ✅ 뉴스 수집 및 저장
def fetch_and_store_news():
    load_existing_urls()

    for keyword in KEYWORDS:
        url = generate_rss_url(keyword)
        feed = feedparser.parse(url)

        for entry in feed.entries:
            news_url = entry.link
            if news_url in existing_urls:
                continue

            title = entry.title
            summary = entry.summary if hasattr(entry, "summary") else ""
            pub_date = datetime(*entry.published_parsed[:6])

            # Supabase에 저장
            supabase.table("ai_news").insert({
                "title": title,
                "summary": summary,
                "url": news_url,
                "keyword": keyword,
                "pub_date": pub_date.isoformat()
            }).execute()

            print(f"[+] 저장됨: {title}")
            existing_urls.add(news_url)

if __name__ == "__main__":
    fetch_and_store_news()
