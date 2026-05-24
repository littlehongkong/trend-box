# site_scraper.py
# 신뢰도 높은 사이트 RSS 직접 수집 → site_news 테이블 저장

import os
import re
import time
import logging
import feedparser
from datetime import datetime, timezone, timedelta
from supabase import create_client
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('site_scraper.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('site_scraper')

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
KST      = timezone(timedelta(hours=9))

# ── 사이트 설정 ───────────────────────────────────────────────
SITES = [
    # ── 국내 (domestic) ──────────────────────────────────────
    {
        "name":         "AI타임스",
        "rss":          "https://www.aitimes.com/rss/allArticle.xml",
        "source_type":  "domestic",
        "source_country": "kor",
        "priority":     10,
        "categories":   ["AI"],
        "keywords":     [],   # 전체 수집 (AI 전문 매체)
    },
    {
        "name":         "ZDNet Korea",
        "rss":          "https://feeds.feedburner.com/zdkorea",
        "source_type":  "domestic",
        "source_country": "kor",
        "priority":     8,
        "categories":   ["AI", "데이터엔지니어링", "RPA"],
        "keywords": [
            "AI", "인공지능", "LLM", "에이전트",
            "데이터", "파이프라인", "자동화", "RPA",
            "클라우드", "머신러닝", "생성형",
            "에이전틱", "agentic", "오케스트레이션",
            "MCP", "RAG", "벡터DB",
            "데이터거버넌스", "데이터품질",
            "Snowflake", "Databricks", "Airflow",
            "망분리", "보안", "사이버",
            "프로세스마이닝", "IDP", "OCR",
        ],
    },
    {
        "name":         "IT조선",
        "rss":          "https://it.chosun.com/rss/allArticle.xml",
        "source_type":  "domestic",
        "source_country": "kor",
        "priority":     7,
        "categories":   ["AI", "데이터엔지니어링", "RPA"],
        "keywords": [
            "AI", "인공지능", "LLM", "에이전트",
            "데이터", "파이프라인", "자동화", "RPA",
            "클라우드", "머신러닝", "생성형",
            "에이전틱", "agentic", "오케스트레이션",
            "MCP", "RAG", "벡터DB",
            "데이터거버넌스", "데이터품질",
            "Snowflake", "Databricks", "Airflow",
            "망분리", "보안", "사이버",
            "프로세스마이닝", "IDP", "OCR",
        ],
    },
    {
        "name":         "GeekNews",
        "rss":          "https://feeds.feedburner.com/geeknews-feed",
        "source_type":  "domestic",
        "source_country": "kor",
        "priority":     9,
        "categories":   ["AI", "데이터엔지니어링", "RPA"],
        "keywords":     [],   # 전체 수집 (개발자 큐레이션)
        "exclude_keywords": [   # 제외 키워드 (새로 추가)
            "스포츠", "연예", "부동산", "주식",
            "맛집", "여행", "패션",
        ]
    },

    # ── 해외 (global) ─────────────────────────────────────────
    {
        "name":         "TechCrunch",
        "rss":          "https://techcrunch.com/feed/",
        "source_type":  "global",
        "source_country": "usa",
        "priority":     8,
        "categories":   ["AI", "RPA"],
        "keywords": [
            "AI", "artificial intelligence", "LLM", "agent",
            "automation", "machine learning",
            "OpenAI", "Anthropic", "Google DeepMind",
            "workflow", "robotic", "process",  # ← 추가
            "no-code", "low-code",  # ← 추가
            "agentic", "orchestration", "multimodal",
            "RAG", "vector", "embedding", "knowledge graph",
            "fine-tuning", "reasoning", "MCP",
            "data pipeline", "data governance",
            "RPA", "workflow automation", "process mining",
            "document AI", "OCR", "IDP",
            "Snowflake", "Databricks", "Hugging Face",
            "GPU", "inference", "on-premise",
        ],
    },
    {
        "name":         "The Verge",
        "rss":          "https://www.theverge.com/rss/index.xml",
        "source_type":  "global",
        "source_country": "usa",
        "priority":     7,
        "categories":   ["AI"],
        "keywords": [
            "AI", "artificial intelligence", "LLM", "agent",
            "OpenAI", "Anthropic", "ChatGPT", "Claude", "Gemini",
            "agentic", "orchestration", "multimodal",
            "RAG", "vector", "embedding", "knowledge graph",
            "fine-tuning", "reasoning", "MCP",
            "data pipeline", "data governance",
            "RPA", "workflow automation", "process mining",
            "document AI", "OCR", "IDP",
            "Snowflake", "Databricks", "Hugging Face",
            "GPU", "inference", "on-premise",
        ],
    },
]

# ── 카테고리 자동 분류 키워드 ─────────────────────────────────
CATEGORY_KEYWORDS = {
    "AI": [
        "AI", "인공지능", "LLM", "에이전트", "agent",
        "ChatGPT", "Claude", "Gemini", "GPT",
        "생성형", "generative", "머신러닝", "machine learning",
        "딥러닝", "deep learning", "OpenAI", "Anthropic",
        "멀티모달", "multimodal", "RAG", "파인튜닝", "fine-tuning",
    ],
    "데이터엔지니어링": [
        "데이터 파이프라인", "data pipeline", "ETL",
        "데이터 웨어하우스", "data warehouse",
        "Snowflake", "Databricks", "BigQuery", "Kafka",
        "데이터 품질", "data quality", "벡터", "vector",
        "knowledge graph", "임베딩", "embedding",
    ],
    "RPA": [
        "RPA", "자동화", "automation", "UiPath",
        "workflow", "로보틱", "robotic",
        "업무 자동화", "지능형 자동화", "프로세스 마이닝",
        "OCR", "문서 AI", "document AI", "IDP",
    ],
}


# ── 유틸 함수 ─────────────────────────────────────────────────
def parse_pub_date(entry) -> str:
    if hasattr(entry, 'published_parsed') and entry.published_parsed:
        dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        return dt.isoformat()
    return datetime.now(KST).isoformat()


def clean_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text or '')
    return re.sub(r'\s+', ' ', text).strip()


def classify_category(title: str, site_categories: list) -> str:
    """제목 기반 카테고리 분류"""
    if len(site_categories) == 1:
        return site_categories[0]

    title_lower = title.lower()
    scores = {}
    for cat in site_categories:
        keywords = CATEGORY_KEYWORDS.get(cat, [])
        score    = sum(1 for kw in keywords if kw.lower() in title_lower)
        if score > 0:
            scores[cat] = score

    return max(scores, key=scores.get) if scores else site_categories[0]


def has_keyword(title: str, keywords: list) -> bool:
    """키워드 필터"""
    if not keywords:
        return True
    title_lower = title.lower()
    return any(kw.lower() in title_lower for kw in keywords)


def flush_batch(batch: list) -> None:
    if not batch:
        return
    try:
        supabase.table('site_news') \
            .upsert(batch, on_conflict='url') \
            .execute()
        logger.info(f"  └ upsert {len(batch)}건 완료")
    except Exception as e:
        logger.error(f"  └ 저장 오류: {e}")
    batch.clear()


# ── 사이트별 수집 ─────────────────────────────────────────────
def fetch_site(site: dict) -> list:
    name     = site['name']
    rss_url  = site['rss']
    keywords = site.get('keywords', [])

    try:
        feed    = feedparser.parse(rss_url)
        entries = getattr(feed, 'entries', [])
        logger.info(f"  [{name}] {len(entries)}건 수신")
    except Exception as e:
        logger.error(f"  [{name}] RSS 수집 실패: {e}")
        return []

    records = []
    for entry in entries:
        title = entry.get('title', '').strip()
        if not title:
            continue

        # 키워드 필터
        if not has_keyword(title, keywords):
            continue

        # 본문 (RSS description이 있으면 저장)
        body = clean_html(entry.get('summary', '') or entry.get('description', ''))
        body = body[:2000] if body else None

        records.append({
            'title':          title[:500],
            'body_text':      body,
            'summary_ko':     None,           # 해외 기사는 processor에서 번역
            'url':            entry.get('link', ''),
            'source':         name,
            'source_type':    site['source_type'],
            'source_country': site['source_country'],
            'category':       classify_category(title, site['categories']),
            'priority':       site['priority'],
            'pub_date':       parse_pub_date(entry),
        })

    logger.info(f"  [{name}] 필터 후 {len(records)}건")
    return records


# ── 메인 실행 ─────────────────────────────────────────────────
def fetch_and_store_sites():
    logger.info("=== 직접 사이트 RSS 수집 시작 ===")
    total = 0
    batch = []
    BATCH_SIZE = 50

    for site in SITES:
        logger.info(f"[{site['name']}] 수집 중...")
        records = fetch_site(site)

        for record in records:
            batch.append(record)
            total += 1
            if len(batch) >= BATCH_SIZE:
                flush_batch(batch)

        time.sleep(2)   # 사이트 간 요청 간격

    flush_batch(batch)
    logger.info(f"=== 수집 완료 | 총 {total}건 저장 ===")
    return total


if __name__ == "__main__":
    fetch_and_store_sites()