import os
import feedparser
import logging
import requests
import re
from datetime import datetime
from supabase import create_client
from dotenv import load_dotenv
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('scraper.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('rss_scraper')

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not all([SUPABASE_URL, SUPABASE_KEY]):
    raise ValueError("Missing Supabase credentials")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
logger.info("Supabase 연결 성공")

# ── 카테고리 & 키워드 ─────────────────────────────────────────
CATEGORIES: dict[str, dict[str, list[str]]] = {

    "AI": {

        # 회사/브랜드 (신규 모델 발표, 전략, 네이밍 변경 캐치)
        "AI_플랫폼": [
            "OpenAI",
            "Anthropic",
            "Google DeepMind",
            "Meta AI",
            "xAI",
            "Mistral AI",
            "Cohere",
            "Perplexity AI",
        ],

        # 모델 패밀리명 (버전 무관, 시리즈 전체 커버)
        # 버전명(숫자)은 제외 → "GPT" 검색 시 GPT-5, GPT-4.1 등 모두 잡힘
        "LLM_모델패밀리": [
            "GPT",  # GPT-5, GPT-4.1-nano 등 전 버전 커버
            "Claude",  # Claude Opus, Sonnet, Haiku, Mistral 등
            "Gemini",  # Gemini 2.5 Pro, Flash 등
            "LLaMA",  # Meta 오픈소스 시리즈
            "Grok",  # xAI 모델
            "Copilot",  # MS 코파일럿 시리즈
            "DeepSeek",  # 중국 오픈소스, 시장 영향력 큼
            "Qwen",  # Alibaba 모델 시리즈
            "Gemma",  # Google 오픈소스 시리즈
        ],

        # LLM 기술 트렌드 (특정 모델 무관한 개념)
        "LLM_트렌드": [
            "LLM",
            "large language model",
            "foundation model",
            "거대언어모델",
            "multimodal model",
            "reasoning model",
            "오픈소스 LLM",
            "context window",  # 주요 경쟁 지표
            "AI benchmark",  # 모델 성능 비교 뉴스
        ],

        # AI 에이전트 (가장 빠르게 성장하는 분야)
        "AI_에이전트": [
            "AI agent",
            "agentic AI",
            "AI 에이전트",
            "multi-agent",
            "MCP protocol",
            "LangGraph",
            "AutoGen",
            "computer use AI",
        ],

        # AI 서비스/제품 (B2B/B2C 활용)
        "AI_서비스": [
            "AI assistant",
            "AI copilot",
            "generative AI",
            "생성형 AI",
            "AI 챗봇",
            "enterprise AI",
        ],

        # 인프라/배포
        "AI_인프라": [
            "vLLM",
            "Ollama",
            "TensorRT-LLM",
            "RAG",
            "vector database",
            "AI inference",
            "GPU cluster",
            "on-premise AI",
        ],

        # 규제/정책/윤리
        "AI_규제정책": [
            "EU AI Act",
            "AI 규제",
            "AI governance",
            "AI safety",
            "AI 윤리",
            "AI copyright",
        ],
    },

    "데이터엔지니어링": {

        # 오케스트레이션/파이프라인 도구
        "파이프라인_도구": [
            "Apache Airflow",
            "dbt",
            "Apache Kafka",
            "Apache Flink",
            "Prefect",
            "Dagster",
            "데이터 파이프라인",
        ],

        # 처리 엔진
        "처리_엔진": [
            "Apache Spark",
            "Databricks",
            "DuckDB",
            "Apache Iceberg",
            "Delta Lake",
            "데이터 레이크하우스",
        ],

        # 데이터 웨어하우스
        "데이터_웨어하우스": [
            "Snowflake",
            "BigQuery",
            "Amazon Redshift",
            "ClickHouse",
            "데이터 웨어하우스",
        ],

        # 데이터 품질/관측
        "데이터_품질": [
            "data quality",
            "data observability",
            "data lineage",
            "데이터 품질",
            "데이터 거버넌스",
            "data governance",
        ],

        # AI와 데이터엔지니어링 교차점
        "AI_데이터": [
            "data engineering AI",
            "AI pipeline",
            "feature store",
            "MLOps",
            "LLMOps",
            "데이터 엔지니어링",
        ],
    },

    "RPA": {

        # RPA 플랫폼
        "RPA_플랫폼": [
            "UiPath",
            "Automation Anywhere",
            "Power Automate",
            "Blue Prism",
            "삼성SDS RPA",
            "RPA 플랫폼",
        ],

        # 지능형 자동화 (RPA + AI 결합 트렌드)
        "지능형_자동화": [
            "intelligent automation",
            "hyperautomation",
            "AI RPA",
            "agentic automation",
            "IDP",                      # Intelligent Document Processing
            "지능형 자동화",
            "업무 자동화 AI",
        ],

        # 프로세스 마이닝
        "프로세스_마이닝": [
            "process mining",
            "Celonis",
            "task mining",
            "프로세스 마이닝",
            "process intelligence",
        ],

        # RPA 트렌드/시장
        "RPA_시장동향": [
            "RPA market",
            "RPA 트렌드",
            "디지털 전환 자동화",
            "업무 자동화 시장",
            "robotic process automation",
        ],
    },
}

# ── 설정 ──────────────────────────────────────────────────────
BATCH_SIZE = 50
LANGUAGES = [
    {"hl": "ko", "gl": "KR", "ceid": "KR:ko", "country_cd": "kor"},
    {"hl": "en-US", "gl": "US", "ceid": "US:en", "country_cd": "usa"},
]

CRAWL_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
}
BODY_MAX_CHARS = 2000
CRAWL_TIMEOUT  = 8


# ── 본문 크롤러 ───────────────────────────────────────────────
def crawl_body(url: str) -> tuple[str | None, str]:
    """
    URL 본문 크롤링
    반환: (본문텍스트 or None, status)
    status: success / failed / paywalled / timeout
    """
    # Google News 리다이렉트 URL 처리
    if "news.google.com" in url:
        try:
            resp = requests.get(
                url, headers=CRAWL_HEADERS,
                timeout=CRAWL_TIMEOUT, allow_redirects=True
            )
            url = resp.url  # 실제 기사 URL로 교체
        except Exception:
            return None, "failed"

    try:
        resp = requests.get(
            url, headers=CRAWL_HEADERS,
            timeout=CRAWL_TIMEOUT, allow_redirects=True
        )

        # 페이월 감지 (구독 유도 패턴)
        paywall_signals = [
            "subscribe", "subscription", "sign in to read",
            "구독", "로그인 후", "회원 전용"
        ]
        if any(s in resp.text.lower() for s in paywall_signals):
            # 페이월이어도 일부 본문이 있을 수 있으니 일단 파싱 시도
            pass

        soup = BeautifulSoup(resp.text, 'html.parser')

        # 불필요 태그 제거
        for tag in soup(['script', 'style', 'nav', 'footer',
                         'header', 'aside', 'form', 'iframe']):
            tag.decompose()

        # RSS description을 fallback으로 먼저 확보
        body = None

        # 본문 후보 선택자 (우선순위 순)
        selectors = [
            'article',
            '[class*="article-body"]',
            '[class*="post-content"]',
            '[class*="entry-content"]',
            '[class*="news-body"]',
            'main',
            '.content',
            'body',
        ]

        for selector in selectors:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator=' ', strip=True)
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) > 200:
                    body = text[:BODY_MAX_CHARS]
                    break

        if body:
            return body, "success"

        return None, "failed"

    except requests.Timeout:
        return None, "timeout"
    except Exception as e:
        logger.debug(f"크롤링 오류 ({url[:60]}): {e}")
        return None, "failed"


# ── 유틸 함수 ─────────────────────────────────────────────────
def build_rss_url(keyword: str, lang: dict) -> str:
    q = keyword.replace(" ", "+")
    return (
        f"https://news.google.com/rss/search?q={q}"
        f"&hl={lang['hl']}&gl={lang['gl']}&ceid={lang['ceid']}"
    )

def clean_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def parse_pub_date(entry) -> datetime:
    if hasattr(entry, 'published_parsed') and entry.published_parsed:
        return datetime(*entry.published_parsed[:6])
    return datetime.now()

def get_source(entry) -> str:
    if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
        return entry.source.title
    return getattr(entry, 'author', 'Unknown')

def flush_batch(batch: list) -> None:
    if not batch:
        return
    try:
        supabase.table('ai_news').upsert(batch, on_conflict='url').execute()
        logger.info(f"  └ upsert {len(batch)}건 완료")
    except Exception as e:
        logger.error(f"  └ 배치 insert 오류: {e}")
    batch.clear()


# ── 핵심 수집 함수 ────────────────────────────────────────────
def build_record(
    entry,
    country_cd: str,
    keyword: str,
    category: str,
    subcategory: str,
) -> dict | None:
    title = entry.get('title', '').strip()
    if not title:
        return None

    return {
        'title':             title[:500],
        'source':            get_source(entry)[:200],
        'url':               entry.get('link', ''),
        'keyword':           keyword,
        'pub_date':          parse_pub_date(entry).isoformat(),
        'category':          category,
        'subcategory':       subcategory,
        'source_country_cd': country_cd,
        # body 관련 필드 전부 제거
    }


# ── 메인 실행 ─────────────────────────────────────────────────
def fetch_and_store_news() -> int:
    logger.info("=== 뉴스 수집 시작 ===")
    total = 0
    batch: list[dict] = []

    for category, subcats in CATEGORIES.items():
        logger.info(f"[대분류] {category}")

        for subcategory, keywords in subcats.items():
            logger.info(f"  [소분류] {subcategory}")

            for keyword in keywords:
                for lang in LANGUAGES:
                    url = build_rss_url(keyword, lang)
                    try:
                        feed    = feedparser.parse(url)
                        entries = getattr(feed, 'entries', [])
                        logger.info(
                            f"    [{lang['country_cd']}] "
                            f"'{keyword}' → {len(entries)}건"
                        )
                    except Exception as e:
                        logger.warning(f"RSS 수집 실패 ({keyword}): {e}")
                        continue

                    for entry in entries:
                        record = build_record(
                            entry, lang['country_cd'],
                            keyword, category, subcategory
                        )
                        if record:
                            batch.append(record)
                            total += 1

                        if len(batch) >= BATCH_SIZE:
                            flush_batch(batch)

    flush_batch(batch)
    logger.info(f"=== 수집 완료 | 총 {total}건 처리 ===")
    return total


if __name__ == "__main__":
    fetch_and_store_news()