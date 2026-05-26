# arxiv_scraper.py

import os
import time
import logging
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
from supabase import create_client
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('arxiv_scraper.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('arxiv_scraper')

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase     = create_client(SUPABASE_URL, SUPABASE_KEY)

KST = timezone(timedelta(hours=9))

# ── arXiv API 설정 ────────────────────────────────────────────
ARXIV_API = "https://export.arxiv.org/api/query"

# 카테고리별 검색 설정
# search_query: arXiv API 쿼리 문법
# cat:cs.AI AND (ti:agent OR ti:LLM) 처럼 조합 가능
SEARCH_CONFIG = {
    "AI": [
        {
            "query": "cat:cs.AI OR cat:cs.LG OR cat:cs.CL",
            "keywords": [
                "agent", "multi-agent", "agentic",
                "large language model", "LLM",
                "reasoning", "chain of thought",
                "instruction tuning", "RLHF",
                "multimodal", "vision language",
                "transformer", "attention",
                "prompt engineering", "in-context learning",
                "hallucination", "alignment",
            ],
            "max_results": 50,
        },
    ],

    "데이터엔지니어링": [
        {
            # AI 데이터 파이프라인 & 검색/지식 기술
            "query": (
                "cat:cs.IR OR cat:cs.DB OR cat:cs.LG OR cat:cs.AI"
            ),
            "keywords": [
                # RAG / 검색 기술
                "retrieval augmented generation", "RAG",
                "GraphRAG", "graph RAG",
                "knowledge graph", "knowledge base",
                "graph neural network",
                # 벡터/임베딩
                "vector database", "vector search",
                "embedding", "semantic search",
                "dense retrieval", "sparse retrieval",
                # 데이터 품질/파이프라인
                "data pipeline", "data quality",
                "data governance", "data lineage",
                "feature store", "MLOps", "LLMOps",
                # 청킹/인덱싱
                "chunking", "indexing strategy",
                "document parsing", "text extraction",
            ],
            "max_results": 50,
        },
    ],

    "RPA": [
        {
            # 문서 AI + 프로세스 자동화
            "query": (
                "cat:cs.AI OR cat:cs.CV OR cat:cs.CL"
            ),
            "keywords": [
                "robotic process automation", "RPA",
                "workflow automation", "process mining",
                "business process", "task automation",
                "document understanding", "document AI",
                "intelligent document processing", "IDP",
                "OCR", "optical character recognition",
                "form understanding", "information extraction",
                "GUI agent", "computer use", "web agent",
                "browser automation", "desktop automation",
            ],
            "max_results": 50,
        },
    ],
}


# ── 유틸 함수 ─────────────────────────────────────────────────
def fetch_arxiv(
    query: str,
    max_results: int = 50,
    start: int = 0,
    retry: int = 3,
    days: int = 7,         # ← 추가
) -> list[dict]:

    # 날짜 범위 계산
    now        = datetime.now(timezone.utc)
    date_from  = (now - timedelta(days=days)).strftime("%Y%m%d")
    date_to    = now.strftime("%Y%m%d")

    # 날짜 필터를 쿼리에 추가
    # submittedDate:[20260516 TO 20260523] 형태
    date_filter = f"submittedDate:[{date_from}0000 TO {date_to}2359]"
    full_query  = f"({query}) AND {date_filter}"

    params = {
        "search_query": full_query,
        "start":        start,
        "max_results":  max_results,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    }

    for attempt in range(retry):
        try:
            resp = requests.get(ARXIV_API, params=params, timeout=30)

            # 429 → 대기 후 재시도
            if resp.status_code == 429:
                wait = 30 * (attempt + 1)  # 30초, 60초, 90초
                logger.warning(f"  429 Rate Limit → {wait}초 대기 후 재시도 ({attempt+1}/{retry})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            break  # 성공 시 루프 탈출

        except requests.exceptions.HTTPError as e:
            if attempt < retry - 1:
                wait = 30 * (attempt + 1)
                logger.warning(f"  HTTP 오류 {e} → {wait}초 대기 후 재시도")
                time.sleep(wait)
                continue
            logger.error(f"arXiv API 오류: {e}")
            return []
        except Exception as e:
            logger.error(f"arXiv API 오류: {e}")
            return []

    # XML 파싱 (기존과 동일)
    ns = {
        'atom':  'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom',
    }
    root    = ET.fromstring(resp.text)
    entries = root.findall('atom:entry', ns)
    papers  = []

    for entry in entries:
        try:
            arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
            title    = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            abstract = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            authors  = ', '.join([
                a.find('atom:name', ns).text
                for a in entry.findall('atom:author', ns)
            ])
            published   = entry.find('atom:published', ns).text[:10]
            url         = f"https://arxiv.org/abs/{arxiv_id}"
            cats        = [c.get('term', '') for c in entry.findall('atom:category', ns)]
            primary_cat = cats[0] if cats else ''

            papers.append({
                'arxiv_id':       arxiv_id,
                'title':          title[:500],
                'abstract':       abstract[:3000],
                'authors':        authors[:500],
                'arxiv_category': primary_cat,
                'url':            url,
                'published_date': published,
            })
        except Exception as e:
            logger.debug(f"파싱 오류: {e}")
            continue

    return papers


def filter_by_keywords(papers: list[dict], keywords: list[str]) -> list[dict]:
    """제목+초록에 키워드 포함 여부로 필터링"""
    result = []
    keywords_lower = [k.lower() for k in keywords]

    for paper in papers:
        text = (paper['title'] + ' ' + paper['abstract']).lower()
        if any(kw in text for kw in keywords_lower):
            result.append(paper)

    return result


def save_papers(papers: list[dict], category: str) -> int:
    """Supabase에 저장 (arxiv_id 기준 upsert)"""
    if not papers:
        return 0

    # category 필드 추가
    for p in papers:
        p['category'] = category

    saved = 0
    batch_size = 20

    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        try:
            supabase.table('arxiv_papers') \
                .upsert(batch, on_conflict='arxiv_id') \
                .execute()
            saved += len(batch)
            logger.info(f"  저장 {saved}건")
        except Exception as e:
            logger.error(f"  저장 오류: {e}")

    return saved


# ── 메인 실행 ─────────────────────────────────────────────────
def fetch_and_store_papers(days:int = 7):
    logger.info(f"=== arXiv 논문 수집 시작 (최근 {days}일) ===")
    total = 0

    for category, configs in SEARCH_CONFIG.items():
        logger.info(f"[{category}] 수집 시작")

        for config in configs:
            papers = fetch_arxiv(
                query=config['query'],
                max_results=config['max_results'],
                days=days,  # ← 추가
            )
            logger.info(f"  arXiv 응답: {len(papers)}건")

            filtered = filter_by_keywords(papers, config['keywords'])
            logger.info(f"  키워드 필터 후: {len(filtered)}건")

            saved  = save_papers(filtered, category)
            total += saved

            # 카테고리 내 config 간 대기
            logger.info(f"  15초 대기 중...")
            time.sleep(15)  # 3초 → 15초

        # 카테고리 간 대기
        logger.info(f"  다음 카테고리까지 20초 대기 중...")
        time.sleep(20)  # 3초 → 20초

    logger.info(f"=== 수집 완료 | 총 {total}건 저장 ===")
    return total


if __name__ == "__main__":
    fetch_and_store_papers(days=7)