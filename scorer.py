# scorer.py
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)  # ← 추가

# 신뢰도 높은 출처
HIGH_AUTHORITY_SOURCES = {
    # 영문
    'techcrunch', 'theverge', 'wired', 'reuters', 'bloomberg',
    'venturebeat', 'ars technica', 'mit technology review',
    # 국문
    'zdnet', '전자신문', '아이뉴스', 'ai타임스', '블로터',
}

# 중요도 높은 시그널 키워드
POSITIVE_SIGNALS = {
    'launch', 'release', 'open-source', 'funding', 'acquisition',
    'breakthrough', 'partnership', 'raises', 'announces',
    '출시', '공개', '발표', '투자', '인수', '협력', '오픈소스',
}

# 노이즈 키워드
NEGATIVE_SIGNALS = {
    'rumor', 'opinion', 'might', 'could', 'speculation',
    '루머', '의견', '전망', '예상', '추측',
    'nfl', 'sports', '스포츠',  # 스포츠 관련
    'battery skills', '배터리 스킬', '배터리',  # 도메인 특화 교육
    'pharmacy', '약국',  # 헬스케어 특화
    'student fellows', 'fellowship',    # 채용/장학
    '교육 프로그램', '커리큘럼',          # 교육 상품
    '단신', '선정',                      # 단순 공지
    '주가', '밸류에이션', 'valuation',    # 주식 분석
}


def score_news(news: dict) -> float:
    title  = news.get('title', '').lower()
    source = news.get('source', '').lower()
    pub_date = news.get('pub_date', '')

    score = 0.0

    # 1. 출처 신뢰도 (0~0.4)
    if any(s in source for s in HIGH_AUTHORITY_SOURCES):
        score += 0.4

    # 2. 제목 긍정 시그널 (0~0.3)
    matches = sum(1 for kw in POSITIVE_SIGNALS if kw in title)
    score += min(matches * 0.15, 0.3)

    # 3. 제목 부정 시그널 (-0.2)
    if any(kw in title for kw in NEGATIVE_SIGNALS):
        score -= 0.2

    # 4. 최신성 (0~0.3)
    try:
        from datetime import datetime, timezone, timedelta
        kst = timezone(timedelta(hours=9))
        pub  = datetime.fromisoformat(pub_date).astimezone(kst)
        now  = datetime.now(kst)
        days_old = (now - pub).days
        score += max(0.3 - days_old * 0.05, 0)
    except Exception:
        pass

    return round(score, 3)


def filter_by_score(
    news_list: list,
    top_n: int = 100,
) -> list:
    """점수 기준 상위 N개만 선택"""
    scored = [(score_news(n), n) for n in news_list]
    scored.sort(key=lambda x: x[0], reverse=True)

    result = [n for _, n in scored[:top_n]]
    logger.info(
        f"스코어링 필터: {len(news_list)}개 → 상위 {len(result)}개 선택"
    )
    return result