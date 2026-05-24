# crawler.py

import re
import time
import logging
import requests
from bs4 import BeautifulSoup
from googlenewsdecoder import gnewsdecoder

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
}
BODY_MAX_CHARS = 2000
TIMEOUT        = 10

PAYWALL_DOMAINS = {
    "nytimes.com", "ft.com", "wsj.com",
    "bloomberg.com", "economist.com",
}


def _is_paywall(url: str) -> bool:
    return any(d in url for d in PAYWALL_DOMAINS)


def _decode_google_url(url: str) -> str | None:
    """Google News URL → 실제 기사 URL 디코딩"""
    if "news.google.com" not in url:
        return url
    try:
        result = gnewsdecoder(url, interval=1)
        if result.get("status"):
            return result["decoded_url"]
        logger.debug(f"디코딩 실패: {result.get('message')}")
        return None
    except Exception as e:
        logger.debug(f"디코딩 오류: {e}")
        return None


def _jina(url: str) -> str | None:
    """Jina AI Reader로 본문 추출"""
    try:
        resp = requests.get(
            f"https://r.jina.ai/{url}",
            headers={"Accept": "text/plain", "X-Return-Format": "text"},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return None

        text = resp.text

        # Markdown Content: 이후만 추출
        if 'Markdown Content:' in text:
            body = text.split('Markdown Content:', 1)[1].strip()
        elif 'Content:' in text:
            body = text.split('Content:', 1)[1].strip()
        else:
            # 구분자 없으면 앞 4줄(메타) 제거
            lines = text.strip().split('\n')
            body = '\n'.join(lines[4:]).strip()

        body = re.sub(r'\s+', ' ', body).strip()
        return body[:BODY_MAX_CHARS] if len(body) > 200 else None

    except Exception as e:
        logger.debug(f"Jina 실패 ({url[:60]}): {e}")
    return None


def _direct(url: str) -> str | None:
    """직접 크롤링 (Jina 실패 시 fallback)"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer',
                         'header', 'aside', 'form', 'iframe']):
            tag.decompose()

        for selector in [
            'article',
            '[class*="article-body"]',
            '[class*="post-content"]',
            '[class*="entry-content"]',
            '[class*="news-body"]',
            'main',
        ]:
            el = soup.select_one(selector)
            if el:
                text = re.sub(
                    r'\s+', ' ', el.get_text(separator=' ')
                ).strip()
                if len(text) > 200:
                    return text[:BODY_MAX_CHARS]

    except Exception as e:
        logger.debug(f"직접 크롤링 실패 ({url[:60]}): {e}")
    return None


def crawl_body(google_url: str) -> tuple[str | None, str]:
    """
    Google News URL → 실제 URL 디코딩 → 본문 크롤링

    반환: (본문 or None, status)
    status:
        success_jina    - Jina 성공
        success_direct  - 직접 크롤링 성공
        decode_failed   - Google URL 디코딩 실패
        paywalled       - 페이월 도메인
        failed          - 모든 방법 실패
    """
    # 1. Google News URL 디코딩
    real_url = _decode_google_url(google_url)
    if not real_url:
        return None, "decode_failed"

    # 2. 페이월 스킵
    if _is_paywall(real_url):
        return None, "paywalled"

    # 3. Jina 시도
    body = _jina(real_url)
    if body:
        return body, "success_jina"

    # 4. 직접 크롤링 fallback
    body = _direct(real_url)
    if body:
        return body, "success_direct"

    return None, "failed"