# news_processor_v2.py
# 변경사항:
#   - _filter_site_news() 추가: site_news LLM 관련성 필터링
#   - _summarize_papers() 수정: 논문 URL 포함
#   - _save_report() 수정: Google News 섹션도 sections_json에 포함
#   - run() 수정: site_news LLM 필터링 적용

import os
import json
import re
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List
from crawler import crawl_body

from supabase import create_client, Client
from dotenv import load_dotenv
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))

MODEL_FAST    = "gpt-5-nano"
MODEL_QUALITY = "gpt-5.4-nano"


# ─────────────────────────────────────────────────────────────
# 1. 분류기
# ─────────────────────────────────────────────────────────────
class NewsClassifier:
    SECTIONS = {
        "new_services":   "🚀 New Services & Launches",
        "updates":        "🔄 Updates & Policy Changes",
        "investment":     "💰 Investment & Business",
        "infrastructure": "🛠 Infrastructure & Dev Tools",
        "trends":         "📊 Technology Trends & Research",
        "other":          "📌 Other",
    }


# ─────────────────────────────────────────────────────────────
# 2. 리포트 생성기
# ─────────────────────────────────────────────────────────────
class WeeklyReportGenerator:

    def __init__(self, client: OpenAI):
        self.client = client

    def generate_report(
        self,
        big_category: str,
        classified: Dict[str, List[Dict]],
        week_label: str,
        paper_section: str = "",
        domestic_news: List[Dict] = None,
        global_news: List[Dict] = None,
    ) -> str:
        section_contents = self._prepare_section_contents(classified)

        has_news   = any(section_contents.values())
        has_direct = bool(domestic_news or global_news)

        if not has_news and not has_direct:
            return f"# {big_category} 주간 리포트\n\n이번 주 관련 뉴스가 없습니다."

        prompt = self._build_report_prompt(
            big_category, week_label, section_contents,
            domestic_news or [], global_news or []
        )

        try:
            resp = self.client.chat.completions.create(
                model=MODEL_QUALITY,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "당신은 IT 업계 전문 뉴스레터 에디터입니다. "
                            "독자는 현업 개발자와 기획자입니다. "
                            "마크다운 형식으로 명확하고 실용적인 인사이트를 제공하세요. "
                            "반드시 한국어로 작성하세요."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
            )
            report_text = resp.choices[0].message.content.strip()

            if paper_section:
                report_text += f"\n\n---\n{paper_section}"

            return report_text

        except Exception as e:
            logger.error(f"리포트 생성 실패: {e}")
            return ""

    def _prepare_section_contents(
        self, classified: Dict[str, List[Dict]]
    ) -> Dict[str, List[Dict]]:
        result = {}
        for section, news_list in classified.items():
            if section == 'other':
                continue
            top_news = news_list[:5]
            enriched = []
            for news in top_news:
                enriched.append({
                    'title':        news.get('title', ''),
                    'url':          news.get('url', ''),
                    'source':       news.get('source', ''),
                    'body_summary': news.get('body_text', '')[:500]
                                    if news.get('body_text') else None,
                })
            result[section] = enriched
        return result

    def _build_report_prompt(
        self,
        big_category: str,
        week_label: str,
        section_contents: Dict[str, List[Dict]],
        domestic_news: List[Dict],
        global_news: List[Dict],
    ) -> str:
        section_labels = NewsClassifier.SECTIONS

        # Google News 섹션
        sections_text = ""
        for section, items in section_contents.items():
            if not items:
                continue
            label = section_labels.get(section, section)
            sections_text += f"\n### {label}\n"
            for item in items:
                sections_text += f"- **{item['title']}** ({item['source']})\n"
                sections_text += f"  URL: {item['url']}\n"
                if item.get('body_summary'):
                    sections_text += f"  본문: {item['body_summary']}\n"

        # 국내 직접 소스
        domestic_text = ""
        for item in domestic_news[:8]:
            domestic_text += f"- **{item['title']}** ({item['source']})\n"
            if item.get('body_text'):
                domestic_text += f"  내용: {item['body_text'][:300]}\n"
            domestic_text += f"  출처: {item['url']}\n"

        # 해외 직접 소스
        global_text = ""
        for item in global_news[:5]:
            title   = item['title']
            summary = item.get('summary_ko') or item.get('body_text', '')[:200]
            global_text += f"- **{title}** ({item['source']})\n"
            if summary:
                global_text += f"  요약: {summary}\n"
            global_text += f"  출처: {item['url']}\n"

        return f"""다음 뉴스를 바탕으로 [{big_category}] 분야 {week_label} 주간 리포트를 작성해주세요.

## 국내 주요 뉴스 (AI타임스/ZDNet Korea/IT조선/GeekNews)
{domestic_text if domestic_text else "해당 없음"}

## 글로벌 동향 (TechCrunch/The Verge)
{global_text if global_text else "해당 없음"}

## 기타 뉴스 (Google News)
{sections_text if sections_text else "해당 없음"}

리포트 형식:
---
# {big_category} 주간 동향 | {week_label}

## 🔑 이번 주 핵심 요약
(3줄 이내 핵심 동향 요약)

## 섹션별 상세

### 📰 국내 주요 뉴스
(AI타임스/ZDNet Korea/IT조선/GeekNews 기사 중심)
(각 항목마다: 제목, 2~3줄 핵심 내용 요약, 출처)

### 🌐 글로벌 동향
(TechCrunch/The Verge 기사 중심, 한국어 요약)
(각 항목마다: 제목, 2~3줄 핵심 내용 요약, 출처)

### 📡 추가 뉴스 (Google News)
(위 섹션에서 다루지 않은 주요 뉴스, 내용 있는 것만)
(각 항목마다: 제목, 1~2줄 요약, 출처)

## 💡 실무 적용 포인트

현재 [{big_category}] 분야 실무자가
실제 업무에 적용 가능한 항목만 작성한다.

작성 규칙:

1. 뉴스 내용 재요약 금지
2. 추상적인 조언 금지
3. "중요하다", "고려해야 한다", "필요하다", "활용 가능하다" 사용 금지
4. 즉시 업무 티켓(Jira/회의 안건) 생성 가능한 수준으로 작성
5. 최대 3개까지만 작성
6. 실제 시스템 / DB / API / 파이프라인 / 운영 관점 포함

각 항목은 아래 형식 유지:

### [한 줄 제목]

[업무 영향]
- 현재 [{big_category}] 업무에서 영향을 받는 영역

[이번 주 액션]
- 바로 실행 가능한 작업 1~3개

[구현 예시]
- DB 테이블 / API / 파이프라인 / 코드 예시 중 하나

[주의사항]
- 운영 시 발생 가능한 문제 또는 제한사항
---

규칙:
- 국내 뉴스와 글로벌 동향 섹션을 반드시 먼저 작성
- 해외 기사는 반드시 한국어로 요약
- 내용 없는 섹션은 생략
- 각 항목에 출처 포함
- 전체 한국어 작성
- 실무자가 바로 활용할 수 있는 인사이트 위주로"""


# ─────────────────────────────────────────────────────────────
# 3. 메인 프로세서
# ─────────────────────────────────────────────────────────────
class WeeklyNewsProcessor:
    BIG_CATEGORIES = ["AI", "데이터엔지니어링", "RPA"]

    # 카테고리별 컨텍스트 (LLM 필터링용)
    CATEGORY_CONTEXT = {
        "AI": "LLM, 에이전트, 생성형AI, AI모델, AI서비스, AI정책, AI인프라, 오케스트레이션, MCP, RAG, 제외: AI 거버넌스 정치 이슈, 노동/임금 분쟁,거시 경제 정책, AI 윤리 선언",
        "데이터엔지니어링": "데이터파이프라인, RAG, 벡터DB, 데이터거버넌스, 지식그래프, 데이터품질, ETL, GraphRAG",
        "RPA": "업무자동화, 프로세스자동화, 문서AI, OCR, 워크플로, RPA, IDP, 에이전틱자동화",
    }

    def __init__(self, supabase: Client, client: OpenAI):
        self.supabase   = supabase
        self.client     = client
        self.report_gen = WeeklyReportGenerator(client)

    # ── 데이터 조회 ───────────────────────────────────────────

    def get_weeks_news(self, days: int = 7) -> Dict[str, List[Dict]]:
        """최근 N일간 대분류별 뉴스 수집 (Google News)"""
        now   = datetime.now(KST)
        start = now - timedelta(days=days)
        result = {}

        for cat in self.BIG_CATEGORIES:
            try:
                resp = self.supabase.table('ai_news') \
                    .select('*') \
                    .eq('category', cat) \
                    .gte('pub_date', start.isoformat()) \
                    .order('pub_date', desc=True) \
                    .execute()

                result[cat] = resp.data or []
                logger.info(f"[{cat}] Google News {len(result[cat])}개 수집")

            except Exception as e:
                logger.error(f"[{cat}] 뉴스 조회 실패: {e}")
                result[cat] = []

        return result

    def get_weeks_site_news(self, days: int = 7) -> Dict[str, Dict[str, List[Dict]]]:
        """최근 N일간 직접 수집 사이트 뉴스 (국내/해외 분리)"""
        now   = datetime.now(KST)
        start = now - timedelta(days=days)

        result = {
            cat: {'domestic': [], 'global': []}
            for cat in self.BIG_CATEGORIES
        }

        for cat in self.BIG_CATEGORIES:
            try:
                resp = self.supabase.table('site_news') \
                    .select('*') \
                    .eq('category', cat) \
                    .gte('pub_date', start.isoformat()) \
                    .order('priority', desc=True) \
                    .order('pub_date', desc=True) \
                    .execute()

                for row in (resp.data or []):
                    source_type = row.get('source_type', 'domestic')
                    if source_type == 'global':
                        result[cat]['global'].append(row)
                    else:
                        result[cat]['domestic'].append(row)

                logger.info(
                    f"[{cat}] 직접소스: "
                    f"국내 {len(result[cat]['domestic'])}개 / "
                    f"해외 {len(result[cat]['global'])}개"
                )

            except Exception as e:
                logger.error(f"[{cat}] 직접소스 조회 실패: {e}")

        return result

    def get_weeks_papers(self, days: int = 7) -> Dict[str, List[Dict]]:
        """최근 N일간 카테고리별 논문 수집"""
        now   = datetime.now(KST)
        start = (now - timedelta(days=days)).date().isoformat()
        result = {}

        for cat in self.BIG_CATEGORIES:
            try:
                resp = self.supabase.table('arxiv_papers') \
                    .select('*') \
                    .eq('category', cat) \
                    .gte('published_date', start) \
                    .order('published_date', desc=True) \
                    .limit(50) \
                    .execute()

                result[cat] = resp.data or []
                logger.info(f"[{cat}] 논문 {len(result[cat])}편")

            except Exception as e:
                logger.error(f"[{cat}] 논문 조회 실패: {e}")
                result[cat] = []

        return result

    # ── 필터링/번역/요약 ──────────────────────────────────────

    def _filter_site_news(
        self, news_list: List[Dict], category: str
    ) -> List[Dict]:
        """
        site_news를 LLM으로 관련성 판단
        - 키워드 방식 한계 극복
        - 새로운 용어/표현도 맥락으로 판단
        - 순천 캐릭터 같은 무관한 기사 제거
        """
        if not news_list:
            return []

        titles_text = "\n".join([
            f"{i+1}. {n.get('title', '')}"
            for i, n in enumerate(news_list)
        ])

        context = self.CATEGORY_CONTEXT.get(category, "IT 기술")

        prompt = f"""다음은 IT 뉴스 제목 목록입니다.
[{category}] 분야 실무자에게 유의미한 기사 번호만 골라주세요.
이 분야의 핵심 관심사: [{context}]

판단 기준:
포함 ✅: 기술 동향, 신규 서비스/모델, 정책/규제, 투자/인수, 실무 적용 사례, 실무에 직접 영향을 주는 규제(예: 망분리 완화, 데이터 거버넌스 의무화)
제외 ❌: 지역 행사/관광/캐릭터, 연예/스포츠, 단순 수상/인사, 
        부동산/금융상품, 건강식품, 광고성 보도자료, IT와 무관한 산업
        AI를 단순 언급만 하는 정치/경제 정책 기사(예: AI 이익 분배, AI 일자리 정책, AI 규제 국회 발의 등 → 실무자가 코드/설계에 바로 활용 불가한 거시 정책)

{titles_text}

JSON: {{"selected": [1, 3, 5, ...]}}"""

        try:
            resp = self.client.chat.completions.create(
                model=MODEL_FAST,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw      = json.loads(resp.choices[0].message.content)
            selected = raw.get('selected', [])
            result   = [news_list[i-1] for i in selected if 0 < i <= len(news_list)]
            logger.info(f"  site_news 필터링: {len(news_list)}개 → {len(result)}개")
            return result

        except Exception as e:
            logger.error(f"site_news 필터링 실패: {e}")
            return news_list  # 실패 시 전체 반환

    def _translate_global_news(
        self, news_list: List[Dict], category: str
    ) -> List[Dict]:
        """해외 기사 한국어 요약 (summary_ko 없는 것만)"""
        to_translate = [n for n in news_list if not n.get('summary_ko')]

        if not to_translate:
            return news_list

        batch      = to_translate[:10]
        items_text = ""
        for i, n in enumerate(batch, 1):
            body = n.get('body_text', '')[:300] or ''
            items_text += f"\n{i}. 제목: {n['title']}\n   내용: {body}\n"

        prompt = f"""다음 {category} 분야 해외 IT 기사들을 한국어로 요약해주세요.
각 기사당 2~3문장으로 핵심 내용만 작성하세요.

{items_text}

JSON: {{"summaries": ["기사1 요약", "기사2 요약", ...]}}"""

        try:
            resp = self.client.chat.completions.create(
                model=MODEL_FAST,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw       = json.loads(resp.choices[0].message.content)
            summaries = raw.get('summaries', [])

            for i, news in enumerate(batch):
                if i < len(summaries):
                    news['summary_ko'] = summaries[i]
                    try:
                        self.supabase.table('site_news') \
                            .update({'summary_ko': summaries[i]}) \
                            .eq('id', news['id']) \
                            .execute()
                    except Exception:
                        pass

        except Exception as e:
            logger.error(f"번역 실패: {e}")

        return news_list

    def _summarize_papers(
        self, papers: List[Dict], category: str, week_label: str
    ) -> str:
        """LLM으로 논문 목록 요약 (URL 포함, 초록 핵심 요약)"""
        if not papers:
            return ""

        papers_text = ""
        for i, p in enumerate(papers[:10], 1):
            abstract = p.get('abstract', '')[:400]  # 초록 더 많이 포함
            url      = p.get('url', '')
            papers_text += f"\n{i}. 제목: {p['title']}\n   URL: {url}\n   초록: {abstract}\n"

        prompt = f"""다음은 {week_label} {category} 분야 arXiv 논문 목록입니다.

{papers_text}

아래 형식으로 연구 동향을 요약해주세요:

### 📚 연구 동향 (arXiv)
(이번 주 주목할 연구 흐름 2~3문장 요약)

**주요 논문:**
- **[논문 제목]**
  - 핵심 기여: (무엇을 해결했는지 1~2줄)
  - 실무 관련성: (어떤 실무 상황에 적용 가능한지 1줄)
  - 링크: [URL 그대로 사용]
(상위 3~5편만 선별, 실무 적용 가능성 높은 것 우선)

규칙:
- 반드시 위 목록에서 제공된 실제 URL을 그대로 사용하세요
- 한국어로 작성하세요
- 실무자가 "왜 이 논문이 중요한가"를 바로 알 수 있게 쓰세요"""

        try:
            resp = self.client.chat.completions.create(
                model=MODEL_QUALITY,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"논문 요약 실패: {e}")
            return ""

    # ── 유틸 ─────────────────────────────────────────────────

    def _get_week_label(self) -> str:
        now      = datetime.now(KST)
        week_num = (now.day - 1) // 7 + 1
        return f"{now.year}년 {now.month}월 {week_num}주차"

    def _rule_based_dedup(self, news_list: List[Dict]) -> List[Dict]:
        seen_urls   = set()
        seen_titles = set()
        result      = []

        for news in news_list:
            url        = news.get('url', '')
            title      = news.get('title', '').strip()
            normalized = re.sub(r'[^\w가-힣]', '', title).lower()

            if url in seen_urls: continue
            if normalized in seen_titles: continue

            prefix = normalized[:20]
            if any(t.startswith(prefix) for t in seen_titles): continue

            seen_urls.add(url)
            seen_titles.add(normalized)
            result.append(news)

        removed = len(news_list) - len(result)
        logger.info(f"규칙 기반 중복 제거: {removed}개 제거, {len(result)}개 남음")
        return result

    def _map_reduce_filter(self, news_list: List[Dict]) -> List[Dict]:
        candidates = []
        batch_size = 50

        for i in range(0, len(news_list), batch_size):
            batch       = news_list[i:i + batch_size]
            titles_text = "\n".join(
                [f"{j+1}. {n.get('title', '')}" for j, n in enumerate(batch)]
            )

            prompt = f"""다음 뉴스 제목에서 IT/AI/데이터/자동화 관련성이 높은 것을
번호 리스트로 골라주세요. 홍보성/루머/광고는 제외하세요.

{titles_text}

JSON: {{"selected": [1, 3, 5, ...]}}"""

            try:
                resp = self.client.chat.completions.create(
                    model=MODEL_FAST,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                raw      = json.loads(resp.choices[0].message.content)
                selected = raw.get('selected', [])
                for idx in selected:
                    actual = idx - 1
                    if 0 <= actual < len(batch):
                        candidates.append(batch[actual])
            except Exception as e:
                logger.error(f"Map-Reduce 필터 실패: {e}")
                candidates.extend(batch[:10])

            time.sleep(0.3)

        return candidates

    def _filter_and_classify(self, news_list: List[Dict]) -> Dict:
        result = {k: [] for k in NewsClassifier.SECTIONS}

        if len(news_list) <= 100:
            batches = [news_list]
        else:
            candidates = self._map_reduce_filter(news_list)
            batches    = [candidates]

        for batch in batches:
            titles_text = "\n".join(
                [f"{j+1}. {n.get('title', '')}" for j, n in enumerate(batch)]
            )

            prompt = f"""다음 뉴스 제목을 카테고리로 분류하세요.

{titles_text}

카테고리:
- new_services: 신제품/신규 모델/서비스 출시
- updates: 기존 서비스 업데이트, 정책 변경
- investment: 투자, M&A, 파트너십
- infrastructure: 개발도구, API, 클라우드, 하드웨어
- trends: 연구, 트렌드, 분석 리포트, 규제
- other: 홍보성/루머/광고/가치 없는 기사

반드시 other로 분류:
- 채용/장학/교육 프로그램 홍보
- 단순 수상/선정 공지
- 주가/밸류에이션 분석
- 의료/게임/에너지/관광 등 IT와 무관한 도메인

JSON: {{"new_services":[1,3],"updates":[2],"investment":[],"infrastructure":[4],"trends":[5],"other":[6]}}"""

            try:
                resp = self.client.chat.completions.create(
                    model=MODEL_FAST,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                raw = json.loads(resp.choices[0].message.content)

                for section, indices in raw.items():
                    if section not in result: continue
                    for idx in indices:
                        actual = idx - 1
                        if 0 <= actual < len(batch):
                            result[section].append(batch[actual])

            except Exception as e:
                logger.error(f"선별/분류 LLM 실패: {e}")
                result['other'].extend(batch)

            time.sleep(0.3)

        selected = sum(len(v) for k, v in result.items() if k != 'other')
        logger.info(f"선별/분류 완료: {len(news_list)}개 → {selected}개 선별")
        return result

    def _enrich_with_body(self, news_list: List[Dict]) -> List[Dict]:
        stats = {
            "success_jina": 0, "success_direct": 0,
            "decode_failed": 0, "paywalled": 0, "failed": 0
        }

        for news in news_list:
            body, status = crawl_body(news.get('url', ''))
            news['body_text'] = body
            news['has_body']  = body is not None
            stats[status]     = stats.get(status, 0) + 1
            time.sleep(1)

        total   = len(news_list)
        success = stats['success_jina'] + stats['success_direct']
        logger.info(
            f"크롤링 완료: {success}/{total} | "
            f"Jina {stats['success_jina']} | "
            f"직접 {stats['success_direct']} | "
            f"실패 {stats['failed']}"
        )
        return news_list

    def _save_report(
        self,
        category: str,
        week_label: str,
        report_text: str,
        classified: Dict[str, List[Dict]],
        domestic_news: List[Dict] = None,
        global_news: List[Dict] = None,
    ) -> None:
        today = datetime.now(KST).date().isoformat()

        sections_data = {}

        # Google News 섹션 (new_services, updates 등)
        for section, items in classified.items():
            if section == 'other': continue
            sections_data[section] = [
                {
                    'title':  news.get('title', ''),
                    'body':   news.get('body_text', '')[:2000] if news.get('body_text') else '',
                    'url':    news.get('real_url') or news.get('url', ''),
                    'source': news.get('source', ''),
                    'pub_date': news.get('pub_date', ''),  # ← 추가
                }
                for news in items[:5]
            ]

        # 국내 직접 소스
        if domestic_news:
            sections_data['domestic'] = [
                {
                    'title':    n.get('title', ''),
                    'body':     n.get('body_text', '')[:2000] if n.get('body_text') else '',
                    'url':      n.get('url', ''),
                    'source':   n.get('source', ''),
                    'priority': n.get('priority', 5),
                    'pub_date': n.get('pub_date', ''),  # ← 추가
                }
                for n in domestic_news[:8]
            ]

        # 해외 직접 소스
        if global_news:
            sections_data['global'] = [
                {
                    'title':      n.get('title', ''),
                    'summary_ko': n.get('summary_ko', ''),
                    'body':       n.get('body_text', '')[:2000] if n.get('body_text') else '',
                    'url':        n.get('url', ''),
                    'source':     n.get('source', ''),
                    'pub_date':   n.get('pub_date', ''),  # ← 추가
                }
                for n in global_news[:5]
            ]

        try:
            self.supabase.table('weekly_reports') \
                .delete() \
                .eq('category', category) \
                .eq('week_label', week_label) \
                .execute()

            self.supabase.table('weekly_reports').insert({
                'category':      category,
                'week_label':    week_label,
                'report_text':   report_text,
                'sections_json': json.dumps(sections_data, ensure_ascii=False),
                'publish_date':  today,
                'created_at':    datetime.now(timezone.utc).isoformat(),
            }).execute()

            logger.info(f"[{category}] 리포트 저장 완료")

        except Exception as e:
            logger.error(f"[{category}] 리포트 저장 실패: {e}")

    # ── 메인 실행 ─────────────────────────────────────────────

    def run(self, days: int = 7) -> None:
        from embedder import semantic_dedup
        from scorer import filter_by_score

        week_label         = self._get_week_label()
        news_by_category   = self.get_weeks_news(days)
        site_news_by_cat   = self.get_weeks_site_news(days)
        papers_by_category = self.get_weeks_papers(days)

        for big_cat in self.BIG_CATEGORIES:
            news_list = news_by_category.get(big_cat, [])
            site_data = site_news_by_cat.get(big_cat, {'domestic': [], 'global': []})

            domestic_news = site_data['domestic']
            global_news   = site_data['global']

            if not news_list and not domestic_news and not global_news:
                logger.info(f"[{big_cat}] 수집된 뉴스 없음, 스킵")
                continue

            logger.info(
                f"\n[{big_cat}] 시작: "
                f"Google {len(news_list)}개 / "
                f"국내직접 {len(domestic_news)}개 / "
                f"해외직접 {len(global_news)}개"
            )

            # ── site_news LLM 필터링 (순천 캐릭터 같은 무관 기사 제거) ──
            if domestic_news:
                domestic_news = self._filter_site_news(domestic_news, big_cat)

            if global_news:
                global_news = self._filter_site_news(global_news, big_cat)

            # ── Google News 처리 ──────────────────────────────
            if news_list:
                news_list = self._rule_based_dedup(news_list)
                logger.info(f"  규칙 중복제거 후: {len(news_list)}개")

                news_list = semantic_dedup(news_list, threshold=0.85)
                logger.info(f"  임베딩 중복제거 후: {len(news_list)}개")

                news_list = filter_by_score(news_list, top_n=100)
                logger.info(f"  스코어링 후: {len(news_list)}개")

                classified = self._filter_and_classify(news_list)

                for section, items in classified.items():
                    if section == 'other' or not items: continue
                    classified[section] = self._enrich_with_body(items[:5])
            else:
                classified = {k: [] for k in NewsClassifier.SECTIONS}

            # ── 해외 직접 소스 한국어 요약 ───────────────────
            if global_news:
                logger.info(f"  해외 기사 {len(global_news)}개 한국어 요약 중...")
                global_news = self._translate_global_news(global_news, big_cat)

            # ── 논문 요약 ────────────────────────────────────
            papers        = papers_by_category.get(big_cat, [])
            paper_section = self._summarize_papers(papers, big_cat, week_label)

            # ── 리포트 생성 ───────────────────────────────────
            report = self.report_gen.generate_report(
                big_category  = big_cat,
                classified    = classified,
                week_label    = week_label,
                paper_section = paper_section,
                domestic_news = domestic_news,
                global_news   = global_news,
            )

            # ── DB 저장 ──────────────────────────────────────
            self._save_report(
                category      = big_cat,
                week_label    = week_label,
                report_text   = report,
                classified    = classified,
                domestic_news = domestic_news,
                global_news   = global_news,
            )

        logger.info("\n=== 전체 처리 완료 ===")


# ─────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────
def main():
    load_dotenv()

    openai_api_key = os.getenv('OPENAI_API_KEY')
    supabase_url   = os.getenv('SUPABASE_URL')
    supabase_key   = os.getenv('SUPABASE_KEY')

    if not all([openai_api_key, supabase_url, supabase_key]):
        raise ValueError(
            "환경변수 누락: OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY 확인"
        )

    client   = OpenAI(api_key=openai_api_key)
    supabase = create_client(supabase_url, supabase_key)

    processor = WeeklyNewsProcessor(supabase, client)
    processor.run(days=7)


if __name__ == "__main__":
    main()