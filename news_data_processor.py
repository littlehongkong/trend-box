import os
import json
import re
import time
import logging
from datetime import datetime, timezone, timedelta
import hashlib
from typing import Dict, List, Optional

import requests
from supabase import create_client, Client
from dotenv import load_dotenv
from deep_translator import GoogleTranslator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_processor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SmartDuplicateFilter:
    """키워드 기반 사전 필터링 + 캐싱 + LLM 기반 의미론적 중복 제거"""

    def __init__(self, openai_api_key: str = None):
        self.content_hashes = set()
        self.title_cache = {}
        self.similarity_threshold = 0.85
        self.openai_api_key = openai_api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-5-nano"
        self._llm_cache = {}

    def _generate_content_hash(self, news_item: Dict) -> str:
        """뉴스 항목의 콘텐츠 해시 생성"""
        content = f"{news_item.get('title', '')}:{news_item.get('source', '')}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _normalize_title(self, title: str) -> str:
        """제목 정규화"""
        title = re.sub(r'[^\w\s가-힣]', '', title.lower())
        title = re.sub(r'\s+', ' ', title).strip()
        return title

    def _extract_keywords(self, title: str) -> set:
        """제목에서 주요 키워드 추출"""
        keywords = set()

        # 영어 단어 (3글자 이상)
        english_words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
        keywords.update(english_words)

        # 한글 단어 (2글자 이상)
        korean_words = re.findall(r'[가-힣]{2,}', title)
        keywords.update(korean_words)

        # 숫자 포함 단어 (버전명, 모델명)
        number_words = re.findall(r'\b\w*\d+\w*\b', title.lower())
        keywords.update(number_words)

        return keywords

    def _make_llm_request(self, messages: List[Dict]) -> Optional[Dict]:
        """LLM API 호출 with caching"""
        if not self.openai_api_key:
            return None

        content = json.dumps(messages, sort_keys=True)
        cache_key = hashlib.md5(content.encode('utf-8')).hexdigest()

        if cache_key in self._llm_cache:
            logger.debug(f"LLM 캐시 히트: {cache_key[:8]}...")
            return self._llm_cache[cache_key]

        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # 캐시에 저장
            self._llm_cache[cache_key] = result
            logger.debug(f"LLM API 호출 성공 및 캐시 저장: {cache_key[:8]}...")

            return result

        except Exception as e:
            logger.error(f"LLM API 호출 실패: {e}")
            return None

    def _check_semantic_duplicates_batch(self, candidates: List[Dict]) -> List[int]:
        """LLM을 사용한 배치 의미론적 중복 검사"""
        if not candidates or len(candidates) < 2 or not self.openai_api_key:
            return []

        # 제목들을 번호와 함께 정리
        titles_text = "\n".join([
            f"{i + 1}. {item['title']}"
            for i, item in enumerate(candidates)
        ])

        prompt = f"""다음 AI 뉴스 제목들 중에서 의미상 중복되는 항목들을 찾아주세요.

{titles_text}

중복 기준:
1. 같은 사건/발표를 다루는 경우
2. 같은 제품/서비스의 같은 업데이트를 다루는 경우
3. 같은 연구/보고서를 다루는 경우
4. 표현만 다르고 실질적으로 같은 내용인 경우

중복되는 그룹이 있다면 다음 JSON 형식으로 응답해주세요:
{{"duplicates": [[1,3], [5,7,9]]}}

중복이 없다면:
{{"duplicates": []}}

각 그룹에서 첫 번째 번호의 기사를 남기고 나머지를 제거할 예정입니다."""

        messages = [{"role": "user", "content": prompt}]
        response = self._make_llm_request(messages)

        if not response:
            return []

        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')

        try:
            # JSON 파싱
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                duplicates = result.get('duplicates', [])

                # 제거할 인덱스들 수집 (각 그룹에서 첫 번째는 보존, 나머지는 제거)
                to_remove = []
                for group in duplicates:
                    if len(group) > 1:
                        # 인덱스를 0-based로 변환하고, 첫 번째 제외하고 나머지 추가
                        to_remove.extend([idx - 1 for idx in group[1:]])

                logger.info(f"LLM 중복 검사 결과: {len(duplicates)}개 그룹, {len(to_remove)}개 항목 제거 예정")
                return to_remove

        except Exception as e:
            logger.error(f"LLM 중복 검사 결과 파싱 실패: {e}")

        return []

    def is_duplicate(self, news_item: Dict, seen_news: List[Dict]) -> bool:
        """빠른 중복 검사 (기존 방식)"""
        title = news_item.get('title', '').strip()
        if not title:
            return True

        # 1. 해시 기반 검사
        content_hash = self._generate_content_hash(news_item)
        if content_hash in self.content_hashes:
            return True

        # 2. 정규화된 제목 검사
        normalized_title = self._normalize_title(title)
        if normalized_title in self.title_cache:
            return True

        # 3. 키워드 기반 유사도 검사
        current_keywords = self._extract_keywords(title)
        if len(current_keywords) == 0:
            return False

        for seen_item in seen_news[-20:]:  # 최근 20개만 비교
            seen_title = seen_item.get('title', '')
            seen_keywords = self._extract_keywords(seen_title)

            if len(seen_keywords) == 0:
                continue

            # Jaccard 유사도 계산
            intersection = len(current_keywords.intersection(seen_keywords))
            union = len(current_keywords.union(seen_keywords))

            if union > 0:
                similarity = intersection / union
                if similarity >= self.similarity_threshold:
                    return True

        # 중복이 아니면 캐시에 추가
        self.content_hashes.add(content_hash)
        self.title_cache[normalized_title] = True

        return False

    def remove_semantic_duplicates_llm(self, news_list: List[Dict]) -> List[Dict]:
        """LLM 기반 의미론적 중복 제거 (배치 처리)"""
        if not news_list or not self.openai_api_key:
            return news_list

        logger.info(f"LLM 기반 의미론적 중복 제거 시작: {len(news_list)}개")

        # 배치 크기 설정 (한 번에 너무 많이 처리하면 토큰 한계 초과)
        batch_size = 30  # 30개씩 처리
        total_removed = 0
        llm_call_count = 0

        filtered_news = news_list.copy()

        # 배치별로 처리
        for batch_start in range(0, len(filtered_news), batch_size):
            batch_end = min(batch_start + batch_size, len(filtered_news))
            batch = filtered_news[batch_start:batch_end]

            # 배치 크기가 2개 미만이면 중복 검사 불필요
            if len(batch) < 2:
                continue

            llm_call_count += 1
            to_remove_indices = self._check_semantic_duplicates_batch(batch)

            if to_remove_indices:
                # 역순으로 제거 (인덱스 변화 방지)
                for idx in sorted(to_remove_indices, reverse=True):
                    actual_idx = batch_start + idx
                    if actual_idx < len(filtered_news):
                        removed_title = filtered_news[actual_idx].get('title', '')[:50]
                        logger.debug(f"LLM 중복 제거: {removed_title}...")
                        filtered_news.pop(actual_idx)
                        total_removed += 1

                # 배치 크기 조정 (제거된 항목 때문에 인덱스가 변했으므로)
                batch_size = max(20, batch_size - len(to_remove_indices))

            # API 호출 간격 (과도한 호출 방지)
            if batch_end < len(news_list):
                time.sleep(0.3)

        logger.info(f"LLM 중복 제거 완료: {total_removed}개 제거, {len(filtered_news)}개 남음")
        logger.info(f"LLM API 호출 횟수: {llm_call_count}회")

        return filtered_news


class KeywordBasedClassifier:
    """키워드 기반 사전 분류기"""

    def __init__(self):
        self.keyword_rules = {
            "new_services": [
                # 신제품 출시, 신규 서비스, 모델/버전
                "출시", "론칭", "런칭", "출범", "공개", "발표", "선보", "데뷔",
                "launch", "release", "unveil", "debut", "introduce", "announce",
                "신규", "새로운", "첫", "최초", "new", "first", "latest",
                "버전", "version", "모델", "model",
                # 신제품 관련 고유 명칭(예: GPT, Claude, Gemini 등)
                "gpt", "claude", "gemini", "llama"
            ],
            "updates": [
                # 기능 개선, 보안, 정책, 운영 안정성, UX/UI 변경
                "업데이트", "개선", "향상", "강화", "추가", "확대", "확장",
                "update", "improve", "enhance", "upgrade", "expand", "extend",
                "보안", "security", "안정성", "stability",
                "거버넌스", "governance",
                "정책", "규제", "가이드라인", "policy", "regulation", "guideline",
                "메모리", "context window", "운영", "operation",
                "UI", "UX", "사용자 환경"
            ],
            "investment": [
                # 투자, M&A, 기업 협력 및 파트너십
                "투자", "펀딩", "조달", "투자유치", "시리즈", "round", "funding", "raise", "series",
                "인수", "합병", "파트너십", "협력", "계약", "acquisition", "merger", "partnership", "deal", "contract",
                "기업", "회사", "스타트업", "company", "startup", "corp",
                "인프라 인수", "매수"
            ],
            "infrastructure": [
                # 개발 도구, API, 클라우드, 하드웨어, 플랫폼(기술적 의미)
                "클라우드", "서버", "gpu", "칩", "반도체", "하드웨어",
                "cloud", "server", "chip", "semiconductor", "hardware",
                "api", "sdk", "도구", "툴", "프레임워크", "framework", "tool", "platform", "development",
                "MCU", "엣지", "edge", "인퍼런스", "배포", "pipeline", "자동화", "automation"
            ],
            "trends": [
                # 기술 동향, 연구, 산업 변화, 법규, 분석 리포트
                "연구", "보고서", "분석", "전망", "예측", "동향", "트렌드",
                "research", "report", "analysis", "forecast", "trend",
                "기술", "혁신", "breakthrough", "technology", "innovation",
                "규제", "policy", "governance", "법", "법률", "legal", "compliance",
                "시장", "시장 변화", "산업", "산업 변화"
            ],
        }

    def classify_by_keywords(self, title: str) -> Optional[str]:
        """키워드 기반 분류"""
        title_lower = title.lower()
        scores = {}

        for category, keywords in self.keyword_rules.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in title_lower:
                    # 키워드 길이에 따른 가중치
                    weight = len(keyword) / 5.0
                    score += weight

            if score > 0:
                scores[category] = score

        if scores:
            # 가장 높은 점수의 카테고리 반환
            return max(scores.items(), key=lambda x: x[1])[0]

        return None


class OptimizedNewsClassifier:
    """최적화된 뉴스 분류기"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.model = "gpt-5-nano"  # 더 저렴한 모델 사용
        self._api_cache = {}

        # 키워드 기반 사전 분류기
        self.keyword_classifier = KeywordBasedClassifier()

        # 뉴스 카테고리 정의
        self.categories = {
            "new_services": "New Services/Launches",
            "updates": "Updates/Policy Changes",
            "investment": "Investment/Business",
            "infrastructure": "Infrastructure/Dev Tools",
            "trends": "Technology Trends",
            "other": "Other News"
        }

    def _get_cache_key(self, content: str) -> str:
        """캐시 키 생성"""
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _make_api_request(self, messages: List[Dict]) -> Optional[Dict]:
        """OpenAI API 호출 with caching"""
        content = json.dumps(messages, sort_keys=True)
        cache_key = self._get_cache_key(content)

        if cache_key in self._api_cache:
            logger.info(f"캐시 히트: {cache_key[:8]}...")
            return self._api_cache[cache_key]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # 캐시에 저장
            self._api_cache[cache_key] = result
            logger.info(f"API 호출 성공 및 캐시 저장: {cache_key[:8]}...")

            return result

        except Exception as e:
            logger.error(f"API 호출 실패: {e}")
            raise

    def classify_news_optimized(self, news_list: List[Dict]) -> Dict[str, List[Dict]]:
        """최적화된 뉴스 분류 - 키워드 우선, LLM은 최소한만"""

        classified = {category: [] for category in self.categories.keys()}
        need_llm_classification = []

        # 1단계: 키워드 기반 사전 분류
        logger.info("1단계: 키워드 기반 사전 분류 시작")

        for i, news in enumerate(news_list):
            title = news.get('title', '')

            # 키워드 기반 분류 시도
            category = self.keyword_classifier.classify_by_keywords(title)

            if category:
                classified[category].append({
                    'index': i,
                    'title': title,
                    'news_data': news
                })
                logger.debug(f"키워드 분류: [{category}] {title[:50]}...")
            else:
                # 키워드로 분류되지 않은 뉴스만 LLM으로 처리
                need_llm_classification.append((i, title, news))

        keyword_classified_count = sum(len(items) for items in classified.values())
        logger.info(f"키워드 분류 완료: {keyword_classified_count}/{len(news_list)}개")

        # 2단계: LLM 분류 (키워드로 분류되지 않은 뉴스만)
        if need_llm_classification:
            logger.info(f"2단계: LLM 분류 시작 ({len(need_llm_classification)}개)")

            # 배치 크기를 크게 늘림 (한 번에 더 많이 처리)
            batch_size = 100

            for batch_start in range(0, len(need_llm_classification), batch_size):
                batch_end = min(batch_start + batch_size, len(need_llm_classification))
                batch = need_llm_classification[batch_start:batch_end]

                batch_titles = [item[1] for item in batch]
                llm_results = self._classify_batch_with_llm(batch_titles)

                # 결과 병합
                for j, (original_idx, title, news_data) in enumerate(batch):
                    category = 'other'  # 기본값

                    # LLM 분류 결과에서 해당 항목 찾기
                    for cat, items in llm_results.items():
                        for item in items:
                            if item['index'] == j:
                                category = cat
                                break
                        if category != 'other':
                            break

                    classified[category].append({
                        'index': original_idx,
                        'title': title,
                        'news_data': news_data
                    })

                # API 호출 간격
                if batch_end < len(need_llm_classification):
                    time.sleep(0.5)

        # 통계 출력
        total_classified = sum(len(items) for items in classified.values())
        logger.info(f"분류 완료: {total_classified}개")
        for category, items in classified.items():
            if items:
                logger.info(f"  {self.categories[category]}: {len(items)}개")

        return classified

    def _classify_batch_with_llm(self, titles: List[str]) -> Dict[str, List[Dict]]:
        """LLM을 사용한 배치 분류"""
        if not titles:
            return {}

        titles_text = "\n".join([f"{i + 1}. {title}" for i, title in enumerate(titles)])

        prompt = f"""다음 AI 뉴스 헤드라인들을 아래 카테고리 기준에 따라 가장 적절한 항목으로 분류해주세요.

{titles_text}

카테고리 설명:
- new_services: 신제품 출시, 신규 모델/버전 공개, 신규 서비스 론칭 관련 뉴스
  예) GPT-5 공개, Claude 신규 요금제 출시

- updates: 기존 서비스 기능 개선, 보안 강화, 정책 및 거버넌스 변경, 사용자 환경(UI/UX) 개선
  예) 보안 업데이트, 메모리 기능 추가, 정책 변경 안내

- investment: 투자 유치, 인수합병(M&A), 전략적 파트너십, 기업 협력 발표
  예) AI 스타트업 투자 유치, 구글 크롬 인수 제안

- infrastructure: AI 개발 도구, API, SDK, 클라우드 인프라, 하드웨어, 배포 자동화 관련
  예) NVIDIA GPU 출시, HuggingFace 라이브러리 업데이트

- trends: 기술 트렌드, 산업 변화, 연구 결과, 법적 규제, 분석 리포트
  예) AI 산업 전망 보고서, 규제 정책 변화, 연구 결과 발표

- other: 위의 어떤 카테고리에도 명확히 속하지 않는 뉴스

중복되는 주제가 있을 경우, 가장 대표적인 하나의 카테고리로만 분류해주세요.


JSON 형식으로만 응답:
{{"new_services": [{{"index": 0, "title": "제목"}}], "updates": [], ...}}

중복되는 주제가 있는 경우, 가장 대표적인 하나의 카테고리로만 분류해주세요.
정확한 JSON 형식만 응답해주세요.
"""

        messages = [{"role": "user", "content": prompt}]
        response = self._make_api_request(messages)

        if not response:
            return {}

        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')

        try:
            # JSON 파싱
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                return result
        except Exception as e:
            logger.error(f"LLM 분류 결과 파싱 실패: {e}")

        return {}

    def generate_summary(self, category: str, news_items: List[Dict]) -> str:
        """카테고리별 요약 생성 - 더 효율적으로"""
        if not news_items or len(news_items) < 3:
            return "**요약에 충분한 정보가 부족합니다.**"

        # 제목만 추출하여 토큰 수 최소화
        titles = [item.get('title', '') for item in news_items[:15]]  # 최대 15개만
        titles_text = "\n".join([f"- {title}" for title in titles if title])

        category_name = self.categories.get(category, category)

        prompt = f"""다음 [{category_name}] 카테고리 AI 뉴스들의 핵심 트렌드를 2-3문장으로 요약하세요:

{titles_text}

개발자 관점의 실용적 인사이트를 중심으로 간결하게 작성해주세요."""

        messages = [{"role": "user", "content": prompt}]
        response = self._make_api_request(messages)

        if not response:
            return f"{category_name} 관련 {len(news_items)}건의 뉴스입니다."

        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        return content.strip()



class OptimizedNewsProcessor:
    """최적화된 뉴스 처리기 (LLM 중복 제거 추가)"""

    def __init__(self, supabase: Client, classifier: OptimizedNewsClassifier):
        self.supabase = supabase
        self.classifier = classifier
        # LLM 중복 제거를 위해 OpenAI API 키 전달
        self.duplicate_filter = SmartDuplicateFilter(classifier.api_key)
        self.kst = timezone(timedelta(hours=9))

    def get_todays_news(self) -> List[Dict]:
        """오늘자 뉴스 데이터 수집"""
        today_kst = datetime.now(self.kst)
        today_start_kst = today_kst.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end_kst = today_kst.replace(hour=23, minute=59, second=59, microsecond=999999)

        today_start_utc = today_start_kst - timedelta(hours=9)
        today_end_utc = today_end_kst - timedelta(hours=9)

        logger.info(f"뉴스 조회 범위: {today_start_kst} ~ {today_end_kst} (KST)")

        try:
            response = self.supabase.table('ai_news') \
                .select('*') \
                .gte('pub_date', today_start_utc.isoformat()) \
                .lte('pub_date', today_end_utc.isoformat()) \
                .eq('is_duplicate', False) \
                .execute()

            news_list = response.data if hasattr(response, 'data') else []
            logger.info(f"조회된 뉴스: {len(news_list)}개")

            # 제목 정리
            for news in news_list:
                if news.get('title'):
                    news['title'] = re.sub(r'[\"'']', '"', news['title']).strip()

            return news_list

        except Exception as e:
            logger.error(f"뉴스 조회 실패: {e}")
            return []

    def remove_duplicates_smart(self, news_list: List[Dict]) -> List[Dict]:
        """스마트 중복 제거 (기존 방식 + LLM 추가)"""
        if len(news_list) <= 1:
            return news_list

        logger.info(f"스마트 중복 제거 시작: {len(news_list)}개")

        # 1단계: 기존 방식으로 중복 제거
        unique_news = []
        news_list.sort(key=lambda x: x.get('pub_date', ''), reverse=True)

        for news in news_list:
            if not self.duplicate_filter.is_duplicate(news, unique_news):
                unique_news.append(news)

        removed_count_step1 = len(news_list) - len(unique_news)
        logger.info(f"1단계 (키워드 기반) 중복 제거: {removed_count_step1}개 제거, {len(unique_news)}개 남음")

        # 2단계: LLM 기반 의미론적 중복 제거
        if len(unique_news) >= 10:  # 충분한 뉴스가 있을 때만 LLM 사용
            final_news = self.duplicate_filter.remove_semantic_duplicates_llm(unique_news)
            removed_count_step2 = len(unique_news) - len(final_news)
            logger.info(f"2단계 (LLM 기반) 중복 제거: {removed_count_step2}개 추가 제거")
        else:
            final_news = unique_news
            logger.info("뉴스 수가 부족하여 LLM 중복 제거 생략")

        total_removed = len(news_list) - len(final_news)
        logger.info(f"총 중복 제거: {total_removed}개 제거, {len(final_news)}개 최종 남음")

        return final_news

    # 나머지 메서드들은 기존과 동일...
    def _is_english_news(self, news_item: Dict) -> bool:
        """영어 뉴스 판별"""
        return news_item.get('source_country_cd') == 'usa'

    def translate_english_news(self, news_list: List[Dict]) -> None:
        """영어 뉴스 제목 번역 (배치 처리)"""
        english_news = [
            news for news in news_list
            if self._is_english_news(news) and not news.get('title_kr') and news.get('title')
        ]

        if not english_news:
            return

        logger.info(f"영어 뉴스 번역 시작: {len(english_news)}개")

        translated_count = 0
        for news in english_news:
            try:
                translated = GoogleTranslator(source='en', target='ko').translate(news['title'])
                if translated:
                    # DB 업데이트
                    self.supabase.table('ai_news') \
                        .update({'title_kr': translated}) \
                        .eq('id', news['id']) \
                        .execute()

                    # 메모리 상의 객체도 업데이트
                    news['title_kr'] = translated
                    translated_count += 1

            except Exception as e:
                logger.error(f"번역 실패 (ID: {news.get('id')}): {e}")

        logger.info(f"번역 완료: {translated_count}개")

    def process_and_save_news(self, news_list: List[Dict]) -> None:
        """뉴스 처리 및 저장"""
        if not news_list:
            logger.info("처리할 뉴스가 없습니다.")
            return

        # 분류
        classified_results = self.classifier.classify_news_optimized(news_list)

        # 각 카테고리별 요약 생성 (뉴스가 충분한 경우만)
        summary_results = {}
        for category, items in classified_results.items():
            news_data = [item['news_data'] for item in items]
            if len(items) >= 3:  # 3개 이상인 경우만 요약 생성
                summary = self.classifier.generate_summary(category, news_data)
                summary_results[category] = {
                    'items': news_data,
                    'summary': summary
                }
                logger.info(f"{self.classifier.categories[category]}: {len(items)}개 (요약 생성)")
            else:
                summary_results[category] = {
                    'items': news_data,
                    'summary': None
                }
                logger.info(f"{self.classifier.categories[category]}: {len(items)}개 (요약 생략)")

        # 데이터베이스 저장
        if summary_results:
            self.save_to_newsletter_sections(summary_results)

    def save_to_newsletter_sections(self, classified_results: Dict[str, Dict]) -> None:
        """분류 결과를 데이터베이스에 저장"""
        today_kst = datetime.now(self.kst).date()

        try:
            # 기존 데이터 삭제
            self.supabase.table('newsletter_sections') \
                .delete() \
                .eq('publish_date', today_kst.isoformat()) \
                .execute()

            sections_to_insert = []
            display_order = 1

            for category, data in classified_results.items():
                news_items = []
                for news in data['items']:
                    news_item = {
                        'id': news['id'],
                        'title': news['title'],
                        'source': news.get('source', ''),
                        'url': news.get('url', ''),
                        'pub_date': news.get('pub_date', '')
                    }

                    if news.get('title_kr'):
                        news_item['title_kr'] = news['title_kr']

                    news_items.append(news_item)

                section_data = {
                    'publish_date': today_kst.isoformat(),
                    'section_name': category,
                    'section_title': self.classifier.categories[category],
                    'summary': data['summary'],
                    'content': news_items,
                    'display_order': display_order,
                    'created_at': datetime.now(timezone.utc).isoformat()
                }

                sections_to_insert.append(section_data)
                display_order += 1

            if sections_to_insert:
                self.supabase.table('newsletter_sections') \
                    .insert(sections_to_insert) \
                    .execute()

                # 처리된 뉴스 플래그 업데이트
                all_news_ids = [news['id'] for data in classified_results.values() for news in data['items']]

                self.supabase.table('ai_news') \
                    .update({'is_processed': True}) \
                    .in_('id', all_news_ids) \
                    .execute()

                logger.info(f"뉴스레터 저장 완료: {len(sections_to_insert)}개 섹션, {len(all_news_ids)}개 뉴스")

        except Exception as e:
            logger.error(f"저장 실패: {e}")
            raise

    def process_daily_news(self) -> None:
        """일일 뉴스 처리 메인 프로세스"""
        start_time = time.time()
        logger.info("=== 최적화된 AI 뉴스 처리 시작 (LLM 중복 제거 포함) ===")

        try:
            # 1. 뉴스 수집
            news_list = self.get_todays_news()
            if not news_list:
                logger.info("처리할 뉴스가 없습니다.")
                return

            # 2. 스마트 중복 제거 (기존 방식 + LLM)
            news_list = self.remove_duplicates_smart(news_list)
            if not news_list:
                logger.info("중복 제거 후 남은 뉴스가 없습니다.")
                return

            # 3. 영어 뉴스 번역
            self.translate_english_news(news_list)

            # 4. 분류 및 저장
            self.process_and_save_news(news_list)

            # 완료 로그
            elapsed_time = time.time() - start_time
            logger.info(f"=== 처리 완료 ===")
            logger.info(f"처리 시간: {elapsed_time:.2f}초")
            logger.info(f"최종 처리된 뉴스: {len(news_list)}개")

        except Exception as e:
            logger.error(f"처리 중 오류: {e}")
            raise


def main():
    """메인 실행 함수"""
    try:
        load_dotenv()

        openai_api_key = os.getenv('OPENAI_API_KEY')
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')

        if not all([openai_api_key, supabase_url, supabase_key]):
            raise ValueError("필수 환경변수가 설정되지 않았습니다.")

        # 클라이언트 초기화
        supabase: Client = create_client(supabase_url, supabase_key)
        classifier = OptimizedNewsClassifier(openai_api_key)
        processor = OptimizedNewsProcessor(supabase, classifier)

        # 뉴스 처리 실행
        processor.process_daily_news()

    except Exception as e:
        logger.error(f"시스템 실행 실패: {e}")
        raise


if __name__ == "__main__":
    main()