import os
import json
import re
import time
import logging
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import uuid

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


class NewsClassifier:
    """Together.ai LLM API를 사용하여 뉴스 분류 및 요약을 처리하는 클래스"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        self.model = "meta-llama/Llama-3-70b-chat-hf"
        self.max_chunk_size = 2000

        # 뉴스 카테고리 정의
        self.categories = {
            "new_services": "New Services/Launches",
            "updates": "Updates/Policy Changes",
            "investment": "Investment/Business",
            "infrastructure": "Infrastructure/Dev Tools",
            "trends": "Technology Trends",
            "other": "Other News"
        }

    def _make_api_request(self, messages: List[Dict], temperature: float = 0.1, max_tokens: int = 3000) -> Optional[
        Dict]:
        """Together.ai API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": 180
        }

        request_id = str(uuid.uuid4())[:8]
        logger.info(f"API 요청 시작 (ID: {request_id})")

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()

            result = response.json()

            # 디버깅용 파일 저장
            with open(f"api_request_{request_id}.json", "w", encoding="utf-8") as f:
                json.dump({"payload": payload, "response": result}, f, ensure_ascii=False, indent=2)

            logger.info(f"API 요청 성공 (ID: {request_id})")
            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"API 요청 실패 (ID: {request_id}): {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"응답 상태 코드: {e.response.status_code}")
                logger.error(f"응답 내용: {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"예상치 못한 오류 (ID: {request_id}): {e}")
            return None

    def _parse_json_response(self, text: str, request_id: str) -> Optional[Dict]:
        """JSON 응답 파싱 - 개선된 버전"""
        try:
            # 1. 마크다운 코드 블록에서 JSON 추출 (더 강력한 패턴)
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_text = json_match.group(1).strip()
                logger.info(f"코드 블록에서 JSON 추출 성공 (ID: {request_id})")
            else:
                # 2. 중괄호로 시작하는 첫 번째 JSON 객체 찾기
                brace_match = re.search(r'\{[\s\S]*\}', text, re.DOTALL)
                if brace_match:
                    json_text = brace_match.group(0).strip()
                    logger.info(f"중괄호 패턴으로 JSON 추출 성공 (ID: {request_id})")
                else:
                    # 3. 전체 텍스트를 JSON으로 시도
                    json_text = text.strip()
                    logger.info(f"전체 텍스트를 JSON으로 시도 (ID: {request_id})")

            # 4. JSON 형식 수정
            json_text = self._fix_json_format(json_text)

            # 5. 첫 번째 파싱 시도
            try:
                result = json.loads(json_text)
                logger.info(f"JSON 파싱 성공 (ID: {request_id})")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"첫 번째 JSON 파싱 실패 (ID: {request_id}): {e}")

                # 6. 부분적 JSON 파싱 시도
                return self._parse_partial_json(json_text, request_id)

        except Exception as e:
            logger.error(f"JSON 파싱 중 예상치 못한 오류 (ID: {request_id}): {e}")

            # 디버깅용 원본 응답 저장
            with open(f"parse_error_{request_id}.txt", "w", encoding="utf-8") as f:
                f.write(f"Original text:\n{text}\n\n")
                f.write(f"Error: {e}")

            return None

    def _fix_json_format(self, json_text: str) -> str:
        """JSON 형식 자동 수정 - 개선된 버전"""
        if not json_text:
            return json_text

        # 기본 정리
        json_text = json_text.strip()

        # JSON이 아닌 텍스트가 앞에 있는 경우 제거
        if not json_text.startswith('{') and not json_text.startswith('['):
            # 첫 번째 { 또는 [ 찾기
            start_pos = -1
            for i, char in enumerate(json_text):
                if char in '{[':
                    start_pos = i
                    break

            if start_pos != -1:
                json_text = json_text[start_pos:]
                logger.info("JSON 시작 부분 추출 완료")

        # JSON이 아닌 텍스트가 뒤에 있는 경우 제거
        if json_text.startswith('{'):
            # 마지막 }까지만 추출
            brace_count = 0
            last_valid_pos = -1

            for i, char in enumerate(json_text):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_valid_pos = i + 1
                        break

            if last_valid_pos > 0:
                json_text = json_text[:last_valid_pos]
                logger.info("JSON 끝 부분 추출 완료")

        # 불완전한 중괄호/대괄호 수정
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        open_brackets = json_text.count('[')
        close_brackets = json_text.count(']')

        # 누락된 닫기 괄호 추가
        if open_braces > close_braces:
            json_text += '}' * (open_braces - close_braces)
            logger.info(f"누락된 중괄호 {open_braces - close_braces}개 추가")

        if open_brackets > close_brackets:
            json_text += ']' * (open_brackets - close_brackets)
            logger.info(f"누락된 대괄호 {open_brackets - close_brackets}개 추가")

        # 잘못된 쉼표 수정 (마지막 요소 뒤의 쉼표)
        json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)

        # 누락된 쉼표 추가 (간단한 경우만)
        json_text = re.sub(r'}\s*{', r'},{', json_text)
        json_text = re.sub(r']\s*\[', r'],[', json_text)

        # 잘못된 따옴표 수정
        json_text = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_text)

        # 줄바꿈 문자가 문자열 안에 있는 경우 처리
        json_text = re.sub(r'("title":\s*"[^"]*)\n([^"]*")', r'\1 \2', json_text)

        return json_text

    def _parse_partial_json(self, json_text: str, request_id: str) -> Optional[Dict]:
        """부분적 JSON 파싱 시도 - 개선된 버전"""
        try:
            # JSON의 기본 구조 확인
            if not json_text.strip().startswith('{'):
                logger.error(f"유효한 JSON 구조가 아님 (ID: {request_id})")
                return None

            # 줄 단위로 점진적 파싱 시도
            lines = json_text.split('\n')

            # 더 작은 단위부터 시작 (5줄씩)
            for chunk_size in [5, 10, 15, 20]:
                if len(lines) < chunk_size:
                    continue

                for i in range(chunk_size, len(lines) + 1, chunk_size):
                    partial_json = '\n'.join(lines[:i])

                    # 기본 구조 유지
                    if not partial_json.strip().startswith('{'):
                        continue

                    # # 중괄호/대괄호 균형 맞추기
                    # balanced_json = self._balance_brackets(partial_json)
                    # 
                    # # 마지막 불완전한 항목 제거
                    # balanced_json = self._remove_incomplete_items(balanced_json)

                    try:
                        result = json.loads(partial_json)
                        logger.info(f"부분적 JSON 파싱 성공 (ID: {request_id}): {i}/{len(lines)} 라인")

                        # 빈 결과가 아닌지 확인
                        if isinstance(result, dict) and any(result.values()):
                            return result

                    except json.JSONDecodeError:
                        continue

            logger.error(f"모든 부분적 파싱 시도 실패 (ID: {request_id})")
            assert False, f"모든 부분적 JSON 파싱 시도가 실패했습니다. (ID: {request_id})"
            return None

        except Exception as e:
            logger.error(f"부분적 JSON 파싱 중 오류 (ID: {request_id}): {e}")
            return None

    def classify_news_batch(self, news_titles: List[str]) -> Dict[str, List[Dict]]:
        """뉴스 배치 분류"""
        if not news_titles:
            return {}

        titles_text = "\n".join([f"{i + 1}. {title}" for i, title in enumerate(news_titles)])

        prompt = f"""Classify these AI news headlines. Return ONLY JSON.

    {titles_text}

    카테고리:
    1. new_services - 새로운 AI 서비스, 제품 출시, 모델 릴리즈
    2. updates - 기존 서비스 업데이트, 정책 변경, 기능 개선
    3. investment - 투자, 인수합병, 비즈니스 파트너십, 기업 소식
    4. infrastructure - AI 인프라, 개발 도구, 플랫폼, 하드웨어
    5. trends - AI 기술 트렌드, 연구 결과, 업계 동향, 분석 보고서
    6. other - 위 카테고리에 속하지 않는 기타 AI 관련 뉴스

    REQUIRED FORMAT:
    {{"new_services": [{{"index": 1, "title": "exact title"}}], ...}}"""

        messages = [{"role": "user", "content": prompt}]

        response = self._make_api_request(messages, temperature=0.3)  # temperature 낮춤
        if not response:
            return {}

        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        request_id = str(uuid.uuid4())[:8]

        parsed_result = self._parse_json_response(content, request_id)
        if not parsed_result:
            return {}

        # 결과 정리 및 검증
        classified = {}
        for category in self.categories.keys():
            classified[category] = []
            if category in parsed_result:
                for item in parsed_result[category]:
                    if isinstance(item, dict) and 'index' in item and 'title' in item:
                        index = item['index'] - 1  # 0-based index로 변환
                        if 0 <= index < len(news_titles):
                            classified[category].append({
                                'index': index,
                                'title': news_titles[index]
                            })

        logger.info(f"분류 완료: {len(news_titles)}개 뉴스")
        return classified

    def generate_summary(self, category: str, news_items: List[Dict]) -> str:
        """카테고리별 통합 요약 생성 - 개발자 관점의 인사이트 제공"""
        if not news_items:
            return ""

        titles_text = "\n".join([f"- {item['title']}" for item in news_items])
        category_name = self.categories.get(category, category)


        prompt = f"""당신은 AI 기술 트렌드를 분석하여 개발자에게 실질적인 인사이트를 제공하는 전문가입니다.
    다음은 [{category_name}] 카테고리에 해당하는 AI 뉴스 헤드라인입니다:

    {titles_text}

    요구사항:
    1. 반드시 한국어로 작성
    2. 뉴스 개수가 너무 적거나, 유의미한 기술 흐름이나 공통 주제가 포착되지 않을 경우 요약을 생략하고, "**요약에 충분한 정보가 부족합니다.**" 라고만 응답하세요.
    3. 그렇지 않다면, 2~3문장 이내로 간결하게 작성하되, 단순 요약이 아니라 **개발자 관점의 핵심 인사이트**를 도출해 주세요.
    4. 뉴스 제목은 직접 언급하지 말고, 공통된 흐름, 기술적 맥락, 시사점 중심으로 설명하세요.
    5. 가능하다면, 개발자가 고려하거나 행동할 수 있는 구체적인 통찰을 포함하세요.

    요약 시작:"""

        messages = [{"role": "user", "content": prompt}]

        response = self._make_api_request(messages, temperature=0.1)
        if not response:
            return f"{category_name} 관련 {len(news_items)}건의 뉴스입니다."

        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        print(content)
        return content.strip()

    def _get_news_language(self, news_item: Dict) -> str:
        """뉴스 언어 판별 (한국어/영어)"""
        # 1. source_country_cd로 먼저 판단
        if news_item.get('source_country_cd') == 'usa':
            return 'english'
        elif news_item.get('source_country_cd') == 'korea':
            return 'korean'

        # 2. 도메인 기반 판단 (fallback)
        english_domains = [
            'techcrunch.com', 'theverge.com', 'wired.com', 'engadget.com',
            'venturebeat.com', 'arstechnica.com', 'cnet.com', 'zdnet.com',
            'bloomberg.com', 'reuters.com', 'cnbc.com', 'wsj.com',
            'nytimes.com', 'washingtonpost.com', 'theguardian.com', 'bbc.com'
        ]

        if news_item.get('link'):
            domain = urlparse(news_item['link']).netloc.lower()
            if any(eng_domain in domain for eng_domain in english_domains):
                return 'english'

        # 3. 제목의 한글 문자 비율로 판단
        title = news_item.get('title', '')
        korean_chars = len([c for c in title if '\uac00' <= c <= '\ud7af'])
        total_chars = len([c for c in title if c.isalpha()])

        if total_chars > 0 and korean_chars / total_chars > 0.3:
            return 'korean'

        return 'english'


    def _get_duplicate_check_groups(self, news_list: List[Dict]) -> Dict[str, List[Dict]]:
        """뉴스를 언어와 출처별로 그룹화"""
        groups = {
            'korean': [],
            'english': [],
            'same_source': {}  # source별 그룹
        }

        for news in news_list:
            language = self._get_news_language(news)
            groups[language].append(news)

            # 동일 출처 그룹화
            source = news.get('source', 'unknown')
            if source not in groups['same_source']:
                groups['same_source'][source] = []
            groups['same_source'][source].append(news)

        logger.info(f"그룹별 뉴스 수: 한국어 {len(groups['korean'])}개, 영어 {len(groups['english'])}개")
        logger.info(f"출처별 그룹: {len(groups['same_source'])}개 출처")

        return groups



    def check_duplicates_batch(self, news_batch: List[Dict], existing_news: List[Dict] = None, check_cross_language: bool = False) -> List[int]:
        """LLM 기반 중복 뉴스 검사 - 수정된 버전"""
        if len(news_batch) <= 1:
            return []

        # 언어별로 그룹화
        batch_groups = self._get_duplicate_check_groups(news_batch)
        existing_groups = self._get_duplicate_check_groups(existing_news or [])

        request_id = str(uuid.uuid4())[:8]
        logger.info(f"중복 검사 시작 (ID: {request_id}): 배치 {len(news_batch)}개")

        # 상세 로깅
        for i, news in enumerate(news_batch):
            lang = self._get_news_language(news)
            logger.debug(f"뉴스 {i + 1} [{lang}] ({news.get('source', 'N/A')}): {news['title'][:50]}...")

        current_titles = []
        existing_titles = []
        title_to_index = {}  # 제목 -> 원본 인덱스 매핑

        # 1. 동일 언어 내 중복 검사를 위한 제목 준비
        for lang in ['korean', 'english']:
            batch_lang_news = batch_groups[lang]
            existing_lang_news = existing_groups[lang]

            if not batch_lang_news:
                continue

            lang_current_titles = []
            lang_existing_titles = []

            for news in batch_lang_news:
                title = news['title']
                # 원본 배치에서의 인덱스 찾기
                original_idx = next(i for i, n in enumerate(news_batch) if n['id'] == news['id'])
                title_to_index[len(current_titles)] = original_idx

                lang_current_titles.append(title)
                current_titles.append(title)

            for news in existing_lang_news:
                lang_existing_titles.append(news['title'])
                existing_titles.append(news['title'])

            logger.debug(f"{lang} 그룹: 현재 {len(lang_current_titles)}개, 기존 {len(lang_existing_titles)}개")

        # 2. 교차 언어 중복 검사 (옵션)
        if check_cross_language and batch_groups['korean'] and batch_groups['english']:
            logger.info("교차 언어 중복 검사 수행")
            # 한국어 뉴스와 영어 뉴스 간 중복 검사 로직 추가 가능

        if not current_titles:
            return []

        all_titles = current_titles + existing_titles
        titles_text = "\n".join([f"{i + 1}. {title}" for i, title in enumerate(all_titles)])

        # 개선된 프롬프트 - 언어와 출처 정보 포함
        prompt = f"""다음 AI 뉴스 헤드라인들 중에서 실질적으로 같은 뉴스 이벤트를 다루는 중복을 찾아주세요:

    {titles_text}

    중복 판단 기준 (모든 조건을 만족해야 함):
    1. 동일한 회사/제품/서비스의 같은 사건/발표를 다루는 경우
    2. 같은 시점의 같은 이벤트를 보도하는 경우  
    3. 단순 키워드 유사성이 아닌 실제 뉴스 내용이 동일한 경우

    주의사항:
    - 같은 회사의 다른 제품/서비스는 중복이 아님
    - 시간차를 두고 발생한 다른 이벤트는 중복이 아님
    - 업데이트/후속 보도는 원본과 다른 뉴스임
    - 1번부터 {len(current_titles)}번까지는 오늘 뉴스
    - {len(current_titles) + 1}번부터 {len(all_titles)}번까지는 기존 뉴스

    duplicates 배열에는 오늘 뉴스(1-{len(current_titles)}) 인덱스만 포함해주세요.

    JSON 형식으로만 응답:
    {{
      "duplicates": [
        {{
          "primary": 1,
          "duplicates": [3, 5],
          "reason": "같은 회사의 동일한 제품 출시 발표"
        }}
      ]
    }}"""

        messages = [{"role": "user", "content": prompt}]

        response = self._make_api_request(messages, temperature=0.1)
        if not response:
            logger.warning(f"중복 검사 API 호출 실패 (ID: {request_id})")
            return []

        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        parsed_result = self._parse_json_response(content, request_id)

        if not parsed_result or 'duplicates' not in parsed_result:
            logger.warning(f"중복 검사 결과 파싱 실패 (ID: {request_id})")
            return []

        # 결과 처리 및 상세 로깅
        duplicate_indices = []
        for dup_group in parsed_result['duplicates']:
            if not isinstance(dup_group, dict):
                continue

            reason = dup_group.get('reason', '이유 없음')
            primary = dup_group.get('primary')
            duplicates_list = dup_group.get('duplicates', [])

            logger.info(f"중복 그룹 발견 (ID: {request_id}): primary={primary}, duplicates={duplicates_list}")
            logger.info(f"중복 이유: {reason}")

            # primary가 오늘 뉴스 범위에 있으면서 기존 뉴스와 중복인 경우
            if isinstance(primary, int) and 1 <= primary <= len(current_titles):
                primary_title = current_titles[primary - 1]

                # 기존 뉴스와의 중복인지 확인
                has_existing_duplicate = any(
                    isinstance(idx, int) and idx > len(current_titles)
                    for idx in duplicates_list
                )

                if has_existing_duplicate:
                    original_idx = title_to_index.get(primary - 1)
                    if original_idx is not None:
                        duplicate_indices.append(original_idx)
                        logger.info(f"기존 뉴스와 중복: [{original_idx}] {primary_title[:50]}...")

            # duplicates 배열의 오늘 뉴스들 처리
            for idx in duplicates_list:
                if isinstance(idx, int) and 1 <= idx <= len(current_titles):
                    original_idx = title_to_index.get(idx - 1)
                    if original_idx is not None and original_idx not in duplicate_indices:
                        duplicate_indices.append(original_idx)
                        dup_title = current_titles[idx - 1]
                        logger.info(f"중복 뉴스: [{original_idx}] {dup_title[:50]}...")

        logger.info(f"중복 검사 완료 (ID: {request_id}): {len(duplicate_indices)}개 중복 발견")
        return sorted(duplicate_indices) if duplicate_indices is not None else []


class NewsProcessor:
    """전체 뉴스 처리 로직을 관리하는 메인 클래스"""

    def __init__(self, supabase: Client, classifier: NewsClassifier):
        self.supabase = supabase
        self.classifier = classifier
        self.batch_size = 20
        self.duplicate_threshold = 0.8
        self.kst = timezone(timedelta(hours=9))

    def _is_english_news(self, news_item: Dict) -> bool:
        """뉴스가 영어로 작성되었는지 확인"""
        # source_country_cd 필드로 영어 뉴스 판단
        if news_item.get('source_country_cd') == 'usa':
            return True
            
        # 이전 호환성을 위한 도메인 기반 체크 (점진적으로 제거 예정)
        english_domains = [
            'techcrunch.com', 'theverge.com', 'wired.com', 'engadget.com',
            'venturebeat.com', 'arstechnica.com', 'cnet.com', 'zdnet.com',
            'bloomberg.com', 'reuters.com', 'cnbc.com', 'wsj.com',
            'nytimes.com', 'washingtonpost.com', 'theguardian.com', 'bbc.com'
        ]
        
        if not news_item.get('url'):
            return False
            
        domain = urlparse(news_item['url']).netloc.lower()
        return any(eng_domain in domain for eng_domain in english_domains)

    def translate_title_to_korean(self, title: str) -> str:
        """뉴스 제목을 한국어로 번역"""
        if not title or not title.strip():
            return ""
            
        try:
            # Google Translator를 사용하여 번역
            translated = GoogleTranslator(source='en', target='ko').translate(title)
            logger.info(f"번역 완료: {title} -> {translated}")
            return translated
        except Exception as e:
            logger.error(f"제목 번역 실패: {e}")
            return ""

    def get_todays_news(self) -> List[Dict]:
        """오늘자 뉴스 데이터 수집"""
        # 현재 KST 시간 기준으로 어제 날짜 계산
        today_kst = datetime.now(self.kst)
        # 오늘 날짜의 시작 (KST 00:00:00)
        today_start_kst = today_kst.replace(hour=0, minute=0, second=0, microsecond=0)

        # 오늘 날짜의 끝 (KST 23:59:59.999999)
        today_end_kst = today_kst.replace(hour=23, minute=59, second=59, microsecond=999999)

        # KST -> UTC 변환 (9시간 전으로 조정)
        today_start_utc = today_start_kst - timedelta(hours=9)
        today_end_utc = today_end_kst - timedelta(hours=9)

        logger.info(f"오늘 뉴스 조회 (KST): {today_start_kst} ~ {today_end_kst}")
        logger.info(f"오늘 뉴스 조회 (UTC): {today_start_utc} ~ {today_end_utc}")

        try:
            # 오늘자 뉴스 조회
            response = self.supabase.table('ai_news') \
                .select('*') \
                .gte('pub_date', today_start_utc.isoformat()) \
                .lte('pub_date', today_end_utc.isoformat()) \
                .execute()

            news_list = response.data if hasattr(response, 'data') else []
            logger.info(f"조회된 뉴스 개수: {len(news_list)}")

            # 제목 정리
            for news in news_list:
                if news.get('title'):
                    news['title'] = re.sub(r'[\"'']', '"', news['title'])
                    news['title'] = news['title'].strip()

            # 영어 뉴스 번역 및 news_list 업데이트 후 반환
            updated_news_list = self.translate_and_update_news(news_list)
            
            return updated_news_list if updated_news_list is not None else news_list

        except Exception as e:
            logger.error(f"뉴스 조회 중 오류 발생: {e}")
            return []

    def translate_and_update_news(self, news_list: List[Dict] = None) -> List[Dict]:
        """
        영어 뉴스 제목을 한국어로 번역하여 DB와 news_list에 업데이트
        
        Args:
            news_list: 업데이트할 뉴스 리스트. 제공된 경우 번역된 제목으로 업데이트됨
            
        Returns:
            업데이트된 news_list 또는 None (오류 발생 시)
        """
        logger.info("영어 뉴스 번역 시작")
        
        if news_list is None:
            return news_list
            
        # 오늘 날짜 범위 계산 (KST 기준)
        today = datetime.now(self.kst)
        today_start_kst = today.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end_kst = today.replace(hour=23, minute=59, second=59, microsecond=999999)

        try:
            # 1. news_list에서 미국 뉴스이면서 title_kr이 없는 항목 찾기
            news_to_translate = [
                news for news in news_list 
                if (news.get('source_country_cd') == 'usa' or self._is_english_news(news)) 
                and not news.get('title_kr') 
                and news.get('title')
            ]
            
            logger.info(f"번역이 필요한 미국 뉴스 {len(news_to_translate)}개 발견")
            
            # 2. 번역 및 업데이트
            updated_count = 0
            for news in news_to_translate:
                try:
                    # 한국어로 번역
                    translated = self.translate_title_to_korean(news['title'])
                    if not translated:
                        continue
                        
                    # DB 업데이트
                    update_result = self.supabase.table('ai_news') \
                        .update({'title_kr': translated}) \
                        .eq('id', news['id']) \
                        .execute()
                    
                    # news_list 업데이트
                    news['title_kr'] = translated
                    updated_count += 1
                    
                    # 로깅 간소화 (너무 많은 로그 방지)
                    if updated_count % 10 == 0 or updated_count == len(news_to_translate):
                        logger.info(f"{updated_count}/{len(news_to_translate)}개 번역 완료")
                        
                except Exception as e:
                    logger.error(f"뉴스 번역/업데이트 실패 (ID: {news.get('id')}): {e}")
            
            logger.info(f"총 {updated_count}개의 뉴스 제목이 번역되어 업데이트되었습니다.")
            
            # 업데이트된 news_list 반환
            return news_list
            
        except Exception as e:
            logger.error(f"뉴스 번역 중 오류 발생: {e}")
            return news_list  # 오류 발생 시 기존 리스트 반환

    def _get_news_language(self, news_item: Dict) -> str:
        """뉴스 언어 판별 (한국어/영어)"""
        # 1. source_country_cd로 먼저 판단
        if news_item.get('source_country_cd') == 'usa':
            return 'english'
        elif news_item.get('source_country_cd') == 'korea':
            return 'korean'

        # 2. 도메인 기반 판단 (fallback)
        english_domains = [
            'techcrunch.com', 'theverge.com', 'wired.com', 'engadget.com',
            'venturebeat.com', 'arstechnica.com', 'cnet.com', 'zdnet.com',
            'bloomberg.com', 'reuters.com', 'cnbc.com', 'wsj.com',
            'nytimes.com', 'washingtonpost.com', 'theguardian.com', 'bbc.com'
        ]

        if news_item.get('url'):
            domain = urlparse(news_item['url']).netloc.lower()
            if any(eng_domain in domain for eng_domain in english_domains):
                return 'english'

        # 3. 제목의 한글 문자 비율로 판단
        title = news_item.get('title', '')
        korean_chars = len([c for c in title if '\uac00' <= c <= '\ud7af'])
        total_chars = len([c for c in title if c.isalpha()])

        if total_chars > 0 and korean_chars / total_chars > 0.3:
            return 'korean'

        return 'english'

    def _get_duplicate_check_groups(self, news_list: List[Dict]) -> Dict[str, List[Dict]]:
        """뉴스를 언어와 출처별로 그룹화"""
        groups = {
            'korean': [],
            'english': [],
            'same_source': {}  # source별 그룹
        }

        for news in news_list:
            language = self._get_news_language(news)
            groups[language].append(news)

            # 동일 출처 그룹화
            source = news.get('source', 'unknown')
            if source not in groups['same_source']:
                groups['same_source'][source] = []
            groups['same_source'][source].append(news)

        logger.info(f"그룹별 뉴스 수: 한국어 {len(groups['korean'])}개, 영어 {len(groups['english'])}개")
        logger.info(f"출처별 그룹: {len(groups['same_source'])}개 출처")

        return groups


    def remove_simple_duplicates(self, news_list: List[Dict]) -> List[Dict]:
        """1단계: 빠른 유사도 검사로 중복 제거"""
        if not news_list:
            return []

        logger.info("1단계 중복 검사 시작 (텍스트 유사도)")

        unique_news = []
        seen_titles = []

        for news in news_list:
            title = news.get('title', '').strip().lower()
            if not title:
                continue

            is_duplicate = False

            for seen_title in seen_titles:
                # 포함 관계 검사
                if title in seen_title or seen_title in title:
                    is_duplicate = True
                    break

                # SequenceMatcher 유사도 검사
                similarity = SequenceMatcher(None, title, seen_title).ratio()
                if similarity >= 0.9:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_news.append(news)
                seen_titles.append(title)

        removed_count = len(news_list) - len(unique_news)
        logger.info(f"1단계 중복 제거 완료: {removed_count}개 제거, {len(unique_news)}개 남음")

        return unique_news

    def remove_same_source_duplicates(self, news_list: List[Dict]) -> List[Dict]:
        """동일 출처 기사의 고급 중복 검사"""
        if len(news_list) <= 1:
            return news_list

        logger.info("동일 출처 중복 검사 시작")

        # 출처별 그룹화
        source_groups = {}
        for i, news in enumerate(news_list):
            source = news.get('source', 'unknown')
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append((i, news))

        duplicate_indices = set()

        # 각 출처별로 중복 검사
        for source, news_group in source_groups.items():
            if len(news_group) <= 1:
                continue

            logger.info(f"출처 '{source}': {len(news_group)}개 기사 중복 검사")

            # 동일 출처 내에서 시간과 제목 유사도 기반 중복 검사
            for i in range(len(news_group)):
                for j in range(i + 1, len(news_group)):
                    idx1, news1 = news_group[i]
                    idx2, news2 = news_group[j]

                    # 이미 중복으로 표시된 뉴스는 건너뛰기
                    if idx1 in duplicate_indices or idx2 in duplicate_indices:
                        continue

                    # 제목 유사도 검사
                    title1 = news1['title'].strip().lower()
                    title2 = news2['title'].strip().lower()
                    similarity = SequenceMatcher(None, title1, title2).ratio()

                    # 게시 시간 차이 검사 (1시간 이내)
                    time_diff = self._calculate_time_difference(news1, news2)

                    # 중복 조건: 높은 유사도 + 짧은 시간차
                    if similarity >= 0.85 and time_diff <= 3600:  # 1시간
                        # 더 최근 뉴스를 유지하고 이전 뉴스를 중복으로 표시
                        if news1.get('pub_date', '') < news2.get('pub_date', ''):
                            duplicate_indices.add(idx1)
                            logger.info(f"동일 출처 중복 제거: {news1['title'][:50]}...")
                        else:
                            duplicate_indices.add(idx2)
                            logger.info(f"동일 출처 중복 제거: {news2['title'][:50]}...")

        # 중복이 아닌 뉴스만 반환
        unique_news = [news for i, news in enumerate(news_list) if i not in duplicate_indices]
        logger.info(f"동일 출처 중복 검사 완료: {len(duplicate_indices)}개 제거, {len(unique_news)}개 남음")

        return unique_news

    def _calculate_time_difference(self, news1: Dict, news2: Dict) -> int:
        """두 뉴스 간의 시간 차이를 초 단위로 계산"""
        try:
            time1 = datetime.fromisoformat(news1.get('pub_date', '').replace('Z', '+00:00'))
            time2 = datetime.fromisoformat(news2.get('pub_date', '').replace('Z', '+00:00'))
            return abs((time1 - time2).total_seconds())
        except:
            return float('inf')  # 시간 정보가 없으면 무한대 차이

    def remove_llm_duplicates(self, news_list: List[Dict]) -> List[Dict]:
        """개선된 2단계: LLM 기반 정교한 중복 검사"""
        if len(news_list) <= 1:
            return news_list

        logger.info("2단계 중복 검사 시작 (개선된 LLM 기반)")

        # 1. 동일 출처 중복 먼저 제거
        news_list = self.remove_same_source_duplicates(news_list)

        if len(news_list) <= 1:
            return news_list

        # 2. 언어별 그룹 확인
        groups = self._get_duplicate_check_groups(news_list)
        korean_count = len(groups['korean'])
        english_count = len(groups['english'])

        logger.info(f"언어별 분포: 한국어 {korean_count}개, 영어 {english_count}개")

        # 3. 기존 뉴스 조회 (언어별 분리)
        today = datetime.now(self.kst)
        today_start_utc = today.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)

        existing_news = []
        try:
            existing_response = self.supabase.table('ai_news') \
                .select('title, source, source_country_cd, url') \
                .gte('pub_date', today_start_utc.isoformat()) \
                .eq('is_processed', True) \
                .limit(150) \
                .execute()
            existing_news = existing_response.data
            logger.info(f"기존 뉴스 조회 완료: {len(existing_news)}개")
        except Exception as e:
            logger.error(f"기존 뉴스 조회 실패: {e}")
            existing_news = []

        duplicate_indices = set()

        # 4. 언어별 분리된 배치 처리
        batch_size = 15  # 배치 크기 축소

        try:
            # 한국어 뉴스 중복 검사
            if korean_count > 0:
                korean_news = groups['korean']
                korean_existing = [n for n in existing_news if self._get_news_language(n) == 'korean']

                for i in range(0, len(korean_news), batch_size):
                    batch = korean_news[i:i + batch_size]
                    logger.info(f"한국어 뉴스 배치 중복 검사: {i + 1}-{i + len(batch)}번째")

                    try:
                        batch_duplicates = self.classifier.check_duplicates_batch(
                            batch, korean_existing, check_cross_language=False
                        )

                        # 전체 리스트에서의 인덱스로 변환
                        for dup_idx in batch_duplicates:
                            if 0 <= dup_idx < len(batch):
                                original_news = batch[dup_idx]
                                global_idx = next(
                                    j for j, news in enumerate(news_list)
                                    if news['id'] == original_news['id']
                                )
                                duplicate_indices.add(global_idx)

                        time.sleep(1)  # API 호출 간격

                    except Exception as batch_error:
                        logger.error(f"한국어 배치 중복 검사 실패: {batch_error}")

            # 영어 뉴스 중복 검사
            if english_count > 0:
                english_news = groups['english']
                english_existing = [n for n in existing_news if self._get_news_language(n) == 'english']

                for i in range(0, len(english_news), batch_size):
                    batch = english_news[i:i + batch_size]
                    logger.info(f"영어 뉴스 배치 중복 검사: {i + 1}-{i + len(batch)}번째")

                    try:
                        batch_duplicates = self.classifier.check_duplicates_batch(
                            batch, english_existing, check_cross_language=False
                        )

                        # 전체 리스트에서의 인덱스로 변환
                        for dup_idx in batch_duplicates:
                            if 0 <= dup_idx < len(batch):
                                original_news = batch[dup_idx]
                                global_idx = next(
                                    j for j, news in enumerate(news_list)
                                    if news['id'] == original_news['id']
                                )
                                duplicate_indices.add(global_idx)

                        time.sleep(1)

                    except Exception as batch_error:
                        logger.error(f"영어 배치 중복 검사 실패: {batch_error}")

        except Exception as e:
            logger.error(f"중복 검사 과정에서 오류: {e}")

        # 5. 중복 플래그 업데이트 및 결과 반환
        if duplicate_indices:
            duplicate_ids = []
            logger.info("=== 중복으로 제거되는 뉴스 목록 ===")
            for idx in sorted(duplicate_indices):
                if 0 <= idx < len(news_list) and 'id' in news_list[idx]:
                    news_item = news_list[idx]
                    duplicate_ids.append(news_item['id'])
                    lang = self._get_news_language(news_item)
                    logger.info(f"[{idx}] [{lang}] ({news_item.get('source', 'N/A')}): {news_item['title'][:80]}...")

            if duplicate_ids:
                try:
                    self.supabase.table('ai_news') \
                        .update({'is_duplicate': True}) \
                        .in_('id', duplicate_ids) \
                        .execute()
                    logger.info(f"중복 플래그 업데이트 완료: {len(duplicate_ids)}개")
                except Exception as e:
                    logger.error(f"중복 플래그 업데이트 실패: {e}")

        # 중복이 아닌 뉴스만 반환
        unique_news = [news for i, news in enumerate(news_list) if i not in duplicate_indices]
        logger.info(f"개선된 2단계 중복 제거 완료: {len(duplicate_indices)}개 제거, {len(unique_news)}개 남음")

        return unique_news

    def classify_and_summarize_news(self, news_list: List[Dict]) -> Dict[str, Dict]:
        """뉴스 분류 및 요약"""
        if not news_list:
            return {}

        logger.info(f"뉴스 분류 시작: {len(news_list)}개")

        # 제목만 추출
        titles = [news['title'] for news in news_list]

        # 배치 단위로 분류
        all_classified = {}
        for category in self.classifier.categories.keys():
            all_classified[category] = []

        for i in range(0, len(titles), self.batch_size):
            batch_titles = titles[i:i + self.batch_size]
            batch_classified = self.classifier.classify_news_batch(batch_titles)

            for category, items in batch_classified.items():
                for item in items:
                    # 전체 인덱스로 변환
                    global_index = i + item['index']
                    if global_index < len(news_list):
                        news_item = news_list[global_index].copy()
                        news_item['title'] = item['title']
                        all_classified[category].append(news_item)

        # 각 카테고리별 요약 생성
        results = {}
        for category, news_items in all_classified.items():
            if news_items:
                # 카테고리당 최대 10개씩 청크 단위로 요약
                summary_chunks = []
                for i in range(0, len(news_items), 10):
                    chunk = news_items[i:i + 10]
                    chunk_summary = self.classifier.generate_summary(category, chunk)
                    if chunk_summary:
                        summary_chunks.append(chunk_summary)

                results[category] = {
                    'items': news_items,
                    'summary': ' '.join(summary_chunks) if summary_chunks else ''
                }

                logger.info(f"{self.classifier.categories[category]}: {len(news_items)}개")

        return results

    def save_to_newsletter_sections(self, classified_results: Dict[str, Dict]) -> None:
        """분류 결과를 데이터베이스에 저장"""
        if not classified_results:
            logger.info("저장할 분류 결과가 없습니다.")
            return

        logger.info("뉴스레터 섹션 저장 시작")

        # 오늘 날짜
        today_kst = datetime.now(self.kst).date()

        try:
            # 기존 데이터 삭제
            self.supabase.table('newsletter_sections') \
                .delete() \
                .eq('publish_date', today_kst.isoformat()) \
                .execute()

            # 새로운 섹션 데이터 생성
            sections_to_insert = []
            display_order = 1

            for category, data in classified_results.items():
                if data['items']:
                    # 뉴스 항목 정보 준비
                    news_items = []
                    for news in data['items']:
                        # Create a new dictionary with the existing fields
                        news_item = {
                            'id': news['id'],
                            'title': news['title'],
                            'source': news.get('source', ''),
                            'url': news.get('url', ''),
                            'pub_date': news.get('pub_date', '')
                        }
                        
                        # Add Korean title if it exists
                        if 'title_kr' in news and news['title_kr']:
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

            # 데이터베이스에 삽입
            if sections_to_insert:
                self.supabase.table('newsletter_sections') \
                    .insert(sections_to_insert) \
                    .execute()

                logger.info(f"뉴스레터 섹션 저장 완료: {len(sections_to_insert)}개 섹션")

                # 처리된 뉴스의 is_processed 플래그 업데이트
                all_news_ids = []
                for data in classified_results.values():
                    for news in data['items']:
                        all_news_ids.append(news['id'])

                if all_news_ids:
                    self.supabase.table('ai_news') \
                        .update({'is_processed': True}) \
                        .in_('id', all_news_ids) \
                        .execute()

                    logger.info(f"처리 완료 플래그 업데이트: {len(all_news_ids)}개")

        except Exception as e:
            logger.error(f"뉴스레터 섹션 저장 실패: {e}")
            raise

    def process_daily_news(self) -> None:
        """일일 뉴스 처리 메인 프로세스"""
        start_time = time.time()
        logger.info("=== AI 뉴스 처리 시스템 시작 ===")

        try:
            # 1. 오늘자 뉴스 수집
            news_list = self.get_todays_news()
            if not news_list:
                logger.info("처리할 뉴스가 없습니다.")
                return

            # 2. 중복 뉴스 제거 (2단계)
            news_list = self.remove_simple_duplicates(news_list)
            news_list = self.remove_llm_duplicates(news_list)

            if not news_list:
                logger.info("중복 제거 후 남은 뉴스가 없습니다.")
                return

            # 3. 뉴스 분류 및 요약
            classified_results = self.classify_and_summarize_news(news_list)

            # 4. 데이터베이스 저장
            self.save_to_newsletter_sections(classified_results)

            # 5. 처리 완료
            elapsed_time = time.time() - start_time
            total_processed = sum(len(data['items']) for data in classified_results.values())

            logger.info(f"=== 처리 완료 ===")
            logger.info(f"처리 시간: {elapsed_time:.2f}초")
            logger.info(f"처리된 뉴스: {total_processed}개")
            logger.info(f"생성된 섹션: {len(classified_results)}개")

        except Exception as e:
            logger.error(f"뉴스 처리 중 오류 발생: {e}")
            raise


def main():
    """메인 실행 함수"""
    try:
        # 1. 환경변수 로드
        load_dotenv()

        together_api_key = os.getenv('TOGETHER_API_KEY')
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')

        if not all([together_api_key, supabase_url, supabase_key]):
            raise ValueError("필수 환경변수가 설정되지 않았습니다.")

        # 2. Supabase 클라이언트 연결
        supabase: Client = create_client(supabase_url, supabase_key)
        logger.info("Supabase 연결 성공")

        # 3. NewsClassifier 인스턴스 생성
        classifier = NewsClassifier(together_api_key)
        logger.info("NewsClassifier 초기화 완료")

        # 4. NewsProcessor 인스턴스 생성
        processor = NewsProcessor(supabase, classifier)
        logger.info("NewsProcessor 초기화 완료")

        # 5. 뉴스 처리 실행
        processor.process_daily_news()

    except Exception as e:
        logger.error(f"시스템 실행 실패: {e}")
        raise


if __name__ == "__main__":
    main()