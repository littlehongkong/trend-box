import os
import json
import re
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import uuid

import requests
from supabase import create_client, Client
from dotenv import load_dotenv

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

        # 카테고리별 맞춤 프롬프트
        category_prompts = {
            "new_services": "새로 출시된 AI 서비스나 모델의 기능, 개발자가 어떻게 활용하거나 새로운 제품/서비스로 확장할 수 있을지 중심으로",
            "updates": "기존 서비스의 업데이트나 정책 변경이 개발 워크플로우, 도구 선택, 협업 방식에 어떤 영향을 미칠지 중심으로",
            "investment": "투자나 인수합병이 AI 생태계, 개발자 도구 또는 커뮤니티 활동에 어떤 기회나 리스크를 제공하는지 중심으로",
            "infrastructure": "새로운 인프라/도구의 기술적 특징과 이를 도입할 때 개발자가 고려해야 할 실질적 요소 중심으로",
            "trends": "기술 트렌드가 개발자의 미래 역량, 학습 방향, 커리어 설계에 어떤 시사점을 주는지 중심으로",
            "other": "AI 산업 전반 또는 문화적 현상이 개발자에게 어떤 간접적 영향이나 통찰을 줄 수 있는지 중심으로"
        }

        focus_area = category_prompts.get(category, category_prompts["other"])

        prompt = f"""당신은 AI 개발 동향을 분석하는 전문 애널리스트입니다.
    다음은 [{category_name}] 카테고리에 포함된 AI 뉴스 헤드라인 목록입니다:

    {titles_text}

    요구사항:
    1. 반드시 한국어로 작성
    2. 2~3문장 이내로 간결하게 작성하되, 단순 요약이 아니라 **개발자 관점의 핵심 인사이트**를 도출할 것
    3. {focus_area}
    4. 개별 뉴스 제목 언급 없이, 공통된 흐름, 기술적 맥락, 시사점 중심으로 설명
    5. 추상적 표현보다는 실제 개발자 행동/결정에 도움이 되는 문장으로 작성

    요약 시작:"""

        messages = [{"role": "user", "content": prompt}]

        response = self._make_api_request(messages, temperature=0.1)
        if not response:
            return f"{category_name} 관련 {len(news_items)}건의 뉴스입니다."

        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        return content.strip()

    def check_duplicates_batch(self, news_batch: List[Dict], existing_news: List[Dict] = None) -> List[int]:
        """LLM 기반 중복 뉴스 검사 - 수정된 버전"""
        if len(news_batch) <= 1:
            return []

        current_titles = [news['title'] for news in news_batch]
        existing_titles = [news['title'] for news in (existing_news or [])]

        all_titles = current_titles + existing_titles
        titles_text = "\n".join([f"{i + 1}. {title}" for i, title in enumerate(all_titles)])

        logger.info(f"중복 검사 시작: 오늘 뉴스 {len(current_titles)}개, 기존 뉴스 {len(existing_titles)}개")

        prompt = f"""다음 AI 뉴스 헤드라인들 중에서 같은 뉴스 이벤트를 다루는 중복된 뉴스를 찾아주세요:

    {titles_text}

    기준:
    - 같은 회사의 같은 제품/서비스/사건을 다루는 경우
    - 단순히 키워드가 비슷한 것이 아닌, 실제로 같은 뉴스 이벤트인 경우
    - 1번부터 {len(current_titles)}번까지는 오늘 뉴스
    - {len(current_titles) + 1}번부터 {len(all_titles)}번까지는 기존 뉴스

    중요: duplicates 배열에는 오늘 뉴스(1-{len(current_titles)}) 인덱스만 포함해주세요.

    JSON 형식으로만 응답해주세요:
    {{
      "duplicates": [
        {{
          "primary": 1,
          "duplicates": [3, 5]
        }}
      ]
    }}

    중복이 없으면:
    {{
      "duplicates": []
    }}"""

        messages = [{"role": "user", "content": prompt}]

        response = self._make_api_request(messages, temperature=0.1)
        if not response:
            return []

        content = response.get('choices', [{}])[0].get('message', {}).get('content', '')
        request_id = str(uuid.uuid4())[:8]

        parsed_result = self._parse_json_response(content, request_id)
        if not parsed_result or 'duplicates' not in parsed_result:
            logger.warning(f"중복 검사 결과 파싱 실패 (ID: {request_id})")
            return []

        # 오늘 뉴스 중에서 중복된 것들의 인덱스 추출
        duplicate_indices = []

        for dup_group in parsed_result['duplicates']:
            if not isinstance(dup_group, dict):
                continue

            # primary 인덱스 처리
            primary = dup_group.get('primary')
            if isinstance(primary, int) and 1 <= primary <= len(current_titles):
                # primary가 오늘 뉴스 범위에 있으면 중복으로 표시하지 않음 (기준점이므로)
                pass

            # duplicates 배열 처리
            duplicates_list = dup_group.get('duplicates', [])
            if not isinstance(duplicates_list, list):
                continue

            for idx in duplicates_list:
                if isinstance(idx, int):
                    # 오늘 뉴스 범위 내의 인덱스만 처리
                    if 1 <= idx <= len(current_titles):
                        duplicate_indices.append(idx - 1)  # 0-based index로 변환
                        logger.debug(f"오늘 뉴스 중복 발견: {idx} -> {current_titles[idx - 1]}")
                    elif idx > len(current_titles):
                        # 기존 뉴스와의 중복인 경우, primary가 오늘 뉴스라면 해당 primary를 중복으로 표시
                        if isinstance(primary, int) and 1 <= primary <= len(current_titles):
                            if primary - 1 not in duplicate_indices:  # 중복 추가 방지
                                duplicate_indices.append(primary - 1)
                                logger.debug(f"기존 뉴스와 중복: {primary} -> {current_titles[primary - 1]}")

        # 중복 제거 및 정렬
        duplicate_indices = sorted(list(set(duplicate_indices)))

        logger.info(f"LLM 중복 검사 완료: {len(duplicate_indices)}개 중복 발견")

        # 디버깅용 로그
        for idx in duplicate_indices[:5]:  # 최대 5개만 로그
            if idx < len(current_titles):
                logger.debug(f"중복 뉴스: {current_titles[idx]}")

        return duplicate_indices


class NewsProcessor:
    """전체 뉴스 처리 로직을 관리하는 메인 클래스"""

    def __init__(self, supabase: Client, classifier: NewsClassifier):
        self.supabase = supabase
        self.classifier = classifier
        self.batch_size = 20
        self.duplicate_threshold = 0.8
        self.kst = timezone(timedelta(hours=9))

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

        logger.info(f"오늘자 뉴스 조회: {today_start_kst} ~ {today_end_kst} (KST)")
        logger.info(f"오늘자 뉴스 조회(utc): {today_start_utc} ~ {today_end_utc} (KST)")

        try:
            response = self.supabase.table('ai_news') \
                .select('*') \
                .gte('pub_date', today_start_utc.isoformat()) \
                .lte('pub_date', today_end_utc.isoformat()) \
                .execute()

            news_list = response.data
            logger.info(f"조회된 뉴스 개수: {len(news_list)}")

            # 제목의 특수문자 정리
            for news in news_list:
                if news.get('title'):
                    news['title'] = re.sub(r'[""''`]', '"', news['title'])
                    news['title'] = news['title'].strip()

            return news_list

        except Exception as e:
            logger.error(f"뉴스 조회 실패: {e}")
            return []

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

    def remove_llm_duplicates(self, news_list: List[Dict]) -> List[Dict]:
        """2단계: LLM 기반 정교한 중복 검사 - 개선된 버전"""
        if len(news_list) <= 1:
            return news_list

        logger.info("2단계 중복 검사 시작 (LLM 기반)")

        # 이전 24시간 뉴스 조회 (최대 100개로 제한)
        today = datetime.now(self.kst)
        today_start_utc = today.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)

        existing_news = []
        try:
            existing_response = self.supabase.table('ai_news') \
                .select('title') \
                .gte('pub_date', today_start_utc.isoformat()) \
                .eq('is_processed', True) \
                .execute()
            existing_news = existing_response.data
            logger.info(f"기존 뉴스 조회 완료: {len(existing_news)}개")
        except Exception as e:
            logger.error(f"기존 뉴스 조회 실패: {e}")
            existing_news = []

        duplicate_indices = set()

        # 배치 단위로 중복 검사 (크기 조정)
        batch_size = min(20, len(news_list))  # 최대 20개

        try:
            for i in range(0, len(news_list), batch_size):
                batch = news_list[i:i + batch_size]
                logger.info(f"배치 중복 검사: {i + 1}-{i + len(batch)}번째 뉴스")

                try:
                    batch_duplicates = self.classifier.check_duplicates_batch(batch, existing_news)

                    # 전체 인덱스로 변환 및 검증
                    for dup_idx in batch_duplicates:
                        global_idx = i + dup_idx
                        if 0 <= global_idx < len(news_list):
                            duplicate_indices.add(global_idx)
                        else:
                            logger.warning(f"잘못된 중복 인덱스: {global_idx} (전체 뉴스: {len(news_list)}개)")

                    logger.info(f"배치 중복 검사 완료: {len(batch_duplicates)}개 중복 발견")

                    # API 호출 간격 조정
                    time.sleep(1)

                except Exception as batch_error:
                    logger.error(f"배치 {i // batch_size + 1} 중복 검사 실패: {batch_error}")
                    continue

        except Exception as e:
            logger.error(f"전체 중복 검사 과정에서 오류: {e}")

        # 중복 플래그 업데이트
        if duplicate_indices:
            duplicate_ids = []
            for idx in duplicate_indices:
                if 0 <= idx < len(news_list) and 'id' in news_list[idx]:
                    duplicate_ids.append(news_list[idx]['id'])

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
        unique_news = []
        for i, news in enumerate(news_list):
            if i not in duplicate_indices:
                unique_news.append(news)

        logger.info(f"2단계 중복 제거 완료: {len(duplicate_indices)}개 제거, {len(unique_news)}개 남음")

        # 중복 제거 상세 로그 (최대 10개)
        removed_count = 0
        for idx in sorted(duplicate_indices)[:10]:
            if idx < len(news_list):
                logger.info(f"중복 제거: {news_list[idx].get('title', 'N/A')}")
                removed_count += 1

        if len(duplicate_indices) > 10:
            logger.info(f"... 외 {len(duplicate_indices) - 10}개 더")

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
                        news_items.append({
                            'id': news['id'],
                            'title': news['title'],
                            'source': news.get('source', ''),
                            'url': news.get('url', ''),
                            'pub_date': news.get('pub_date', '')
                        })

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