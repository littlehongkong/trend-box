import os
import logging
from supabase import create_client
from dotenv import load_dotenv
import resend
import time
import difflib
from typing import List, Dict, Any, Optional
from collections import defaultdict
import requests
from datetime import datetime, timedelta

# Resend 예외 처리를 위한 임포트 제거

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('newsletter_sender.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('newsletter_sender')

# Load environment variables
load_dotenv()

# Initialize Resend
resend.api_key = os.getenv("RESEND_API_KEY")
if not resend.api_key:
    logger.error("RESEND_API_KEY not found in environment variables")
    raise ValueError("RESEND_API_KEY environment variable is required")

# Supabase 설정
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_todays_news() -> List[Dict[str, Any]]:
    """
    최근 1시간 이내에 생성된 뉴스를 Supabase에서 가져옵니다.
    
    Returns:
        List[Dict[str, Any]]: 최근 3시간 이내의 뉴스 리스트 또는 빈 리스트
    """
    logger.info("Fetching recent news from Supabase (last 3 three)...")
    
    try:
        # 현재 시간으로부터 1시간 전 시간 계산 (KST 기준)
        now = datetime.utcnow()
        three_hour_ago = now - timedelta(hours=3)

        # KST 시간대를 고려하여 포맷팅 (YYYY-MM-DD HH:MM:SS)
        time_format = '%Y-%m-%d %H:%M:%S'
        three_hour_ago_str = three_hour_ago.strftime(time_format)
        current_time_str = now.strftime(time_format)
        
        logger.info(f"Querying news published between {three_hour_ago_str} and {current_time_str}")
        
        # pub_date 필드를 기준으로 최근 3시간 이내 발행된 기사 조회
        response = supabase.table('ai_news') \
            .select('*') \
            .gte('created_at', three_hour_ago_str) \
            .lte('created_at', current_time_str) \
            .order('pub_date', desc=True) \
            .execute()
            
        logger.info(f"Found {len(response.data) if hasattr(response, 'data') else 0} new articles")
        
        news_count = len(response.data) if hasattr(response, 'data') else 0
        logger.info(f"Fetched {news_count} news items for today")
        
        return response.data if hasattr(response, 'data') else []
        
    except Exception as e:
        logger.error(f"Error fetching today's news: {str(e)}", exc_info=True)
        return []

def summarize_with_ai(texts: List[str], category: str, max_retries: int = 2) -> Optional[str]:
    """
    Together.ai API를 사용하여 텍스트를 요약합니다.
    
    Args:
        texts: 요약할 텍스트 리스트
        category: 뉴스 카테고리
        max_retries: 최대 재시도 횟수
        
    Returns:
        str: 요약된 텍스트 또는 None (실패 시)
    """
    logger.info(f"Starting summarization for category: {category}")
    
    if not texts:
        logger.warning("No text provided for summarization")
        return None
        
    # API 키 로드
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        logger.error("TOGETHER_API_KEY not found in environment variables")
        return None
    
    # 모든 텍스트를 하나로 결합 (너무 길지 않게 조절)
    combined_text = "\n".join(texts[:10])
    if len(combined_text) > 4000:
        logger.debug(f"Truncating text from {len(combined_text)} to 4000 characters")
        combined_text = combined_text[:4000]
    
    # 프롬프트 생성
    prompt = f"""'{category}' 카테고리의 뉴스 기사들입니다. 
이 뉴스들의 주요 내용을 3-5개의 핵심 포인트로 요약해주세요.
각 포인트는 간결한 문장으로 작성해주세요.

{combined_text}

요약:
1. """
    
    logger.debug(f"Generated prompt for category: {category}")
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries + 1} - Sending request to Together.ai")
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "meta-llama/Llama-3-8b-chat-hf",
                "prompt": prompt,
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9
            }
            
            # API 요청 시간 측정
            start_time = time.time()
            
            response = requests.post(
                "https://api.together.xyz/v1/completions",
                headers=headers,
                json=payload,
                timeout=60  # 60초 타임아웃
            )
            
            elapsed = time.time() - start_time
            logger.debug(f"API response received in {elapsed:.2f} seconds")
            
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"API response: {result}")
                
                if 'choices' in result and len(result['choices']) > 0:
                    generated_text = result['choices'][0]['text'].strip()
                    logger.debug(f"Generated text length: {len(generated_text)} characters")
                    
                    # 유효성 검사: [INST] 태그가 있으면 재시도
                    if '[INST]' in generated_text or '[/INST]' in generated_text:
                        if attempt < max_retries:
                            logger.warning(f"Invalid response format detected, retrying... ({attempt + 1}/{max_retries})")
                            time.sleep(1)  # 잠시 대기 후 재시도
                            continue
                        else:
                            logger.warning("Max retries reached with invalid format")
                    
                    # 불필요한 태그 제거
                    generated_text = generated_text.replace('[/INST]', '').replace('[INST]', '').strip()
                    
                    # 숫자로 시작하는 포인트만 필터링
                    points = []
                    for line in generated_text.split('\n'):
                        line = line.strip()
                        if line and (line[0].isdigit() or line.startswith('-')):
                            # 라인에서 숫자/불릿 이후의 텍스트만 추출
                            point = line[line.find('.') + 1:].strip() if '.' in line else line[1:].strip()
                            if point and len(point) > 3:  # 최소 길이 검사
                                points.append(point)
                    
                    logger.info(f"Extracted {len(points)} valid points from response")
                    
                    if points:  # 유효한 포인트가 있으면 반환
                        summary = '\n'.join(points[:5])
                        logger.info(f"Successfully generated summary with {len(points)} points")
                        return summary
                    
                    # 포인트가 없으면 재시도
                    if attempt < max_retries:
                        logger.warning(f"No valid points found, retrying... ({attempt + 1}/{max_retries})")
                        time.sleep(1)  # 잠시 대기 후 재시도
                        continue
                        
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                if response.status_code >= 500:  # 서버 에러인 경우에만 재시도
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)  # 지수 백오프
                        continue
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during summarization (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)  # 지수 백오프
                continue
        except Exception as e:
            logger.error(f"Unexpected error during summarization (attempt {attempt + 1}): {str(e)}", exc_info=True)
            if attempt < max_retries:
                time.sleep(1)
                continue
    
    logger.warning(f"Failed to generate summary for category: {category} after {max_retries + 1} attempts")
    return None

def normalize_text(text: str) -> str:
    """
    텍스트를 정규화합니다.
    - 특수문자 제거
    - 공백 정규화
    - 소문자 변환
    
    Args:
        text: 정규화할 텍스트
        
    Returns:
        str: 정규화된 텍스트
    """
    import re
    # 특수문자 제거
    text = re.sub(r'[^\w\s]', ' ', text)
    # 공백 정규화
    text = ' '.join(text.split())
    # 소문자 변환
    return text.lower()

def remove_source_suffix(text: str) -> str:
    """
    뉴스 제목에서 출처 정보를 제거합니다.
    예: '제목 - 출처' -> '제목'
    
    Args:
        text: 처리할 텍스트
        
    Returns:
        str: 출처가 제거된 텍스트
    """
    # ' - ' 또는 ' | ' 또는 ' : '로 분리하여 첫 번째 부분만 취함
    for sep in [' - ', ' | ', ' : ']:
        if sep in text:
            text = text.split(sep)[0].strip()
    return text

def calculate_similarity(str1: str, str2: str) -> float:
    """
    두 문자열 간의 유사도를 0~1 사이의 값으로 반환합니다.
    
    Args:
        str1: 첫 번째 문자열
        str2: 두 번째 문자열
        
    Returns:
        float: 0~1 사이의 유사도 점수 (1에 가까울수록 유사함)
    """
    # 원본 유사도
    original_similarity = difflib.SequenceMatcher(None, str1, str2).ratio()
    
    # 출처 제거 후 유사도
    clean_str1 = remove_source_suffix(str1)
    clean_str2 = remove_source_suffix(str2)
    clean_similarity = difflib.SequenceMatcher(None, clean_str1, clean_str2).ratio()
    
    # 정규화된 텍스트 유사도
    norm_str1 = normalize_text(clean_str1)
    norm_str2 = normalize_text(clean_str2)
    norm_similarity = difflib.SequenceMatcher(None, norm_str1, norm_str2).ratio()
    
    # 가장 높은 유사도 반환
    return max(original_similarity, clean_similarity, norm_similarity)

def analyze_tech_trends(news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    뉴스 아이템을 분석하여 기술 트렌드를 추출합니다.
    
    Args:
        news_items: 분석할 뉴스 아이템 리스트
        
    Returns:
        List[Dict[str, Any]]: 추출된 기술 트렌드 리스트
    """
    # 기술 키워드 카테고리 정의
    tech_categories = {
        'AI/ML': ['AI', '머신러닝', '딥러닝', '인공지능', 'LLM', 'GPT', '모델', '학습', '추론', '생성형 AI'],
        '클라우드': ['클라우드', 'AWS', 'GCP', 'Azure', '서버리스', '도커', '쿠버네티스', '인프라', 'DevOps'],
        '웹/모바일': ['웹', '모바일', '앱', '프론트엔드', '백엔드', 'API', '프레임워크', 'React', 'Next.js', 'Vue', 'Flutter'],
        '데이터': ['데이터', '분석', '빅데이터', '데이터베이스', 'SQL', 'NoSQL', '벡터DB', 'RAG', 'ETL']
    }
    
    # 카테고리별 점수 초기화
    category_scores = {category: defaultdict(int) for category in tech_categories}
    
    # 뉴스 아이템 분석
    for item in news_items:
        title = item.get('title', '')
        description = item.get('description', '')
        content = f"{title} {description}"
        
        # 각 카테고리별로 점수 계산
        for category, keywords in tech_categories.items():
            score = sum(content.count(keyword) for keyword in keywords)
            if score > 0:
                # 출처 수집을 위해 아이템 ID 저장
                if 'items' not in category_scores[category]:
                    category_scores[category]['items'] = []
                category_scores[category]['items'].append(item)
                category_scores[category]['score'] += score
    
    # 상위 트렌드 추출
    trends = []
    for category, data in category_scores.items():
        if 'items' in data and data['items']:
            # 대표 아이템 선택 (가장 최근 아이템)
            latest_item = max(data['items'], key=lambda x: x.get('pub_date', ''))
            
            # 간단한 요약 생성
            title = latest_item.get('title', '')
            source = latest_item.get('source', '출처 없음')
            
            # 중복 제거를 위한 키 생성
            trend_key = f"{category}_{title[:30]}"
            
            trends.append({
                'category': category,
                'title': title,
                'description': latest_item.get('description', ''),
                'source': source,
                'score': data['score'],
                'key': trend_key,
                'url': latest_item.get('url', '#')
            })
    
    # 점수 기준 정렬 및 중복 제거
    seen = set()
    unique_trends = []
    for trend in sorted(trends, key=lambda x: x['score'], reverse=True):
        if trend['key'] not in seen:
            seen.add(trend['key'])
            unique_trends.append(trend)
    
    return unique_trends[:4]  # 상위 4개만 반환

def remove_duplicate_news(news_items: List[Dict[str, Any]], similarity_threshold: float = 0.75) -> List[Dict[str, Any]]:
    """
    제목의 유사도가 높은 중복 뉴스를 제거합니다.
    
    Args:
        news_items: 뉴스 아이템 리스트
        similarity_threshold: 유사도 임계값 (0~1), 이 값보다 높으면 중복으로 판단
        
    Returns:
        List[Dict[str, Any]]: 중복이 제거된 뉴스 아이템 리스트
    """
    if not news_items:
        return []
        
    # 출처 정보가 더 긴 뉴스를 우선적으로 유지하기 위해 정렬
    sorted_news = sorted(news_items, 
                        key=lambda x: len(x.get('title', '')), 
                        reverse=True)
    
    unique_news = []
    seen_titles = []
    removed_count = 0
    
    for item in sorted_news:
        title = item.get('title', '').strip()
        if not title:
            continue
            
        is_duplicate = False
        
        # 출처 제거된 제목으로 비교
        clean_title = remove_source_suffix(title)
        
        # 이미 본 제목들과 유사도 비교
        for seen_item in unique_news:
            seen_title = seen_item.get('title', '').strip()
            seen_clean = remove_source_suffix(seen_title)
            
            # 출처가 다른 경우에만 유사도 체크
            if clean_title != seen_clean:
                similarity = calculate_similarity(title, seen_title)
                if similarity > similarity_threshold:
                    is_duplicate = True
                    removed_count += 1
                    logger.info(f"Removing duplicate article (similarity: {similarity:.2f}):\n  - Original: {seen_title}\n  - Duplicate: {title}")
                    break
        
        if not is_duplicate:
            unique_news.append(item)
    
    logger.info(f"Removed {removed_count} duplicate articles (kept {len(unique_news)} unique articles)")
    return unique_news

def format_newsletter() -> Optional[str]:
    """
    Supabase의 newsletter_sections 테이블에서 오늘 날짜의 섹션들을 가져와 뉴스레터를 생성합니다.
    
    Returns:
        str: 생성된 HTML 뉴스레터 또는 None
    """
    # KST 기준 오늘 날짜 (YYYY-MM-DD)
    today_kst = datetime.now().strftime('%Y-%m-%d')
    logger.info(f"Fetching newsletter sections for date: {today_kst}")
    
    try:
        # Supabase에서 오늘 날짜의 섹션 조회 (is_published=True인 항목만)
        response = supabase.table('newsletter_sections') \
            .select('*') \
            .eq('publish_date', today_kst) \
            .eq('is_published', True) \
            .order('display_order') \
            .execute()
        
        sections = response.data if hasattr(response, 'data') else []
        
        if not sections:
            logger.info(f"No published sections found for date: {today_kst}")
            return None
            
        logger.info(f"Found {len(sections)} sections for today's newsletter")
        
        # HTML 생성
        html_content = f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
            <h1 style="color: #1a365d;">📰 AI 뉴스레터 - {today_kst}</h1>
            <p style="color: #4a5568;">AI 분야의 최신 소식과 인사이트를 전해드립니다.</p>
        """
        
        # 섹션별 렌더링
        for section in sections:
            section_name = section.get('section_name', '')
            section_title = section.get('section_title', '')
            content = section.get('content', [])
            summary = section.get('summary', '')
            
            if not content or not isinstance(content, list):
                logger.warning(f"Skipping section {section_name} - invalid content format")
                continue
                
            # 섹션 헤더
            html_content += f"""
            <div style="margin-top: 30px; padding: 20px; background-color: #f9fafb; border-radius: 8px;">
                <h2 style="color: #2d3748; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;">
                    {section_title}
                </h2>
            """
            
            # 요약이 있는 경우 표시
            if summary:
                html_content += f"""
                <div style="background: #edf2f7; padding: 12px 15px; border-radius: 6px; margin-bottom: 15px; font-size: 0.95em; line-height: 1.5; color: #2d3748;">
                    {summary}
                </div>
                """
            
            # 콘텐츠 항목들
            html_content += """
                <div style="margin-top: 15px;">
                    <ul style="list-style: none; padding: 0; margin: 0;">
            """
            
            # 각 콘텐츠 항목 렌더링
            for item in content:
                if not isinstance(item, dict):
                    continue
                    
                title = item.get('title', '')
                url = item.get('url', '#')
                description = item.get('description', '')
                source = item.get('source', '')
                
                html_content += f"""
                <li style="margin-bottom: 20px; background: white; padding: 18px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.05);">
                    <a href="{url}" target="_blank" style="color: #2563eb; font-weight: 600; font-size: 1.05em; text-decoration: none; line-height: 1.4; display: block; margin-bottom: 8px;">
                        {title}
                    </a>
                """
                
                if description:
                    html_content += f"""
                    <p style="color: #4b5563; margin: 8px 0 10px 0; line-height: 1.5; font-size: 0.95em;">
                        {description}
                    </p>
                    """
                
                if source:
                    html_content += f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                        <span style="color: #6b7280; font-size: 0.85em;">{source}</span>
                    </div>
                    """
                
                html_content += "</li>"
            
            # 섹션 닫기
            html_content += """
                    </ul>
                </div>
            </div>
            """
        
        # 푸터
        html_content += f"""
        <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 0.9em;">
            <p>이 뉴스레터는 {today_kst} 기준으로 작성되었습니다.</p>
            <p style="margin-top: 10px;">뉴스레터 구독을 원하지 않으시면 <a href="#" style="color: #4f46e5;">여기</a>를 클릭해 구독을 취소하실 수 있습니다.</p>
        </div>
        </div>
        """
        
        return html_content
        
    except Exception as e:
        logger.error(f"Error formatting newsletter: {str(e)}", exc_info=True)
        return None

    return html_content



def send_newsletter():
    """
    뉴스레터를 이메일로 발송합니다.
    
    Returns:
        bool: 이메일 발송 성공 여부
    """
    logger.info("Starting newsletter sending process...")
    
    # 수신자 이메일 주소 확인
    recipient_email = os.getenv("RECIPIENT_EMAIL", "tlstjscjswo@gmail.com")
    if not recipient_email:
        logger.error("No recipient email address found")
        return False
    
    logger.info(f"Preparing to send newsletter to: {recipient_email}")
    
    # 뉴스레터 HTML 생성
    logger.info("Generating newsletter HTML content...")
    html_content = format_newsletter()
    
    if not html_content:
        logger.error("No HTML content generated for the newsletter")
        return False
    
    logger.debug(f"Generated HTML content length: {len(html_content)} characters")
    
    # 이메일 제목 설정 (시간대 표시)
    current_time = datetime.now()
    email_subject = f"🤖 AI/LLM 실시간 업데이트 - {current_time.strftime('%m/%d %H:%M')} 기준"
    
    try:
        logger.info("Sending email via Resend API...")
        
        # 이메일 발송 요청
        response = resend.Emails.send({
            "from": "AI Newsletter <onboarding@resend.dev>",
            "to": [recipient_email],
            "subject": email_subject,
            "html": html_content
        })
        
        # 응답 로깅
        if isinstance(response, dict) and 'id' in response:
            logger.info(f"Email sent successfully! Email ID: {response['id']}")
            return True
        else:
            logger.error(f"Unexpected response from Resend API: {response}")
            return False
            
    except Exception as e:
        if hasattr(e, 'status_code'):
            logger.error(f"Resend API error (HTTP {e.status_code}): {str(e)}")
        else:
            logger.error(f"Resend API error: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error while sending email: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    try:
        logger.info("=" * 50)
        logger.info("Starting AI Newsletter Sender")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)
        
        # Send the newsletter
        success = send_newsletter()
        
        if success:
            logger.info("Newsletter sent successfully!")
        else:
            logger.error("Failed to send newsletter")
            
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
    except Exception as e:
        logger.critical(f"Critical error in main execution: {str(e)}", exc_info=True)
    finally:
        logger.info("=" * 50)
        logger.info(f"Process completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)
