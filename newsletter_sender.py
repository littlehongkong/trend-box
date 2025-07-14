import os
import requests
import logging
from datetime import datetime, timedelta
from supabase import create_client
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import resend
import re
import time

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
        List[Dict[str, Any]]: 최근 1시간 이내의 뉴스 리스트 또는 빈 리스트
    """
    logger.info("Fetching recent news from Supabase (last 1 hour)...")
    
    try:
        # 현재 시간으로부터 1시간 전 시간 계산 (KST 기준)
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        
        # KST 시간대를 고려하여 포맷팅 (YYYY-MM-DD HH:MM:SS)
        time_format = '%Y-%m-%d %H:%M:%S'
        one_hour_ago_str = one_hour_ago.strftime(time_format)
        current_time_str = now.strftime(time_format)
        
        logger.info(f"Querying news published between {one_hour_ago_str} and {current_time_str}")
        
        # pub_date 필드를 기준으로 최근 1시간 이내 발행된 기사 조회
        response = supabase.table('ai_news') \
            .select('*') \
            .gte('pub_date', one_hour_ago_str) \
            .lte('pub_date', current_time_str) \
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
                "top_p": 0.9,
                "stop": ["\n\n"]  # 두 줄 연속 개행에서 중단
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

def format_newsletter() -> Optional[str]:
    """
    오늘 수집된 뉴스 기반으로 HTML 형식의 뉴스레터를 생성합니다.
    
    Returns:
        str: 생성된 HTML 뉴스레터 또는 None (뉴스가 없는 경우)
    """
    today = datetime.now().strftime('%Y년 %m월 %d일')
    
    # 1. 오늘의 전체 뉴스 가져오기
    news_items = fetch_todays_news()
    if not news_items:
        print("오늘의 뉴스가 없습니다.")
        return None
    
    # 2. 키워드(카테고리)별로 뉴스 분류
    news_by_keyword = {}
    for item in news_items:
        keyword = item.get('keyword', '기타')
        if keyword not in news_by_keyword:
            news_by_keyword[keyword] = []
        news_by_keyword[keyword].append(item)
    
    # 3. 전체 요약 생성 (Vietnam.vn 제외)
    all_news_texts = [
        f"{item.get('title', '')}. {item.get('description', '')}"
        for items in news_by_keyword.values() for item in items[:3]  # 각 카테고리별 상위 3개만 사용
        if item.get('source') != 'Vietnam.vn'  # Vietnam.vn 제외
    ]
    overall_summary = summarize_with_ai(all_news_texts, "전체 요약") if all_news_texts else None
    
    # 중복 제거를 위한 전역 세트
    all_used_points = set()
    
    # 4. HTML 컨텐츠 생성 시작
    html_content = f"""
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
        <h1 style="color: #1a365d;">🤖 AI/LLM 뉴스레터 - {today}</h1>
        <p style="color: #4a5568;">오늘의 AI/LLM 관련 최신 소식을 요약해드립니다.</p>
    """
    
    # 5. 전체 요약 섹션 추가 (개발자 관점의 주요 하이라이트)
    if overall_summary:
        html_content += """
        <div style="margin: 20px 0; padding: 20px; background-color: #f0f9ff; border-radius: 8px; border-left: 4px solid #3b82f6;">
            <h2 style="color: #1e40af; margin-top: 0;">📋 개발자 핵심 하이라이트</h2>
            <ul style="padding-left: 20px;">
        """
        
        # 개발자 관련 키워드 필터링
        dev_keywords = ['API', 'SDK', '모델', '출시', '업데이트', '버전', '기술', '보안', '성능', '개발', '프레임워크', '라이브러리', '오픈소스', 'GitHub']
        
        # 전체 요약 포인트 추가 (개발자 관련 필터링 적용)
        points_added = 0
        unique_points = set()
        
        for point in [p.strip() for p in overall_summary.split('\n') if p.strip()]:
            # 특수문자 정규화
            point = re.sub(r'[.]+', '.', point)
            point = re.sub(r'[!?]+', lambda x: x.group(0)[0], point)
            
            # 개발자 관련 포인트만 필터링 (중복 제거 및 최대 5개)
            if (any(keyword in point for keyword in dev_keywords) and 
                point not in unique_points and 
                len(point) > 10 and 
                points_added < 5):
                
                # 출처 정보 추가 (Vietnam.vn 제외)
                source_info = ""
                for items in news_by_keyword.values():
                    for item in items:
                        source = item.get('source', '')
                        if source == 'Vietnam.vn':
                            continue
                        if item.get('title', '') in point or any(word in point for word in item.get('title', '').split()):
                            if source:
                                source_info = f'<span style="color: #4b5563; font-size: 0.9em;">(출처: {source})</span>'
                            break
                    if source_info:
                        break
                
                # 중복 체크 및 포인트 추가
                point_key = point.lower().strip('.!?\n\r\t ')
                if point_key not in all_used_points:
                    html_content += f'<li style="margin-bottom: 8px; line-height: 1.5;">{point} {source_info}</li>'
                    all_used_points.add(point_key)
                    points_added += 1
        
        # 개발자 관련 포인트가 부족한 경우, 일반 포인트로 보완
        if points_added < 3:
            for point in [p.strip() for p in overall_summary.split('\n') if p.strip()]:
                point = re.sub(r'[.]+', '.', point)
                point = re.sub(r'[!?]+', lambda x: x.group(0)[0], point)
                
                if (point not in unique_points and 
                    len(point) > 10 and 
                    points_added < 5):
                    
                    html_content += f'<li style="margin-bottom: 8px; line-height: 1.5;">{point}</li>'
                    unique_points.add(point)
                    points_added += 1
        
        html_content += """
            </ul>
        </div>
        """
    
    # 6. 각 키워드별 섹션 생성
    for keyword, items in news_by_keyword.items():
        # 상위 5개 항목 추출
        top_items = items[:5]
        
        # 카테고리별 요약 생성
        category_texts = [
            f"{item.get('title', '')}. {item.get('description', '')}"
            for item in top_items
        ]
        category_summary = summarize_with_ai(category_texts, keyword) if category_texts else None
        
        # 섹션 헤더
        html_content += f"""
        <div style="margin: 30px 0; padding: 20px; background-color: #f7fafc; border-radius: 8px;">
            <h2 style="color: #2d3748; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;">
                🔍 {keyword} 관련 소식
            </h2>
        """
        
        # 카테고리 요약 내용 추가
        if category_summary:
            html_content += """
            <div style="margin: 15px 0; padding: 15px; background-color: white; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h3 style="color: #2b6cb0; margin-top: 0;">📌 주요 포인트</h3>
                <ul style="padding-left: 20px; margin: 10px 0 0 0;">
            """
            
            # 요약 포인트 추가 (최대 3개)
            points_added = 0
            for point in [p.strip() for p in category_summary.split('\n') if p.strip()]:
                # 테이블 형식이나 불필요한 문자 제거
                point = re.sub(r'\|.*$', '', point)  # 파이프 문자 이후 모두 제거
                point = re.sub(r'\s+', ' ', point)   # 연속된 공백을 하나로
                point = re.sub(r'^[0-9\s\-\*•]+', '', point)  # 번호나 글머리 기호 제거
                point = point.strip()
                
                # 특수문자 반복 제거
                point = re.sub(r'[.]+', '.', point)
                point = re.sub(r'[!?]+', lambda x: x.group(0)[0], point)
                
                # 유효한 문장인지 확인 (최소 길이 및 의미 있는 텍스트인지)
                if (len(point) > 10 and 
                    not any(c.isdigit() for c in point) and  # 숫자만 있는 라인 제외
                    not re.match(r'^[\s\W]+$', point) and  # 특수문자만 있는 라인 제외
                    points_added < 3):  # 3개로 제한
                    
                    html_content += f'<li style="margin-bottom: 8px; line-height: 1.5;">{point}</li>'
                    points_added += 1
            
            html_content += """
                </ul>
            </div>
            """
        
        # 상세 뉴스 항목 추가 (상위 3개만 기본 표시, 나머지는 접기)
        html_content += """
            <h3 style="color: #2b6cb0; margin: 20px 0 10px 0;">📰 상세 기사</h3>
            <ul id="main-articles-{keyword}" style="list-style: none; padding: 0; margin: 0 0 10px 0;">
        """.format(keyword=keyword.replace(' ', '-'))
        
        # 상위 3개 기사 표시
        for i, item in enumerate(top_items[:3]):
            if item.get('source') == 'Vietnam.vn':
                continue
                
            title = item.get('title', '제목 없음')
            url = item.get('url', '#')
            source = item.get('source', '출처 미상')
            description = item.get('description', '')
            
            pub_date = datetime.fromisoformat(item.get('pub_date', datetime.now().isoformat()))
            formatted_date = pub_date.strftime('%Y년 %m월 %d일 %H:%M')
            
            html_content += f"""
            <li style="margin-bottom: 16px; padding: 16px; background-color: white; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <a href="{url}" target="_blank" style="color: #2b6cb0; text-decoration: none; font-weight: 600; font-size: 1.1em; display: block; margin-bottom: 6px;">
                    {title}
                </a>
                <div style="color: #4a5568; margin: 6px 0; line-height: 1.5;">
                    {description}
                </div>
                <div style="color: #718096; font-size: 0.9em; display: flex; justify-content: space-between; align-items: center; margin-top: 8px;">
                    <div>
                        <span style="display: inline-block; margin-right: 12px;">📰 {source}</span>
                        <span style="color: #9ca3af;">⏰ {formatted_date}</span>
                    </div>
                    <a href="{url}" target="_blank" style="color: #4f46e5; text-decoration: none; font-weight: 500; white-space: nowrap;">기사 보기 →</a>
                </div>
            </li>
            """
        
        html_content += """
            </ul>
        """
        
        # 중복/유사 뉴스 섹션 (4개 이상인 경우에만 표시)
        if len(top_items) > 3:
            related_count = len(top_items) - 3
            html_content += """
            <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #e5e7eb;">
                <h3 style="color: #4b5563; font-size: 0.95em; margin: 0 0 12px 0; padding-bottom: 8px; border-bottom: 1px dashed #d1d5db;">
                    🔄 중복/유사 뉴스 ({0}건)
                </h3>
                <ul style="list-style: none; padding: 0; margin: 0;">
            """.format(related_count)
            
            for item in top_items[3:]:
                if item.get('source') == 'Vietnam.vn':
                    continue
                    
                title = item.get('title', '제목 없음')
                url = item.get('url', '#')
                source = item.get('source', '출처 미상')
                pub_date = datetime.fromisoformat(item.get('pub_date', datetime.now().isoformat()))
                formatted_date = pub_date.strftime('%Y년 %m월 %d일 %H:%M')
                
                # 유사도 점수가 있으면 표시 (없으면 생략)
                similarity = item.get('similarity')
                similarity_badge = f'<span style="display: inline-block; background-color: #e5e7eb; color: #4b5563; font-size: 0.75em; padding: 1px 6px; border-radius: 4px; margin-left: 6px;">유사도 {similarity}%</span>' if similarity else ''
                
                html_content += f"""
                <li style="margin-bottom: 10px; padding: 10px; background-color: #f8fafc; border-radius: 6px; border-left: 2px solid #d1d5db;">
                    <a href="{url}" target="_blank" style="color: #4b5563; text-decoration: none; font-weight: 500; display: block; margin-bottom: 4px; line-height: 1.4;">
                        {title}
                    </a>
                    <div style="color: #6b7280; font-size: 0.85em; margin-top: 4px;">
                        <span style="margin-right: 12px;">📰 {source}</span>
                        <span style="margin-right: 12px;">⏰ {formatted_date}</span>
                        {similarity_badge}
                    </div>
                </li>
                """
            
            html_content += """
                </ul>
            </div>
            """
        
        html_content += """
            </ul>
        </div>
        """
    
    # 푸터 추가
    html_content += f"""
    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 0.9em;">
        <p>이 뉴스레터는 자동으로 발송되었습니다. 수신 거부를 원하시면 회신 바랍니다.</p>
        <p style="margin-top: 10px; font-size: 0.9em;">
            발신: AI 뉴스레터 봇 | {today} 발행
        </p>
    </div>
    </div>
    
    <style>
    .related-articles-toggle {{
        color: #4f46e5;
        text-decoration: none;
        font-size: 0.9em;
        display: inline-block;
        margin-top: 5px;
        cursor: pointer;
    }}
    .related-articles-toggle:hover {{
        text-decoration: underline;
    }}
    </style>
    """
    
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
