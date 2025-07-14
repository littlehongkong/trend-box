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

    # 2. 카테고리(category) 기준으로 뉴스 분류 (수정)
    news_by_category = {}
    for item in news_items:
        category = item.get('category') or '기타'
        if category not in news_by_category:
            news_by_category[category] = []
        news_by_category[category].append(item)

    # 3. 전체 요약 생성 (Vietnam.vn 제외)
    all_news_texts = [
        f"{item.get('title', '')}. {item.get('description', '')}"
        for items in news_by_category.values() for item in items[:3]
        if item.get('source') != 'Vietnam.vn'
    ]
    overall_summary = summarize_with_ai(all_news_texts, "전체 요약") if all_news_texts else None

    all_used_points = set()

    # 4. HTML 컨텐츠 생성 시작
    html_content = f"""
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
        <h1 style="color: #1a365d;">🤖 AI/LLM 뉴스레터 - {today}</h1>
        <p style="color: #4a5568;">오늘의 AI/LLM 관련 최신 소식을 요약해드립니다.</p>
    """

    # 5. 전체 요약 섹션 추가 (기존과 동일)

    # 6. 카테고리별 섹션 생성 (수정)
    for category, items in news_by_category.items():
        top_items = items[:5]

        category_texts = [
            f"{item.get('title', '')}. {item.get('description', '')}"
            for item in top_items
        ]
        category_summary = summarize_with_ai(category_texts, category) if category_texts else None

        html_content += f"""
        <div style="margin: 30px 0; padding: 20px; background-color: #f7fafc; border-radius: 8px;">
            <h2 style="color: #2d3748; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;">
                🔍 {category} 관련 소식
            </h2>
        """

        if category_summary:
            html_content += """
            <div style="margin: 15px 0; padding: 15px; background-color: white; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <h3 style="color: #2b6cb0; margin-top: 0;">📌 주요 포인트</h3>
                <ul style="padding-left: 20px; margin: 10px 0 0 0;">
            """
            for point in category_summary.split('\n'):
                point = point.strip()
                if point:
                    html_content += f'<li style="margin-bottom: 8px; line-height: 1.5;">{point}</li>'
            html_content += """
                </ul>
            </div>
            """

        html_content += """
            <h3 style="color: #2b6cb0; margin: 20px 0 10px 0;">📰 상세 기사</h3>
            <ul style="list-style: none; padding: 0;">
        """

        for item in top_items:
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
                <div style="color: #718096; font-size: 0.9em; margin-top: 8px;">
                    📰 {source} | ⏰ {formatted_date}
                </div>
            </li>
            """

        html_content += """
            </ul>
        </div>
        """

    # 7. 푸터 추가
    html_content += f"""
    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 0.9em;">
        <p>이 뉴스레터는 자동으로 발송되었습니다. 수신 거부를 원하시면 회신 바랍니다.</p>
        <p style="margin-top: 10px; font-size: 0.9em;">
            발신: AI 뉴스레터 봇 | {today} 발행
        </p>
    </div>
    </div>
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
