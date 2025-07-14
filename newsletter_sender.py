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
import difflib
import json
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, Counter
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

def calculate_similarity(str1: str, str2: str) -> float:
    """
    두 문자열 간의 유사도를 0~1 사이의 값으로 반환합니다.
    
    Args:
        str1: 첫 번째 문자열
        str2: 두 번째 문자열
        
    Returns:
        float: 0~1 사이의 유사도 점수 (1에 가까울수록 유사함)
    """
    return difflib.SequenceMatcher(None, str1, str2).ratio()

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
                'source': source,
                'score': data['score'],
                'key': trend_key
            })
    
    # 점수 기준 정렬 및 중복 제거
    seen = set()
    unique_trends = []
    for trend in sorted(trends, key=lambda x: x['score'], reverse=True):
        if trend['key'] not in seen:
            seen.add(trend['key'])
            unique_trends.append(trend)
    
    return unique_trends[:4]  # 상위 4개만 반환

def remove_duplicate_news(news_items: List[Dict[str, Any]], similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    제목의 유사도가 높은 중복 뉴스를 제거합니다.
    
    Args:
        news_items: 뉴스 아이템 리스트
        similarity_threshold: 유사도 임계값 (0~1), 이 값보다 높으면 중복으로 판단
        
    Returns:
        List[Dict[str, Any]]: 중복이 제거된 뉴스 아이템 리스트
    """
    unique_news = []
    seen_titles = []
    
    for item in news_items:
        title = item.get('title', '').strip()
        if not title:
            continue
            
        is_duplicate = False
        
        # 이미 본 제목들과 유사도 비교
        for seen_title in seen_titles:
            similarity = calculate_similarity(title, seen_title)
            if similarity > similarity_threshold:
                is_duplicate = True
                logger.info(f"Removing duplicate article (similarity: {similarity:.2f}): {title}")
                break
                
        if not is_duplicate:
            unique_news.append(item)
            seen_titles.append(title)
    
    logger.info(f"Removed {len(news_items) - len(unique_news)} duplicate articles")
    return unique_news

def format_newsletter() -> Optional[str]:
    """
    오늘 수집된 뉴스 기반으로 AI 툴/서비스 브리핑 형태의 HTML 뉴스레터를 생성합니다.

    Returns:
        str: 생성된 HTML 뉴스레터 또는 None (뉴스가 없는 경우)
    """
    today = datetime.now().strftime('%Y년 %m월 %d일 %H:%M')

    # 1. 최근 1시간 내 뉴스 수집
    news_items = fetch_todays_news()
    if not news_items:
        print("오늘의 AI 관련 신규 서비스/업데이트 소식이 없습니다.")
        return None
        
    # 2. 중복 뉴스 제거
    news_items = remove_duplicate_news(news_items)

    # 2. 신규 서비스/툴 중심으로 구성
    service_news = [item for item in news_items if '출시' in item.get('title', '') or '서비스' in item.get('title', '') or '공개' in item.get('title', '')]
    update_news = [item for item in news_items if item not in service_news]

    html_content = f"""
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px;">
        <h1 style="color: #1a365d;">🤖 AI 툴/서비스 실시간 브리핑 - {today}</h1>
        <p style="color: #4a5568;">방금 발표된 AI 관련 서비스/툴/업데이트를 개발자에게 전달합니다.</p>
    """

    # 3. 신규 서비스/툴 섹션
    html_content += """
    <div style="margin-top: 30px; padding: 20px; background-color: #f7fafc; border-radius: 8px;">
        <h2 style="color: #2d3748; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;">
            🚀 오늘 출시된 AI 서비스/툴
        </h2>
        <ul style="list-style: none; padding: 0;">
    """

    if service_news:
        for item in service_news:
            title = item.get('title', '제목 없음')
            url = item.get('url', '#')
            description = item.get('description', '')
            source = item.get('source', '출처 미상')
            pub_date = datetime.fromisoformat(item.get('pub_date', datetime.now().isoformat()))
            formatted_date = pub_date.strftime('%Y-%m-%d %H:%M')

            html_content += f"""
            <li style="margin-bottom: 20px; background: white; padding: 16px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <a href="{url}" target="_blank" style="color: #2b6cb0; font-weight: bold; font-size: 1.1em; text-decoration: none;">{title}</a>
                <p style="color: #4a5568; margin: 8px 0;">{description}</p>
                <p style="color: #718096; font-size: 0.9em;">{source} | {formatted_date}</p>
            </li>
            """
    else:
        html_content += "<p>오늘 출시된 신규 서비스/툴 정보가 없습니다.</p>"

    html_content += "</ul></div>"

    # 4. 업데이트/정책 변경 섹션
    html_content += """
    <div style="margin-top: 30px; padding: 20px; background-color: #edf2f7; border-radius: 8px;">
        <h2 style="color: #2d3748; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px;">
            🛠️ 주요 업데이트/정책 변경
        </h2>
        <ul style="list-style: none; padding: 0;">
    """

    if update_news:
        for item in update_news:
            title = item.get('title', '제목 없음')
            url = item.get('url', '#')
            description = item.get('description', '')
            source = item.get('source', '출처 미상')
            pub_date = datetime.fromisoformat(item.get('pub_date', datetime.now().isoformat()))
            formatted_date = pub_date.strftime('%Y-%m-%d %H:%M')

            html_content += f"""
            <li style="margin-bottom: 20px; background: white; padding: 16px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                <a href="{url}" target="_blank" style="color: #2b6cb0; font-weight: bold; font-size: 1.1em; text-decoration: none;">{title}</a>
                <p style="color: #4a5568; margin: 8px 0;">{description}</p>
                <p style="color: #718096; font-size: 0.9em;">{source} | {formatted_date}</p>
            </li>
            """
    else:
        html_content += "<p>오늘 확인된 업데이트/정책 변경이 없습니다.</p>"

    html_content += "</ul></div>"
    
    # 5. 현재 주목할 기술 트렌드 섹션 추가
    tech_trends = analyze_tech_trends(update_news + service_news)
    
    if tech_trends:
        # 카테고리별 색상 매핑
        category_colors = {
            'AI/ML': {'bg': '#ede9fe', 'text': '#5b21b6'},
            '클라우드': {'bg': '#e0f2fe', 'text': '#0369a1'},
            '웹/모바일': {'bg': '#fef3c7', 'text': '#92400e'},
            '데이터': {'bg': '#dcfce7', 'text': '#166534'}
        }
        
        html_content += """
        <div style="margin-top: 40px; padding: 25px; background-color: #f5f3ff; border-radius: 8px;">
            <h2 style="color: #5b21b6; border-bottom: 2px solid #c4b5fd; padding-bottom: 10px; margin-top: 0;">
                📈 현재 주목할 기술 트렌드
            </h2>
            <div style="margin-top: 20px;">
                <h3 style="color: #5b21b6; margin-bottom: 15px;">개발자 필독! 핵심 기술 동향</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px;">
        """
        
        for trend in tech_trends:
            category = trend['category']
            colors = category_colors.get(category, {'bg': '#f3f4f6', 'text': '#4b5563'})
            
            html_content += f"""
                <div style="background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <div style="display: flex; align-items: center; margin-bottom: 12px;">
                        <span style="background-color: {colors['bg']}; color: {colors['text']}; font-weight: 600; padding: 4px 12px; border-radius: 12px; font-size: 0.9em;">
                            {category}
                        </span>
                    </div>
                    <h4 style="margin: 0 0 10px 0; color: #1f2937;">{trend['title']}</h4>
                    <p style="color: #6b7280; font-size: 0.85em; margin-top: 8px;">
                        출처: {trend['source']}
                    </p>
                </div>
            """
        
        html_content += """
                </div>
                <div style="margin-top: 20px; padding: 15px; background-color: #f8fafc; border-radius: 6px; border-right: 4px solid #c4b5fd;">
                    <p style="margin: 0; color: #4b5563; font-size: 0.9em; font-style: italic;">
                        💡 개발자 TIP: 최신 기술 트렌드를 놓치지 않으려면 공식 문서와 깃허브 트렌드를 정기적으로 확인하세요. 
                        새로운 기술을 배울 때는 핵심 개념을 이해한 후 프로젝트에 적용해보는 것이 가장 효과적입니다.
                    </p>
                </div>
            </div>
        </div>
        """

    # 6. 푸터 추가
    html_content += f"""
    <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb; color: #6b7280; font-size: 0.9em;">
        <p>이 뉴스레터는 실시간으로 수집된 AI 정보로 구성됩니다.</p>
        <p>수신 거부를 원하시면 회신 바랍니다.</p>
        <p style="margin-top: 10px; font-size: 0.9em;">발신: AI 레이더 | {today} 발행</p>
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
