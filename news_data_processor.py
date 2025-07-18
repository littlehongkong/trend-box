import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from supabase import create_client
import httpx
import asyncio
import re
import json
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('news_processor.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('news_processor')

# Load environment variables
load_dotenv()

# Configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
MODEL_NAME = "meta-llama/Llama-3-70b-chat-hf"
BATCH_SIZE = 50  # Number of headlines to process in one batch

# Supabase setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not all([SUPABASE_URL, SUPABASE_KEY]):
    logger.error("Supabase credentials not found in environment variables")
    raise ValueError("Missing Supabase credentials")

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Successfully connected to Supabase")
except Exception as e:
    logger.error(f"Failed to connect to Supabase: {str(e)}")
    raise

# Category mapping for database storage
CATEGORY_MAPPING = {
    "🚀 New Services/Launches": "new_services",
    "🛠️ Updates/Policy Changes": "updates_policy",
    "📊 Investment/Business": "investment_business",
    "⚙️ Infrastructure/Dev Tools": "infrastructure",
    "📈 Technology Trends (Auto)": "tech_trends",
    "📰 Other News": "other_news"
}

class NewsClassifier:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.max_chunk_size = 2000  # Reduced chunk size to be safer

    async def process_headlines(self, headlines: List[str]) -> Dict[str, Any]:
        """Process headlines to get both classification and summary using Together.ai
        
        Args:
            headlines: List of news headlines to process
            
        Returns:
            Dict containing 'classification' and 'summary' keys with their respective data
        """
        if not headlines:
            return {"classification": {}, "summary": {}}

        # Phase 1: Classification only
        classification_results = await self._classify_headlines(headlines)
        
        # Phase 2: Summarization based on classified results
        summary_results = await self._summarize_headlines(classification_results)
        
        return {
            "classification": classification_results,
            "summary": summary_results
        }
    
    async def _classify_headlines(self, headlines: List[str]) -> Dict[str, List[str]]:
        """Classify headlines into categories"""
        if not headlines:
            return {}

        # Process in chunks
        all_results = {category: [] for category in CATEGORY_MAPPING.keys()}
        current_chunk = []
        current_length = 0

        for headline in headlines:
            headline_length = len(headline) + 10  # Buffer for formatting
            
            if current_chunk and (current_length + headline_length) > self.max_chunk_size:
                chunk_result = await self._process_classification_chunk(current_chunk)
                self._merge_classification_results(all_results, chunk_result)
                current_chunk = []
                current_length = 0
            
            current_chunk.append(headline)
            current_length += headline_length
        
        if current_chunk:
            chunk_result = await self._process_classification_chunk(current_chunk)
            self._merge_classification_results(all_results, chunk_result)
        
        return all_results
    
    async def _summarize_headlines(self, classified_news: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Generate summaries for classified headlines"""
        if not classified_news or not any(classified_news.values()):
            return {}
            
        summary_results = {category: [] for category in CATEGORY_MAPPING.keys()}
        
        for category, headlines in classified_news.items():
            if not headlines:
                continue
                
            # Process in chunks if there are many headlines
            chunk_size = 10  # Smaller chunk size for summarization
            for i in range(0, len(headlines), chunk_size):
                chunk = headlines[i:i + chunk_size]
                summary = await self._process_summary_chunk(category, chunk)
                if summary and category in summary_results:
                    summary_results[category].extend(summary.get(category, []))
        
        return summary_results

    async def _process_classification_chunk(self, headlines: List[str]) -> Dict[str, List[str]]:
        """Process a single chunk of headlines for classification only"""
        if not headlines:
            logger.warning("No headlines provided for classification")
            return {}
            
        prompt = self._build_classification_prompt(headlines)
        logger.info(f"Classifying {len(headlines)} headlines, prompt length: {len(prompt)} chars")
        logger.debug(f"Classification prompt: {prompt[:500]}...")
        
        try:
            logger.info("Sending request to Together API...")
            response = await self._make_api_request(prompt)
            
            if not response:
                logger.error("No response received from API")
                return {}
                
            logger.debug(f"API Response: {json.dumps(response, ensure_ascii=False, indent=2)[:1000]}...")
            
            if 'choices' not in response or not response['choices']:
                logger.error(f"Unexpected API response format: {response}")
                return {}
                
            message = response['choices'][0].get('message', {})
            if not message or 'content' not in message:
                logger.error(f"No content in API response: {response}")
                return {}
                
            content = message['content']
            logger.debug(f"Raw classification content: {content[:1000]}...")
            
            # Save raw response for debugging
            with open('classification_response.txt', 'w', encoding='utf-8') as f:
                f.write(content)
                
            result = self._parse_classification_response(content)
            logger.info(f"Successfully parsed {sum(len(items) for items in result.values())} classified items")
            return result
            
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}", exc_info=True)
            # Save error context for debugging
            try:
                with open('classification_error.txt', 'w', encoding='utf-8') as f:
                    f.write(f"Error: {str(e)}\n\n")
                    f.write(f"Response: {str(response) if 'response' in locals() else 'No response'}\n")
            except Exception as save_error:
                logger.error(f"Failed to save error details: {save_error}")
            return {}
    
    async def _process_summary_chunk(self, category: str, headlines: List[str]) -> Dict[str, List[str]]:
        """Process a single chunk of headlines for summarization"""
        if not headlines or not category:
            return {}
            
        prompt = self._build_summary_prompt(category, headlines)
        logger.info(f"Summarizing {len(headlines)} headlines for {category}, prompt length: {len(prompt)} chars")
        
        try:
            response = await self._make_api_request(prompt)
            if not response:
                return {}
                
            content = response['choices'][0]['message']['content']
            return self._parse_summary_response(category, content)
            
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}", exc_info=True)
            return {}
    
    async def _make_api_request(self, prompt: str) -> Dict[str, Any]:
        """Make API request to Together API"""
        request_id = str(uuid.uuid4())[:8]  # Generate a unique ID for this request
        logger.info(f"[Req {request_id}] Sending API request to {TOGETHER_API_URL}")
        
        try:
            request_data = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 4000
            }
            
            # Log request details (without the full prompt)
            logger.debug(f"[Req {request_id}] Request data: {json.dumps({
                'model': MODEL_NAME,
                'message_length': len(prompt),
                'temperature': 0.3
            })}")
            
            # Save the full request for debugging
            with open(f'api_request_{request_id}.json', 'w', encoding='utf-8') as f:
                json.dump({
                    'url': TOGETHER_API_URL,
                    'headers': {k: '***' if k.lower() == 'authorization' else v 
                              for k, v in self.headers.items()},
                    'data': request_data
                }, f, ensure_ascii=False, indent=2)
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Log the request
                logger.debug(f"[Req {request_id}] Sending request to {TOGETHER_API_URL}")
                
                try:
                    response = await client.post(
                        TOGETHER_API_URL,
                        headers=self.headers,
                        json=request_data
                    )
                    
                    # Log response status and headers
                    logger.info(f"[Req {request_id}] Response status: {response.status_code}")
                    logger.debug(f"[Req {request_id}] Response headers: {dict(response.headers)}")
                    
                    # Handle error responses
                    if response.status_code >= 400:
                        error_detail = response.text
                        logger.error(f"[Req {request_id}] API Error {response.status_code}: {error_detail}")
                        
                        # Save error response for debugging
                        with open(f'api_error_{request_id}.txt', 'w', encoding='utf-8') as f:
                            f.write(f"Status: {response.status_code}\n")
                            f.write(f"Headers: {dict(response.headers)}\n\n")
                            f.write(error_detail)
                        
                        if response.status_code == 422:
                            logger.error(f"[Req {request_id}] Problematic prompt (first 500 chars): {prompt[:500]}...")
                        return None
                    
                    # Parse the successful response
                    try:
                        result = response.json()
                        logger.debug(f"[Req {request_id}] Successfully parsed JSON response")
                        
                        # Save the full response for debugging
                        with open(f'api_response_{request_id}.json', 'w', encoding='utf-8') as f:
                            json.dump(result, f, ensure_ascii=False, indent=2)
                        
                        if 'choices' not in result or not result['choices']:
                            logger.error(f"[Req {request_id}] Unexpected response format: 'choices' not found")
                            return None
                            
                        return result
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"[Req {request_id}] Failed to parse JSON response: {str(e)}")
                        logger.error(f"[Req {request_id}] Response text: {response.text[:1000]}...")
                        return None
                    
                except httpx.RequestError as e:
                    logger.error(f"[Req {request_id}] Request failed: {str(e)}")
                    return None
                
        except Exception as e:
            logger.error(f"[Req {request_id}] Unexpected error in API request: {str(e)}", exc_info=True)
            
        return None

    def _merge_classification_results(self, all_results: Dict[str, List[str]], new_results: Dict[str, List[str]]):
        """Merge new classification results into the accumulated results with deduplication"""
        if not new_results:
            return
            
        for category, items in new_results.items():
            if category not in all_results:
                all_results[category] = []
                
            # Add items that aren't already in the results
            existing_items = set(all_results[category])
            for item in items:
                # Simple exact match check first
                if item not in existing_items:
                    # More thorough check for similar items
                    is_duplicate = False
                    for existing in existing_items:
                        if self._is_similar_news(item, existing):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        all_results[category].append(item)
                        existing_items.add(item)
    
    def _is_similar_news(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two news items are similar using simple text comparison"""
        # Simple implementation - can be enhanced with more sophisticated similarity measures
        # like Jaccard similarity, Levenshtein distance, or embeddings
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Exact match
        if text1 == text2:
            return True
            
        # Check if one is a substring of the other (with some leeway)
        if len(text1) > 10 and len(text2) > 10:  # Only if both are reasonably long
            if text1 in text2 or text2 in text1:
                return True
                
        # Check for high word overlap (simple implementation)
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return False
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        similarity = len(intersection) / len(union) if union else 0
        
        return similarity >= threshold

    def _build_classification_prompt(self, headlines: List[str]) -> str:
        """Build the prompt for the LLM to classify headlines
        
        Args:
            headlines: List of news headlines to classify
            
        Returns:
            Formatted prompt string for classification
        """
        headlines_text = "\n".join([f"- {h}" for h in headlines])
        
        return f"""You are an AI/ML news classification expert. Your task is to analyze and classify each news headline into exactly one of the following categories:

## CATEGORIES (MUST USE THESE EXACT NAMES):

🚀 New Services/Launches: New AI products, services, or platforms being launched
🛠️ Updates/Policy Changes: Technical updates, API changes, or policy updates
📊 Investment/Business: Funding, acquisitions, partnerships with technical implications
⚙️ Infrastructure/Dev Tools: Technical tools, libraries, or infrastructure updates
📈 Technology Trends (Auto): Technical innovations or research findings
📰 Other News: Only use this if the headline doesn't fit any other category

## INSTRUCTIONS:
1. Classify EACH headline into ONE of the categories above.
2. Use the EXACT category names as provided above.
3. If uncertain, use "📰 Other News" only if the headline truly doesn't fit other categories.
4. Group similar headlines about the same topic together.
5. Return ONLY valid JSON with no additional text.

## OUTPUT FORMAT (JSON):
{{
    "🚀 New Services/Launches": ["headline 1", "headline 2", ...],
    "🛠️ Updates/Policy Changes": ["headline 3", ...],
    "📊 Investment/Business": [],
    "⚙️ Infrastructure/Dev Tools": [],
    "📈 Technology Trends (Auto)": [],
    "📰 Other News": []
}}

## HEADLINES TO CLASSIFY:
{headlines_text}"""

    def _build_summary_prompt(self, category: str, headlines: List[str]) -> str:
        """Build the prompt for the LLM to summarize headlines in a specific category
        
        Args:
            category: The category these headlines belong to
            headlines: List of news headlines to summarize
            
        Returns:
            Formatted prompt string for summarization
        """
        headlines_text = "\n".join([f"- {h}" for h in headlines])
        
        return f"""You are an AI/ML technical analyst. Your task is to write concise, insightful summaries in Korean for the following news headlines in the category: {category}

## INSTRUCTIONS:
1. Write 1-2 sentence summaries in Korean for each headline.
2. Focus on the key technical or business implications.
3. Be concise but informative.
4. Return ONLY valid JSON with no additional text.

## OUTPUT FORMAT (JSON):
{{
    "{category}": [
        "요약 1",
        "요약 2",
        ...
    ]
}}

## HEADLINES TO SUMMARIZE:
{headlines_text}"""

    def _extract_json_from_markdown(self, content: str) -> str:
        """Extract JSON content from markdown code block"""
        # Check if content is wrapped in markdown code block
        code_block_start = content.find('```')
        if code_block_start != -1:
            # Find the start of the actual JSON (after the first ```)
            json_start = content.find('{', code_block_start)
            if json_start == -1:
                # If no { after ```, try to find any JSON-like content
                json_start = code_block_start + 3  # Skip the ```
            
            # Find the end of the code block
            code_block_end = content.find('```', code_block_start + 3)
            if code_block_end == -1:
                # If no closing ```, take everything after the first ```
                return content[json_start:].strip()
            else:
                return content[json_start:code_block_end].strip()
        return content.strip()

    def _parse_classification_response(self, content: str) -> Dict[str, List[str]]:
        """Parse the classification response from the API"""
        result = {category: [] for category in CATEGORY_MAPPING.keys()}
        
        if not content or not content.strip():
            logger.error("Empty response content received from API")
            return result
            
        logger.debug(f"Raw classification response: {content[:500]}")
        
        try:
            # First, try to parse the content as is
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from markdown
                logger.info("Direct JSON parsing failed, trying to extract from markdown...")
                json_content = self._extract_json_from_markdown(content)
                parsed = json.loads(json_content)
            
            logger.debug(f"Successfully parsed JSON: {json.dumps(parsed, ensure_ascii=False, indent=2)[:500]}")
            
            # Handle the parsed data
            if not isinstance(parsed, dict):
                logger.error(f"Expected a dictionary but got: {type(parsed).__name__}")
                return result
                
            for category, items in parsed.items():
                # Normalize category name
                normalized_category = category.strip()
                
                # Ensure the category exists in our mapping
                if normalized_category not in result:
                    logger.warning(f"Unexpected category in response: {normalized_category}")
                    continue
                    
                if not isinstance(items, list):
                    logger.warning(f"Expected list of items for category {normalized_category}, got {type(items).__name__}")
                    continue
                    
                # Clean and add each item
                for item in items:
                    if isinstance(item, str) and item.strip():
                        result[normalized_category].append(item.strip())
                    else:
                        logger.warning(f"Skipping invalid item in category {normalized_category}: {item}")
            
            # Log the number of items in each category for debugging
            for category, items in result.items():
                if items:
                    logger.debug(f"Category {category} has {len(items)} items")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse classification response: {e}")
            logger.error(f"Response content (first 1000 chars): {content[:1000]}")
            
            # Save the problematic response for debugging
            with open('failed_parse_response.txt', 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Try to find where the JSON might be malformed
            try:
                # Try to find the start of JSON content
                start_idx = content.find('{')
                if start_idx > 0:
                    logger.warning(f"Found JSON starting at position {start_idx}")
                    logger.warning(f"Content before JSON: {content[:start_idx]}")
                    # Try to parse from the first {
                    json_content = content[start_idx:]
                    # Try to find the end of JSON object
                    brace_count = 0
                    in_string = False
                    escape = False
                    
                    for i, char in enumerate(json_content):
                        if char == '"' and not escape:
                            in_string = not in_string
                        elif char == '\\' and in_string:
                            escape = not escape
                            continue
                        elif char == '{' and not in_string:
                            brace_count += 1
                        elif char == '}' and not in_string:
                            brace_count -= 1
                            if brace_count == 0:
                                json_content = json_content[:i+1]
                                break
                        escape = False
                    
                    logger.info(f"Extracted JSON content: {json_content[:100]}...")
                    parsed = json.loads(json_content)
                    logger.info("Successfully parsed JSON after extraction")
                    return self._parse_classification_response(json_content)
            except Exception as inner_e:
                logger.error("Failed to fix JSON content", exc_info=True)
                
        except Exception as e:
            logger.error(f"Unexpected error parsing classification response: {e}", exc_info=True)
            
        return result
    
    def _parse_summary_response(self, expected_category: str, content: str) -> Dict[str, List[str]]:
        """Parse the summary response from the API for a specific category"""
        result = {expected_category: []}
        
        try:
            # Try to parse the JSON response
            parsed = json.loads(content)
            
            # Get the summaries for the expected category
            if expected_category in parsed and isinstance(parsed[expected_category], list):
                for item in parsed[expected_category]:
                    if isinstance(item, str) and item.strip():
                        result[expected_category].append(item.strip())
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse summary response: {e}")
            logger.debug(f"Response content: {content[:500]}...")
        except Exception as e:
            logger.error(f"Error parsing summary response: {e}", exc_info=True)
            
        return {}
    
    def _fix_encoding(self, data):
        """Helper function to fix encoding issues in the response"""
        if isinstance(data, str):
            try:
                return data.encode('latin1').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                return data
        elif isinstance(data, list):
            return [self._fix_encoding(item) for item in data]
        elif isinstance(data, dict):
            return {self._fix_encoding(k): self._fix_encoding(v) for k, v in data.items()}
        return data


class NewsProcessor:
    def __init__(self, supabase_client, classifier: NewsClassifier):
        self.supabase = supabase_client
        self.classifier = classifier
        # Set timezone to KST (UTC+9)
        self.kst = timezone(timedelta(hours=9))
        self.today = datetime.now(self.kst).date()
        self.section_titles = {
            "new_services": "🚀 New AI Services & Launches",
            "updates_policy": "🛠️ AI Updates & Policy Changes",
            "investment_business": "📊 AI Investments & Business",
            "infrastructure": "⚙️ AI Infrastructure & Dev Tools",
            "tech_trends": "📈 AI Technology Trends",
            "other_news": "📰 Other AI News"
        }

    async def process_news(self):
        """Main method to process news"""
        logger.info("Starting news processing...")
        
        # Get today's news
        news_items = self._get_todays_news()
        if not news_items:
            logger.info("No news found for today.")
            return
            
        logger.info(f"Found {len(news_items)} news items for today.")
        
        # Filter out duplicates
        unique_news = await self._find_duplicates(news_items)
        if not unique_news:
            logger.info("No unique news items to process after duplicate removal.")
            return
            
        logger.info(f"Processing {len(unique_news)} unique news items.")
        
        # Extract headlines
        headlines = [item['title'] for item in unique_news]
        
        # Process headlines to get both classification and summary
        result = await self.classifier.process_headlines(headlines)
        
        # Save to database and mark as processed
        await self._save_classification(result, unique_news)
        
        logger.info("News processing completed successfully.")
    
    def _clean_title(self, title: str) -> str:
        """Clean title by replacing single quotes with similar-looking special characters
        
        Args:
            title: The title to clean
            
        Returns:
            str: Cleaned title with single quotes replaced
        """
        if not title:
            return title
            
        # Replace single quotes with similar-looking special characters
        # Left single quote with '`' (backtick)
        # Right single quote with '\u2019' (right single quotation mark)
        # Single quote in the middle of a word with '\u02BC' (modifier letter apostrophe)
        cleaned = title.replace("'", "\u2019")  # Replace all single quotes with right single quotation mark
        
        # Handle cases where single quotes are used as apostrophes in the middle of words
        cleaned = cleaned.replace("\u2019s ", "\u02BCs ")  # 's -> ʼs
        cleaned = cleaned.replace("\u2019t ", "\u02BCt ")  # 't -> ʼt
        cleaned = cleaned.replace("\u2019re ", "\u02BCre ") # 're -> ʼre
        cleaned = cleaned.replace("\u2019ve ", "\u02BCve ") # 've -> ʼve
        cleaned = cleaned.replace("\u2019ll ", "\u02BCll ") # 'll -> ʼll
        cleaned = cleaned.replace("\u2019d ", "\u02BCd ")   # 'd -> ʼd
        cleaned = cleaned.replace("\u2019m ", "\u02BCm ")   # 'm -> ʼm
        
        # Handle leading single quotes (like in contractions or quotes at the beginning)
        if cleaned.startswith("\u2019"):
            cleaned = "`" + cleaned[1:]
            
        return cleaned

    def _get_todays_news(self) -> List[Dict[str, Any]]:
        """Fetch today's news from the database
        
        Returns:
            List of news items from today that haven't been marked as duplicates
            
        Note:
            Date range is set to KST 00:00:00 to 23:59:59.999 of the current day
            which translates to UTC 15:00:00 (previous day) to 14:59:59.999 (current day)
        """
        try:
            # Get current date in KST
            kst = timezone(timedelta(hours=9))
            utc = timezone.utc
            
            # Get current date components in KST
            kst_now = datetime.now(kst)
            year = kst_now.year
            month = kst_now.month
            day = kst_now.day
            
            # Calculate UTC time range that corresponds to KST 00:00:00 - 23:59:59.999
            # KST 00:00:00 = UTC 15:00:00 (previous day)
            utc_start = datetime(year, month, day - 1, 15, 0, 0, 0, tzinfo=utc)
            # KST 23:59:59.999 = UTC 14:59:59.999 (current day)
            utc_end = datetime(year, month, day, 14, 59, 59, 999000, tzinfo=utc)
            
            logger.info(f"Fetching news from {utc_start} to {utc_end}")
            
            # Query Supabase with UTC time range
            response = self.supabase.table('ai_news') \
                .select('*') \
                .gte('pub_date', utc_start.isoformat()) \
                .lte('pub_date', utc_end.isoformat()) \
                .order('pub_date', desc=True) \
                .limit(20) \
                .execute()
                
            if not response.data:
                logger.warning("No news items found in the database for the specified time range.")
                return []
                
            logger.info(f"Found {len(response.data)} news items in the database.")
            
            # Clean titles before returning
            for item in response.data:
                if 'title' in item and item['title']:
                    item['title'] = self._clean_title(item['title'])
                    
            return response.data
            
        except Exception as e:
            logger.error(f"Error fetching today's news: {str(e)}")
            return []
            
    def _simple_similarity(self, title1: str, title2: str) -> bool:
        """Check if two titles are similar using simple text matching
        
        Args:
            title1: First title to compare
            title2: Second title to compare
            
        Returns:
            bool: True if titles are considered similar
        """
        from difflib import SequenceMatcher
        
        # Remove common words that don't affect meaning
        common_words = {'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but',
                      '은', '는', '이', '가', '을', '를', '에', '에서', '으로', '로', '의', '과', '와', '도', '만', '까지'}
        
        # Simple case: if one title is contained in another
        if title1 in title2 or title2 in title1:
            return True
            
        # Check sequence similarity
        similarity = SequenceMatcher(None, title1, title2).ratio()
        return similarity >= 0.9

    async def _find_duplicates_with_llm(self, current_batch: List[Dict[str, Any]], 
                                     previous_uniques: List[Dict[str, Any]] = None) -> tuple[list[Dict[str, Any]], list[str]]:
        """Use LLM to find duplicate news items in a batch, considering previous unique items
        
        Args:
            current_batch: List of news items in current batch
            previous_uniques: List of previously identified unique items to check against
            
        Returns:
            tuple: (list of unique news items, list of duplicate news IDs)
        """
        if not current_batch:
            return [], []
            
        # Combine current batch with previous uniques for comparison
        all_items = (previous_uniques or []) + current_batch
        
        # If we have too many items, split into chunks that fit within token limits
        max_items_per_batch = 20  # Adjust based on token limits
        
        unique_items = {}
        duplicate_ids = set()
        
        # Process in chunks to avoid token limits
        for i in range(0, len(all_items), max_items_per_batch):
            chunk = all_items[i:i + max_items_per_batch]
            
            # Prepare the prompt for the LLM
            prompt = """
            You are a news analysis expert. Your task is to identify duplicate news articles from the following list.
            Articles are considered duplicates if they report the same news story, even if the wording is different.
            
            Instructions:
            1. Group the articles by the news story they report on.
            2. For each group, keep the most informative/complete version (prioritize items with more details).
            3. If an article is a duplicate of one in a previous batch, mark it as duplicate.
            4. Return a JSON object with the following structure:
            {
                "groups": [
                    {
                        "primary_id": "id_of_article_to_keep",
                        "duplicate_ids": ["id1", "id2", ...]
                    },
                    ...
                ]
            }
            
            Articles to analyze:
            """
            
            # Add articles to the prompt with metadata
            for item in chunk:
                prompt += f"\nID: {item['id']}\n"
                prompt += f"Title: {item['title']}\n"
                if 'description' in item:
                    prompt += f"Description: {item['description'][:200]}...\n"
                prompt += f"Source: {item.get('source', 'N/A')}\n"
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        TOGETHER_API_URL,
                        headers={
                            "Authorization": f"Bearer {TOGETHER_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": MODEL_NAME,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.1,
                            "max_tokens": 4000
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    content = result['choices'][0]['message']['content']

                    # Clean up the response to handle markdown code blocks and extract JSON
                    content = content.strip()
                    
                    # Try to find JSON in markdown code blocks first
                    json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', content, re.DOTALL)
                    if json_match:
                        try:
                            analysis = json.loads(json_match.group(1).strip())
                        except json.JSONDecodeError:
                            # If parsing fails, try to find any JSON in the content
                            json_match = re.search(r'({[\s\S]*})', content, re.DOTALL)
                            if json_match:
                                try:
                                    analysis = json.loads(json_match.group(1).strip())
                                except json.JSONDecodeError:
                                    analysis = {'groups': []}
                            else:
                                analysis = {'groups': []}
                    else:
                        # If no code block, try to find JSON directly in the response
                        try:
                            analysis = json.loads(content)
                        except json.JSONDecodeError:
                            # Try to find any JSON object in the content
                            json_match = re.search(r'({[\s\S]*})', content, re.DOTALL)
                            if json_match:
                                try:
                                    analysis = json.loads(json_match.group(1).strip())
                                except:
                                    analysis = {'groups': []}
                            else:
                                analysis = {'groups': []}
                    
                    # Log the extracted analysis for debugging
                    logger.debug(f"Extracted analysis: {json.dumps(analysis, ensure_ascii=False, indent=2)}")
                    
                    # Process the results
                    if not isinstance(analysis, dict) or 'groups' not in analysis:
                        logger.warning(f"Unexpected LLM response format: {content}")
                        analysis = {'groups': []}
                        
                    for group in analysis.get('groups', []):
                        primary_id = str(group.get('primary_id'))
                        if not primary_id:
                            continue
                            
                        # Find the primary item
                        primary_item = next((item for item in chunk if str(item['id']) == primary_id), None)
                        if not primary_item:
                            continue
                            
                        # Add to unique items if not already there
                        if primary_id not in unique_items:
                            unique_items[primary_id] = primary_item
                        
                        # Mark duplicates
                        for dup_id in group.get('duplicate_ids', []):
                            dup_id = str(dup_id)
                            if dup_id != primary_id:  # Make sure we're not marking the primary as duplicate
                                duplicate_ids.add(dup_id)
                                
                                # If a previously unique item is marked as duplicate, update our records
                                if dup_id in unique_items:
                                    del unique_items[dup_id]
                                    
            except Exception as e:
                logger.error(f"Error in LLM batch duplicate detection: {str(e)}")
                # In case of error, treat all in this chunk as unique
                for item in chunk:
                    item_id = str(item['id'])
                    if item_id not in duplicate_ids:
                        unique_items[item_id] = item
        
        # Filter out any items that were marked as duplicates
        unique_items = {k: v for k, v in unique_items.items() if k not in duplicate_ids}
        
        # Only return items from current batch (not previous uniques)
        current_batch_ids = {str(item['id']) for item in current_batch}
        batch_duplicates = [did for did in duplicate_ids if did in current_batch_ids]
        batch_uniques = [item for item in unique_items.values() 
                        if str(item['id']) in current_batch_ids]
        
        return batch_uniques, batch_duplicates

    async def _find_duplicates(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify and mark duplicate news items using a two-step approach
        
        Args:
            news_items: List of news items to check for duplicates
            
        Returns:
            List of non-duplicate news items ready for processing
        """
        if not news_items:
            return []
            
        logger.info(f"Starting duplicate detection for {len(news_items)} news items")
        
        # Sort by publish date (newer first) to keep the most recent version
        news_items.sort(key=lambda x: x.get('pub_date', ''), reverse=True)
        
        # Step 1: Simple similarity check (fast)
        unique_news = []
        potential_duplicates = []
        
        # First pass: check for obvious duplicates using simple similarity
        for item in news_items:
            is_duplicate = False
            
            for unique_item in unique_news:
                if self._simple_similarity(item['title'], unique_item['title']):
                    # Found a duplicate using simple comparison
                    logger.info(f"Found simple duplicate: '{item['title']}' is similar to '{unique_item['title']}'")
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                # If no obvious duplicate found, keep for LLM check
                potential_duplicates.append(item)
            
            # Keep track of unique items for cross-batch comparison
            if not is_duplicate:
                unique_news.append(item)
        
        logger.info(f"After simple deduplication: {len(unique_news)} unique items, {len(potential_duplicates)} items for LLM check")
        
        # Step 2: Use LLM for more sophisticated duplicate detection
        if len(potential_duplicates) > 1:  # Need at least 2 items to compare
            # Get previously processed unique items from database for cross-batch comparison
            try:
                # Get unique items from the last 24 hours to compare with
                yesterday = datetime.now(self.kst) - timedelta(days=1)
                try:
                    # Try with the correct table name (adjust 'ai_news' to your actual table name)
                    result = self.supabase.table('ai_news')\
                        .select('*')\
                        .eq('is_duplicate', False)\
                        .gte('pub_date', yesterday.isoformat())\
                        .order('pub_date', desc=True)\
                        .execute()
                    
                    previous_uniques = result.data if hasattr(result, 'data') else []
                except Exception as e:
                    logger.warning(f"Could not fetch previous uniques: {str(e)}")
                    previous_uniques = []
                logger.info(f"Found {len(previous_uniques)} previously processed unique items for comparison")
                
            except Exception as e:
                logger.error(f"Error fetching previous unique items: {str(e)}")
                previous_uniques = []
            
            # Process in batches with LLM
            all_llm_uniques = []
            all_duplicate_ids = set()
            
            for i in range(0, len(potential_duplicates), BATCH_SIZE):
                batch = potential_duplicates[i:i + BATCH_SIZE]
                logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(potential_duplicates)-1)//BATCH_SIZE + 1} with {len(batch)} items")
                
                # Include previously identified uniques in the comparison
                llm_uniques, duplicate_ids = await self._find_duplicates_with_llm(
                    batch, 
                    previous_uniques + all_llm_uniques
                )
                
                all_llm_uniques.extend(llm_uniques)
                all_duplicate_ids.update(duplicate_ids)
            
            # Mark LLM-identified duplicates
            if all_duplicate_ids:
                logger.info(f"LLM identified {len(all_duplicate_ids)} duplicate items")
                self._mark_as_duplicate(list(all_duplicate_ids))
            
            # Update unique items with LLM results
            unique_news = [item for item in unique_news 
                         if str(item['id']) not in all_duplicate_ids]
            unique_news.extend(all_llm_uniques)
        
        # Mark all remaining items as processed
        processed_ids = [item['id'] for item in unique_news]
        if processed_ids:
            self._mark_as_processed([{'id': pid} for pid in processed_ids])
        
        logger.info(f"Final unique items after deduplication: {len(unique_news)}")
        return unique_news
        
    async def _is_similar(self, title1: str, title2: str) -> bool:
        """Check if two news titles refer to the same news using LLM
        
        Args:
            title1: First title to compare
            title2: Second title to compare
            
        Returns:
            bool: True if titles are considered to be about the same news
        """
        # If titles are exactly the same, return True immediately
        if title1.strip().lower() == title2.strip().lower():
            return True
            
        # Prepare the prompt for the LLM
        prompt = f"""
        You are a news analysis expert. Your task is to determine if two news headlines are about the same news story.
        
        Headline 1: "{title1}"
        Headline 2: "{title2}"
        
        Consider the following factors in your analysis:
        1. Are the main subjects/entities the same?
        2. Are they reporting the same event or development?
        3. Are the key details (who, what, when, where, why) consistent?
        
        Respond with a JSON object containing:
        {{
            "is_same_news": boolean,
            "reasoning": "Brief explanation of your decision"
        }}
        """
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    TOGETHER_API_URL,
                    headers={
                        "Authorization": f"Bearer {TOGETHER_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": MODEL_NAME,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1,  # Low temperature for more deterministic responses
                        "max_tokens": 200
                    }
                )
                response.raise_for_status()
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                # Parse the JSON response
                import json
                try:
                    analysis = json.loads(content)
                    logger.info(f"LLM analysis for '{title1}' vs '{title2}': {analysis.get('reasoning', 'No reasoning provided')}")
                    return analysis.get('is_same_news', False)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM response: {content}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error in LLM similarity check: {str(e)}")
            # Fallback to simple similarity check if LLM fails
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, title1, title2).ratio()
            return similarity >= 0.85  # Slightly higher threshold for fallback
            

    
    async def _save_classification(self, result: Dict[str, Any], news_items: List[Dict[str, Any]]):
        """Save classified news and summaries to the newsletter_sections table
        
        Args:
            result: Dictionary containing 'classification' and 'summary' keys
            news_items: List of news items to process
        """
        try:
            date_str = self.today.strftime('%Y-%m-%d')
            logger.info(f"[저장 시작] 날짜: {date_str}, 카테고리 수: {len(result.get('classification', {}))}")
            
            # Log input data for debugging
            logger.debug(f"[분류 결과] {json.dumps(result.get('classification', {}), ensure_ascii=False, indent=2)}")
            logger.debug(f"[요약 결과] {json.dumps(result.get('summary', {}), ensure_ascii=False, indent=2)}")
            
            # Delete existing entries for the date
            logger.info(f"[기존 데이터 삭제] 날짜: {date_str}")
            try:
                delete_result = self.supabase.table('newsletter_sections')\
                    .delete()\
                    .eq('publish_date', date_str)\
                    .execute()
                logger.info(f"[기존 데이터 삭제 완료] 삭제된 레코드 수: {len(delete_result.data) if hasattr(delete_result, 'data') else '알 수 없음'}")
            except Exception as e:
                logger.error(f"[오류] 기존 데이터 삭제 실패: {str(e)}")
                raise
            
            # Create a mapping of (id, title) to news item for more accurate lookup
            news_key_to_item = {(item['id'], item['title']): item for item in news_items}
            processed_news_ids = set()  # Track processed news IDs to prevent duplicates
            
            # Get classification and summary from result
            classification = result.get('classification', {})
            summaries = result.get('summary', {})
            
            logger.info(f"[처리 시작] 총 {len(classification)}개 카테고리, {len(news_items)}개 뉴스 항목")
            
            # Process each category
            for category_display, headlines in classification.items():
                if not headlines:
                    logger.debug(f"[카테고리 건너뜀] {category_display}: 빈 헤드라인")
                    continue
                    
                category_key = CATEGORY_MAPPING.get(category_display)
                if not category_key:
                    logger.warning(f"[경고] 알 수 없는 카테고리: {category_display}")
                    continue
                
                logger.info(f"[카테고리 처리 중] {category_display} ({len(headlines)}개 항목)")
                
                # Prepare content for this category
                section_content = []
                for idx, headline in enumerate(headlines, 1):
                    # Find matching news item by both ID and title that hasn't been processed yet
                    matching_items = [item for (id, title), item in news_key_to_item.items() 
                                    if title == headline and id not in processed_news_ids]
                    
                    if not matching_items:
                        logger.debug(f"  - [항목 {idx}] 일치하는 뉴스 없음: {headline}")
                        continue
                        
                    # Take the first matching item
                    news_item = matching_items[0]
                    try:
                        # Always use self.today for consistency
                        pub_date = datetime.combine(self.today, datetime.min.time()).astimezone(self.kst).isoformat()
                            
                        item_data = {
                            'id': news_item['id'],
                            'title': headline,
                            'source': news_item.get('source', 'Unknown'),
                            'url': news_item.get('url', ''),
                            'published_at': pub_date,
                        }
                        section_content.append(item_data)
                        processed_news_ids.add(news_item['id'])
                        logger.debug(f"  - [항목 {idx}] 추가됨: {headline}")
                        
                    except Exception as e:
                        logger.error(f"[오류] 항목 처리 실패: {headline}, 오류: {str(e)}")
                
                if not section_content:
                    logger.warning(f"[경고] {category_display}에 저장할 유효한 항목이 없습니다.")
                    continue
                
                logger.info(f"[카테고리 준비 완료] {category_display}: {len(section_content)}개 항목")
                
                # Get summary for this category, if available
                category_summary = summaries.get(category_display, [])
                summary_text = '\n'.join([f"• {insight}" for insight in category_summary])
                
                logger.debug(f"[요약] {category_display}: {summary_text[:100]}...")
                logger.debug(f"[내용 샘플] {json.dumps(section_content[0], ensure_ascii=False, indent=2) if section_content else '없음'}")
                
                # Prepare section data with consistent date
                section_data = {
                    'section_name': category_key,
                    'section_title': self.section_titles.get(category_key, category_key),
                    'content': section_content,
                    'summary': summary_text,
                    'display_order': list(CATEGORY_MAPPING.values()).index(category_key) + 1,  # 1-based index
                    'publish_date': date_str,  # Use the same date_str for consistency
                    'is_published': True,
                    'updated_at': datetime.now(timezone.utc).isoformat()
                }
                
                # Insert the new section
                try:
                    self.supabase.table('newsletter_sections')\
                        .insert(section_data)\
                        .execute()
                    logger.info(f"Inserted new section: {category_key}")
                except Exception as e:
                    logger.error(f"Error inserting section {category_key}: {str(e)}")
                    raise
                
                # Mark news items as processed
                news_ids = [item['id'] for item in section_content]
                self._mark_as_processed(news_ids)
                
        except Exception as e:
            logger.error(f"Error saving classification: {str(e)}")
            raise
    
    def _mark_as_duplicate(self, news_ids: List[int]) -> None:
        """Mark news items as duplicates in the database."""
        if not news_ids:
            return
            
        try:
            update_data = {'is_duplicate': True}
            # Only include updated_at if the column exists

            self.supabase.table('ai_news')\
                .update(update_data)\
                .in_('id', news_ids)\
                .execute()
            logger.info(f"Marked {len(news_ids)} news items as duplicates")
        except Exception as e:
            logger.error(f"Error marking news as duplicates: {str(e)}")
            raise
    
    def _mark_as_processed(self, news_items: Union[List[Dict[str, Any]], List[int]]) -> None:
        """Mark news items as processed in the database."""
        if not news_items:
            return
            
        try:
            # Handle both list of dicts and list of IDs
            if isinstance(news_items[0], dict):
                news_ids = [item['id'] for item in news_items if isinstance(item, dict) and 'id' in item]
            else:
                news_ids = [item for item in news_items if isinstance(item, int)]
                
            if not news_ids:
                return
                
            update_data = {'is_processed': True}
            # Only include updated_at if the column exists

            self.supabase.table('ai_news')\
                .update(update_data)\
                .in_('id', news_ids)\
                .execute()
            logger.info(f"Marked {len(news_ids)} news items as processed")
        except Exception as e:
            logger.error(f"Error marking news as processed: {str(e)}")
            raise

async def main():
    # Initialize Supabase client
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    together_api_key = os.getenv('TOGETHER_API_KEY')
    
    if not all([supabase_url, supabase_key, together_api_key]):
        logger.error("Missing required environment variables. Please ensure the following are set:\n"
                   "- SUPABASE_URL\n"
                   "- SUPABASE_KEY\n"
                   "- TOGETHER_API_KEY")
        return
    
    # Initialize clients
    supabase = create_client(supabase_url, supabase_key)
    classifier = NewsClassifier(api_key=together_api_key)
    
    logger.info("Starting news processing...")
    
    try:
        # Initialize and run processor
        processor = NewsProcessor(supabase, classifier)
        await processor.process_news()
        logger.info("News processing completed successfully.")
    except Exception as e:
        logger.error(f"Error during news processing: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(main())
