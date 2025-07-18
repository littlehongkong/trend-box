import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Union
from dotenv import load_dotenv
from supabase import create_client
import httpx
import asyncio
import pytz

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

        # Process in chunks
        all_results = {"classification": {}, "summary": {}}
        current_chunk = []
        current_length = 0

        for headline in headlines:
            # Estimate token count (roughly 4 chars per token)
            headline_length = len(headline) + 10  # Add some buffer for formatting
            
            # If adding this headline would exceed the chunk size, process current chunk
            if current_chunk and (current_length + headline_length) > self.max_chunk_size:
                chunk_result = await self._process_chunk(current_chunk)
                self._merge_results(all_results, chunk_result)
                current_chunk = []
                current_length = 0
            
            current_chunk.append(headline)
            current_length += headline_length
        
        # Process any remaining headlines
        if current_chunk:
            chunk_result = await self._process_chunk(current_chunk)
            self._merge_results(all_results, chunk_result)
        
        return all_results

    async def _process_chunk(self, headlines: List[str]) -> Dict[str, Any]:
        """Process a single chunk of headlines"""
        if not headlines:
            return {"classification": {}, "summary": {}}
            
        prompt = self._build_prompt(headlines)
        logger.info(f"Processing chunk with {len(headlines)} headlines, prompt length: {len(prompt)} chars")
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                request_data = {
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 4000
                }
                
                # Log request details (without sensitive data)
                logger.debug(f"Sending request to {TOGETHER_API_URL} with model {MODEL_NAME}")
                
                response = await client.post(
                    TOGETHER_API_URL,
                    headers=self.headers,
                    json=request_data
                )
                
                # Log response status and headers
                logger.debug(f"Response status: {response.status_code}")
                
                # For 4xx errors, log the response content
                if response.status_code >= 400:
                    error_detail = response.text
                    logger.error(f"API Error {response.status_code}: {error_detail}")
                    
                    # If it's a 422 error, log the problematic prompt
                    if response.status_code == 422:
                        logger.error(f"Problematic prompt (first 500 chars): {prompt[:500]}...")
                    
                    return {"classification": {}, "summary": {}}
                
                response.raise_for_status()
                result = response.json()
                
                if 'choices' not in result or not result['choices']:
                    logger.error("Unexpected response format: 'choices' not found")
                    return {"classification": {}, "summary": {}}
                    
                content = result['choices'][0]['message']['content']
                return self._parse_response(content)
                
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error occurred: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
        except Exception as e:
            logger.error(f"Unexpected error processing chunk: {str(e)}", exc_info=True)
            
        return {"classification": {}, "summary": {}}

    def _merge_results(self, all_results: Dict[str, Any], new_results: Dict[str, Any]):
        """Merge new results into the accumulated results"""
        # Initialize categories if they don't exist
        for category in CATEGORY_MAPPING.keys():
            if category not in all_results["classification"]:
                all_results["classification"][category] = []
            if category not in all_results["summary"]:
                all_results["summary"][category] = []
        
        # Merge classifications
        for category, items in new_results.get("classification", {}).items():
            if category in all_results["classification"]:
                all_results["classification"][category].extend(items)
        
        # Merge summaries
        for category, items in new_results.get("summary", {}).items():
            if category in all_results["summary"]:
                all_results["summary"][category].extend(items)

    def _build_prompt(self, headlines: List[str]) -> str:
        """Build the prompt for the LLM to classify and summarize headlines
        
        Args:
            headlines: List of news headlines to process
            
        Returns:
            Formatted prompt string
        """
        headlines_text = "\n".join([f"- {h}" for h in headlines])
        
        return f"""You are an AI/LLM news curator and technical analyst. Your task is to analyze and summarize AI/ML news for developers.

## INSTRUCTIONS:

1. **REMOVE DUPLICATES**: If multiple headlines describe the same news, keep only the most informative version.
2. **CLASSIFY** each unique headline into exactly one category.
3. **GENERATE** 3-5 key technical insights per category in Korean.

### CATEGORIES:

🚀 New Services/Launches: New AI products, services, or platforms being launched
🛠️ Updates/Policy Changes: Technical updates, API changes, or policy updates
📊 Investment/Business: Funding, acquisitions, partnerships with technical implications
⚙️ Infrastructure/Dev Tools: Technical tools, libraries, or infrastructure updates
📈 Technology Trends: Technical innovations or research findings
📰 Other News: AI-related news not fitting other categories

## OUTPUT FORMAT:

### CLASSIFICATION:
[Category Name]:
- [Headline 1]
- [Headline 2]

### KEY INSIGHTS (Korean):
[Category Name]:
1. [기술적 인사이트 1: 구체적이고 실행 가능한 내용]
2. [기술적 인사이트 2: 개발자 관점에서의 시사점]

## GUIDELINES FOR INSIGHTS:
- **기술적 초점**: 개발자에게 유용한 기술적 세부사항 포함
- **중복 제거**: 유사한 내용은 하나로 통합
- **구체성**: 모호한 표현 대신 구체적인 기술 스택, 아키텍처, 성능 수치 등 포함
- **실행 가능성**: 개발자가 활용할 수 있는 실질적인 조언 제공
- **간결성**: 불필요한 수식어 제거하고 핵심 내용만 전달

## PROCESS THESE HEADLINES:
{headlines_text}"""

    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM response to extract both classifications and summaries
        
        Args:
            content: Raw response content from the LLM
            
        Returns:
            Dict with 'classification' and 'summary' keys containing parsed data
        """
        result = {
            "classification": {
                "🚀 New Services/Launches": [],
                "🛠️ Updates/Policy Changes": [],
                "📊 Investment/Business": [],
                "⚙️ Infrastructure/Dev Tools": [],
                "📈 Technology Trends (Auto)": [],
                "📰 Other News": []
            },
            "summary": {
                "🚀 New Services/Launches": [],
                "🛠️ Updates/Policy Changes": [],
                "📊 Investment/Business": [],
                "⚙️ Infrastructure/Dev Tools": [],
                "📈 Technology Trends (Auto)": [],
                "📰 Other News": []
            }
        }
        
        current_section = None
        is_summary = False
        
        # Split content into classification and summary parts
        parts = content.split("### KEY INSIGHTS")
        classification_part = parts[0]
        summary_part = parts[1] if len(parts) > 1 else ""
        
        # Parse classification section
        for line in classification_part.split('\n'):
            line = line.strip()
            if not line or line.startswith('##'):
                continue
                
            # Check for category headers
            if line.endswith(':'):
                for category in result["classification"]:
                    if line.startswith(category):
                        current_section = category
                        break
            # Add headlines to current category
            elif current_section and line.startswith('-'):
                headline = line[1:].strip()
                if headline:
                    result["classification"][current_section].append(headline)
        
        # Parse summary section if it exists
        if summary_part:
            current_section = None
            for line in summary_part.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Check for category headers in summary
                if line.endswith(':'):
                    for category in result["summary"]:
                        if line.startswith(category):
                            current_section = category
                            break
                # Add numbered insights
                elif current_section and line[0].isdigit() and '. ' in line:
                    insight = line.split('. ', 1)[1].strip()
                    if insight:
                        result["summary"][current_section].append(insight)
        
        return result

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
                .execute()
                
            if not response.data:
                logger.warning("No news items found in the database for the specified time range.")
                return []
                
            logger.info(f"Found {len(response.data)} news items in the database.")
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
                    
                    # Parse the JSON response
                    import json
                    import re
                    
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
            # First, delete all sections for today's date
            try:
                delete_result = self.supabase.table('newsletter_sections')\
                    .delete()\
                    .eq('publish_date', self.today.strftime('%Y-%m-%d'))\
                    .execute()
                logger.info(f"Deleted {len(delete_result.data or [])} existing sections for {self.today.strftime('%Y-%m-%d')}")
            except Exception as e:
                logger.error(f"Error deleting existing sections: {str(e)}")
                raise
            
            # Create a mapping of headline to news item for quick lookup
            headline_to_news = {item['title']: item for item in news_items}
            
            # Get classification and summary from result
            classification = result.get('classification', {})
            summaries = result.get('summary', {})
            
            # Process each category
            for category_display, headlines in classification.items():
                if not headlines:
                    continue
                    
                category_key = CATEGORY_MAPPING.get(category_display)
                if not category_key:
                    logger.warning(f"Unknown category: {category_display}")
                    continue
                
                # Prepare content for this category
                section_content = []
                for headline in headlines:
                    if headline in headline_to_news:
                        news_item = headline_to_news[headline]
                        section_content.append({
                            'id': news_item['id'],
                            'title': headline,
                            'source': news_item.get('source', 'Unknown'),
                            'url': news_item.get('url', ''),
                            'published_at': datetime.fromisoformat(news_item.get('pub_date')).replace(tzinfo=timezone.utc).astimezone(self.kst).isoformat() if news_item.get('pub_date') else None,
                        })
                
                if not section_content:
                    continue
                
                # Get summary for this category, if available
                category_summary = summaries.get(category_display, [])
                summary_text = '\n'.join([f"• {insight}" for insight in category_summary])
                
                # Prepare section data
                section_data = {
                    'section_name': category_key,
                    'section_title': self.section_titles.get(category_key, category_key),
                    'content': section_content,
                    'summary': summary_text,
                    'display_order': list(CATEGORY_MAPPING.values()).index(category_key) + 1,  # 1-based index
                    'publish_date': self.today.strftime('%Y-%m-%d'),
                    'is_published': False,
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
