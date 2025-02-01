import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from langchain_community.tools import DuckDuckGoSearchRun
import google.generativeai as genai
from langchain.chains.combine_documents import create_stuff_documents_chain
from lxml import html
import time
from langchain_core.embeddings import Embeddings
import random
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def get_gemini_model(model_name: str = "gemini-pro", temperature: float = 0.4) -> genai.GenerativeModel:
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error getting Gemini model: {e}")
        return None

def get_generation_config(temperature: float = 0.4) -> Dict:
    return {
        "temperature": temperature,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

def get_safety_settings() -> List[Dict[str, str]]:
    return [
        {"category": category, "threshold": "BLOCK_NONE"}
        for category in [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
    ]

def generate_gemini_response(model: genai.GenerativeModel, prompt: str) -> str:
    try:
        response = model.generate_content(
            prompt,
            generation_config=get_generation_config(),
            safety_settings=get_safety_settings()
        )
        if response.candidates and len(response.candidates) > 0:
            return response.candidates[0].content.parts[0].text
    except Exception as e:
        print(f"Error generating Gemini response: {e}")
    return ''

def create_search_prompt(query: str, context: str = "") -> str:
    system_prompt = """You are a smart assistant designed to determine whether a query needs data from a web search or can be answered using a document database. 
    Consider the provided context if available. 
    If the query requires external information, No context is provided, Irrelevant context is present, or latest information is required, then output the special token <SEARCH> 
    followed by relevant keywords extracted from the query to optimize for search engine results. 
    Ensure the keywords are concise and relevant. If document data is sufficient, simply return blank."""
    
    return f"{system_prompt}\n\nContext: {context}\n\nQuery: {query}" if context else f"{system_prompt}\n\nQuery: {query}"

def create_summary_prompt(content: str) -> str:
    return f"""Please provide a comprehensive yet concise summary of the following content, highlighting the most important points and maintaining factual accuracy. Organize the information in a clear and coherent manner:

Content to summarize:
{content}

Summary:"""

def init_selenium_driver() -> webdriver.Chrome:
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        return webdriver.Chrome(options=chrome_options)
    except Exception as e:
        print(f"Error initializing Selenium driver: {e}")
        return None

def extract_static_page(url: str) -> Optional[str]:
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        return soup.get_text(separator=" ", strip=True)[:5000]
    except requests.exceptions.RequestException as e:
        print(f"Error extracting static page: {e}")
        return None

def extract_dynamic_page(url: str, driver: webdriver.Chrome) -> Optional[str]:
    try:
        driver.get(url)
        time.sleep(random.uniform(2, 5))
        body = driver.find_element(By.TAG_NAME, "body")
        ActionChains(driver).move_to_element(body).perform()
        time.sleep(random.uniform(2, 5))
        tree = html.fromstring(driver.page_source)
        text = tree.xpath('//body//text()')
        return ' '.join(text).strip()[:1000]
    except Exception as e:
        print(f"Error extracting dynamic page: {e}")
        return None

def scrape_page(url: str) -> Optional[str]:
    try:
        if "javascript" in url or "dynamic" in url:
            driver = init_selenium_driver()
            if driver:
                text = extract_dynamic_page(url, driver)
                driver.quit()
            else:
                text = None
        else:
            text = extract_static_page(url)
        return text
    except Exception as e:
        print(f"Error scraping page: {e}")
        return None

def scrape_web(urls: List[str], max_urls: int = 5) -> List[str]:
    texts = []
    for url in tqdm(urls[:max_urls], desc="Scraping websites"):
        text = scrape_page(url)
        if text:
            texts.append(text)
    return texts

def check_search_needed(model: genai.GenerativeModel, query: str, context: str) -> Tuple[bool, Optional[str]]:
    try:
        response = generate_gemini_response(model, create_search_prompt(query, context))
        if "<SEARCH>" in response:
            return True, response.split("<SEARCH>")[1].strip()
    except Exception as e:
        print(f"Error checking if search is needed: {e}")
    return False, None

def summarize_content(model: genai.GenerativeModel, content: str) -> str:
    try:
        return generate_gemini_response(model, create_summary_prompt(content))
    except Exception as e:
        print(f"Error summarizing content: {e}")
        return ''

def process_query(query: str, context: str = '') -> Dict:
    try:
        model = get_gemini_model()
        if not model:
            return {"error": "Failed to initialize Gemini model"}
        
        search_tool = DuckDuckGoSearchRun()
        
        needs_search, search_terms = check_search_needed(model, query, context)
        
        result = {
            "original_query": query,
            "needs_search": needs_search,
            "search_terms": search_terms,
            "web_content": None,
            "summary": None,
            "raw_response": None
        }
        
        if needs_search:
            search_results = search_tool.run(search_terms)
            result["web_content"] = search_results
            result["summary"] = summarize_content(model, search_results)
        
        return result
    except Exception as e:
        print(f"Error processing query: {e}")
        return {"error": str(e)}

def run(question: str, context: str = '') -> None:
    try:
        result = process_query(question, context)
        print("\nQuery Results:")
        print(f"Search needed: {result['needs_search']}")
        
        if result['needs_search']:
            print(f"\nSearch terms used: {result['search_terms']}")
            print("\nSummary of findings:")
            print(result['summary'])
    except Exception as e:
        print(f"Error running query: {e}")

def main(question: str, context: str = '') -> Dict:
    try:
        return process_query(question, context)
    except Exception as e:
        print(f"Error in main function: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    question = "What are the symptoms of COVID-19?"
    context = "I am a student researching common illnesses for a school project."
    main(question, context)