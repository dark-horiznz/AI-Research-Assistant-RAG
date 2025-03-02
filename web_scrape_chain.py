import streamlit as st
import google.generativeai as genai
import requests
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
import os
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from lxml import html

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# Set up Gemini API
def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("No Gemini API key found. Please set it in the .env file or in the Streamlit secrets.")
        st.stop()
    genai.configure(api_key=api_key)
    return api_key

# Gemini model functions
def get_gemini_model(model_name="gemini-1.5-pro", temperature=0.4):
    return genai.GenerativeModel(model_name)

def get_generation_config(temperature=0.4):
    return {
        "temperature": temperature,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

def get_safety_settings():
    return [
        {"category": category, "threshold": "BLOCK_NONE"}
        for category in [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ]
    ]

def generate_gemini_response(model, prompt):
    response = model.generate_content(
        prompt,
        generation_config=get_generation_config(),
        safety_settings=get_safety_settings()
    )
    if response.candidates and len(response.candidates) > 0:
        return response.candidates[0].content.parts[0].text
    return ''

# Prompt creation functions
def create_search_prompt(query, context=""):
    system_prompt = """You are a smart assistant designed to determine whether a query needs data from a web search or can be answered using a document database. 
    Consider the provided context if available. 
    If the query requires external information, No context is provided, Irrelevent context is present or latest information is required, then output the special token <SEARCH> 
    followed by relevant keywords extracted from the query to optimize for search engine results. 
    Ensure the keywords are concise and relevant. If document data is sufficient, simply return blank."""
    
    if context:
        return f"{system_prompt}\n\nContext: {context}\n\nQuery: {query}"
    
    return f"{system_prompt}\n\nQuery: {query}"

def create_summary_prompt(content):
    return f"""Please provide a comprehensive yet concise summary of the following content, highlighting the most important points and maintaining factual accuracy. Organize the information in a clear and coherent manner:

Content to summarize:
{content}

Summary:"""

# Web scraping functions
def init_selenium_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def extract_static_page(url):
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        text = soup.get_text(separator=" ", strip=True)
        return text[:5000]

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching page: {e}")
        return None
        
def extract_dynamic_page(url, driver):
    try:
        driver.get(url)
        time.sleep(random.uniform(2, 5))
        
        body = driver.find_element(By.TAG_NAME, "body")
        ActionChains(driver).move_to_element(body).perform()
        time.sleep(random.uniform(2, 5))
        
        page_source = driver.page_source
        tree = html.fromstring(page_source)
        
        text = tree.xpath('//body//text()')
        text_content = ' '.join(text).strip()
        return text_content[:1000]

    except Exception as e:
        st.error(f"Error fetching dynamic page: {e}")
        return None

def scrape_page(url):
    if "javascript" in url or "dynamic" in url:
        driver = init_selenium_driver()
        text = extract_dynamic_page(url, driver)
        driver.quit()
    else:
        text = extract_static_page(url)
    
    return text

def scrape_web(urls, max_urls=5):
    texts = []
    
    for url in urls[:max_urls]:
        text = scrape_page(url)
        
        if text:
            texts.append(text)
        else:
            st.warning(f"Failed to retrieve content from {url}")
            
    return texts

# Main functions
def check_search_needed(model, query, context):
    prompt = create_search_prompt(query, context)
    response = generate_gemini_response(model, prompt)
    
    if "<SEARCH>" in response:
        search_terms = response.split("<SEARCH>")[1].strip()
        return True, search_terms
    return False, None

def summarize_content(model, content):
    prompt = create_summary_prompt(content)
    return generate_gemini_response(model, prompt)

def process_query(query, context=''):
    with st.spinner("Processing query..."):
        model = get_gemini_model()
        search_tool = DuckDuckGoSearchRun()
        
        needs_search, search_terms = check_search_needed(model, query, context)
        
        result = {
            "original_query": query,
            "needs_search": needs_search,
            "search_terms": search_terms,
            "web_content": None,
            "summary": None
        }
        
        if needs_search:
            with st.spinner(f"Searching the web for: {search_terms}"):
                search_results = search_tool.run(search_terms)
                result["web_content"] = search_results
            
            with st.spinner("Summarizing search results..."):
                summary = summarize_content(model, search_results)
                result["summary"] = summary
        
    return result

# UI Components
def display_header():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://i.imgur.com/oLgq2zK.png", width=80)  # Replace with your logo if needed
    with col2:
        st.title("AI Research Assistant")
    st.markdown("---")

def display_results(result):
    if result["needs_search"]:
        st.success("Research completed!")
        
        with st.expander("Search Details", expanded=False):
            st.subheader("Search Terms Used")
            st.info(result["search_terms"])
            
            st.subheader("Raw Web Content")
            st.text_area("Web Content", result["web_content"], height=200)
        
        st.subheader("Summary of Findings")
        st.markdown(result["summary"])
    else:
        st.info("Based on the analysis, no web search was needed for this query.")

# Main application
def main():
    display_header()
    
    # Setup API key
    api_key = setup_gemini()
    
    # Input area
    with st.form("query_form"):
        query = st.text_area("Enter your research question", height=100, 
                           placeholder="E.g., What are the latest developments in quantum computing?")
        context = st.text_area("Optional: Add any context", height=100, 
                               placeholder="Add any additional context that might help with the research")
        submit_button = st.form_submit_button("🔍 Research")
    # Add error handling for Gemini API
    st.sidebar.subheader("API Status")
    try:
        # Test if Gemini model is accessible 
        test_model = get_gemini_model()
        test_response = test_model.generate_content(
            "test",
            generation_config=get_generation_config(),
            safety_settings=get_safety_settings()
        )
        st.sidebar.success("✅ Gemini API connected")
    except Exception as e:
        st.sidebar.error(f"❌ Gemini API Error: {str(e)}")
        st.error("There was an issue connecting to the Gemini API. Please check your API key or try again later.")
        st.info("Common issues: Invalid API key, API quota exceeded, or using an unsupported model version.")
    if submit_button and query:
        result = process_query(query, context)
        display_results(result)
        
    # Settings sidebar (for future extension)
    with st.sidebar:
        st.subheader("About")
        st.write("This AI Research Assistant uses Gemini and LangChain to search and summarize web content.")
        st.write("---")
        st.write("Powered by:")
        st.write("- Google Gemini API")
        st.write("- LangChain")
        st.write("- DuckDuckGo Search")

if __name__ == "__main__":
    main()