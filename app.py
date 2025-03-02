import os
import streamlit as st
import pdfplumber
import requests
import google.generativeai as genai
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_community.tools import DuckDuckGoSearchRun
from pinecone import Pinecone
from dotenv import load_dotenv
import numpy as np
import time
import random
from typing import List
import arxiv
import wikipedia
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from lxml import html
import base64
import os
import streamlit as st
import pdfplumber
import requests
import google.generativeai as genai
# Load environment variables
load_dotenv()

# Get API keys from environment variables
groq_key = os.getenv("GROQ_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gemini_key)
# Check if all required API keys are available
if not gemini_key:
    st.error("Gemini API key is missing. Please set either GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")
    
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

#-------------------------------------------------------------
# UTILITY FUNCTIONS
#-------------------------------------------------------------

# Gemini Embeddings class
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model_name = "models/embedding-001"
    
    def embed_documents(self, texts):
        return [self._convert_to_float32(genai.embed_content(
            model=self.model_name, content=text, task_type="retrieval_document"
        )["embedding"]) for text in texts]
    
    def embed_query(self, text):
        response = genai.embed_content(
            model=self.model_name, content=text, task_type="retrieval_query"
        )
        return self._convert_to_float32(response["embedding"])
    
    @staticmethod
    def _convert_to_float32(embedding):
        return np.array(embedding, dtype=np.float32).tolist()

# PDF handling functions
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def read_data_from_doc(uploaded_file):
    docs = []
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            table_text = "\n".join([
                "\n".join(["\t".join(cell if cell is not None else "" for cell in row) for row in table])
                for table in tables if table
            ]) if tables else ""
            images = page.images
            image_text = f"[{len(images)} image(s) detected]" if images else ""
            content = f"{text}\n\n{table_text}\n\n{image_text}".strip()
            if content:
                docs.append(Document(page_content=content, metadata={"page": i + 1}))
    return docs

def make_chunks(docs, chunk_len=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_len, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    return [Document(page_content=chunk.page_content, metadata=chunk.metadata) for chunk in chunks]

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

def summarize_text(text):
    model = get_gemini_model()
    prompt_text = f"Summarize the following research paper very concisely:\n{text[:5000]}"  # Truncate to 5000 chars
    summary = generate_gemini_response(model, prompt_text)
    return summary

#-------------------------------------------------------------
# RESEARCH ASSISTANT MODULE
#-------------------------------------------------------------

def download_pdf(pdf_url, save_path="temp_paper.pdf"):
    try:
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            return save_path
    except Exception as e:
        st.error(f"Error downloading PDF: {e}")
    return None

def search_arxiv(query, max_results=2):
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    
    arxiv_docs = []
    
    for result in client.results(search):
        pdf_link = next((link.href for link in result.links if 'pdf' in link.href), None)
        
        # Download, extract, and summarize PDF if link exists
        if pdf_link:
            with st.spinner(f"Processing arXiv paper: {result.title}"):
                pdf_path = download_pdf(pdf_link)
                if pdf_path:
                    text = extract_text_from_pdf(pdf_path)
                    summary = summarize_text(text)
                    # Clean up downloaded file
                    if os.path.exists(pdf_path):
                        os.remove(pdf_path)
                else:
                    summary = "PDF could not be downloaded."
        else:
            summary = "No PDF available."

        content = f"""
        **Title:** {result.title}
        **Authors:** {', '.join(author.name for author in result.authors)}
        **Published:** {result.published.strftime('%Y-%m-%d')}
        **Abstract:** {result.summary}
        **PDF Summary:** {summary}
        **PDF Link:** {pdf_link if pdf_link else 'Not available'}
        """

        arxiv_docs.append(Document(page_content=content, metadata={"source": "arXiv", "title": result.title}))
    
    return arxiv_docs

def search_wikipedia(query, max_results=2):
    try:
        page_titles = wikipedia.search(query, results=max_results)
        wiki_docs = []
        for title in page_titles:
            try:
                with st.spinner(f"Processing Wikipedia article: {title}"):
                    page = wikipedia.page(title)
                    wiki_docs.append(Document(
                        page_content=page.content[:2000], 
                        metadata={"source": "Wikipedia", "title": title}
                    ))
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
                st.warning(f"Error retrieving Wikipedia page {title}: {e}")
        return wiki_docs
    except Exception as e:
        st.error(f"Error searching Wikipedia: {e}")
        return []

class ResearchAssistant:
    def __init__(self):
        # Initialize LLM
        self.llm = ChatGroq(
            api_key=groq_key,
            model="llama3-70b-8192",
            temperature=0.2
        )
        
        # Set up the prompt template
        self.prompt = ChatPromptTemplate.from_template("""
        You are an expert research assistant. Use the following context to answer the question. 
        If you don't know the answer, say so, but try your best to find relevant information 
        from the provided context and additional context.
        
        Context from user documents:
        {context}
        
        Additional context from research sources:
        {additional_context}
        
        Question: {input}
        
        Answer:
        """)
        
        # Set up the question-answer chain
        self.question_answer_chain = create_stuff_documents_chain(
            self.llm, self.prompt
        )

    def retrieve_documents(self, query):
        user_context = []
        
        # Get documents from arXiv and Wikipedia
        arxiv_docs = search_arxiv(query)
        wiki_docs = search_wikipedia(query)
        
        summarized_context = []
        for doc in arxiv_docs:
            summarized_context.append(f"**ArXiv - {doc.metadata.get('title', 'Unknown Title')}**:\n{doc.page_content}...")
            
        for doc in wiki_docs:
            summarized_context.append(f"**Wikipedia - {doc.metadata.get('title', 'Unknown Title')}**:\n{doc.page_content}...")
            
        return user_context, summarized_context
    
    def chat(self, question):
        user_context, summarized_context = self.retrieve_documents(question)
        
        input_data = {
            "input": question,
            "context": "\n\n".join(user_context),
            "additional_context": "\n\n".join(summarized_context)
        }
        
        with st.spinner("Generating answer..."):
            # Use the LLM directly
            prompt_text = f"""
            Question: {question}
            
            Additional context:
            {input_data['additional_context']}
            
            Please provide a comprehensive answer based on the above information.
            """
            response = self.llm.invoke(prompt_text)
            return response.content, summarized_context

#-------------------------------------------------------------
# DOCUMENT QA MODULE
#-------------------------------------------------------------

# Initialize retrieval chain
@st.cache_resource(show_spinner=False)
def get_retrieval_chain(uploaded_file, model):
    with st.spinner("Processing document... This may take a minute."):
        # Configure embeddings
        genai.configure(api_key=gemini_key)
        embeddings = GeminiEmbeddings(api_key=gemini_key)
        
        # Read and process document
        docs = read_data_from_doc(uploaded_file)
        splits = make_chunks(docs)
        
        # Set up vector store
        pc = Pinecone(api_key=pinecone_key)
        
        # Check if index exists, create it if not
        indexes = pc.list_indexes()
        index_name = "research-rag"
        if index_name not in [idx.name for idx in indexes]:
            pc.create_index(
                name=index_name,
                dimension=768,  # Dimension for embeddings
                metric="cosine"
            )
            
        vectorstore = PineconeVectorStore.from_documents(
            splits,
            embeddings,
            index_name=index_name,
        )
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # Set up LLM and chain
        llm = ChatGroq(model_name=model, temperature=0.75, api_key=groq_key)
        
        system_prompt = """
        You are an AI assistant answering questions based on retrieved documents and additional context. 
        Use the provided context from both database retrieval and additional sources to answer the question. 

        - **Discard irrelevant context:** If one of the contexts (retrieved or additional) does not match the question, ignore it.
        - **Highlight conflicting information:** If multiple sources provide conflicting information, explicitly mention it by saying:
          - "According to the retrieved context, ... but as per internet sources, ..."
          - "According to the retrieved context, ... but as per internet sources, ..."
        - **Prioritize accuracy:** If neither context provides a relevant answer, say "I don't know" instead of guessing.

        Provide concise yet informative answers, ensuring clarity and completeness.

        Retrieved Context: {context}
        Additional Context: {additional_context}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}\n\nRetrieved Context: {context}\n\nAdditional Context: {additional_context}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return chain

#-------------------------------------------------------------
# WEB SEARCH MODULE
#-------------------------------------------------------------

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

# Main web search functions
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

#-------------------------------------------------------------
# MAIN APP
#-------------------------------------------------------------

def display_header():
    st.title("🔍 AI Research Assistant")
    st.markdown("Your all-in-one tool for research, document analysis, and web search")

def main():
    # App header
    display_header()
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        app_mode = st.radio("Choose a mode:", 
                          ["Research Assistant", "Document Q&A", "Web Search"])
        
        st.markdown("---")
        st.subheader("About")
        st.markdown("""
        This AI Research Assistant helps you find and analyze information from various sources:
        - arXiv papers
        - Wikipedia articles
        - Your own uploaded documents
        - Web search results
        """)
        
        # API keys status
        st.markdown("---")
        st.subheader("API Status")
        
        if groq_key:
            st.success("✅ Groq API connected")
        else:
            st.error("❌ Groq API key missing")
            
        if gemini_key:
            st.success("✅ Gemini API connected")
        else:
            st.error("❌ Gemini API key missing")
            
        if pinecone_key:
            st.success("✅ Pinecone API connected")
        else:
            st.error("❌ Pinecone API key missing")
    
    # Research Assistant Mode
    if app_mode == "Research Assistant":
        st.header("Research Assistant")
        st.markdown("Ask research questions and get answers from arXiv papers and Wikipedia.")
        
        # Initialize session state for chat history
        if "research_history" not in st.session_state:
            st.session_state.research_history = []
            
        # Initialize Research Assistant
        if "research_assistant" not in st.session_state:
            with st.spinner("Initializing Research Assistant..."):
                st.session_state.research_assistant = ResearchAssistant()
                
        # Input area
        with st.form(key="research_form"):
            question = st.text_input("Your research question:", key="research_question")
            submit_button = st.form_submit_button("Search")
            
        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.research_history = []
            st.rerun()
        
        # Process query when submitted
        if submit_button and question:
            # Add user query to chat history
            st.session_state.research_history.append({"role": "user", "content": question})
            
            # Get response from assistant
            answer, sources = st.session_state.research_assistant.chat(question)
            
            # Add assistant response to chat history
            st.session_state.research_history.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })
        
        # Display chat history
        for message in st.session_state.research_history:
            if message["role"] == "user":
                st.write(f"👤 **You:** {message['content']}")
            else:
                st.write(f"🤖 **AI Assistant:**")
                st.markdown(message["content"])
                
                # Display sources in expandable section
                if message.get("sources"):
                    with st.expander("View Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.markdown(source)
                            st.markdown("---")
    
    # Document Q&A Mode
    elif app_mode == "Document Q&A":
        st.header("Document Q&A")
        st.markdown("Upload a PDF document and ask questions about it.")
        
        # Model selection
        model_name = st.selectbox(
            "Select Groq Model",
            [
                "llama3-70b-8192",
                "gemma2-9b-it",
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama-guard-3-8b",
                "mixtral-8x7b-32768",
                "deepseek-r1-distill-llama-70b",
                "llama-3.2-1b-preview"
            ],
            index=0
        )
        
        # Initialize session state for conversation history
        if 'document_conversation' not in st.session_state:
            st.session_state.document_conversation = []
        
        # File upload
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        
        if uploaded_file:
            try:
                chain = get_retrieval_chain(
                    uploaded_file, 
                    model_name
                )
                
                # Show success message
                st.success("Document processed successfully! You can now ask questions.")
                
                # Display conversation history
                for q, a in st.session_state.document_conversation:
                    with st.chat_message("user"):
                        st.write(q)
                    with st.chat_message("assistant"):
                        st.write(a)
                
                # Question input
                question = st.chat_input("Ask a question about your document...")
                
                if question:
                    with st.chat_message("user"):
                        st.write(question)
                        
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            additional_context = ""  # Can be modified to add external context if needed
                            result = chain.invoke({
                                "input": question,
                                "additional_context": additional_context
                            })
                            answer = result['answer']
                            st.write(answer)
                    
                    # Store in conversation history
                    st.session_state.document_conversation.append((question, answer))
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                
        elif not (groq_key and gemini_key and pinecone_key):
            st.warning("Please make sure all API keys are properly configured.")
    
    # Web Search Mode
    else:
        st.header("Web Search")
        st.markdown("Search the web for answers to your questions.")
        
        # Input area
        with st.form("web_query_form"):
            query = st.text_area("Enter your research question", height=100, 
                               placeholder="E.g., What are the latest developments in quantum computing?")
            context = st.text_area("Optional: Add any context", height=100, 
                                   placeholder="Add any additional context that might help with the research")
            submit_button = st.form_submit_button("🔍 Research")
        
        if submit_button and query:
            result = process_query(query, context)
            
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

if __name__ == "__main__":
    main()