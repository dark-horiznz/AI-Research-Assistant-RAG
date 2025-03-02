import os
import requests
import pdfplumber
import arxiv
import wikipedia
import numpy as np
import streamlit as st
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain.schema import Document
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any, Optional

# Load environment variables from .env file
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV', 'research-rag')

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

def get_generation_config():
    return {
        'temperature': 0.4,
        'top_p': 0.95,
        'top_k': 40,
        'max_output_tokens': 4096,
    }

def get_safety_settings():
    return [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

def get_gemini_model(model_name="gemini-1.5-pro", temperature=0.4):
    return genai.GenerativeModel(model_name)

def generate_gemini_response(model, prompt):
    response = model.generate_content(
        prompt,
        generation_config=get_generation_config(),
        safety_settings=get_safety_settings()
    )
    if response.candidates and len(response.candidates) > 0:
        return response.candidates[0].content.parts[0].text
    return ''

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

def summarize_text(text):
    model = get_gemini_model()
    prompt_text = f"Summarize the following research paper very concisely:\n{text[:5000]}"  # Truncate to 5000 chars
    summary = generate_gemini_response(model, prompt_text)
    return summary

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
                with st.spinner(f"Retrieving Wikipedia article: {title}"):
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
    def __init__(self, embeddings_model="text-embedding-ada-002"):
        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Initialize vector store with the index
        index_name = "research-assistant"
        
        # Check if index exists, if not, create a simple placeholder embedding function
        class SimpleEmbeddings(Embeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                # Return a simple placeholder embedding for each text
                return [[0.0] * 1536 for _ in texts]
            
            def embed_query(self, text: str) -> List[float]:
                # Return a simple placeholder embedding
                return [0.0] * 1536
        
        try:
            # Try to get the index, if it doesn't exist, handle that case
            indexes = pc.list_indexes()
            if index_name not in [idx.name for idx in indexes]:
                st.warning(f"Warning: Index '{index_name}' not found. Using mock vector store.")
                self.vector_store = None
            else:
                self.vector_store = PineconeVectorStore(
                    index_name=index_name, 
                    embedding=SimpleEmbeddings()
                )
        except Exception as e:
            st.error(f"Error connecting to Pinecone: {e}")
            self.vector_store = None
        
        # Initialize LLM
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama3-70b-8192",
            temperature=0.2
        )
        
        # Create retriever if vector store is available
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 3}
        ) if self.vector_store else None
        
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
        
        # Set up the retrieval chain if retriever is available
        if self.retriever:
            self.chain = create_retrieval_chain(self.retriever, self.question_answer_chain)
        else:
            self.chain = None

    def retrieve_documents(self, query):
        user_context = []
        
        # Get documents from vector store if available
        if self.retriever:
            retrieved_docs = self.retriever.get_relevant_documents(query)
            for doc in retrieved_docs:
                user_context.append(f"**{doc.metadata.get('source', 'Unknown Source')}**: {doc.page_content}...")
        
        # Get documents from arXiv and Wikipedia
        with st.spinner("Searching arXiv papers..."):
            arxiv_docs = search_arxiv(query)
        with st.spinner("Searching Wikipedia articles..."):
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
            if self.chain:
                result = self.chain.invoke(input_data)
                return result["answer"], summarized_context
            else:
                # If chain isn't available, just use the LLM directly
                prompt_text = f"""
                Question: {question}
                
                Additional context:
                {input_data['additional_context']}
                
                Please provide a comprehensive answer based on the above information.
                """
                response = self.llm.invoke(prompt_text)
                return response.content, summarized_context

def main():
    st.set_page_config(
        page_title="AI Research Assistant",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔍 AI Research Assistant")
    st.markdown("Ask me any research question, and I'll search for relevant information.")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # Initialize Research Assistant
    if "assistant" not in st.session_state:
        with st.spinner("Initializing Research Assistant..."):
            st.session_state.assistant = ResearchAssistant()
            
    # Input area
    with st.form(key="query_form"):
        question = st.text_input("Your research question:", key="question_input")
        submit_button = st.form_submit_button("Search")
        
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Process query when submitted
    if submit_button and question:
        # Add user query to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Get response from assistant
        answer, sources = st.session_state.assistant.chat(question)
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources
        })
    
    # Display chat history
    for message in st.session_state.chat_history:
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
                        st.markdown(source[:500] + "..." if len(source) > 500 else source)
                        st.markdown("---")
    
    # Additional info in sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This AI Research Assistant helps you find and analyze information from various sources:
        - arXiv papers
        - Wikipedia articles
        - Your own uploaded documents (coming soon)
        """)
        
        st.header("Tips")
        st.markdown("""
        - Ask specific questions for better results
        - Include keywords related to your research area
        - Check the sources for more detailed information
        """)

if __name__ == "__main__":
    main()
# from langchain_core.embeddings import Embeddingsho
# from langchain.schema import Document
# from pinecone import Pinecone
# from dotenv import load_dotenv
# from typing import List, Dict, Tuple, Any, Optional

# # Load environment variables from .env file
# load_dotenv()

# # API Keys
# GROQ_API_KEY = os.getenv('GROQ_API_KEY')
# GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# PINECONE_ENV = os.getenv('PINECONE_ENV', 'research-rag')

# # Configure Gemini
# genai.configure(api_key=GEMINI_API_KEY)

# def get_generation_config():
#     return {
#         'temperature': 0.4,
#         'top_p': 0.95,
#         'top_k': 40,
#         'max_output_tokens': 4096,
#     }

# def get_safety_settings():
#     return [
#         {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#         {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#         {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#         {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
#     ]

# def get_gemini_model(model_name="gemini-pro", temperature=0.4):
#     return genai.GenerativeModel(model_name)

# def generate_gemini_response(model, prompt):
#     response = model.generate_content(
#         prompt,
#         generation_config=get_generation_config(),
#         safety_settings=get_safety_settings()
#     )
#     if response.candidates and len(response.candidates) > 0:
#         return response.candidates[0].content.parts[0].text
#     return ''

# def download_pdf(pdf_url, save_path="temp_paper.pdf"):
#     try:
#         response = requests.get(pdf_url)
#         if response.status_code == 200:
#             with open(save_path, "wb") as file:
#                 file.write(response.content)
#             return save_path
#     except Exception as e:
#         print(f"Error downloading PDF: {e}")
#     return None

# def extract_text_from_pdf(pdf_path):
#     text = ""
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             for page in pdf.pages:
#                 extracted_text = page.extract_text()
#                 if extracted_text:
#                     text += extracted_text + "\n"
#         return text.strip()
#     except Exception as e:
#         print(f"Error extracting text from PDF: {e}")
#         return ""

# def summarize_text(text):
#     model = get_gemini_model()
#     prompt_text = f"Summarize the following research paper very concisely:\n{text[:5000]}"  # Truncate to 5000 chars
#     summary = generate_gemini_response(model, prompt_text)
#     return summary

# def search_arxiv(query, max_results=2):
#     client = arxiv.Client()
#     search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    
#     arxiv_docs = []
    
#     for result in client.results(search):
#         pdf_link = next((link.href for link in result.links if 'pdf' in link.href), None)
        
#         # Download, extract, and summarize PDF if link exists
#         if pdf_link:
#             pdf_path = download_pdf(pdf_link)
#             if pdf_path:
#                 text = extract_text_from_pdf(pdf_path)
#                 summary = summarize_text(text)
#                 # Clean up downloaded file
#                 if os.path.exists(pdf_path):
#                     os.remove(pdf_path)
#             else:
#                 summary = "PDF could not be downloaded."
#         else:
#             summary = "No PDF available."

#         content = f"""
#         **Title:** {result.title}
#         **Authors:** {', '.join(author.name for author in result.authors)}
#         **Published:** {result.published.strftime('%Y-%m-%d')}
#         **Abstract:** {result.summary}
#         **PDF Summary:** {summary}
#         **PDF Link:** {pdf_link if pdf_link else 'Not available'}
#         """

#         arxiv_docs.append(Document(page_content=content, metadata={"source": "arXiv", "title": result.title}))
    
#     return arxiv_docs

# def search_wikipedia(query, max_results=2):
#     try:
#         page_titles = wikipedia.search(query, results=max_results)
#         wiki_docs = []
#         for title in page_titles:
#             try:
#                 page = wikipedia.page(title)
#                 wiki_docs.append(Document(
#                     page_content=page.content[:2000], 
#                     metadata={"source": "Wikipedia", "title": title}
#                 ))
#             except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
#                 print(f"Error retrieving Wikipedia page {title}: {e}")
#         return wiki_docs
#     except Exception as e:
#         print(f"Error searching Wikipedia: {e}")
#         return []

# class ResearchAssistant:
#     def __init__(self, embeddings_model="text-embedding-ada-002"):
#         # Initialize Pinecone
#         pc = Pinecone(api_key=PINECONE_API_KEY)
        
#         # Initialize vector store with the index
#         index_name = "research-assistant"
        
#         # Check if index exists, if not, create a simple placeholder embedding function
#         class SimpleEmbeddings(Embeddings):
#             def embed_documents(self, texts: List[str]) -> List[List[float]]:
#                 # Return a simple placeholder embedding for each text
#                 return [[0.0] * 1536 for _ in texts]
            
#             def embed_query(self, text: str) -> List[float]:
#                 # Return a simple placeholder embedding
#                 return [0.0] * 1536
        
#         try:
#             # Try to get the index, if it doesn't exist, handle that case
#             indexes = pc.list_indexes()
#             if index_name not in [idx.name for idx in indexes]:
#                 print(f"Warning: Index '{index_name}' not found. Using mock vector store.")
#                 self.vector_store = None
#             else:
#                 self.vector_store = PineconeVectorStore(
#                     index_name=index_name, 
#                     embedding=SimpleEmbeddings()
#                 )
#         except Exception as e:
#             print(f"Error connecting to Pinecone: {e}")
#             self.vector_store = None
        
#         # Initialize LLM
#         self.llm = ChatGroq(
#             api_key=GROQ_API_KEY,
#             model="llama3-70b-8192",
#             temperature=0.2
#         )
        
#         # Create retriever if vector store is available
#         self.retriever = self.vector_store.as_retriever(
#             search_kwargs={"k": 3}
#         ) if self.vector_store else None
        
#         # Set up the prompt template
#         self.prompt = ChatPromptTemplate.from_template("""
#         You are an expert research assistant. Use the following context to answer the question. 
#         If you don't know the answer, say so, but try your best to find relevant information 
#         from the provided context and additional context.
        
#         Context from user documents:
#         {context}
        
#         Additional context from research sources:
#         {additional_context}
        
#         Question: {input}
        
#         Answer:
#         """)
        
#         # Set up the question-answer chain
#         self.question_answer_chain = create_stuff_documents_chain(
#             self.llm, self.prompt
#         )
        
#         # Set up the retrieval chain if retriever is available
#         if self.retriever:
#             self.chain = create_retrieval_chain(self.retriever, self.question_answer_chain)
#         else:
#             self.chain = None

#     def retrieve_documents(self, query):
#         user_context = []
        
#         # Get documents from vector store if available
#         if self.retriever:
#             retrieved_docs = self.retriever.get_relevant_documents(query)
#             for doc in retrieved_docs:
#                 user_context.append(f"**{doc.metadata.get('source', 'Unknown Source')}**: {doc.page_content}...")
        
#         # Get documents from arXiv and Wikipedia
#         arxiv_docs = search_arxiv(query)
#         wiki_docs = search_wikipedia(query)
        
#         summarized_context = []
#         for doc in arxiv_docs:
#             summarized_context.append(f"**ArXiv - {doc.metadata.get('title', 'Unknown Title')}**:\n{doc.page_content}...")
            
#         for doc in wiki_docs:
#             summarized_context.append(f"**Wikipedia - {doc.metadata.get('title', 'Unknown Title')}**:\n{doc.page_content}...")
            
#         return user_context, summarized_context
    
#     def chat(self, question):
#         user_context, summarized_context = self.retrieve_documents(question)
        
#         input_data = {
#             "input": question,
#             "context": "\n\n".join(user_context),
#             "additional_context": "\n\n".join(summarized_context)
#         }
        
#         if self.chain:
#             result = self.chain.invoke(input_data)
#             return result["answer"], summarized_context
#         else:
#             # If chain isn't available, just use the LLM directly
#             prompt_text = f"""
#             Question: {question}
            
#             Additional context:
#             {input_data['additional_context']}
            
#             Please provide a comprehensive answer based on the above information.
#             """
#             response = self.llm.invoke(prompt_text)
#             return response.content, summarized_context

# def main():
#     print("Initializing Research Assistant...")
#     assistant = ResearchAssistant()
    
#     print("\nWelcome to the AI Research Assistant!")
#     print("Ask me any research question, and I'll search for relevant information.")
#     print("Type 'exit' to quit.\n")
    
#     while True:
#         question = input("\nYour question: ")
#         if question.lower() in ['exit', 'quit']:
#             break
            
#         print("\nSearching for information...")
#         answer, sources = assistant.chat(question)
        
#         print("\n=== Answer ===")
#         print(answer)
        
#         print("\n=== Sources ===")
#         for i, source in enumerate(sources, 1):
#             print(f"\n[Source {i}]")
#             print(source[:300] + "..." if len(source) > 300 else source)
        
#         print("\n" + "-"*50)

# if __name__ == "__main__":
#     main()