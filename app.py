import os
import tempfile
import streamlit as st
import pdfplumber
import arxiv
import google.generativeai as genai
import numpy as np
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from typing import List, Dict
import requests
from io import BytesIO

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
search_tool = DuckDuckGoSearchRun()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

INDEX_NAMES = {
    "openai": "rag",
    "groq": "gemini-rag",
    "research": "research-rag"
}

GROQ_MODELS = [
    "gemma2-9b-it", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", 
    "mixtral-8x7b-32768", "deepseek-r1-distill-llama-70b"
]

class GeminiEmbeddings:
    def __init__(self):
        self.model_name = "models/embedding-001"
        self._dimension = 768  
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of documents."""
        try:
            return [self._embed_text(text) for text in texts]
        except Exception as e:
            st.error(f"Embedding error: {str(e)}")
            return []
    
    def embed_query(self, text: str) -> List[float]:
        """Create embeddings for a query string."""
        return self._embed_text(text)
    
    def _embed_text(self, text: str) -> List[float]:
        """Helper function to embed a single text."""
        try:
            response = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            embedding = response["embedding"]
            return np.array(embedding, dtype=np.float32).tolist()
        except Exception as e:
            st.error(f"Embedding generation error: {str(e)}")
            return [0.0] * self._dimension

class ResearchEngine:
    @staticmethod
    def _download_and_process_pdf(pdf_url: str, metadata: dict = None) -> List[Document]:
        """Download and process a PDF from a URL."""
        try:
            response = requests.get(pdf_url, timeout=30)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(response.content)
                    docs = DocumentProcessor.process_pdf(tmp.name)
                    if metadata:
                        for doc in docs:
                            doc.metadata.update(metadata)
                    os.unlink(tmp.name)
                    return docs
            return []
        except Exception as e:
            st.warning(f"Error processing PDF from {pdf_url}: {str(e)}")
            return []

    @staticmethod
    def fetch_and_process_arxiv_papers(query: str) -> List[Document]:
        """Fetch and process papers from arXiv."""
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=2,
                sort_by=arxiv.SortCriterion.Relevance
            )
            documents = []
            
            for result in client.results(search):
                try:
                    metadata = {
                        "title": result.title,
                        "authors": ", ".join(a.name for a in result.authors),
                        "published": result.published.strftime('%Y-%m-%d'),
                        "url": result.pdf_url,
                        "source": "arXiv",
                        "abstract": result.summary
                    }
                    docs = ResearchEngine._download_and_process_pdf(result.pdf_url, metadata)
                    documents.extend(docs)
                except Exception as e:
                    st.warning(f"Error processing paper {result.title}: {str(e)}")
                    continue
            
            return documents
        except Exception as e:
            st.error(f"arXiv error: {str(e)}")
            return []

    @staticmethod
    def process_pdf_links(pdf_links: List[str], titles: List[str] = None) -> List[Document]:
        """Process a list of PDF links directly."""
        documents = []
        for i, pdf_url in enumerate(pdf_links):
            try:
                metadata = {
                    "title": titles[i] if titles and i < len(titles) else f"Paper {i+1}",
                    "url": pdf_url,
                    "source": "Custom PDF",
                }
                docs = ResearchEngine._download_and_process_pdf(pdf_url, metadata)
                documents.extend(docs)
            except Exception as e:
                st.warning(f"Error processing PDF from {pdf_url}: {str(e)}")
                continue
        return documents

    @staticmethod
    def research_chain(question: str, model_name: str, mode: str = "arxiv", pdf_links: List[str] = None, titles: List[str] = None) -> str:
        """Enhanced research chain with multiple modes."""
        try:
            if mode == "arxiv":
                docs = ResearchEngine.fetch_and_process_arxiv_papers(question)
            elif mode == "custom_pdfs" and pdf_links:
                docs = ResearchEngine.process_pdf_links(pdf_links, titles)
            else:
                return "Invalid research mode or missing PDF links"

            if not docs:
                return "No relevant documents found or could not process PDFs."
            
            embeddings = GeminiEmbeddings()
            vectorstore = VectorStoreManager.get_vectorstore(docs, embeddings, INDEX_NAMES["research"])
            if not vectorstore:
                return "Error: Could not process documents"
            
            llm = ChatGroq(model_name=model_name)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            prompt = ChatPromptTemplate.from_template("""
                Based on the following research documents:
                {context}
                
                Question: {input}
                
                Provide a comprehensive analysis with specific citations to the source papers.
                For each point, mention which paper it comes from using the title or number.
                Include relevant quotes where appropriate.
                
                Structure your response as follows:
                1. Main findings
                2. Supporting evidence
                3. Relevant quotes
                4. Sources used
            """)
            
            chain = create_retrieval_chain(
                retriever,
                create_stuff_documents_chain(llm, prompt)
            )
            result = chain.invoke({"input": question})
            return result["answer"]
        except Exception as e:
            return f"Research Error: {str(e)}"
class DocumentProcessor:
    @staticmethod
    def process_pdf(pdf_path: str) -> List[Document]:
        """Process a PDF file and return a list of Document objects."""
        if not pdf_path:
            return []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                docs = []
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        docs.append(Document(
                            page_content=text.strip(),
                            metadata={
                                "page": i + 1,
                                "source": pdf_path,
                                "type": "pdf"
                            }
                        ))
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                return text_splitter.split_documents(docs)
        except Exception as e:
            st.error(f"PDF processing error: {str(e)}")
            return []

class VectorStoreManager:
    @staticmethod
    def get_vectorstore(docs: List[Document], embeddings, index_name: str) -> PineconeVectorStore:
        """Create or get a vector store for the given documents."""
        try:
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=768,  
                    metric="cosine"
                )
            
            return PineconeVectorStore.from_documents(
                documents=docs,
                embedding=embeddings,
                index_name=index_name
            )
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None

    @staticmethod
    def clear_index(index_name: str):
        """Clear all vectors from the specified index."""
        try:
            if index_name in pc.list_indexes().names():
                index = pc.Index(index_name)
                index.delete(delete_all=True)
                st.success(f"Successfully cleared {index_name} index")
            else:
                st.warning(f"Index {index_name} does not exist")
        except Exception as e:
            st.error(f"Error clearing index: {str(e)}")

class AIChains:
    @staticmethod
    def openai_chain(question: str, context: str = "", pdf_path: str = None) -> str:
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
            embeddings = OpenAIEmbeddings()
            
            if pdf_path:
                docs = DocumentProcessor.process_pdf(pdf_path)
                vectorstore = VectorStoreManager.get_vectorstore(docs, embeddings, INDEX_NAMES["openai"])
                if not vectorstore:
                    return "Error: Could not process document"
                    
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                prompt = ChatPromptTemplate.from_template("""
                    Context: {context}
                    Additional Info: {additional_context}
                    Question: {input}
                    Provide a detailed answer with citations:
                """)
                
                chain = create_retrieval_chain(
                    retriever, 
                    create_stuff_documents_chain(llm, prompt)
                )
                result = chain.invoke({
                    "input": question,
                    "additional_context": context
                })
                return result["answer"]
            
            return llm.invoke(f"{context}\nQuestion: {question}").content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"

    @staticmethod
    def groq_chain(question: str, model_name: str, context: str = "", pdf_path: str = None) -> str:
        try:
            llm = ChatGroq(model_name=model_name)
            embeddings = GeminiEmbeddings()
            
            if pdf_path:
                docs = DocumentProcessor.process_pdf(pdf_path)
                vectorstore = VectorStoreManager.get_vectorstore(docs, embeddings, INDEX_NAMES["groq"])
                if not vectorstore:
                    return "Error: Could not process document"
                
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                prompt = ChatPromptTemplate.from_template("""
                    Context: {context}
                    Additional Info: {additional_context}
                    Question: {input}
                    Provide a detailed answer with citations:
                """)
                
                chain = create_retrieval_chain(
                    retriever,
                    create_stuff_documents_chain(llm, prompt)
                )
                result = chain.invoke({
                    "input": question,
                    "additional_context": context
                })
                return result["answer"]
            
            return llm.invoke(f"{context}\nQuestion: {question}").content
        except Exception as e:
            return f"Groq Error: {str(e)}"

    @staticmethod
    def research_chain(question: str, model_name: str, mode: str = "arxiv", pdf_links: List[str] = None, titles: List[str] = None) -> str:
        try:
            if mode == "arxiv":
                docs = ResearchEngine.fetch_and_process_arxiv_papers(question)
            elif mode == "custom_pdfs" and pdf_links:
                docs = ResearchEngine.process_pdf_links(pdf_links, titles)
            else:
                return "Invalid research mode or missing PDF links"

            if not docs:
                return "No relevant documents found."
            
            embeddings = GeminiEmbeddings()
            vectorstore = VectorStoreManager.get_vectorstore(
                docs, 
                embeddings, 
                INDEX_NAMES["research"]
            )
            if not vectorstore:
                return "Error: Could not process research papers"
            
            llm = ChatGroq(model_name=model_name)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            prompt = ChatPromptTemplate.from_template("""
                Based on the following research papers:
                {context}
                
                Question: {input}
                
                Provide a detailed analysis with specific citations:
            """)
            
            chain = create_retrieval_chain(
                retriever,
                create_stuff_documents_chain(llm, prompt)
            )
            result = chain.invoke({"input": question})
            return result["answer"]
        except Exception as e:
            return f"Research Error: {str(e)}"

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Base styles */
    :root {
        --primary-color: #7c3aed;
        --secondary-color: #4f46e5;
        --background-color: #f9fafb;
        --text-color: #111827;
    }

    /* Main container */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
    }

    /* Chat messages */
    .stChatMessage {
        background-color: white;
        border-radius: 1rem;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        animation: slideIn 0.3s ease-out;
    }

    /* User message specific */
    .user-message {
        background-color: #f3f4f6;
        margin-left: auto;
        max-width: 80%;
    }

    /* Assistant message specific */
    .assistant-message {
        background-color: white;
        margin-right: auto;
        max-width: 80%;
    }

    /* Input container */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(to right, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }

    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: white;
    }

    /* Animations */
    @keyframes slideIn {
        from {
            transform: translateY(20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }

    /* Chat container */
    .chat-container {
        margin-bottom: 120px;
        padding: 1rem;
    }

    /* Search container */
    .search-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 1000;
    }

    /* Input field */
    .stTextInput input {
        border-radius: 0.5rem;
        border: 2px solid #e5e7eb;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stTextInput input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(124,58,237,0.2);
    }

    /* Helper buttons */
    .helper-buttons {
        display: flex;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }

    .helper-button {
        background-color: #f3f4f6;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-size: 0.875rem;
        color: #4b5563;
        cursor: pointer;
        transition: all 0.2s ease;
    }

    .helper-button:hover {
        background-color: #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

with st.sidebar:
    st.title("🤖 AI Research Assistant")
    
    st.subheader("💬 Chat Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Chat"):
            st.session_state.messages = []
    with col2:
        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
    
    st.divider()
    
    st.subheader("🛠️ Model Settings")
    model_choice = st.selectbox("Select Model", ["OpenAI", "Groq"], key="model_choice")
    
    if model_choice == "Groq":
        groq_model = st.selectbox("Model Version", GROQ_MODELS)
    
    st.subheader("📊 Database Controls")
    selected_index = st.selectbox("Select Index", list(INDEX_NAMES.values()))
    if st.button("🗑️ Clear Selected Index"):
        VectorStoreManager.clear_index(selected_index)
    
    st.subheader("📄 Document Upload")
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.getvalue())
            st.session_state.pdf_path = tmp.name
            st.success("✅ PDF uploaded successfully!")

st.header("AI Research Assistant", divider="rainbow")

with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources"):
                    st.write(message["sources"])
    st.markdown('</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    prompt = st.text_input("Ask me anything...", key="chat_input", 
                          placeholder="Type your message here...")
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        web_search = st.button("🌐 Web")
    with col2:
        research_search = st.button("📚 Research")
    
    if prompt and (web_search or research_search or st.session_state.get("chat_input")):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    context = ""
                    sources = {}
                    
                    if web_search:
                        with st.spinner("🌐 Searching the web..."):
                            web_results = search_tool.run(prompt)
                            context += f"Web Search Results:\n{web_results}\n"
                            sources["Web"] = web_results
                    
                    if research_search:
                        with st.spinner("📚 Analyzing research papers..."):
                            if model_choice == "Groq":
                                research_response = AIChains.research_chain(prompt, groq_model)
                                sources["Research"] = research_response
                            else:
                                st.warning("ℹ️ Research mode is only available with Groq models")
                                research_response = ""
                            context += f"\nResearch Context:\n{research_response}\n"
                    
                    if not (web_search or research_search):
                        with st.spinner("💭 Generating response..."):
                            if model_choice == "OpenAI":
                                response = AIChains.openai_chain(
                                    question=prompt,
                                    context=context,
                                    pdf_path=st.session_state.pdf_path
                                )
                            else:  
                                response = AIChains.groq_chain(
                                    question=prompt,
                                    model_name=groq_model,
                                    context=context,
                                    pdf_path=st.session_state.pdf_path
                                )
                    else:
                        response = context
                    
                    st.markdown(response)
                    
                    if sources:
                        with st.expander("📚 View Sources"):
                            for source_type, content in sources.items():
                                st.subheader(f"{source_type} Sources")
                                st.markdown(content)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "sources": sources if sources else None
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
st.sidebar.subheader("📚 Research Mode")
research_mode = st.sidebar.radio(
    "Select Research Mode",
    ["arXiv", "Custom PDFs"]
)

if research_mode == "Custom PDFs":
    pdf_links = st.sidebar.text_area(
        "Enter PDF URLs (one per line)",
        placeholder="https://example.com/paper1.pdf\nhttps://example.com/paper2.pdf"
    )
    pdf_titles = st.sidebar.text_area(
        "Enter Paper Titles (one per line)",
        placeholder="Paper 1 Title\nPaper 2 Title"
    )
    
    pdf_links_list = [url.strip() for url in pdf_links.split('\n') if url.strip()] if pdf_links else []
    pdf_titles_list = [title.strip() for title in pdf_titles.split('\n') if title.strip()] if pdf_titles else []

if research_search:
    with st.spinner("📚 Analyzing research papers..."):
        if model_choice == "Groq":
            if research_mode == "Custom PDFs" and pdf_links_list:
                research_response = AIChains.research_chain(
                    prompt, 
                    groq_model,
                    mode="custom_pdfs",
                    pdf_links=pdf_links_list,
                    titles=pdf_titles_list
                )
            else:
                research_response = AIChains.research_chain(
                    prompt, 
                    groq_model,
                    mode="arxiv"
                )
            sources["Research"] = research_response
        else:
            st.warning("ℹ️ Research mode is only available with Groq models")
            research_response = ""
        context += f"\nResearch Context:\n{research_response}\n"

if st.session_state.pdf_path and not pdf_file:
    try:
        os.unlink(st.session_state.pdf_path)
        st.session_state.pdf_path = None
    except Exception as e:
        st.error(f"Error cleaning up temporary files: {str(e)}")

st.markdown("""
<div style='position: fixed; bottom: 150px; left: 0; right: 0; text-align: center; padding: 10px; font-size: 0.8em; color: #666;'>
    Made with ❤️ by Aditya Singh
</div>
""", unsafe_allow_html=True)
