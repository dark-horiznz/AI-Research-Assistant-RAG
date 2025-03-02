import os
import streamlit as st
import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
# Get API keys from environment variables
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")
gemini_key = os.getenv("GEMINI_API_KEY")
pinecone_env_name = "reserach-rag"
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
import google.generativeai as genai
import numpy as np
from langchain_core.embeddings import Embeddings
from typing import List

# Set page configuration
st.set_page_config(page_title="Document Q&A with Groq", layout="wide")

# Sidebar for API keys
with st.sidebar:
    
    # Model selection
    st.title("Model Selection")
    model_name = st.selectbox(
        "Select Groq Model",
        [
            "gemma2-9b-it",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-guard-3-8b",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "deepseek-r1-distill-llama-70b",
            "llama-3.3-70b-specdec",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview"
        ],
        index=5
    )

# Main content
st.title("RAG-powered Document Assistant")
st.write("Upload a document and ask questions about it!")

# Gemini Embeddings implementation
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

# Function to read PDF
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

# Function to create chunks
def make_chunks(docs, chunk_len=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_len, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    return [Document(page_content=chunk.page_content, metadata=chunk.metadata) for chunk in chunks]

# Initialize session state for conversation history
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []

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
        pc = Pinecone(pinecone_key)
        vectorstore = PineconeVectorStore.from_documents(
            splits,
            embeddings,
            index_name="research-rag",
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
        for q, a in st.session_state['conversation']:
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
            st.session_state['conversation'].append((question, answer))
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
elif uploaded_file:
    st.warning("Please enter all required API keys in the sidebar.")