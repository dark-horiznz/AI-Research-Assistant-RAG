import os
import streamlit as st
import requests
import pdfplumber
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.embeddings import Embeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
import arxiv
import wikipedia
from io import BytesIO
from typing import List, Dict

load_dotenv()

models = [
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
]

def read_data_from_doc(pdf_path: str) -> List[Document]:
    docs = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
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
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return docs

def make_chunks(docs: List[Document], chunk_len: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_len, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(docs)
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return []

class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        try:
            genai.configure(api_key=api_key)
            self.model_name = "models/embedding-001"
        except Exception as e:
            print(f"Error configuring Gemini API: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            return [self._convert_to_float32(genai.embed_content(model=self.model_name, content=text, task_type="retrieval_document")["embedding"]) for text in texts]
        except Exception as e:
            print(f"Error embedding documents: {e}")
            return []

    def embed_query(self, text: str) -> List[float]:
        try:
            response = genai.embed_content(model=self.model_name, content=text, task_type="retrieval_query")
            return self._convert_to_float32(response["embedding"])
        except Exception as e:
            print(f"Error embedding query: {e}")
            return []

    @staticmethod
    def _convert_to_float32(embedding: List[float]) -> List[float]:
        return np.array(embedding, dtype=np.float32).tolist()

def initialize_vectorstore(docs: List[Document]) -> PineconeVectorStore:
    try:
        embeddings = GeminiEmbeddings(api_key=os.environ["GEMINI_API_KEY"])
        pc = Pinecone(os.environ["PINECONE_API_KEY"])
        return PineconeVectorStore.from_documents(
            docs,
            embeddings,
            index_name=os.environ["PINECONE_ENV"]
        )
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None

system_prompt = """
You are an AI assistant answering questions based on retrieved documents and additional context. 
Use the provided context from both database retrieval and additional sources to answer the question. 

- **Discard irrelevant context:** If one of the contexts (retrieved or additional) does not match the question, ignore it.
- **Highlight conflicting information:** If multiple sources provide conflicting information, explicitly mention it by saying:
  - "According to the retrieved context, ... but as per internet sources, ..."
  - "According to the retrieved context, ... but as per additional context, ..."
- **Prioritize accuracy:** If neither context provides a relevant answer, say "I don't know" instead of guessing.

Provide concise yet informative answers, ensuring clarity and completeness.

Retrieved Context: {context}
Additional Context: {additional_context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}\n\nRetrieved Context: {context}\n\nAdditional Context: {additional_context}"),
    ]
)

def create_chat_pipeline(vectorstore: PineconeVectorStore, model_name: str) -> Dict:
    try:
        llm = ChatGroq(model_name=model_name, temperature=0.75, api_key=os.environ["GROQ_API_KEY"])
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        return create_retrieval_chain(retriever, question_answer_chain)
    except Exception as e:
        print(f"Error creating chat pipeline: {e}")
        return None

def chat_with_llm(chain, question: str, additional_context: str = "") -> str:
    try:
        input_data = {
            "input": question,
            "additional_context": additional_context  
        }
        return chain.invoke(input_data)
    except Exception as e:
        print(f"Error during chat with LLM: {e}")
        return ""

def main(pdf_path: str, question: str, model_name: str, additional_context: str = "") -> str:
    try:
        docs = read_data_from_doc(pdf_path)
        chunks = make_chunks(docs)
        vectorstore = initialize_vectorstore(chunks)
        if vectorstore is None:
            return "Error initializing vector store."
        chain = create_chat_pipeline(vectorstore, model_name)
        if chain is None:
            return "Error creating chat pipeline."
        return chat_with_llm(chain, question, additional_context)
    except Exception as e:
        print(f"Error in main function: {e}")
        return "An error occurred."

if __name__ == "__main__":
    pdf_path = "sample.pdf"
    question = "Tell me about the highest-rated Game of Thrones episodes"
    model_name = "llama-3.3-70b-versatile" 
    additional_context = ""

    result = main(pdf_path, question, model_name, additional_context)
    print(result)
