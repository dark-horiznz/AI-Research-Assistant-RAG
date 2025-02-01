import os
import pdfplumber
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List, Dict, Tuple

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.75)
genai.configure(api_key=GEMINI_API_KEY)

# Web search tool
search_tool = DuckDuckGoSearchRun()

def get_gemini_model(model_name: str = "gemini-pro") -> genai.GenerativeModel:
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        print(f"Error getting Gemini model: {e}")
        return None

def create_search_prompt(query: str, context: str = "") -> str:
    system_prompt = """You are a smart assistant designed to determine whether a query needs data from a web search or can be answered using a document database. 
    If the query requires external information, contains no context, has outdated context, or is best answered with real-time data, return "<SEARCH>" followed by optimized search terms. 
    Otherwise, return an empty response."""
    
    return f"{system_prompt}\n\nQuery: {query}\nContext: {context}"

def check_search_needed(model: genai.GenerativeModel, query: str, context: str) -> Tuple[bool, str]:
    try:
        response = model.generate_content(create_search_prompt(query, context))
        if "<SEARCH>" in response.text:
            return True, response.text.split("<SEARCH>")[1].strip()
    except Exception as e:
        print(f"Error checking if search is needed: {e}")
    return False, ""

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

def make_chunks(docs: List[Document], chunk_len=1000, chunk_overlap=200) -> List[Document]:
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_len, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(docs)
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return []

def create_vector_store(docs: List[Document]) -> PineconeVectorStore:
    try:
        return PineconeVectorStore.from_documents(
            docs,
            embeddings,
            index_name=PINECONE_ENV
        )
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

system_prompt = """
You are an AI assistant answering questions based on retrieved documents and additional context. 
Use the provided context from both database retrieval and additional sources to answer the question. 

- Discard irrelevant context: If one of the contexts (retrieved or additional) does not match the question, ignore it.
- Highlight conflicting information: If multiple sources provide conflicting information, explicitly mention it.
- Prioritize accuracy: If neither context provides a relevant answer, say "I don't know" instead of guessing.

Retrieved Context: {context}
Additional Context: {additional_context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}\n\nRetrieved Context: {context}\n\nAdditional Context: {additional_context}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

def create_chat_pipeline(vectorstore: PineconeVectorStore):
    try:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
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
        return "An error occurred while processing your request."

def main(pdf_path, question, additional_context=""):
    try:
        docs = read_data_from_doc(pdf_path)
        chunks = make_chunks(docs)
        vectorstore = create_vector_store(chunks)
        if vectorstore is None:
            return "Failed to create vector store."
        chain = create_chat_pipeline(vectorstore)
        if chain is None:
            return "Failed to create chat pipeline."

        # Check if web search is needed
        model = get_gemini_model()
        needs_search, search_terms = check_search_needed(model, question, additional_context)

        if needs_search:
            print(f"Performing web search for: {search_terms}")
            search_results = search_tool.run(search_terms)
            return chat_with_llm(chain, question, search_results)

        return chat_with_llm(chain, question, additional_context)
    except Exception as e:
        print(f"Error in main function: {e}")
        return "An error occurred while processing your request."

if __name__ == "__main__":
    pdf_path = "sample.pdf"
    question = "What are the key insights from this document?"
    additional_context = "This document covers financial reports for Q3."

    result = main(pdf_path, question, additional_context)
    print(result)
