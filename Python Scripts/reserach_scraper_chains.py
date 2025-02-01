import os
import requests
import pdfplumber
from langchain_groq import ChatGroq
import google.generativeai as genai
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_core.embeddings import Embeddings
from langchain.schema import Document
from pinecone import Pinecone
import numpy as np
import arxiv
import wikipedia
from dotenv import load_dotenv

load_dotenv()

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    raise ValueError("GEMINI_API_KEY is missing from environment variables.")

def get_gemini_model(model_name="gemini-pro", temperature=0.4):
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        raise RuntimeError(f"Error initializing Gemini model: {e}")

def generate_gemini_response(model, prompt):
    try:
        response = model.generate_content(
            prompt,
            generation_config=get_generation_config(),
            safety_settings=get_safety_settings()
        )
        if response.candidates and len(response.candidates) > 0:
            return response.candidates[0].content.parts[0].text
        return ''
    except Exception as e:
        raise RuntimeError(f"Error generating Gemini response: {e}")

def download_pdf(pdf_url, save_path="temp_paper.pdf"):
    try:
        response = requests.get(pdf_url, timeout=10)
        response.raise_for_status()
        with open(save_path, "wb") as file:
            file.write(response.content)
        return save_path
    except requests.RequestException as e:
        raise RuntimeError(f"Error downloading PDF: {e}")

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Error extracting text from PDF: {e}")

def summarize_text(text):
    try:
        model = get_gemini_model()
        prompt_text = f"Summarize the following research paper very concisely:\n{text[:5000]}"
        return generate_gemini_response(model, prompt_text)
    except Exception as e:
        raise RuntimeError(f"Error summarizing text: {e}")

def search_arxiv(query, max_results=2):
    try:
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
        arxiv_docs = []
        for result in client.results(search):
            pdf_link = next((link.href for link in result.links if 'pdf' in link.href), None)
            summary = "No PDF available."
            if pdf_link:
                pdf_path = download_pdf(pdf_link)
                if pdf_path:
                    text = extract_text_from_pdf(pdf_path)
                    summary = summarize_text(text)
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
    except Exception as e:
        raise RuntimeError(f"Error searching arXiv: {e}")

def search_wikipedia(query, max_results=2):
    try:
        page_titles = wikipedia.search(query, results=max_results)
        return [Document(page_content=wikipedia.page(title).content[:2000], metadata={"source": "Wikipedia", "title": title}) for title in page_titles]
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
        return []
    except Exception as e:
        raise RuntimeError(f"Error searching Wikipedia: {e}")

def retrieve_documents(query):
    try:
        retrieved_docs = retriever.get_relevant_documents(query)
        arxiv_docs = search_arxiv(query)
        wiki_docs = search_wikipedia(query)
        summarized_context = []
        user_context = []
        for doc in retrieved_docs:
            user_context.append(f"**{doc.metadata.get('source', 'Unknown Source')}**: {doc.page_content}...")
        for doc in arxiv_docs:
            summarized_context.append(f"**ArXiv - {doc.metadata.get('title', 'Unknown Title')}**:\n{doc.page_content}...")
        for doc in wiki_docs:
            summarized_context.append(f"**Wikipedia - {doc.metadata.get('title', 'Unknown Title')}**:\n{doc.page_content}...")
        return user_context , summarized_context
    except Exception as e:
        raise RuntimeError(f"Error retrieving documents: {e}")

try:
    chain = create_retrieval_chain(retriever, question_answer_chain)
except Exception as e:
    raise RuntimeError(f"Error initializing retrieval chain: {e}")

def chat_with_llm(question):
    try:
        user_context , summarized_context = retrieve_documents(question)
        input_data = {
            "input": question,
            "context": "\n\n".join(user_context),  
            "additional_context": "\n\n".join(summarized_context)
        }
        return chain.invoke(input_data)
    except Exception as e:
        raise RuntimeError(f"Error during chat execution: {e}")

def main():
    try:
        out = chat_with_llm('Black hole mergers')
        print(out['answer'])
    except Exception as e:
        print(f"Execution failed: {e}")

if __name__ == "__main__":
    main()
