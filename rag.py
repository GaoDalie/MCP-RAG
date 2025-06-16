from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import search
from langchain_core.documents import Document
import os
import asyncio
from langchain_ollama import OllamaEmbeddings
from langchain_mistralai import MistralAIEmbeddings

async def create_rag(links: list[str]) -> FAISS:
    try:
        model_name = os.getenv("MODEL", "your name model") 
        # Change any embedding you want, whether Ollama or MistralAIEmbeddings
        embeddings = MistralAIEmbeddings(
            model=model_name,
            chunk_size=64
        )
        # embeddings = OpenAIEmbeddings(
        #     model=model_name,
        #     openai_api_key=os.getenv("OPENAI_API_KEY"),
        #     openai_api_base=os.getenv("OPENAI_API_BASE"),
        #     chunk_size=64
        # )
        documents = []
        # Use asyncio.gather to process all URL requests in parallel
        tasks = [search.get_web_content(url) for url in links]
        results = await asyncio.gather(*tasks)
        for result in results:
            documents.extend(result)
        
        # Text chunking processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500,
            length_function=len,
            is_separator_regex=False,
        )
        split_documents = text_splitter.split_documents(documents)
        # print(documents)
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error in create_rag: {str(e)}")
        raise

async def create_rag_from_documents(documents: list[Document]) -> FAISS:
    """
    Create a RAG system directly from a list of documents to avoid repeated web scraping
    
    Args:
        documents: List of already fetched documents
        
    Returns:
        FAISS: Vector store object
    """
    try:
        model_name = os.getenv("MODEL")
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            chunk_size=64)
        
        # embeddings = OpenAIEmbeddings(
        #     model=model_name,
        #     openai_api_key=os.getenv("OPENAI_API_KEY"),
        #     openai_api_base=os.getenv("OPENAI_API_BASE"),
        #     chunk_size=64
        # )
        
        # Text chunking processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=500,
            length_function=len,
            is_separator_regex=False,
        )
        split_documents = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error in create_rag_from_documents: {str(e)}")
        raise

async def search_rag(query: str, vectorstore: FAISS) -> list[Document]:
    """
    Search the RAG system with a query
    
    Args:
        query: Search query string
        vectorstore: FAISS vector store to search against
        
    Returns:
        list[Document]: List of relevant documents
    """
    return vectorstore.similarity_search(query, k=3)