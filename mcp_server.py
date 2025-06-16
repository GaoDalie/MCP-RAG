import asyncio
from mcp.server.fastmcp import FastMCP
import rag
import search
import logging
import os
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

mcp = FastMCP(
    name="web_search", 
    version="1.0.0",
    description="Web search capability using Exa API , Firecrawl API  that provides real-time internet search results and use RAG to search for relevant data. Supports both basic and advanced search with filtering options including domain restrictions, text inclusion requirements, and date filtering. Returns formatted results with titles, URLs, publication dates, and content summaries."
)

@mcp.tool()
async def search_web_tool(query: str) -> str:
    logger.info(f"Searching web for query: {query}")
    formatted_results, raw_results = await search.search_web(query)
    
    if not raw_results:
        return "No search results found."
    
    urls = [result.url for result in raw_results if hasattr(result, 'url')]
    if not urls:
        return "No valid URLs found in search results."
        
    vectorstore = await rag.create_rag(urls)
    rag_results = await rag.search_rag(query, vectorstore)
    
    # You can optionally include the formatted search results in the output
    full_results = f"{formatted_results}\n\n### RAG Results:\n\n"
    full_results += '\n---\n'.join(doc.page_content for doc in rag_results)
    
    return full_results

@mcp.tool()
async def get_web_content_tool(url: str) -> str:
    try:
        documents = await asyncio.wait_for(search.get_web_content(url), timeout=15.0)
        if documents:
            return '\n\n'.join([doc.page_content for doc in documents])
        return "Unable to retrieve web content."
    except asyncio.TimeoutError:
        return "Timeout occurred while fetching web content. Please try again later."
    except Exception as e:
        return f"An error occurred while fetching web content: {str(e)}"
    

if __name__ == "__main__":
    mcp.run(transport="sse")

