import asyncio
from dotenv import load_dotenv
import os
from exa_py import Exa
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.document_loaders.firecrawl import FireCrawlLoader
import requests
# Load .env variables
load_dotenv(override=True)

# Initialize the Exa client
exa_api_key = os.getenv("EXA_API_KEY", "your_api")
exa = Exa(api_key=exa_api_key)
os.environ['FIRECRAWL_API_KEY'] = 'your_api'
# Default search config
websearch_config = {
    "parameters": {
        "default_num_results": 5,
        "include_domains": []
    }
}

# Constants for web content fetching
MAX_RETRIES = 3
FIRECRAWL_TIMEOUT = 30  # seconds

async def search_web(query: str, num_results: int = None) -> Tuple[str, list]:
    """Search the web using the Exa API and return formatted and raw results"""
    try:
        search_args = {
            "num_results": num_results or websearch_config["parameters"]["default_num_results"]
        }

        search_results = exa.search_and_contents(
            query,
            summary={"query": "Main points and key takeaways"},
            **search_args
        )

        formatted_results = format_search_results(search_results)
        return formatted_results, search_results.results
    except Exception as e:
        return f"An error occurred while searching with Exa: {e}", []

def format_search_results(search_results):
    if not search_results.results:
        return "No results found."

    markdown_results = "### Search Results:\n\n"
    for idx, result in enumerate(search_results.results, 1):
        title = result.title if hasattr(result, 'title') and result.title else "No title"
        url = result.url
        published_date = f" (Published: {result.published_date})" if hasattr(result, 'published_date') and result.published_date else ""

        markdown_results += f"**{idx}.** [{title}]({url}){published_date}\n"

        if hasattr(result, 'summary') and result.summary:
            markdown_results += f"> **Summary:** {result.summary}\n\n"
        else:
            markdown_results += "\n"

    return markdown_results

async def get_web_content(url: str) -> List[Document]:
    """Get web content and convert to document list."""
    for attempt in range(MAX_RETRIES):
        try:
            # Create FireCrawlLoader instance
            loader = FireCrawlLoader(
                url=url,
                mode="scrape"
            )
            
            # Use timeout protection
            documents = await asyncio.wait_for(loader.aload(), timeout=FIRECRAWL_TIMEOUT)
            
            # Return results if documents retrieved successfully
            if documents and len(documents) > 0:
                return documents
            
            # Retry if no documents but no exception
            print(f"No documents retrieved from {url} (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)  # Wait 1 second before retrying
                continue
                
        except requests.exceptions.HTTPError as e:
            if "Website Not Supported" in str(e):
                # Create a minimal document with error info
                print(f"Website not supported by FireCrawl: {url}")
                content = f"Content from {url} could not be retrieved: Website not supported by FireCrawl API."
                return [Document(page_content=content, metadata={"source": url, "error": "Website not supported"})]
            else:
                print(f"HTTP error retrieving content from {url}: {str(e)} (attempt {attempt + 1}/{MAX_RETRIES})")
                
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)
                continue
            raise
        except Exception as e:
            print(f"Error retrieving content from {url}: {str(e)} (attempt {attempt + 1}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1)
                continue
            raise
    
    # Return empty list if all retries failed
    return []
# # Async main function to test
# async def main():
#     query = "OpenAI ChatGPT updates 2024"
#     formatted_results, raw_results = await search_web(query)
#     print(formatted_results)
#     print(f"Number of raw results: {len(raw_results)}")

# # Run if script is executed directly
# if __name__ == "__main__":
#     asyncio.run(main())