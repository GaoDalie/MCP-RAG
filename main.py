import streamlit as st
import asyncio
import os
import search
import rag

# Configure environment (consider using st.secrets in production)


st.title("Web Search & RAG System MCP")
st.write("Enter a query to search the web and get enhanced results with RAG")

# Create input field
query = st.text_input("Search query")

# Create a placeholder for the results
results_placeholder = st.empty()

# When the user submits a query
if query:
    with st.spinner("Searching and processing..."):
        # Use asyncio to run the async functions
        async def run_search_and_rag():
            try:
                # Show progress
                progress = st.progress(0)
                status_text = st.empty()
                
                # Search step
                status_text.text("Searching the web...")
                formatted_results, raw_results = await search.search_web(query)
                progress.progress(0.3)
                
                if not raw_results:
                    st.error("No search results found.")
                    return
                
                # Extract URLs
                status_text.text("Extracting URLs...")
                urls = [result.url for result in raw_results if hasattr(result, 'url')]
                progress.progress(0.5)
                
                if not urls:
                    st.error("No valid URLs found in search results.")
                    return
                
                # Create RAG
                status_text.text(f"Processing {len(urls)} URLs for RAG...")
                vectorstore = await rag.create_rag(urls)
                progress.progress(0.8)
                
                # Search RAG
                status_text.text("Retrieving enhanced results...")
                rag_results = await rag.search_rag(query, vectorstore)
                progress.progress(1.0)
                
                # Clear progress indicators
                progress.empty()
                status_text.empty()
                
                # Display results
                st.subheader("Search Results")
                st.write(formatted_results)
                
                st.subheader("RAG Enhanced Results")
                for i, doc in enumerate(rag_results):
                    with st.expander(f"Result {i+1}"):
                        st.write(doc.page_content)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)
        
        # Run the async function
        asyncio.run(run_search_and_rag())

st.sidebar.title("About")
st.sidebar.info(
    "This app uses web search MCP Tool and Retrieval Augmented Generation (RAG) MCP Tool "
    "to provide enhanced search results based on your query."
)

# Add some tips in the sidebar
st.sidebar.title("Tips")
st.sidebar.write(
    "- For best results, use specific queries\n"
    "- The system processes multiple URLs, so it may take a moment\n"
    "- Click on the expanders to view detailed RAG results"
)