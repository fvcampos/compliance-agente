"""
Collection of tools available to the Agent for external data retrieval.
"""

import logging
from typing import List

from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryResponse

from src.utils.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

@tool
def retrieve_documents(
    query: str,
    chunk_limit: int) -> str:

    """
    Searches the vector database for documents relevant to the user query.
    
    Use this tool when you need to find specific information from the 
    company policies, compliance documents, or technical manuals to answer 
    a user question.

    Args:
        query (str): The search string to look up in the database. 
                     Example: "What is the spending limit for travel?"
        chunk_limit (int): The maximum number of document chunks to retrieve.

    Returns:
        str: A formatted string containing the top retrieved document chunks
             and their sources.
    """
    logger.info(f"Tool 'retrieve_documents' invoked with query: '{query}'")

    try:
        client: QdrantClient = QdrantClient(url=settings.QDRANT_URL)

        # Note: client.query() automatically handles the embedding of the 
        # input text using FastEmbed, matching the 'ingest.py' logic.
        results: List[QueryResponse] = client.query(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_text=query,
            limit=chunk_limit  # Retrieve top N most relevant chunks
        )

        if not results:
            logger.warning("No documents found for query.")
            return "No relevant documents found in the database."

        # Format the results into a context string for the LLM
        context_parts: List[str] = []
        for res in results:
            # Accessing the payload (metadata) + content
            # Note: The structure depends on how Qdrant returns the
            # QueryResponse In the Python client, 'document' is the text content
            # if managed by FastEmbed
            content: str = getattr(res, "document", "No content available")
            source: str = res.metadata.get("source", "Unknown Source")
            page: str = res.metadata.get("page", "Unknown Page")

            chunk_text = (
                f"--- Document Chunk ---\n"
                f"Source: {source} (Page {page})\n"
                f"Content: {content}\n"
            )
            context_parts.append(chunk_text)

        final_context = "\n".join(context_parts)
        logger.info(f"Retrieved {len(results)} documents successfully.")
        return final_context

    except Exception as e:
        # Raising error is not a good idea to avoid the agent to crash
        logger.error(f"Error querying Qdrant: {e}", exc_info=True)
        return f"Error retrieving documents: {str(e)}"


if __name__ == "__main__":
    # Simple local test to verify the tool works without the Agent
    test_query = "remote work"
    test_chunk_limit = 5
    logging.info(
        f"Testing 'retrieve_documents' tool with query: '{test_query}'")

    logging.info(
        retrieve_documents.invoke({
            "query": test_query,
            "chunk_limit": test_chunk_limit
        })
    )
