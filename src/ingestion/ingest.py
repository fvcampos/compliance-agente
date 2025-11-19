'''

Collection of functions and classes to handle document ingestion into Qdrant
vector database.

'''

import os
import logging

from qdrant_client import QdrantClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)

def ingest_docs(chunk_size: int = 500, chunk_overlap: int = 50) -> None:

    """
    Ingests a PDF document into a Qdrant vector database.
    This function performs the following steps:
    1. Connects to a Qdrant instance using settings configuration.
    2. Loads a specific PDF file ('data/raw_pdfs/policy.pdf').
    3. Splits the document content into smaller text chunks using a recursive
        character splitter.
    4. Generates embeddings for these chunks (using FastEmbed via Qdrant
        client).
    5. Indexes the vectors and metadata into the specified Qdrant collection.
    Args:
        chunk_size (int, optional): The maximum size of each text chunk in
            characters. Defaults to 500.
        chunk_overlap (int, optional): The number of characters to overlap
            between adjacent chunks to maintain context. Defaults to 50.
    Returns:
        None
    Raises:
        FileNotFoundError: Implicitly handled by logging an error if the source
            PDF path does not exist.
        QdrantClientError: If connection to the Qdrant instance fails.
    """

    logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}...")
    client: QdrantClient = QdrantClient(url=settings.QDRANT_URL)

    pdf_path: str = "data/raw_pdfs/policy.pdf"
    if not os.path.exists(pdf_path):
        logger.error(f"File not found: {pdf_path}")
        return

    logger.info(f"Loading {pdf_path}...")
    loader: PyPDFLoader = PyPDFLoader(pdf_path)
    docs: list = loader.load()

    text_splitter: RecursiveCharacterTextSplitter = \
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    splits: list = text_splitter.split_documents(docs)
    logger.info(f"Split into {len(splits)} chunks.")

    logger.info(
        "Indexing into Qdrant collection " +
        f"'{settings.QDRANT_COLLECTION_NAME}'...")

    client.add(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        documents=[doc.page_content for doc in splits],
        metadata=[doc.metadata for doc in splits],
        ids=None # Auto-generate IDs
    )

    logger.info(
        f"Success! Indexed {len(splits)} chunks into" +
        f"'{settings.QDRANT_COLLECTION_NAME}'")

if __name__ == "__main__":
    ingest_docs()
