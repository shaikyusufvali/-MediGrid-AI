import chromadb
import uuid
from pathlib import Path


# Project root directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Persistent database location
CHROMA_PATH = BASE_DIR / "data" / "chroma_db"

COLLECTION_NAME = "medigrid_knowledge"


# Initialize persistent ChromaDB
client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)


# Create or access collection
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


def add_chunks(chunks, embeddings):
    """
    Store text chunks, embeddings and metadata.
    """

    if not chunks:
        raise ValueError("Chunks cannot be empty")

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count mismatch")

    ids = [
        str(uuid.uuid4()) for _ in chunks
    ]

    texts = [
        chunk["text"] for chunk in chunks
    ]

    metadatas = [
        chunk["metadata"] for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return {
        "status": "success",
        "chunks_stored": len(chunks)
    }


def get_collection_count():
    """
    Return total stored chunks.
    """

    return collection.count()


def search_chunks(query_embedding, top_k=3):
    """
    Search relevant chunks using query embedding.
    """

    if not query_embedding:
        raise ValueError("Query embedding cannot be empty")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, max(collection.count(), 1))
    )

    return results