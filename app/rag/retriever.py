from app.rag.embeddings import generate_query_embedding
from app.rag.vector_store import search_chunks, get_collection_count


def retrieve_relevant_chunks(query, top_k=3):
    """
    Retrieve relevant PDF chunks for a user query.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    # Check whether knowledge exists
    if get_collection_count() == 0:
        return []

    # Convert user question into embedding
    query_embedding = generate_query_embedding(query)

    # Search ChromaDB
    results = search_chunks(
        query_embedding=query_embedding,
        top_k=top_k
    )

    retrieved_documents = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        retrieved_documents.append({
            "text": text,
            "source": metadata.get("source", "Unknown"),
            "page": metadata.get("page", "Unknown"),
            "distance": distance
        })

    return retrieved_documents