from app.rag.document_loader import load_pdf_document
from app.rag.text_chunker import split_documents
from app.rag.embeddings import generate_embeddings
from app.rag.retriever import retrieve_relevant_chunks
from app.rag.vector_store import add_chunks


def ingest_pdf(pdf_bytes: bytes, filename: str):
    """
    Complete PDF ingestion pipeline.

    1. Extract PDF text
    2. Split text into chunks
    3. Generate embeddings
    4. Store chunks in ChromaDB
    """

    # Step 1: Load PDF
    documents = load_pdf_document(
        pdf_bytes=pdf_bytes,
        filename=filename
    )

    # Step 2: Split into chunks
    chunks = split_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100
    )

    if not chunks:
        raise ValueError("No text chunks generated from PDF")

    # Step 3: Generate embeddings
    texts = [
        chunk["text"] for chunk in chunks
    ]

    embeddings = generate_embeddings(texts)

    # Step 4: Store in ChromaDB
    storage_result = add_chunks(
        chunks=chunks,
        embeddings=embeddings
    )

    return {
        "status": "success",
        "filename": filename,
        "pages_processed": len(documents),
        "chunks_created": len(chunks),
        "chunks_stored": storage_result["chunks_stored"],
        "message": "PDF successfully added to RAG knowledge base"
    }


def ask_rag(question: str, top_k: int = 3):
    """
    Retrieve relevant information for a user question.

    Returns relevant text with source and page details.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    results = retrieve_relevant_chunks(
        query=question,
        top_k=top_k
    )

    if not results:
        return {
            "status": "no_knowledge",
            "question": question,
            "context": [],
            "message": "No relevant knowledge found. Upload a PDF first."
        }

    return {
        "status": "success",
        "question": question,
        "context": results,
        "message": "Relevant information retrieved successfully"
    }