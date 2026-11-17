
def split_documents(
    documents,
    chunk_size=500,
    chunk_overlap=100
):
    """
    Split PDF documents into smaller text chunks.

    Each chunk preserves its original PDF page metadata.
    """

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size"
        )

    chunks = []

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "metadata": metadata.copy()
                })

            start += chunk_size - chunk_overlap

    return chunks
