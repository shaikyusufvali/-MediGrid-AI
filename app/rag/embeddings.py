from sentence_transformers import SentenceTransformer


# Local embedding model
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_embedding_model():
    """
    Load the embedding model only once.
    CPU is used for compatibility.
    """
    global _model

    if _model is None:
        _model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

    return _model


def generate_embeddings(texts):
    """
    Convert multiple text chunks into numerical vectors.
    """

    if not texts:
        raise ValueError("Text list cannot be empty")

    if not all(isinstance(text, str) and text.strip() for text in texts):
        raise ValueError("Every item must contain valid text")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.tolist()


def generate_query_embedding(query):
    """
    Convert a user's question into an embedding vector.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty")

    embeddings = generate_embeddings([query])

    return embeddings[0]
