"""Local sentence embeddings for creator-specific similarity learning."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=2)
def _model(model_name: str, cache_dir: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError("Semantic learning requires sentence-transformers. Install requirements.txt first.") from error
    return SentenceTransformer(model_name, cache_folder=cache_dir)


def embed_texts(texts: list[str], cache_dir: Path, model_name: str = DEFAULT_EMBEDDING_MODEL) -> list[list[float]]:
    """Return L2-normalized vectors; the first run downloads the model locally."""
    if not texts:
        return []
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectors = _model(model_name, str(cache_dir)).encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [vector.tolist() for vector in vectors]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Normalized vectors make cosine similarity a simple dot product."""
    return sum(a * b for a, b in zip(left, right))
