"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer

# Đồng bộ config với Task 4
from task4_chunking_indexing import (
    EMBEDDING_MODEL,
    VECTOR_STORE,
    WEAVIATE_COLLECTION,
)

CHROMA_DB_PATH = Path(__file__).parent.parent / "chroma_db"
FAISS_INDEX_PATH = Path(__file__).parent.parent / "faiss_index"

# Model singleton — tránh load lại mỗi lần gọi
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _embed_query(query: str) -> list[float]:
    model = _get_model()
    return model.encode(query, normalize_embeddings=True).tolist()


# =============================================================================
# Backend implementations
# =============================================================================

def _search_chromadb(query_embedding: list[float], top_k: int) -> list[dict]:
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection(name=WEAVIATE_COLLECTION)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        # ChromaDB cosine space: distance ∈ [0, 2], 0 = identical
        score = 1.0 - dist
        hits.append({
            "content": doc,
            "score": score,
            "metadata": {
                "source": meta.get("source", ""),
                "doc_type": meta.get("type", ""),
                "chunk_index": meta.get("chunk_index", -1),
            },
        })

    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _search_weaviate(query_embedding: list[float], top_k: int) -> list[dict]:
    import weaviate
    from weaviate.classes.query import MetadataQuery

    client = weaviate.connect_to_local()
    try:
        collection = client.collections.get(WEAVIATE_COLLECTION)
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )
        hits = [
            {
                "content": obj.properties["content"],
                "score": 1.0 - obj.metadata.distance,
                "metadata": {
                    "source": obj.properties.get("source", ""),
                    "doc_type": obj.properties.get("doc_type", ""),
                    "chunk_index": obj.properties.get("chunk_index", -1),
                },
            }
            for obj in results.objects
        ]
    finally:
        client.close()

    return sorted(hits, key=lambda x: x["score"], reverse=True)


def _search_faiss(query_embedding: list[float], top_k: int) -> list[dict]:
    import faiss
    import json
    import numpy as np

    index = faiss.read_index(str(FAISS_INDEX_PATH / "index.faiss"))
    with open(FAISS_INDEX_PATH / "metadata.json", encoding="utf-8") as f:
        metadata_store = json.load(f)

    query_vector = np.array([query_embedding], dtype="float32")
    scores, indices = index.search(query_vector, top_k)

    hits = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = metadata_store[idx]
        meta = entry["metadata"]
        hits.append({
            "content": entry["content"],
            "score": float(score),  # Inner Product score, higher = more similar
            "metadata": {
                "source": meta.get("source", ""),
                "doc_type": meta.get("type", ""),
                "chunk_index": meta.get("chunk_index", -1),
            },
        })

    return sorted(hits, key=lambda x: x["score"], reverse=True)


# =============================================================================
# Public API
# =============================================================================

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    query_embedding = _embed_query(query)

    if VECTOR_STORE == "chromadb":
        print("Using ChromaDB for semantic search")
        return _search_chromadb(query_embedding, top_k)
    elif VECTOR_STORE == "weaviate":
        print("Using Weaviate for semantic search")
        return _search_weaviate(query_embedding, top_k)
    elif VECTOR_STORE == "faiss":
        print("Using Faiss for semantic search")
        return _search_faiss(query_embedding, top_k)
    else:
        raise ValueError(f"Unknown VECTOR_STORE: {VECTOR_STORE}")


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=10)
    for r in results:
        # print(f"[{r['score']:.3f}] {r['content'][:100]}...")
        print(f"[{r['score']:.3f}] {r['content'][:]}")