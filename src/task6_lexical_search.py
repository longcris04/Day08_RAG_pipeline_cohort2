"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

from task4_chunking_indexing import VECTOR_STORE, WEAVIATE_COLLECTION

CHROMA_DB_PATH = Path(__file__).parent.parent / "chroma_db"

# Singleton cache
_bm25: BM25Okapi | None = None
_corpus: list[dict] = []


# =============================================================================
# Corpus loading
# =============================================================================

def _load_corpus_from_chromadb() -> list[dict]:
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_collection(name=WEAVIATE_COLLECTION)

    # Lấy toàn bộ documents (không dùng query)
    result = collection.get(include=["documents", "metadatas"])

    corpus = []
    for doc, meta in zip(result["documents"], result["metadatas"]):
        corpus.append({
            "content": doc,
            "metadata": {
                "source": meta.get("source", ""),
                "doc_type": meta.get("type", ""),
                "chunk_index": meta.get("chunk_index", -1),
            },
        })
    return corpus


def _load_corpus_from_weaviate() -> list[dict]:
    import weaviate

    client = weaviate.connect_to_local()
    try:
        collection = client.collections.get(WEAVIATE_COLLECTION)
        corpus = []
        for obj in collection.iterator():
            corpus.append({
                "content": obj.properties["content"],
                "metadata": {
                    "source": obj.properties.get("source", ""),
                    "doc_type": obj.properties.get("doc_type", ""),
                    "chunk_index": obj.properties.get("chunk_index", -1),
                },
            })
    finally:
        client.close()
    return corpus


def _load_corpus_from_faiss() -> list[dict]:
    import json

    faiss_path = Path(__file__).parent.parent / "faiss_index" / "metadata.json"
    with open(faiss_path, encoding="utf-8") as f:
        metadata_store = json.load(f)

    return [
        {
            "content": entry["content"],
            "metadata": {
                "source": entry["metadata"].get("source", ""),
                "doc_type": entry["metadata"].get("type", ""),
                "chunk_index": entry["metadata"].get("chunk_index", -1),
            },
        }
        for entry in metadata_store
    ]


def _load_corpus() -> list[dict]:
    if VECTOR_STORE == "chromadb":
        return _load_corpus_from_chromadb()
    elif VECTOR_STORE == "weaviate":
        return _load_corpus_from_weaviate()
    elif VECTOR_STORE == "faiss":
        return _load_corpus_from_faiss()
    else:
        raise ValueError(f"Unknown VECTOR_STORE: {VECTOR_STORE}")


# =============================================================================
# Tokenizer
# =============================================================================

def _tokenize(text: str) -> list[str]:
    """
    Tokenize văn bản tiếng Việt.
    Dùng underthesea nếu có (tốt hơn cho tiếng Việt), fallback về split().
    """
    try:
        from underthesea import word_tokenize
        return word_tokenize(text.lower())
    except ImportError:
        return text.lower().split()


# =============================================================================
# Index builder
# =============================================================================

def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi instance
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def _get_index() -> tuple[BM25Okapi, list[dict]]:
    """Trả về (bm25, corpus), load lazy một lần duy nhất."""
    global _bm25, _corpus
    if _bm25 is None:
        print(f"  Loading corpus from {VECTOR_STORE} ...")
        _corpus = _load_corpus()
        print(f"  Building BM25 index over {len(_corpus)} chunks ...")
        _bm25 = build_bm25_index(_corpus)
    return _bm25, _corpus


# =============================================================================
# Public API
# =============================================================================

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    bm25, corpus = _get_index()

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": corpus[idx]["content"],
                "score": float(scores[idx]),
                "metadata": corpus[idx]["metadata"],
            })

    return results  # đã sorted descending theo argsort[::-1]


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
