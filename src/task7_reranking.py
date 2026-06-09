"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import os
import numpy as np
from typing import Optional

# Cross-encoder singleton
_cross_encoder = None

# Model dùng cho MMR embedding (đồng bộ với Task 4/5)
_mmr_embed_model = None


# =============================================================================
# Helpers
# =============================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        # BAAI/bge-reranker-v2-m3: multilingual, tốt cho tiếng Việt, chạy local
        _cross_encoder = CrossEncoder("BAAI/bge-reranker-v2-m3")
    return _cross_encoder


def _get_embed_model():
    global _mmr_embed_model
    if _mmr_embed_model is None:
        from sentence_transformers import SentenceTransformer
        from task4_chunking_indexing import EMBEDDING_MODEL
        _mmr_embed_model = SentenceTransformer(EMBEDDING_MODEL)
    return _mmr_embed_model


# =============================================================================
# Method 1: Cross-Encoder
# =============================================================================

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    jina_api_key = os.getenv("JINA_API_KEY")

    if jina_api_key:
        # Option A: Jina Reranker API (jina-reranker-v2-base-multilingual)
        import requests

        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={"Authorization": f"Bearer {jina_api_key}"},
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": [c["content"] for c in candidates],
                "top_n": top_k,
            },
        )
        response.raise_for_status()
        reranked = response.json()["results"]
        return [
            {**candidates[r["index"]], "score": r["relevance_score"]}
            for r in reranked
        ]

    else:
        # Option B: Local cross-encoder BAAI/bge-reranker-v2-m3
        model = _get_cross_encoder()
        pairs = [(query, c["content"]) for c in candidates]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        return [
            {**cand, "score": float(score)}
            for score, cand in ranked[:top_k]
        ]


# =============================================================================
# Method 2: MMR (Maximal Marginal Relevance)
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    # Nếu candidates chưa có embedding, tự embed
    model = _get_embed_model()
    for c in candidates:
        if "embedding" not in c:
            c["embedding"] = model.encode(
                c["content"], normalize_embeddings=True
            ).tolist()

    selected_indices: list[int] = []
    remaining_indices = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining_indices:
            relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])

            if not selected_indices:
                max_sim_to_selected = 0.0
            else:
                max_sim_to_selected = max(
                    _cosine_sim(candidates[idx]["embedding"], candidates[s]["embedding"])
                    for s in selected_indices
                )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [
        {**candidates[i], "score": float(_cosine_sim(query_embedding, candidates[i]["embedding"]))}
        for i in selected_indices
    ]


# =============================================================================
# Method 3: RRF (Reciprocal Rank Fusion)
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Unified interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
    ranked_lists: Optional[list[list[dict]]] = None,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval (dùng cho cross_encoder và mmr)
        top_k: Số lượng kết quả sau rerank
        method: "cross_encoder" | "mmr" | "rrf"
        ranked_lists: Dùng cho RRF — list of ranked lists từ nhiều retriever
        lambda_param: Dùng cho MMR — trade-off relevance vs diversity

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    elif method == "mmr":
        model = _get_embed_model()
        query_embedding = model.encode(query, normalize_embeddings=True).tolist()
        return rerank_mmr(query_embedding, candidates, top_k, lambda_param)

    elif method == "rrf":
        if ranked_lists is None:
            raise ValueError("RRF cần ranked_lists — truyền danh sách kết quả từ nhiều retriever")
        return rerank_rrf(ranked_lists, top_k)

    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]

    print("=== Cross-Encoder ===")
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2, method="cross_encoder")
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")

    print("\n=== MMR ===")
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2, method="mmr")
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")

    print("\n=== RRF ===")
    list1 = [dummy_candidates[0], dummy_candidates[2]]
    list2 = [dummy_candidates[2], dummy_candidates[1]]
    results = rerank("hình phạt tàng trữ ma tuý", [], top_k=2, method="rrf", ranked_lists=[list1, list2])
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")
