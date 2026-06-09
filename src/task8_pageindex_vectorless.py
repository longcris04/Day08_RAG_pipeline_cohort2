"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Lưu ý:
    - PageIndex chỉ hỗ trợ file PDF (không nhận DOCX hay markdown)
    - Retrieval là async: submit_query → poll get_retrieval đến khi completed
    - doc_ids được lưu local vào pageindex_doc_ids.json để tái sử dụng
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
DOC_IDS_FILE = Path(__file__).parent.parent / "pageindex_doc_ids.json"

POLL_INTERVAL = 2    # giây giữa mỗi lần poll
POLL_TIMEOUT  = 120  # giây tối đa chờ kết quả


def _get_client():
    if not PAGEINDEX_API_KEY:
        raise ValueError(
            "PAGEINDEX_API_KEY chưa được set.\n"
            "Đăng ký tại https://pageindex.ai/ và thêm vào file .env:\n"
            "  PAGEINDEX_API_KEY=your_key_here"
        )
    from pageindex import PageIndexClient
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_doc_ids() -> dict[str, str]:
    """Đọc mapping filename → doc_id từ file JSON local."""
    if DOC_IDS_FILE.exists():
        return json.loads(DOC_IDS_FILE.read_text(encoding="utf-8"))
    return {}


def _save_doc_ids(mapping: dict[str, str]):
    DOC_IDS_FILE.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _poll_retrieval(client, retrieval_id: str) -> dict:
    """Poll get_retrieval cho đến khi completed hoặc timeout."""
    deadline = time.time() + POLL_TIMEOUT
    while time.time() < deadline:
        result = client.get_retrieval(retrieval_id)
        status = result.get("status", "")
        if status == "completed":
            return result
        if status in ("failed", "error"):
            raise RuntimeError(f"Retrieval {retrieval_id} failed: {result}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Retrieval {retrieval_id} không hoàn thành sau {POLL_TIMEOUT}s")


# =============================================================================
# Public API
# =============================================================================

def upload_documents() -> dict[str, str]:
    """
    Upload toàn bộ PDF documents từ data/landing/ lên PageIndex.
    DOCX và markdown không được hỗ trợ — chỉ upload .pdf.

    Returns:
        mapping filename → doc_id
    """
    pi = _get_client()
    existing = _load_doc_ids()

    pdf_files = list(LANDING_DIR.rglob("*.pdf"))
    if not pdf_files:
        print(f"  [!] Không tìm thấy .pdf files trong {LANDING_DIR}")
        return existing

    for pdf_file in pdf_files:
        name = pdf_file.name
        if name in existing:
            print(f"  (skip) {name} — đã upload (doc_id={existing[name]})")
            continue

        print(f"  Uploading {name} ...")
        try:
            resp = pi.submit_document(file_path=str(pdf_file))
        except Exception as e:
            err = str(e)
            if "InsufficientCredits" in err:
                print(f"  [!] Không đủ credits trên PageIndex account.")
                print(f"  [!] Truy cập https://pageindex.ai/ để nạp thêm credits.")
                break
            raise
        doc_id = resp["doc_id"]
        existing[name] = doc_id
        print(f"  ✓ {name} → doc_id={doc_id}")

    _save_doc_ids(existing)
    print(f"\n  Tổng: {len(existing)} PDFs đã có doc_id.")
    return existing


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.

    Luồng: load doc_ids → submit_query mỗi doc → poll → aggregate → top_k.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa (gộp từ tất cả documents)

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'
        }
        Sorted by score descending.
    """
    pi = _get_client()
    doc_ids = _load_doc_ids()

    if not doc_ids:
        raise RuntimeError(
            "Chưa có doc_ids. Hãy chạy upload_documents() trước."
        )

    all_hits: list[dict] = []

    for filename, doc_id in doc_ids.items():
        # Kiểm tra document đã ready chưa
        if not pi.is_retrieval_ready(doc_id):
            print(f"  [!] {filename} chưa ready, bỏ qua.")
            continue

        # Submit query và poll kết quả
        submit_resp = pi.submit_query(doc_id=doc_id, query=query)
        retrieval_id = submit_resp["retrieval_id"]
        result = _poll_retrieval(pi, retrieval_id)

        # Parse kết quả — API trả về list trong key "results" hoặc "passages"
        passages = result.get("results") or result.get("passages") or []
        for item in passages:
            text = item.get("text") or item.get("content") or ""
            score = float(item.get("score", 0.0))
            all_hits.append({
                "content": text,
                "score": score,
                "metadata": {
                    "source": filename,
                    "doc_id": doc_id,
                    **{k: v for k, v in item.items() if k not in ("text", "content", "score")},
                },
                "source": "pageindex",
            })

    all_hits.sort(key=lambda x: x["score"], reverse=True)
    return all_hits[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        doc_ids = upload_documents()

        if not doc_ids:
            print("\n[!] Không có doc_ids — bỏ qua bước query.")
            print("    Nạp thêm credits tại https://pageindex.ai/ rồi chạy lại.")
        else:
            print("\nTest query:")
            results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
            for r in results:
                print(f"[{r['score']:.3f}] {r['content'][:100]}...")
