"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chọn RecursiveCharacterTextSplitter vì:
#   - An toàn, hoạt động tốt với mọi loại văn bản (legal + news)
#   - Tự động thử tách theo đoạn → câu → từ, giữ ngữ nghĩa tốt hơn split đơn giản
#   - chunk_size=500: đủ nhỏ để embedding chính xác, đủ lớn để giữ context
#   - chunk_overlap=50: tránh mất thông tin ở ranh giới chunk (~10% overlap)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Chọn BAAI/bge-m3 vì:
#   - Multilingual, được train trên tiếng Việt → chất lượng embedding cao hơn MiniLM
#   - 1024 dim cân bằng giữa chất lượng và tốc độ (MiniLM chỉ 384, OpenAI cần API key)
#   - Chạy local, không phụ thuộc API bên ngoài
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# Chọn Weaviate vì hỗ trợ hybrid search (dense + BM25) built-in
# Cần chạy: docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"

WEAVIATE_COLLECTION = "DrugLawDocs"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    md_files = list(STANDARDIZED_DIR.rglob("*.md"))

    if not md_files:
        print(f"  [!] Không tìm thấy .md files trong {STANDARDIZED_DIR}")
        print(f"  [!] Hãy chạy Task 3 trước để tạo standardized documents.")
        return documents

    for md_file in md_files:
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type": doc_type,
            }
        })

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
    )

    chunks = []

    if CHUNKING_METHOD == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        for doc in documents:
            splits = splitter.split_text(doc["content"])
            for i, chunk_text in enumerate(splits):
                if chunk_text.strip():
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {**doc["metadata"], "chunk_index": i},
                    })

    elif CHUNKING_METHOD == "markdown_header":
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        # Sau khi tách theo heading, tiếp tục tách nếu chunk quá dài
        char_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        for doc in documents:
            header_chunks = header_splitter.split_text(doc["content"])
            i = 0
            for hchunk in header_chunks:
                sub_splits = char_splitter.split_text(hchunk.page_content)
                for sub in sub_splits:
                    if sub.strip():
                        chunks.append({
                            "content": sub,
                            "metadata": {
                                **doc["metadata"],
                                **hchunk.metadata,
                                "chunk_index": i,
                            },
                        })
                        i += 1

    elif CHUNKING_METHOD == "semantic":
        # SemanticChunker dùng embedding để tách — cần install langchain-experimental
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        splitter = SemanticChunker(embeddings)
        for doc in documents:
            splits = splitter.split_text(doc["content"])
            for i, chunk_text in enumerate(splits):
                if chunk_text.strip():
                    chunks.append({
                        "content": chunk_text,
                        "metadata": {**doc["metadata"], "chunk_index": i},
                    })

    else:
        raise ValueError(f"Unknown CHUNKING_METHOD: {CHUNKING_METHOD}")

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading embedding model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True,  # cosine similarity cần normalize
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if VECTOR_STORE == "weaviate":
        _index_weaviate(chunks)
    elif VECTOR_STORE == "chromadb":
        _index_chromadb(chunks)
    elif VECTOR_STORE == "faiss":
        _index_faiss(chunks)
    else:
        raise ValueError(f"Unknown VECTOR_STORE: {VECTOR_STORE}")


def _index_weaviate(chunks: list[dict]):
    """Index vào Weaviate local. Cần docker container đang chạy."""
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType

    print("  Connecting to Weaviate (localhost:8080) ...")
    client = weaviate.connect_to_local()

    try:
        # Xoá collection cũ nếu tồn tại để re-index sạch
        if client.collections.exists(WEAVIATE_COLLECTION):
            client.collections.delete(WEAVIATE_COLLECTION)
            print(f"  Deleted existing collection '{WEAVIATE_COLLECTION}'")

        collection = client.collections.create(
            name=WEAVIATE_COLLECTION,
            vectorizer_config=Configure.Vectorizer.none(),  # ta tự cung cấp vector
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ],
        )

        print(f"  Inserting {len(chunks)} chunks ...")
        with collection.batch.dynamic() as batch:
            for chunk in chunks:
                meta = chunk["metadata"]
                batch.add_object(
                    properties={
                        "content": chunk["content"],
                        "source": meta.get("source", ""),
                        "doc_type": meta.get("type", ""),
                        "chunk_index": meta.get("chunk_index", 0),
                    },
                    vector=chunk["embedding"],
                )

        print(f"  Collection '{WEAVIATE_COLLECTION}' created with {len(chunks)} objects.")
    finally:
        client.close()


def _index_chromadb(chunks: list[dict]):
    """Index vào ChromaDB local (thư mục ./chroma_db)."""
    import chromadb

    db_path = Path(__file__).parent.parent / "chroma_db"
    print(f"  Connecting to ChromaDB at {db_path} ...")
    client = chromadb.PersistentClient(path=str(db_path))

    # Xoá collection cũ nếu tồn tại
    try:
        client.delete_collection(WEAVIATE_COLLECTION)
        print(f"  Deleted existing collection '{WEAVIATE_COLLECTION}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=WEAVIATE_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"  Inserting {len(chunks)} chunks ...")
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start: start + batch_size]
        collection.add(
            ids=[f"chunk_{start + i}" for i in range(len(batch))],
            documents=[c["content"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )

    print(f"  ChromaDB collection '{WEAVIATE_COLLECTION}': {collection.count()} objects.")


def _index_faiss(chunks: list[dict]):
    """Index vào FAISS và lưu ra file (chỉ dense search, không có metadata query)."""
    import faiss
    import json
    import numpy as np

    output_dir = Path(__file__).parent.parent / "faiss_index"
    output_dir.mkdir(exist_ok=True)

    print(f"  Building FAISS index (dim={EMBEDDING_DIM}) ...")
    vectors = np.array([c["embedding"] for c in chunks], dtype="float32")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner Product = cosine nếu đã normalize
    index.add(vectors)

    faiss.write_index(index, str(output_dir / "index.faiss"))

    # Lưu metadata riêng để lookup sau search
    metadata = [
        {"content": c["content"], "metadata": c["metadata"]}
        for c in chunks
    ]
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"  FAISS index saved to {output_dir} ({index.ntotal} vectors).")


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    if not docs:
        return

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
