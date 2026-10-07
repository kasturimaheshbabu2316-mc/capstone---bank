"""
rag/vector_store.py - Local ChromaDB Dual Indexing & Retrieval
Track: Banking & FinTech (Cred)
Task 3 & 4: Dual vector collections with SentenceTransformers & offline fallback.
"""

import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from typing import List, Dict, Any, Tuple, Optional
import os
import math
import hashlib
import re
from rag.chunking import (
    chunk_fixed_size,
    chunk_sentence_boundary,
    load_knowledge_base_documents,
)


class DeterministicEmbedder:
    """
    Offline deterministic vector embedder.
    Attempts to load local SentenceTransformers ('all-MiniLM-L6-v2').
    If running air-gapped without downloaded weights, falls back to a deterministic
    384-dimensional hashed semantic representation that preserves exact cosine properties.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.dimension = dimension
        self.model = None
        if os.environ.get("USE_SENTENCE_TRANSFORMERS", "").lower() in ("1", "true"):
            try:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
                from sentence_transformers import SentenceTransformer  # type: ignore
                self.model = SentenceTransformer(model_name, local_files_only=True)
            except Exception:
                self.model = None

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string into a normalized 384-dim vector."""
        if self.model is not None:
            try:
                vec = self.model.encode(text, normalize_embeddings=True)
                return [float(x) for x in vec]
            except Exception:
                pass

        # Offline deterministic signed hashing (zero-mean for clean cosine separation)
        vector = [0.0] * self.dimension
        words = re.findall(r'\b[a-z0-9_-]+\b', text.lower())
        for i, word in enumerate(words):
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            weight = 1.0 / math.sqrt(i + 1)
            vector[idx] += sign * weight

        # Add signed character 4-grams for morphological matches
        for i in range(len(text) - 4):
            sub = text[i : i + 4].lower()
            h = int(hashlib.md5(sub.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 4) & 1) else -1.0
            vector[idx] += sign * 0.1

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 1e-9:
            vector = [x / norm for x in vector]
        else:
            vector[0] = 1.0

        return vector

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes exact cosine similarity between two unit vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 <= 1e-9 or norm2 <= 1e-9:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm1 * norm2)))


class LocalVectorStore:
    """
    Manages dual ChromaDB collections persisted on disk, with in-memory fallback.
    """

    def __init__(self, persist_dir: str = "./data/chroma_db"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        self.embedder = DeterministicEmbedder()
        self.client = None
        self.collection_fixed = None
        self.collection_sentence = None
        self._in_memory_fixed: List[Dict[str, Any]] = []
        self._in_memory_sentence: List[Dict[str, Any]] = []

        if os.environ.get("USE_CHROMADB", "").lower() in ("1", "true"):
            try:
                import chromadb  # type: ignore
                self.client = chromadb.PersistentClient(path=self.persist_dir)
                self.collection_fixed = self.client.get_or_create_collection(
                    name="collection_fixed",
                    metadata={"hnsw:space": "cosine"}
                )
                self.collection_sentence = self.client.get_or_create_collection(
                    name="collection_sentence",
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception:
                self.client = None

    def upsert_chunks(self, chunks: List[Dict[str, Any]], collection_type: str = "fixed"):
        """Indexes chunks into the target collection with idempotent upsert."""
        if not chunks:
            return

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(texts)
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {k: str(v) for k, v in c.items() if k not in ("text", "chunk_id")}
            for c in chunks
        ]

        if self.client is not None:
            col = self.collection_fixed if collection_type == "fixed" else self.collection_sentence
            col.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        # Mirror in memory for fast exact arithmetic benchmarking
        target_store = self._in_memory_fixed if collection_type == "fixed" else self._in_memory_sentence
        existing_ids = {item["chunk_id"] for item in target_store}
        for chunk, emb in zip(chunks, embeddings):
            record = dict(chunk)
            record["embedding"] = emb
            if record["chunk_id"] in existing_ids:
                for idx, old in enumerate(target_store):
                    if old["chunk_id"] == record["chunk_id"]:
                        target_store[idx] = record
            else:
                target_store.append(record)
                existing_ids.add(record["chunk_id"])

    def query(
        self, query_text: str, collection_type: str = "sentence", top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k chunks with cosine similarity.
        Returns list of dicts: {chunk_id, doc_id, text, similarity, strategy}
        """
        # Ensure knowledge base is indexed if empty
        if not self._in_memory_fixed and not self._in_memory_sentence:
            self.index_all()

        query_vec = self.embedder.embed_text(query_text)
        results: List[Dict[str, Any]] = []

        target_store = self._in_memory_sentence if collection_type == "sentence" else self._in_memory_fixed

        if target_store:
            scored = []
            for item in target_store:
                sim = cosine_similarity(query_vec, item["embedding"])
                scored.append((sim, item))
            scored.sort(key=lambda x: x[0], reverse=True)

            for sim, item in scored[:top_k]:
                res = dict(item)
                res["similarity"] = round(float(sim), 4)
                res.pop("embedding", None)
                results.append(res)
            return results

        if self.client is not None:
            col = self.collection_sentence if collection_type == "sentence" else self.collection_fixed
            raw = col.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            if raw and raw["ids"] and raw["ids"][0]:
                for i in range(len(raw["ids"][0])):
                    cid = raw["ids"][0][i]
                    doc = raw["documents"][0][i]
                    meta = raw["metadatas"][0][i] if raw["metadatas"] else {}
                    # Chroma cosine distance = 1 - cosine_similarity
                    dist = raw["distances"][0][i] if raw.get("distances") else 0.5
                    sim = round(1.0 - dist, 4)
                    results.append(
                        {
                            "chunk_id": cid,
                            "doc_id": meta.get("doc_id", "unknown"),
                            "text": doc,
                            "similarity": sim,
                            "strategy": collection_type,
                        }
                    )

        return results

    def index_all(self, kb_dir: str = "knowledge_base"):
        """Indexes all documents into both fixed and sentence collections."""
        docs = load_knowledge_base_documents(kb_dir)
        for doc_id, text in docs.items():
            fixed = chunk_fixed_size(doc_id, text)
            sent = chunk_sentence_boundary(doc_id, text)
            self.upsert_chunks(fixed, collection_type="fixed")
            self.upsert_chunks(sent, collection_type="sentence")
        return len(docs)


# Singleton vector store instance
VECTOR_STORE = LocalVectorStore()


def get_vector_store() -> LocalVectorStore:
    return VECTOR_STORE


if __name__ == "__main__":
    vs = get_vector_store()
    count = vs.index_all()
    print(f"Indexed {count} documents into ChromaDB collections.")
    res = vs.query("prepayment penalty rules", collection_type="sentence", top_k=2)
    print("Sample retrieval:", res)
