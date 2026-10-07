"""
rag/chunking.py - Dual Chunking Strategies for Knowledge Base
Track: Banking & FinTech (Cred)
Task 3: Fixed-size with overlap vs Sentence-boundary chunking.
"""

from typing import List, Dict, Any
import re
import os


def chunk_fixed_size(
    doc_id: str, text: str, chunk_size: int = 200, overlap: int = 40
) -> List[Dict[str, Any]]:
    """
    Strategy A: Fixed-size character window with character overlap.
    """
    chunks: List[Dict[str, Any]] = []
    text_clean = text.strip()
    if not text_clean:
        return chunks

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be strictly greater than overlap.")

    chunk_idx = 0
    start = 0
    n = len(text_clean)

    while start < n:
        end = min(start + chunk_size, n)
        chunk_text = text_clean[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_fixed_{chunk_idx:03d}",
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "strategy": "fixed",
                    "start_char": start,
                    "end_char": end,
                }
            )
            chunk_idx += 1
        if end >= n:
            break
        start += step

    return chunks


def split_into_sentences(text: str) -> List[str]:
    """Splits text along grammatical sentence boundaries."""
    # Matches terminal punctuation followed by space or end of string
    sentence_end = re.compile(r'(?<=[.!?])\s+')
    raw_sentences = sentence_end.split(text.strip())
    return [s.strip() for s in raw_sentences if s.strip()]


def chunk_sentence_boundary(
    doc_id: str, text: str, sentences_per_chunk: int = 1
) -> List[Dict[str, Any]]:
    """
    Strategy B: Sentence-boundary chunking preserving semantic complete thoughts.
    """
    chunks: List[Dict[str, Any]] = []
    sentences = split_into_sentences(text)
    if not sentences:
        return chunks

    chunk_idx = 0
    for i in range(0, len(sentences), sentences_per_chunk):
        batch = sentences[i : i + sentences_per_chunk]
        chunk_text = " ".join(batch).strip()
        if chunk_text:
            chunks.append(
                {
                    "chunk_id": f"{doc_id}_sent_{chunk_idx:03d}",
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "strategy": "sentence",
                    "sentence_count": len(batch),
                }
            )
            chunk_idx += 1

    return chunks


def load_knowledge_base_documents(kb_dir: str = "knowledge_base") -> Dict[str, str]:
    """Loads all authoritative txt policy documents from the directory."""
    documents: Dict[str, str] = {}
    if not os.path.exists(kb_dir):
        return documents

    for filename in sorted(os.listdir(kb_dir)):
        if filename.endswith(".txt"):
            doc_id = os.path.splitext(filename)[0]
            with open(os.path.join(kb_dir, filename), "r", encoding="utf-8") as f:
                documents[doc_id] = f.read().strip()

    return documents


if __name__ == "__main__":
    docs = load_knowledge_base_documents()
    print(f"Loaded {len(docs)} documents.")
    if docs:
        first_id = list(docs.keys())[0]
        fixed = chunk_fixed_size(first_id, docs[first_id])
        sent = chunk_sentence_boundary(first_id, docs[first_id])
        print(f"Sample doc '{first_id}': Fixed chunks={len(fixed)}, Sent chunks={len(sent)}")
