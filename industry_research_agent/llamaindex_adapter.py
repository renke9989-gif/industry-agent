"""Optional LlamaIndex adapter for the private document store.

The core application does not depend on LlamaIndex. When installed, this
adapter exposes the same local document root through a VectorStoreIndex for
experiments or migration to a LlamaIndex-based retrieval pipeline.
"""
from __future__ import annotations

from pathlib import Path


def build_index(source_root: str | Path, persist_dir: str | Path | None = None):
    try:
        from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex, load_index_from_storage
    except ImportError as exc:
        raise RuntimeError("LlamaIndex adapter requires llama-index-core") from exc
    persist = Path(persist_dir) if persist_dir else None
    if persist and (persist / "docstore.json").exists():
        return load_index_from_storage(StorageContext.from_defaults(persist_dir=str(persist)))
    documents = SimpleDirectoryReader(str(source_root), recursive=True).load_data()
    index = VectorStoreIndex.from_documents(documents)
    if persist:
        persist.mkdir(parents=True, exist_ok=True); index.storage_context.persist(persist_dir=str(persist))
    return index
