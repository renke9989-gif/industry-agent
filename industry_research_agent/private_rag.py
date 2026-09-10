"""Small, dependency-light private document retriever.

Documents are chunked and persisted in SQLite. Retrieval uses a deterministic
TF-IDF cosine score (with a token-overlap fallback), so the feature works
offline and can later be swapped for an embedding/vector backend without
changing the Evidence contract.
"""
from __future__ import annotations

import hashlib, json, os, re, sqlite3, threading
from contextlib import closing
from pathlib import Path
from typing import Any

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover
    TfidfVectorizer = cosine_similarity = None


class PrivateDocumentStore:
    def __init__(self, path: str | Path = ".data/private_rag.sqlite3", *, chunk_chars: int = 900, overlap: int = 120,
                 retrieval_backend: str | None = None, embedding_model: str | None = None,
                 reranker_model: str | None = None):
        self.path = str(path); self.chunk_chars = max(200, int(chunk_chars)); self.overlap = max(0, int(overlap)); self.lock = threading.RLock()
        self.retrieval_backend = (retrieval_backend or os.getenv("PRIVATE_RAG_RETRIEVAL", "tfidf")).lower()
        self.embedding_model = embedding_model or os.getenv("PRIVATE_RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        self.reranker_model = reranker_model or os.getenv("PRIVATE_RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
        self._embedder = None; self._reranker = None; self._chroma = None; self._collection = None
        Path(self.path).parent.mkdir(parents=True, exist_ok=True) if self.path != ":memory:" else None
        self._init()

    def _connect(self):
        c = sqlite3.connect(self.path, check_same_thread=False); c.row_factory = sqlite3.Row; return c

    def _init(self):
        with closing(self._connect()) as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, title TEXT NOT NULL, sha256 TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL, page INTEGER, UNIQUE(doc_id, chunk_index));
            CREATE INDEX IF NOT EXISTS ix_chunks_doc ON chunks(doc_id);
            """)
            c.commit()

    @staticmethod
    def _read(path: Path) -> tuple[str, list[int | None]]:
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown", ".txt", ".rst", ".csv", ".json"}:
            return path.read_text(encoding="utf-8", errors="ignore"), []
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader
                pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
                return "\n".join(pages), list(range(1, len(pages) + 1))
            except ImportError as exc:
                raise RuntimeError("PDF indexing requires optional dependency pypdf") from exc
        raise ValueError(f"unsupported private document type: {suffix}")

    def _chunks(self, text: str, pages: list[int | None]) -> list[tuple[str, int | None]]:
        clean = re.sub(r"\r\n?", "\n", text).strip(); out=[]; start=0
        while start < len(clean):
            end=min(len(clean), start+self.chunk_chars); part=clean[start:end].strip()
            if part: out.append((part, None))
            if end >= len(clean): break
            start=max(start+1, end-self.overlap)
        if pages and out:
            # Best-effort page attribution based on character offsets.
            page_len=max(1, len(clean)//len(pages))
            out=[(part, min(len(pages), max(1, (i*self.chunk_chars)//page_len+1))) for i,(part,_) in enumerate(out)]
        return out

    def index_file(self, path: str | Path) -> dict[str, Any]:
        p=Path(path).resolve(); text, pages=self._read(p); digest=hashlib.sha256(text.encode()).hexdigest(); doc_id=hashlib.sha256(str(p).encode()).hexdigest()[:24]
        chunks=self._chunks(text,pages)
        with self.lock, closing(self._connect()) as c:
            c.execute("INSERT OR REPLACE INTO documents(doc_id,path,title,sha256) VALUES(?,?,?,?)", (doc_id,str(p),p.stem,digest))
            c.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            for i,(part,page) in enumerate(chunks): c.execute("INSERT INTO chunks(chunk_id,doc_id,chunk_index,text,page) VALUES(?,?,?,?,?)", (f"{doc_id}-{i}",doc_id,i,part,page))
            c.commit()
        if self.retrieval_backend in {"hybrid", "vector", "chroma"}:
            self._upsert_vectors(doc_id, chunks)
        return {"doc_id":doc_id,"path":str(p),"title":p.stem,"chunk_count":len(chunks),"sha256":digest}

    def _ensure_vector_backend(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("vector retrieval requires chromadb and sentence-transformers") from exc
        vector_path = os.getenv("PRIVATE_RAG_VECTOR_PATH", str(Path(self.path).with_suffix(".chroma")))
        self._chroma = chromadb.PersistentClient(path=vector_path)
        self._collection = self._chroma.get_or_create_collection("private_documents", metadata={"hnsw:space": "cosine"})
        self._embedder = SentenceTransformer(self.embedding_model)
        return self._collection

    def _upsert_vectors(self, doc_id: str, chunks: list[tuple[str, int | None]]) -> None:
        collection = self._ensure_vector_backend()
        ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
        if not ids:
            return
        collection.delete(where={"doc_id": doc_id})
        embeddings = self._embedder.encode([part for part, _ in chunks], normalize_embeddings=True).tolist()
        collection.upsert(ids=ids, documents=[part for part, _ in chunks], embeddings=embeddings,
                          metadatas=[{"doc_id": doc_id, "chunk_index": i, "page": page or 0} for i, (_, page) in enumerate(chunks)])

    def _vector_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        collection = self._ensure_vector_backend()
        embedding = self._embedder.encode([query], normalize_embeddings=True).tolist()
        result = collection.query(query_embeddings=embedding, n_results=max(1, min(top_k * 3, 50)), include=["documents", "metadatas", "distances"])
        hits=[]
        for text, meta, distance in zip(result.get("documents", [[]])[0], result.get("metadatas", [[]])[0], result.get("distances", [[]])[0]):
            hits.append({"chunk_id": f"{meta['doc_id']}-{meta['chunk_index']}", "doc_id": meta["doc_id"], "chunk_index": meta["chunk_index"], "page": meta.get("page") or None, "text": text, "vector_score": round(max(0.0, 1.0 - float(distance)), 4)})
        return hits

    def _rerank(self, query: str, hits: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if os.getenv("PRIVATE_RAG_RERANK", "false").lower() not in {"1", "true", "yes"} or not hits:
            return hits[:top_k]
        try:
            from sentence_transformers import CrossEncoder
            if self._reranker is None: self._reranker = CrossEncoder(self.reranker_model)
            scores = self._reranker.predict([(query, hit["text"]) for hit in hits])
            for hit, score in zip(hits, scores): hit["rerank_score"] = round(float(score), 4)
            return sorted(hits, key=lambda item: item.get("rerank_score", 0), reverse=True)[:top_k]
        except Exception:
            return hits[:top_k]

    def index_dir(self, root: str | Path, recursive: bool = True) -> list[dict[str, Any]]:
        base=Path(root); patterns=("*.md","*.markdown","*.txt","*.rst","*.csv","*.json","*.pdf"); files=[]
        for pat in patterns: files.extend(base.rglob(pat) if recursive else base.glob(pat))
        return [self.index_file(p) for p in sorted(set(files))]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        with self.lock, closing(self._connect()) as c: rows=[dict(r) for r in c.execute("SELECT c.*,d.path,d.title FROM chunks c JOIN documents d ON d.doc_id=c.doc_id").fetchall()]
        if not rows or not query.strip(): return []
        texts=[r["text"] for r in rows]
        vector_hits = []
        if self.retrieval_backend in {"hybrid", "vector", "chroma"}:
            try:
                vector_hits = self._vector_search(query, top_k)
            except Exception:
                vector_hits = []
        if TfidfVectorizer is not None:
            matrix=TfidfVectorizer(analyzer="char", ngram_range=(2,4), min_df=1).fit_transform(texts+[query]); scores=cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        else:
            q=set(re.findall(r"[\w\u4e00-\u9fff]+", query.lower())); scores=[len(q & set(re.findall(r"[\w\u4e00-\u9fff]+", t.lower()))) for t in texts]
        lexical = {r["chunk_id"]: float(score) for score, r in zip(scores, rows)}
        by_id = {r["chunk_id"]: r for r in rows}
        candidates = set(lexical)
        for item in vector_hits:
            candidates.add(item["chunk_id"])
            if item["chunk_id"] not in by_id:
                continue
        merged=[]
        vector_scores={item["chunk_id"]: item.get("vector_score", 0.0) for item in vector_hits}
        for chunk_id in candidates:
            row=by_id.get(chunk_id)
            if not row: continue
            lex=lexical.get(chunk_id, 0.0); vec=vector_scores.get(chunk_id, 0.0)
            score=(0.5*lex + 0.5*vec) if vector_hits else lex
            merged.append({"chunk_id":row["chunk_id"],"doc_id":row["doc_id"],"title":row["title"],"path":row["path"],"chunk_index":row["chunk_index"],"page":row["page"],"text":row["text"],"score":round(float(score),4),"lexical_score":round(lex,4),"vector_score":round(vec,4)})
        return self._rerank(query, sorted(merged, key=lambda item: item["score"], reverse=True), max(1,min(int(top_k),20)))

    def close(self): pass
