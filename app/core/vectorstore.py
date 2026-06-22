import logging
import pickle
import threading
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger("chat.vectorstore")

_MODEL_NAME = "all-MiniLM-L6-v2"
_DIMENSION = 384  # output dim for all-MiniLM-L6-v2


class VectorStoreManager:
    """
    Per-session FAISS indexes backed by sentence-transformers embeddings.
    Indexes are persisted to disk so they survive server restarts.
    Thread-safe via a single lock — FAISS is not thread-safe by itself.
    """

    def __init__(self, store_dir: str = "./vector_store"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._indexes: dict[str, faiss.Index] = {}
        self._metadata: dict[str, list[dict]] = {}

        log.info("[VECTOR] loading embedding model %s ...", _MODEL_NAME)
        self._model = SentenceTransformer(_MODEL_NAME)
        log.info("[VECTOR] embedding model ready  dim=%d", _DIMENSION)

    # ── disk I/O ───────────────────────────────────────────────────────────────

    def _paths(self, session_id: str) -> tuple[Path, Path]:
        d = self.store_dir / session_id
        d.mkdir(exist_ok=True)
        return d / "index.faiss", d / "metadata.pkl"

    def _load(self, session_id: str):
        """Load from disk if not already in memory. Must be called under lock."""
        if session_id in self._indexes:
            return
        idx_path, meta_path = self._paths(session_id)
        if idx_path.exists() and meta_path.exists():
            self._indexes[session_id] = faiss.read_index(str(idx_path))
            with open(meta_path, "rb") as f:
                self._metadata[session_id] = pickle.load(f)
            log.info(
                "[VECTOR] index loaded  session=%s  vectors=%d",
                session_id, self._indexes[session_id].ntotal,
            )
        else:
            # IndexFlatIP with normalized vectors = cosine similarity
            self._indexes[session_id] = faiss.IndexFlatIP(_DIMENSION)
            self._metadata[session_id] = []

    def _save(self, session_id: str):
        idx_path, meta_path = self._paths(session_id)
        faiss.write_index(self._indexes[session_id], str(idx_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self._metadata[session_id], f)

    # ── embedding ──────────────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

    # ── public API ─────────────────────────────────────────────────────────────

    def add_message(self, session_id: str, role: str, content: str):
        """Embed and index a single message. Called after saving to DB."""
        if not content.strip():
            return
        with self._lock:
            self._load(session_id)
            vec = self._embed([content])
            self._indexes[session_id].add(vec)
            self._metadata[session_id].append({"role": role, "content": content})
            self._save(session_id)

    def backfill_if_needed(self, session_id: str, messages: list[dict]):
        """
        Index all messages when the on-disk index doesn't exist yet
        (e.g. first semantic search after adding this feature to an existing DB).
        No-op if the index already has vectors.
        """
        with self._lock:
            self._load(session_id)
            if self._indexes[session_id].ntotal > 0 or not messages:
                return
            log.info(
                "[VECTOR] backfilling %d messages  session=%s", len(messages), session_id
            )
            texts = [m["content"] for m in messages]
            vecs = self._embed(texts)
            self._indexes[session_id].add(vecs)
            self._metadata[session_id] = [
                {"role": m["role"], "content": m["content"]} for m in messages
            ]
            self._save(session_id)
            log.info(
                "[VECTOR] backfill done  session=%s  vectors=%d",
                session_id, self._indexes[session_id].ntotal,
            )

    def search(self, session_id: str, query: str, k: int = 3) -> list[dict]:
        """Return up to k most relevant messages with their cosine similarity scores."""
        with self._lock:
            self._load(session_id)
            index = self._indexes[session_id]
            if index.ntotal == 0:
                return []
            k = min(k, index.ntotal)
            vec = self._embed([query])
            scores, indices = index.search(vec, k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    meta = self._metadata[session_id][idx]
                    results.append({
                        "role": meta["role"],
                        "content": meta["content"],
                        "relevance_score": round(float(score), 3),
                    })
            return results

    def search_all(self, query: str, k: int = 5) -> list[dict]:
        """
        Search across every session's index. Computes the embedding once, then
        queries each loaded/on-disk index under a single lock acquisition.
        Returns top k results globally, each tagged with its session_id.
        """
        vec = self._embed([query])  # expensive — done once outside the loop
        session_dirs = [d for d in self.store_dir.iterdir() if d.is_dir()]

        all_results = []
        with self._lock:
            for d in session_dirs:
                sid = d.name
                self._load(sid)
                index = self._indexes.get(sid)
                if index is None or index.ntotal == 0:
                    continue
                k_local = min(k, index.ntotal)
                scores, indices = index.search(vec, k_local)
                for score, idx in zip(scores[0], indices[0]):
                    if idx >= 0:
                        meta = self._metadata[sid][idx]
                        all_results.append({
                            "session_id": sid,
                            "role": meta.get("role", "unknown"),
                            "content": meta.get("content", ""),
                            "relevance_score": round(float(score), 3),
                        })

        all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return all_results[:k]

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        final_response: str,
        external_tool_names: list[str],
    ):
        """
        Index a completed turn as a single rich document.
        Stores user message + tool context + response summary so semantic_search
        can find turns where specific tools were used or topics were discussed.
        """
        if not user_message.strip():
            return
        parts = [f"User: {user_message}"]
        if external_tool_names:
            parts.append(f"Tools used: {', '.join(external_tool_names)}")
        if final_response:
            parts.append(f"Assistant: {final_response[:600]}")
        doc = "\n".join(parts)

        with self._lock:
            self._load(session_id)
            vec = self._embed([doc])
            self._indexes[session_id].add(vec)
            self._metadata[session_id].append({
                "role": "turn",
                "content": doc,
                "user_message": user_message,
                "external_tools": external_tool_names,
            })
            self._save(session_id)

    def delete_session(self, session_id: str):
        """Remove the index from disk and memory. Called when a session is deleted."""
        import shutil
        with self._lock:
            d = self.store_dir / session_id
            if d.exists():
                shutil.rmtree(d)
            self._indexes.pop(session_id, None)
            self._metadata.pop(session_id, None)
        log.info("[VECTOR] index deleted  session=%s", session_id)


# Single instance shared across all requests
vector_store = VectorStoreManager()
