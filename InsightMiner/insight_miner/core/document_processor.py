"""Document loading, chunking, and index management."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import threading
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from insight_miner.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    SUPPORTED_EXTS,
    get_bm25_path,
    get_chroma_dir,
    get_docs_dir,
    get_graph_path,
    get_kb_dir,
    get_manifest_path,
)


# ── Document loading ──

def load_document(fpath: str | Path) -> str:
    fpath = Path(fpath)
    ext = fpath.suffix.lower()
    if ext == ".pdf":
        loader: TextLoader = PyPDFLoader(str(fpath))  # type: ignore[assignment]
    elif ext in (".docx", ".doc"):
        loader = Docx2txtLoader(str(fpath))
    else:
        loader = TextLoader(str(fpath), encoding="utf-8")
    pages = loader.load()
    return "\n".join(p.page_content for p in pages)


# ── Tokenizer for BM25 ──

def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for m in re.finditer(r"[a-zA-Z0-9]+|[一-鿿]", text):
        t = m.group()
        if re.match(r"[a-zA-Z0-9]", t):
            tokens.append(t.lower())
        else:
            tokens.append(t)
    bigrams = []
    for i in range(len(tokens) - 1):
        if re.match(r"[一-鿿]", tokens[i]) and re.match(r"[一-鿿]", tokens[i + 1]):
            bigrams.append(tokens[i] + tokens[i + 1])
    return tokens + bigrams


# ── Entity graph (used during both indexing and retrieval) ──

_ENG_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "i", "you", "he",
    "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
    "your", "his", "its", "our", "their", "this", "that", "these", "those",
    "and", "or", "but", "if", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
}


def extract_entities(text: str) -> list[str]:
    entities: set[str] = set()
    for m in re.finditer(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+)*", text):
        if len(m.group()) >= 3:
            entities.add(m.group().lower())
    for m in re.finditer(r"[a-zA-Z0-9]{3,}", text):
        t = m.group().lower()
        if t not in _ENG_STOPWORDS and not t.isdigit():
            entities.add(t)
    for m in re.finditer(r"[一-鿿]{2,}", text):
        entities.add(m.group())
    return [e for e in entities if len(e) >= 2]


class GraphRetriever:
    def __init__(self):
        self.graph = None
        self.entity_to_chunks: dict[str, set[int]] = {}
        self.chunk_texts: list[str] = []
        self.chunk_metas: list[dict] = []

    def build(self, chunk_texts: list[str], chunk_metas: list[dict]):
        import networkx as nx
        self.chunk_texts = chunk_texts
        self.chunk_metas = chunk_metas
        self.entity_to_chunks.clear()
        all_entities: set[str] = set()

        for idx, text in enumerate(chunk_texts):
            entities = extract_entities(text)
            for e in entities:
                self.entity_to_chunks.setdefault(e, set()).add(idx)
                all_entities.add(e)

        self.graph = nx.Graph()
        for chunk_entities in self.entity_to_chunks.values():
            elist = list(chunk_entities)
            for i in range(len(elist)):
                for j in range(i + 1, len(elist)):
                    if self.graph.has_edge(elist[i], elist[j]):
                        self.graph[elist[i]][elist[j]]["weight"] += 1
                    else:
                        self.graph.add_edge(elist[i], elist[j], weight=1)

        # orphan entities as isolated nodes
        for e in all_entities:
            if e not in self.graph:
                self.graph.add_node(e)

    def retrieve(self, query: str, top_k: int = 40, query_entities: list[str] | None = None):
        if self.graph is None or self.graph.number_of_nodes() == 0:
            return []

        if query_entities is None or not query_entities:
            query_entities = extract_entities(query)

        matched = self._find_matching_entities(query_entities)
        if not matched:
            return []

        # BFS traversal scoring
        _SCORE_EXACT = 3.0
        _SCORE_NEIGHBOR = 2.0
        scores: dict[int, float] = {}

        for entity, match_type in matched:
            for chunk_idx in self.entity_to_chunks.get(entity, set()):
                if match_type == 0:
                    scores[chunk_idx] = scores.get(chunk_idx, 0) + _SCORE_EXACT
                else:
                    scores[chunk_idx] = scores.get(chunk_idx, 0) + _SCORE_NEIGHBOR

            if self.graph.has_node(entity):
                for neighbor in self.graph.neighbors(entity):
                    if isinstance(neighbor, int):
                        weight = self.graph[entity][neighbor].get("weight", 1)
                        boost = 1.0 + 0.5 * weight / (weight + 5.0)
                        dist = 1
                        scores[neighbor] = scores.get(neighbor, 0) + max(
                            _SCORE_NEIGHBOR * boost / dist, 0.1
                        )

        # chunk chaining bonus
        doc_chunks: dict[str, list[int]] = {}
        for idx in scores:
            fn = self.chunk_metas[idx].get("filename", "")
            doc_chunks.setdefault(fn, []).append(idx)

        bonus: dict[int, float] = {}
        for fn, indices in doc_chunks.items():
            sorted_idx = sorted(indices)
            for idx in sorted_idx:
                ci = self.chunk_metas[idx].get("chunk_index", 0)
                for delta in (-1, 1):
                    neighbor = ci + delta
                    for jdx in sorted_idx:
                        if self.chunk_metas[jdx].get("chunk_index", 0) == neighbor:
                            if idx in scores:
                                bonus[jdx] = bonus.get(jdx, 0) + 0.05
                            else:
                                bonus[jdx] = bonus.get(jdx, 0) + scores.get(idx, _SCORE_NEIGHBOR) * 0.1

        for idx, b in bonus.items():
            scores[idx] = scores.get(idx, 0) + b

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def _find_matching_entities(self, query_entities: list[str]):
        matched: list[tuple[str, int]] = []
        for qe in query_entities:
            qe_lower = qe.lower()
            # exact match
            if qe_lower in self.entity_to_chunks or (self.graph and self.graph.has_node(qe_lower)):
                matched.append((qe_lower, 0))
                continue
            # substring / contains
            found = False
            for ge in self.entity_to_chunks:
                if qe_lower in ge or ge in qe_lower:
                    matched.append((ge, 1))
                    found = True
            if not found and self.graph:
                for node in self.graph.nodes():
                    if isinstance(node, str) and (qe_lower in node or node in qe_lower):
                        matched.append((node, 1))
                        break
        return matched

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump({"graph": self.graph, "entity_to_chunks": self.entity_to_chunks}, f)

    def load(self, path: str | Path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.graph = data["graph"]
        self.entity_to_chunks = data["entity_to_chunks"]


# ── Knowledge Base Index ──

def _hash_file(fpath: str | Path) -> str:
    h = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class KnowledgeBaseIndex:
    """Manages all index artifacts for one knowledge base."""

    def __init__(self, kb_id: str):
        self.kb_id = kb_id

        # Models (set via initialize or load_models)
        self.embeddings_model: HuggingFaceEmbeddings | None = None
        self.reranker_model: CrossEncoder | None = None

        # ChromaDB collections
        self.chunk_collection: Chroma | None = None

        # BM25
        self.bm25: BM25Okapi | None = None
        self.chunk_texts: list[str] = []
        self.chunk_ids: list[str] = []
        self.chunk_metas: list[dict] = []

        # Entity graph
        self.graph_retriever = GraphRetriever()

        # Sync
        self._lock = threading.Lock()
        self._dirty_bm25 = False
        self._dirty_graph = False

    # ── Model loading ──

    def load_models(self):
        if self.embeddings_model is None:
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"local_files_only": True},
            )
        if self.reranker_model is None:
            self.reranker_model = CrossEncoder(
                "BAAI/bge-reranker-base",
                local_files_only=True,
            )

    # ── Index building ──

    def ensure_dirs(self):
        get_kb_dir(self.kb_id).mkdir(parents=True, exist_ok=True)
        get_chroma_dir(self.kb_id).mkdir(parents=True, exist_ok=True)
        get_docs_dir(self.kb_id).mkdir(parents=True, exist_ok=True)

    def _doc_id(self, fname: str) -> str:
        import hashlib
        return hashlib.md5(fname.encode()).hexdigest()[:8]

    def _make_chunk_id(self, fname: str, idx: int) -> str:
        return f"{self._doc_id(fname)}_{idx:06d}"

    def _chroma_exists(self) -> bool:
        return (get_chroma_dir(self.kb_id) / "chroma.sqlite3").exists()

    def _list_kb_files(self) -> list[Path]:
        docs_dir = get_docs_dir(self.kb_id)
        if not docs_dir.exists():
            return []
        return [f for f in sorted(docs_dir.iterdir()) if f.suffix.lower() in SUPPORTED_EXTS]

    def _load_existing_chroma(self):
        self.chunk_collection = Chroma(
            collection_name="chunks",
            embedding_function=self.embeddings_model,
            persist_directory=str(get_chroma_dir(self.kb_id)),
        )

    def _rebuild_internal_from_chroma(self):
        if self.chunk_collection is None:
            return
        all_data = self.chunk_collection.get(include=["documents", "metadatas"])
        self.chunk_texts = all_data.get("documents", []) or []
        self.chunk_ids = all_data.get("ids", []) or []
        self.chunk_metas = all_data.get("metadatas", []) or []
        self._build_bm25()
        self._build_graph()
        self._dirty_bm25 = False
        self._dirty_graph = False

    def _full_rebuild(self):
        import shutil
        chroma_dir = get_chroma_dir(self.kb_id)
        if chroma_dir.exists():
            shutil.rmtree(str(chroma_dir))
        chroma_dir.mkdir(parents=True, exist_ok=True)

        docs_dir = get_docs_dir(self.kb_id)
        docs_dir.mkdir(parents=True, exist_ok=True)

        all_chunk_texts: list[str] = []
        all_chunk_ids: list[str] = []
        all_chunk_metas: list[dict] = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        for fpath in self._list_kb_files():
            try:
                text = load_document(fpath)
            except Exception:
                continue
            chunks = splitter.split_text(text)
            doc_id = self._doc_id(fpath.name)
            for i, chunk in enumerate(chunks):
                cid = f"{doc_id}_{i:06d}"
                all_chunk_texts.append(chunk)
                all_chunk_ids.append(cid)
                all_chunk_metas.append({
                    "filename": fpath.name,
                    "chunk_id": cid,
                    "chunk_index": i,
                })

        self.chunk_texts = all_chunk_texts
        self.chunk_ids = all_chunk_ids
        self.chunk_metas = all_chunk_metas

        if all_chunk_texts:
            self.chunk_collection = Chroma.from_texts(
                texts=all_chunk_texts,
                embedding=self.embeddings_model,
                ids=all_chunk_ids,
                metadatas=all_chunk_metas,
                persist_directory=str(get_chroma_dir(self.kb_id)),
            )
        else:
            self.chunk_collection = Chroma(
                collection_name="chunks",
                embedding_function=self.embeddings_model,
                persist_directory=str(get_chroma_dir(self.kb_id)),
            )

        self._build_bm25()
        self._build_graph()
        self._save_manifest()
        self._dirty_bm25 = False
        self._dirty_graph = False

    # ── BM25 ──

    def _build_bm25(self):
        if not self.chunk_texts:
            self.bm25 = None
            return
        tokenized = [tokenize(t) for t in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

    def _save_bm25(self):
        path = get_bm25_path(self.kb_id)
        with open(path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "texts": self.chunk_texts,
                "ids": self.chunk_ids,
                "metas": self.chunk_metas,
            }, f)

    def _load_bm25(self):
        path = get_bm25_path(self.kb_id)
        if not path.exists():
            self._build_bm25()
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunk_texts = data["texts"]
        self.chunk_ids = data["ids"]
        self.chunk_metas = data["metas"]

    def bm25_search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        if self.bm25 is None:
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: -x[1])
        return [(idx, float(s)) for idx, s in indexed if s > 0][:k]

    # ── Entity graph ──

    def _build_graph(self):
        self.graph_retriever.build(self.chunk_texts, self.chunk_metas)

    def _save_graph(self):
        self.graph_retriever.save(get_graph_path(self.kb_id))

    def _load_graph(self):
        path = get_graph_path(self.kb_id)
        if not path.exists():
            self._build_graph()
            return
        self.graph_retriever.load(path)
        self.graph_retriever.chunk_texts = self.chunk_texts
        self.graph_retriever.chunk_metas = self.chunk_metas

    # ── Manifest ──

    def _compute_manifest(self):
        files = {}
        for fpath in self._list_kb_files():
            files[fpath.name] = {
                "mtime": fpath.stat().st_mtime,
                "hash": _hash_file(fpath),
            }
        return {
            "version": 2,
            "files": files,
        }

    def _save_manifest(self):
        path = get_manifest_path(self.kb_id)
        with open(path, "w") as f:
            json.dump(self._compute_manifest(), f, indent=2)

    def _load_manifest(self) -> dict | None:
        path = get_manifest_path(self.kb_id)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def detect_changes(self) -> dict:
        """Returns {new: [fnames], modified: [fnames], deleted: [fnames]}."""
        manifest = self._load_manifest()
        current_files = {f.name: f for f in self._list_kb_files()}
        result: dict = {"new": [], "modified": [], "deleted": []}

        if manifest is None or manifest.get("version") != 2:
            result["new"] = [f.name for f in self._list_kb_files()]
            return result

        prev = manifest.get("files", {})

        for fname, fpath in current_files.items():
            if fname not in prev:
                result["new"].append(fname)
            elif _hash_file(fpath) != prev[fname].get("hash"):
                result["modified"].append(fname)

        for fname in prev:
            if fname not in current_files:
                result["deleted"].append(fname)

        return result

    # ── Document CRUD ──

    def add_document(self, fname: str) -> bool:
        """Add a single document's chunks to the index. Returns True if successful."""
        fpath = get_docs_dir(self.kb_id) / fname
        if not fpath.exists():
            return False
        try:
            text = load_document(fpath)
        except Exception:
            return False

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_text(text)
        doc_id = self._doc_id(fname)
        texts: list[str] = []
        ids: list[str] = []
        metas: list[dict] = []

        for i, chunk in enumerate(chunks):
            cid = f"{doc_id}_{i:06d}"
            texts.append(chunk)
            ids.append(cid)
            metas.append({
                "filename": fname,
                "chunk_id": cid,
                "chunk_index": i,
            })

        with self._lock:
            if self.chunk_collection is not None and texts:
                self.chunk_collection.add_texts(texts=texts, ids=ids, metadatas=metas)
            self.chunk_texts.extend(texts)
            self.chunk_ids.extend(ids)
            self.chunk_metas.extend(metas)
            self._dirty_bm25 = True
            self._dirty_graph = True

        return True

    def remove_document(self, fname: str):
        """Remove all chunks belonging to a document."""
        with self._lock:
            if self.chunk_collection is not None:
                existing = self.chunk_collection.get(where={"filename": fname})
                if existing and existing.get("ids"):
                    self.chunk_collection.delete(ids=existing["ids"])

            keep_texts: list[str] = []
            keep_ids: list[str] = []
            keep_metas: list[dict] = []
            for t, i, m in zip(self.chunk_texts, self.chunk_ids, self.chunk_metas):
                if m.get("filename") != fname:
                    keep_texts.append(t)
                    keep_ids.append(i)
                    keep_metas.append(m)

            self.chunk_texts = keep_texts
            self.chunk_ids = keep_ids
            self.chunk_metas = keep_metas
            self._dirty_bm25 = True
            self._dirty_graph = True

    def finalize(self):
        """Persist BM25 and graph if dirty, then save manifest."""
        with self._lock:
            if self._dirty_bm25:
                self._build_bm25()
                self._save_bm25()
                self._dirty_bm25 = False
            if self._dirty_graph:
                self._build_graph()
                self._save_graph()
                self._dirty_graph = False
        self._save_manifest()

    def initialize(self):
        """Load existing indices or rebuild from scratch."""
        self.ensure_dirs()
        if self._chroma_exists():
            self._load_existing_chroma()
            self._load_bm25()
            self._load_graph()
            changes = self.detect_changes()
            self._apply_changes(changes)
        else:
            self._full_rebuild()

    def _apply_changes(self, changes: dict):
        dirty = False
        for fname in changes.get("deleted", []):
            self.remove_document(fname)
            dirty = True
        for fname in changes.get("modified", []):
            self.remove_document(fname)
            self.add_document(fname)
            dirty = True
        for fname in changes.get("new", []):
            self.add_document(fname)
            dirty = True
        if dirty:
            self.finalize()

    # ── Retrieval helpers ──

    def dense_search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """Semantic search via ChromaDB. Returns [(chunk_index, score)]."""
        if self.chunk_collection is None:
            return []
        results = self.chunk_collection.similarity_search_with_score(query, k=k)
        # ChromaDB returns (Document, score) where score is L2 distance (lower=better)
        indexed: list[tuple[int, float]] = []
        for doc, score in results:
            cid = doc.metadata.get("chunk_id", "")
            try:
                idx = self.chunk_ids.index(cid)
            except ValueError:
                continue
            # Normalize: convert L2 to a 0-1 score (1 = best)
            normalized = 1.0 / (1.0 + score)
            indexed.append((idx, normalized))
        indexed.sort(key=lambda x: -x[1])
        return indexed

    def graph_search(self, query: str, k: int = 40, query_entities: list[str] | None = None) -> list[tuple[int, float]]:
        return self.graph_retriever.retrieve(query, top_k=k, query_entities=query_entities)

    @staticmethod
    def rrf_fusion(*ranked_lists: list[tuple[int, float]], k: int = 60) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (idx, _) in enumerate(ranked):
                scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
        result = sorted(scores.items(), key=lambda x: -x[1])
        return result

    def rerank(self, query: str, docs: list[str], top_k: int = 5) -> list[tuple[str, float]]:
        if not docs or self.reranker_model is None:
            return [(d, 0.0) for d in docs]
        pairs = [[query, d[:1024]] for d in docs]
        scores = self.reranker_model.predict(pairs)
        indexed = list(zip(docs, scores.tolist()))
        indexed.sort(key=lambda x: -x[1])
        return indexed[:top_k]
