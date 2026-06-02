"""IngestionPipeline — document processing pipeline.

Stages:
  1. Parse: extract text from PDF / DOCX / TXT
  2. Chunk: split text into overlapping chunks
  3. Embed: generate vector embeddings
  4. Index: store in ChromaDB, update BM25 + entity graph
"""

from __future__ import annotations

import logging
from pathlib import Path

from insight_miner.config import CHUNK_OVERLAP, CHUNK_SIZE, get_docs_dir
from insight_miner.core.document_processor import (
    KnowledgeBaseIndex,
    load_document,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates document ingestion for a single file."""

    def __init__(self, kb_index: KnowledgeBaseIndex):
        self._kb_index = kb_index
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

    async def run(self, filename: str) -> bool:
        """Process a single document end-to-end.

        Returns True on success, False on failure.
        """
        fpath = get_docs_dir(self._kb_index.kb_id) / filename
        if not fpath.exists():
            logger.error("pipeline file_not_found kb=%s file=%s", self._kb_index.kb_id, filename)
            return False

        try:
            text = load_document(fpath)
        except Exception as e:
            logger.error("pipeline parse_fail kb=%s file=%s error=%s", self._kb_index.kb_id, filename, e)
            return False

        chunks = self._splitter.split_text(text)
        doc_id = self._doc_id(filename)

        texts: list[str] = []
        ids: list[str] = []
        metas: list[dict] = []

        for i, chunk in enumerate(chunks):
            cid = f"{doc_id}_{i:06d}"
            texts.append(chunk)
            ids.append(cid)
            metas.append({
                "filename": filename,
                "chunk_id": cid,
                "chunk_index": i,
            })

        with self._kb_index._lock:
            if self._kb_index.chunk_collection is not None and texts:
                self._kb_index.chunk_collection.add_texts(
                    texts=texts,
                    ids=ids,
                    metadatas=metas,
                )
            self._kb_index.chunk_texts.extend(texts)
            self._kb_index.chunk_ids.extend(ids)
            self._kb_index.chunk_metas.extend(metas)
            self._kb_index._dirty_bm25 = True
            self._kb_index._dirty_graph = True

        # Persist BM25 + graph
        self._kb_index.finalize()
        logger.info(
            "pipeline done kb=%s file=%s chunks=%d",
            self._kb_index.kb_id, filename, len(chunks),
        )
        return True

    @staticmethod
    def _doc_id(fname: str) -> str:
        import hashlib
        return hashlib.md5(fname.encode()).hexdigest()[:8]
