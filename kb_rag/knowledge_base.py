"""知识库模块 — RagKnowledgeBase 类封装所有 KB 状态、持久化与检索。

特性：
  - 摘要缓存到 .kb_summaries.json，仅对新增/变更文件调 LLM 生成摘要
  - ChromaDB 持久化到 chroma_db/，通过 .manifest.json 做增量更新
  - 每文档内容哈希 + chunk 数记录在 manifest，仅差异部分增量操作
  - 嵌入模型与重排序模型并发加载 (asyncio.gather + run_in_executor)
  - 混合检索：语义 (dense) + BM25 (sparse) + 图检索 (graph) → RRF → CrossEncoder
"""

import os
import re
import json
import pickle
import hashlib
import shutil
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langchain.messages import HumanMessage
from rank_bm25 import BM25Okapi

from .config import (
    logger, KB_DIR, CHROMA_DIR, SUMMARY_CACHE_PATH,
    EMBEDDING_MODEL_NAME, _SUPPORTED_EXTS, LLM_PROVIDER,
)
from .llm import create_llm
from .graph_retriever import GraphRetriever

# ── 格式转换层：任意格式 → 纯文本 ──────────────────────────────────────


def load_document(fpath: str) -> list:
    """通过 langchain_community 的 loader 将任意格式转为纯文本。"""
    ext = os.path.splitext(fpath)[1].lower() or ".txt"

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader(fpath).load()

    if ext in (".docx", ".doc"):
        from langchain_community.document_loaders import Docx2txtLoader
        return Docx2txtLoader(fpath).load()

    return TextLoader(fpath, encoding="utf-8").load()


# ════════════════════════════════════════════════════════════════════════
# RagKnowledgeBase
# ════════════════════════════════════════════════════════════════════════

class RagKnowledgeBase:
    """封装知识库的所有状态和操作。

    Attributes:
        embeddings_model: HuggingFace 嵌入模型
        reranker_model: CrossEncoder 重排序模型
        summary_collection: ChromaDB 摘要索引
        chunk_collection: ChromaDB 片段索引
        descriptions: {filename: summary} 文档摘要映射
        file_mtimes: {filename: mtime} 文件修改时间
    """

    def __init__(self, kb_dir: str = KB_DIR, chroma_dir: str = CHROMA_DIR,
                 summary_cache_path: str = SUMMARY_CACHE_PATH,
                 embedding_model_name: str = EMBEDDING_MODEL_NAME):
        self.kb_dir = kb_dir
        self.chroma_dir = chroma_dir
        self.summary_cache_path = summary_cache_path
        self.embedding_model_name = embedding_model_name

        # 模型
        self.embeddings_model: HuggingFaceEmbeddings | None = None
        self.reranker_model: CrossEncoder | None = None

        # ChromaDB 索引
        self.summary_collection: Chroma | None = None
        self.chunk_collection: Chroma | None = None

        # 摘要缓存
        self.descriptions: dict[str, str] = {}
        self.file_mtimes: dict[str, float] = {}

        # BM25 索引
        self._bm25_index: BM25Okapi | None = None
        self._chunk_texts: list[str] = []
        self._chunk_ids: list[str] = []
        self._chunk_metas: list[dict] = []

        # BM25 持久化路径
        self._bm25_path = os.path.join(chroma_dir, "bm25_index.pkl")

        # 实体共现图
        self._graph_retriever: GraphRetriever | None = None
        self._graph_path = os.path.join(chroma_dir, "entity_graph.pkl")

        # 增量更新脏标记
        self._dirty_bm25 = False
        self._dirty_graph = False
        self._lock = threading.Lock()

        # Manifest 路径
        self._manifest_path = os.path.join(chroma_dir, ".manifest.json")

    # ── 公共生命周期 ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """一站式启动：并发加载模型 → 加载摘要缓存 → 刷新摘要 → 构建/复用索引。"""
        os.makedirs(self.kb_dir, exist_ok=True)
        os.makedirs(self.chroma_dir, exist_ok=True)

        await self._preload_models_concurrent()
        self._load_summary_cache()
        await self._refresh_descriptions()
        self._build_or_reuse_index()
        logger.info("[KB] RagKnowledgeBase 初始化完成")

    # ── 模型加载（并发）────────────────────────────────────────────────

    async def _preload_models_concurrent(self) -> None:
        """并发加载嵌入模型和重排序模型。"""
        if self.embeddings_model is not None and self.reranker_model is not None:
            return

        loop = asyncio.get_event_loop()

        def _load_embed():
            try:
                self.embeddings_model = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                )
                logger.info(f"[Preload] 嵌入模型加载完成（{self.embedding_model_name}）")
            except Exception as e:
                logger.warning(f"[Preload] 嵌入模型加载失败: {e}")

        def _load_rerank():
            try:
                self.reranker_model = CrossEncoder("BAAI/bge-reranker-base")
                logger.info("[Preload] 重排序模型加载完成")
            except Exception as e:
                logger.warning(f"[Preload] 重排序模型加载失败（将跳过精排）: {e}")

        await asyncio.gather(
            loop.run_in_executor(None, _load_embed),
            loop.run_in_executor(None, _load_rerank),
        )

    # ── 摘要缓存 I/O ───────────────────────────────────────────────────

    def _load_summary_cache(self) -> None:
        """从 .kb_summaries.json 加载缓存的摘要。"""
        if not os.path.exists(self.summary_cache_path):
            return
        try:
            with open(self.summary_cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.descriptions = data.get("descriptions", {})
            raw_mtimes = data.get("file_mtimes", {})
            self.file_mtimes = {k: float(v) for k, v in raw_mtimes.items()}
            logger.info(f"[KB] 已加载 {len(self.descriptions)} 条摘要缓存")
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("[KB] 摘要缓存损坏，将重新生成")
            self.descriptions = {}
            self.file_mtimes = {}

    def _save_summary_cache(self) -> None:
        """将当前摘要写入 .kb_summaries.json。"""
        try:
            with open(self.summary_cache_path, "w", encoding="utf-8") as f:
                json.dump({
                    "descriptions": self.descriptions,
                    "file_mtimes": {k: str(v) for k, v in self.file_mtimes.items()},
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[KB] 保存摘要缓存失败: {e}")

    # ── 摘要刷新 ───────────────────────────────────────────────────────

    async def _refresh_descriptions(self) -> None:
        """对比文件 mtime，仅对新增或变更的文件调 LLM 生成摘要。"""
        kb_files = self._list_kb_files()
        if not kb_files:
            self.descriptions = {}
            self.file_mtimes = {}
            self._save_summary_cache()
            logger.info("[KB] 无可加载文档，跳过摘要生成")
            return

        for cached in list(self.descriptions.keys()):
            if cached not in kb_files:
                del self.descriptions[cached]
                self.file_mtimes.pop(cached, None)

        pending = []
        for fname in kb_files:
            fpath = os.path.join(self.kb_dir, fname)
            mtime = os.path.getmtime(fpath)
            if fname in self.descriptions and self.file_mtimes.get(fname) == mtime:
                continue
            pending.append((fname, fpath, mtime))

        if not pending:
            logger.info(f"[KB] 摘要全部命中缓存，共 {len(self.descriptions)} 个文档")
            return

        llm = create_llm(LLM_PROVIDER, temperature=0.0)

        for fname, fpath, mtime in pending:
            try:
                docs = load_document(fpath)
                content = "\n".join(doc.page_content for doc in docs)[:2000]
            except Exception:
                content = "（无法读取文件内容）"

            summary_prompt = (
                f"请用一句话概括以下文档的主题内容，不超过30个汉字。\n\n"
                f"文档内容（前2000字符）:\n{content}"
            )
            try:
                response = await llm.ainvoke([HumanMessage(content=summary_prompt)])
                summary = response.content.strip()
            except Exception:
                summary = os.path.splitext(fname)[0]

            self.descriptions[fname] = summary
            self.file_mtimes[fname] = mtime
            logger.info(f"[KB] 摘要生成: {fname} → {summary[:50]}")

        self._save_summary_cache()
        logger.info(f"[KB] 摘要就绪，共 {len(self.descriptions)} 个文档")

    # ── 文档 ID / 哈希 ────────────────────────────────────────────────

    @staticmethod
    def _doc_id(fname: str) -> str:
        """从文件名派生的稳定短 ID，用于 chunk_id 前缀。"""
        return hashlib.md5(fname.encode("utf-8")).hexdigest()[:8]

    @staticmethod
    def _hash_file(fpath: str) -> str:
        """快速文件内容哈希（流式读取，适合大文件）。"""
        h = hashlib.md5()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def _make_chunk_id(self, fname: str, idx: int) -> str:
        """生成全局唯一的 chunk ID：{doc_id}_{index:06d}。"""
        return f"{self._doc_id(fname)}_{idx:06d}"

    # ── Manifest / 增量变更检测 ───────────────────────────────────────

    def _compute_manifest(self) -> dict:
        """生成当前 KB 文件的 manifest 快照（含内容哈希 + chunk 数）。"""
        files = {}
        for fname in self._list_kb_files():
            fpath = os.path.join(self.kb_dir, fname)
            files[fname] = {
                "mtime": os.path.getmtime(fpath),
                "hash": self._hash_file(fpath),
            }
        return {
            "version": 2,
            "files": files,
            "summary_count": len(self.descriptions),
        }

    def _manifest_unchanged(self) -> bool:
        """检查 manifest 是否与持久化版本一致。"""
        if not os.path.exists(self._manifest_path):
            return False
        if not os.path.isdir(self.chroma_dir):
            return False
        if not os.path.exists(os.path.join(self.chroma_dir, "chroma.sqlite3")):
            return False
        if not os.path.exists(self._bm25_path):
            return False
        if not os.path.exists(self._graph_path):
            return False
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # 版本 1 旧格式 → 强制迁移
            if saved.get("version") != 2:
                return False
            # 检查文件是否有变化（只比 hash，不比 mtime）
            current = self._compute_manifest()
            cur_files = current.get("files", {})
            old_files = saved.get("files", {})
            if set(cur_files) != set(old_files):
                return False
            for fname in cur_files:
                if cur_files[fname]["hash"] != old_files[fname]["hash"]:
                    return False
            return saved.get("summary_count") == current.get("summary_count")
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            return False

    def _save_manifest(self) -> None:
        """将当前 manifest 写入 chroma_db/.manifest.json。"""
        os.makedirs(self.chroma_dir, exist_ok=True)
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._compute_manifest(), f, ensure_ascii=False, indent=2)

    def _detect_file_changes(self) -> dict[str, list[str]]:
        """对比 manifest 和文件系统，返回 {new, modified, deleted}。

        仅支持 version 2 格式；旧格式或缺失视作所有文件为新。
        """
        saved: dict[str, dict] = {}
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == 2:
                saved = data.get("files", {})
            # 旧格式（version 1 或纯 dict）→ ignored，所有文件标记为 new
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        current_files = set(self._list_kb_files())
        saved_files = set(saved.keys())

        new_files = current_files - saved_files
        deleted_files = saved_files - current_files

        modified_files = set()
        for fname in current_files & saved_files:
            fpath = os.path.join(self.kb_dir, fname)
            cur_hash = self._hash_file(fpath)
            if cur_hash != saved[fname].get("hash", ""):
                modified_files.add(fname)

        return {
            "new": sorted(new_files),
            "modified": sorted(modified_files),
            "deleted": sorted(deleted_files),
        }

    # ── ChromaDB 索引构建 / 复用 / 增量更新 ──────────────────────────

    def _chroma_db_exists(self) -> bool:
        """ChromaDB 持久化目录是否存在且可用。"""
        return (
            os.path.isdir(self.chroma_dir)
            and os.path.exists(os.path.join(self.chroma_dir, "chroma.sqlite3"))
        )

    def _load_existing_chroma(self) -> None:
        """加载已有的 ChromaDB 集合。"""
        self.summary_collection = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embeddings_model,
            collection_name="summaries",
        )
        self.chunk_collection = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embeddings_model,
            collection_name="chunks",
        )

    def _full_rebuild(self) -> None:
        """全量重建所有索引（首次构建 / manifest 版本不兼容时）。"""
        kb_files = self._list_kb_files()
        if not kb_files:
            self.summary_collection = None
            self.chunk_collection = None
            return

        if os.path.exists(self.chroma_dir):
            shutil.rmtree(self.chroma_dir)
            os.makedirs(self.chroma_dir)

        logger.info("[KB] 开始全量构建索引...")

        # ── 摘要索引 ──
        summary_texts, summary_ids, summary_metas = [], [], []
        for fname in kb_files:
            summary = self.descriptions.get(fname, os.path.splitext(fname)[0])
            summary_texts.append(summary)
            summary_ids.append(fname)
            summary_metas.append({"filename": fname})

        self.summary_collection = Chroma.from_texts(
            texts=summary_texts,
            embedding=self.embeddings_model,
            ids=summary_ids,
            metadatas=summary_metas,
            persist_directory=self.chroma_dir,
            collection_name="summaries",
        )
        logger.info(f"[KB] 摘要索引就绪，{len(summary_texts)} 个文档")

        # ── 片段索引（逐文档分块，使用 doc-based chunk ID）──
        all_chunk_texts, all_chunk_ids, all_chunk_metas = [], [], []
        for fname in kb_files:
            fpath = os.path.join(self.kb_dir, fname)
            try:
                docs = load_document(fpath)
            except Exception as e:
                logger.warning(f"[KB] 加载文件 {fname} 失败: {e}")
                continue

            splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
            splits = splitter.split_documents(docs)

            for i, doc in enumerate(splits):
                all_chunk_texts.append(doc.page_content)
                all_chunk_ids.append(self._make_chunk_id(fname, i))
                all_chunk_metas.append({
                    "filename": fname,
                    "chunk_id": self._make_chunk_id(fname, i),
                    "chunk_index": i,
                })

        if not all_chunk_texts:
            self.chunk_collection = None
            self._save_manifest()
            return

        self._chunk_texts = all_chunk_texts
        self._chunk_ids = all_chunk_ids
        self._chunk_metas = all_chunk_metas

        self.chunk_collection = Chroma.from_texts(
            texts=all_chunk_texts,
            embedding=self.embeddings_model,
            ids=all_chunk_ids,
            metadatas=all_chunk_metas,
            persist_directory=self.chroma_dir,
            collection_name="chunks",
        )
        logger.info(
            f"[KB] 片段索引就绪，{len(all_chunk_texts)} 个 chunk，{len(kb_files)} 个文档"
        )

        self._build_bm25()
        self._build_entity_graph()
        self._save_manifest()

    def _build_or_reuse_index(self) -> None:
        """智能索引构建：全量 or 加载缓存 or 增量更新。"""
        if self.embeddings_model is None:
            logger.warning("[KB] 嵌入模型不可用，跳过索引构建")
            self.summary_collection = None
            self.chunk_collection = None
            return

        kb_files = self._list_kb_files()
        if not kb_files:
            logger.warning("[KB] 无可加载文档，跳过索引构建")
            self.summary_collection = None
            self.chunk_collection = None
            return

        # 首次运行或无 ChromaDB → 全量构建
        if not self._chroma_db_exists():
            self._full_rebuild()
            return

        # Manifest 版本不兼容 → 全量重建（避免旧格式导致增量异常）
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                _v = json.load(f).get("version", 1)
        except (FileNotFoundError, json.JSONDecodeError):
            _v = 0
        if _v != 2:
            logger.info(f"[KB] Manifest 版本不兼容 (v{_v})，触发全量重建")
            self._full_rebuild()
            return

        # Manifest 未变 → 从缓存加载
        if self._manifest_unchanged():
            logger.info("[KB] Manifest 未变更，加载已有索引")
            self._load_existing_chroma()
            if not self._load_bm25():
                logger.warning("[KB] BM25 缺失，将从 ChromaDB 重建")
                self._dirty_bm25 = True
            if not self._load_entity_graph():
                logger.warning("[KB] 实体图缺失，将从 ChromaDB 重建")
                self._dirty_graph = True
            if self._dirty_bm25 or self._dirty_graph:
                self._rebuild_internal_from_chroma()
            logger.info("[KB] 索引加载完成（复用已有）")
            return

        # Manifest 变更 → 加载 ChromaDB + 增量更新
        logger.info("[KB] 检测到文件变更，开始增量更新...")
        self._load_existing_chroma()
        changes = self._detect_file_changes()
        logger.info(
            f"[KB] 变更: +{len(changes['new'])} ~{len(changes['modified'])} "
            f"-{len(changes['deleted'])}"
        )
        self._apply_file_changes(changes)
        self._finalize_index()

    # ── 增量更新 ─────────────────────────────────────────────────────

    def _apply_file_changes(self, changes: dict[str, list[str]]) -> None:
        """对新增/修改/删除的文档执行增量更新。"""
        with self._lock:
            # 先删后改：先处理删除和修改中的旧数据
            for fname in changes["deleted"]:
                self._delete_document_chunks(fname)

            for fname in changes["modified"]:
                self._delete_document_chunks(fname)

            # 再处理新增和修改中的新数据
            for fname in changes["new"]:
                self._add_document_chunks(fname)

            for fname in changes["modified"]:
                self._add_document_chunks(fname)

    def _delete_document_chunks(self, fname: str) -> int:
        """从 ChromaDB 删除指定文档的所有 chunk。"""
        if self.chunk_collection is None:
            return 0
        try:
            result = self.chunk_collection.get(where={"filename": fname})
            ids = result.get("ids", [])
            if ids:
                self.chunk_collection.delete(ids=ids)
            logger.info(f"[KB] 删除文档 chunk: {fname} ({len(ids)} 条)")
            self._dirty_bm25 = True
            self._dirty_graph = True
            return len(ids)
        except Exception as e:
            logger.warning(f"[KB] 删除文档 chunk 失败 {fname}: {e}")
            return 0

    def _add_document_chunks(self, fname: str) -> int:
        """对单个文档分块、嵌入并加入 ChromaDB。"""
        fpath = os.path.join(self.kb_dir, fname)
        try:
            docs = load_document(fpath)
        except Exception as e:
            logger.warning(f"[KB] 加载文件失败 {fname}: {e}")
            return 0

        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        splits = splitter.split_documents(docs)

        chunk_texts = [doc.page_content for doc in splits]
        chunk_ids = [self._make_chunk_id(fname, i) for i in range(len(chunk_texts))]
        chunk_metas = [
            {
                "filename": fname,
                "chunk_id": cid,
                "chunk_index": i,
            }
            for i, cid in enumerate(chunk_ids)
        ]

        if self.chunk_collection is not None:
            self.chunk_collection.add_texts(
                texts=chunk_texts,
                ids=chunk_ids,
                metadatas=chunk_metas,
            )
        else:
            # 首次添加（理论上不会发生，防御性代码）
            self.chunk_collection = Chroma.from_texts(
                texts=chunk_texts,
                embedding=self.embeddings_model,
                ids=chunk_ids,
                metadatas=chunk_metas,
                persist_directory=self.chroma_dir,
                collection_name="chunks",
            )

        logger.info(
            f"[KB] 新增文档 chunk: {fname} ({len(chunk_texts)} 条)"
        )
        self._dirty_bm25 = True
        self._dirty_graph = True
        return len(chunk_texts)

    def _finalize_index(self) -> None:
        """增量更新结束后：重建脏索引 + 持久化 manifest。"""
        if self._dirty_bm25 or self._dirty_graph:
            self._rebuild_internal_from_chroma()
        self._save_manifest()
        self._dirty_bm25 = False
        self._dirty_graph = False
        logger.info("[KB] 增量更新完成")

    def _rebuild_internal_from_chroma(self) -> None:
        """从 ChromaDB 重建 _chunk_texts/_chunk_ids/_chunk_metas + BM25 + 图。"""
        if self.chunk_collection is None:
            return
        try:
            result = self.chunk_collection.get(include=["documents", "metadatas"])
            texts = result.get("documents", [])
            ids = result.get("ids", [])
            metas = result.get("metadatas", [])
            if not texts:
                return
            # 重建 chunk_index 元数据
            for i, meta in enumerate(metas):
                meta["chunk_index"] = i
            self._chunk_texts = texts
            self._chunk_ids = ids
            self._chunk_metas = metas

            if self._dirty_bm25:
                self._build_bm25()
            if self._dirty_graph:
                self._build_entity_graph()
        except Exception as e:
            logger.warning(f"[KB] 从 ChromaDB 重建内部索引失败: {e}")

    # ── 分词 ───────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """中英混合文本分词：英文/数字 token + 中文字符 unigram + bigram。"""
        tokens = []
        for m in re.finditer(r"[a-zA-Z0-9]+", text.lower()):
            tokens.append(m.group())
        chinese = re.findall(r"[一-鿿]", text)
        tokens.extend(chinese)
        for i in range(len(chinese) - 1):
            tokens.append(chinese[i] + chinese[i + 1])
        return tokens

    # ── BM25 索引构建 / 持久化 ───────────────────────────────────────────

    def _build_bm25(self) -> None:
        """构建 BM25Okapi 索引并序列化到 chroma_db/。"""
        if not self._chunk_texts:
            self._bm25_index = None
            return

        tokenized = [self._tokenize(t) for t in self._chunk_texts]
        self._bm25_index = BM25Okapi(tokenized)

        data = {
            "index": self._bm25_index,
            "texts": self._chunk_texts,
            "ids": self._chunk_ids,
            "metas": self._chunk_metas,
        }
        try:
            with open(self._bm25_path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"[KB] BM25 索引构建完成，{len(self._chunk_texts)} 个 chunk")
        except Exception as e:
            logger.warning(f"[KB] BM25 索引持久化失败: {e}")

    def _load_bm25(self) -> bool:
        """从 chroma_db/ 加载已有 BM25 索引。返回是否加载成功。"""
        if not os.path.exists(self._bm25_path):
            return False
        try:
            with open(self._bm25_path, "rb") as f:
                data = pickle.load(f)
            self._bm25_index = data["index"]
            self._chunk_texts = data["texts"]
            self._chunk_ids = data["ids"]
            self._chunk_metas = data["metas"]
            logger.info(f"[KB] BM25 索引加载完成（复用），{len(self._chunk_texts)} 个 chunk")
            return True
        except Exception as e:
            logger.warning(f"[KB] BM25 索引加载失败，将重建: {e}")
            self._bm25_index = None
            return False

    # ── 实体共现图构建 / 加载 ──────────────────────────────────────

    def _build_entity_graph(self) -> None:
        """构建实体共现图并持久化。"""
        if not self._chunk_texts:
            self._graph_retriever = None
            return
        self._graph_retriever = GraphRetriever()
        self._graph_retriever.build(self._chunk_texts, self._chunk_metas)
        self._graph_retriever.save(self._graph_path)

    def _load_entity_graph(self) -> bool:
        """从 pickle 加载实体图。"""
        self._graph_retriever = GraphRetriever()
        if not self._graph_retriever.load(self._graph_path):
            self._graph_retriever = None
            return False
        self._graph_retriever.chunk_texts = self._chunk_texts
        self._graph_retriever.chunk_metas = self._chunk_metas
        return True

    def _bm25_search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        """BM25 关键词检索，返回 top-k 的 [(chunk_index, score), ...]（降序）。"""
        if self._bm25_index is None or not self._chunk_texts:
            return []
        tokens = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokens)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        top = indexed[:k]
        logger.info(f"[KB] BM25 检索 top-{k}: {len(top)} 条，最高分 {top[0][1]:.2f}" if top else "[KB] BM25 检索无结果")
        return [(idx, float(score)) for idx, score in top]

    # ── RRF 融合 ────────────────────────────────────────────────────────

    def _rrf_fusion(self,
                    *ranked_lists: list[tuple[int, float]],
                    k: int = 60) -> list[tuple[int, float]]:
        """RRF 倒数排名融合，支持任意数量检索结果列表。"""
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (idx, _) in enumerate(ranked, 1):
                scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)
        merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        counts = " + ".join(str(len(r)) for r in ranked_lists)
        logger.info(f"[KB] RRF 融合: {counts} → {len(merged)} 条")
        return merged

    # ── 检索 ───────────────────────────────────────────────────────────

    RERANK_LOW_THRESHOLD = -2.0

    async def retrieve(self, question: str, query_entities: list[str] | None = None):
        """混合检索，返回 (检索内容, 置信度分数)。

        三路并行 — 语义 (dense) + BM25 (sparse) + 图检索 (graph)
        → RRF 融合 → CrossEncoder 精排。

        Args:
            query_entities: 可选的 LLM 提取实体列表，供图检索使用
        """
        if self.chunk_collection is None:
            return ("[知识库不可用] 未找到已索引的文档。", -999.0)

        loop = asyncio.get_event_loop()

        def _dense_search():
            """语义检索：ChromaDB chunk top-k。"""
            results = self.chunk_collection.similarity_search_with_score(
                question, k=20
            )
            dense_ranked: list[tuple[int, float]] = []
            for doc, distance in results:
                # 先用 chunk_id 查找，兼容无 chunk_id 的旧索引
                chunk_id = doc.metadata.get("chunk_id", "")
                if chunk_id and chunk_id in self._chunk_ids:
                    idx = self._chunk_ids.index(chunk_id)
                else:
                    # 降级：用 chunk_index
                    idx = doc.metadata.get("chunk_index")
                    if idx is None:
                        try:
                            idx = self._chunk_texts.index(doc.page_content)
                        except ValueError:
                            continue
                dense_ranked.append((int(idx), float(distance)))
            logger.info(f"[KB] 语义检索 top-20: {len(dense_ranked)} 条")
            return dense_ranked

        def _sparse_search():
            return self._bm25_search(question, k=20)

        def _graph_search():
            if self._graph_retriever is None:
                return []
            return self._graph_retriever.retrieve(
                question, top_k=40, query_entities=query_entities
            )

        dense_ranked, sparse_ranked, graph_ranked = await asyncio.gather(
            loop.run_in_executor(None, _dense_search),
            loop.run_in_executor(None, _sparse_search),
            loop.run_in_executor(None, _graph_search),
        )

        merged = self._rrf_fusion(dense_ranked, sparse_ranked, graph_ranked, k=60)
        if not merged:
            logger.warning("[KB] RRF 融合后无结果")
            return ("[知识库] 未检索到相关内容。", -999.0)

        top_n = min(10, len(merged))
        candidate_texts: list[str] = []
        for chunk_idx, _ in merged[:top_n]:
            if chunk_idx < len(self._chunk_texts):
                candidate_texts.append(self._chunk_texts[chunk_idx])

        top_score = 0.0
        contents_with_scores: list[tuple[str, float]] = []
        if self.reranker_model is not None and candidate_texts:
            ranked = self.rerank(question, candidate_texts, top_k=3, return_scores=True)
            contents_with_scores = ranked
            top_score = float(ranked[0][1]) if ranked else -999.0
        else:
            contents_with_scores = [(text, 0.0) for text in candidate_texts[:3]]

        logger.info(f"[KB] 检索完成，精排 Top 分数: {top_score:.2f}")

        parts = []
        for i, (text, score) in enumerate(contents_with_scores, 1):
            parts.append(f"--- 文档片段 {i} (score: {score:.3f}) ---\n{text}")
        return ("\n\n".join(parts), top_score)

    # ── 重排序 ─────────────────────────────────────────────────────────

    def rerank(self, query: str, docs: list[str], top_k: int = 5,
               return_scores: bool = False):
        """用 CrossEncoder 对文档片段精排。"""
        if self.reranker_model is None or len(docs) <= 1:
            if return_scores:
                return [(doc, 0.0) for doc in docs[:top_k]]
            return docs[:top_k]
        pairs = [[query, doc[:1024]] for doc in docs]
        scores = self.reranker_model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        if return_scores:
            return [(doc, float(score)) for doc, score in ranked[:top_k]]
        return [doc for doc, _ in ranked[:top_k]]

    # ── 实用方法 ───────────────────────────────────────────────────────

    def get_descriptions_text(self) -> str:
        """返回所有文档摘要的拼接字符串，供意图分类使用。"""
        if not self.descriptions:
            return "暂无已索引文档"
        return "、".join(self.descriptions.values())

    def _list_kb_files(self) -> list[str]:
        """扫描 kb_dir 下所有可加载的文档文件。"""
        if not os.path.isdir(self.kb_dir):
            return []
        return sorted(
            f for f in os.listdir(self.kb_dir)
            if os.path.isfile(os.path.join(self.kb_dir, f))
            and os.path.splitext(f)[1].lower() in _SUPPORTED_EXTS
        )
