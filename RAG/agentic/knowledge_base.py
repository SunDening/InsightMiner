"""知识库模块 — RagKnowledgeBase 类封装所有 KB 状态、持久化与检索。

特性：
  - 摘要缓存到 .kb_summaries.json，仅对新增/变更文件调 LLM 生成摘要
  - ChromaDB 持久化到 chroma_db/，通过 .manifest.json 检测变更，跳过高成本重建
  - 嵌入模型与重排序模型并发加载 (asyncio.gather + run_in_executor)
"""

import os
import json
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from langchain.messages import HumanMessage

from .config import (
    logger, KB_DIR, CHROMA_DIR, SUMMARY_CACHE_PATH,
    EMBEDDING_MODEL_NAME, _SUPPORTED_EXTS, LLM_PROVIDER,
)
from .llm import create_llm

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
        """并发加载嵌入模型和重排序模型。

        离线模式由 .env 中的 TRANSFORMERS_OFFLINE=1 / HF_HUB_OFFLINE=1 控制，
        在 config.py 的 load_dotenv() 阶段即生效，早于任何 huggingface/transformers 的 import，
        确保模型从本地缓存秒级加载。如需下载新模型，临时注释 .env 中对应行即可。
        """
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

        # 清除已删除文件的缓存
        for cached in list(self.descriptions.keys()):
            if cached not in kb_files:
                del self.descriptions[cached]
                self.file_mtimes.pop(cached, None)

        # 找出需要生成摘要的文件
        pending = []
        for fname in kb_files:
            fpath = os.path.join(self.kb_dir, fname)
            mtime = os.path.getmtime(fpath)
            if fname in self.descriptions and self.file_mtimes.get(fname) == mtime:
                continue  # 缓存命中
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

    # ── Manifest / 变更检测 ────────────────────────────────────────────

    def _compute_manifest(self) -> dict:
        """生成当前 KB 文件的 manifest 快照。"""
        files = {}
        for fname in self._list_kb_files():
            fpath = os.path.join(self.kb_dir, fname)
            files[fname] = os.path.getmtime(fpath)
        return {"files": files, "summary_count": len(self.descriptions)}

    def _manifest_unchanged(self) -> bool:
        """检查 manifest 是否与持久化版本一致。"""
        if not os.path.exists(self._manifest_path):
            return False
        if not os.path.isdir(self.chroma_dir):
            return False
        if not os.path.exists(os.path.join(self.chroma_dir, "chroma.sqlite3")):
            return False
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            return saved == self._compute_manifest()
        except (json.JSONDecodeError, FileNotFoundError):
            return False

    def _save_manifest(self) -> None:
        """将当前 manifest 写入 chroma_db/.manifest.json。"""
        os.makedirs(self.chroma_dir, exist_ok=True)
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._compute_manifest(), f, ensure_ascii=False, indent=2)

    # ── ChromaDB 索引构建 / 复用 ──────────────────────────────────────

    def _build_or_reuse_index(self) -> None:
        """构建双层索引，或从 persist_directory 加载已有索引。"""
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

        # Manifest 未变 → 从磁盘直接加载已有 ChromaDB
        if self._manifest_unchanged():
            logger.info("[KB] Manifest 未变更，加载已有 ChromaDB 索引")
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
            logger.info(f"[KB] 索引加载完成（复用已有）")
            return

        # Manifest 变更 → 删除旧数据重建
        if os.path.exists(self.chroma_dir):
            shutil.rmtree(self.chroma_dir)
            os.makedirs(self.chroma_dir)

        logger.info("[KB] 开始构建双层索引...")

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

        # ── 片段索引 ──
        all_docs = []
        for fname in kb_files:
            fpath = os.path.join(self.kb_dir, fname)
            try:
                all_docs.extend(load_document(fpath))
            except Exception as e:
                logger.warning(f"[KB] 加载文件 {fname} 失败: {e}")

        if not all_docs:
            self.chunk_collection = None
            self._save_manifest()
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
        splits = splitter.split_documents(all_docs)

        chunk_texts = [doc.page_content for doc in splits]
        chunk_ids = [f"chunk_{i}" for i in range(len(chunk_texts))]
        chunk_metas = []
        for doc in splits:
            src = doc.metadata.get("source", "")
            fname = os.path.basename(src) if src else "unknown"
            chunk_metas.append({"filename": fname})

        self.chunk_collection = Chroma.from_texts(
            texts=chunk_texts,
            embedding=self.embeddings_model,
            ids=chunk_ids,
            metadatas=chunk_metas,
            persist_directory=self.chroma_dir,
            collection_name="chunks",
        )
        logger.info(f"[KB] 片段索引就绪，{len(chunk_texts)} 个 chunk，{len(kb_files)} 个文档")

        self._save_manifest()

    # ── 检索 ───────────────────────────────────────────────────────────

    # ChromaDB 默认余弦距离，1.0 = 正交（完全不相关），0.0 = 完全相同
    SUMMARY_DISTANCE_THRESHOLD = 1.0

    # bge-reranker-base 经验阈值：>0 相关，-2~0 边界，<-2 不相关
    RERANK_LOW_THRESHOLD = -2.0

    def retrieve(self, question: str):
        """分层检索，返回 (检索内容, 置信度分数)。

        Stage1 摘要匹配 → 命中则限定文档内 Stage2 精搜；
        Stage1 全挂 → 全量 Stage2 精搜（不限定文档）+ CrossEncoder 评分。
        置信度用于 synthesize 阶段判断是否标注"LLM 生成"。
        """
        if self.summary_collection is None or self.chunk_collection is None:
            return ("[知识库不可用] 未找到已索引的文档。", -999.0)

        # Stage 1: 摘要匹配
        max_summaries = min(5, self.summary_collection._collection.count())
        summary_results = self.summary_collection.similarity_search_with_score(
            question, k=max_summaries
        )
        matched_files: list[str] = []
        for doc, distance in summary_results:
            if distance < self.SUMMARY_DISTANCE_THRESHOLD:
                fname = doc.metadata.get("filename", "")
                if fname and fname not in matched_files:
                    matched_files.append(fname)
        matched_set = set(matched_files)
        logger.info(
            f"[KB] Stage1 摘要匹配 → 命中 {len(matched_files)}/{max_summaries} 个文档"
            f"（阈值 {self.SUMMARY_DISTANCE_THRESHOLD}）: {matched_files}"
        )

        # Stage 2: chunk 精搜
        raw_chunks = self.chunk_collection.similarity_search(question, k=10)

        if matched_files:
            # 有命中文档 → 限定范围内检索
            filtered = [doc for doc in raw_chunks if doc.metadata.get("filename", "") in matched_set]
            if len(filtered) < 3:
                filtered = raw_chunks
                logger.info(f"[KB] Stage2 过滤后仅 {len(filtered)} 条，放宽到全文档")
        else:
            # Stage1 全挂 → 不限文档，全量检索
            filtered = raw_chunks
            logger.info("[KB] Stage1 摘要全部未命中，Stage2 全量检索")

        contents = [doc.page_content for doc in filtered]
        top_score = 0.0

        if self.reranker_model is not None:
            ranked = self.rerank(question, contents, top_k=3, return_scores=True)
            contents = [doc for doc, _ in ranked]
            top_score = float(ranked[0][1]) if ranked else -999.0
        else:
            contents = contents[:3]

        logger.info(f"[KB] 检索完成，精排 Top 分数: {top_score:.2f}")

        parts = []
        for i, text in enumerate(contents, 1):
            parts.append(f"--- 文档片段 {i} ---\n{text}")
        return ("\n\n".join(parts), top_score)

    # ── 重排序 ─────────────────────────────────────────────────────────

    def rerank(self, query: str, docs: list[str], top_k: int = 5,
               return_scores: bool = False):
        """用 CrossEncoder 对文档片段精排。

        Args:
            return_scores: 为 True 时返回 List[Tuple[str, float]]，否则返回 List[str]
        """
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
