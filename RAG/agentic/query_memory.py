"""Query Memory — 记录成功查询作为 few-shot 示例。

将成功的 (问题, SQL, 表列表) 存入 ChromaDB。
新查询时检索相似历史查询，注入 SQL prompt 作为少数示例参考。
"""

import os
import json
import time

from langchain_chroma import Chroma

from .config import logger, QUERY_MEMORY_PATH, MAX_FEWSHOT_EXAMPLES


class QueryMemory:
    """查询记忆库，存储成功查询并提供语义检索。

    ChromaDB collection: query_history
    复用外部嵌入模型（通常来自 KB 的 HuggingFaceEmbeddings）。
    """

    def __init__(self, persist_dir: str = QUERY_MEMORY_PATH,
                 embeddings_model=None,
                 max_examples: int = MAX_FEWSHOT_EXAMPLES):
        self.persist_dir = persist_dir
        self.embeddings_model = embeddings_model
        self.max_examples = max_examples
        self.collection: Chroma | None = None

    def initialize(self) -> None:
        """初始化 ChromaDB collection。"""
        if self.embeddings_model is None:
            logger.warning("[QueryMemory] 嵌入模型不可用，跳过初始化")
            return
        os.makedirs(self.persist_dir, exist_ok=True)
        self.collection = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings_model,
            collection_name="query_history",
        )
        logger.info(f"[QueryMemory] 初始化完成，collection=query_history")

    def add(self, question: str, rewritten: str, sql: str,
            tables: list[str]) -> None:
        """记录一条成功查询。"""
        if self.collection is None:
            return
        table_count = len(tables) if tables else 0
        doc = json.dumps({
            "question": question,
            "rewritten": rewritten,
            "sql": sql,
            "tables": tables,
            "table_count": table_count,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False)
        # 用 question 作为嵌入输入
        meta = {
            "tables": ",".join(tables) if tables else "",
            "table_count": table_count,
        }
        try:
            self.collection.add_texts(
                texts=[question],
                metadatas=[meta],
                ids=[f"q_{int(time.time() * 1000000)}"],
            )
            logger.info(f"[QueryMemory] 记录查询: {question[:60]}...")
        except Exception as e:
            logger.warning(f"[QueryMemory] 记录失败: {e}")

    def search(self, question: str, k: int | None = None) -> str:
        """检索与当前查询最相似的历史成功查询。

        优先返回表数量少且语义相似的查询示例。

        Returns:
            格式化的 few-shot 示例文本（可直接注入 prompt），
            如无命中则返回空字符串。
        """
        if self.collection is None:
            return ""
        k = k or self.max_examples
        # 多取一些候选，用于重排序
        fetch_k = max(k * 3, 10)
        try:
            results = self.collection.similarity_search_with_score(question, k=fetch_k)
        except Exception as e:
            logger.warning(f"[QueryMemory] 检索失败: {e}")
            return ""

        if not results:
            return ""

        # 解析候选并计算重排序分数：相似度 + 简单查询偏好
        candidates = []
        for doc, sim_score in results:
            try:
                entry = json.loads(doc.page_content)
            except json.JSONDecodeError:
                continue
            tc = entry.get("table_count", 99)
            # 表数量越少优先级越高，表数量为 0 或缺失则放在最后
            if tc <= 0:
                tc = 99
            # 简单偏好：表越少加分越多（1张表 +0.5, 3张表 +0.17, 5张表 +0.1）
            simplicity_bonus = 0.5 / tc
            adjusted_score = sim_score + simplicity_bonus
            candidates.append((adjusted_score, entry))

        # 按调整后的分数升序（分数越低越好，因为 similarity_search_with_score 返回的是距离）
        candidates.sort(key=lambda x: x[0])

        examples = []
        for _, entry in candidates:
            examples.append(
                f"  问题: {entry.get('question', '')}\n"
                f"  SQL: {entry.get('sql', '')}\n"
                f"  使用的表: {', '.join(entry.get('tables', []))}\n"
            )
            if len(examples) >= self.max_examples:
                break

        if not examples:
            return ""

        header = (
            f"\n\n## 历史相似查询（共 {len(examples)} 条，"
            f"优先简单查询，请参考其 SQL 风格和表选择逻辑）\n"
        )
        return header + "\n".join(examples)
