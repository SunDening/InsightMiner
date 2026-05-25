"""Schema Indexer — 从 JSON 构建表/列索引，按查询动态检索相关 Schema。

双 ChromaDB 索引 + FK 关系图，检索管线复用 KB 的 dense+BM25→RRF→CrossEncoder 模式。
"""

import os
import re
import json
import pickle
import shutil
import asyncio

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from .config import (
    logger, SCHEMA_JSON_PATH, TABLE_DESC_JSON_PATH,
    SCHEMA_CHROMA_DIR, SCHEMA_ENRICHED_PATH,
    MAX_SCHEMA_TABLES, MAX_SCHEMA_COLUMNS_PER_TABLE,
    MAX_SCHEMA_TABLES_BASE, MAX_FK_EXPAND, MAX_SCHEMA_TABLES_CAP,
    MAX_SCHEMA_CHARS_CAP,
    ALLOWED_JOIN_TABLES,
    GARBAGE_TABLES,
)


class SchemaIndexer:
    """管理数据库 Schema 元数据的索引与检索。

    数据来源：schema.json（列名、类型、主键、关系、中英文描述）
             table_desc.json（表名 + 模块级业务描述）

    双层 ChromaDB 索引：
      - table_index：表级文档，用于"该查询需要哪几张表"
      - column_index：列级文档，用于"这些表里哪几列相关"

    外键关系图 (fk_graph)：用于自动发现表间 Join 路径。
    """

    def __init__(self, schema_json_path: str = SCHEMA_JSON_PATH,
                 table_desc_path: str = TABLE_DESC_JSON_PATH,
                 chroma_dir: str = SCHEMA_CHROMA_DIR,
                 embeddings_model=None, reranker_model=None):
        self.schema_json_path = schema_json_path
        self.table_desc_path = table_desc_path
        self.chroma_dir = chroma_dir

        # 外部注入的模型（可复用 KB 已加载的实例）
        self.embeddings_model = embeddings_model
        self.reranker_model = reranker_model

        # ChromaDB 索引
        self.table_index: Chroma | None = None
        self.column_index: Chroma | None = None

        # 元数据
        self.tables_meta: dict[str, dict] = {}       # name → {description, module, columns, pk}
        self.fk_graph: dict[str, list[tuple]] = {}   # table → [(col, target_table, target_col)]

        # BM25 索引
        self._table_bm25: BM25Okapi | None = None
        self._table_texts: list[str] = []
        self._table_names: list[str] = []
        self._col_bm25: BM25Okapi | None = None
        self._col_texts: list[str] = []
        self._col_refs: list[tuple] = []  # [(table_name, column_name), ...]

        # 持久化路径
        self._manifest_path = os.path.join(chroma_dir, ".schema_manifest.json")
        self._bm25_table_path = os.path.join(chroma_dir, "bm25_tables.pkl")
        self._bm25_col_path = os.path.join(chroma_dir, "bm25_columns.pkl")

    # ── 公共生命周期 ──────────────────────────────────────────────────

    async def initialize(self) -> None:
        """一站式启动：加载 JSON → 构建/复用索引 → 构建 FK 图。"""
        os.makedirs(self.chroma_dir, exist_ok=True)

        self._load_metadata()
        self._build_or_reuse_indices()
        self._build_fk_graph()
        logger.info(f"[SchemaIndexer] 初始化完成，{len(self.tables_meta)} 张表，"
                    f"{len(self.fk_graph)} 条 FK 关系")

    # ── 元数据加载 ────────────────────────────────────────────────────

    def _load_metadata(self) -> None:
        """从 schema.json + table_desc.json 加载并合并元数据。

        优先使用 .schema_enriched.json 中的 LLM 富化描述，
        如不存在则回退到原始 schema.json。
        """
        enriched_data = None
        if os.path.exists(SCHEMA_ENRICHED_PATH):
            with open(SCHEMA_ENRICHED_PATH, "r", encoding="utf-8") as f:
                enriched_data = json.load(f)
            logger.info("[SchemaIndexer] 加载 LLM 富化 Schema 描述")

        if enriched_data is None:
            if not os.path.exists(self.schema_json_path):
                logger.warning(
                    f"[SchemaIndexer] schema.json 不存在: {self.schema_json_path}"
                )
                return
            logger.info("[SchemaIndexer] 未找到富化描述，使用原始 schema.json")
            self._load_from_raw_json()
            return

        # 从富化数据构建 tables_meta
        for t in enriched_data:
            name = t["name"]
            if name in GARBAGE_TABLES:
                continue

            columns = []
            for c in t.get("columns", []):
                col_name = c["name"]
                # 合并富化列描述
                col_enriched_desc = (
                    t.get("col_enriched_desc", {}).get(col_name, "")
                )
                columns.append({
                    "name": col_name,
                    "type": c.get("type", "VARCHAR"),
                    "size": c.get("size"),
                    "nullable": c.get("nullable", True),
                    "primary_key": c.get("primary_key", False),
                    "description_en": c.get("description_en", ""),
                    "description_zh": c.get("description_zh", ""),
                    "enriched_desc": col_enriched_desc,
                })

            self.tables_meta[name] = {
                "description": t.get("description", ""),
                "module": t.get("module", ""),
                "primary_key": t.get("primary_key", []),
                "columns": columns,
                "relationships": t.get("relationships", []),
                "enriched_desc": t.get("table_enriched_desc", ""),
            }

        logger.info(
            f"[SchemaIndexer] 已加载 {len(self.tables_meta)} 张表的元数据"
            f"（含 LLM 富化描述）"
        )

    def _load_from_raw_json(self) -> None:
        """回退方案：从原始 schema.json + table_desc.json 加载。"""
        with open(self.schema_json_path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)

        table_descriptions: dict[str, str] = {}
        table_modules: dict[str, str] = {}
        if os.path.exists(self.table_desc_path):
            with open(self.table_desc_path, "r", encoding="utf-8") as f:
                desc_data = json.load(f)
            for module in desc_data.get("modules", []):
                mod_name = module.get("module_name", "")
                for t in module.get("tables", []):
                    table_descriptions[t["table_name"]] = t.get("description", "")
                    table_modules[t["table_name"]] = mod_name

        for t in schema_data.get("tables", []):
            name = t["name"]
            if name in GARBAGE_TABLES:
                continue
            columns = []
            for c in t.get("columns", []):
                columns.append({
                    "name": c["name"],
                    "type": c.get("type", "VARCHAR"),
                    "size": c.get("size"),
                    "nullable": c.get("nullable", True),
                    "primary_key": c.get("primary_key", False),
                    "description_en": c.get("description_en", ""),
                    "description_zh": c.get("description_zh", ""),
                    "enriched_desc": "",
                })
            self.tables_meta[name] = {
                "description": table_descriptions.get(name, ""),
                "module": table_modules.get(name, ""),
                "primary_key": t.get("primary_key", []),
                "columns": columns,
                "relationships": t.get("relationships", []),
                "enriched_desc": "",
            }

    # ── Manifest ──────────────────────────────────────────────────────

    def _compute_manifest(self) -> dict:
        return {
            "schema_mtime": os.path.getmtime(self.schema_json_path),
            "table_desc_mtime": os.path.getmtime(self.table_desc_path)
            if os.path.exists(self.table_desc_path) else 0,
            "table_count": len(self.tables_meta),
        }

    def _manifest_unchanged(self) -> bool:
        if not os.path.exists(self._manifest_path):
            return False
        if not os.path.isdir(self.chroma_dir):
            return False
        if not os.path.exists(os.path.join(self.chroma_dir, "chroma.sqlite3")):
            return False
        if not os.path.exists(self._bm25_table_path):
            return False
        try:
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            return saved == self._compute_manifest()
        except (json.JSONDecodeError, FileNotFoundError):
            return False

    def _save_manifest(self) -> None:
        os.makedirs(self.chroma_dir, exist_ok=True)
        with open(self._manifest_path, "w", encoding="utf-8") as f:
            json.dump(self._compute_manifest(), f, ensure_ascii=False, indent=2)

    # ── 索引构建 / 复用 ──────────────────────────────────────────────

    def _build_or_reuse_indices(self) -> None:
        """构建表级 + 列级 ChromaDB 索引，或从磁盘复用。"""
        if self.embeddings_model is None:
            logger.warning("[SchemaIndexer] 嵌入模型不可用，跳过索引构建")
            return
        if not self.tables_meta:
            return

        if self._manifest_unchanged():
            logger.info("[SchemaIndexer] Manifest 未变更，加载已有索引")
            self.table_index = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embeddings_model,
                collection_name="schema_tables",
            )
            self.column_index = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embeddings_model,
                collection_name="schema_columns",
            )
            if not self._load_bm25():
                self._rebuild_bm25_from_chroma()
            logger.info(f"[SchemaIndexer] 索引加载完成（复用）")
            return

        # Manifest 变更 → 重建
        if os.path.exists(self.chroma_dir):
            shutil.rmtree(self.chroma_dir)
            os.makedirs(self.chroma_dir)

        logger.info("[SchemaIndexer] 开始构建 Schema 索引...")

        # ── 表级文档 ──
        table_texts, table_ids, table_metas = [], [], []
        for name, meta in self.tables_meta.items():
            enriched = meta.get("enriched_desc", "")
            col_names = ", ".join(c["name"] for c in meta["columns"])
            if enriched:
                # 优先使用 LLM 富化描述
                text = enriched
            else:
                # 回退：技术描述
                text = (
                    f"{name} | {meta['description']} | {meta['module']} | "
                    f"COLUMNS: {col_names}"
                )
            table_texts.append(text)
            table_ids.append(name)
            table_metas.append({"table_name": name})

        self._table_texts = table_texts
        self._table_names = list(self.tables_meta.keys())

        self.table_index = Chroma.from_texts(
            texts=table_texts, embedding=self.embeddings_model,
            ids=table_ids, metadatas=table_metas,
            persist_directory=self.chroma_dir, collection_name="schema_tables",
        )
        logger.info(f"[SchemaIndexer] 表级索引就绪，{len(table_texts)} 张表")

        # ── 列级文档 ──
        col_texts, col_ids, col_metas, col_refs = [], [], [], []
        for name, meta in self.tables_meta.items():
            for c in meta["columns"]:
                enriched_col = c.get("enriched_desc", "")
                if enriched_col:
                    text = enriched_col
                else:
                    desc_parts = []
                    if c["description_zh"]:
                        desc_parts.append(c["description_zh"])
                    if c["description_en"]:
                        desc_parts.append(c["description_en"])
                    desc = " | ".join(desc_parts)
                    text = f"{name}.{c['name']} | {c['type']} | {desc}"
                col_texts.append(text)
                col_ids.append(f"{name}.{c['name']}")
                col_metas.append({"table_name": name, "column_name": c["name"]})
                col_refs.append((name, c["name"]))

        self._col_texts = col_texts
        self._col_refs = col_refs

        self.column_index = Chroma.from_texts(
            texts=col_texts, embedding=self.embeddings_model,
            ids=col_ids, metadatas=col_metas,
            persist_directory=self.chroma_dir, collection_name="schema_columns",
        )
        logger.info(f"[SchemaIndexer] 列级索引就绪，{len(col_texts)} 列")

        self._build_bm25()
        self._save_bm25()
        self._save_manifest()

    # ── 分词（同 knowledge_base.py）───────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = []
        for m in re.finditer(r"[a-zA-Z0-9_]+", text.lower()):
            tokens.append(m.group())
        chinese = re.findall(r"[一-鿿]", text)
        tokens.extend(chinese)
        for i in range(len(chinese) - 1):
            tokens.append(chinese[i] + chinese[i + 1])
        return tokens

    # ── BM25 ─────────────────────────────────────────────────────────

    def _build_bm25(self) -> None:
        if self._table_texts:
            self._table_bm25 = BM25Okapi([self._tokenize(t) for t in self._table_texts])
        if self._col_texts:
            self._col_bm25 = BM25Okapi([self._tokenize(t) for t in self._col_texts])

    def _save_bm25(self) -> None:
        for path, data in [
            (self._bm25_table_path, (self._table_bm25, self._table_texts, self._table_names)),
            (self._bm25_col_path, (self._col_bm25, self._col_texts, self._col_refs)),
        ]:
            try:
                with open(path, "wb") as f:
                    pickle.dump(data, f)
            except Exception as e:
                logger.warning(f"[SchemaIndexer] BM25 序列化失败: {e}")

    def _load_bm25(self) -> bool:
        for path, is_ok in [
            (self._bm25_table_path, False), (self._bm25_col_path, False),
        ]:
            if not os.path.exists(path):
                return False
        try:
            with open(self._bm25_table_path, "rb") as f:
                self._table_bm25, self._table_texts, self._table_names = pickle.load(f)
            with open(self._bm25_col_path, "rb") as f:
                self._col_bm25, self._col_texts, self._col_refs = pickle.load(f)
            logger.info(f"[SchemaIndexer] BM25 索引加载完成（复用）")
            return True
        except Exception as e:
            logger.warning(f"[SchemaIndexer] BM25 加载失败: {e}")
            return False

    def _rebuild_bm25_from_chroma(self) -> None:
        if self.table_index is None:
            return
        try:
            result = self.table_index.get(include=["documents"])
            self._table_texts = result.get("documents", [])
            self._table_names = [m.get("table_name", "") for m in result.get("metadatas", [])]
        except Exception:
            pass
        try:
            result = self.column_index.get(include=["documents", "metadatas"])
            self._col_texts = result.get("documents", [])
            metas = result.get("metadatas", [])
            self._col_refs = [(m.get("table_name", ""), m.get("column_name", "")) for m in metas]
        except Exception:
            pass
        self._build_bm25()
        self._save_bm25()

    # ── BM25 检索 ────────────────────────────────────────────────────

    def _bm25_search(self, query: str, category: str = "table", k: int = 20
                     ) -> list[tuple[int, float]]:
        bm25 = self._table_bm25 if category == "table" else self._col_bm25
        if bm25 is None:
            return []
        tokens = self._tokenize(query)
        scores = bm25.get_scores(tokens)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in indexed[:k]]

    # ── RRF 融合（同 knowledge_base.py）───────────────────────────────

    @staticmethod
    def _rrf_fusion(dense_ranked: list[tuple[int, float]],
                    sparse_ranked: list[tuple[int, float]],
                    k: int = 60) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for rank, (idx, _) in enumerate(dense_ranked, 1):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)
        for rank, (idx, _) in enumerate(sparse_ranked, 1):
            scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # ── 重排序 ───────────────────────────────────────────────────────

    def _rerank(self, query: str, docs: list[str], top_k: int = 5) -> list[str]:
        if self.reranker_model is None or len(docs) <= 1:
            return docs[:top_k]
        pairs = [[query, doc[:1024]] for doc in docs]
        scores = self.reranker_model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    # ── FK 关系图 ───────────────────────────────────────────────────

    def _build_fk_graph(self) -> None:
        """从 schema.json 的 relationships 构建外键邻接表。"""
        self.fk_graph.clear()
        for name, meta in self.tables_meta.items():
            self.fk_graph.setdefault(name, [])
            seen = set()
            for rel in meta.get("relationships", []):
                col = rel.get("column", "")
                target = rel.get("target_table", "")
                target_col = rel.get("target_column", "")
                if col and target:
                    key = (col, target, target_col)
                    if key not in seen:
                        seen.add(key)
                        self.fk_graph[name].append(key)

        logger.info(f"[SchemaIndexer] FK 图就绪，{sum(len(v) for v in self.fk_graph.values())} 条边")

    # ── Join Path 发现 ───────────────────────────────────────────────

    def _find_join_paths(self, selected_tables: list[str]) -> list[dict]:
        """BFS 计算选中表集合之间的最小连接路径。

        Returns:
            [{from_table, from_col, to_table, to_col}, ...]
            每个元素代表一条需要显式告知 LLM 的 Join 条件。
        """
        if len(selected_tables) <= 1:
            return []

        # BFS 从第一个表出发，逐步连接其余表
        connected = {selected_tables[0]}
        joins = []
        remaining = list(selected_tables[1:])

        for target in remaining:
            # BFS 找到最短路径
            path = self._bfs_shortest_path(connected, target)
            if path:
                for i in range(len(path) - 1):
                    src, dst = path[i], path[i + 1]
                    # 找到 src → dst 的具体列
                    for col, tgt_tbl, tgt_col in self.fk_graph.get(src, []):
                        if tgt_tbl == dst:
                            joins.append({
                                "from_table": src, "from_col": col,
                                "to_table": dst, "to_col": tgt_col,
                            })
                            break
                    else:
                        # 反向找 dst → src
                        for col, tgt_tbl, tgt_col in self.fk_graph.get(dst, []):
                            if tgt_tbl == src:
                                joins.append({
                                    "from_table": dst, "from_col": col,
                                    "to_table": src, "to_col": tgt_col,
                                })
                                break
                connected.add(target)

        return joins

    def _bfs_shortest_path(self, start_set: set, target: str) -> list[str] | None:
        """BFS 从 start_set 中任意节点出发，找到到达 target 的最短路径。"""
        if target in start_set:
            return [target]

        visited = {t: None for t in start_set}
        queue = list(start_set)

        # 构建双向邻接（Symmetrize FK graph）
        neighbors: dict[str, list[str]] = {}
        for tbl, edges in self.fk_graph.items():
            neighbors.setdefault(tbl, [])
            for _, tgt, _ in edges:
                if tgt in self.fk_graph or tgt in neighbors:
                    neighbors.setdefault(tbl, []).append(tgt)
                    neighbors.setdefault(tgt, []).append(tbl)

        for node in queue[:]:
            if node not in visited:
                visited[node] = None

        while queue:
            current = queue.pop(0)
            for neighbor in neighbors.get(current, []):
                if neighbor not in visited:
                    visited[neighbor] = current
                    queue.append(neighbor)
                    if neighbor == target:
                        # 回溯路径
                        path = [target]
                        while path[-1] not in start_set:
                            path.append(visited[path[-1]])
                        path.reverse()
                        return path
        return None

    # ═══════════════════════════════════════════════════════════════════
    # 桥接表发现
    # ═══════════════════════════════════════════════════════════════════

    def find_bridge_tables(self, target_tables: list[str]) -> list[str]:
        """给定用户指定的表集合，找出连接它们所需的桥接表。

        仅在 ALLOWED_JOIN_TABLES 范围内寻找桥接表。
        """
        if len(target_tables) <= 1:
            return []

        # 构建双向邻接图（仅白名单内的表）
        neighbors: dict[str, set[str]] = {}
        for tbl, edges in self.fk_graph.items():
            if tbl not in ALLOWED_JOIN_TABLES:
                continue
            neighbors.setdefault(tbl, set())
            for _, tgt, _ in edges:
                if tgt in self.fk_graph and tgt in ALLOWED_JOIN_TABLES:
                    neighbors[tbl].add(tgt)
                    neighbors.setdefault(tgt, set()).add(tbl)

        # 检查每对指定的表之间是否可以直连
        bridges = set()
        for i, t1 in enumerate(target_tables):
            for t2 in target_tables[i + 1:]:
                # 直连检查
                t1_neighbors = neighbors.get(t1, set())
                t2_neighbors = neighbors.get(t2, set())
                if t2 in t1_neighbors or t1 in t2_neighbors:
                    continue  # 可直连，不需要桥接表

                # BFS 找最短路径，中间节点即为桥接表
                path = self._bfs_to(t1, t2, neighbors)
                if path and len(path) > 2:
                    # path[0]=t1, path[-1]=t2, 中间的都是桥接表
                    for mid in path[1:-1]:
                        if mid not in target_tables:
                            bridges.add(mid)

        return list(bridges)

    def _bfs_to(self, start: str, target: str,
                neighbors: dict[str, set[str]]) -> list[str] | None:
        """BFS 找 start → target 最短路径。"""
        if start == target:
            return [start]
        visited = {start: None}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for nb in neighbors.get(current, set()):
                if nb not in visited:
                    visited[nb] = current
                    queue.append(nb)
                    if nb == target:
                        path = [target]
                        while path[-1] != start:
                            path.append(visited[path[-1]])
                        path.reverse()
                        return path
        return None

    def _expand_fk_neighbors(self, selected: list[str],
                             max_expand: int) -> list[str]:
        """沿 FK 关系扩展 1-hop 邻居表。

        对已选中的每张表，收集其外键邻居，按连通度降序排列，
        取 top-N 作为扩展表。高连通度的桥接表（如 notice）自然优先。

        Args:
            selected: 已选中的表名列表
            max_expand: 最大扩展表数

        Returns:
            按优先级排列的邻居表名列表（不包含已选中的表）
        """
        neighbor_scores: dict[str, int] = {}
        for tbl in selected:
            for _, tgt, _ in self.fk_graph.get(tbl, []):
                if (tgt not in selected and tgt in self.fk_graph
                        and tgt in ALLOWED_JOIN_TABLES):
                    neighbor_scores[tgt] = neighbor_scores.get(tgt, 0) + 1
            # 反向：哪些表指向 selected 中的表
            for src, edges in self.fk_graph.items():
                for _, tgt, _ in edges:
                    if (tgt == tbl and src not in selected
                            and src in self.fk_graph
                            and src in ALLOWED_JOIN_TABLES):
                        neighbor_scores[src] = neighbor_scores.get(src, 0) + 1

        # 按连通度降序
        ranked = sorted(neighbor_scores.items(), key=lambda x: x[1], reverse=True)
        result = [tbl for tbl, _ in ranked[:max_expand]]
        logger.info(
            f"[SchemaIndexer] FK 扩展 {len(selected)} →  "
            f"+{len(result)} 张邻居表"
        )
        return result

    # ═══════════════════════════════════════════════════════════════════
    # 公共 API
    # ═══════════════════════════════════════════════════════════════════

    def get_table_list_text(self) -> str:
        """返回所有表名及描述的紧凑字符串，供意图分类使用。"""
        if not self.tables_meta:
            return "暂无数据库表信息"
        parts = []
        for name, meta in self.tables_meta.items():
            desc = meta.get("description", "")
            parts.append(f"{name}（{desc}）" if desc else name)
        return "、".join(parts)

    async def build_schema_context(self, query: str,
                                    force_tables: list[str] | None = None,
                                    entity_scope: list[str] | None = None) -> str:
        """动态检索并组装相关 Schema 上下文。

        Pipeline:
          1. 表级混合检索 (dense + BM25 → RRF) → top-K 表
          2. 列级混合检索 → 每表 top-M 列 + PK/FK 列
          3. FK 图计算 Join 路径
          4. 组装为 LLM 可读的文本块

        Args:
            query: 用户查询
            force_tables: 可选，强制包含的表名列表（放在检索结果最前面）
            entity_scope: 可选，限定检索范围的表名列表（由 EntityRouter 提供）
        """
        if self.table_index is None or not self.tables_meta:
            return "（Schema 索引不可用）"

        force_set = set(t.strip() for t in (force_tables or [])
                        if t.strip() in self.tables_meta)
        scope_set = set(t.strip() for t in (entity_scope or [])
                        if t.strip() in self.tables_meta) if entity_scope else None

        loop = asyncio.get_event_loop()

        # ── Stage 1: 表级检索 ──
        def _dense_table_search():
            results = self.table_index.similarity_search_with_score(query, k=20)
            ranked = []
            for doc, distance in results:
                tbl_name = doc.metadata.get("table_name", "")
                if tbl_name in self.tables_meta:
                    idx = self._table_names.index(tbl_name) if tbl_name in self._table_names else -1
                    if idx >= 0:
                        ranked.append((idx, float(distance)))
            return ranked

        dense_table, sparse_table = await asyncio.gather(
            loop.run_in_executor(None, _dense_table_search),
            loop.run_in_executor(None, lambda: self._bm25_search(query, "table", 20)),
        )

        merged = self._rrf_fusion(dense_table, sparse_table, k=60)

        # 优先从 JOIN 白名单中找表，不够再补其他表
        if ALLOWED_JOIN_TABLES:
            allowed_merged = [(idx, score) for idx, score in merged
                              if idx < len(self._table_names)
                              and self._table_names[idx] in ALLOWED_JOIN_TABLES]
            other_merged = [(idx, score) for idx, score in merged
                            if idx < len(self._table_names)
                            and self._table_names[idx] not in ALLOWED_JOIN_TABLES]
            # 白名单表排前面，其他表排后面
            merged = allowed_merged + other_merged

        # 实体范围过滤：限定在 entity_scope 内的表
        if scope_set:
            filtered = [(idx, score) for idx, score in merged
                        if idx < len(self._table_names)
                        and self._table_names[idx] in scope_set]
            if filtered:
                merged = filtered
            else:
                logger.info("[SchemaIndexer] 实体范围内无匹配表，使用全库检索")

        if not merged:
            return "（未找到相关表，请检查查询）"

        # 取 top-K 张表
        # 自适应 K: 用户指定表 → 扩大槽位；FK 扩展 → 补充邻域表
        base_k = min(MAX_SCHEMA_TABLES_BASE, len(merged))
        # 用户指定表不计入检索基线
        forced_count = len(force_set)
        top_k = base_k + forced_count  # 基线 + 用户强制表
        # 安全阀
        top_k = min(MAX_SCHEMA_TABLES_CAP, top_k, len(merged) + forced_count)
        selected_tables = []
        candidate_texts = []
        for idx, _ in merged[:max(top_k + 5, 10)]:
            if idx < len(self._table_names):
                name = self._table_names[idx]
                if name not in selected_tables:
                    selected_tables.append(name)
                    candidate_texts.append(self._table_texts[idx])
            if len(selected_tables) >= top_k + 5:
                break

        # CrossEncoder 精排表
        if self.reranker_model is not None and len(candidate_texts) > 1:
            # 保存候选文本对应的表名（rerank 后通过文本内容回查表名）
            candidate_map = {text: name for text, name in
                             zip(candidate_texts, selected_tables)}
            candidate_texts = self._rerank(query, candidate_texts, top_k=top_k)
            ranked_names = []
            for text in candidate_texts:
                name = candidate_map.get(text, "")
                if name and name in selected_tables:
                    ranked_names.append(name)
            selected_tables = ranked_names[:top_k]
        else:
            selected_tables = selected_tables[:top_k]

        # 强制包含用户指定的表（放在最前面）
        for ft in reversed(list(force_set)):
            if ft in selected_tables:
                selected_tables.remove(ft)
            selected_tables.insert(0, ft)

        # FK 邻域扩展：沿外键扩展 1-hop，发现间接相关的表
        if len(selected_tables) < MAX_SCHEMA_TABLES_CAP:
            expanded = self._expand_fk_neighbors(selected_tables, MAX_FK_EXPAND)
            for tbl in expanded:
                if tbl not in selected_tables:
                    selected_tables.append(tbl)

        # 截断到安全上限
        selected_tables = selected_tables[:MAX_SCHEMA_TABLES_CAP]

        if not selected_tables:
            return "（未找到相关表）"

        # ── Stage 2: 列级检索 ──
        selected_columns: dict[str, list[str]] = {}
        fk_columns: dict[str, set[str]] = {}

        for tbl in selected_tables:
            meta = self.tables_meta.get(tbl)
            if meta is None:
                continue

            # 固定包含：所有主键列
            must_include = set(meta.get("primary_key", []))
            selected_columns[tbl] = list(must_include)

            # 固定包含：所有 FK 列（用于 join）
            fk_columns[tbl] = set()
            for col, _, _ in self.fk_graph.get(tbl, []):
                fk_columns[tbl].add(col)
                must_include.add(col)

            # 收集已包含的列名（去重用）
            already = must_include.copy()

            # 列级检索：在列索引中搜索
            def _dense_col_search():
                # 用 "{table_name}" 过滤会太限制，这里先做全局列检索
                results = self.column_index.similarity_search_with_score(
                    f"{tbl} {query}", k=30
                )
                ranked = []
                for doc, distance in results:
                    col_tbl = doc.metadata.get("table_name", "")
                    if col_tbl == tbl:
                        col_name = doc.metadata.get("column_name", "")
                        if col_name and col_name not in already:
                            try:
                                idx = self._col_texts.index(doc.page_content)
                            except ValueError:
                                continue
                            ranked.append((idx, float(distance)))
                return ranked

            # 全局列检索（用表名限定 BM25）
            def _sparse_col_search():
                results = self._bm25_search(f"{tbl} {query}", "column", k=50)
                filtered = []
                for idx, score in results:
                    if idx < len(self._col_refs):
                        ref_tbl, col_name = self._col_refs[idx]
                        if ref_tbl == tbl and col_name not in already:
                            filtered.append((idx, score))
                return filtered

            dense_col, sparse_col = await asyncio.gather(
                loop.run_in_executor(None, _dense_col_search),
                loop.run_in_executor(None, _sparse_col_search),
            )

            col_merged = self._rrf_fusion(dense_col, sparse_col, k=60)

            # 取列候选文本用于精排
            col_candidates = []
            col_candidate_names = []
            for idx, _ in col_merged[:10]:
                if idx < len(self._col_refs):
                    ref_tbl, col_name = self._col_refs[idx]
                    if ref_tbl == tbl and col_name not in already:
                        col_candidates.append(self._col_texts[idx])
                        col_candidate_names.append(col_name)

            if self.reranker_model is not None and len(col_candidates) > 1:
                # 保存候选文本对应的列名
                col_candidate_map = dict(zip(col_candidates, col_candidate_names))
                col_candidates = self._rerank(
                    f"{tbl} {query}", col_candidates,
                    top_k=MAX_SCHEMA_COLUMNS_PER_TABLE,
                )
                final_cols = []
                for text in col_candidates:
                    col_name = col_candidate_map.get(text, "")
                    if col_name and col_name not in already:
                        final_cols.append(col_name)
                        already.add(col_name)
            else:
                final_cols = [n for n in col_candidate_names[:MAX_SCHEMA_COLUMNS_PER_TABLE]
                              if n not in already]
                for n in final_cols:
                    already.add(n)

            selected_columns[tbl].extend(final_cols)

        # ── Stage 3: Join Paths ──
        join_paths = self._find_join_paths(selected_tables)

        # ── Stage 4: 组装文本 ──
        parts = []
        parts.append(f"## 相关表 ({len(selected_tables)}/{len(self.tables_meta)})\n")

        for tbl in selected_tables:
            meta = self.tables_meta.get(tbl)
            if meta is None:
                continue
            desc = meta.get("description", "")
            module = meta.get("module", "")
            enriched = meta.get("enriched_desc", "")
            pk_str = ", ".join(meta.get("primary_key", []))

            joinable = "✓JOIN" if tbl in ALLOWED_JOIN_TABLES else "✗单表"
            header = f"### [{joinable}] {tbl}"
            if desc:
                header += f" — {desc}"
            if module:
                header += f" ({module})"
            parts.append(header)
            parts.append(f"PK: {pk_str}" if pk_str else "PK: (none)")

            # 优先展示 LLM 富化表描述
            if enriched:
                # 提取富化描述中的关键行（去掉 [TBL] 前缀行）
                lines = enriched.strip().split("\n")
                for line in lines:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("[TBL]"):
                        parts.append(stripped)

            parts.append("关键列:")
            for col_name in selected_columns.get(tbl, []):
                col_info = None
                for c in meta["columns"]:
                    if c["name"] == col_name:
                        col_info = c
                        break
                if col_info is None:
                    continue

                flags = []
                if col_name in meta.get("primary_key", []):
                    flags.append("PK")
                if col_name in fk_columns.get(tbl, set()):
                    for fc, tt, tc in self.fk_graph.get(tbl, []):
                        if fc == col_name:
                            flags.append(f"FK → {tt}.{tc}")
                            break

                flag_str = f" ({', '.join(flags)})" if flags else ""
                # 优先使用 LLM 富化列描述
                enriched_col = col_info.get("enriched_desc", "")
                desc_str = col_info.get("description_zh") or col_info.get("description_en") or ""
                if enriched_col:
                    desc_str = enriched_col
                parts.append(f"  - {col_name} {col_info['type']}{flag_str} — {desc_str}")
            parts.append("")

        # 表间关联（仅展示白名单内 JOIN）
        if join_paths:
            whitelist_joins = [
                jp for jp in join_paths
                if jp["from_table"] in ALLOWED_JOIN_TABLES
                and jp["to_table"] in ALLOWED_JOIN_TABLES
            ]
            if whitelist_joins:
                parts.append("## 表间关联")
                seen_joins = set()
                for jp in whitelist_joins:
                    key = (jp["from_table"], jp["to_table"])
                    if key not in seen_joins:
                        seen_joins.add(key)
                        parts.append(f"- {jp['from_table']}.{jp['from_col']} → "
                                     f"{jp['to_table']}.{jp['to_col']}")
            parts.append("")

        # JOIN 白名单约束标注
        single_only_tables = [t for t in selected_tables
                              if t not in ALLOWED_JOIN_TABLES]
        joinable_tables = [t for t in selected_tables
                           if t in ALLOWED_JOIN_TABLES]

        parts.append("## SQL 注意事项")
        parts.append("- 数据库类型：Microsoft Access (.mdb)，使用 Jet SQL 语法")
        parts.append("- 字符串连接使用 & 而非 CONCAT()")
        parts.append("- 限制行数使用 SELECT TOP N 而非 LIMIT")
        parts.append("- 不支持 WITH ... AS (CTE)，请使用子查询代替")
        parts.append("- 时间类型为 DATETIME，用 #YYYY-MM-DD HH:MM:SS# 格式表示")
        parts.append("- 字符串比较优先用 = 精确匹配，模糊搜索用 LIKE '%keyword%'")
        parts.append("- 聚合查询使用 GROUP BY 和聚合函数")
        parts.append("- 默认 SELECT TOP 50")

        if single_only_tables:
            parts.append(
                f"\n## ⚠️ JOIN 约束\n"
                f"以下表只允许单表查询，禁止 JOIN：{', '.join(single_only_tables)}\n"
                f"以下表允许 JOIN：{', '.join(joinable_tables) if joinable_tables else '无'}"
            )

        result = "\n".join(parts)
        # 字符数上限保护
        if len(result) > MAX_SCHEMA_CHARS_CAP:
            result = result[:MAX_SCHEMA_CHARS_CAP] + "\n（Schema 上下文已截断）"
        logger.info(f"[SchemaIndexer] schema 组装完成：{len(selected_tables)} 表, "
                    f"{sum(len(v) for v in selected_columns.values())} 列, "
                    f"{len(result)} 字符")
        return result
