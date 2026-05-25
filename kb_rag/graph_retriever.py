"""实体共现图检索器 — KB RAG 的第三条检索路径。

索引时：从 chunk 中提取实体 → 构建 entity↔chunk 映射 → 构建共现加权图
检索时：从查询中提取实体 → 图匹配 + BFS 遍历 → 按匹配类型+距离打分 → chunk chaining
"""

import os
import re
import pickle
import logging

import networkx as nx

logger = logging.getLogger("kb_rag.graph")

# ── 匹配类型基准分 ──────────────────────────────────────────
_SCORE_EXACT = 3.0
_SCORE_NEIGHBOR = 2.0
_SCORE_EXPANDED = 1.0

# ── 英文停用词 ─────────────────────────────────────────────
_STOPWORDS = frozenset({
    "the", "this", "that", "what", "how", "why", "which", "when",
    "where", "with", "from", "into", "they", "them", "their",
    "these", "those", "have", "has", "had", "been", "being",
    "were", "was", "are", "is", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "not", "no", "nor", "but", "and", "or", "for", "so", "yet",
    "also", "just", "very", "too", "much", "more", "most",
    "some", "any", "each", "every", "both", "few", "many",
    "about", "above", "after", "again", "all", "am", "an",
    "at", "be", "because", "been", "below",
    "between", "during", "if", "into", "of", "off",
    "on", "once", "only", "other", "our", "out", "over",
    "own", "same", "she", "than",
    "to", "under", "until", "up", "was",
    "were", "while", "you", "your",
})


class GraphRetriever:
    """实体共现图检索器。

    Attributes:
        graph: NetworkX 无向图，节点=实体名，边权重=共现次数
        entity_to_chunks: {entity_lower: set[chunk_index]}
        chunk_texts / chunk_metas: 引用外部列表，供 chunk chaining 使用
    """

    def __init__(self):
        self.graph: nx.Graph | None = None
        self.entity_to_chunks: dict[str, set[int]] = {}
        self.chunk_texts: list[str] = []
        self.chunk_metas: list[dict] = []

    # ── 索引构建 ──────────────────────────────────────────────

    def build(self, chunk_texts: list[str], chunk_metas: list[dict]) -> None:
        """构建实体共现图。"""
        self.chunk_texts = chunk_texts
        self.chunk_metas = chunk_metas
        self.graph = nx.Graph()
        self.entity_to_chunks = {}

        for chunk_idx, text in enumerate(chunk_texts):
            entities = self._extract_entities(text)
            if not entities:
                continue

            # entity → chunk 映射
            for ent in entities:
                self.entity_to_chunks.setdefault(ent, set()).add(chunk_idx)

            # 实体共现边（同一 chunk 内的所有实体两两建边）
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    a, b = entities[i], entities[j]
                    if self.graph.has_edge(a, b):
                        self.graph[a][b]["weight"] += 1
                    else:
                        self.graph.add_edge(a, b, weight=1)

        # 孤立实体也加入图（确保节点存在）
        for ent in self.entity_to_chunks:
            if ent not in self.graph:
                self.graph.add_node(ent)

        logger.info(
            f"[Graph] 图构建完成: {len(self.graph.nodes)} 个实体, "
            f"{len(self.graph.edges)} 条边, "
            f"{len(self.entity_to_chunks)} 个实体→chunk 映射"
        )

    # ── 检索入口 ──────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 40,
                 query_entities: list[str] | None = None) -> list[tuple[int, float]]:
        """基于图的检索入口。

        返回 list[tuple[chunk_index, score]]，按分数降序，最多 top_k 条。
        支持外部传入 query_entities（如来自 LLM 提取）。
        """
        if self.graph is None or not self.entity_to_chunks:
            return []

        # 优先使用外部传入的实体，否则 fallback 到启发式提取
        entities = (
            query_entities
            if query_entities
            else self._extract_entities(query)
        )
        if not entities:
            return []

        matched = self._find_matching_entities(entities)
        if not matched:
            logger.info("[Graph] 查询实体在图谱中无匹配")
            return []

        results = self._traverse_and_score(matched)
        results = self._apply_chunk_chaining(results)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ── 实体提取（启发式）──────────────────────────────────────

    @staticmethod
    def extract_entities(text: str) -> list[str]:
        """公开的实体提取方法，供外部调用。"""
        return GraphRetriever._extract_entities(text)

    @staticmethod
    def _extract_entities(text: str) -> list[str]:
        """启发式实体提取。

        策略：
          - 英文大写/标题大小写短语（专有名词）
          - 英文/数字 token ≥ 3 字符（术语、缩写）
          - 中文连续 2+ 字
          - 去重 + 过滤停用词
        """
        entities = set()

        # 1. 专有名词：连续标题大小写单词
        for m in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
            candidate = m.group().strip()
            words = candidate.split()
            if len(words) <= 5:
                entities.add(candidate.lower())

        # 2. 英文/数字 token（≥3 字符）
        for m in re.finditer(r"[a-zA-Z0-9_\-]{3,}", text):
            entities.add(m.group().lower())

        # 3. 中文连续 2+ 字
        for m in re.finditer(r"[一-鿿]{2,}", text):
            entities.add(m.group())

        # 过滤
        result = []
        for e in entities:
            e = e.strip()
            if len(e) >= 2 and e not in _STOPWORDS:
                result.append(e)
        return result

    # ── 图匹配 ────────────────────────────────────────────────

    def _find_matching_entities(
        self, query_entities: list[str]
    ) -> list[tuple[str, int]]:
        """在图节点中查找匹配，返回 [(entity, match_type), ...]。

        match_type: 0=精确匹配, 1=子串/包含匹配
        """
        matched: list[tuple[str, int]] = []
        node_map = {n.lower(): n for n in self.graph.nodes}

        for qe in query_entities:
            qe_lower = qe.lower()

            # 精确匹配
            if qe_lower in node_map:
                matched.append((node_map[qe_lower], 0))
                continue

            # 子串/包含匹配
            for gn_lower, gn_orig in node_map.items():
                if qe_lower in gn_lower or gn_lower in qe_lower:
                    matched.append((gn_orig, 1))
                    break

        return matched

    # ── BFS 遍历 + 打分 ──────────────────────────────────────

    def _traverse_and_score(
        self, matched_entities: list[tuple[str, int]]
    ) -> list[tuple[int, float]]:
        """BFS 遍历实体图并给 chunk 打分。"""
        chunk_scores: dict[int, float] = {}
        visited: set[str] = set()

        exact_entities = [e for e, t in matched_entities if t == 0]

        # 精确匹配实体 → 直接命中 chunk（最高分）
        for entity in exact_entities:
            visited.add(entity)
            for idx in self.entity_to_chunks.get(entity, set()):
                chunk_scores[idx] = max(chunk_scores.get(idx, 0), _SCORE_EXACT)

        # BFS 展开所有匹配实体
        for entity, _ in matched_entities:
            if entity not in self.graph:
                continue
            try:
                distances = nx.single_source_shortest_path_length(
                    self.graph, entity, cutoff=3
                )
            except nx.NetworkXError:
                continue

            for neighbor, distance in distances.items():
                if neighbor in visited and distance > 0:
                    continue
                visited.add(neighbor)

                if distance == 0:
                    continue

                base = _SCORE_NEIGHBOR if distance == 1 else _SCORE_EXPANDED / distance
                # 共现权重加成
                bonus = 0.0
                if self.graph.has_edge(entity, neighbor):
                    w = self.graph[entity][neighbor].get("weight", 1)
                    bonus = 0.5 * (w / (w + 5.0))

                score = base + bonus
                for idx in self.entity_to_chunks.get(neighbor, set()):
                    chunk_scores[idx] = max(chunk_scores.get(idx, 0), score)

        return list(chunk_scores.items())

    # ── Chunk Chaining ───────────────────────────────────────

    def _apply_chunk_chaining(
        self,
        results: list[tuple[int, float]],
        adjacency_bonus: float = 0.1,
    ) -> list[tuple[int, float]]:
        """对强匹配 chunk 的文档内相邻 chunk 给予奖励。"""
        if not results or not self.chunk_metas:
            return results

        score_map = dict(results)
        boosted = dict(score_map)

        for chunk_idx, score in results:
            if score < _SCORE_NEIGHBOR:
                continue
            fname = self.chunk_metas[chunk_idx].get("filename", "")
            for offset in (-1, 1):
                neighbor = chunk_idx + offset
                if neighbor < 0 or neighbor >= len(self.chunk_metas):
                    continue
                if self.chunk_metas[neighbor].get("filename") != fname:
                    continue
                if neighbor not in score_map:
                    boosted[neighbor] = score * adjacency_bonus
                elif score_map[neighbor] < score:
                    boosted[neighbor] = score_map[neighbor] + 0.05

        return list(boosted.items())

    # ── 持久化 ────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """序列化到 pickle。"""
        data = {
            "graph": self.graph,
            "entity_to_chunks": self.entity_to_chunks,
        }
        try:
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"[Graph] 已持久化到 {path}")
        except Exception as e:
            logger.warning(f"[Graph] 持久化失败: {e}")

    def load(self, path: str) -> bool:
        """从 pickle 加载。返回是否成功。"""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.graph = data["graph"]
            self.entity_to_chunks = data["entity_to_chunks"]
            logger.info(
                f"[Graph] 从 {path} 加载完成: "
                f"{len(self.graph.nodes)} 实体, {len(self.graph.edges)} 边"
            )
            return True
        except Exception as e:
            logger.warning(f"[Graph] 加载失败: {e}")
            self.graph = None
            self.entity_to_chunks = {}
            return False
