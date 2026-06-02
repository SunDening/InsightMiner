# InsightMiner — 未来扩展方向

> 当前四个 Phase 已全部实施完成。以下是根据架构讨论梳理的后续可扩展方向。

---

## 1. 检索通道增强

### 1.1 PostgreSQL FTS + zhparser 替换 BM25

**现状**：`BM25Channel` 使用 `rank-bm25`（Python 内存库），不支持增量更新（dirty flag 全量重建），中文分词仅 30 行正则。

**方案**：PG 已跑着，加一行 FTS 索引 + zhparser 中文分词器，写一个 `PgFtsChannel` 替换 `BM25Channel`。

```yaml
# docker-compose.yml 改动
# 当前:
image: postgres:16-alpine
# 改为（内置 zhparser）:
image: registry.cn-hangzhou.aliyuncs.com/hzfy/postgresql-zhparser:16
```

```sql
-- 加 FTS 索引（一次）
ALTER TABLE chunk ADD COLUMN content_tsv tsvector
  GENERATED ALWAYS AS (to_tsvector('zhparser', content)) STORED;
CREATE INDEX ON chunk USING GIN (content_tsv);

-- 查询
SELECT chunk_id FROM chunk
WHERE content_tsv @@ to_tsquery('zhparser', 'leveldb & 数据')
ORDER BY ts_rank(content_tsv, query) DESC LIMIT 20;
```

**收益**：增量更新（行级）、中文词级切分（非单字）、短语/排除/高亮查询、不占 Python 内存。

**时机**：当 pickle 重建 > 10s 或 BM25 中文召回明显不足时。

### 1.2 pg_search (ParadeDB)

**pg_search** 是 ParadeDB 的 PG 扩展，基于 Tantivy（Rust 搜索引擎库），提供独立于 PG FTS 的 BM25 搜索引擎。

| 维度 | PG FTS + zhparser | pg_search |
|------|-------------------|-----------|
| 算法 | `ts_rank()`（类 TF-IDF） | BM25（词频+长度归一化） |
| 中文 | zhparser 分词 | 内置 CJK 分析器 |
| 增量 | 行级自动 | 行级自动 |
| 部署 | 社区 PG 镜像 | ParadeDB 定制镜像 |
| 运维 | 成熟稳定 | 较新（2023+） |

**建议**：先用 zhparser + PG FTS，如果中文排名质量不够再用 pg_search 替换。

---

## 2. 稀疏检索方案对比总结

```
                     ┌─────────────┐
                     │  rank-bm25   │ ← 当前，零依赖，<1万文档够用
                     │  (内存pickle) │    瓶颈：增量重建、中文分词
                     └──────┬──────┘
                            ↓
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
   ┌──────────┐    ┌──────────────┐    ┌──────────────┐
   │ PG FTS + │    │  pg_search   │    │ Elasticsearch│
   │ zhparser │    │  (ParadeDB)  │    │              │
   ├──────────┤    ├──────────────┤    ├──────────────┤
   │ PG 内置   │    │ PG 扩展       │    │ 独立服务      │
   │ tsvector  │    │ Tantivy BM25 │    │ 分布式集群    │
   │ 中文 OK   │    │ 中文 OK      │    │ 中文 OK      │
   │ 零额外部署 │    │ 定制 PG 镜像  │    │ 重（~300MB） │
   └──────────┘    └──────────────┘    └──────────────┘
         ↑                 ↑                  ↑
    性价比最高         性能最好         已有团队运维时
   PG 已跑着就用       RAG 场景最佳     才值得引入
```

---

## 3. 现有 BM25 够用 vs 需要升级的判断标准

| 信号 | 说明 |
|------|------|
| **文档 < 5000** | BM25Okapi 完全够用，零依赖，不用换 |
| **pickle 重建 > 10s** | `_build_bm25()` 耗时太长，考虑 PG FTS 增量方案 |
| **中文长尾词召回差** | 自定义 tokenizer 中文字颗粒度不够，上 zhparser |
| **需要短语/排除查询** | BM25 不支持，上 PG FTS / pg_search |
| **Python 内存紧张** | BM25 的 pickle 加载所有 token 到内存，PG 方案缓解 |
| **面试需要展开** | "当前 BM25 够用，预留了 SearchChannel 接口，可替换为 pg_search" |
