import time

import torch
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM

# 重排序模型：CrossEncoder 对 (query, doc) 逐对打分，精度远超向量相似度
from sentence_transformers import CrossEncoder

# 自动检测设备：CUDA 可用则用 GPU，否则回退 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] Using: {DEVICE}")

# Stage 1: Multi-query generation  (LLM 扩写 → 5 个子查询)
# Stage 2: Vector retrieval        (每个子查询独立检索)
# Stage 3: RRF fusion              (多路结果加权融合，高召回)
# Stage 4: CrossEncoder rerank     (逐对精排，高精度)   ← 新增
# Stage 5: Top-6 as context        (精度最高的文档入上下文)
# Stage 6: LLM generation          (CME 专家 JSON 输出)


# ============================================================
# 1. Load & index the document
# ============================================================
loader = TextLoader("cme_event.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
splits = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda"},       # 强制使用 GPU
)
vector_db = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 10})

llm = OllamaLLM(model="gemma4:e4b")

# 加载 HuggingFace 重排序模型（一次加载，全局复用）
# BAAI/bge-reranker-base 是中文社区广泛使用的高质量 CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-base", device="cuda")  # 强制使用 GPU

# ============================================================
# 2. Multi-query generation — expand one question into N diverse sub-queries
# ============================================================
def generate_multi_queries(question: str, n: int = 4) -> list[str]:
    """Use the LLM to generate diverse, complementary sub-queries from the original question."""
    prompt = (
        f"You are a space weather analysis expert specializing in Coronal Mass Ejections (CMEs).\n"
        f"Generate {n} diverse, complementary search queries based on the user's question about CME events.\n"
        f"Each query should approach the topic from a different angle (e.g., speed, source location, impact probability, "
        f"flare association, timing, CME type) to maximize retrieval coverage.\n"
        f"Return ONLY the queries themselves, one per line. Do NOT number or prefix them.\n\n"
        f"User question: {question}\n\nQueries:"
    )
    response = llm.invoke(prompt)
    queries = [q.strip().lstrip("0123456789. -") for q in response.strip().split("\n") if q.strip()]
    # Always include the original question
    return [question] + queries[:n]


# ============================================================
# 3. Reciprocal Rank Fusion — fuse results from multiple retrievals
# ============================================================
def reciprocal_rank_fusion(
    query_results: list[list], k: int = 60
) -> list[tuple[object, float]]:
    """
    Each query_results[i] is a ranked list of documents from one sub-query.
    RRF score(doc) = sum over queries of 1 / (k + rank_of_doc_in_that_query).
    Returns a list of (doc, score) sorted by descending RRF score.
    """
    doc_score: dict[str, tuple[object, float]] = {}
    for ranked_list in query_results:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc.page_content
            score = 1.0 / (k + rank + 1)
            if doc_id in doc_score:
                doc_score[doc_id] = (doc, doc_score[doc_id][1] + score)
            else:
                doc_score[doc_id] = (doc, score)
    return sorted(doc_score.values(), key=lambda x: x[1], reverse=True)


# ============================================================
# 4. CrossEncoder reranking — fine-grained relevance scoring
# ============================================================
def rerank_with_cross_encoder(
    question: str, docs_with_scores: list[tuple[object, float]]
) -> list[tuple[object, float]]:
    """
    Use a CrossEncoder (BAAI/bge-reranker-base) to score each (question, doc) pair.
    The CrossEncoder reads both query and document jointly, capturing semantic
    relevance more precisely than vector similarity or RRF fusion alone.

    Pipeline position: RRF (high recall) → CrossEncoder rerank (high precision).
    """
    # 去重：同一个 chunk 可能被多个子查询命中，RRF 已按 content 去重
    seen = set()
    unique_docs = []
    for doc, _ in docs_with_scores:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique_docs.append(doc)

    if not unique_docs:
        return []

    # 构造 (question, doc_content) 对，批量喂给 CrossEncoder
    pairs = [[question, doc.page_content] for doc in unique_docs]
    scores = reranker.predict(pairs)

    # 按 CrossEncoder 得分降序排列
    doc_score_pairs = list(zip(unique_docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return doc_score_pairs


# ============================================================
# 5. RAG Fusion + Rerank pipeline
#    multi-query → retrieval → RRF → CrossEncoder rerank → generation
# ============================================================
def rag_fusion_rerank_ask(question: str) -> str:
    t_total = time.time()

    # --- Stage 1: Multi-query generation ---
    t0 = time.time()
    sub_queries = generate_multi_queries(question)
    t1 = time.time()
    print(f"[S1] Multi-query generation:      {t1 - t0:.1f}s  ({len(sub_queries)} queries)")

    # --- Stage 2: Retrieve for each sub-query ---
    all_results = []
    for q in sub_queries:
        docs = retriever.invoke(q)
        all_results.append(docs)
    t2 = time.time()
    print(f"[S2] Vector retrieval:            {t2 - t1:.1f}s  ({sum(len(r) for r in all_results)} total hits)")

    # --- Stage 3: RRF fusion — combine multiple ranked lists ---
    fused = reciprocal_rank_fusion(all_results)
    t3 = time.time()
    print(f"[S3] RRF fusion:                  {t3 - t2:.1f}s  ({len(fused)} unique docs)")

    # --- Stage 4: CrossEncoder reranking — fine-grained relevance ---
    reranked = rerank_with_cross_encoder(question, fused)
    t4 = time.time()
    print(f"[S4] CrossEncoder rerank:         {t4 - t3:.1f}s  (top score: {reranked[0][1]:.4f})")

    # --- Stage 5: Top-N after rerank as context ---
    top_docs = reranked[:6]
    context = "\n\n---\n\n".join([doc.page_content for doc, score in top_docs])

    # --- Stage 6: Final generation with domain-specific prompt ---
    prompt = (
        f"You are a space weather analysis expert specializing in Coronal Mass Ejections (CMEs). "
        f"Your role is to analyze CME event data and provide accurate, insightful answers to user queries.\n\n"

        f"## KNOWLEDGE BASE STRUCTURE\n"
        f"The knowledge base contains CME events with the following fields:\n"
        f"- event_id: Unique identifier string (e.g., 'CME-001')\n"
        f"- event_type: Fixed value 'cme'\n"
        f"- start_time: CME start time in ISO 8601 format (e.g., '2026-05-07T12:36:00Z')\n"
        f"- peak_time: Peak brightness time in ISO 8601 format\n"
        f"- end_time: CME end time in ISO 8601 format\n"
        f"- duration_minutes: Total duration in minutes (number)\n"
        f"- speed_km_s: Ejection speed in km/s (number, typically 400-2500)\n"
        f"- half_angle_deg: Angular width half-angle in degrees (number, typically 10-65)\n"
        f"- source_location: Solar source location in heliographic coordinates (e.g., 'N15W25')\n"
        f"- type: CME classification - 'full_halo' (Earth-directed wide), 'partial_halo', or 'normal'\n"
        f"- earth_directed: Boolean - whether CME is directed toward Earth\n"
        f"- impact_probability: Probability of Earth impact from 0 to 1 (number)\n"
        f"- arrival_time_predicted: Predicted Earth arrival time in ISO 8601, or null if not available\n"
        f"- associated_flare_class: Associated solar flare class (B/C/M/X followed by number, e.g., 'M5.6')\n"
        f"- confidence_score: Event confidence score from 0 to 1 (number)\n\n"

        f"## TASK\n"
        f"Based on the CME event data provided in the Context section, answer the user's question thoroughly and insightfully.\n"
        f"When answering:\n"
        f"- Only use information from the provided Context\n"
        f"- If the Context doesn't contain enough information to answer, state that clearly\n"
        f"- For comparative questions, highlight differences between events\n"
        f"- For analytical questions, provide reasoning based on CME characteristics\n"
        f"- Use ONLY plain text — NO Markdown formatting (no **bold**, *italic*, `code`, # headings, bullet lists with - or *, etc.)\n\n"

        f"## OUTPUT FORMAT\n"
        f"Return ONLY a valid JSON object with the following fields:\n"
        f'{{\n'
        f'  "direct_answer": "Direct answer to the question",\n'
        f'  "evidence": ["event_id: CME-001, speed: 800 km/s", "..."],\n'
        f'  "insights": "Additional insights or trends observed (or null if none)",\n'
        f'  "confidence": "High" | "Medium" | "Low"\n'
        f'}}\n\n'

        f"## CONTEXT (Retrieved from Knowledge Base)\n"
        f"{context}\n\n"

        f"## QUESTION\n"
        f"{question}\n\n"

        f"## ANSWER\n"
    )
    result = llm.invoke(prompt)
    t5 = time.time()
    print(f"[S6] LLM generation:              {t5 - t4:.1f}s")
    print(f"[==] Total pipeline:              {t5 - t_total:.1f}s")
    return result


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print(rag_fusion_rerank_ask("比较CME-001、CME-002和CME-004的速度和撞击概率，哪个对地球威胁最大？"))
