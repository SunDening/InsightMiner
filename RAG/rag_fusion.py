from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM


# ============================================================
# 1. Load & index the document (same as before)
# ============================================================
loader = TextLoader("cme_event.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
splits = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 10})

llm = OllamaLLM(model="gemma4:e4b")

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
# 4. RAG Fusion pipeline — multi-query → RRF → final generation
# ============================================================
def rag_fusion_ask(question: str) -> str:
    # Step A: Generate diverse sub-queries
    sub_queries = generate_multi_queries(question)
    print(f"[Generated {len(sub_queries)} sub-queries]")

    # Step B: Retrieve for each sub-query
    all_results = []
    for q in sub_queries:
        docs = retriever.invoke(q)
        all_results.append(docs)

    # Step C: RRF fusion — combine & rerank
    fused = reciprocal_rank_fusion(all_results)
    fused.sort(key=lambda x: x[1], reverse=True)

    # Step D: Top-N after fusion as context
    top_docs = fused[:6]
    context = "\n\n---\n\n".join([doc.page_content for doc, score in top_docs])

    # Step E: Final generation with domain-specific prompt
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
    return llm.invoke(prompt)


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    print(rag_fusion_ask("列出所有撞击概率大于0.8且朝向地球的CME事件。"))
