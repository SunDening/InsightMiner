"""Utility functions migrated from kb_rag/utils.py."""

import json
import re


def extract_json(text: str) -> str | None:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.+?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        return m.group(0)
    return None


def sanitize_text(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\|", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def repair_json(json_str: str) -> str | None:
    json_str = json_str.strip()
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)
    m = re.search(r"\{.*\}", json_str, re.DOTALL)
    if not m:
        return None
    json_str = m.group(0)
    json_str = re.sub(
        r'(?<=[{,])\s*([a-zA-Z_]\w*)\s*:',
        r'"\1":',
        json_str,
    )
    return json_str


def parse_json_response(raw: str, question: str = "") -> dict:
    fallback = {
        "answer": sanitize_text(raw),
        "rewritten_query": "",
        "data_sources": [],
    }

    extracted = extract_json(raw)
    if not extracted:
        return fallback

    try:
        data = json.loads(extracted)
    except json.JSONDecodeError:
        repaired = repair_json(extracted)
        if not repaired:
            return fallback
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return fallback

    data.setdefault("answer", "")
    data.setdefault("rewritten_query", "")
    data.setdefault("data_sources", [])

    data["answer"] = sanitize_text(data["answer"])
    data["rewritten_query"] = sanitize_text(data["rewritten_query"])

    for src in data.get("data_sources", []):
        src.setdefault("source_type", "")
        src.setdefault("evidences", [])
        if "evidence" in src and "evidences" not in src:
            src["evidences"] = src.pop("evidence")
        for ev in src.get("evidences", []):
            for key in ("evidence", "content", "text"):
                if key in ev:
                    ev["evidence"] = sanitize_text(ev[key])
                    break
            ev.setdefault("evidence", "")
            ev.setdefault("score", 0.0)

    for key in ("method", "description", "analysis"):
        data.pop(key, None)

    return data


def format_history(messages: list, max_pairs: int = 6) -> str:
    pairs = []
    i = 0
    while i < len(messages) - 1:
        if messages[i].type == "human" and messages[i + 1].type == "ai":
            q = messages[i].content[:300]
            a = messages[i + 1].content[:300]
            pairs.append(f"用户: {q}\n助手: {a}")
            i += 2
        else:
            i += 1
    recent = pairs[-max_pairs:]
    return "\n\n".join(recent)
