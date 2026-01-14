from __future__ import annotations

import logging
from typing import TypedDict, Any

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from providers import init_llm
from vector_store import pg_engine

_log = logging.getLogger(__name__)


# =========================
# LangGraph: retrieval + answer
# =========================
class RAGState(TypedDict):
    question: str
    top_k: int
    evidence: list[dict[str, Any]]
    answer: str


def node_retrieve( state: RAGState ) -> dict:
    _log.info("graph: node_retrieve start top_k=%d question_len=%d", state['top_k'], len(state['question']))

    pairs = pg_engine.similarity_search_with_score(
        state["question"],
        k = state["top_k"],
    )
    _log.info("graph: retrieved pairs=%d", len(pairs))

    evidence = []
    for doc, score in pairs:
        evidence.append(
            {
                "source": doc.metadata.get("source_file", "unknown"),
                "locator": doc.metadata.get("page_human", "chunk"),
                "snippet": (doc.page_content or "")[:900],
                "score": float(score),
            },
        )
    _log.info("graph: evidence built=%d", len(evidence))
    return { "evidence": evidence }


def node_answer( state: RAGState ) -> dict:
    _log.info("graph: node_answer start evidence_in=%d", len(state.get('evidence', [])))
    llm = init_llm()

    evidence = sorted(state.get("evidence", []), key = lambda x: x["score"])[:20]
    if not evidence:
        _log.warning("graph: no evidence -> returning fallback")
        return { "answer": "В источниках ничего релевантного не найдено по данному вопросу." }

    lines = []
    for i, e in enumerate(evidence, start = 1):
        loc = e["locator"]
        lines.append(f"[{i}] src={e['source']} loc={loc}\n{e['snippet']}\n")
    evidence_block = "\n".join(lines)

    system = (
        "Ты RAG-ассистент. Отвечай СТРОГО по Evidence.\n"
        "Правила:\n"
        "- Не используй внешние знания.\n"
        "- Каждое существенное утверждение снабжай ссылкой [n].\n"
        "- Если данных недостаточно — так и скажи.\n"
        "- В конце добавь раздел 'Источники:' со списком [n] -> source.\n"
        "Язык: русский.\n"
    )

    user = (
        f"Вопрос: {state['question']}\n\n"
        f"Evidence:\n{evidence_block}\n\n"
        "Сформируй ответ:\n"
        "1) краткий вывод\n"
        "2) детали/обоснование\n"
    )

    _log.info("graph: llm_invoke start evidence_used=%d", len(evidence))
    resp = llm.invoke(
        [
            { "role": "system", "content": system },
            { "role": "user", "content": user },
        ],
    )
    text = resp.content if hasattr(resp, "content") else str(resp)

    if "Источники:" not in text:
        foot = ["\nИсточники:"]
        for i, e in enumerate(evidence, start = 1):
            foot.append(f"[{i}] {e['source']} ({e['locator']})")
        text = text.rstrip() + "\n" + "\n".join(foot)

    _log.info("graph: llm_invoke done answer_chars=%d", len(text))
    return { "answer": text }


def build_graph( ):
    g = StateGraph(RAGState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("answer", node_answer)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", END)

    _log.info("graph: build_graph compile done")
    return g.compile()


GRAPH = build_graph()
