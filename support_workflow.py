"""
AI-assisted customer support workflow (LangGraph skeleton).

Flow: classify -> retrieve -> draft -> assess -> HUMAN REVIEW -> send -> log
The graph cannot reach send_response without a human decision.

pip install langgraph langchain-anthropic pydantic
export ANTHROPIC_API_KEY=...
"""
import json
import time
from typing import Literal, TypedDict

import os

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

# No ANTHROPIC_API_KEY set -> run in MOCK mode (rule-based, free, offline).
# Set the key -> the same graph uses a real LLM.
USE_MOCK = not os.getenv("ANTHROPIC_API_KEY")
if not USE_MOCK:
    from langchain_anthropic import ChatAnthropic
    llm = ChatAnthropic(model="claude-sonnet-5", temperature=0)
print(f"Mode: {'MOCK (no API key)' if USE_MOCK else 'LIVE (Claude)'}")
CONFIDENCE_THRESHOLD = 0.7

INTENT_KEYWORDS = {
    "complaint": ["angry", "terrible", "worst", "complain", "unacceptable"],
    "refund": ["refund", "return", "money back"],
    "order_status": ["order", "tracking", "shipping", "delivery"],
    "product_info": ["price", "feature", "size", "color"],
}


def _mock_intent(question: str):
    q = question.lower()
    for intent, words in INTENT_KEYWORDS.items():
        if any(w in q for w in words):
            return intent, 0.9
    return "other", 0.4


def _mock_draft(state) -> dict:
    if state["docs"]:
        top = state["docs"][0]
        return {"draft": f"Thanks for reaching out. {top['text']}",
                "missing_info": [], "confidence": 0.85}
    return {"draft": "Thanks for reaching out. A team member will follow up shortly.",
            "missing_info": ["relevant policy or document"], "confidence": 0.3}

# Replace with your real source (the AI Workspace Integration layer:
# Drive/Notion/Confluence/vector store). Keep the return shape the same.
KNOWLEDGE_BASE = [
    {"id": "refund-policy", "text": "Refunds are available within 30 days of purchase with proof of order."},
    {"id": "shipping", "text": "Standard shipping takes 3-5 business days. Tracking is emailed on dispatch."},
]


class Intent(BaseModel):
    intent: Literal["order_status", "refund", "product_info", "complaint", "other"]
    confidence: float


class Draft(BaseModel):
    reply: str
    missing_info: list[str]  # facts needed but not found in the docs
    confidence: float


class State(TypedDict, total=False):
    question: str
    intent: str
    intent_confidence: float
    docs: list[dict]
    draft: str
    missing_info: list[str]
    confidence: float
    escalated: bool
    escalation_reason: str
    review: dict
    final_reply: str
    status: str


def classify_intent(state: State) -> dict:
    if USE_MOCK:
        intent, conf = _mock_intent(state["question"])
        return {"intent": intent, "intent_confidence": conf}
    out = llm.with_structured_output(Intent).invoke(
        f"Classify this customer question.\n\n{state['question']}"
    )
    return {"intent": out.intent, "intent_confidence": out.confidence}


def retrieve_docs(state: State) -> dict:
    # Placeholder keyword match. Swap for embeddings / workspace search.
    words = set(state["question"].lower().split())
    scored = [(len(words & set(d["text"].lower().split())), d) for d in KNOWLEDGE_BASE]
    return {"docs": [d for score, d in sorted(scored, key=lambda x: -x[0]) if score > 0][:3]}


def draft_response(state: State) -> dict:
    context = "\n".join(f"[{d['id']}] {d['text']}" for d in state["docs"]) or "NO DOCUMENTS FOUND"
    if USE_MOCK:
        return _mock_draft(state)
    out = llm.with_structured_output(Draft).invoke(
        "You draft replies for a support team. Use ONLY the context below. "
        "Never promise refunds, dates, or exceptions. List anything you'd need "
        "but can't find in missing_info. Be honest in confidence (0-1).\n\n"
        f"Context:\n{context}\n\nQuestion: {state['question']}"
    )
    return {"draft": out.reply, "missing_info": out.missing_info, "confidence": out.confidence}


def assess_confidence(state: State) -> dict:
    reasons = []
    if not state["docs"]:
        reasons.append("no relevant documents found")
    if state["missing_info"]:
        reasons.append(f"missing info: {', '.join(state['missing_info'])}")
    if min(state["confidence"], state["intent_confidence"]) < CONFIDENCE_THRESHOLD:
        reasons.append("low model confidence")
    if state["intent"] in ("complaint", "other"):
        reasons.append(f"intent '{state['intent']}' always needs a human")
    return {"escalated": bool(reasons), "escalation_reason": "; ".join(reasons)}


def human_review(state: State) -> dict:
    # Pauses the graph. Resume with Command(resume={"action": ..., "edited_reply": ...})
    decision = interrupt({
        "question": state["question"],
        "intent": state["intent"],
        "draft": state["draft"],
        "sources": [d["id"] for d in state["docs"]],
        "escalated": state["escalated"],
        "escalation_reason": state["escalation_reason"],
    })
    return {"review": decision}


def route_after_review(state: State) -> Literal["send_response", "log_interaction"]:
    return "send_response" if state["review"]["action"] in ("approve", "edit") else "log_interaction"


def send_response(state: State) -> dict:
    review = state["review"]
    reply = review.get("edited_reply") if review["action"] == "edit" else state["draft"]
    # TODO: call your email / helpdesk API here.
    print(f"SENT: {reply}")
    return {"final_reply": reply, "status": "sent"}


def log_interaction(state: State) -> dict:
    status = state.get("status", "rejected")
    record = {
        "ts": time.time(),
        "question": state["question"],
        "intent": state["intent"],
        "sources": [d["id"] for d in state["docs"]],
        "ai_draft": state["draft"],
        "escalated": state["escalated"],
        "human_action": state["review"]["action"],
        "final_reply": state.get("final_reply"),
        "status": status,
    }
    with open("interactions.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    return {"status": status}


builder = StateGraph(State)
for name, fn in [
    ("classify_intent", classify_intent), ("retrieve_docs", retrieve_docs),
    ("draft_response", draft_response), ("assess_confidence", assess_confidence),
    ("human_review", human_review), ("send_response", send_response),
    ("log_interaction", log_interaction),
]:
    builder.add_node(name, fn)

builder.add_edge(START, "classify_intent")
builder.add_edge("classify_intent", "retrieve_docs")
builder.add_edge("retrieve_docs", "draft_response")
builder.add_edge("draft_response", "assess_confidence")
builder.add_edge("assess_confidence", "human_review")
builder.add_conditional_edges("human_review", route_after_review)
builder.add_edge("send_response", "log_interaction")
builder.add_edge("log_interaction", END)

# Use a persistent checkpointer (SQLite/Postgres) in production so a paused
# review survives restarts.
graph = builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "ticket-001"}}
    result = graph.invoke({"question": "Can I get a refund? I bought this 10 days ago."}, cfg)

    payload = result["__interrupt__"][0].value
    print("REVIEWER SEES:", json.dumps(payload, indent=2))

    # Simulate the human approving. Use action "edit" + "edited_reply", or "reject".
    graph.invoke(Command(resume={"action": "approve"}), cfg)
