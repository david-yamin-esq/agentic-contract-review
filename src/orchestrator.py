"""LangGraph orchestrator.

Wires the agents into a state graph and configures the human-in-the-loop
gate using LangGraph's `interrupt_before` mechanism. The graph checkpoints
its state to SQLite, so a paused review can be resumed in a separate process
or after the user closes the browser.

Flow:

    classify -> extract -> compare -> score -> verify -> [HITL gate] -> finalize

The HITL gate is implemented as a routing edge, not a node. After the
verifier writes `requires_human_review` into state, the graph either pauses
(interrupt_before='hitl') or skips straight to finalize.
"""

from __future__ import annotations
import sqlite3
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from config import CHECKPOINT_DB_PATH
from src.state import ReviewState
from src.agents.classifier import classify
from src.agents.extractor import extract
from src.agents.playbook_agent import compare
from src.agents.risk_scorer import score
from src.agents.verifier import verify
from src.audit import log_event


def _route_after_verify(state: ReviewState) -> Literal["hitl", "finalize"]:
    return "hitl" if state.get("requires_human_review") else "finalize"


def _hitl_node(state: ReviewState) -> ReviewState:
    """The HITL node itself is a no-op — its purpose is to be the
    interrupt point. When the graph is resumed, the orchestrator will have
    written hitl_decision into state via update_state(), and this node
    just passes that through."""
    decision = state.get("hitl_decision")
    if decision:
        log_event(
            run_id=state["run_id"],
            stage="hitl",
            event_type="human_decision",
            payload=dict(decision),
        )
        # Apply any modifications the reviewer made to clauses
        modified = decision.get("modified_clauses", {}) or {}
        if modified:
            updated = []
            for c in state.get("clauses", []):
                override = modified.get(c["clause_id"])
                if override:
                    c = {**c, **override}
                updated.append(c)
            state = {**state, "clauses": updated}
    return state


def _finalize(state: ReviewState) -> ReviewState:
    """Produce a final report string. Kept simple and deterministic."""
    lines = []
    lines.append(f"# Contract Review Report")
    lines.append("")
    lines.append(f"**Contract:** {state.get('contract_name')}")
    lines.append(f"**Type:** {state.get('contract_type')} "
                 f"(confidence {state.get('contract_type_confidence', 0):.2f})")
    lines.append(f"**Overall risk score:** {state.get('overall_risk_score')}/100")
    lines.append(f"**Overall rationale:** {state.get('overall_risk_rationale')}")
    decision = state.get("hitl_decision")
    if decision:
        lines.append(f"**Human review:** {decision.get('decision')} "
                     f"by {decision.get('reviewer')}")
        if decision.get("notes"):
            lines.append(f"  Notes: {decision['notes']}")
    lines.append("")
    lines.append("## Clause findings")
    lines.append("")
    for c in state.get("clauses", []):
        lines.append(f"### {c['clause_type']} (risk {c['risk_score']}/100)")
        lines.append(f"_Location: {c.get('location_hint', 'n/a')}_")
        lines.append("")
        comp = c.get("comparison", {})
        lines.append(f"- Alignment: **{comp.get('alignment')}**")
        if comp.get("deviations"):
            for d in comp["deviations"]:
                lines.append(f"  - [{d.get('severity')}] {d.get('issue')}")
        lines.append(f"- Rationale: {c.get('risk_rationale', '')}")
        if comp.get("suggested_redline"):
            lines.append("")
            lines.append("**Suggested redline:**")
            lines.append("")
            lines.append(f"> {comp['suggested_redline']}")
        lines.append("")

    findings = state.get("verifier_findings", [])
    if findings:
        lines.append("## Verifier findings")
        lines.append("")
        for f in findings:
            lines.append(f"- [{f['severity']}] {f['issue']} (clause {f['clause_id']})")

    return {**state, "final_report": "\n".join(lines)}


def build_graph():
    """Build and return a compiled LangGraph with HITL interrupt."""
    g = StateGraph(ReviewState)

    g.add_node("classify", classify)
    g.add_node("extract", extract)
    g.add_node("compare", compare)
    g.add_node("score", score)
    g.add_node("verify", verify)
    g.add_node("hitl", _hitl_node)
    g.add_node("finalize", _finalize)

    g.set_entry_point("classify")
    g.add_edge("classify", "extract")
    g.add_edge("extract", "compare")
    g.add_edge("compare", "score")
    g.add_edge("score", "verify")
    g.add_conditional_edges("verify", _route_after_verify, {
        "hitl": "hitl",
        "finalize": "finalize",
    })
    g.add_edge("hitl", "finalize")
    g.add_edge("finalize", END)

    # SqliteSaver persists graph state. The `interrupt_before=["hitl"]`
    # tells LangGraph to pause execution before the hitl node runs,
    # so the UI can collect a human decision.
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=["hitl"],
    )
