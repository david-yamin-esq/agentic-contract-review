"""Streamlit UI for the contract review pipeline.

The app has three states it cycles through:

1. UPLOAD: user picks or pastes a contract.
2. REVIEW: pipeline ran, paused at HITL gate, user approves/rejects/modifies.
3. DONE: final report and audit trail visible.

The graph state is persisted in a SqliteSaver, so a paused review survives
a browser refresh — the user just re-enters the same run_id.
"""

from __future__ import annotations
import json
import uuid
from pathlib import Path

import streamlit as st

from config import CONTRACTS_DIR
from src.orchestrator import build_graph
from src.audit import start_run, complete_run, get_events_for_run, get_run_summary
from src.rag import index_playbook


st.set_page_config(page_title="Contract Review Pipeline", layout="wide")

# Session state init
for key, default in [
    ("graph", None),
    ("run_id", None),
    ("thread_id", None),
    ("state", None),
    ("phase", "upload"),  # upload | review | done
]:
    if key not in st.session_state:
        st.session_state[key] = default


def get_graph():
    if st.session_state.graph is None:
        st.session_state.graph = build_graph()
    return st.session_state.graph


# ---- Sidebar: settings + playbook management ----
with st.sidebar:
    st.header("Pipeline controls")

    if st.button("(Re)build playbook index", help="Re-embed data/playbook/playbook.md"):
        with st.spinner("Indexing playbook..."):
            n = index_playbook()
        st.success(f"Indexed {n} playbook sections")

    st.divider()
    st.caption("Run ID")
    st.code(st.session_state.run_id or "(none yet)")

    if st.button("Reset session"):
        for k in ("graph", "run_id", "thread_id", "state", "phase"):
            st.session_state[k] = None if k != "phase" else "upload"
        st.rerun()


# ---- Phase 1: upload ----
if st.session_state.phase == "upload":
    st.title("Agentic Contract Review")
    st.write(
        "Upload a contract or pick a sample. The pipeline will classify it, "
        "extract key clauses, compare each against the playbook, score risk, "
        "and run a verifier pass before pausing for human review."
    )

    col1, col2 = st.columns(2)
    contract_text = ""
    contract_name = ""

    with col1:
        st.subheader("Upload")
        uploaded = st.file_uploader("Plain text or markdown", type=["txt", "md"])
        if uploaded is not None:
            contract_text = uploaded.read().decode("utf-8")
            contract_name = uploaded.name

    with col2:
        st.subheader("Or pick a sample")
        samples = sorted(CONTRACTS_DIR.glob("*.txt")) + sorted(CONTRACTS_DIR.glob("*.md"))
        sample_names = ["(none)"] + [s.name for s in samples]
        choice = st.selectbox("Sample contract", sample_names)
        if choice != "(none)":
            sample_path = CONTRACTS_DIR / choice
            contract_text = sample_path.read_text(encoding="utf-8")
            contract_name = choice

    if contract_text:
        with st.expander("Preview contract text", expanded=False):
            st.text(contract_text[:3000] + ("..." if len(contract_text) > 3000 else ""))

    if st.button("Run review pipeline", type="primary", disabled=not contract_text):
        run_id = start_run(contract_name)
        thread_id = str(uuid.uuid4())
        st.session_state.run_id = run_id
        st.session_state.thread_id = thread_id

        graph = get_graph()
        config = {"configurable": {"thread_id": thread_id}}

        initial: dict = {
            "run_id": run_id,
            "contract_name": contract_name,
            "contract_text": contract_text,
        }

        with st.spinner("Running pipeline (classify → extract → compare → score → verify)..."):
            # Stream until interrupt or completion
            for event in graph.stream(initial, config=config, stream_mode="values"):
                st.session_state.state = event

        # After streaming, check whether we're at the HITL interrupt
        snapshot = graph.get_state(config)
        if snapshot.next and "hitl" in snapshot.next:
            st.session_state.phase = "review"
        else:
            # No HITL needed (low risk, no findings) — pipeline finished
            st.session_state.phase = "done"
        st.rerun()


# ---- Phase 2: HITL review ----
elif st.session_state.phase == "review":
    state = st.session_state.state or {}
    st.title("Human review required")

    risk = state.get("overall_risk_score", 0)
    color = "red" if risk >= 75 else ("orange" if risk >= 50 else "yellow")
    st.markdown(
        f"**Overall risk:** :{color}[{risk}/100] — "
        f"{state.get('overall_risk_rationale', '')}"
    )

    findings = state.get("verifier_findings", [])
    if findings:
        st.warning(f"Verifier raised {len(findings)} finding(s)")
        for f in findings:
            st.markdown(
                f"- **[{f['severity']}]** {f['issue']} _(clause {f['clause_id']})_"
            )

    st.subheader("Clause-by-clause findings")
    modified_clauses: dict[str, dict] = {}
    for clause in state.get("clauses", []):
        with st.expander(
            f"{clause['clause_type']} — risk {clause['risk_score']}/100",
            expanded=clause["risk_score"] >= 50,
        ):
            st.markdown(f"_Location: {clause.get('location_hint', 'n/a')}_")
            st.markdown("**Clause text (as extracted):**")
            st.text(clause["text"])

            comp = clause.get("comparison", {})
            st.markdown(f"**Alignment:** `{comp.get('alignment')}`")
            if comp.get("deviations"):
                st.markdown("**Deviations:**")
                for d in comp["deviations"]:
                    st.markdown(f"- [{d.get('severity')}] {d.get('issue')}")
            st.markdown(f"**AI rationale:** {clause.get('risk_rationale', '')}")
            if comp.get("suggested_redline"):
                st.markdown("**Suggested redline:**")
                st.code(comp["suggested_redline"])

            # Reviewer override controls
            override_score = st.number_input(
                "Override risk score (leave as-is to accept AI)",
                min_value=0,
                max_value=100,
                value=int(clause["risk_score"]),
                key=f"score_{clause['clause_id']}",
            )
            if override_score != clause["risk_score"]:
                modified_clauses[clause["clause_id"]] = {
                    "risk_score": override_score
                }

    st.divider()
    st.subheader("Decision")
    reviewer = st.text_input("Your name", value="reviewer")
    decision = st.radio("Decision", ["approve", "reject", "modify"], horizontal=True)
    notes = st.text_area("Notes (required for reject/modify)", value="")

    if st.button("Submit decision", type="primary"):
        graph = get_graph()
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        # Inject the human decision into graph state, then resume
        graph.update_state(config, {
            "hitl_decision": {
                "decision": decision,
                "reviewer": reviewer,
                "notes": notes,
                "modified_clauses": modified_clauses,
            }
        })

        with st.spinner("Resuming pipeline..."):
            for event in graph.stream(None, config=config, stream_mode="values"):
                st.session_state.state = event

        complete_run(
            run_id=st.session_state.run_id,
            final_status=decision,
            final_risk_score=state.get("overall_risk_score"),
        )
        st.session_state.phase = "done"
        st.rerun()


# ---- Phase 3: done ----
elif st.session_state.phase == "done":
    state = st.session_state.state or {}
    st.title("Review complete")

    if state.get("final_report"):
        st.markdown(state["final_report"])

    st.divider()
    st.subheader("Audit trail")
    events = get_events_for_run(st.session_state.run_id)
    summary = get_run_summary(st.session_state.run_id)
    st.json({"run": summary, "event_count": len(events)})

    with st.expander("All events", expanded=False):
        for e in events:
            payload = json.loads(e["payload_json"])
            st.markdown(
                f"**{e['stage']}** / {e['event_type']} — "
                f"{e['timestamp']} "
                f"({e.get('input_tokens') or 0} in / "
                f"{e.get('output_tokens') or 0} out tokens)"
            )
            st.json(payload, expanded=False)

    audit_export = json.dumps(
        {"run": summary, "events": events}, indent=2, default=str
    )
    st.download_button(
        "Download audit trail (JSON)",
        audit_export,
        file_name=f"audit_{st.session_state.run_id}.json",
        mime="application/json",
    )

    if state.get("final_report"):
        st.download_button(
            "Download report (Markdown)",
            state["final_report"],
            file_name=f"report_{st.session_state.run_id}.md",
            mime="text/markdown",
        )
