"""Playbook comparison agent.

For each extracted clause, retrieve the most relevant playbook positions and
ask the LLM whether the clause aligns with the playbook, where it deviates,
and what redline would bring it into alignment.
"""

from __future__ import annotations
from src.llm import call_llm
from src.rag import retrieve_playbook_position
from src.state import ReviewState


SYSTEM_PROMPT = """You compare a contract clause against approved playbook
positions and identify deviations.

Return strict JSON:
{
  "alignment": "aligned" | "partial" | "deviation" | "no_position",
  "deviations": [
    {"issue": "<short description>", "severity": "low" | "medium" | "high"}
  ],
  "suggested_redline": "<redlined alternative text, or empty if aligned>",
  "rationale": "<one or two sentences>"
}

Rules:
- "aligned" only if the clause materially matches the playbook position.
- "no_position" if the playbook does not address this clause type.
- Cite playbook language by paraphrase only — do not quote it back verbatim.
- No prose outside the JSON object."""


def compare(state: ReviewState) -> ReviewState:
    run_id = state["run_id"]
    updated_clauses = []

    for clause in state.get("clauses", []):
        # Retrieve top playbook matches for this clause
        matches = retrieve_playbook_position(
            run_id=run_id,
            clause_text=clause["text"],
            clause_type=clause["clause_type"],
            top_k=2,
        )

        playbook_context = "\n\n---\n\n".join(
            m["document"] for m in matches
        ) or "(no playbook entries retrieved)"

        user = (
            f"Clause type: {clause['clause_type']}\n"
            f"Clause text:\n{clause['text']}\n\n"
            f"Relevant playbook positions:\n{playbook_context}"
        )

        result = call_llm(
            run_id=run_id,
            stage="playbook",
            system=SYSTEM_PROMPT,
            user=user,
            expect_json=True,
            max_tokens=1500,
        )

        parsed = result.get("parsed") or {}
        clause["playbook_matches"] = matches
        clause["comparison"] = {
            "alignment": parsed.get("alignment", "no_position"),
            "deviations": parsed.get("deviations", []),
            "suggested_redline": parsed.get("suggested_redline", ""),
            "rationale": parsed.get("rationale", ""),
        }
        updated_clauses.append(clause)

    return {**state, "clauses": updated_clauses}
