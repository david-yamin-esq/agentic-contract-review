"""Risk scorer agent.

Per-clause and overall risk scoring. Uses the playbook comparison output as
its primary input. Scores are 0-100 to make the HITL threshold logic simple.
"""

from __future__ import annotations
import json

from src.llm import call_llm
from src.state import ReviewState


PER_CLAUSE_SYSTEM = """You assign a risk score (0-100) to a contract clause.

Inputs you receive: the clause text and a playbook-comparison summary.

Scoring guide:
- 0-20: aligned with playbook, low risk
- 21-50: minor deviations, manageable
- 51-75: material deviation, negotiation recommended
- 76-100: severe deviation or unusual exposure, escalate

Return strict JSON:
{
  "risk_score": <int 0-100>,
  "rationale": "<one or two sentences>"
}
No prose outside the JSON object."""


OVERALL_SYSTEM = """You assign an overall risk score (0-100) to a contract
based on the per-clause risk scores and rationales provided.

The overall score is NOT a simple average. Consider:
- Whether any single clause is high enough to dominate (e.g. an uncapped
  indemnity is enough to escalate even if everything else is fine).
- Whether deviations compound (e.g. weak liability cap + broad indemnity).
- Whether missing clauses (e.g. no governing law) imply additional risk.

Return strict JSON:
{
  "overall_risk_score": <int 0-100>,
  "rationale": "<two or three sentences>"
}
No prose outside the JSON object."""


def score(state: ReviewState) -> ReviewState:
    run_id = state["run_id"]
    updated_clauses = []

    for clause in state.get("clauses", []):
        comparison = clause.get("comparison", {})
        user = (
            f"Clause type: {clause['clause_type']}\n"
            f"Clause text:\n{clause['text']}\n\n"
            f"Playbook comparison:\n{json.dumps(comparison, indent=2)}"
        )
        result = call_llm(
            run_id=run_id,
            stage="risk_scorer",
            system=PER_CLAUSE_SYSTEM,
            user=user,
            expect_json=True,
            max_tokens=400,
        )
        parsed = result.get("parsed") or {}
        clause["risk_score"] = int(parsed.get("risk_score", 0))
        clause["risk_rationale"] = parsed.get("rationale", "")
        updated_clauses.append(clause)

    # Overall score
    summary = [
        {
            "clause_type": c["clause_type"],
            "risk_score": c["risk_score"],
            "rationale": c["risk_rationale"],
            "alignment": c.get("comparison", {}).get("alignment"),
        }
        for c in updated_clauses
    ]
    overall = call_llm(
        run_id=run_id,
        stage="risk_scorer",
        system=OVERALL_SYSTEM,
        user=f"Per-clause scores:\n{json.dumps(summary, indent=2)}",
        expect_json=True,
        max_tokens=500,
    )
    overall_parsed = overall.get("parsed") or {}

    return {
        **state,
        "clauses": updated_clauses,
        "overall_risk_score": int(overall_parsed.get("overall_risk_score", 0)),
        "overall_risk_rationale": overall_parsed.get("rationale", ""),
    }
