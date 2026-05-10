"""Clause extractor agent.

Pulls the highest-leverage clauses out of the contract — the ones a reviewer
would actually want to negotiate. Each extracted clause includes a
location_hint so the verifier can later check the quote back against source.
"""

from __future__ import annotations
import uuid
from src.llm import call_llm
from src.state import ReviewState, Clause


# Clause types the extractor is asked to look for. Tunable per contract type
# in production; kept generic here.
TARGET_CLAUSES = [
    "limitation_of_liability",
    "indemnification",
    "confidentiality",
    "termination",
    "governing_law",
    "intellectual_property",
    "data_protection",
    "warranty",
    "payment_terms",
    "non_compete",
]


SYSTEM_PROMPT = f"""You are a legal clause extractor. Given a contract, find
each instance of the following clause types and return their verbatim text.

Clause types to look for:
{chr(10).join(f"- {c}" for c in TARGET_CLAUSES)}

Return strict JSON:
{{
  "clauses": [
    {{
      "clause_type": "<one of the types above>",
      "text": "<verbatim clause text, copied exactly from the contract>",
      "location_hint": "<section number, heading, or other locator>"
    }}
  ]
}}

Rules:
- Quote text VERBATIM. Do not paraphrase or summarise.
- If a clause type is not present, omit it from the list.
- If multiple instances exist, include each separately.
- If no listed clauses are found, return {{"clauses": []}}.
- No prose outside the JSON object."""


def extract(state: ReviewState) -> ReviewState:
    contract_text = state["contract_text"]
    user = (
        f"Contract type: {state.get('contract_type', 'unknown')}\n\n"
        f"Contract text:\n\n{contract_text}"
    )

    result = call_llm(
        run_id=state["run_id"],
        stage="extractor",
        system=SYSTEM_PROMPT,
        user=user,
        expect_json=True,
        max_tokens=4000,
    )

    parsed = result.get("parsed") or {}
    raw = parsed.get("clauses", [])

    clauses: list[Clause] = []
    for c in raw:
        clauses.append(
            Clause(
                clause_id=str(uuid.uuid4())[:8],
                clause_type=c.get("clause_type", "other"),
                text=c.get("text", "").strip(),
                location_hint=c.get("location_hint", ""),
                playbook_matches=[],
                comparison={},
                risk_score=0,
                risk_rationale="",
            )
        )

    return {**state, "clauses": clauses}
