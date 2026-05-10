"""Classifier agent.

Identifies the type of contract (NDA, MSA, SaaS, employment, etc.) and a
confidence score. Cheap first step that lets downstream agents apply
type-specific playbook entries.
"""

from __future__ import annotations
from src.llm import call_llm
from src.state import ReviewState


SYSTEM_PROMPT = """You are a legal classifier. Given the text of a contract,
identify its primary type using a short canonical label.

Allowed labels (use exactly one):
- nda
- msa
- saas_subscription
- employment
- consulting_services
- licensing
- distribution
- procurement
- other

Return strict JSON:
{
  "contract_type": "<label>",
  "confidence": <float 0-1>,
  "reasoning": "<one sentence>"
}
No prose outside the JSON object."""


def classify(state: ReviewState) -> ReviewState:
    contract_text = state["contract_text"]
    # Cap input to first 6k chars — type is almost always identifiable from the opening
    user = f"Classify this contract:\n\n{contract_text[:6000]}"

    result = call_llm(
        run_id=state["run_id"],
        stage="classifier",
        system=SYSTEM_PROMPT,
        user=user,
        expect_json=True,
        max_tokens=300,
    )

    parsed = result.get("parsed") or {}
    return {
        **state,
        "contract_type": parsed.get("contract_type", "other"),
        "contract_type_confidence": float(parsed.get("confidence", 0.0)),
    }
