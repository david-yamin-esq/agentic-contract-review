"""Verifier agent.

Independently checks the agents' outputs against the source contract.
Specifically:

1. For each extracted clause, verifies the quoted text actually appears in
   the contract (catches the most common hallucination — fabricated quotes).
2. For each comparison, asks the LLM whether the rationale is supported by
   the clause text and the playbook context.

Findings feed the HITL gate. If the verifier flags a high-severity issue,
the workflow forces human review regardless of the headline risk score.
"""

from __future__ import annotations
import re

from src.llm import call_llm
from src.state import ReviewState, VerifierFinding


SYSTEM_PROMPT = """You are an independent verifier checking an AI's contract
review for accuracy and hallucination.

You are given a clause as quoted by the AI and the corresponding source text
from the contract. Determine whether the quote is faithful (allowing for
trivial whitespace differences only) and whether the AI's stated rationale is
supported.

Return strict JSON:
{
  "quote_faithful": true | false,
  "rationale_supported": true | false,
  "issue": "<short description, or empty string if no issue>",
  "severity": "low" | "medium" | "high"
}
No prose outside the JSON object."""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _quote_in_source(quote: str, source: str) -> bool:
    """Coarse fidelity check — exact match after whitespace normalization."""
    return _normalize(quote) in _normalize(source)


def verify(state: ReviewState) -> ReviewState:
    run_id = state["run_id"]
    contract_text = state["contract_text"]
    findings: list[VerifierFinding] = []

    for clause in state.get("clauses", []):
        # Cheap mechanical check first
        if not _quote_in_source(clause["text"], contract_text):
            findings.append(
                VerifierFinding(
                    clause_id=clause["clause_id"],
                    issue="Extracted quote not found verbatim in source contract",
                    severity="high",
                    quote_from_ai=clause["text"][:200],
                    quote_from_source=None,
                )
            )
            # Don't bother LLM-verifying a fabricated quote
            continue

        # LLM-based rationale check
        comparison = clause.get("comparison", {})
        user = (
            f"AI-extracted clause text:\n{clause['text']}\n\n"
            f"AI's comparison rationale: {comparison.get('rationale', '')}\n"
            f"AI's alignment verdict: {comparison.get('alignment', '')}\n\n"
            f"Source contract excerpt (verbatim):\n"
            f"{_find_source_excerpt(clause['text'], contract_text)}"
        )

        result = call_llm(
            run_id=run_id,
            stage="verifier",
            system=SYSTEM_PROMPT,
            user=user,
            expect_json=True,
            max_tokens=500,
        )
        parsed = result.get("parsed") or {}

        if not parsed.get("quote_faithful", True) or not parsed.get(
            "rationale_supported", True
        ):
            findings.append(
                VerifierFinding(
                    clause_id=clause["clause_id"],
                    issue=parsed.get("issue", "Unspecified verifier finding"),
                    severity=parsed.get("severity", "medium"),
                    quote_from_ai=clause["text"][:200],
                    quote_from_source=None,
                )
            )

    # HITL gate trigger
    has_high_severity = any(f["severity"] == "high" for f in findings)
    risk_above_threshold = (
        state.get("overall_risk_score", 0) >= _hitl_threshold()
    )
    requires_human_review = has_high_severity or risk_above_threshold

    return {
        **state,
        "verifier_findings": findings,
        "requires_human_review": requires_human_review,
    }


def _hitl_threshold() -> int:
    from config import HITL_RISK_THRESHOLD
    return HITL_RISK_THRESHOLD


def _find_source_excerpt(quote: str, source: str, window: int = 300) -> str:
    """Return a window of source text around the (normalized) quote location."""
    norm_source = _normalize(source)
    norm_quote = _normalize(quote)
    idx = norm_source.find(norm_quote)
    if idx == -1:
        return source[:window * 2]
    # Map back to original indexing approximately
    approx_start = max(0, int(idx * len(source) / max(len(norm_source), 1)) - window)
    approx_end = min(len(source), approx_start + len(quote) + window * 2)
    return source[approx_start:approx_end]
