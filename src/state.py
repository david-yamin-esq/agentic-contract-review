"""Typed state shared across the LangGraph workflow.

LangGraph passes this dict-like state between nodes. Each agent reads what it
needs and writes its outputs into specific keys.
"""

from __future__ import annotations
from typing import TypedDict, Optional, Any


class Clause(TypedDict, total=False):
    clause_id: str
    clause_type: str
    text: str
    location_hint: str  # e.g. "Section 4.2" or character offsets
    playbook_matches: list[dict[str, Any]]
    comparison: dict[str, Any]  # alignment, deviations, suggested redline
    risk_score: int  # 0-100
    risk_rationale: str


class VerifierFinding(TypedDict):
    clause_id: str
    issue: str
    severity: str  # low | medium | high
    quote_from_ai: str
    quote_from_source: Optional[str]


class HITLDecision(TypedDict, total=False):
    decision: str  # approve | reject | modify
    reviewer: str
    notes: str
    modified_clauses: dict[str, dict[str, Any]]  # clause_id -> overrides


class ReviewState(TypedDict, total=False):
    run_id: str
    contract_name: str
    contract_text: str
    contract_type: str
    contract_type_confidence: float
    clauses: list[Clause]
    overall_risk_score: int
    overall_risk_rationale: str
    verifier_findings: list[VerifierFinding]
    requires_human_review: bool
    hitl_decision: Optional[HITLDecision]
    final_report: Optional[str]
    error: Optional[str]
