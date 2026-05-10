"""Central configuration. All tunable values live here so prompts, thresholds,
and model choices can be version-controlled and audited."""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT = Path(__file__).parent.resolve()
DATA_DIR = ROOT / "data"
PLAYBOOK_DIR = DATA_DIR / "playbook"
CONTRACTS_DIR = DATA_DIR / "sample_contracts"
AUDIT_DIR = DATA_DIR / "audit"
AUDIT_DB_PATH = AUDIT_DIR / "audit.db"
CHECKPOINT_DB_PATH = AUDIT_DIR / "checkpoints.db"
CHROMA_DIR = AUDIT_DIR / "chroma"

# LLM
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-5-20250929")

# HITL thresholds
HITL_RISK_THRESHOLD = int(os.getenv("HITL_RISK_THRESHOLD", "60"))

# Prompt versions — bump these when you change a prompt so audit logs
# can be filtered by prompt generation
PROMPT_VERSIONS = {
    "classifier": "v1",
    "extractor": "v1",
    "playbook": "v1",
    "risk_scorer": "v1",
    "verifier": "v1",
}

# Ensure dirs exist
for d in (DATA_DIR, PLAYBOOK_DIR, CONTRACTS_DIR, AUDIT_DIR, CHROMA_DIR):
    d.mkdir(parents=True, exist_ok=True)
