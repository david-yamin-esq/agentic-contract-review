# Agentic Contract Review

**A multi-agent contract review pipeline built around audit-grade governance.**

Most legal AI products demonstrate capability and skip the harder question: how do you defend the system's outputs when something goes wrong? This pipeline is built around that question. Five specialized agents extract clauses, compare each against an embedded playbook via RAG, score risk, and pause for human review when warranted. An independent verifier checks the agents' work before any conclusions are finalized. Every LLM call, retrieval, and human decision is logged in sufficient detail to reconstruct the run six months later.

The system is roughly 1,500 lines of Python and runs locally on a developer laptop with an Anthropic API key.

## What this catches that simpler systems miss

In the first production run against a vendor-favorable services agreement, the comparison agent claimed Section 3 had no termination provisions. The verifier read the source contract independently, found that Section 3.2 explicitly granted Vertex termination rights, and raised a high-severity finding:

> "AI claims termination provisions are 'omitted entirely' but source shows Section 3.2 exists with Vertex termination rights. AI did not review complete termination section before concluding provisions were missing."

That finding came from the verifier, not the comparison agent. The verifier reads source-of-truth directly, runs a different prompt asking "is the quote faithful and is the rationale supported?", and forms an independent judgment. When its judgment disagrees with the comparison agent's, it raises a finding that forces the pipeline to pause for human review — regardless of the headline risk score.

This is governance implemented as plumbing rather than policy. The pause-and-resume is an architectural property of the pipeline, with an audit trail to prove it.

## Architecture

```
[contract] → classify → extract → compare → score → verify → [HITL gate?] → finalize → [report + audit]
                                      ↑
                                   playbook (RAG, ChromaDB)
```

Five agents, sequenced through LangGraph with SQLite checkpointing:

- **Classify** identifies the contract type (NDA, MSA, SaaS, employment, consulting, licensing, etc.)
- **Extract** pulls verbatim clauses across ten target types (limitation of liability, indemnification, IP, data protection, termination, confidentiality, governing law, warranty, payment terms, non-compete)
- **Compare** retrieves the top-2 playbook positions for each clause via vector similarity, then asks the LLM to compare the clause against those positions and identify deviations
- **Score** assigns a 0–100 risk score per clause, then a separate overall contract score that explicitly considers compounding deviations rather than averaging
- **Verify** independently checks for hallucination — a mechanical fidelity check (does the AI's quote actually appear in the source?) plus an LLM-based rationale check

If the overall risk score crosses threshold (default 60) **or** the verifier raises any high-severity finding, the pipeline pauses at a human-in-the-loop gate. The reviewer sees per-clause findings with override fields, submits a decision (approve / reject / modify with notes), and the graph resumes from where it paused. All decisions and overrides are written to the audit log.

## Design decisions

The architectural choices that distinguish this from a generic agent demo:

**LangGraph over CrewAI / AutoGen.** The pipeline is a state machine with conditional routing and a hard pause-and-resume requirement that must survive process restarts. LangGraph's SQLite checkpointer plus `interrupt_before` model this exactly. CrewAI optimizes for agents that decide their own order; AutoGen for multi-turn conversations. Either would mean fighting the framework.

**Independent verifier, not self-critique.** Asking the same agent to grade its own output fails the same way human self-grading fails. A verifier with a different prompt, given source-of-truth context, catches a different distribution of errors. The mechanical "is this quote in the source?" check catches the most common hallucination class without spending an LLM call.

**Audit logging at the LLM chokepoint.** Every agent calls `src/llm.py:call_llm`. There is no code path that calls the Anthropic SDK directly. The audit log is therefore complete by construction rather than by discipline.

**Prompt versions in every audit row.** Every LLM call records the prompt version active at call time. The audit log can be filtered by prompt generation. The current registry is a Python dict — the smallest contract a real prompt registry would honor.

**Per-clause RAG, not whole-document.** The playbook is split into H2 sections and embedded individually. Each clause queries for the top-2 most relevant playbook positions. More accurate than dumping the whole playbook into context, and cheaper than reranking after broad retrieval.

## Cost and runtime

Reconstructed from the audit log of an actual run on the included aggressive sample contract:

- **$0.33 per contract** at Claude Sonnet 4.5 list pricing
- **66 LLM calls** (1 classifier + 1 extractor + 21 compare + 22 score + 21 verifier) on a contract that produced 21 extracted clauses
- **21 RAG retrievals** — one per extracted clause
- **88 audit events** total
- **~5 minutes** end-to-end runtime, sequential

At 10,000 contracts per month — the volume of a midsize ALSP contract operation — that's roughly $3,300/month in LLM spend. The bottleneck at that scale is human review capacity, not AI throughput. Production hardening would parallelize the per-clause loops (currently sequential) and apply prompt caching to the repeated playbook context, both of which would cut runtime and cost meaningfully.

## What this version does not yet do

Each of these is a deliberate scope boundary in v1 and a target for the next iteration:

**Calibration of the per-clause risk scorer at the extreme end.** The current scoring prompt weights deviation count more heavily than severity-when-extreme, which produces well-calibrated scores in the middle of the range but undershoots on walkaway-level clauses (a $1,000 vendor liability cap currently scores in the high 80s where it should arguably score 95+). The fix is a revised scoring prompt validated against a labeled evaluation set — work that belongs alongside a proper eval harness rather than on its own.

**A formal verifier evaluation harness.** The mechanical fidelity check (does the AI's quote actually appear in the source?) is robust by construction. The LLM rationale check is a softer signal — it catches obvious hallucinations more reliably than subtle ones. Quantifying that performance requires synthetic hallucination datasets and a recall measurement. This is significant enough that it likely becomes its own standalone tool: a citation and rationale verifier that drops alongside any LLM-based legal product, not just this one.

**Document ingestion beyond plain text.** Real contracts arrive as PDFs and DOCXs, sometimes scanned. The architecture is content-agnostic — text in, structured output — so adding ingestion is a discrete pre-processing layer (`pdfplumber`, `python-docx`, OCR for scans) rather than a rewrite. Scoped for v2.

**Parallelization of the per-clause loops.** Compare, score, and verify currently run sequentially over each extracted clause. LangGraph's `Send` API supports fan-out, and prompt caching against the repeated playbook context would compound the speedup. Both are straightforward and would meaningfully reduce both runtime and cost; deferred to v2 because the sequential version is easier to demonstrate and audit.

**Production hardening.** Multi-tenancy, SSO, RBAC, secrets management, observability, queue-based orchestration in place of inline `graph.stream()` — the standard list of things that turn a single-tenant working system into a production deployment. The architecture supports those additions without redesign; building them is a separate effort from demonstrating that the agentic and governance patterns work.

**Variance reduction across runs.** Even at temperature=0, repeated runs produce slightly different outputs — clause scores can shift ±10 points and verifier findings can appear in one run and not another. The architecture currently mitigates this by keying HITL decisions on the more-stable overall score rather than individual clause scores. A v2 approach would add ensemble runs with majority voting on borderline cases, which trades some additional cost for substantially tighter consistency.

## Running it locally

Requires Python 3.12, an Anthropic API key, and ~3 GB of disk space for the local embedding model and dependencies.

```bash
git clone https://github.com/david-yamin-esq/agentic-contract-review.git
cd agentic-contract-review
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env               # edit and add your ANTHROPIC_API_KEY
streamlit run app.py
```

In the Streamlit UI, click **(Re)build playbook index** in the sidebar (one-time), select `aggressive_services.txt` from the sample dropdown, and click **Run review pipeline**. A live progress panel shows each stage executing; after ~5 minutes the pipeline pauses at the HITL gate. Submit any decision to see the final report and download the audit trail.

## Project structure

```
agentic-contract-review/
├── app.py                          # Streamlit UI (upload → review → done)
├── config.py                       # Paths, model, thresholds, prompt versions
├── requirements.txt
├── src/
│   ├── state.py                    # Typed graph state
│   ├── audit.py                    # SQLite audit log
│   ├── progress.py                 # UI progress reporter (singleton)
│   ├── llm.py                      # LLM chokepoint with audit + retries
│   ├── rag.py                      # ChromaDB playbook retrieval
│   ├── orchestrator.py             # LangGraph wiring + HITL interrupt
│   └── agents/
│       ├── classifier.py           # Contract type
│       ├── extractor.py            # Verbatim clause pull
│       ├── playbook_agent.py       # RAG-driven comparison
│       ├── risk_scorer.py          # Per-clause + overall scoring
│       └── verifier.py             # Hallucination check
└── data/
    ├── playbook/playbook.md        # Sample playbook (mock)
    └── sample_contracts/
        └── aggressive_services.txt # Demonstration contract
```

A detailed annotated walkthrough of every file is in `docs/code_walkthrough.txt`. A slide-deck summary of the architecture, verifier story, and economics is in `docs/agentic-contract-review-deck.pptx`.

## About

Built by **David Yamin** — twenty-two-year AmLaw 100 commercial litigation partner (Bingham McCutchen, now Morgan Lewis); eight years senior data scientist at a major league baseball club; M.S. Applied Data Science (Syracuse), J.D. (NYU). Massachusetts and Florida bar admissions; AIGP credential in progress; Stanford Law and UC Berkeley legal AI programs (2026).

This project is part of an ongoing portfolio of legal AI engineering work focused on the intersection of legal practice and operational AI infrastructure — the things that have to be true for AI to be deployable in environments where being wrong has visible consequences. More repositories forthcoming.

For inquiries: [LinkedIn](https://linkedin.com/in/davidyamin)

## License

MIT. Use, modify, and adapt freely.
