"""LLM client wrapper.

All agents call into this module rather than the Anthropic SDK directly. That
gives us a single chokepoint for: (1) audit logging, (2) retries, (3) prompt
version tracking, and (4) provider swapping. Swap the implementation here to
move to Azure OpenAI or another provider without touching any agent code.
"""

from __future__ import annotations
import json
from typing import Any, Optional

from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from config import ANTHROPIC_API_KEY, LLM_MODEL, PROMPT_VERSIONS
from src.audit import log_event


_client: Optional[Anthropic] = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_llm(
    *,
    run_id: str,
    stage: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    expect_json: bool = False,
) -> dict[str, Any]:
    """Make an LLM call with audit logging.

    Returns a dict with keys: text, parsed (if expect_json), usage.
    """
    client = _get_client()
    prompt_version = PROMPT_VERSIONS.get(stage, "v1")

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )

    text = "".join(block.text for block in response.content if block.type == "text")
    parsed: Optional[Any] = None
    parse_error: Optional[str] = None

    if expect_json:
        # Tolerate fenced code blocks that wrap JSON
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            parse_error = str(e)

    log_event(
        run_id=run_id,
        stage=stage,
        event_type="llm_call",
        model=LLM_MODEL,
        prompt_version=prompt_version,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        payload={
            "system": system,
            "user": user,
            "response_text": text,
            "parsed": parsed,
            "parse_error": parse_error,
        },
    )

    return {
        "text": text,
        "parsed": parsed,
        "parse_error": parse_error,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }
