"""Keyed agentic mode: run Lumen's own Planner + SQLAgent + VegaLiteAgent headless.

Enabled only when an LLM is configured (an API key in the environment). The host asks in natural
language and Lumen writes and runs the SQL and builds the chart itself, over the session workspace.

Headless recipe (validated, no Lumen change needed): a Planner with agents=[ChatAgent, SQLAgent,
VegaLiteAgent], interface=None, and both ``sources`` and ``source`` pre-set in context (so no
interactive SourceAgent is needed); then respond() to compute the plan and plan.execute() to run it.
"""

from __future__ import annotations

import os
from typing import Optional

from . import viz
from ._shims import _run_coro
from .session import session

# provider -> (env var, candidate Lumen Llm class names, default model)
_PROVIDERS = [
    ("anthropic", "ANTHROPIC_API_KEY", ("AnthropicAI", "Anthropic"), "claude-sonnet-4-5"),
    ("openai", "OPENAI_API_KEY", ("OpenAI",), "gpt-4o"),
]

_llm = None


def configure_llm() -> Optional[str]:
    """Configure the server-side LLM from environment keys. Returns the provider name, or None."""
    global _llm
    from lumen.ai import llm as llm_module

    model_override = os.environ.get("LUMEN_MCP_LLM_MODEL")
    for provider, env_var, class_names, default_model in _PROVIDERS:
        if not os.environ.get(env_var):
            continue
        cls = None
        for name in class_names:
            cls = getattr(llm_module, name, None)
            if cls is not None:
                break
        if cls is None:
            continue
        _llm = cls(model_kwargs={"default": {"model": model_override or default_model}}, timeout=180)
        return provider
    return None


def set_key(api_key: str, provider: str = "openai", model: Optional[str] = None) -> str:
    """Enable keyed mode at runtime by setting a provider key, then (re)configuring the LLM."""
    entry = next((item for item in _PROVIDERS if item[0] == provider), None)
    if entry is None:
        raise ValueError(f"Unknown provider {provider!r}. Use one of {[item[0] for item in _PROVIDERS]}.")
    os.environ[entry[1]] = api_key
    if model:
        os.environ["LUMEN_MCP_LLM_MODEL"] = model
    if configure_llm() is None:
        raise RuntimeError("Failed to configure the LLM with the provided key.")
    return provider


async def _run(prompt: str) -> dict:
    from lumen.ai.agents import ChatAgent, SQLAgent, VegaLiteAgent
    from lumen.ai.coordinator.planner import Planner
    from lumen.ai.editors import VegaLiteEditor

    source = session.require_source()
    coordinator = Planner(
        llm=_llm,
        agents=[ChatAgent, SQLAgent, VegaLiteAgent],
        interface=None,
        context={"sources": [source], "source": source},
    )
    plan = await coordinator.respond([{"role": "user", "content": prompt}], coordinator.context)
    outputs, out_context = await plan.execute(coordinator.context)

    result: dict = {"prompt": prompt, "sql": out_context.get("sql"), "table": out_context.get("table")}

    # Render any chart Lumen produced through our own delivery path (inline PNG + HTML + ui_uri).
    derived = out_context.get("source")
    table = out_context.get("table")
    for output in outputs or []:
        if isinstance(output, VegaLiteEditor) and derived is not None and table:
            spec = getattr(output.component, "spec", None)
            if spec:
                session.source = derived
                chart = viz.render_vegalite(spec, table)
                result.update({key: chart[key] for key in ("chart_id", "spec", "png_path", "html_path", "ui_uri")})
            break

    # First non-empty text output as a summary.
    for output in outputs or []:
        obj = getattr(output, "object", None)
        if isinstance(obj, str) and obj.strip():
            result["summary"] = obj.strip()
            break

    return result


def lumen_ask(prompt: str) -> dict:
    """Answer a natural-language request by running Lumen's own agents over the workspace.

    Requires keyed mode (an LLM configured via an API key). Lumen writes and runs the SQL and builds
    the chart itself. Returns the generated SQL, the chart (id + paths + ui_uri), and a summary.
    """
    if _llm is None:
        raise RuntimeError(
            "Keyed mode is not configured. Start the server with an LLM API key "
            "(OPENAI_API_KEY or ANTHROPIC_API_KEY) to enable lumen_ask."
        )
    return _run_coro(_run(prompt))
