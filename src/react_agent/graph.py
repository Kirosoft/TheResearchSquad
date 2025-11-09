"""Define a multi-agent research and reporting workflow.

Agents:
1. Supervisor (entry) - orchestrates workflow. DOES NOT use tools. Must output strict JSON:
    {"next": "research|analyze|finalize|end", "instructions": "...", "notes": "..."}
    - "research": delegate to Researcher
    - "analyze": delegate to Analyst
    - "finalize": delegate to ReportBuilder
    - "end": finish immediately
2. Researcher - may call search tools. Outputs JSON:
    {"research_complete": true, "findings": ["..."]}
3. Analyst - synthesizes findings + supervisor notes. Outputs JSON:
    {"analysis_complete": true, "sections": {"summary": "...", "details": "..."}, "needs_additional_research": false}
4. ReportBuilder - produces FINAL human-readable formatted report (NOT JSON) and ends graph.

All handoffs rely on parsing the last AIMessage content as JSON (except ReportBuilder final output).
If parsing fails when JSON is required, an error is raised.
"""

from collections.abc import Iterable
from datetime import UTC, datetime
from functools import wraps
from typing import Dict, List, Literal, cast
import json

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model


def require_roles(allowed_roles: Iterable[str]):
    """Decorator enforcing role-based access for graph node functions.
    Preserves the original (state, runtime, ...) signature expected by LangGraph.
    If unauthorized, returns an AIMessage denial so downstream routing can detect
    and terminate without raising exceptions.
    """
    allowed = set(allowed_roles)
    def decorator(fn):
        @wraps(fn)
        async def wrapped(state: State, runtime: Runtime[Context], *args, **kwargs):
            roles: set[str] = set()
            # Simulate roles stored on state (InputState.roles) if present
            # normally config.configurable
            if not roles and getattr(state, "roles", None):
                if isinstance(state.roles, list):
                    roles = set(state.roles)
            if not roles or roles.isdisjoint(allowed):
                denial = AIMessage(content=(
                    f"Not authorized to run {fn.__name__}. Required any of {sorted(allowed)}; got {sorted(roles)}."
                ))
                return {"messages": [denial]}
            # Proceed to underlying node
            return await fn(state, runtime, *args, **kwargs)
        return wrapped
    return decorator



# --------------------------------------------------
# Supervisor (renamed from previous call_model)
# --------------------------------------------------

@require_roles(["admin", "finance"])
async def supervisor(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Supervisor agent: plans next step, never uses tools.

    System prompt instructs model to output STRICT JSON schema:
    {"next": "research|analyze|finalize|end", "instructions": "...", "notes": "..."}
    """
    model = load_chat_model(runtime.context.model)  # no tools bound

    system_message = (
        "You are the SUPERVISOR orchestrating a research & analysis workflow.\n"
        "ALWAYS respond with STRICT valid JSON matching schema: \n"
        "{\n  \"next\": one of [research, analyze, finalize, end],\n  \"instructions\": string (concise directive for the target agent),\n  \"notes\": optional string\n}\n"
        "Do not include markdown fences.\n"
        f"System time: {datetime.now(tz=UTC).isoformat()}"
    )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": system_message},
                *state.messages,
            ]
        ),
    )

    # Validate JSON (raise early if invalid so user can correct prompt/behavior)
    try:
        data = json.loads(str(response.content))
        if not isinstance(data, dict) or "next" not in data:
            raise ValueError("Supervisor must output a dict with 'next'.")
        if data["next"] not in {"research", "analyze", "finalize", "end"}:
            raise ValueError("Invalid 'next' value from Supervisor.")
    except Exception as exc:  # noqa: BLE001
        # Replace content with error message (still JSON for traceability)
        response = AIMessage(
            id=response.id,
            content=json.dumps(
                {
                    "error": "Invalid supervisor JSON output",
                    "details": str(exc),
                }
            ),
        )

    return {"messages": [response]}


# --------------------------------------------------
# Researcher (tool-using agent)
# --------------------------------------------------


async def researcher(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Researcher agent: performs web searches using bound tools.

    Must output JSON: {"research_complete": true, "findings": [string, ...]}
    If additional searches are required, include tool calls before final JSON.
    """
    model = load_chat_model(runtime.context.model).bind_tools(TOOLS)

    system_message = (
        "You are the RESEARCHER.\n"
        "INSTRUCTIONS:\n"
        "1. Use the search tool for ALL factual retrieval (no unsupported prior knowledge).\n"
        "2. Accumulate concise atomic fact strings.\n"
        "3. When finished, OUTPUT STRICT JSON ONLY (no prose, no markdown fences). Schema example:\n"
        "{\n  \"research_complete\": true,\n  \"findings\": [\"Fact 1...\", \"Fact 2...\"]\n}\n"
        "4. If you still need more tool calls, produce a tool call instead of the final JSON.\n"
        "5. 'findings' MUST be an array of strings (not objects).\n"
        "If you deviate, the system will raise a JSON schema error."
    )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": system_message},
                *state.messages,
            ]
        ),
    )

    # If tool calls present we skip validation until after tools resolve
    if not response.tool_calls:
        raw_content = response.content
        # Some providers may return list segments; normalize to string
        if isinstance(raw_content, list):  # type: ignore[redundant-cast]
            raw_content = "".join(str(part) for part in raw_content)
        content_str = str(raw_content).strip()
        # Attempt resilient JSON extraction (grab first {...} span if extra text leaked)
        json_candidate = content_str
        if not (content_str.startswith("{") and content_str.endswith("}")):
            first = content_str.find("{")
            last = content_str.rfind("}")
            if first != -1 and last != -1 and last > first:
                json_candidate = content_str[first : last + 1]
        try:
            data = json.loads(json_candidate)
            # Coerce findings to list[str] if possible
            findings = data.get("findings")
            if isinstance(findings, list):
                findings = [str(f).strip() for f in findings if str(f).strip()]
            else:
                # Allow single string -> list
                if isinstance(findings, str) and findings.strip():
                    findings = [findings.strip()]
            # Validate required fields
            if not (
                isinstance(data, dict)
                and (data.get("research_complete") is True or data.get("complete") is True)
                and isinstance(findings, list)
                and all(isinstance(x, str) for x in findings)
            ):
                raise ValueError("Researcher JSON schema mismatch")
            # Normalize schema (support 'complete' synonym)
            normalized = {
                "research_complete": True,
                "findings": findings,
            }
            response = AIMessage(id=response.id, content=json.dumps(normalized))
        except Exception as exc:  # noqa: BLE001
            response = AIMessage(
                id=response.id,
                content=json.dumps(
                    {
                        "error": "Invalid researcher JSON output",
                        "details": str(exc),
                        "raw": content_str[:500],  # truncate for safety
                    }
                ),
            )

    return {"messages": [response]}


# --------------------------------------------------
# Analyst (non-tool agent)
# --------------------------------------------------


async def analyst(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """Analyst agent: synthesizes findings into structured report parts.

    Outputs STRICT JSON: {"analysis_complete": true, "sections": {"summary": str, "details": str}, "needs_additional_research": bool}
    """
    model = load_chat_model(runtime.context.model)
    system_message = (
        "You are the ANALYST. You receive research findings and supervisor notes.\n"
        "Produce JSON ONLY: {\n  \"analysis_complete\": true,\n  \"sections\": {\"summary\": str, \"details\": str},\n  \"needs_additional_research\": bool\n}\n"
        "The summary is a high-level answer; details contain bullet-style elaboration."
    )
    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": system_message},
                *state.messages,
            ]
        ),
    )
    try:
        data = json.loads(str(response.content))
        if not (
            isinstance(data, dict)
            and data.get("analysis_complete") is True
            and isinstance(data.get("sections"), dict)
            and "summary" in data["sections"]
            and "details" in data["sections"]
            and isinstance(data.get("needs_additional_research"), bool)
        ):
            raise ValueError("Analyst JSON schema mismatch")
    except Exception as exc:  # noqa: BLE001
        response = AIMessage(
            id=response.id,
            content=json.dumps(
                {"error": "Invalid analyst JSON output", "details": str(exc)}
            ),
        )
    return {"messages": [response]}


# --------------------------------------------------
# ReportBuilder (final formatting agent)
# --------------------------------------------------


async def report_builder(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:
    """ReportBuilder agent: converts latest analysis JSON into a polished human-readable report.

    Output MUST be nicely formatted plain text (NOT JSON). Include title, executive summary,
    detailed findings, and optional next steps. This node ends the graph.
    """
    model = load_chat_model(runtime.context.model)
    system_message = (
        "You are the REPORT BUILDER. Convert the most recent analysis JSON into a well-formatted report.\n"
        "Output plain text only (no JSON, no markdown fences). Use clear headings in ALL CAPS."
    )
    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": system_message},
                *state.messages,
            ]
        ),
    )
    return {"messages": [response]}
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Register nodes
builder.add_node(supervisor)
builder.add_node(researcher)
builder.add_node(analyst)
builder.add_node(report_builder)
builder.add_node("tools", ToolNode(TOOLS))

# Entry point
builder.add_edge("__start__", "supervisor")


def route_supervisor_output(state: State) -> Literal[
    "researcher", "analyst", "report_builder", "__end__"
]:
    """Route based on Supervisor JSON 'next' field."""
    last = state.messages[-1]
    if not isinstance(last, AIMessage):
        raise ValueError("Last message from supervisor must be AIMessage")
    try:
        data = json.loads(str(last.content))
        nxt = data.get("next")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Supervisor output not valid JSON: {exc}") from exc
    if nxt == "research":
        return "researcher"
    if nxt == "analyze":
        return "analyst"
    if nxt == "finalize":
        return "report_builder"
    if nxt == "end":
        return "__end__"
    raise ValueError(f"Unknown next directive from supervisor: {nxt}")


def route_researcher_output(state: State) -> Literal["tools", "supervisor"]:
    """If researcher still has tool calls execute them; else return to supervisor."""
    last = state.messages[-1]
    if not isinstance(last, AIMessage):
        raise ValueError("Researcher last message must be AIMessage")
    if last.tool_calls:
        return "tools"
    return "supervisor"


def route_analyst_output(state: State) -> Literal["supervisor"]:
    """Always send control back to supervisor after analyst completes."""
    return "supervisor"


# Conditional edges
builder.add_conditional_edges("supervisor", route_supervisor_output)
builder.add_conditional_edges("researcher", route_researcher_output)
builder.add_conditional_edges("analyst", route_analyst_output)

# Tools always return to researcher
builder.add_edge("tools", "researcher")

# Compile graph
graph = builder.compile(name="Research & Reporting Agent")
