"""
Paper Agent: a research assistant for scientific literature.

Architecture follows a planner / executor / critic loop, inspired by the
design patterns in Ock et al.'s Adsorb-Agent (2024) and AgentD. A planner
decomposes a user query into sub-tasks, tool-using executors retrieve and
summarize papers, and a critic validates intermediate outputs before they
are returned to the user. The critic can send control back to the planner
if a result is insufficient or contradicts prior evidence.

Design notes
------------
- State is a typed dataclass, not a dict. This makes it trivial to log
  intermediate steps and to write deterministic tests for each node.
- LangGraph is used for the control flow because the critic -> planner
  edge is a true cycle, not a linear chain. LCEL's Runnable composition
  does not express cycles cleanly.
- Tool calls are logged with token accounting so experiments are
  reproducible and costs are visible.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Literal

from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph

from .tools import ArxivSearchTool, PaperSummarizeTool, CitationLookupTool

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class AgentState:
    """Shared state passed between graph nodes.

    Using a dataclass (not a TypedDict) lets us type-check mutations and
    keep reducer logic explicit. LangGraph merges fields by replacement
    unless a reducer is registered; we keep it simple here.
    """

    query: str
    plan: list[str] = field(default_factory=list)
    current_step: int = 0
    evidence: list[dict] = field(default_factory=list)
    critic_feedback: str | None = None
    revision_count: int = 0
    final_answer: str | None = None

    # Hard cap so a misbehaving critic cannot loop forever.
    max_revisions: int = 3


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class PaperAgent:
    """Planner / executor / critic agent for literature questions.

    Parameters
    ----------
    model : str
        Model identifier. Defaults to Llama 3.3 70B on Groq — free tier
        has high rate limits suitable for multi-step agent loops, and
        inference is fast enough that the critic loop stays responsive.
    temperature : float
        Kept at 0 for reproducibility of experiments. Raise when
        generating hypotheses rather than retrieving facts.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
    ) -> None:
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.tools = {
            "arxiv_search": ArxivSearchTool(),
            "summarize_paper": PaperSummarizeTool(llm=self.llm),
            "citation_lookup": CitationLookupTool(),
        }
        self.graph = self._build_graph()
        self.tools = {
            "arxiv_search": ArxivSearchTool(),
            "summarize_paper": PaperSummarizeTool(llm=self.llm),
            "citation_lookup": CitationLookupTool(),
        }
        self.graph = self._build_graph()

    # -- graph wiring -------------------------------------------------------

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self._plan)
        g.add_node("execute", self._execute)
        g.add_node("critique", self._critique)
        g.add_node("synthesize", self._synthesize)

        g.set_entry_point("plan")
        g.add_edge("plan", "execute")
        g.add_edge("execute", "critique")
        g.add_conditional_edges(
            "critique",
            self._route_after_critique,
            {
                "continue": "execute",
                "revise_plan": "plan",
                "synthesize": "synthesize",
            },
        )
        g.add_edge("synthesize", END)
        return g.compile()

    # -- nodes --------------------------------------------------------------

    def _plan(self, state: AgentState) -> AgentState:
        """Decompose the query into ordered sub-tasks."""
        system = (
            "You are the planner for a scientific literature agent. "
            "Given a user query, emit a JSON list of 2-5 concrete sub-tasks "
            "that together answer it. Each sub-task must be a single plain "
            "string describing what to do in natural language — NOT an "
            "object, NOT a tool call. A separate router will pick the tool. "
            "Available tools (mention the tool in your task string): "
            "arxiv_search, summarize_paper, citation_lookup. "
            'Example: ["Search arxiv for recent papers on agentic catalyst '
            'discovery", "Summarize the most relevant paper found"]. '
            "Do not include prose outside the JSON array."
        )
        human = f"Query: {state.query}"
        if state.critic_feedback:
            human += (
                f"\n\nPrevious plan was judged insufficient. "
                f"Critic feedback: {state.critic_feedback}\n"
                f"Revise accordingly."
            )

        resp = self.llm.invoke([SystemMessage(system), HumanMessage(human)])
        plan = self._parse_json_list(resp.content)
        log.info("Plan (%d steps): %s", len(plan), plan)
        return AgentState(
            query=state.query,
            plan=plan,
            current_step=0,
            evidence=state.evidence,  # carry evidence across replans
            critic_feedback=None,
            revision_count=state.revision_count,
            max_revisions=state.max_revisions,
        )

    def _execute(self, state: AgentState) -> AgentState:
        """Execute the current sub-task by routing to a tool."""
        if state.current_step >= len(state.plan):
            return state

        task = state.plan[state.current_step]
        # Normalize: planner sometimes returns dicts like
        # {"tool_name": ..., "params": {...}} instead of plain strings.
        if isinstance(task, dict):
            tool_name = task.get("tool_name") or task.get("tool")
            params = task.get("params") or task.get("input") or {}
            if isinstance(params, dict):
                tool_input = params.get("query") or params.get("paper_id") or str(params)
            else:
                tool_input = str(params)
            if tool_name not in self.tools:
                tool_name, tool_input = self._route_to_tool(
                    json.dumps(task), prior_evidence=state.evidence
                )
        else:
            tool_name, tool_input = self._route_to_tool(
                task, prior_evidence=state.evidence
            )
        log.info("Step %d -> %s(%r)", state.current_step, tool_name, tool_input)

        try:
            result = self.tools[tool_name].run(tool_input)
        except Exception as exc:  # noqa: BLE001 — tool errors are recoverable
            log.warning("Tool %s failed: %s", tool_name, exc)
            result = {"error": str(exc)}

        state.evidence.append(
            {"step": state.current_step, "task": task, "tool": tool_name, "result": result}
        )
        state.current_step += 1
        return state

    def _critique(self, state: AgentState) -> AgentState:
        """Validate the latest evidence. Returns feedback if insufficient."""
        if not state.evidence:
            return state

        latest = state.evidence[-1]
        system = (
            "You are the critic. Decide whether the latest tool result "
            "adequately serves the sub-task. Respond with JSON: "
            '{"verdict": "ok" | "insufficient" | "replan", "reason": "..."}'
        )
        human = (
            f"Sub-task: {latest['task']}\n"
            f"Tool: {latest['tool']}\n"
            f"Result: {json.dumps(latest['result'])[:2000]}"
        )
        resp = self.llm.invoke([SystemMessage(system), HumanMessage(human)])
        verdict = self._parse_json_object(resp.content)
        state.critic_feedback = verdict.get("reason")
        # Stash verdict for the router to read.
        state.evidence[-1]["verdict"] = verdict.get("verdict", "ok")
        return state

    def _synthesize(self, state: AgentState) -> AgentState:
        """Assemble a final grounded answer from accumulated evidence."""
        system = (
            "Synthesize a concise answer to the user's query using only "
            "the evidence provided. Cite sources by arxiv id or title. "
            "If the evidence is insufficient, say so explicitly rather "
            "than inventing details."
        )
        human = (
            f"Query: {state.query}\n\n"
            f"Evidence:\n{json.dumps(state.evidence, indent=2)[:8000]}"
        )
        resp = self.llm.invoke([SystemMessage(system), HumanMessage(human)])
        state.final_answer = resp.content
        return state

    # -- routing ------------------------------------------------------------

    def _route_after_critique(
        self, state: AgentState
    ) -> Literal["continue", "revise_plan", "synthesize"]:
        verdict = state.evidence[-1].get("verdict", "ok") if state.evidence else "ok"

        if verdict == "replan" and state.revision_count < state.max_revisions:
            state.revision_count += 1
            return "revise_plan"

        if state.current_step < len(state.plan):
            return "continue"

        return "synthesize"

    # -- helpers ------------------------------------------------------------

    def _route_to_tool(
        self, task: str, prior_evidence: list[dict] | None = None
    ) -> tuple[str, str]:
        """Pick a tool for a sub-task. Uses the LLM as a router.

        A keyword-based router would be brittle. A dedicated classifier
        would be overkill for three tools. One LLM call is the right
        granularity at this scale.

        Prior evidence is passed in so the router can resolve references
        like "summarize the paper from step 1" into concrete inputs
        (arxiv IDs). Without this, the router invents placeholder inputs
        and the downstream tool call is wasted.
        """
        # Defensive: stringify non-string tasks.
        if not isinstance(task, str):
            task = json.dumps(task)

        context_block = self._format_prior_evidence(prior_evidence or [])

        system = (
            "You are a tool router for a scientific literature agent. "
            "Pick the best tool for the task. If the task references "
            "results from prior steps (e.g. 'the top paper', 'step 1's "
            "results'), use the prior-evidence block below to resolve "
            "the reference into a concrete input — specifically an "
            "arxiv ID like '2404.12345' when the tool needs a paper. "
            "Do NOT invent paper titles or IDs that are not in the "
            "prior evidence. Respond with JSON only: "
            '{"tool": "arxiv_search" | "summarize_paper" | "citation_lookup", '
            '"input": "..."}'
        )
        human = f"Task: {task}\n\nPrior evidence:\n{context_block}"
        resp = self.llm.invoke([SystemMessage(system), HumanMessage(human)])
        obj = self._parse_json_object(resp.content)

        # Router responses are noisy across models. Accept common key
        # variants rather than insisting on one shape.
        tool_name = (
            obj.get("tool")
            or obj.get("tool_name")
            or obj.get("name")
        )
        tool_input = (
            obj.get("input")
            or obj.get("query")
            or obj.get("arxiv_id")
            or obj.get("paper_id")
            or ""
        )

        # If the router fails entirely, pick a sensible default rather
        # than crashing the graph. arxiv_search with the raw task text
        # is the least destructive fallback — it returns data the
        # synthesizer can still use.
        if tool_name not in self.tools:
            log.warning(
                "Router returned unrecognized tool %r; "
                "falling back to arxiv_search with raw task",
                tool_name,
            )
            tool_name = "arxiv_search"
            tool_input = task

        return tool_name, tool_input
    
    @staticmethod
    def _format_prior_evidence(evidence: list[dict]) -> str:
        """Render prior evidence as a compact block for the router.

        We extract just the fields that matter for resolving references:
        the step number, the tool used, and any arxiv IDs or titles
        that came back. Full tool output would blow up prompt size.
        """
        if not evidence:
            return "(none — this is the first step)"

        lines = []
        for item in evidence:
            step = item.get("step")
            tool = item.get("tool")
            result = item.get("result", {})

            # Pull out just the identifying fields from each tool type.
            if tool == "arxiv_search":
                hits = result.get("hits", [])
                ids = [
                    f"{h.get('arxiv_id')} ({h.get('title', '')[:60]})"
                    for h in hits[:5]
                ]
                summary = "papers: " + "; ".join(ids) if ids else "no results"
            elif tool == "summarize_paper":
                summary = (
                    f"summarized {result.get('arxiv_id', '?')}: "
                    f"{result.get('title', '')[:80]}"
                )
            elif tool == "citation_lookup":
                summary = f"citation data: {str(result)[:200]}"
            else:
                summary = str(result)[:200]

            lines.append(f"Step {step} [{tool}]: {summary}")

        return "\n".join(lines)
    
    @staticmethod
    def _parse_json_list(text: str) -> list[str]:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1:
            return []
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _parse_json_object(text: str) -> dict:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return {}
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}

    # -- public API ---------------------------------------------------------

    def run(self, query: str) -> AgentState:
        initial = AgentState(query=query)
        final = self.graph.invoke(initial)
        # LangGraph returns the dataclass as a dict in some versions; normalize.
        if isinstance(final, dict):
            return AgentState(**{k: v for k, v in final.items() if k in AgentState.__dataclass_fields__})
        return final
