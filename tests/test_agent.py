"""
Unit tests for the paper agent.

These cover the deterministic parts of the agent — JSON parsing, tool
contracts, state transitions, the router's termination condition. The
LLM-dependent nodes are exercised in integration tests (not included
here; they require an API key).
"""

from __future__ import annotations

import pytest

from src.agent import AgentState, PaperAgent
from src.tools import PaperSummarizeTool


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


class TestJsonParsing:
    def test_plan_list_parsed_from_clean_json(self):
        out = PaperAgent._parse_json_list('["step one", "step two"]')
        assert out == ["step one", "step two"]

    def test_plan_list_parsed_from_prose_wrapped_json(self):
        # LLMs often wrap JSON in explanatory text.
        text = 'Sure! Here is the plan:\n["a", "b"]\nLet me know if...'
        assert PaperAgent._parse_json_list(text) == ["a", "b"]

    def test_plan_list_returns_empty_on_garbage(self):
        assert PaperAgent._parse_json_list("no json here") == []

    def test_plan_list_returns_empty_on_malformed_json(self):
        assert PaperAgent._parse_json_list("[unclosed") == []

    def test_object_parsed_from_prose_wrapped_json(self):
        text = 'Verdict: {"verdict": "ok", "reason": "looks good"}'
        out = PaperAgent._parse_json_object(text)
        assert out == {"verdict": "ok", "reason": "looks good"}


# ---------------------------------------------------------------------------
# arxiv id extraction
# ---------------------------------------------------------------------------


class TestArxivIdExtraction:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("summarize 2404.12345", "2404.12345"),
            ("https://arxiv.org/abs/2301.00001v2", "2301.00001v2"),
            ("the paper 2410.99999 please", "2410.99999"),
            ("legacy id cs.LG/0401001", "cs.LG/0401001"),
        ],
    )
    def test_extracts_known_formats(self, text, expected):
        assert PaperSummarizeTool._extract_arxiv_id(text) == expected

    def test_returns_none_when_no_id(self):
        assert PaperSummarizeTool._extract_arxiv_id("just a title") is None


# ---------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------


class TestRouting:
    def _agent(self):
        # We do not instantiate the real LLM — we only call routing logic.
        agent = PaperAgent.__new__(PaperAgent)
        return agent

    def test_synthesize_when_plan_exhausted_and_verdict_ok(self):
        agent = self._agent()
        state = AgentState(
            query="q",
            plan=["a"],
            current_step=1,
            evidence=[{"verdict": "ok"}],
        )
        assert agent._route_after_critique(state) == "synthesize"

    def test_continue_when_more_steps_remain(self):
        agent = self._agent()
        state = AgentState(
            query="q",
            plan=["a", "b"],
            current_step=1,
            evidence=[{"verdict": "ok"}],
        )
        assert agent._route_after_critique(state) == "continue"

    def test_replan_on_critic_request_under_budget(self):
        agent = self._agent()
        state = AgentState(
            query="q",
            plan=["a"],
            current_step=1,
            evidence=[{"verdict": "replan"}],
            revision_count=0,
            max_revisions=3,
        )
        assert agent._route_after_critique(state) == "revise_plan"
        assert state.revision_count == 1

    def test_replan_budget_is_respected(self):
        # Critic asks to replan but we've hit the cap. Graph must terminate.
        agent = self._agent()
        state = AgentState(
            query="q",
            plan=["a"],
            current_step=1,
            evidence=[{"verdict": "replan"}],
            revision_count=3,
            max_revisions=3,
        )
        # Should not return "revise_plan" once budget is exhausted.
        assert agent._route_after_critique(state) != "revise_plan"
