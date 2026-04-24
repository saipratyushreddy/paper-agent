# Paper Agent

A small LLM agent for scientific literature, built to explore the
planner / executor / critic design pattern that has become common in
recent tool-using agents for scientific domains (e.g. Adsorb-Agent, 2024).

The goal is not to build a production literature tool. It is to have a
clean, testable reference implementation of the architecture so I can
reason about where the pattern succeeds and where it breaks.

## What it does

Given a natural-language query about the literature, the agent:

1. **Plans** — decomposes the query into 2–5 sub-tasks.
2. **Executes** — routes each sub-task to one of three tools
   (`arxiv_search`, `summarize_paper`, `citation_lookup`).
3. **Critiques** — an LLM-as-judge validates the tool output against
   the sub-task. It can return `ok`, `insufficient`, or `replan`.
4. **Loops or synthesizes** — on `replan` the graph edges back to the
   planner with the critic's feedback. Otherwise it proceeds to
   synthesis.

A hard cap on revisions (default 3) prevents the critic from driving
the graph into a non-terminating loop. This is an obvious failure mode
for critic-in-the-loop architectures and is worth enforcing explicitly.

## Why this shape

Three design decisions worth calling out:

**LangGraph, not LCEL.** The critic → planner edge is a true cycle.
LCEL's `Runnable` composition expresses DAGs elegantly but cycles
require awkward recursion. LangGraph is a thin wrapper over a state
machine, which is the honest representation.

**Typed dataclass state.** Using a `@dataclass` for agent state (rather
than a `TypedDict` or raw `dict`) makes the reducer semantics visible:
you can see at a glance which fields survive a replan and which reset.
It also makes the router unit-testable without an LLM in the loop — see
`tests/test_agent.py::TestRouting`.

**LLM-based tool routing.** With three tools, a keyword router would be
brittle and a trained classifier would be overkill. One LLM call per
step is the right granularity; it also means adding a fourth tool is a
prompt edit, not a code change.

## Running it

```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_...

python -m examples.cli "What are recent agentic approaches to catalyst discovery?"
```

Tests run without an API key (the LLM-dependent nodes are not covered
by unit tests; integration tests are gated on `ANTHROPIC_API_KEY`):

```bash
pytest tests/ -v
```

## Layout

```
src/
  agent.py     # planner / executor / critic graph
  tools.py     # arxiv_search, summarize_paper, citation_lookup
examples/
  cli.py       # minimal entry point
tests/
  test_agent.py
```

## What this is not

- It does not read PDF full text. The summarizer uses the arXiv
  abstract. Section-aware chunking and map-reduce summarization are
  the natural next step and are deliberately out of scope for v0.1.
- It has no persistence. Each query starts fresh. Caching tool
  responses (especially Semantic Scholar, which rate-limits aggressively)
  is the next obvious addition.
- The critic is a single LLM call. A stronger critic would compare
  against a ground-truth retrieval set; a weaker one would be a rubric
  check. Both are worth comparing empirically.

## Things I'd want to measure next

If I were extending this as research rather than a demo:

- How often does the critic's `replan` signal actually improve the
  final answer vs. burning tokens? My prior is that most replans do
  not help, and the right policy is to allow zero or one, not three.
- How sensitive is the planner to prompt wording? In my informal
  testing, asking for "2–5 sub-tasks" gives qualitatively better plans
  than "a plan" — but I have not run this carefully.
- What's the failure mode when a sub-task is outside the tools' scope?
  Currently the agent invents a plausible-looking tool call and the
  critic does not reliably catch it. A tool-scope check before
  execution would help.

---

Built as a reference implementation while preparing for PhD work on
LLM agents for scientific discovery.
