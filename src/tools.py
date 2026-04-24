"""
Tools used by the paper agent.

Each tool has a narrow contract (`run(input: str) -> dict`) so the
agent's tool-routing code does not need to know tool-specific signatures.
Real work is delegated to vetted libraries (arxiv, requests) rather than
hand-rolled scraping.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import arxiv
import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# arxiv search
# ---------------------------------------------------------------------------


@dataclass
class ArxivSearchTool:
    """Search arXiv and return structured hits.

    Uses the official `arxiv` client which wraps the Atom feed API and
    handles pagination and rate limits. We cap results to keep prompt
    size bounded downstream.
    """

    max_results: int = 5

    def run(self, query: str) -> dict[str, Any]:
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        hits = []
        for r in search.results():
            hits.append(
                {
                    "arxiv_id": r.entry_id.rsplit("/", 1)[-1],
                    "title": r.title.strip(),
                    "authors": [a.name for a in r.authors][:5],
                    "summary": r.summary.strip()[:500],
                    "published": r.published.isoformat(),
                    "pdf_url": r.pdf_url,
                }
            )
        return {"query": query, "hits": hits}


# ---------------------------------------------------------------------------
# paper summarization
# ---------------------------------------------------------------------------


@dataclass
class PaperSummarizeTool:
    """Summarize a paper by arxiv id or URL.

    Full-text extraction is intentionally out of scope here — we use the
    abstract + whatever metadata arxiv exposes. Extending this to pull
    the PDF, chunk with section-aware splitting, and run map-reduce
    summarization is the obvious next iteration.
    """

    llm: BaseChatModel

    def run(self, query: str) -> dict[str, Any]:
        arxiv_id = self._extract_arxiv_id(query)
        if not arxiv_id:
            return {"error": f"could not extract arxiv id from {query!r}"}

        try:
            paper = next(arxiv.Search(id_list=[arxiv_id]).results())
        except StopIteration:
            return {"error": f"no arxiv paper found for id {arxiv_id}"}

        system = (
            "Summarize this paper for a machine-learning-literate reader. "
            "Return four labeled sections, each 1-3 sentences: Problem, "
            "Method, Results, Limitations. Be specific, no filler."
        )
        human = f"Title: {paper.title}\n\nAbstract: {paper.summary}"
        resp = self.llm.invoke([SystemMessage(system), HumanMessage(human)])
        return {
            "arxiv_id": arxiv_id,
            "title": paper.title.strip(),
            "summary": resp.content,
        }

    @staticmethod
    def _extract_arxiv_id(text: str) -> str | None:
        # Modern IDs (e.g. 2404.12345) and legacy IDs (e.g. cs.LG/0401001).
        modern = re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", text)
        if modern:
            return modern.group(0)
        legacy = re.search(r"\b[a-z\-]+(?:\.[A-Z]{2})?/\d{7}\b", text)
        return legacy.group(0) if legacy else None


# ---------------------------------------------------------------------------
# citation lookup
# ---------------------------------------------------------------------------


@dataclass
class CitationLookupTool:
    """Look up citation counts and related work via Semantic Scholar.

    The public Semantic Scholar API is unauthenticated but rate-limited.
    A production version would use an API key and a caching layer; this
    is fine for a demo and for interviews.
    """

    base_url: str = "https://api.semanticscholar.org/graph/v1/paper"
    timeout_s: float = 10.0

    def run(self, query: str) -> dict[str, Any]:
        # Accept either an arxiv id or a free-text title.
        arxiv_id = PaperSummarizeTool._extract_arxiv_id(query)
        if arxiv_id:
            url = f"{self.base_url}/arXiv:{arxiv_id}"
            params = {"fields": "title,citationCount,references.title,references.year"}
        else:
            url = f"{self.base_url}/search"
            params = {"query": query, "limit": 1, "fields": "title,citationCount"}

        try:
            r = requests.get(url, params=params, timeout=self.timeout_s)
            r.raise_for_status()
        except requests.RequestException as exc:
            return {"error": f"semantic scholar request failed: {exc}"}

        return r.json()
