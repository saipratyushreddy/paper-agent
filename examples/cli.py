"""
Quick CLI for the paper agent.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m examples.cli "What are recent agentic approaches to catalyst discovery?"
"""

from __future__ import annotations

import logging
import sys

from src import PaperAgent


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    query = " ".join(sys.argv[1:])
    agent = PaperAgent()
    state = agent.run(query)

    print("\n" + "=" * 72)
    print("QUERY:", query)
    print("=" * 72)
    print("\nPLAN:")
    for i, step in enumerate(state.plan):
        print(f"  {i + 1}. {step}")

    print(f"\nEVIDENCE GATHERED: {len(state.evidence)} step(s)")
    print(f"PLAN REVISIONS: {state.revision_count}")

    print("\nANSWER:")
    print(state.final_answer or "(no answer produced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
