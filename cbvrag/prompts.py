from __future__ import annotations

from typing import Iterable


def answer_prompt(question: str, selected_snippets: Iterable[str], branch_summary: str, global_summary: str) -> str:
    snippets = "\n".join([f"- {s}" for s in selected_snippets])
    return (
        "Answer the question using only the snippets. Keep it concise.\n"
        f"Question: {question}\n"
        f"Branch summary: {branch_summary}\n"
        f"Global summary: {global_summary}\n"
        f"Snippets:\n{snippets}\n"
        "Final answer:"
    )


def counterfactual_prompt(question: str, branch_type: str) -> str:
    return f"Create one short counterfactual hypothesis for this question ({branch_type}): {question}"


def verify_prompt(question: str, claim: str, snippets: Iterable[str]) -> str:
    limited = list(snippets)[:3]
    text = "\n".join([f"- {s}" for s in limited])
    return (
        "Return one label: supported, contradicted, unknown.\n"
        f"Question: {question}\n"
        f"Claim: {claim}\n"
        f"Evidence:\n{text}\n"
        "Label:"
    )
