from __future__ import annotations
from typing import Iterable


def answer_prompt(question: str, selected_snippets: Iterable[str], branch_summary: str, global_summary: str) -> str:
    snippets = list(selected_snippets)
    if not snippets:
        return (
            "Answer the following question as concisely as possible.\n\n"
            f"Question: {question}\n\n"
        )

    evidence_text = "\n".join([f"[{i+1}] {s}" for i, s in enumerate(snippets)])

    context_parts = []
    if (branch_summary or "").strip():
        context_parts.append(f"Additional context: {branch_summary.strip()}")
    if (global_summary or "").strip():
        context_parts.append(f"Summary: {global_summary.strip()}")
    context_block = ("\n" + "\n".join(context_parts) + "\n") if context_parts else "\n"

    return (
        "You are a precise question answering system.\n"
        "Use ONLY the evidence below to answer the question.\n\n"
        f"Evidence:\n{evidence_text}\n"
        f"{context_block}\n"
        f"Question: {question}\n\n"
        "Answer (1-5 words only):"
    )


def counterfactual_prompt(question: str, branch_type: str) -> str:
    return (
        f"Generate one short counterfactual hypothesis for this question "
        f"({branch_type}): {question}"
    )


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
