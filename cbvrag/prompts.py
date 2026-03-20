from __future__ import annotations

from typing import Iterable


def answer_prompt(question: str, selected_snippets: Iterable[str], branch_summary: str, global_summary: str) -> str:
    snippets = list(selected_snippets)
    if not snippets:
        # Fallback: no retrieved evidence
        return (
            "Answer the following question as concisely as possible (1-5 words).\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    evidence_text = "\n".join([f"- {s}" for s in snippets])

    context_parts = []
    if (branch_summary or "").strip():
        context_parts.append(f"Branch summary: {branch_summary.strip()}")
    if (global_summary or "").strip():
        context_parts.append(f"Global summary: {global_summary.strip()}")
    context_block = ("\n" + "\n".join(context_parts) + "\n") if context_parts else "\n"

    # KEY FIX: No template placeholders like "[your concise answer here]".
    # The prompt ends with "Answer:" as the only generation prefix.
    # The model completes directly from here — no template to leak.
    return (
        "Answer the question using ONLY the evidence snippets provided. "
        "Be concise — 1 to 5 words only. "
        "Do not add information not present in the snippets.\n\n"
        f"Question: {question}\n"
        f"{context_block}"
        f"Evidence snippets:\n{evidence_text}\n\n"
        "Answer:"
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