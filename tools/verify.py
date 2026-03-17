"""Two-tier NLI verifier for CBV-RAG.

This module provides a cost-efficient, two-tier verification pipeline that
checks whether each *claim* in a generated answer is *entailed* by the
retrieved document passages.

Tier 1 — Cheap NLI (always run)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Uses the ``microsoft/deberta-large-mnli`` sequence-classification model to
compute an entailment score for each (passage, claim) pair.  This is fast,
purely local, and requires no LLM calls.

* If the score exceeds *cheap_threshold* (default 0.7) → claim is **verified**.
* If the score is below *uncertain_low* (default 0.4) → claim is **rejected**.
* If the score falls in the *uncertain zone* [uncertain_low, cheap_threshold]
  → escalate to Tier 2.

Tier 2 — LLM verification (uncertain zone only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Calls the LLM with a structured verification prompt to adjudicate uncertain
cases.  Tier 2 is intentionally expensive and is only triggered when Tier 1
cannot confidently classify a claim.

Primary entry point
-------------------
:func:`verify_answer` — routes each claim through the appropriate tier and
returns a structured verification record with per-claim results and overall
support statistics.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_NLI_MODEL_ID = "microsoft/deberta-large-mnli"

# Module-level singletons — loaded lazily.
_nli_tokenizer = None
_nli_model = None
_nli_device: Optional[str] = None


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_nli_model(device: str = "cpu") -> None:
    """Load the DeBERTa-large-MNLI model (idempotent)."""
    global _nli_tokenizer, _nli_model, _nli_device

    if _nli_model is not None and _nli_device == device:
        return

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading NLI model %s on %s …", _NLI_MODEL_ID, device)
    _nli_tokenizer = AutoTokenizer.from_pretrained(_NLI_MODEL_ID)
    _nli_model = AutoModelForSequenceClassification.from_pretrained(_NLI_MODEL_ID)
    _nli_model.to(device)
    _nli_model.eval()
    _nli_device = device
    logger.info("NLI model loaded.")


def _entailment_prob(premise: str, hypothesis: str, device: str = "cpu") -> float:
    """Return the entailment probability P(entailment | premise, hypothesis).

    DeBERTa-large-MNLI label ordering: 0=contradiction, 1=neutral, 2=entailment.

    Parameters
    ----------
    premise:
        Retrieved document passage acting as evidence.
    hypothesis:
        Claim from the answer to be verified.
    device:
        Torch device string.

    Returns
    -------
    float
        Entailment probability in [0, 1].
    """
    _load_nli_model(device)
    inputs = _nli_tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(_nli_device)

    with torch.no_grad():
        logits = _nli_model(**inputs).logits  # shape: (1, 3)

    probs = F.softmax(logits, dim=-1)
    return float(probs[0, 2].item())  # index 2 = entailment


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

def _split_into_claims(answer: str) -> List[str]:
    """Split an answer string into individual *claims* for verification.

    Strategy:
    1. Split on sentence-ending punctuation (``. ! ?``).
    2. Strip whitespace and filter empty strings.
    3. Limit to a maximum of 10 claims to keep runtime bounded.

    Parameters
    ----------
    answer:
        The full answer text to decompose.

    Returns
    -------
    List[str]
        List of non-empty claim strings.
    """
    # Split on sentence boundaries, keeping short claims.
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    claims = [s.strip() for s in sentences if s.strip()]
    return claims[:10]  # cap at 10 to avoid runaway NLI calls


# ---------------------------------------------------------------------------
# Tier 1 — cheap NLI verification
# ---------------------------------------------------------------------------

def _tier1_verify_claim(
    claim: str,
    docs: List[str],
    device: str = "cpu",
) -> Tuple[float, str]:
    """Run Tier-1 NLI verification for a single claim.

    Computes the maximum entailment score across all passages in *docs*.

    Parameters
    ----------
    claim:
        The claim string to verify.
    docs:
        List of retrieved document passages used as premises.
    device:
        Torch device string.

    Returns
    -------
    Tuple[float, str]
        ``(score, tier)`` where *score* is in [0, 1] and *tier* is
        always ``"tier1"``.
    """
    if not docs:
        return 0.0, "tier1"
    score = max(_entailment_prob(doc, claim, device=device) for doc in docs)
    return score, "tier1"


# ---------------------------------------------------------------------------
# Tier 2 — LLM verification
# ---------------------------------------------------------------------------

def _tier2_verify_claim(
    claim: str,
    docs: List[str],
    llm: Any,
    tier1_score: float,
) -> Tuple[float, str]:
    """Run Tier-2 LLM verification for a claim in the *uncertain zone*.

    Constructs a structured prompt that asks the LLM to judge whether the
    retrieved passages support the claim.  The response is parsed for one of
    three labels: ``supported``, ``contradicted``, ``uncertain``.

    The label is then mapped to a float score:
    * ``"supported"``    → 0.85
    * ``"contradicted"`` → 0.10
    * ``"uncertain"``    → the original Tier-1 score (pass-through)

    Parameters
    ----------
    claim:
        The claim to verify.
    docs:
        Retrieved document passages.
    llm:
        An LLMEngine-compatible object with a ``.generate()`` method.
    tier1_score:
        The Tier-1 score, used as a fallback when the LLM returns
        ``"uncertain"``.

    Returns
    -------
    Tuple[float, str]
        ``(score, tier)`` where *tier* is always ``"tier2"``.
    """
    passages_text = "\n".join(f"[{i+1}] {doc[:300]}" for i, doc in enumerate(docs[:5]))
    prompt = (
        "You are a fact-verification assistant. Determine whether the CLAIM is "
        "supported by the PASSAGES below.\n\n"
        f"CLAIM: {claim}\n\n"
        f"PASSAGES:\n{passages_text}\n\n"
        "Return exactly one label: supported, contradicted, or uncertain.\n"
        "Label:"
    )
    verdict, _ = llm.generate(prompt, max_new_tokens=8, temperature=0.0, name="verify_tier2")
    verdict_lower = verdict.strip().lower()

    if "supported" in verdict_lower:
        score = 0.85
    elif "contradicted" in verdict_lower:
        score = 0.10
    else:
        score = tier1_score  # uncertain — fall back to tier1 estimate

    logger.debug("tier2 verdict=%r  score=%.3f  claim=%r", verdict_lower, score, claim[:60])
    return score, "tier2"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def verify_answer(
    answer: str,
    docs: List[str],
    llm: Optional[Any] = None,
    cheap_threshold: float = 0.7,
    uncertain_low: float = 0.4,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Verify an answer against retrieved document passages using two-tier NLI.

    Routing logic per claim
    -----------------------
    1. Run **Tier 1** (DeBERTa NLI) to get an entailment score.
    2. If score ≥ *cheap_threshold* → ``"verified"`` (Tier 1 only).
    3. If score < *uncertain_low* → ``"rejected"`` (Tier 1 only).
    4. If score ∈ [uncertain_low, cheap_threshold) → escalate to **Tier 2**
       (LLM structured verification) — only if *llm* is provided.  If no LLM
       is provided, the Tier-1 score is used and the tier is recorded as
       ``"tier1_no_llm"``.

    Parameters
    ----------
    answer:
        The generated answer text to verify.
    docs:
        Retrieved document passages used as NLI premises.
    llm:
        Optional LLMEngine-compatible object.  Required for Tier-2 escalation.
        If ``None``, uncertain-zone claims are left at their Tier-1 score.
    cheap_threshold:
        Entailment score above which a claim is considered verified without
        LLM involvement (default ``0.7``).
    uncertain_low:
        Lower bound of the uncertain zone.  Claims with score below this
        value are rejected outright (default ``0.4``).
    device:
        Torch device string used for the NLI model (default ``"cpu"``).

    Returns
    -------
    dict with keys:
        * ``"claim_results"`` (List[dict]): per-claim breakdown with keys:
            - ``"claim"`` (str)
            - ``"score"`` (float) — final entailment score in [0, 1]
            - ``"tier"`` (str) — ``"tier1"``, ``"tier2"``, or ``"tier1_no_llm"``
            - ``"verdict"`` (str) — ``"verified"``, ``"rejected"``, or
              ``"uncertain"`` (when Tier 2 is unavailable)
        * ``"overall_score"`` (float): mean score across all claims.
        * ``"num_verified"`` (int): number of verified claims.
        * ``"num_rejected"`` (int): number of rejected claims.
        * ``"num_uncertain"`` (int): number of uncertain (unresolved) claims.
        * ``"num_tier2_calls"`` (int): number of LLM Tier-2 calls made.
        * ``"cheap_threshold"`` (float)
        * ``"uncertain_low"`` (float)

    Examples
    --------
    >>> result = verify_answer(
    ...     answer="The Eiffel Tower is in Paris. It was built in 1889.",
    ...     docs=["The Eiffel Tower, located in Paris, was completed in 1889."],
    ...     cheap_threshold=0.7,
    ...     uncertain_low=0.4,
    ... )
    >>> result["num_verified"]  # doctest: +SKIP
    2
    """
    claims = _split_into_claims(answer)
    if not claims:
        return {
            "claim_results": [],
            "overall_score": 0.0,
            "num_verified": 0,
            "num_rejected": 0,
            "num_uncertain": 0,
            "num_tier2_calls": 0,
            "cheap_threshold": cheap_threshold,
            "uncertain_low": uncertain_low,
        }

    claim_results: List[Dict[str, Any]] = []
    num_tier2_calls = 0

    for claim in claims:
        # --- Tier 1 ---
        t1_score, _ = _tier1_verify_claim(claim, docs, device=device)

        if t1_score >= cheap_threshold:
            # Clearly supported — no need for LLM.
            verdict = "verified"
            final_score = t1_score
            tier = "tier1"
            logger.debug("tier1 VERIFIED  score=%.3f  claim=%r", t1_score, claim[:60])

        elif t1_score < uncertain_low:
            # Clearly not supported — skip Tier 2.
            verdict = "rejected"
            final_score = t1_score
            tier = "tier1"
            logger.debug("tier1 REJECTED  score=%.3f  claim=%r", t1_score, claim[:60])

        else:
            # Uncertain zone → escalate to Tier 2 if LLM available.
            if llm is not None:
                final_score, tier = _tier2_verify_claim(claim, docs, llm, t1_score)
                num_tier2_calls += 1
                if final_score >= cheap_threshold:
                    verdict = "verified"
                elif final_score < uncertain_low:
                    verdict = "rejected"
                else:
                    verdict = "uncertain"
            else:
                # No LLM provided — keep tier1 estimate, mark as uncertain.
                final_score = t1_score
                tier = "tier1_no_llm"
                verdict = "uncertain"
                logger.debug(
                    "tier1_no_llm UNCERTAIN  score=%.3f  claim=%r", t1_score, claim[:60]
                )

        claim_results.append(
            {
                "claim": claim,
                "score": float(final_score),
                "tier": tier,
                "verdict": verdict,
            }
        )

    scores = [r["score"] for r in claim_results]
    overall_score = sum(scores) / len(scores) if scores else 0.0

    result = {
        "claim_results": claim_results,
        "overall_score": float(overall_score),
        "num_verified": sum(1 for r in claim_results if r["verdict"] == "verified"),
        "num_rejected": sum(1 for r in claim_results if r["verdict"] == "rejected"),
        "num_uncertain": sum(1 for r in claim_results if r["verdict"] == "uncertain"),
        "num_tier2_calls": num_tier2_calls,
        "cheap_threshold": cheap_threshold,
        "uncertain_low": uncertain_low,
    }

    logger.info(
        "verify_answer: claims=%d  verified=%d  rejected=%d  uncertain=%d  "
        "tier2_calls=%d  overall_score=%.4f",
        len(claim_results),
        result["num_verified"],
        result["num_rejected"],
        result["num_uncertain"],
        num_tier2_calls,
        overall_score,
    )

    return result
