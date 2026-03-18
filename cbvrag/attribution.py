"""Attribution scoring for CBV-RAG.

This module implements the **Attr score** — a training-free, model-agnostic
measure of how well a set of retrieved documents *attributes* (supports) an
answer to a query.

Formula
-------
    Attr(q, D) = alpha * GD(q, D) + (1 - alpha) * PS(q, D, Q_tilde)

where:

* **GD** (Grounded Directness) — measures how strongly the retrieved documents
  *entail* a claim derived from the query.  Uses a DeBERTa-large-MNLI NLI
  model: the documents are treated as the *premise* and the query (or a
  short claim derived from it) is the *hypothesis*.  The returned value is
  the softmax probability assigned to the *entailment* class.

* **PS** (Parametric Stability / counterfactual resistance) — measures how
  exclusively the documents support the *original* query rather than a set of
  adversarially-constructed *counterfactual* queries Q̃.  A high PS score
  means the documents are specifically grounded to the original query and do
  not equally support alternative (false) premises.

    PS(q, D, Q̃) = 1 − max_{q̃ ∈ Q̃} ent(D, q̃) / (ent(D, q) + eps)

  where ent(D, x) is the maximum entailment probability over all documents
  in D for hypothesis x.

Neither component requires access to LLM log-probabilities; both rely solely
on the NLI model's classification head.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_NLI_MODEL_ID = "microsoft/deberta-large-mnli"

# Question words that are stripped when converting a raw query into an NLI
# hypothesis (used when no explicit answer_text is available).
_QUESTION_WORD_RE = re.compile(
    r"^(who|what|where|when|why|how|is|are|was|were|did|does)\s+",
    re.IGNORECASE,
)


def _query_to_hypothesis(query: str) -> str:
    """Strip leading question words from *query* to produce a hypothesis string."""
    return _QUESTION_WORD_RE.sub("", query).strip()
_EPS = 1e-8

# Module-level singletons — loaded lazily so that importing this module does
# not immediately pull in heavy model weights.
_nli_tokenizer = None
_nli_model = None
_nli_device: Optional[str] = None


def _load_nli_model(device: str = "cpu") -> None:
    """Load the DeBERTa-large-MNLI model and tokenizer (idempotent)."""
    global _nli_tokenizer, _nli_model, _nli_device

    if _nli_model is not None and _nli_device == device:
        return  # already loaded on the correct device

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("Loading NLI model %s on device %s …", _NLI_MODEL_ID, device)
    _nli_tokenizer = AutoTokenizer.from_pretrained(_NLI_MODEL_ID)
    _nli_model = AutoModelForSequenceClassification.from_pretrained(_NLI_MODEL_ID)
    _nli_model.to(device)
    _nli_model.eval()
    _nli_device = device
    logger.info("NLI model loaded.")


def _entailment_prob(premise: str, hypothesis: str, device: str = "cpu") -> float:
    """Return the *entailment* probability for a single (premise, hypothesis) pair.

    The DeBERTa-large-MNLI label ordering is:
        0 → contradiction, 1 → neutral, 2 → entailment

    Parameters
    ----------
    premise:
        The text that acts as evidence (a retrieved document passage).
    hypothesis:
        The claim to be checked against the premise (query-derived).
    device:
        Torch device string, e.g. ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    float
        Probability in [0, 1] assigned to the *entailment* label.
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
    # Label index 2 = entailment for deberta-large-mnli
    entailment_idx = 2
    return float(probs[0, entailment_idx].item())


def _max_entailment(docs: List[str], hypothesis: str, device: str = "cpu") -> float:
    """Return the maximum entailment probability across all documents for *hypothesis*.

    Parameters
    ----------
    docs:
        List of document / passage strings used as premises.
    hypothesis:
        The claim to verify.
    device:
        Torch device string.

    Returns
    -------
    float
        max_{d in docs} P(entailment | d, hypothesis).
        Returns 0.0 if *docs* is empty.
    """
    if not docs:
        return 0.0
    return max(_entailment_prob(doc, hypothesis, device=device) for doc in docs)


def grounded_directness(
    query: str,
    docs: List[str],
    device: str = "cpu",
    answer_text: Optional[str] = None,
) -> float:
    """Compute the **GD** (Grounded Directness) component of the Attr score.

    GD measures how strongly the retrieved document set *D* entails a claim
    derived from the query *q*.

    The NLI hypothesis is chosen as follows:

    * If *answer_text* is provided (the model's predicted answer), it is used
      directly as the hypothesis — this is more precise than the raw question.
    * Otherwise, leading question words (Who/What/Where/When/Why/How/Is/Are/
      Was/Were/Did/Does) are stripped from *query* and the remainder is used.

    Implementation
    --------------
    For each document ``d`` in *D* we compute P(entailment | d, hypothesis)
    using the DeBERTa-large-MNLI model.  GD is the *maximum* over all documents:

        GD(q, D) = max_{d ∈ D} P(entailment | d, hypothesis)

    Taking the maximum (rather than the mean) reflects the RAG setting where
    a *single* highly relevant passage is sufficient to ground the answer.

    Parameters
    ----------
    query:
        The original user question (used to derive the hypothesis when
        *answer_text* is ``None``).
    docs:
        Retrieved document passages (plain text strings).
    device:
        Torch device string (default ``"cpu"``).
    answer_text:
        Optional predicted answer string.  When provided it is used directly
        as the NLI hypothesis instead of the stripped query.

    Returns
    -------
    float
        GD score in [0, 1].  Higher is better (stronger grounding).
    """
    hypothesis = answer_text if answer_text is not None else _query_to_hypothesis(query)
    return _max_entailment(docs, hypothesis=hypothesis, device=device)


def parametric_stability(
    query: str,
    docs: List[str],
    counterfactual_queries: List[str],
    device: str = "cpu",
    eps: float = _EPS,
    answer_text: Optional[str] = None,
) -> float:
    """Compute the **PS** (Parametric Stability) component of the Attr score.

    PS measures *counterfactual resistance*: how exclusively the documents
    support the *original* query rather than adversarially-constructed
    alternatives (Q̃).  High PS indicates that the document set is genuinely
    grounded in the original query and not merely a generic passage that would
    entail any plausible claim.

    Formula
    -------
        PS(q, D, Q̃) = 1 − max_{q̃ ∈ Q̃} ent(D, q̃) / (ent(D, q) + eps)

    where ``ent(D, x) = max_{d ∈ D} P(entailment | d, x)``.

    The score is clipped to [0, 1]:
    * PS → 1 : counterfactuals are not entailed — documents are exclusively
      relevant to the original query.
    * PS → 0 : the best counterfactual is entailed as strongly as the original
      query — the documents are generic or misleading.

    Parameters
    ----------
    query:
        The original user question used to compute the denominator
        ``ent(D, q)``.
    docs:
        Retrieved document passages (plain text strings).
    counterfactual_queries:
        A list of alternative / adversarial query strings (Q̃).  If empty,
        PS defaults to 1.0 (no evidence of instability).
    device:
        Torch device string (default ``"cpu"``).
    eps:
        Small constant added to the denominator to prevent division by zero
        (default ``1e-8``).
    answer_text:
        Optional predicted answer string.  When provided it is used as the
        hypothesis for ``ent(D, q)`` instead of the stripped query.

    Returns
    -------
    float
        PS score in [0, 1].  Higher is better (more counterfactual-resistant).
    """
    if not counterfactual_queries or not docs:
        return 1.0

    hypothesis = answer_text if answer_text is not None else _query_to_hypothesis(query)
    ent_original = _max_entailment(docs, hypothesis=hypothesis, device=device)
    ent_max_counter = max(
        _max_entailment(docs, hypothesis=cq, device=device)
        for cq in counterfactual_queries
    )

    ps = 1.0 - ent_max_counter / (ent_original + eps)
    # Clip to [0, 1]: PS can be negative when counterfactuals are entailed
    # *more* strongly than the original (rare but possible with generic docs).
    return float(max(0.0, min(1.0, ps)))


def compute_attr(
    query: str,
    docs: List[str],
    counterfactual_queries: List[str],
    alpha: float = 0.5,
    device: str = "cpu",
    answer_text: Optional[str] = None,
) -> dict:
    """Compute the composite **Attr** attribution score.

    Attr(q, D) = alpha * GD(q, D) + (1 - alpha) * PS(q, D, Q̃)

    This is the primary entry point for the attribution module.

    Parameters
    ----------
    query:
        The original user question.
    docs:
        Retrieved document passages to evaluate (plain text strings).
        An empty list results in Attr = 0.0.
    counterfactual_queries:
        Adversarial / counterfactual variants of the query (Q̃).  Used to
        compute the PS component.  An empty list yields PS = 1.0.
    alpha:
        Mixing coefficient in [0, 1].

        * ``alpha = 1.0`` → pure GD (only grounding quality matters).
        * ``alpha = 0.0`` → pure PS (only counterfactual resistance matters).
        * ``alpha = 0.5`` (default) → equal weight.
    device:
        Torch device string passed to the NLI model (default ``"cpu"``).
    answer_text:
        Optional predicted answer string.  When provided it is used as the
        NLI hypothesis for both the GD and PS components instead of the
        stripped query.  Pass the model's predicted answer (``pred`` field)
        for more precise attribution scoring.

    Returns
    -------
    dict with keys:
        * ``"attr"`` (float): composite score in [0, 1].
        * ``"gd"`` (float): GD component score.
        * ``"ps"`` (float): PS component score.
        * ``"alpha"`` (float): the mixing coefficient used.
        * ``"num_docs"`` (int): number of documents evaluated.
        * ``"num_counterfactuals"`` (int): number of counterfactual queries.

    Examples
    --------
    >>> result = compute_attr(
    ...     query="Who founded Apple Inc.?",
    ...     docs=["Apple Inc. was co-founded by Steve Jobs, Steve Wozniak, and Ronald Wayne."],
    ...     counterfactual_queries=["Who founded Microsoft?"],
    ...     alpha=0.5,
    ... )
    >>> result["attr"]  # doctest: +SKIP
    0.73
    """
    if not docs:
        return {
            "attr": 0.0,
            "gd": 0.0,
            "ps": 1.0,
            "alpha": alpha,
            "num_docs": 0,
            "num_counterfactuals": len(counterfactual_queries),
        }

    gd = grounded_directness(query, docs, device=device, answer_text=answer_text)
    ps = parametric_stability(query, docs, counterfactual_queries, device=device, answer_text=answer_text)
    attr = alpha * gd + (1.0 - alpha) * ps

    logger.debug(
        "compute_attr: query=%r  gd=%.4f  ps=%.4f  attr=%.4f  alpha=%.2f",
        query[:80],
        gd,
        ps,
        attr,
        alpha,
    )

    return {
        "attr": float(attr),
        "gd": float(gd),
        "ps": float(ps),
        "alpha": float(alpha),
        "num_docs": len(docs),
        "num_counterfactuals": len(counterfactual_queries),
    }
