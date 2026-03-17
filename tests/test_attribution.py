"""Unit tests for cbvrag/attribution.py and related components.

These tests use small toy examples designed to run **without a GPU** and
without loading real model weights.  The DeBERTa NLI model is patched with a
lightweight mock so that all tests execute quickly in CI.

Tests
-----
1. compute_attr returns a float in [0, 1].
2. A perfectly entailed doc scores higher GD than a random / unrelated doc.
3. A doc that also entails a counterfactual query scores lower PS than one
   that does not.
4. The null branch generate_null_branch produces a non-empty string.
5. The two-tier verifier routes to Tier 2 only in the uncertain zone.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock NLI model setup
#
# We patch cbvrag.attribution._entailment_prob (and tools.verify._entailment_prob)
# so that:
#   - Passage contains the word "entailed"  → entailment prob = 0.90
#   - Hypothesis contains "counterfactual"  → entailment prob = 0.80
#   - Otherwise                             → entailment prob = 0.10
#
# This gives us deterministic, interpretable behaviour for each test.
# ---------------------------------------------------------------------------

def _mock_entailment_prob_attr(premise: str, hypothesis: str, device: str = "cpu") -> float:
    """Deterministic mock for cbvrag.attribution._entailment_prob."""
    if "entailed" in premise.lower():
        return 0.90
    if "counterfactual" in hypothesis.lower():
        return 0.80
    return 0.10


def _mock_entailment_prob_verify(premise: str, hypothesis: str, device: str = "cpu") -> float:
    """Deterministic mock for tools.verify._entailment_prob."""
    premise_l = premise.lower()
    hypothesis_l = hypothesis.lower()
    if "entailed" in premise_l and "verified" in hypothesis_l:
        return 0.85  # clearly above cheap_threshold
    if "uncertain" in hypothesis_l:
        return 0.55  # sits in uncertain zone [0.4, 0.7]
    if "rejected" in hypothesis_l:
        return 0.20  # clearly below uncertain_low
    return 0.85


# ---------------------------------------------------------------------------
# Test 1: compute_attr returns a float in [0, 1]
# ---------------------------------------------------------------------------

@patch("cbvrag.attribution._entailment_prob", side_effect=_mock_entailment_prob_attr)
def test_compute_attr_range(mock_ent):
    from cbvrag.attribution import compute_attr

    result = compute_attr(
        query="Who founded Apple?",
        docs=["Apple was founded by Steve Jobs."],
        counterfactual_queries=["Who founded Microsoft?"],
        alpha=0.5,
    )

    assert isinstance(result, dict), "compute_attr should return a dict"
    attr = result["attr"]
    assert isinstance(attr, float), "attr score should be a float"
    assert 0.0 <= attr <= 1.0, f"attr score {attr} should be in [0, 1]"
    assert 0.0 <= result["gd"] <= 1.0, "GD should be in [0, 1]"
    assert 0.0 <= result["ps"] <= 1.0, "PS should be in [0, 1]"
    assert result["num_docs"] == 1
    assert result["num_counterfactuals"] == 1


# ---------------------------------------------------------------------------
# Test 2: A perfectly entailed doc scores higher GD than an unrelated doc
# ---------------------------------------------------------------------------

@patch("cbvrag.attribution._entailment_prob", side_effect=_mock_entailment_prob_attr)
def test_gd_entailed_higher_than_random(mock_ent):
    from cbvrag.attribution import grounded_directness

    # "entailed" in premise triggers high entailment (0.90)
    gd_high = grounded_directness(
        query="Who founded Apple?",
        docs=["Apple was entailed by Steve Jobs."],
    )

    # No "entailed" → low entailment (0.10)
    gd_low = grounded_directness(
        query="Who founded Apple?",
        docs=["The weather is sunny today."],
    )

    assert gd_high > gd_low, (
        f"Entailed doc GD ({gd_high:.3f}) should exceed random doc GD ({gd_low:.3f})"
    )
    assert gd_high == pytest.approx(0.90, abs=1e-6)
    assert gd_low == pytest.approx(0.10, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 3: Doc entailing counterfactual → lower PS than doc that does not
# ---------------------------------------------------------------------------

@patch("cbvrag.attribution._entailment_prob", side_effect=_mock_entailment_prob_attr)
def test_ps_lower_when_counterfactual_entailed(mock_ent):
    from cbvrag.attribution import parametric_stability

    # Hypothesis = "counterfactual" → mock returns 0.80 for counterfactual
    # The doc itself is unrelated (0.10 for original query).
    ps_unstable = parametric_stability(
        query="Who founded Apple?",
        docs=["Unrelated document text."],
        counterfactual_queries=["Who founded counterfactual corp?"],
    )

    # No "counterfactual" in query → mock returns 0.10 for everything.
    ps_stable = parametric_stability(
        query="Who founded Apple?",
        docs=["Apple was entailed by Steve Jobs."],  # original entailment = 0.90
        counterfactual_queries=["Who founded Microsoft?"],  # no 'counterfactual' → 0.10
    )

    assert ps_stable > ps_unstable, (
        f"Stable PS ({ps_stable:.3f}) should exceed unstable PS ({ps_unstable:.3f})"
    )
    # ps_unstable: 1 - 0.80/(0.10 + 1e-8) → clipped to 0.0
    assert ps_unstable == pytest.approx(0.0, abs=1e-4)
    # ps_stable: 1 - 0.10/(0.90 + 1e-8) ≈ 0.889
    assert ps_stable == pytest.approx(1.0 - 0.10 / (0.90 + 1e-8), abs=1e-4)


# ---------------------------------------------------------------------------
# Test 4: generate_null_branch produces a non-empty string
# ---------------------------------------------------------------------------

def test_null_branch_nonempty():
    from cbvrag.runner import generate_null_branch

    mock_llm = MagicMock()
    mock_llm.generate.return_value = ("Paris is the capital of France.", {"total_tokens": 20})

    result = generate_null_branch(question="What is the capital of France?", llm=mock_llm)

    assert result["branch_type"] == "null"
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0, "Null branch answer should be non-empty"
    assert result["parametric_hallucination_risk"] is False
    mock_llm.generate.assert_called_once()


# ---------------------------------------------------------------------------
# Test 5: Two-tier verifier routes to Tier 2 only in the uncertain zone
# ---------------------------------------------------------------------------

@patch("tools.verify._entailment_prob", side_effect=_mock_entailment_prob_verify)
def test_verifier_tier_routing(mock_ent):
    from tools.verify import verify_answer

    # Claims designed to hit specific zones:
    #   "verified claim"  → premise with 'entailed' + hypothesis with 'verified' → 0.85 ≥ 0.7 → tier1
    #   "uncertain claim" → 0.55 ∈ [0.4, 0.7]                                   → tier2
    #   "rejected claim"  → 0.20 < 0.4                                           → tier1 rejected
    answer = "This is a verified claim. This is an uncertain claim. This is a rejected claim."
    docs = ["This passage is entailed by evidence.", "Another unrelated passage."]

    mock_llm = MagicMock()
    # Tier-2 LLM response for the uncertain claim.
    mock_llm.generate.return_value = ("supported", {"total_tokens": 5})

    result = verify_answer(
        answer=answer,
        docs=docs,
        llm=mock_llm,
        cheap_threshold=0.7,
        uncertain_low=0.4,
    )

    # Exactly one Tier-2 call should have been made (for the uncertain claim).
    assert result["num_tier2_calls"] == 1, (
        f"Expected exactly 1 Tier-2 call, got {result['num_tier2_calls']}"
    )

    tiers_used = {r["tier"] for r in result["claim_results"]}
    assert "tier2" in tiers_used, "Tier 2 should have been triggered for uncertain claim"
    assert "tier1" in tiers_used, "Tier 1 should have handled non-uncertain claims"

    # Check overall structure.
    assert "claim_results" in result
    assert "overall_score" in result
    assert 0.0 <= result["overall_score"] <= 1.0


@patch("tools.verify._entailment_prob", side_effect=_mock_entailment_prob_verify)
def test_verifier_no_tier2_without_llm(mock_ent):
    """Without an LLM, all uncertain-zone claims stay at tier1_no_llm."""
    from tools.verify import verify_answer

    answer = "This is an uncertain claim."
    docs = ["Another unrelated passage."]

    result = verify_answer(
        answer=answer,
        docs=docs,
        llm=None,  # no LLM provided
        cheap_threshold=0.7,
        uncertain_low=0.4,
    )

    assert result["num_tier2_calls"] == 0
    for cr in result["claim_results"]:
        if cr["verdict"] == "uncertain":
            assert cr["tier"] == "tier1_no_llm", (
                f"Expected tier1_no_llm for uncertain claim without LLM, got {cr['tier']}"
            )


# ---------------------------------------------------------------------------
# Test 6: compute_attr with empty docs returns zero attr
# ---------------------------------------------------------------------------

@patch("cbvrag.attribution._entailment_prob", side_effect=_mock_entailment_prob_attr)
def test_compute_attr_empty_docs(mock_ent):
    from cbvrag.attribution import compute_attr

    result = compute_attr(
        query="Who founded Apple?",
        docs=[],
        counterfactual_queries=["Who founded Microsoft?"],
    )

    assert result["attr"] == 0.0
    assert result["gd"] == 0.0
    assert result["num_docs"] == 0


# ---------------------------------------------------------------------------
# Test 7: PS defaults to 1.0 with empty counterfactuals
# ---------------------------------------------------------------------------

@patch("cbvrag.attribution._entailment_prob", side_effect=_mock_entailment_prob_attr)
def test_ps_empty_counterfactuals(mock_ent):
    from cbvrag.attribution import parametric_stability

    ps = parametric_stability(
        query="Who founded Apple?",
        docs=["Apple was entailed by Steve Jobs."],
        counterfactual_queries=[],  # no counterfactuals
    )

    assert ps == pytest.approx(1.0, abs=1e-6), (
        "PS should be 1.0 when no counterfactual queries are provided"
    )
