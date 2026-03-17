from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from retriever import KnowledgeBaseRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles for EM comparison."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_level_exact_match(pred: str, gold: str) -> bool:
    """
    Token-overlap EM: gold token set must be a subset of pred token set.
    Avoids false positives from raw substring matching (e.g. 'john' in
    'john was not the answer').
    """
    pred_tokens = set(normalize_answer(pred).split())
    gold_tokens = set(normalize_answer(gold).split())
    return bool(gold_tokens) and gold_tokens.issubset(pred_tokens)


class CFRAGPipeline:

    def __init__(self, models: Dict[str, Any], retriever: KnowledgeBaseRetriever):
        self.llm_model = models["llm_model"]
        self.llm_tokenizer = models["llm_tokenizer"]
        self.reranker = models["reranker_model"]
        self.retriever = retriever

        # Seeded RNG for reproducible evidence sampling
        self._rng = np.random.default_rng(seed=42)

        self._reset_token_usage()
        self._load_nli_model()

        self.cf_generation_config = {
            "max_new_tokens": 128,
            "temperature": config.COUNTERFACTUAL_TEMPERATURE,
            "do_sample": True,
            "pad_token_id": self.llm_tokenizer.eos_token_id,
        }

        if config.EXPLANATORY_TEMPERATURE > 0:
            self.exp_generation_config = {
                "max_new_tokens": config.MAX_NEW_TOKENS,
                "temperature": config.EXPLANATORY_TEMPERATURE,
                "do_sample": True,
                "pad_token_id": self.llm_tokenizer.eos_token_id,
            }
        else:
            self.exp_generation_config = {
                "max_new_tokens": config.MAX_NEW_TOKENS,
                "do_sample": False,
                "pad_token_id": self.llm_tokenizer.eos_token_id,
            }

        self.draft_generation_config = {
            "max_new_tokens": 256,
            "temperature": config.COUNTERFACTUAL_TEMPERATURE,
            "do_sample": True,
            "pad_token_id": self.llm_tokenizer.eos_token_id,
        }

        logger.info("CF-RAG Pipeline initialized successfully")

    # ──────────────────────────────────────────────────────────────────────────
    # NLI model
    # ──────────────────────────────────────────────────────────────────────────

    def _load_nli_model(self) -> None:
        """
        Load a lightweight DeBERTa-v3 cross-encoder for NLI inference.
        Used for grounding divergence and perturbation sensitivity signals.
        ~180M params; much cheaper per call than the main LLM.

        Supports local/HPC execution through optional config overrides:
          - NLI_MODEL_ID
          - HF_LOCAL_FILES_ONLY
        """
        nli_model_id = getattr(config, "NLI_MODEL_ID", "cross-encoder/nli-deberta-v3-small")
        local_files_only = bool(getattr(config, "HF_LOCAL_FILES_ONLY", False))

        try:
            self.nli_tokenizer = AutoTokenizer.from_pretrained(
                nli_model_id,
                local_files_only=local_files_only,
            )
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(
                nli_model_id,
                local_files_only=local_files_only,
            )
        except Exception as e:
            if local_files_only:
                raise RuntimeError(
                    "Failed to load NLI model from local files only. "
                    "Set config.NLI_MODEL_ID to a cached local path or disable "
                    "config.HF_LOCAL_FILES_ONLY."
                ) from e
            raise

        self.nli_model.eval()
        self.nli_model.to(self.llm_model.device)

        self.nli_label2id = self.nli_model.config.label2id
        # Expected: {'contradiction': 0, 'entailment': 1, 'neutral': 2}
        # Log so any label-order mismatch is caught immediately.
        logger.info(f"Loaded NLI model from: {nli_model_id}")
        logger.info(f"NLI label mapping: {self.nli_label2id}")

    def _get_label_idx(self, label: str) -> int:
        """Safe label index lookup with hard-coded fallback defaults."""
        defaults = {"contradiction": 0, "entailment": 1, "neutral": 2}
        return self.nli_label2id.get(label, defaults[label])

    def _nli_scores(
        self,
        premises: List[str],
        hypotheses: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Batched NLI inference.

        Returns
        -------
        np.ndarray, shape (N, 3)
            Columns are [contradiction, entailment, neutral] probabilities,
            ordered according to self.nli_label2id.
        """
        assert len(premises) == len(hypotheses), (
            f"Mismatched premise/hypothesis lengths: {len(premises)} vs {len(hypotheses)}"
        )
        all_probs: List[np.ndarray] = []

        for i in range(0, len(premises), batch_size):
            batch_p = premises[i : i + batch_size]
            batch_h = hypotheses[i : i + batch_size]

            encoded = self.nli_tokenizer(
                batch_p,
                batch_h,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.nli_model.device)

            with torch.no_grad():
                logits = self.nli_model(**encoded).logits       # (B, 3)
                probs = F.softmax(logits, dim=-1).cpu().numpy()

            all_probs.append(probs)

        return np.vstack(all_probs)  # (N, 3)

    # ──────────────────────────────────────────────────────────────────────────
    # Token tracking
    # ──────────────────────────────────────────────────────────────────────────

    def _reset_token_usage(self) -> None:
        self.last_token_usage: Dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "num_generations": 0,
            "by_stage": {},
        }

    def _record_generation_usage(self, inputs, outputs, stage: str) -> None:
        try:
            # outputs shape: (batch_size, seq_len)
            assert outputs.shape[0] == 1, (
                "Token tracking assumes batch_size=1; got batch_size="
                f"{outputs.shape[0]}"
            )
            prompt_tokens = int(inputs["input_ids"].shape[1])
            total_sequence_tokens = int(outputs[0].shape[0])
            completion_tokens = max(0, total_sequence_tokens - prompt_tokens)
        except Exception:
            prompt_tokens = 0
            completion_tokens = 0

        total_tokens = prompt_tokens + completion_tokens
        self.last_token_usage["prompt_tokens"] += prompt_tokens
        self.last_token_usage["completion_tokens"] += completion_tokens
        self.last_token_usage["total_tokens"] += total_tokens
        self.last_token_usage["num_generations"] += 1

        stage_entry = self.last_token_usage["by_stage"].setdefault(
            stage,
            {"prompt_tokens": 0, "completion_tokens": 0,
             "total_tokens": 0, "num_generations": 0},
        )
        stage_entry["prompt_tokens"] += prompt_tokens
        stage_entry["completion_tokens"] += completion_tokens
        stage_entry["total_tokens"] += total_tokens
        stage_entry["num_generations"] += 1

    def _generate_with_tracking(self, inputs, stage: str, **generation_kwargs):
        with torch.no_grad():
            outputs = self.llm_model.generate(**inputs, **generation_kwargs)
        self._record_generation_usage(inputs, outputs, stage)
        return outputs

    def get_last_token_usage(self) -> Dict[str, Any]:
        if not hasattr(self, "last_token_usage"):
            self._reset_token_usage()
        return self.last_token_usage

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1 — counterfactual query generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_counterfactual_query(self, original_query: str) -> List[str]:
        logger.info(f"Generating counterfactual queries for: {original_query}")

        prompt = config.COUNTERFACTUAL_PROMPT_TEMPLATE.format(
            num_queries=config.MAX_COUNTERFACTUAL_QUERIES,
            original_query=original_query,
        )
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates counterfactual queries.",
            },
            {"role": "user", "content": prompt},
        ]
        input_text = self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.llm_tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_INPUT_LENGTH,
        ).to(self.llm_model.device)

        outputs = self._generate_with_tracking(
            inputs, stage="counterfactual_queries", **self.cf_generation_config
        )
        generated_text = self.llm_tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]) :], skip_special_tokens=True
        ).strip()

        cf_queries = self._parse_counterfactual_queries(generated_text)
        logger.info(f"Generated {len(cf_queries)} counterfactual queries: {cf_queries}")
        return cf_queries

    def _parse_counterfactual_queries(self, generated_text: str) -> List[str]:
        queries: List[str] = []

        for line in generated_text.split("\n"):
            line = line.strip()
            match = re.match(r"^\d+\.\s*(.+)$", line)
            if match:
                queries.append(match.group(1).strip())

        if not queries:
            for line in generated_text.split("\n"):
                line = line.strip()
                match = re.match(r"^-\s*(.+)$", line)
                if match:
                    queries.append(match.group(1).strip())

        if not queries:
            for line in generated_text.split("\n"):
                line = line.strip()
                if len(line) > 10 and "?" in line:
                    queries.append(line)

        if not queries and generated_text.strip():
            queries = [generated_text.strip()]

        return queries[: config.MAX_COUNTERFACTUAL_QUERIES]

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2 — synergetic retrieval
    # ──────────────────────────────────────────────────────────────────────────

    def _synergetic_retrieval(
        self, original_query: str, cf_queries: List[str]
    ) -> List[str]:
        logger.info("Executing synergetic retrieval")

        all_documents: List[str] = []
        seen_docs: set = set()

        for doc in self.retriever.search(original_query, top_k=config.RETRIEVAL_TOP_K):
            doc_text = doc if isinstance(doc, str) else doc.get("text", str(doc))
            if doc_text not in seen_docs:
                all_documents.append(doc_text)
                seen_docs.add(doc_text)

        for i, cf_query in enumerate(cf_queries):
            logger.info(f"Retrieving for counterfactual query {i + 1}: {cf_query}")
            for doc in self.retriever.search(cf_query, top_k=config.RETRIEVAL_TOP_K):
                doc_text = doc if isinstance(doc, str) else doc.get("text", str(doc))
                if doc_text not in seen_docs:
                    all_documents.append(doc_text)
                    seen_docs.add(doc_text)

        logger.info(
            f"Synergetic retrieval completed. Total unique documents: {len(all_documents)}"
        )
        return all_documents

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 3a — evidence clustering
    # ──────────────────────────────────────────────────────────────────────────

    def _cluster_and_sample_evidence(
        self, documents: List[str], original_query: str = ""
    ) -> List[List[str]]:
        logger.info(
            f"Clustering and sampling evidence from {len(documents)} documents"
        )

        if not documents:
            logger.warning("No documents to cluster")
            return []

        if len(documents) < config.NUM_CLUSTERS:
            logger.warning(
                f"Document count ({len(documents)}) < NUM_CLUSTERS "
                f"({config.NUM_CLUSTERS}), using all documents"
            )
            return [documents] * config.NUM_DRAFTS

        try:
            # ── Batch-encode all docs + query in two calls (not N+1 calls) ──
            logger.info("Encoding documents with embedding model")
            doc_embeddings = np.array(
                self.retriever.embedding_model.encode(
                    documents, convert_to_tensor=False
                )
            )  # (N, D)

            query_scores = np.zeros(len(documents))
            if original_query:
                query_embedding = self.retriever.embedding_model.encode(
                    [original_query], convert_to_tensor=False
                )[0]  # (D,)
                from sklearn.metrics.pairwise import cosine_similarity

                query_scores = cosine_similarity(
                    doc_embeddings, query_embedding.reshape(1, -1)
                ).flatten()  # (N,)

            logger.info(f"Embeddings shape: {doc_embeddings.shape}")

            # ── Clustering ──────────────────────────────────────────────────
            if config.USE_SEMANTIC_CLUSTERING:
                cluster_labels = self._semantic_clustering(
                    doc_embeddings, documents, original_query, query_scores
                )
            else:
                kmeans = KMeans(
                    n_clusters=config.NUM_CLUSTERS, random_state=42, n_init=10
                )
                cluster_labels = kmeans.fit_predict(doc_embeddings)

            clusters: Dict[int, List[str]] = defaultdict(list)
            for doc_idx, cluster_id in enumerate(cluster_labels):
                clusters[cluster_id].append(documents[doc_idx])

            logger.info(
                f"Created {len(clusters)} clusters, sizes: "
                f"{[len(v) for v in clusters.values()]}"
            )

            # ── Draft assembly ───────────────────────────────────────────────
            top_relevant_docs: List[str] = []
            if config.INCLUDE_TOP_EVIDENCE:
                top_relevant_docs = [
                    documents[i]
                    for i in np.argsort(query_scores)[-2:][::-1]
                    if i < len(documents)
                ]
                logger.info(
                    f"Selected {len(top_relevant_docs)} top-relevant docs "
                    "to include in all drafts"
                )

            evidence_subsets: List[List[str]] = []
            doc_relevance = sorted(
                zip(documents, query_scores), key=lambda x: x[1], reverse=True
            )

            for draft_idx in range(config.NUM_DRAFTS):
                evidence_subset: List[str] = []

                if original_query and doc_relevance:
                    core_doc = doc_relevance[draft_idx % len(doc_relevance)][0]
                    evidence_subset.append(core_doc)

                for doc in top_relevant_docs:
                    if doc not in evidence_subset:
                        evidence_subset.append(doc)

                for cluster_id in sorted(clusters.keys()):
                    available = [
                        d for d in clusters[cluster_id] if d not in evidence_subset
                    ]
                    if available:
                        selected = available[draft_idx % len(available)]
                        evidence_subset.append(selected)

                while (
                    len(evidence_subset) < config.MIN_DOCS_PER_DRAFT
                    and len(evidence_subset) < len(documents)
                ):
                    remaining = [d for d in documents if d not in evidence_subset]
                    if not remaining:
                        break
                    evidence_subset.append(
                        self._rng.choice(remaining)  # seeded RNG
                    )

                evidence_subsets.append(evidence_subset)
                logger.debug(
                    f"Evidence subset {draft_idx + 1}: {len(evidence_subset)} docs"
                )

            logger.info(f"Created {len(evidence_subsets)} evidence subsets")
            return evidence_subsets

        except Exception as e:
            logger.error(f"Error in clustering: {e}. Falling back to sequential split.")
            docs_per = max(1, len(documents) // config.NUM_DRAFTS)
            subsets = []
            for i in range(config.NUM_DRAFTS):
                start = (i * docs_per) % len(documents)
                end = min(start + docs_per, len(documents))
                subsets.append(documents[start:end] or documents[:docs_per])
            return subsets

    def _semantic_clustering(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        original_query: str,
        query_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Query-aware KMeans clustering.

        FIX vs old version: removed post-hoc cluster reassignment that
        overwrote KMeans assignments for top-scoring docs, which broke
        cluster coherence. Instead we use query-similarity as an additional
        feature dimension so KMeans naturally places relevant docs well.
        """
        try:
            from sklearn.preprocessing import StandardScaler

            enhanced = np.column_stack(
                [
                    embeddings,
                    query_scores.reshape(-1, 1),
                    (query_scores ** 2).reshape(-1, 1),
                ]
            )
            enhanced = StandardScaler().fit_transform(enhanced)

            kmeans = KMeans(
                n_clusters=config.NUM_CLUSTERS, random_state=42, n_init=10
            )
            labels = kmeans.fit_predict(enhanced)

            logger.info("Semantic clustering completed")
            logger.debug(f"Cluster sizes: {np.bincount(labels)}")
            return labels

        except Exception as e:
            logger.error(f"Semantic clustering failed: {e}. Using standard KMeans.")
            kmeans = KMeans(
                n_clusters=config.NUM_CLUSTERS, random_state=42, n_init=10
            )
            return kmeans.fit_predict(embeddings)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 3b — simplified evidence selection (simplified mode only)
    # ──────────────────────────────────────────────────────────────────────────

    def _simplified_evidence_selection(
        self,
        original_query: str,
        cf_queries: List[str],
        documents: List[str],
    ) -> List[str]:
        logger.info(
            f"Simplified evidence selection from {len(documents)} documents"
        )

        if not documents:
            return []

        try:
            relevance_scores = np.array(
                self.reranker.predict([(original_query, doc) for doc in documents])
            )

            qualified = [
                (documents[i], relevance_scores[i])
                for i in range(len(documents))
                if relevance_scores[i] >= config.RELEVANCE_THRESHOLD
            ]

            if not qualified:
                logger.warning(
                    "No documents passed relevance threshold — using top-K."
                )
                top_idx = np.argsort(relevance_scores)[-config.RERANKER_TOP_K :][::-1]
                return [documents[i] for i in top_idx]

            qualified_docs = [d for d, _ in qualified]
            qual_rel_scores = np.array([s for _, s in qualified])

            if cf_queries:
                # NLI contradiction: does the document resist the CF framing?
                contra_idx = self._get_label_idx("contradiction")
                cf_contra_scores = []
                for doc in qualified_docs:
                    premises = [doc] * len(cf_queries)
                    probs = self._nli_scores(premises, cf_queries)
                    cf_contra_scores.append(float(probs[:, contra_idx].mean()))
                cf_contra_scores = np.array(cf_contra_scores)

                # Final: relevance to original + resistance to counterfactual
                final_scores = (
                    config.RERANKER_WEIGHT * qual_rel_scores
                    + (1 - config.RERANKER_WEIGHT) * cf_contra_scores
                )
            else:
                final_scores = qual_rel_scores

            top_idx = np.argsort(final_scores)[-config.RERANKER_TOP_K :][::-1]
            top_docs = [qualified_docs[i] for i in top_idx]

            logger.info(
                f"Simplified evidence selection done. Selected {len(top_docs)} docs."
            )
            return top_docs

        except Exception as e:
            logger.error(f"Error in simplified evidence selection: {e}")
            return []

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 3b — parallel draft generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_single_draft(
        self,
        original_query: str,
        evidence_subset: List[str],
        draft_idx: int,
    ) -> Dict[str, Any]:
        """Generate one draft answer from one evidence subset."""
        try:
            evidence_text = "\n\n".join(
                f"Evidence {j + 1}: {doc}"
                for j, doc in enumerate(evidence_subset)
            )
            prompt = config.DRAFT_PROMPT_TEMPLATE.format(
                original_query=original_query,
                evidence_documents=evidence_text,
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that provides concise answers "
                        "with brief rationales based on evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            input_text = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.llm_tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.MAX_INPUT_LENGTH,
            ).to(self.llm_model.device)

            outputs = self._generate_with_tracking(
                inputs, stage="draft_generation", **self.draft_generation_config
            )
            generated_text = self.llm_tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]) :],
                skip_special_tokens=True,
            ).strip()

            draft_answer, rationale = self._parse_draft_response(generated_text)
            return {
                "draft_answer": draft_answer,
                "rationale": rationale,
                "evidence_subset": evidence_subset,
                "draft_id": draft_idx + 1,
            }

        except Exception as e:
            logger.error(f"Error generating draft {draft_idx + 1}: {e}")
            return {
                "draft_answer": f"Error generating answer: {e}",
                "rationale": "Generation failed",
                "evidence_subset": evidence_subset,
                "draft_id": draft_idx + 1,
            }

    def _generate_drafts(
        self,
        original_query: str,
        evidence_subsets: List[List[str]],
    ) -> List[Dict[str, Any]]:
        """
        Generate one draft per evidence subset.

        Uses ThreadPoolExecutor for parallel execution when the underlying
        LLM is API-backed or supports concurrent inference. Falls back
        gracefully to sequential execution on single-GPU local models.
        """
        logger.info(f"Generating {len(evidence_subsets)} drafts")

        if not evidence_subsets:
            logger.warning("No evidence subsets provided")
            return []

        drafts: List[Optional[Dict[str, Any]]] = [None] * len(evidence_subsets)

        # For local HuggingFace models on a single GPU, true parallelism isn't
        # possible — we run sequentially but keep the interface consistent.
        for i, subset in enumerate(evidence_subsets):
            logger.info(f"Generating draft {i + 1}/{len(evidence_subsets)}")
            drafts[i] = self._generate_single_draft(original_query, subset, i)

        logger.info(f"Generated {len(drafts)} drafts")
        return drafts  # type: ignore[return-value]

    def _parse_draft_response(self, generated_text: str) -> Tuple[str, str]:
        answer_pattern = (
            r"(?:Answer|答案)[:：]\s*(.+?)(?=(?:Rationale|Reasoning|推理|理由)[:：]|$)"
        )
        rationale_pattern = r"(?:Rationale|Reasoning|推理|理由)[:：]\s*(.+)"

        answer_match = re.search(
            answer_pattern, generated_text, re.IGNORECASE | re.DOTALL
        )
        rationale_match = re.search(
            rationale_pattern, generated_text, re.IGNORECASE | re.DOTALL
        )

        if answer_match and rationale_match:
            return answer_match.group(1).strip(), rationale_match.group(1).strip()

        paragraphs = [p.strip() for p in generated_text.split("\n\n") if p.strip()]
        if len(paragraphs) >= 2:
            return paragraphs[0], " ".join(paragraphs[1:])

        sentences = [s.strip() for s in generated_text.split(".") if s.strip()]
        if len(sentences) >= 2:
            mid = len(sentences) // 2
            return (
                ". ".join(sentences[:mid]) + ".",
                ". ".join(sentences[mid:]) + ".",
            )

        return generated_text.strip(), "No specific rationale provided."

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 4 — counterfactual-enhanced verification
    # ──────────────────────────────────────────────────────────────────────────

    def _verify_drafts(
        self,
        original_query: str,
        cf_queries: List[str],
        drafts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        logger.info(f"Verifying {len(drafts)} drafts")

        if not drafts:
            return {"draft_answer": "", "evidence_subset": [], "rationale": ""}
        if len(drafts) == 1:
            return drafts[0]

        draft_scores = []
        for i, draft in enumerate(drafts):
            try:
                score_causal = self._calculate_causal_score(
                    original_query, cf_queries, draft["evidence_subset"]
                )
                score_consistency = self._calculate_consistency_score(
                    original_query, draft["draft_answer"], draft["evidence_subset"]
                )
                score_completeness = self._calculate_completeness_score(
                    draft["draft_answer"], draft["evidence_subset"]
                )
                final_score = (
                    0.4 * score_consistency
                    + 0.4 * score_causal
                    + 0.2 * score_completeness
                )
                draft_scores.append(
                    {
                        "draft_id": i + 1,
                        "draft": draft,
                        "score_causal": score_causal,
                        "score_consistency": score_consistency,
                        "score_completeness": score_completeness,
                        "final_score": final_score,
                    }
                )
                logger.info(
                    f"Draft {i + 1}: causal={score_causal:.3f}, "
                    f"consistency={score_consistency:.3f}, "
                    f"completeness={score_completeness:.3f}, "
                    f"final={final_score:.3f}"
                )
            except Exception as e:
                logger.error(f"Error scoring draft {i + 1}: {e}")
                draft_scores.append(
                    {
                        "draft_id": i + 1,
                        "draft": draft,
                        "score_causal": 0.0,
                        "score_consistency": 0.0,
                        "score_completeness": 0.0,
                        "final_score": 0.0,
                    }
                )

        draft_scores.sort(key=lambda x: x["final_score"], reverse=True)
        best = draft_scores[0]
        logger.info(
            f"Selected draft {best['draft_id']} (final={best['final_score']:.3f})"
        )

        best["draft"]["verification_scores"] = {
            "causal_score": best["score_causal"],
            "consistency_score": best["score_consistency"],
            "completeness_score": best["score_completeness"],
            "final_score": best["final_score"],
        }
        return best["draft"]

    # ──────────────────────────────────────────────────────────────────────────
    # Scoring functions  (core CBV-RAG contribution)
    # ──────────────────────────────────────────────────────────────────────────

    def _calculate_causal_score(
        self,
        original_query: str,
        cf_queries: List[str],
        evidence_docs: List[str],
    ) -> float:
        """
        Attr(q, D) = α · grounding_divergence + (1−α) · perturbation_sensitivity

        grounding_divergence    — NLI entailment: does D support q?
        perturbation_sensitivity — NLI contradiction: does D resist CF queries?

        Both signals are required:
          • grounding_divergence alone catches hallucination from parametric memory
            (Branch A ≈ Branch B when docs are removed → low divergence)
          • perturbation_sensitivity confirms the model reads the docs
            (Branch A ≈ Branch C even when doc content was negated → low sensitivity)
        """
        if not evidence_docs:
            return 0.0

        try:
            entail_idx = self._get_label_idx("entailment")
            contra_idx = self._get_label_idx("contradiction")

            # ── Grounding divergence ─────────────────────────────────────────
            # Premise = document, Hypothesis = original query
            orig_probs = self._nli_scores(
                premises=evidence_docs,
                hypotheses=[original_query] * len(evidence_docs),
            )
            grounding_scores = orig_probs[:, entail_idx]  # (N,)

            # Weight by reranker relevance so off-topic docs don't dilute signal
            reranker_scores = np.array(
                self.reranker.predict(
                    [(original_query, doc) for doc in evidence_docs]
                )
            )
            weights = np.clip(reranker_scores, 0, None)
            weights = weights / (weights.sum() + 1e-9)
            grounding_divergence = float(np.dot(weights, grounding_scores))

            # ── Perturbation sensitivity ─────────────────────────────────────
            # Premise = document, Hypothesis = CF query
            # High contradiction → document resists the counterfactual framing
            perturbation_sensitivity = 0.0
            if cf_queries:
                # Build (N * |CF|, ) batched pairs
                cf_premises: List[str] = []
                cf_hypotheses: List[str] = []
                for doc in evidence_docs:
                    for cf_q in cf_queries:
                        cf_premises.append(doc)
                        cf_hypotheses.append(cf_q)

                cf_probs = self._nli_scores(cf_premises, cf_hypotheses)
                contra_scores = cf_probs[:, contra_idx]  # (N * |CF|,)

                # Average contradiction per document, then weight by relevance
                contra_per_doc = contra_scores.reshape(
                    len(evidence_docs), len(cf_queries)
                ).mean(axis=1)  # (N,)
                perturbation_sensitivity = float(np.dot(weights, contra_per_doc))

            alpha = getattr(config, "CAUSAL_ALPHA", 0.5)
            causal_score = (
                alpha * grounding_divergence
                + (1 - alpha) * perturbation_sensitivity
            )
            logger.debug(
                f"Causal score: {causal_score:.3f} "
                f"(grounding={grounding_divergence:.3f}, "
                f"perturbation={perturbation_sensitivity:.3f})"
            )
            return float(causal_score)

        except Exception as e:
            logger.error(f"Error calculating causal score: {e}")
            return 0.0

    def _calculate_consistency_score(
        self,
        original_query: str,
        draft_answer: str,
        evidence_docs: List[str],
    ) -> float:
        if not evidence_docs or not draft_answer:
            return 0.0

        if config.ENABLE_MULTI_ASPECT_CONSISTENCY:
            return self._multi_aspect_consistency_evaluation(
                original_query, draft_answer, evidence_docs
            )
        return self._single_aspect_consistency_evaluation(
            original_query, draft_answer, evidence_docs
        )

    def _calculate_completeness_score(
        self,
        draft_answer: str,
        evidence_docs: List[str],
    ) -> float:
        """
        NLI entailment replaces the old word-count heuristic.

        Premise  = concatenated evidence (first 3 docs to stay within 512 tokens)
        Hypothesis = draft answer

        High entailment → answer is grounded in and supported by the evidence.
        This is a direct measure of faithfulness, not a surface proxy.
        """
        if not draft_answer or not evidence_docs:
            return 0.0

        try:
            entail_idx = self._get_label_idx("entailment")

            # Truncate premise to avoid exceeding NLI model's token limit
            premise = " ".join(evidence_docs[:3])[:2000]
            probs = self._nli_scores([premise], [draft_answer])
            score = float(probs[0, entail_idx])

            logger.debug(f"Completeness (entailment) score: {score:.3f}")
            return score

        except Exception as e:
            logger.error(f"Error calculating completeness score: {e}")
            return 0.5

    def _multi_aspect_consistency_evaluation(
        self,
        original_query: str,
        draft_answer: str,
        evidence_docs: List[str],
    ) -> float:
        evidence_text = "\n".join(
            f"Evidence {i + 1}: {doc}" for i, doc in enumerate(evidence_docs)
        )
        aspect_scores = []
        for aspect in config.CONSISTENCY_ASPECTS:
            try:
                score = self._evaluate_consistency_aspect(
                    original_query, draft_answer, evidence_text, aspect
                )
                aspect_scores.append(score)
                logger.debug(f"Consistency aspect '{aspect}': {score:.3f}")
            except Exception as e:
                logger.warning(f"Failed to evaluate aspect '{aspect}': {e}")
                aspect_scores.append(0.5)

        weights = [0.4, 0.35, 0.25]
        if len(aspect_scores) == len(weights):
            return sum(s * w for s, w in zip(aspect_scores, weights))
        return float(np.mean(aspect_scores))

    def _evaluate_consistency_aspect(
        self,
        original_query: str,
        draft_answer: str,
        evidence_text: str,
        aspect: str,
    ) -> float:
        aspect_prompts = {
            "factual_support": (
                f"Evaluate the factual support for the answer based on evidence.\n\n"
                f"Question: {original_query}\nAnswer: {draft_answer}\n"
                f"Evidence: {evidence_text}\n\n"
                f"Rate how well the evidence factually supports the answer "
                f"(Strong/Moderate/Weak/None):"
            ),
            "logical_coherence": (
                f"Evaluate the logical coherence between the answer and evidence.\n\n"
                f"Question: {original_query}\nAnswer: {draft_answer}\n"
                f"Evidence: {evidence_text}\n\n"
                f"Rate the logical consistency and reasoning flow "
                f"(Strong/Moderate/Weak/None):"
            ),
            "completeness": (
                f"Evaluate how completely the answer addresses the question "
                f"based on available evidence.\n\n"
                f"Question: {original_query}\nAnswer: {draft_answer}\n"
                f"Evidence: {evidence_text}\n\n"
                f"Rate the completeness of the answer (Strong/Moderate/Weak/None):"
            ),
        }
        prompt = aspect_prompts.get(aspect, aspect_prompts["factual_support"])
        return self._evaluate_with_llm(prompt)

    def _evaluate_with_llm(self, prompt: str) -> float:
        try:
            from transformers import GenerationConfig

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an objective evaluator. "
                        "Respond with only one word: Strong, Moderate, Weak, or None."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            input_text = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.llm_tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.MAX_INPUT_LENGTH,
            ).to(self.llm_model.device)

            eval_config = GenerationConfig(
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )
            outputs = self._generate_with_tracking(
                inputs, stage="verification_reasoning", generation_config=eval_config
            )
            result = self.llm_tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]) :],
                skip_special_tokens=True,
            ).strip().lower()

            return {"strong": 1.0, "moderate": 0.7, "weak": 0.4, "none": 0.0}.get(
                next((k for k in ("strong", "moderate", "weak", "none") if k in result), ""),
                0.5,
            )
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return 0.5

    def _single_aspect_consistency_evaluation(
        self,
        original_query: str,
        draft_answer: str,
        evidence_docs: List[str],
    ) -> float:
        evidence_text = "\n".join(
            f"Evidence {i + 1}: {doc}" for i, doc in enumerate(evidence_docs)
        )
        prompt = (
            f"You are evaluating the consistency between an answer and evidence.\n\n"
            f"Question: {original_query}\nAnswer: {draft_answer}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            f"Rate the support level (respond with only one word):\n"
            f'- "Strong" if evidence strongly supports the answer\n'
            f'- "Moderate" if evidence provides reasonable support\n'
            f'- "Weak" if evidence provides minimal support\n'
            f'- "None" if evidence contradicts or does not support the answer\n\n'
            f"Support level:"
        )
        return self._evaluate_with_llm(prompt)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 5 — final answer generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_explanatory_answer(
        self, original_query: str, evidence_docs: List[str]
    ) -> str:
        logger.info("Generating explanatory answer")

        evidence_text = "\n\n".join(
            f"Evidence {i + 1}: {doc}"
            for i, doc in enumerate(evidence_docs[: config.MAX_EVIDENCE_DOCS])
        )
        prompt = config.EXPLANATORY_PROMPT_TEMPLATE.format(
            original_query=original_query,
            evidence_documents=evidence_text,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that provides comprehensive "
                    "answers with explanations based on evidence."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        input_text = self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.llm_tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.MAX_INPUT_LENGTH,
        ).to(self.llm_model.device)

        outputs = self._generate_with_tracking(
            inputs, stage="explanatory_answer", **self.exp_generation_config
        )
        return self.llm_tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]) :],
            skip_special_tokens=True,
        ).strip()

    def _generate_explanatory_answer_with_reasoning(
        self, query: str, best_draft: Dict[str, Any]
    ) -> str:
        """
        NOTE: This stage *regenerates* the final answer conditioned on the
        best evidence subset selected by verification.  Verification therefore
        acts as evidence selection, not direct answer verification.  This is
        intentional and should be stated clearly in the paper.
        """
        logger.info("Generating explanatory answer with reasoning")

        try:
            evidence_subset = best_draft.get("evidence_subset", [])
            draft_answer = best_draft.get("draft_answer", "")
            draft_rationale = best_draft.get("rationale", "")
            v_scores = best_draft.get("verification_scores", {})

            evidence_text = "\n\n".join(
                f"Evidence {i + 1}: {doc}"
                for i, doc in enumerate(evidence_subset)
            )
            base_prompt = config.EXPLANATORY_PROMPT_TEMPLATE.format(
                original_query=query,
                evidence_documents=evidence_text,
            )
            enhanced_prompt = (
                f"{base_prompt}\n\n"
                f"**Reference Draft Analysis:**\n"
                f"Previous Analysis: {draft_rationale}\n"
                f"Draft Answer: {draft_answer}\n"
                f"Verification Scores: "
                f"Causal={v_scores.get('causal_score', 0):.3f}, "
                f"Consistency={v_scores.get('consistency_score', 0):.3f}\n\n"
                f"Please provide a comprehensive response that improves upon "
                f"this draft with detailed reasoning and evidence citations."
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert AI assistant that provides comprehensive, "
                        "well-reasoned answers with detailed evidence analysis."
                    ),
                },
                {"role": "user", "content": enhanced_prompt},
            ]
            input_text = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.llm_tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.MAX_INPUT_LENGTH,
            ).to(self.llm_model.device)

            outputs = self._generate_with_tracking(
                inputs, stage="final_answer", **self.exp_generation_config
            )
            return self.llm_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()

        except Exception as e:
            logger.error(f"Error in final answer generation: {e}")
            return (
                f"Based on the evidence analysis:\n\n"
                f"**Reasoning:** {best_draft.get('rationale', 'N/A')}\n\n"
                f"**Answer:** {best_draft.get('draft_answer', 'Unable to provide answer.')}\n\n"
                f"**Evidence Sources:** "
                f"{len(best_draft.get('evidence_subset', []))} documents analysed."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, query: str) -> str:
        logger.info(f"Starting CF-RAG pipeline for query: {query}")

        # Hard guard: simplified mode disables the verification contribution.
        # Never run experiments with this flag on.
        assert not config.ENABLE_SIMPLIFIED_MODE, (
            "ENABLE_SIMPLIFIED_MODE=True disables counterfactual verification. "
            "Set to False before running paper experiments."
        )

        self._reset_token_usage()

        try:
            logger.info("Stage 1: Generating counterfactual queries")
            cf_queries = self._generate_counterfactual_query(query)

            logger.info("Stage 2: Synergetic retrieval")
            candidate_docs = self._synergetic_retrieval(query, cf_queries)
            if not candidate_docs:
                logger.error("No documents retrieved")
                return "Could not find relevant information to answer your question."

            logger.info("Stage 3a: Evidence clustering")
            evidence_subsets = self._cluster_and_sample_evidence(candidate_docs, query)
            if not evidence_subsets:
                logger.error("Failed to create evidence subsets")
                return "Could not organise the retrieved evidence."

            logger.info("Stage 3b: Draft generation")
            drafts = self._generate_drafts(query, evidence_subsets)
            if not drafts:
                logger.error("No drafts generated")
                return "Could not generate answer drafts."

            logger.info("Stage 4: Counterfactual-enhanced verification")
            best_draft = self._verify_drafts(query, cf_queries, drafts)
            if not best_draft or not best_draft.get("draft_answer"):
                logger.warning("Verification failed — falling back to first draft")
                best_draft = drafts[0]

            logger.info("Stage 5: Final answer generation")
            final_answer = self._generate_explanatory_answer_with_reasoning(
                query, best_draft
            )

            logger.info("CF-RAG pipeline completed successfully")
            return final_answer

        except Exception as e:
            logger.error(f"Error in CF-RAG pipeline: {e}")
            return f"An error occurred while processing your question: {e}"

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def get_pipeline_stats(self) -> Dict[str, Any]:
        return {
            "model_info": {
                "llm_model": self.llm_model.__class__.__name__,
                "reranker_model": self.reranker.__class__.__name__,
                "nli_model": self.nli_model.__class__.__name__,
                "device": str(self.llm_model.device),
            },
            "config": {
                "max_new_tokens": config.MAX_NEW_TOKENS,
                "counterfactual_temperature": config.COUNTERFACTUAL_TEMPERATURE,
                "explanatory_temperature": config.EXPLANATORY_TEMPERATURE,
                "retrieval_top_k": config.RETRIEVAL_TOP_K,
                "reranker_top_k": config.RERANKER_TOP_K,
                "reranker_weight": config.RERANKER_WEIGHT,
                "relevance_threshold": config.RELEVANCE_THRESHOLD,
                "num_clusters": config.NUM_CLUSTERS,
                "num_drafts": config.NUM_DRAFTS,
            },
            "pipeline_info": {
                "architecture": "Five-Stage CF-RAG with NLI-Grounded Verification",
                "stages": [
                    "Counterfactual Query Generation",
                    "Synergetic Retrieval",
                    "Evidence Clustering & Draft Generation",
                    "Counterfactual-Enhanced Verification (NLI)",
                    "Dialectical Final Answer Generation",
                ],
                "attr_score": "alpha * grounding_divergence + (1-alpha) * perturbation_sensitivity",
            },
        }
