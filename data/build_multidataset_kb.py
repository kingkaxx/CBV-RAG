from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import load_dataset


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_ws(text).lower().encode("utf-8")).hexdigest()


def split_words_chunk(text: str, chunk_words: int = 140, overlap_words: int = 30) -> List[str]:
    words = normalize_ws(text).split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_words - overlap_words)
    while start < len(words):
        end = min(len(words), start + chunk_words)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


@dataclass
class QAExample:
    qid: str
    dataset: str
    split: str
    question: str
    answers: List[str]
    evidence_docs: List[Dict[str, Any]]
    support_titles: List[str]
    support_sentences: List[str]


def _hotpot(split: str, cache_dir: str, limit: Optional[int]) -> List[QAExample]:
    ds = load_dataset("hotpot_qa", "distractor", split=split, cache_dir=cache_dir)
    rows = ds.select(range(min(limit, len(ds)))) if limit else ds
    out = []
    for i, ex in enumerate(rows):
        titles = ex.get("context", {}).get("title", []) or []
        sents = ex.get("context", {}).get("sentences", []) or []
        docs = []
        for t, sent_list in zip(titles, sents):
            docs.append({"title": t, "text": " ".join(sent_list), "sentences": sent_list})
        sf = ex.get("supporting_facts", {}) or {}
        support_titles = sf.get("title", []) or []
        support_sentences = []
        for t, idx in zip(sf.get("title", []) or [], sf.get("sent_id", []) or []):
            for dt, sent_list in zip(titles, sents):
                if dt == t and 0 <= int(idx) < len(sent_list):
                    support_sentences.append(sent_list[int(idx)])
        out.append(QAExample(str(ex.get("id", i)), "hotpotqa", split, ex["question"], [ex["answer"]], docs, support_titles, support_sentences))
    return out


def _trivia(split: str, cache_dir: str, limit: Optional[int]) -> List[QAExample]:
    ds = load_dataset("trivia_qa", "unfiltered", split=split, cache_dir=cache_dir)
    rows = ds.select(range(min(limit, len(ds)))) if limit else ds
    out = []
    for i, ex in enumerate(rows):
        docs = []
        e = ex.get("entity_pages") or {}
        for title, text in zip(e.get("title", []) or [], e.get("wiki_context", []) or []):
            docs.append({"title": title, "text": text})
        sr = ex.get("search_results") or {}
        for title, text in zip(sr.get("title", []) or [], sr.get("description", []) or []):
            docs.append({"title": title, "text": text})
        answers = list(dict.fromkeys((ex.get("answer") or {}).get("aliases", []) or []))
        out.append(QAExample(str(ex.get("question_id", i)), "triviaqa", split, ex["question"], answers, docs, [], []))
    return out


def _popqa(split: str, cache_dir: str, limit: Optional[int]) -> List[QAExample]:
    ds = load_dataset("akariasai/PopQA", split=split, cache_dir=cache_dir)
    rows = ds.select(range(min(limit, len(ds)))) if limit else ds
    out = []
    for i, ex in enumerate(rows):
        title = ex.get("s_wiki_title") or ex.get("subj") or f"popqa_{i}"
        docs = [{"title": title, "text": f"{ex.get('subj','')} {ex.get('prop','')} {ex.get('obj','')}"}]
        out.append(QAExample(str(ex.get("id", i)), "popqa", split, ex["question"], [ex.get("obj", "")], docs, [], []))
    return out


def _pubhealth(split: str, cache_dir: str, limit: Optional[int]) -> List[QAExample]:
    ds = load_dataset("bigbio/pubhealth", "pubhealth_source", split=split, cache_dir=cache_dir)
    rows = ds.select(range(min(limit, len(ds)))) if limit else ds
    label_map = {0: "False", 1: "True", 2: "Mixture", 3: "Unproven"}
    out = []
    for i, ex in enumerate(rows):
        text = "\n".join([ex.get("main_text", ""), ex.get("explanation", "")]).strip()
        docs = [{"title": f"pubhealth_{i}", "text": text or ex.get("claim", "")}] 
        out.append(QAExample(str(ex.get("id", i)), "pubhealth", split, ex.get("claim", ""), [label_map.get(ex.get("label"), "Unknown")], docs, [], []))
    return out


def _musique(split: str, cache_dir: str, limit: Optional[int]) -> List[QAExample]:
    ds = load_dataset("dgslibisey/MuSiQue", split=split, cache_dir=cache_dir)
    rows = ds.select(range(min(limit, len(ds)))) if limit else ds
    out = []
    for i, ex in enumerate(rows):
        docs, support_titles = [], []
        for p in ex.get("paragraphs", []) or []:
            title = p.get("title", f"musique_{i}")
            text = p.get("paragraph_text", "")
            docs.append({"title": title, "text": text})
            if p.get("is_supporting"):
                support_titles.append(title)
        out.append(QAExample(str(ex.get("id", i)), "musique", split, ex.get("question", ""), [ex.get("answer", "")], docs, support_titles, []))
    return out


LOADERS = {
    "hotpotqa": _hotpot,
    "triviaqa": _trivia,
    "popqa": _popqa,
    "pubhealth": _pubhealth,
    "musique": _musique,
}


def iter_examples(datasets: List[str], split: str, cache_dir: str, limit_per_dataset: Optional[int]) -> Iterable[QAExample]:
    for name in datasets:
        if name not in LOADERS:
            raise ValueError(f"Unsupported dataset '{name}'")
        for ex in LOADERS[name](split, cache_dir, limit_per_dataset):
            yield ex


def build_kb_rows(examples: Iterable[QAExample], chunk_words: int, overlap_words: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    qa_rows: List[Dict[str, Any]] = []
    kb_rows: List[Dict[str, Any]] = []
    seen_doc_hashes = set()

    for ex in examples:
        qa_rows.append(
            {
                "qid": ex.qid,
                "dataset": ex.dataset,
                "split": ex.split,
                "question": ex.question,
                "answers": ex.answers,
                "support_titles": ex.support_titles,
                "support_sentences": ex.support_sentences,
            }
        )

        for d_i, doc in enumerate(ex.evidence_docs):
            title = normalize_ws(doc.get("title") or f"{ex.dataset}_{ex.qid}_{d_i}")
            text = normalize_ws(doc.get("text", ""))
            if not text:
                continue
            doc_id = text_hash(f"{title}::{text[:200]}")
            full_hash = text_hash(text)
            if full_hash in seen_doc_hashes:
                continue
            seen_doc_hashes.add(full_hash)
            for c_i, chunk in enumerate(split_words_chunk(text, chunk_words=chunk_words, overlap_words=overlap_words)):
                kb_rows.append(
                    {
                        "dataset": ex.dataset,
                        "title": title,
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_{c_i}",
                        "text": chunk,
                        "metadata": {"qid": ex.qid, "split": ex.split, "source_dataset": ex.dataset, "chunk_index": c_i},
                    }
                )
    return qa_rows, kb_rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build unified QA manifest + global KB chunk JSONL.")
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa", "triviaqa", "popqa", "pubhealth", "musique"])
    ap.add_argument("--split", default="validation")
    ap.add_argument("--cache_dir", default="./huggingface_cache")
    ap.add_argument("--limit_per_dataset", type=int, default=None)
    ap.add_argument("--chunk_words", type=int, default=140)
    ap.add_argument("--overlap_words", type=int, default=30)
    ap.add_argument("--qa_out", default="data/multidataset_qa.jsonl")
    ap.add_argument("--kb_out", default="data/global_kb_chunks.jsonl")
    args = ap.parse_args()

    examples = list(iter_examples(args.datasets, args.split, args.cache_dir, args.limit_per_dataset))
    qa_rows, kb_rows = build_kb_rows(examples, chunk_words=args.chunk_words, overlap_words=args.overlap_words)
    write_jsonl(Path(args.qa_out), qa_rows)
    write_jsonl(Path(args.kb_out), kb_rows)
    print(json.dumps({"examples": len(qa_rows), "chunks": len(kb_rows), "qa_out": args.qa_out, "kb_out": args.kb_out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
