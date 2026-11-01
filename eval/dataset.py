"""
Evaluation Dataset Loader, Filter, and Question Sampler.
Loads HotpotQA dev set, filters to questions where ALL gold supporting titles exist
in the 250k corpus, samples N questions with fixed seed (default 42), and builds
the three evaluation sets: ANSWERABLE, PARTIAL, UNANSWERABLE.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional
from datasets import load_dataset

from eval.caro_core import CorpusAssets, PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "eval" / "data"
QUESTIONS_FILE = DATA_DIR / "questions.jsonl"


def get_questions_filepath(seed: int = 42) -> Path:
    if seed == 123:
        return DATA_DIR / "questions_seed123.jsonl"
    elif seed == 42:
        return QUESTIONS_FILE
    else:
        return DATA_DIR / f"questions_seed{seed}.jsonl"


def prepare_evaluation_dataset(
    sample_size: int = 500,
    seed: int = 42,
    force_rebuild: bool = False,
    device: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Loads HotpotQA validation set, matches gold titles to the 250k corpus,
    and produces a sampled set of questions saved to eval/data/questions.jsonl (seed 42)
    or eval/data/questions_seed123.jsonl (seed 123 excluding seed 42).
    Returns (sampled_questions, stats_dict).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target_file = get_questions_filepath(seed)

    if target_file.exists() and not force_rebuild:
        print(f"[Dataset] Loading existing sampled questions from {target_file}...")
        questions = []
        with open(target_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        if len(questions) >= sample_size:
            print(f"[Dataset] Reusing {len(questions)} cached questions.")
            return questions[:sample_size], {"reused_from_cache": True, "count": len(questions), "file_path": str(target_file)}

    print("[Dataset] Loading corpus metadata to map gold titles...")
    CorpusAssets.load(device=device)
    title_to_doc_ids = CorpusAssets.title_to_doc_ids

    # Build case-insensitive lookup
    ci_title_to_doc_ids = {k.casefold(): v for k, v in title_to_doc_ids.items()}

    print("[Dataset] Downloading/loading official HotpotQA validation set via datasets...")
    try:
        raw_ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    except Exception as e:
        raise RuntimeError(f"Failed to load HotpotQA validation dataset: {e}")

    total_dev = len(raw_ds)
    matched_questions = []
    dropped_count = 0

    for item in raw_ds:
        qid = item["id"]
        q_text = item["question"]
        gold_ans = item["answer"]
        raw_titles = item["supporting_facts"]["title"]

        # Deduplicate titles while preserving order
        unique_gold_titles = list(dict.fromkeys(t.strip() for t in raw_titles if t.strip()))
        if not unique_gold_titles:
            dropped_count += 1
            continue

        # Check coverage of ALL gold titles
        all_found = True
        gold_title_to_dids = {}
        for title in unique_gold_titles:
            # Try exact match first, then case-insensitive
            dids = title_to_doc_ids.get(title) or ci_title_to_doc_ids.get(title.casefold())
            if not dids:
                all_found = False
                break
            gold_title_to_dids[title] = dids

        if not all_found:
            dropped_count += 1
            continue

        matched_questions.append({
            "question_id": qid,
            "question": q_text,
            "gold_answer": gold_ans,
            "gold_titles": unique_gold_titles,
            "gold_title_to_doc_ids": gold_title_to_dids,
            "type": item.get("type", "unknown"),
            "level": item.get("level", "unknown")
        })

    match_rate = len(matched_questions) / total_dev
    print(f"[Dataset] HotpotQA Dev total: {total_dev}")
    print(f"[Dataset] Questions with ALL gold titles in corpus: {len(matched_questions)} ({match_rate * 100:.2f}%)")
    print(f"[Dataset] Dropped questions: {dropped_count} ({(dropped_count / total_dev) * 100:.2f}%)")

    # If seed == 123, exclude seed-42 question IDs
    if seed == 123:
        seed42_ids = set()
        if QUESTIONS_FILE.exists():
            with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        seed42_ids.add(json.loads(line)["question_id"])
        # Also sample seed 42 from matched_questions to cover up to sample_size
        rng42 = random.Random(42)
        s42 = rng42.sample(matched_questions, min(sample_size, len(matched_questions)))
        seed42_ids.update(q["question_id"] for q in s42)

        candidate_questions = [q for q in matched_questions if q["question_id"] not in seed42_ids]
        print(f"[Dataset] Seed 123: Excluded {len(seed42_ids)} seed-42 questions. Eligible candidates: {len(candidate_questions)}")
    else:
        candidate_questions = matched_questions

    # Sample with fixed seed
    rng = random.Random(seed)
    n_sample = min(sample_size, len(candidate_questions))
    sampled = rng.sample(candidate_questions, n_sample)

    processed_questions = []
    for idx, item in enumerate(sampled):
        gold_titles = item["gold_titles"]
        gold_map = item["gold_title_to_doc_ids"]

        # Unanswerable excludes ALL doc_ids of ALL gold titles
        all_excluded_doc_ids = []
        for dids in gold_map.values():
            all_excluded_doc_ids.extend(dids)
        all_excluded_doc_ids = sorted(set(all_excluded_doc_ids))

        # Partial excludes ALL doc_ids of ONE gold title (selected deterministically)
        part_idx = rng.randint(0, len(gold_titles) - 1)
        part_title = gold_titles[part_idx]
        part_excluded_doc_ids = sorted(set(gold_map[part_title]))

        record = {
            "sample_index": idx,
            "question_id": item["question_id"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "gold_titles": gold_titles,
            "corpus_gold_doc_ids": gold_map,
            "partial_removed_title": part_title,
            "partial_excluded_doc_ids": part_excluded_doc_ids,
            "unanswerable_excluded_doc_ids": all_excluded_doc_ids,
            "type": item["type"],
            "level": item["level"]
        }
        processed_questions.append(record)

    # Save to target_file
    with open(target_file, "w", encoding="utf-8") as f:
        for q in processed_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    stats = {
        "total_dev_questions": total_dev,
        "matched_with_all_gold": len(matched_questions),
        "match_rate": match_rate,
        "dropped_questions": dropped_count,
        "sample_size": len(processed_questions),
        "seed": seed,
        "file_path": str(target_file)
    }
    print(f"[Dataset] Saved {len(processed_questions)} sampled questions to {target_file}")
    return processed_questions, stats


def get_evaluation_items(
    questions: List[Dict[str, Any]],
    target_sets: Tuple[str, ...] = ("ANSWERABLE", "PARTIAL", "UNANSWERABLE")
) -> List[Dict[str, Any]]:
    """
    Expands sampled questions into distinct evaluation items labeled with their set_type.
    """
    items = []
    for q in questions:
        for st in target_sets:
            if st == "ANSWERABLE":
                excluded = []
            elif st == "PARTIAL":
                excluded = q["partial_excluded_doc_ids"]
            elif st == "UNANSWERABLE":
                excluded = q["unanswerable_excluded_doc_ids"]
            else:
                raise ValueError(f"Unknown set_type: {st}")

            items.append({
                "sample_index": q["sample_index"],
                "question_id": q["question_id"],
                "set_type": st,
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "gold_titles": q["gold_titles"],
                "excluded_doc_ids": excluded,
                "partial_removed_title": q.get("partial_removed_title") if st == "PARTIAL" else None
            })
    return items
