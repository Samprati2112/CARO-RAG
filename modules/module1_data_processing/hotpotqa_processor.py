import json
from typing import Any, Dict, List, Optional

from datasets import load_dataset

from config.config import HOTPOTQA_CONFIG_CANDIDATES, HOTPOTQA_DATASET_NAMES, HOTPOTQA_DOCS_OUTPUT, HOTPOTQA_QUESTIONS_OUTPUT
from modules.module1_data_processing.processor import normalize_text, validate_jsonl_file, write_jsonl


class HotPotQAProcessor:
    def __init__(self, dataset_name: str = "hotpotqa/hotpot_qa", config_name: Optional[str] = None):
        self.dataset_name = dataset_name
        self.config_name = config_name

    def load_dataset(self):
        for candidate in HOTPOTQA_DATASET_NAMES:
            for cfg in HOTPOTQA_CONFIG_CANDIDATES:
                try:
                    ds = load_dataset(candidate, cfg)
                    self.config_name = cfg
                    print(f"Loaded HotPotQA dataset {candidate} with config {cfg}")
                    return ds
                except Exception:
                    continue
        raise ValueError(f"Could not load HotPotQA dataset from {HOTPOTQA_DATASET_NAMES} using configs {HOTPOTQA_CONFIG_CANDIDATES}")

    def print_dataset_summary(self, dataset):
        print("DATASET NAME: hotpotqa/hotpot_qa")
        print("AVAILABLE SPLITS:", list(dataset.keys()))
        for split_name in dataset.keys():
            split = dataset[split_name]
            print(f"SPLIT {split_name}: records={len(split)} columns={split.column_names}")
            if len(split) > 0:
                sample = split[0]
                print("SAMPLE RECORD:")
                print(json.dumps(sample, ensure_ascii=False, indent=2)[:1500])

    def _doc_id(self, question_id: Any, index: int) -> str:
        return f"hotpotqa_{question_id}_{index}"

    def _normalize_passage(self, passage: Any) -> str:
        if isinstance(passage, dict):
            text = passage.get("text") or passage.get("passage") or passage.get("content") or ""
            title = passage.get("title") or passage.get("wiki_title") or ""
            return f"{title}\n{text}" if title and text else text or title
        if isinstance(passage, list):
            return "\n\n".join(self._normalize_passage(item) for item in passage if item is not None)
        return normalize_text(passage)

    def _build_question_record(self, record: Dict[str, Any], split: str) -> Optional[Dict[str, Any]]:
        question_id = record.get("id") or record.get("question_id") or record.get("_id")
        question = normalize_text(record.get("question"))
        answer = normalize_text(record.get("answer"))
        if not question or not question_id:
            return None
        return {
            "question_id": str(question_id),
            "question": question,
            "answer": answer,
            "source": "HotPotQA",
            "split": split,
            "metadata": {"question_type": "multi-hop"},
        }

    def _build_document_records(self, record: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
        question_id = record.get("id") or record.get("question_id") or record.get("_id") or "unknown"
        context = record.get("context") or {}
        titles = []
        paragraphs: List[List[str]] = []

        if isinstance(context, dict):
            titles = context.get("title") or []
            paragraphs = context.get("sentences") or []
        if not paragraphs and isinstance(record.get("passages"), list):
            passages = record.get("passages")
            for passage in passages:
                if isinstance(passage, dict):
                    titles.append(passage.get("title") or "")
                    text = self._normalize_passage(passage)
                    paragraphs.append([text])

        results: List[Dict[str, Any]] = []
        supporting = record.get("supporting_facts") or record.get("supportingFacts") or {}
        support_titles = []
        if isinstance(supporting, dict):
            support_titles = supporting.get("title") or []

        if not paragraphs:
            return results

        for idx, para in enumerate(paragraphs):
            if isinstance(para, list):
                text = " ".join(normalize_text(piece) for piece in para if normalize_text(piece))
            else:
                text = self._normalize_passage(para)
            if not text:
                continue
            title = titles[idx] if idx < len(titles) else ""
            is_supporting = bool(title and title in support_titles)
            results.append({
                "doc_id": self._doc_id(question_id, idx),
                "text": text,
                "title": title,
                "source": "HotPotQA",
                "question_id": str(question_id),
                "split": split,
                "metadata": {"is_supporting": is_supporting, "title": title},
            })
        return results

    def process_dataset(self, dataset):
        all_questions: List[Dict[str, Any]] = []
        all_documents: List[Dict[str, Any]] = []
        empty_records = 0
        duplicates_removed = 0
        summary = {
            "dataset": "HotPotQA",
            "input_records": 0,
            "processed_questions": 0,
            "processed_documents": 0,
            "empty_records_removed": 0,
            "duplicates_removed": 0,
            "validation_errors": 0,
            "documents_by_split": {},
            "questions_by_split": {},
            "raw_splits": list(dataset.keys()),
        }
        seen_question_ids = set()
        seen_doc_ids = set()

        for split_name in dataset.keys():
            split = dataset[split_name]
            summary["documents_by_split"][split_name] = 0
            summary["questions_by_split"][split_name] = 0
            summary["input_records"] += len(split)
            for record in split:
                if record is None:
                    empty_records += 1
                    continue
                question_record = self._build_question_record(record, split_name)
                if question_record is not None:
                    qid = question_record["question_id"]
                    if qid in seen_question_ids:
                        duplicates_removed += 1
                        continue
                    seen_question_ids.add(qid)
                    all_questions.append(question_record)
                    summary["questions_by_split"][split_name] += 1

                for doc in self._build_document_records(record, split_name):
                    if doc["doc_id"] in seen_doc_ids:
                        duplicates_removed += 1
                        continue
                    seen_doc_ids.add(doc["doc_id"])
                    all_documents.append(doc)
                    summary["documents_by_split"][split_name] += 1

        summary["processed_questions"] = len(all_questions)
        summary["processed_documents"] = len(all_documents)
        summary["empty_records_removed"] = empty_records
        summary["duplicates_removed"] = duplicates_removed
        return {"questions": all_questions, "documents": all_documents, "summary": summary}

    def save_outputs(self, processed):
        questions = processed["questions"]
        documents = processed["documents"]
        write_jsonl(HOTPOTQA_QUESTIONS_OUTPUT, questions)
        write_jsonl(HOTPOTQA_DOCS_OUTPUT, documents)

        q_validation = validate_jsonl_file(HOTPOTQA_QUESTIONS_OUTPUT, kind="HotPotQA questions", expected_id_key="question_id", required_fields=["question", "answer", "source", "split", "metadata"])
        d_validation = validate_jsonl_file(HOTPOTQA_DOCS_OUTPUT, kind="HotPotQA documents", expected_id_key="doc_id", required_fields=["text", "source", "split", "metadata"])
        processed["summary"]["validation_errors"] = q_validation["validation_errors"] + d_validation["validation_errors"]

        summary = {
            "dataset": "HotPotQA",
            "input_records": processed["summary"]["input_records"],
            "processed_questions": len(questions),
            "processed_documents": len(documents),
            "empty_records_removed": processed["summary"]["empty_records_removed"],
            "duplicates_removed": processed["summary"]["duplicates_removed"],
            "validation_errors": processed["summary"]["validation_errors"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
