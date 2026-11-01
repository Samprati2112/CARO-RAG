import json
from typing import Any, Dict, List, Optional

from datasets import DatasetDict, IterableDatasetDict, load_dataset

from config.config import QRECC_CONVERSATIONS_OUTPUT, QRECC_DATASET_NAMES, QRECC_DOCS_OUTPUT, QRECC_QUESTIONS_OUTPUT
from modules.module1_data_processing.processor import normalize_text, validate_jsonl_file, write_jsonl


class QReCCProcessor:
    def __init__(self, dataset_name: str = "svakulenk0/qrecc"):
        self.dataset_name = dataset_name

    def load_dataset(self):
        candidates = list(dict.fromkeys(QRECC_DATASET_NAMES))
        last_error = None
        for candidate in candidates:
            for loader in [
                lambda: load_dataset(candidate),
                lambda: load_dataset(candidate, streaming=True),
            ]:
                try:
                    ds = loader()
                    if ds is None:
                        continue
                    if isinstance(ds, IterableDatasetDict):
                        materialized = {}
                        for split_name, split_ds in ds.items():
                            materialized[split_name] = list(split_ds)
                        print(f"Loaded QReCC dataset {candidate} via streaming fallback")
                        return materialized
                    if isinstance(ds, DatasetDict) or hasattr(ds, "keys"):
                        print(f"Loaded QReCC dataset {candidate}")
                        return ds
                except Exception as exc:
                    last_error = exc
                    continue
        if last_error is not None:
            raise ValueError(f"Could not load QReCC dataset from {candidates}: {last_error}")
        raise ValueError(f"Could not load QReCC dataset from {candidates}")

    def print_dataset_summary(self, dataset):
        print("DATASET NAME: svakulenk0/qrecc")
        print("AVAILABLE SPLITS:", list(dataset.keys()))
        for split_name in dataset.keys():
            split = dataset[split_name]
            # Handle both Dataset objects and lists (from streaming materialization)
            if isinstance(split, list):
                column_names = list(split[0].keys()) if split else []
                print(f"SPLIT {split_name}: records={len(split)} columns={column_names}")
            else:
                column_names = split.column_names
                print(f"SPLIT {split_name}: records={len(split)} columns={column_names}")
            if len(split) > 0:
                sample = split[0]
                print("SAMPLE RECORD:")
                print(json.dumps(sample, ensure_ascii=False, indent=2)[:1500])

    def _conversation_id(self, record: Dict[str, Any]) -> str:
        return str(record.get("conversation_id") or record.get("conversationId") or record.get("id") or record.get("conversation") or record.get("Conversation_no") or "unknown")

    def _turn_id(self, record: Dict[str, Any], fallback: int) -> int:
        for key in ["turn_id", "turn", "turnNumber", "Turn_no"]:
            if key in record:
                try:
                    return int(record[key])
                except Exception:
                    pass
        return fallback + 1

    def _history_from_record(self, record: Dict[str, Any]) -> List[str]:
        history: List[str] = []
        for key in ["history", "conversation_history", "previous_questions", "context", "Context"]:
            value = record.get(key)
            if isinstance(value, list):
                for item in value:
                    cleaned = normalize_text(item)
                    if cleaned:
                        history.append(cleaned)
            elif isinstance(value, str):
                cleaned = normalize_text(value)
                if cleaned:
                    history.append(cleaned)
        return history

    def _build_question_record(self, record: Dict[str, Any], split: str, conversation_id: str, turn_id: int) -> Optional[Dict[str, Any]]:
        question = normalize_text(record.get("question") or record.get("current_question") or record.get("query") or record.get("utterance") or record.get("Question"))
        answer = normalize_text(record.get("answer") or record.get("response") or record.get("gold_answer") or record.get("Answer"))
        if not question:
            return None
        return {
            "conversation_id": conversation_id,
            "turn_id": turn_id,
            "question": question,
            "answer": answer,
            "conversation_history": self._history_from_record(record),
            "source": "QReCC",
            "split": split,
            "metadata": {"question_type": "conversational"},
        }

    def _build_document_records(self, record: Dict[str, Any], split: str, conversation_id: str, turn_id: int) -> List[Dict[str, Any]]:
        doc_candidates: List[Any] = []
        for key in ["documents", "passages", "retrieved_documents", "context_documents", "ctx", "Context", "Passages"]:
            value = record.get(key)
            if isinstance(value, list):
                doc_candidates.extend(value)
            elif isinstance(value, dict):
                doc_candidates.append(value)

        results: List[Dict[str, Any]] = []
        seen_doc_ids = set()

        for idx, doc in enumerate(doc_candidates):
            if isinstance(doc, dict):
                text = doc.get("text") or doc.get("passage") or doc.get("content") or doc.get("passage_text") or ""
                title = doc.get("title") or doc.get("doc_title") or ""
            elif isinstance(doc, str):
                text = doc
                title = ""
            else:
                continue
            text = normalize_text(text)
            if not text:
                continue
            doc_id = f"qrecc_{conversation_id}_{turn_id}_{idx}"
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            results.append({
                "doc_id": doc_id,
                "text": text,
                "title": title,
                "source": "QReCC",
                "conversation_id": conversation_id,
                "turn_id": turn_id,
                "split": split,
                "metadata": {"title": title, "question_type": "conversational"},
            })
        return results

    def _build_conversation_record(self, records: List[Dict[str, Any]], split: str) -> Dict[str, Any]:
        conversation_id = self._conversation_id(records[0]) if records else "unknown"
        turns = []
        history: List[str] = []
        for idx, record in enumerate(records):
            question = normalize_text(record.get("question") or record.get("current_question") or record.get("query") or record.get("utterance") or record.get("Question"))
            answer = normalize_text(record.get("answer") or record.get("response") or record.get("gold_answer") or record.get("Answer"))
            if not question:
                continue
            turn_id = self._turn_id(record, idx)
            turns.append({"turn_id": turn_id, "question": question, "answer": answer})
            history.append(question)
        return {
            "conversation_id": conversation_id,
            "source": "QReCC",
            "split": split,
            "turns": turns,
            "conversation_history": history,
            "metadata": {"question_type": "conversational"},
        }

    def process_dataset(self, dataset):
        all_questions: List[Dict[str, Any]] = []
        all_documents: List[Dict[str, Any]] = []
        all_conversations: List[Dict[str, Any]] = []
        empty_records = 0
        duplicates_removed = 0
        summary = {
            "dataset": "QReCC",
            "input_records": 0,
            "processed_questions": 0,
            "processed_documents": 0,
            "processed_conversations": 0,
            "empty_records_removed": 0,
            "duplicates_removed": 0,
            "validation_errors": 0,
            "documents_by_split": {},
            "questions_by_split": {},
            "raw_splits": list(dataset.keys()),
        }

        seen_q_keys = set()
        seen_doc_ids = set()
        seen_conversation_ids = set()

        for split_name in dataset.keys():
            split = dataset[split_name]
            summary["documents_by_split"][split_name] = 0
            summary["questions_by_split"][split_name] = 0
            summary["input_records"] += len(split)

            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for record in split:
                if record is None:
                    empty_records += 1
                    continue
                conv_id = self._conversation_id(record)
                grouped.setdefault(conv_id, []).append(record)

            for conversation_id, records in grouped.items():
                records = [
                    record
                    for _, record in sorted(
                        enumerate(records),
                        key=lambda item: self._turn_id(item[1], item[0]),
                    )
                ]
                conversation_record = self._build_conversation_record(records, split_name)
                if conversation_id not in seen_conversation_ids:
                    seen_conversation_ids.add(conversation_id)
                    all_conversations.append(conversation_record)
                    summary["processed_conversations"] += 1

                for idx, record in enumerate(records):
                    question_record = self._build_question_record(record, split_name, conversation_id, self._turn_id(record, idx))
                    if question_record is not None:
                        key = (conversation_id, question_record["turn_id"])
                        if key in seen_q_keys:
                            duplicates_removed += 1
                            continue
                        seen_q_keys.add(key)
                        all_questions.append(question_record)
                        summary["questions_by_split"][split_name] += 1

                    for doc in self._build_document_records(record, split_name, conversation_id, self._turn_id(record, idx)):
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
        return {"questions": all_questions, "documents": all_documents, "conversations": all_conversations, "summary": summary}

    def save_outputs(self, processed):
        write_jsonl(QRECC_QUESTIONS_OUTPUT, processed["questions"])
        write_jsonl(QRECC_DOCS_OUTPUT, processed["documents"])
        write_jsonl(QRECC_CONVERSATIONS_OUTPUT, processed["conversations"])

        q_validation = validate_jsonl_file(QRECC_QUESTIONS_OUTPUT, kind="QReCC questions", expected_id_key="conversation_id", required_fields=["question", "source", "split", "metadata"])
        d_validation = validate_jsonl_file(QRECC_DOCS_OUTPUT, kind="QReCC documents", expected_id_key="doc_id", required_fields=["text", "source", "split", "metadata"])
        c_validation = validate_jsonl_file(QRECC_CONVERSATIONS_OUTPUT, kind="QReCC conversations", expected_id_key="conversation_id", required_fields=["source", "split", "metadata"])
        processed["summary"]["validation_errors"] = q_validation["validation_errors"] + d_validation["validation_errors"] + c_validation["validation_errors"]

        summary = {
            "dataset": "QReCC",
            "input_records": processed["summary"]["input_records"],
            "processed_questions": len(processed["questions"]),
            "processed_documents": len(processed["documents"]),
            "conversations": len(processed["conversations"]),
            "empty_records_removed": processed["summary"]["empty_records_removed"],
            "duplicates_removed": processed["summary"]["duplicates_removed"],
            "validation_errors": processed["summary"]["validation_errors"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary
