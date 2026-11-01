import json
from pathlib import Path

from datasets import DatasetDict

from config.config import HOTPOTQA_DOCS_OUTPUT, HOTPOTQA_QUESTIONS_OUTPUT, QRECC_CONVERSATIONS_OUTPUT, QRECC_DOCS_OUTPUT, QRECC_QUESTIONS_OUTPUT
from modules.module1_data_processing.hotpotqa_processor import HotPotQAProcessor
from modules.module1_data_processing.qrecc_processor import QReCCProcessor


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_hotpotqa_loading_and_format():
    ds = HotPotQAProcessor().load_dataset()
    assert isinstance(ds, dict)
    split_name = next(iter(ds.keys()))
    assert len(ds[split_name]) > 0
    processor = HotPotQAProcessor()
    processed = processor.process_dataset(ds)
    assert len(processed["questions"]) > 0
    assert len(processed["documents"]) > 0
    for item in processed["questions"]:
        assert item["question"]
        assert item["source"] == "HotPotQA"
        assert item["metadata"]["question_type"] == "multi-hop"
    for item in processed["documents"]:
        assert item["text"]
        assert item["source"] == "HotPotQA"
        assert "metadata" in item and isinstance(item["metadata"], dict)


def test_qrecc_loading_and_conversation_order():
    ds = QReCCProcessor().load_dataset()
    assert isinstance(ds, (dict, DatasetDict))
    split_name = next(iter(ds.keys()))
    assert len(ds[split_name]) > 0
    processor = QReCCProcessor()
    processed = processor.process_dataset(ds)
    assert len(processed["questions"]) > 0
    assert len(processed["conversations"]) > 0
    for conversation in processed["conversations"]:
        turn_ids = [turn["turn_id"] for turn in conversation["turns"]]
        assert turn_ids == sorted(turn_ids)
        for turn in conversation["turns"]:
            assert turn["question"]
            assert turn["answer"] is not None


def test_jsonl_outputs_are_valid_and_unique():
    hotpot = HotPotQAProcessor().process_dataset(HotPotQAProcessor().load_dataset())
    qrecc = QReCCProcessor().process_dataset(QReCCProcessor().load_dataset())
    HotPotQAProcessor().save_outputs(hotpot)
    QReCCProcessor().save_outputs(qrecc)

    expected_counts = {
        HOTPOTQA_DOCS_OUTPUT: len(hotpot["documents"]),
        HOTPOTQA_QUESTIONS_OUTPUT: len(hotpot["questions"]),
        QRECC_DOCS_OUTPUT: len(qrecc["documents"]),
        QRECC_QUESTIONS_OUTPUT: len(qrecc["questions"]),
        QRECC_CONVERSATIONS_OUTPUT: len(qrecc["conversations"]),
    }
    expected_id_keys = {
        HOTPOTQA_DOCS_OUTPUT: "doc_id",
        HOTPOTQA_QUESTIONS_OUTPUT: "question_id",
        QRECC_DOCS_OUTPUT: "doc_id",
        QRECC_QUESTIONS_OUTPUT: "conversation_id",
        QRECC_CONVERSATIONS_OUTPUT: "conversation_id",
    }

    for path in expected_counts:
        assert path.exists(), f"Missing JSONL output: {path}"
        row_count = 0
        ids = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_count += 1
                assert expected_id_keys[path] in row
                if path == QRECC_QUESTIONS_OUTPUT:
                    ids.append((row["conversation_id"], row["turn_id"]))
                else:
                    ids.append(row[expected_id_keys[path]])
                assert row.get("source")
                assert row.get("split")
                assert row.get("metadata")

        assert row_count == expected_counts[path]
        assert row_count > 0
        assert len(ids) == len(set(ids))


def test_required_metadata_present():
    for path in [HOTPOTQA_DOCS_OUTPUT, HOTPOTQA_QUESTIONS_OUTPUT, QRECC_DOCS_OUTPUT, QRECC_QUESTIONS_OUTPUT, QRECC_CONVERSATIONS_OUTPUT]:
        if not path.exists():
            continue
        rows = read_jsonl(path)
        for row in rows:
            assert "metadata" in row and isinstance(row["metadata"], dict)
            assert row["metadata"]
