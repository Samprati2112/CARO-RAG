import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, (list, tuple)):
        parts = [normalize_text(item) for item in value if item is not None]
        return " ".join(part for part in parts if part)
    return str(value).strip()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: str | Path, records: Iterable[Dict[str, Any]]) -> Path:
    out_path = ensure_dir(path)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            if record is None:
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL record in {p}: {line[:200]} -> {exc}") from exc
    return records


def is_nonempty_text(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(is_nonempty_text(item) for item in value)
    return True


def safe_get(mapping: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if isinstance(mapping, dict) and key in mapping:
            return mapping[key]
    return None


def describe_dataset(name: str, config: Optional[str] = None, limit: int = 3) -> Dict[str, Any]:
    ds = load_dataset(name, config) if config else load_dataset(name)
    split_names = list(ds.keys())
    sample_records: List[Dict[str, Any]] = []
    for split_name in split_names:
        split = ds[split_name]
        for idx in range(min(limit, len(split))):
            sample_records.append({"split": split_name, "record": split[idx]})

    return {
        "dataset_name": name,
        "config": config,
        "splits": split_names,
        "split_sizes": {split: len(ds[split]) for split in split_names},
        "columns": list(ds[split_names[0]].column_names),
        "sample_records": sample_records,
    }


def validate_records(records: Iterable[Dict[str, Any]], *, kind: str, expected_id_key: str, required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    required_fields = required_fields or ["source", "split"]
    seen_ids = set()
    duplicate_ids: List[str] = []
    missing_metadata = 0
    empty_records = 0
    validation_errors = 0
    total = 0

    for record in records:
        total += 1
        if not isinstance(record, dict):
            empty_records += 1
            validation_errors += 1
            continue
        if not any(is_nonempty_text(v) for v in record.values()):
            empty_records += 1
            continue

        record_id = record.get(expected_id_key)
        if expected_id_key not in record or not is_nonempty_text(record_id):
            validation_errors += 1
            continue
        record_id_str = str(record_id)
        if record_id_str in seen_ids:
            duplicate_ids.append(record_id_str)
        else:
            seen_ids.add(record_id_str)

        for field in required_fields:
            value = record.get(field)
            if field not in record or not is_nonempty_text(value):
                validation_errors += 1
                if field == "metadata":
                    missing_metadata += 1

        if "metadata" not in record or not isinstance(record.get("metadata"), dict):
            missing_metadata += 1
            validation_errors += 1

    return {
        "kind": kind,
        "total": total,
        "unique_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "empty_records": empty_records,
        "missing_metadata": missing_metadata,
        "validation_errors": validation_errors,
    }


def validate_jsonl_file(path: str | Path, *, kind: str, expected_id_key: str, required_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {
            "kind": kind,
            "total": 0,
            "unique_ids": 0,
            "duplicate_ids": [],
            "empty_records": 0,
            "missing_metadata": 0,
            "validation_errors": 0,
            "malformed_jsonl_records": 0,
            "missing_file": True,
        }

    required_fields = required_fields or ["source", "split"]
    seen_ids = set()
    duplicate_ids: List[str] = []
    missing_metadata = 0
    empty_records = 0
    validation_errors = 0
    total = 0
    malformed = 0

    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                validation_errors += 1
                continue

            total += 1
            if not isinstance(record, dict):
                empty_records += 1
                validation_errors += 1
                continue
            if not any(is_nonempty_text(value) for value in record.values()):
                empty_records += 1
                continue

            record_id = record.get(expected_id_key)
            if expected_id_key not in record or not is_nonempty_text(record_id):
                validation_errors += 1
                continue
            record_id_str = str(record_id)
            if record_id_str in seen_ids:
                duplicate_ids.append(record_id_str)
            else:
                seen_ids.add(record_id_str)

            for field in required_fields:
                value = record.get(field)
                if field not in record or not is_nonempty_text(value):
                    validation_errors += 1
                    if field == "metadata":
                        missing_metadata += 1

            if "metadata" not in record or not isinstance(record.get("metadata"), dict):
                missing_metadata += 1
                validation_errors += 1

    return {
        "kind": kind,
        "total": total,
        "unique_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "empty_records": empty_records,
        "missing_metadata": missing_metadata,
        "validation_errors": validation_errors,
        "malformed_jsonl_records": malformed,
        "missing_file": False,
    }


def get_dataset_candidates(name: str, options: Optional[List[str]] = None) -> List[str]:
    options = options or []
    candidates: List[str] = []
    for option in options:
        candidates.append(option)
    if name not in candidates:
        candidates.insert(0, name)
    return candidates
