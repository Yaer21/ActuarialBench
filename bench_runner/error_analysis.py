from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ERROR_TYPES = [
    "tool_call_failure",
    "function_usage_error",
    "modeling_calculation_error",
]
CORRECT_TYPE = "correct"


def _score(row: Dict[str, Any]) -> float:
    try:
        return float(row.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _is_empty(value: Any) -> bool:
    return not _text(value).strip()


def _path_exists(run_root: Optional[Path], raw_path: Any) -> bool:
    if run_root is None or not raw_path:
        return False
    path = Path(str(raw_path))
    if path.is_absolute():
        return path.exists()
    return (run_root / path).exists()


def _looks_like_syntax_failure(text: str) -> bool:
    lower = text.lower()
    markers = [
        "syntaxerror",
        "invalid syntax",
        "unexpected symbol",
        "unexpected end of input",
        "parse error",
        "error in parse",
        "unexpected token",
    ]
    return any(marker in lower for marker in markers)


def _looks_like_function_usage_failure(text: str) -> bool:
    lower = text.lower()
    markers = [
        "object ",
        "not found",
        "could not find function",
        "undefined columns selected",
        "subscript out of bounds",
        "no applicable method",
        "unused argument",
        "argument is missing",
        "non-numeric argument",
        "replacement has",
        "cannot open file",
        "no such file",
        "filenotfounderror",
        "keyerror",
        "nameerror",
        "attributeerror",
        "indexerror",
        "valueerror",
        "typeerror",
        "sheet",
        "worksheet",
        "column",
        "row",
    ]
    return any(marker in lower for marker in markers)


def classify_spreadsheet_error(row: Dict[str, Any], run_root: Optional[Path] = None) -> str:
    if _score(row) >= 1.0:
        return CORRECT_TYPE
    if _is_empty(row.get("raw_response")):
        return "tool_call_failure"

    error = _text(row.get("error"))
    if error:
        if _looks_like_syntax_failure(error):
            return "tool_call_failure"
        if _looks_like_function_usage_failure(error):
            return "function_usage_error"
        return "tool_call_failure"

    number_file_exists = _path_exists(run_root, row.get("number_file"))
    formula_file_exists = _path_exists(run_root, row.get("formula_file"))
    if run_root is not None and (number_file_exists or formula_file_exists):
        return "modeling_calculation_error"
    return "modeling_calculation_error"


def classify_code_error(row: Dict[str, Any]) -> str:
    if bool(row.get("passed")) or _score(row) >= 1.0:
        return CORRECT_TYPE
    if _is_empty(row.get("raw_response")):
        return "tool_call_failure"
    if _is_empty(row.get("extracted_code")):
        return "tool_call_failure"

    combined_error = "\n".join(
        _text(row.get(key))
        for key in ("error", "stderr", "stdout")
        if not _is_empty(row.get(key))
    )
    if not bool(row.get("execution_success")):
        if _looks_like_syntax_failure(combined_error):
            return "tool_call_failure"
        if _looks_like_function_usage_failure(combined_error):
            return "function_usage_error"
        if _is_empty(combined_error):
            return "tool_call_failure"
        return "function_usage_error"

    return "modeling_calculation_error"


def classify_error(dataset: str, row: Dict[str, Any], run_root: Optional[Path] = None) -> str:
    if dataset == "LV3.1":
        return classify_spreadsheet_error(row, run_root=run_root)
    if dataset == "LV3.2":
        return classify_code_error(row)
    raise ValueError(f"Unsupported dataset for error analysis: {dataset}")


def insert_error_type(row: Dict[str, Any], error_type: str) -> Dict[str, Any]:
    updated: Dict[str, Any] = {}
    inserted = False
    for key, value in row.items():
        if key == "error" and not inserted:
            updated["error_type"] = error_type
            inserted = True
        if key != "error_type":
            updated[key] = value
    if not inserted:
        updated["error_type"] = error_type
    return updated


def annotate_questions(dataset: str, questions: Iterable[Dict[str, Any]], run_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    annotated = []
    for row in questions:
        error_type = classify_error(dataset, row, run_root=run_root)
        annotated.append(insert_error_type(row, error_type))
    return annotated


def summarize_error_types(questions: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(questions)
    wrong_rows = [row for row in rows if row.get("error_type") != CORRECT_TYPE]
    counts = Counter(row.get("error_type", "tool_call_failure") for row in wrong_rows)
    total_errors = len(wrong_rows)
    percentages = {
        error_type: (counts.get(error_type, 0) / total_errors if total_errors else 0.0)
        for error_type in ERROR_TYPES
    }
    return {
        "total_questions": len(rows),
        "total_errors": total_errors,
        "counts": {error_type: counts.get(error_type, 0) for error_type in ERROR_TYPES},
        "percentages": percentages,
    }


def dump_results(payload: Dict[str, Any]) -> str:
    if "questions" not in payload or not isinstance(payload["questions"], list):
        return json.dumps(payload, ensure_ascii=False, indent=2)
    head = {key: value for key, value in payload.items() if key != "questions"}
    lines = ["{"]
    for key, value in head.items():
        lines.append(f'  {json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)},')
    lines.append('  "questions": [')
    for index, question in enumerate(payload["questions"]):
        suffix = "," if index < len(payload["questions"]) - 1 else ""
        lines.append(f"    {json.dumps(question, ensure_ascii=False)}{suffix}")
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines)


def annotate_result_file(path: Path, dataset: Optional[str] = None) -> Dict[str, Any]:
    resolved_dataset = dataset or path.stem.replace("results_", "")
    data = json.loads(path.read_text(encoding="utf-8"))
    run_root = path.parent
    data["questions"] = annotate_questions(resolved_dataset, data.get("questions", []), run_root=run_root)
    summary = summarize_error_types(data["questions"])
    data["error_type_summary"] = summary
    path.write_text(dump_results(data), encoding="utf-8")
    return summary
