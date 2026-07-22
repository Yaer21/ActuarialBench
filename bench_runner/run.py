# -*- coding: utf-8 -*-
"""Unified benchmark execution entry point.

This module keeps the task-specific evaluation logic in each benchmark package,
but exposes one CLI for the four benchmark families:

  mcq, case, spreadsheet, code
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from bench_runner.prompts import (  # noqa: E402
    CasePromptContext,
    CodePromptContext,
    SYSTEM_PROMPTS,
    build_case_user_prompt,
    build_code_user_prompt,
)
from bench_runner.error_analysis import classify_code_error, insert_error_type  # noqa: E402
from models.factory import create_model, load_config  # noqa: E402
from bench_runner.spreadsheet import run_spreadsheet_benchmark  # noqa: E402


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (_ROOT / path)


def _load_model(model_key: str, config_path: str):
    load_dotenv(_ROOT / ".env")
    config = load_config(str(_repo_path(config_path)))
    return create_model(model_key, config)


def _safe_name(value: str) -> str:
    import re

    safe = re.sub(r"[^\w.\-]+", "_", str(value), flags=re.ASCII)
    return safe.strip("_") or "model"


def _dataset_result_name(test_data: str | Path) -> str:
    return f"results_{Path(test_data).stem}.json"


def _make_result_run_dir(output_root: str | Path, model_name: str) -> Path:
    shared_run_dir = os.environ.get("BENCH_RUN_DIR")
    if shared_run_dir:
        run_dir = _repo_path(shared_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _repo_path(output_root) / f"{_safe_name(model_name)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _result_summary(model_name: str, dataset: str, timestamp: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    return {
        "model": model_name,
        "dataset": dataset,
        "timestamp": timestamp,
        "total_questions": total,
        "average_score": round(sum(float(row.get("score") or 0.0) for row in rows) / total, 6) if total else 0.0,
        "questions": rows,
    }


def _parse_int_list(raw: str) -> List[int]:
    if not raw:
        return []
    values: List[int] = []
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(int(chunk))
    return values


def _load_json(path: str | Path) -> Any:
    with _repo_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _dumps_results(payload: Dict[str, Any]) -> str:
    """Dump result JSON while keeping each question record on one line."""

    if "questions" not in payload or not isinstance(payload["questions"], list):
        return json.dumps(payload, ensure_ascii=False, indent=2)

    head = {key: value for key, value in payload.items() if key != "questions"}
    lines = ["{"]
    items = list(head.items())
    for key, value in items:
        lines.append(f'  {json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)},')
    lines.append('  "questions": [')
    for idx, question in enumerate(payload["questions"]):
        suffix = "," if idx < len(payload["questions"]) - 1 else ""
        lines.append(f"    {json.dumps(question, ensure_ascii=False)}{suffix}")
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines)


def _question_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        return list(data.get("questions", []))
    return [q for q in data if isinstance(q, dict) and "question" in q]


def _qid(question: Dict[str, Any]) -> int:
    return int(question.get("id", question.get("question_num")))


def _group_id(question: Dict[str, Any]) -> Optional[int]:
    for key in ("group_id", "case_id"):
        if question.get(key) not in ("", None):
            try:
                return int(question[key])
            except (TypeError, ValueError):
                return None
    return None


def _format_options(options: Any) -> str:
    if isinstance(options, dict):
        return "\n".join(f"{key}) {value}" for key, value in options.items())
    if isinstance(options, list):
        return "\n".join(f"{chr(65 + idx)}) {value}" for idx, value in enumerate(options))
    return str(options)


def _option_keys(options: Any) -> List[str]:
    if isinstance(options, dict):
        return [str(key).upper() for key in options.keys()]
    if isinstance(options, list):
        return [chr(65 + idx) for idx in range(len(options))]
    return []


def _build_mcq_prompt(question: Dict[str, Any], num_fewshot: int) -> str:
    from bench_runner.prompts import MCQ_FEWSHOT_EXAMPLES

    parts: List[str] = []
    if num_fewshot > 0:
        parts.append("Here are some example questions and their answers:")
        for idx, example in enumerate(MCQ_FEWSHOT_EXAMPLES[:num_fewshot], start=1):
            parts.append(
                f"Example {idx}:\n"
                f"{example['question']}\n\n"
                f"Options:\n{_format_options(example['options'])}\n\n"
                f"Answer: {example['answer']}"
            )
        parts.append("Now solve the actual question.")

    parts.append(
        f"{question['question']}\n\n"
        f"Options:\n{_format_options(question.get('options', {}))}\n\n"
        f"Answer with exactly one option letter."
    )
    return "\n\n".join(parts)


def _extract_single_choice(response: str, valid_options: List[str]) -> Optional[str]:
    import re

    if not response:
        return None
    valid = [key.upper() for key in valid_options]
    valid_set = set(valid)
    text = response.strip().upper()
    if len(text) <= 8:
        for char in text:
            if char in valid_set:
                return char

    answer_match = re.search(r"(?:ANSWER|FINAL)\s*:?\s*([A-Z])\b", text)
    if answer_match and answer_match.group(1) in valid_set:
        return answer_match.group(1)

    standalone = [m.group(1) for m in re.finditer(r"\b([A-Z])\b", text) if m.group(1) in valid_set]
    if standalone:
        return standalone[-1]

    for char in text:
        if char in valid_set:
            return char
    return None


def _normalize_multi_select_answer(response: str, valid_options: List[str]) -> str:
    import re

    if not response:
        return ""

    valid_set = {str(option).upper() for option in valid_options}
    if not valid_set:
        return ""

    text = str(response).upper()
    marker = re.search(r"(?:FINAL\s+ANSWER|FINAL|ANSWER)\s*:?\s*(.+)$", text, re.DOTALL)
    if marker:
        text = marker.group(1)

    compact_candidates = re.findall(r"\b[A-Z]{2,}\b", text)
    for token in reversed(compact_candidates):
        if all(char in valid_set for char in token):
            return "".join(sorted(set(token)))

    letters = [match.group(1) for match in re.finditer(r"\b([A-Z])\b", text) if match.group(1) in valid_set]
    if letters:
        return "".join(sorted(set(letters)))

    letters = [char for char in text if char in valid_set]
    return "".join(sorted(set(letters)))


def _score_multi_select_answer(prediction: str, truth: str) -> float:
    pred = "".join(sorted(set(str(prediction or "").upper())))
    gold = "".join(sorted(set(str(truth or "").upper())))
    return 1.0 if pred and pred == gold else 0.0


def run_mcq(args: argparse.Namespace) -> Dict[str, Any]:
    model = _load_model(args.model, args.config)
    questions = _question_list(_load_json(args.test_data))
    question_filter = set(_parse_int_list(args.questions))
    if question_filter:
        questions = [
            q for q in questions
            if _qid(q) in question_filter
        ]
    group_filter = set(_parse_int_list(args.group_ids))
    if group_filter:
        questions = [
            q for q in questions
            if _group_id(q) in group_filter
        ]
    if args.limit:
        questions = questions[: args.limit]

    run_dir = _make_result_run_dir(args.output_dir, args.model)
    result_path = run_dir / _dataset_result_name(args.test_data)
    timestamp = datetime.now().isoformat()
    results: Dict[str, Any] = {
        "model": args.model,
        "dataset": Path(args.test_data).stem,
        "timestamp": timestamp,
        "total_questions": len(questions),
        "average_score": 0.0,
        "questions": [],
    }

    total_score = 0.0
    for idx, question in enumerate(questions):
        prompt = _build_mcq_prompt(question, args.num_fewshot)
        started = time.perf_counter()
        details = model.generate_with_retry_details(
            prompt,
            retry_times=3,
            system_prompt=SYSTEM_PROMPTS["mcq"],
        )
        elapsed = round(time.perf_counter() - started, 2)
        raw_response = details.get("raw_response", "") or ""
        final_response = details.get("final_response", "") or raw_response
        predicted = _extract_single_choice(final_response, _option_keys(question.get("options", {})))
        correct = str(question.get("answer", "")).upper()
        score = 1.0 if predicted == correct else 0.0
        total_score += score
        results["questions"].append(
            {
                "id": _qid(question) if ("id" in question or "question_num" in question) else idx + 1,
                "source": question.get("source", ""),
                "correct_answer": correct,
                "raw_response": raw_response,
                "extracted_answer": predicted,
                "score": score,
                "input_token": details.get("input_tokens"),
                "response_time_seconds": elapsed,
            }
        )
        results["average_score"] = total_score / len(results["questions"])
        _write_text(result_path, _dumps_results(results))

    if not questions:
        _write_text(result_path, _dumps_results(results))
    print(result_path)
    return results


def run_case(args: argparse.Namespace) -> Dict[str, Any]:
    model = _load_model(args.model, args.config)
    records = _load_json(args.test_data)
    if isinstance(records, dict):
        records = records.get("groups", records.get("questions", []))

    case_dir = _repo_path(args.case_dir) if args.case_dir else None
    table_dir = _repo_path(args.table_dir) if args.table_dir else None
    items: List[Dict[str, Any]] = []

    for record in records:
        case_value = record.get("case", "")
        table_value = record.get("table", "")
        if case_dir and case_value and (case_dir / case_value).exists():
            case_material = (case_dir / case_value).read_text(encoding="utf-8")
        else:
            case_material = str(case_value or "")
        if table_dir and table_value and (table_dir / table_value).exists():
            table_material = (table_dir / table_value).read_text(encoding="utf-8")
        else:
            table_material = str(table_value or "")

        stem_keys = sorted(
            [key for key in record.keys() if key.startswith("stem") and isinstance(record.get(key), dict)],
            key=lambda value: int(value[4:]) if value[4:].isdigit() else 0,
        )
        for stem_key in stem_keys:
            stem_obj = record[stem_key]
            q_keys = sorted(
                [key for key in stem_obj.keys() if key.startswith("q") and isinstance(stem_obj.get(key), dict)],
                key=lambda value: int(value[1:]) if value[1:].isdigit() else 0,
            )
            for q_key in q_keys:
                q_obj = stem_obj[q_key]
                items.append(
                    {
                        "id": q_obj.get("id") or f"G{record.get('group_id', record.get('id', ''))}_{stem_key}_{q_key}",
                        "case_id": record.get("group_id", record.get("id", "")),
                        "source": record.get("source", ""),
                        "year": record.get("year", ""),
                        "season": record.get("season", ""),
                        "subject": record.get("subject", record.get("source", "")),
                        "stem_id": stem_key,
                        "question_id": q_key,
                        "case_material": case_material,
                        "table_material": table_material,
                        "stem_text": stem_obj.get("stem", ""),
                        "question_text": q_obj.get("question", ""),
                        "options": q_obj.get("options", {}),
                        "reference_answer": q_obj.get("answer", ""),
                    }
                )

    question_filter = set(_parse_int_list(args.questions))
    if question_filter:
        items = [
            item for item in items
            if isinstance(item.get("id"), int) and item["id"] in question_filter
        ]
    group_filter = set(_parse_int_list(args.group_ids))
    if group_filter:
        items = [
            item for item in items
            if _group_id(item) in group_filter
        ]
    if args.limit:
        items = items[: args.limit]

    run_dir = _make_result_run_dir(args.output_dir, args.model)
    result_path = run_dir / _dataset_result_name(args.test_data)
    timestamp = datetime.now().isoformat()

    details_rows: List[Dict[str, Any]] = []
    total_score = 0.0
    summary: Dict[str, Any] = {
        "model": args.model,
        "dataset": Path(args.test_data).stem,
        "timestamp": timestamp,
        "total_questions": 0,
        "average_score": 0.0,
        "questions": details_rows,
    }
    for item in items:
        prompt = build_case_user_prompt(
            CasePromptContext(
                case_id=item["case_id"],
                year=str(item["year"]),
                season=str(item["season"]),
                subject=str(item["subject"]),
                stem_id=item["stem_id"],
                question_id=item["question_id"],
                case_material=item["case_material"],
                table_material=item["table_material"],
                stem_text=item["stem_text"],
                question_text=item["question_text"],
                options=item["options"],
            )
        )
        started = time.perf_counter()
        response = model.generate_with_retry_details(prompt, retry_times=3, system_prompt=SYSTEM_PROMPTS["case"])
        elapsed = round(time.perf_counter() - started, 2)
        raw_response = response.get("raw_response", "") or ""
        final_response = response.get("final_response", "") or raw_response
        parsed = _normalize_multi_select_answer(final_response, list(item["options"].keys()))
        score = _score_multi_select_answer(parsed, item["reference_answer"])
        total_score += score
        details_rows.append(
            {
                "id": item["id"],
                "source": item["source"],
                "correct_answer": item["reference_answer"],
                "raw_response": raw_response,
                "extracted_answer": parsed,
                "score": score,
                "response_time_seconds": elapsed,
                "input_token": response.get("input_tokens"),
            }
        )
        summary = _result_summary(args.model, Path(args.test_data).stem, timestamp, details_rows)
        _write_text(result_path, _dumps_results(summary))

    if not items:
        _write_text(result_path, _dumps_results(summary))
    print(result_path)
    return summary


def run_spreadsheet(args: argparse.Namespace) -> Dict[str, Any]:
    model = _load_model(args.model, args.config)
    return run_spreadsheet_benchmark(
        model=model,
        model_key=args.model,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        limit=args.limit,
        group_ids=_parse_int_list(args.group_ids),
        question_ids=_parse_int_list(args.questions),
    )


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _load_existing_code_response(existing_run: Path, question_num: int) -> Dict[str, str]:
    from bench_runner.code import extract_student_code

    question_dir = existing_run / f"Q{question_num:03d}"
    raw_response = _read_text(question_dir / "raw_response.txt")
    extracted_code = _read_text(question_dir / "extracted_code.R")
    if not extracted_code and raw_response:
        extracted_code = extract_student_code(raw_response)
    if not raw_response:
        raw_response = extracted_code
    return {"raw_response": raw_response, "extracted_code": extracted_code}


def _collect_code_questions(
    benchmark: Dict[str, Any],
    question_numbers: List[int],
    group_ids: List[int],
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    questions = benchmark.get("questions", [])
    if question_numbers:
        wanted = set(question_numbers)
        questions = [q for q in questions if int(q["id"]) in wanted]
    if group_ids:
        wanted_groups = set(group_ids)
        questions = [q for q in questions if _group_id(q) in wanted_groups]
    if limit and limit > 0:
        return questions[:limit]
    return questions


def _code_scoring_mode(question: Dict[str, Any]) -> str:
    scoring = question.get("scoring", {})
    if isinstance(scoring, dict):
        return scoring.get("mode", "numeric")
    return "numeric"


def _code_reference_values(question: Dict[str, Any]) -> List[Any]:
    scoring = question.get("scoring", {})
    if isinstance(scoring, list):
        return scoring
    if isinstance(scoring, dict):
        reference = scoring.get("reference", [])
        return reference if isinstance(reference, list) else [reference]
    return []


def _code_line_count(code: str) -> int:
    return len([line for line in str(code or "").splitlines() if line.strip()])


def run_code(args: argparse.Namespace) -> Dict[str, Any]:
    from bench_runner.code import CodeBenchScorer, build_files_snapshot, extract_student_code, resolve_question_files

    benchmark_path = _repo_path(args.test_data).resolve()
    with benchmark_path.open("r", encoding="utf-8") as f:
        benchmark = json.load(f)

    question_numbers = _parse_int_list(args.questions)
    questions = _collect_code_questions(benchmark, question_numbers, _parse_int_list(args.group_ids), args.limit)

    reuse_run = _repo_path(args.reuse_run_dir).resolve() if args.reuse_run_dir else None
    model = None
    model_name = args.model
    if reuse_run is None:
        model = _load_model(args.model, args.config)
    else:
        model_name = reuse_run.name.split("_20")[0] if "_20" in reuse_run.name else reuse_run.name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{timestamp}" if reuse_run is None else f"{reuse_run.name}_rerun_{timestamp}"
    output_root = (_repo_path(args.output_dir) / run_name).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / _dataset_result_name(args.test_data)

    scorer = CodeBenchScorer(benchmark_path)
    system_prompt = SYSTEM_PROMPTS["code"]

    started_at = datetime.now()
    result_rows: List[Dict[str, Any]] = []

    for question in questions:
        question_id = int(question["id"])
        raw_response = ""
        extracted_code = ""
        response_time_seconds = None
        input_token = None
        files_snapshot = ""
        resolved_files_for_prompt: List[Path] = []

        try:
            resolved_files_for_prompt = resolve_question_files(question, scorer.benchmark_dir)
            files_snapshot = build_files_snapshot(resolved_files_for_prompt, scorer.runner)
            if reuse_run is not None:
                existing = _load_existing_code_response(reuse_run, question_id)
                raw_response = existing["raw_response"]
                extracted_code = existing["extracted_code"]
            else:
                ctx = CodePromptContext(
                    question=question["question"],
                    files=question.get("files", []),
                    files_snapshot=files_snapshot,
                    knowledge_tags=[],
                )
                user_prompt = build_code_user_prompt(ctx)
                response_started = time.perf_counter()
                details = model.generate_with_retry_details(
                    user_prompt,
                    retry_times=3,
                    system_prompt=system_prompt,
                )
                response_time_seconds = round(time.perf_counter() - response_started, 2)
                input_token = details.get("input_tokens")
                raw_response = details.get("raw_response", "") or ""
                final_response = details.get("final_response", "") or raw_response
                extracted_code = extract_student_code(final_response)

            with tempfile.TemporaryDirectory() as temp_dir:
                exec_result = scorer.execute_code(question, extracted_code, workspace=Path(temp_dir))
                score_result = scorer.score_execution(question, exec_result)

                stdout = exec_result.get("stdout", "")
                stderr = exec_result.get("stderr", "")

                result_entry = {
                    "id": question_id,
                    "source": question.get("source", ""),
                    "resolved_files": exec_result.get("resolved_files", []),
                    "files_snapshot": files_snapshot,
                    "raw_response": raw_response,
                    "extracted_code": extracted_code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "parsed_numbers": score_result.get("parsed_numbers", []),
                    "reference": _code_reference_values(question),
                    "execution_success": exec_result.get("exit_code", -1) == 0,
                    "passed": score_result.get("passed", False),
                    "score": score_result.get("avg_score", 0.0),
                    "input_token": input_token,
                    "response_time_seconds": response_time_seconds,
                    "execution_time_seconds": exec_result.get("execution_time_seconds"),
                    "code_length": _code_line_count(extracted_code),
                    "error": None,
                }
                result_entry = insert_error_type(result_entry, classify_code_error(result_entry))
        except Exception as exc:
            result_entry = {
                "id": question_id,
                "source": question.get("source", ""),
                "resolved_files": [str(path) for path in resolved_files_for_prompt],
                "files_snapshot": files_snapshot,
                "raw_response": raw_response,
                "extracted_code": extracted_code,
                "stdout": "",
                "stderr": "",
                "parsed_numbers": [],
                "reference": _code_reference_values(question),
                "execution_success": False,
                "passed": False,
                "score": 0.0,
                "input_token": input_token,
                "response_time_seconds": response_time_seconds,
                "execution_time_seconds": None,
                "code_length": _code_line_count(extracted_code),
                "error": str(exc),
            }
            result_entry = insert_error_type(result_entry, classify_code_error(result_entry))

        result_rows.append(result_entry)
        total_done = len(result_rows)
        partial = _result_summary(model_name, Path(args.test_data).stem, started_at.isoformat(), result_rows)
        _write_text(result_path, _dumps_results(partial))

    results = _result_summary(model_name, Path(args.test_data).stem, started_at.isoformat(), result_rows)
    _write_text(result_path, _dumps_results(results))
    print(result_path)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run actuarial benchmarks through one entry point.")
    parser.add_argument("--bench", choices=["mcq", "case", "spreadsheet", "code"], required=True)
    parser.add_argument("--model", required=True, help="Model key in config.yaml.")
    parser.add_argument("--test-data", default="", help="Benchmark JSON path.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-dir", default="", help="Output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Run at most this many selected questions.")
    parser.add_argument("--num-fewshot", type=int, default=2, help="MCQ few-shot examples; ignored by other benches.")
    parser.add_argument("--questions", default="", help="Comma-separated question ids for any bench, e.g. 5 or 253,254.")
    parser.add_argument("--group-ids", default="", help="Comma-separated group ids for benches that contain group/case ids.")
    parser.add_argument("--case-dir", default="", help="Case bench only: directory for external case text files.")
    parser.add_argument("--table-dir", default="", help="Case bench only: directory for external table text files.")
    parser.add_argument("--reuse-run-dir", default="", help="Code bench only: reuse model responses from a previous run.")
    return parser


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.test_data:
        args.test_data = {
            "mcq": "data_all/LV1.1.json",
            "case": "data_all/LV2.json",
            "spreadsheet": "data_all/LV3.1.json",
            "code": "data_all/LV3.2.json",
        }[args.bench]
    if not args.output_dir:
        args.output_dir = {
            "mcq": "results",
            "case": "results",
            "spreadsheet": "results",
            "code": "results",
        }[args.bench]
    return args


def main() -> None:
    args = apply_defaults(build_parser().parse_args())
    runners = {
        "mcq": run_mcq,
        "case": run_case,
        "spreadsheet": run_spreadsheet,
        "code": run_code,
    }
    runners[args.bench](args)


if __name__ == "__main__":
    main()
