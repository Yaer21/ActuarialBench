# -*- coding: utf-8 -*-
"""Unified benchmark scoring and summary entry point.

The score command reads an existing run directory or result file and produces a
fresh summary for the selected benchmark family. For code bench it can also
re-execute extracted R code against the configured scoring rules.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (_ROOT / path)


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _qid(question: Dict[str, Any]) -> int:
    return int(question["id"])


def _iter_json_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    else:
        yield from sorted(path.rglob("*.json"))


def score_mcq(args: argparse.Namespace) -> Dict[str, Any]:
    result_path = _repo_path(args.run_path)
    if result_path.is_dir():
        candidates = sorted(result_path.glob("results_*.json"))
        candidate = candidates[-1] if candidates else result_path / "results.json"
        if not candidate.exists() and not candidates:
            candidate = result_path / "results_temp.json"
        result_path = candidate

    data = _read_json(result_path)
    questions = data.get("questions", [])
    total = len(questions)
    failed = sum(1 for q in questions if q.get("error") or q.get("failed"))
    if any("score" in q for q in questions):
        correct = sum(1 for q in questions if float(q.get("score") or 0.0) >= 1.0)
        score_sum = sum(float(q.get("score") or 0.0) for q in questions)
    else:
        correct = sum(1 for q in questions if q.get("is_correct"))
        score_sum = float(correct)
    valid = total - failed
    summary = {
        "bench": "mcq",
        "source_result": str(result_path.relative_to(_ROOT)),
        "total_questions": total,
        "correct_count": correct,
        "failed_count": failed,
        "valid_count": valid,
        "accuracy": score_sum / valid if valid else 0.0,
    }
    return _maybe_write_summary(args, summary)


def score_case(args: argparse.Namespace) -> Dict[str, Any]:
    run_path = _repo_path(args.run_path)
    candidates = [p for p in _iter_json_files(run_path) if p.name != "score_summary.json"]
    if not candidates:
        raise FileNotFoundError(f"No JSON result files found under {run_path}")
    result_path = max(candidates, key=lambda p: p.stat().st_mtime)
    data = _read_json(result_path)

    details = data.get("questions", data.get("details", []))
    scored: List[Dict[str, Any]] = []
    total_score = 0.0
    for row in details:
        pred = "".join(sorted(set(str(row.get("extracted_answer", row.get("parsed_answer", ""))).upper())))
        gold = "".join(sorted(set(str(row.get("correct_answer", row.get("reference_answer", ""))).upper())))
        score = 1.0 if pred and pred == gold else 0.0
        total_score += score
        scored.append(
            {
                "id": row.get("id"),
                "correct_answer": gold,
                "extracted_answer": pred,
                "score": score,
            }
        )

    summary = {
        "bench": "case",
        "source_result": str(result_path.relative_to(_ROOT)),
        "total_questions": len(scored),
        "average_score": total_score / len(scored) if scored else 0.0,
        "details": scored,
    }
    return _maybe_write_summary(args, summary)


def score_spreadsheet(args: argparse.Namespace) -> Dict[str, Any]:
    run_path = _repo_path(args.run_path)
    if run_path.is_file() and run_path.name in {"LV3.1.json", "results_LV3.1.json"}:
        data = _read_json(run_path)
        questions = data.get("questions", [])
        summary = {
            "bench": "spreadsheet",
            "run_path": str(run_path.relative_to(_ROOT)),
            "total_questions": len(questions),
            "scored_questions": sum(1 for row in questions if row.get("error") is None),
            "average_score": data.get("average_score", 0.0),
            "details": questions,
        }
        return _maybe_write_summary(args, summary)

    if run_path.is_dir():
        dataset_files = sorted(
            path for path in run_path.rglob("*.json")
            if path.name in {"LV3.1.json", "results_LV3.1.json"}
        )
        if dataset_files:
            data = _read_json(dataset_files[-1])
            questions = data.get("questions", [])
            summary = {
                "bench": "spreadsheet",
                "run_path": str(run_path.relative_to(_ROOT)),
                "total_questions": len(questions),
                "scored_questions": sum(1 for row in questions if row.get("error") is None),
                "average_score": data.get("average_score", 0.0),
                "details": questions,
            }
            return _maybe_write_summary(args, summary)

        response_files = sorted(run_path.rglob("result.json"))
        if not response_files:
            response_files = sorted(run_path.rglob("response.json"))
    else:
        response_files = [run_path]
    rows: List[Dict[str, Any]] = []
    for path in response_files:
        data = _read_json(path)
        score = data.get("score", {}) or {}
        rows.append(
            {
                "group_id": data.get("group_id"),
                "id": data.get("id"),
                "status": data.get("status"),
                "score_status": score.get("status"),
                "score": score.get("score", 0.0),
                "completeness": score.get("completeness"),
                "response_file": str(path.relative_to(_ROOT)),
            }
        )

    scored = [row for row in rows if row.get("score_status") == "scored"]
    avg = sum(float(row.get("score") or 0.0) for row in scored) / len(scored) if scored else 0.0
    summary = {
        "bench": "spreadsheet",
        "run_path": str(run_path.relative_to(_ROOT)),
        "total_questions": len(rows),
        "scored_questions": len(scored),
        "average_score": avg,
        "details": rows,
    }
    return _maybe_write_summary(args, summary)


def _load_code_benchmark(path: Path) -> Dict[int, Dict[str, Any]]:
    data = _read_json(path)
    return {_qid(q): q for q in data.get("questions", [])}


def _score_existing_code_result(
    *,
    question: Dict[str, Any],
    result_path: Path,
    scorer: Any,
    rescore: bool,
) -> Dict[str, Any]:
    result = _read_json(result_path) if result_path.exists() else {}
    question_dir = result_path.parent
    extracted_code = (question_dir / "extracted_code.R").read_text(encoding="utf-8") if (question_dir / "extracted_code.R").exists() else ""

    if rescore and extracted_code:
        with tempfile.TemporaryDirectory() as temp_dir:
            exec_result = scorer.execute_code(question, extracted_code, workspace=Path(temp_dir))
            score_result = scorer.score_execution(question, exec_result)
        result.update(
            {
                "score": score_result,
                "exit_code": exec_result.get("exit_code", -1),
                "execution_time_seconds": exec_result.get("execution_time_seconds"),
                "question_score": score_result.get("avg_score", 0.0),
                "passed": score_result.get("passed", False),
                "error": None,
            }
        )
        _write_json(result_path, result)

    score = result.get("score", {}) or {}
    score_dict = score if isinstance(score, dict) else {}
    question_score = result.get("question_score")
    if question_score is None and isinstance(score, (int, float)):
        question_score = score
    if question_score is None:
        question_score = score_dict.get("avg_score", 0.0)
    return {
        "id": _qid(question),
        "mode": score_dict.get("mode", "numeric"),
        "execution_success": bool(result.get("execution_success")) if "execution_success" in result else result.get("exit_code") == 0,
        "passed": bool(result.get("passed")),
        "question_score": float(question_score or 0.0),
        "result_file": str(result_path.relative_to(_ROOT)),
    }


def score_code(args: argparse.Namespace) -> Dict[str, Any]:
    run_path = _repo_path(args.run_path)
    benchmark_path = _repo_path(args.test_data)
    benchmark = _load_code_benchmark(benchmark_path)

    rows: List[Dict[str, Any]] = []
    result_files = sorted(run_path.glob("results_LV3.2.json")) if run_path.is_dir() else []
    if run_path.is_file() and run_path.name == "results_LV3.2.json":
        result_files = [run_path]
    if result_files:
        data = _read_json(result_files[-1])
        for row in data.get("questions", []):
            rows.append(
                {
                    "id": int(row.get("id")),
                    "mode": "numeric",
                    "execution_success": bool(row.get("execution_success")),
                    "passed": bool(row.get("passed")),
                    "question_score": float(row.get("score") or 0.0),
                    "result_file": str(result_files[-1].relative_to(_ROOT)),
                }
            )
    else:
        scorer = None
        if args.rescore:
            from bench_runner.code import CodeBenchScorer

            scorer = CodeBenchScorer(benchmark_path)
        for result_path in sorted(run_path.glob("Q*/result.json")):
            qnum = int(result_path.parent.name.lstrip("Q"))
            question = benchmark.get(qnum)
            if not question:
                continue
            rows.append(
                _score_existing_code_result(
                    question=question,
                    result_path=result_path,
                    scorer=scorer,
                    rescore=args.rescore,
                )
            )

    total = len(rows)
    passed = sum(1 for row in rows if row["passed"])
    execution_success = sum(1 for row in rows if row["execution_success"])
    avg = sum(row["question_score"] for row in rows) / total if total else 0.0
    summary = {
        "bench": "code",
        "run_path": str(run_path.relative_to(_ROOT)),
        "benchmark": str(benchmark_path.relative_to(_ROOT)),
        "total_questions": total,
        "execution_success_count": execution_success,
        "passed_count": passed,
        "failed_count": total - passed,
        "average_score": avg,
        "details": rows,
    }
    return _maybe_write_summary(args, summary)


def score_error_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    from bench_runner.error_analysis import annotate_result_file

    run_path = _repo_path(args.run_path)
    result_files: List[Path] = []
    if run_path.is_file():
        result_files = [run_path]
    else:
        result_files = sorted(
            path for path in run_path.rglob("results_LV3.*.json")
            if path.name in {"results_LV3.1.json", "results_LV3.2.json"}
        )

    summaries = {}
    for result_file in result_files:
        dataset = result_file.stem.replace("results_", "")
        summary = annotate_result_file(result_file, dataset=dataset)
        summaries[str(result_file.relative_to(_ROOT))] = summary

    output = {
        "bench": "error_analysis",
        "run_path": str(run_path.relative_to(_ROOT)),
        "result_files": len(result_files),
        "summaries": summaries,
    }
    return _maybe_write_summary(args, output)


def _maybe_write_summary(args: argparse.Namespace, summary: Dict[str, Any]) -> Dict[str, Any]:
    if args.write_summary:
        output_path = _repo_path(args.output) if args.output else _repo_path(args.run_path) / "score_summary.json"
        _write_json(output_path, summary)
        print(output_path)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score or summarize an existing benchmark run.")
    parser.add_argument("--bench", choices=["mcq", "case", "spreadsheet", "code", "error_analysis"], required=True)
    parser.add_argument("--run-path", required=True, help="Run directory or result JSON file.")
    parser.add_argument("--test-data", default="data_all/LV3.2.json", help="Required for code rescore.")
    parser.add_argument("--rescore", action="store_true", help="Code bench only: re-execute extracted R code before summarizing.")
    parser.add_argument("--write-summary", action="store_true", default=True)
    parser.add_argument("--output", default="", help="Optional score summary output path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scorers = {
        "mcq": score_mcq,
        "case": score_case,
        "spreadsheet": score_spreadsheet,
        "code": score_code,
        "error_analysis": score_error_analysis,
    }
    scorers[args.bench](args)


if __name__ == "__main__":
    main()
