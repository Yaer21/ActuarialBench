from __future__ import annotations

import json
import csv
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


_ROOT = Path(__file__).resolve().parents[1]
_NUMBER_RE = re.compile(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?")
DEFAULT_NUMERIC_RELATIVE_ERROR = 0.01


def extract_student_code(student_output: str) -> str:
    student_code = (student_output or "").strip()
    fenced = re.search(r"```(?:R|r)?\s*(.*?)```", student_code, re.DOTALL)
    if fenced:
        student_code = fenced.group(1).strip()
    elif student_code.startswith("```"):
        lines = student_code.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        student_code = "\n".join(lines).strip()
    return student_code


def extract_numbers_for_comparison(text: str) -> List[float]:
    values: List[float] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        line = re.sub(r"^\s*\[\d+\]\s*", "", line)
        if re.match(r"^\s*\d+\s+", line):
            tokens = line.split()
            if len(tokens) >= 3 and tokens[0].isdigit():
                line = " ".join(tokens[1:])
        for match in _NUMBER_RE.findall(line):
            values.append(float(match))
    return values


def _relative_match(reference: float, candidate: float, relative_error: float) -> bool:
    if abs(reference) < 1e-12:
        return abs(candidate - reference) <= 1e-12
    return abs(candidate - reference) / abs(reference) <= relative_error


def compare_numeric_reference(stdout: str, references: List[Any], relative_error: float) -> Dict[str, Any]:
    candidate_numbers = extract_numbers_for_comparison(stdout)
    refs = [float(x) for x in references]
    matched_refs = 0
    available = list(candidate_numbers)

    for ref in refs:
        found_at: Optional[int] = None
        for idx, cand in enumerate(available):
            if _relative_match(ref, cand, relative_error):
                found_at = idx
                break
        if found_at is not None:
            matched_refs += 1
            available.pop(found_at)

    numeric_pass = matched_refs == len(refs) if refs else False
    return {
        "numeric_score": 1.0 if numeric_pass else 0.0,
        "numeric_pass": numeric_pass,
        "matched_target_count": 1 if numeric_pass and refs else 0,
        "total_target_count": 1 if refs else 0,
        "parsed_numbers": candidate_numbers,
        "target_results": [
            {
                "reference": refs,
                "matched_refs": matched_refs,
                "total_refs": len(refs),
                "matched": numeric_pass,
            }
        ] if refs else [],
    }


def resolve_question_files(question: Dict[str, Any], benchmark_dir: Path) -> List[Path]:
    resolved: List[Path] = []
    for ref in question.get("files", []) or []:
        ref_path = Path(ref)
        base_name = ref_path.name
        search_roots = [
            _ROOT / "ground_truth" / "raw_rdata",
        ]
        for root in search_roots:
            candidates = [
                (root / ref_path).resolve(),
                (root / base_name).resolve(),
            ]
            candidate = next((path for path in candidates if path.exists()), None)
            if candidate is not None:
                resolved.append(candidate)
                break
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_text(value: Any, max_len: int = 120) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def _csv_snapshot(path: Path, max_rows: int = 12, max_cols: int = 40) -> str:
    try:
        sample = path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
        dialect = csv.Sniffer().sniff(sample) if sample.strip() else csv.get_dialect("ex" + "cel")
    except csv.Error:
        dialect = csv.get_dialect("ex" + "cel")

    rows: List[List[str]] = []
    row_count = 0
    max_column_count = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file_obj:
        reader = csv.reader(file_obj, dialect)
        for row in reader:
            row_count += 1
            max_column_count = max(max_column_count, len(row))
            if len(rows) < max_rows:
                rows.append([_display_text(cell, 80) for cell in row[:max_cols]])

    lines = [
        "FILE_SNAPSHOT_BEGIN",
        f"file_name: {path.name}",
        f"file_type: csv",
        f"file_sha256: {_file_sha256(path)}",
        f"row_count: {row_count}",
        f"max_column_count: {max_column_count}",
        f"preview_rows: {len(rows)}",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(f"R{idx}: " + " | ".join(row))
    if max_column_count > max_cols:
        lines.append(f"NOTE: preview truncated to first {max_cols} columns.")
    lines.append("FILE_SNAPSHOT_END")
    return "\n".join(lines)


def _plain_text_snapshot(path: Path, max_lines: int = 30) -> str:
    lines = [
        "FILE_SNAPSHOT_BEGIN",
        f"file_name: {path.name}",
        f"file_type: text",
        f"file_sha256: {_file_sha256(path)}",
    ]
    try:
        text_lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except UnicodeDecodeError:
        text_lines = []
    lines.append(f"line_count: {len(text_lines)}")
    for idx, line in enumerate(text_lines[:max_lines], start=1):
        lines.append(f"L{idx}: {_display_text(line, 180)}")
    if len(text_lines) > max_lines:
        lines.append(f"NOTE: preview truncated to first {max_lines} lines.")
    lines.append("FILE_SNAPSHOT_END")
    return "\n".join(lines)


def _r_string(value: str) -> str:
    return json.dumps(value)


def _rdata_snapshot(path: Path, docker_runner: "DockerRunner") -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        shutil.copy2(path, workspace / path.name)
        script = f"""
options(warn = 1)
file_name <- {_r_string(path.name)}
cat("FILE_SNAPSHOT_BEGIN\\n")
cat("file_name: ", file_name, "\\n", sep = "")
cat("file_type: RData\\n")
env <- new.env(parent = emptyenv())
loaded <- tryCatch(load(file_name, envir = env), error = function(e) e)
if (inherits(loaded, "error")) {{
  cat("load_error: ", conditionMessage(loaded), "\\n", sep = "")
}} else {{
  cat("objects: ", paste(loaded, collapse = ", "), "\\n", sep = "")
  for (obj_name in loaded) {{
    obj <- get(obj_name, envir = env)
    cat("OBJECT ", obj_name, " | class=", paste(class(obj), collapse = "/"),
        " | typeof=", typeof(obj), "\\n", sep = "")
    dims <- dim(obj)
    if (!is.null(dims)) cat("dim: ", paste(dims, collapse = " x "), "\\n", sep = "")
    nms <- names(obj)
    if (!is.null(nms)) cat("names: ", paste(utils::head(nms, 80), collapse = ", "), "\\n", sep = "")
    if (is.data.frame(obj) || is.matrix(obj)) {{
      cat("head:\\n")
      print(utils::head(obj, 8))
    }} else if (is.atomic(obj)) {{
      cat("values_head:\\n")
      print(utils::head(obj, 30))
      cat("summary:\\n")
      print(summary(obj))
    }} else {{
      cat("structure:\\n")
      utils::str(obj, max.level = 2, vec.len = 8)
    }}
  }}
}}
cat("FILE_SNAPSHOT_END\\n")
"""
        result = docker_runner.run_r_code(script, workspace)
        if result.get("exit_code") == 0:
            return result.get("stdout", "").strip()
        return "\n".join(
            [
                "FILE_SNAPSHOT_BEGIN",
                f"file_name: {path.name}",
                "file_type: RData",
                f"file_sha256: {_file_sha256(path)}",
                f"snapshot_error: {_display_text(result.get('stderr', ''), 300)}",
                "FILE_SNAPSHOT_END",
            ]
        )


def build_files_snapshot(resolved_files: List[Path], docker_runner: Optional["DockerRunner"] = None) -> str:
    max_chars = int(os.environ.get("CODE_FILES_SNAPSHOT_MAX_CHARS", "120000"))
    if not resolved_files:
        return ""

    blocks: List[str] = []
    for path in resolved_files:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            blocks.append(_csv_snapshot(path))
        elif suffix in {".txt", ".tsv"}:
            blocks.append(_plain_text_snapshot(path))
        elif suffix in {".rdata", ".rda"} and docker_runner is not None:
            blocks.append(_rdata_snapshot(path, docker_runner))
        else:
            blocks.append(
                "\n".join(
                    [
                        "FILE_SNAPSHOT_BEGIN",
                        f"file_name: {path.name}",
                        f"file_type: {suffix.lstrip('.') or 'unknown'}",
                        f"file_sha256: {_file_sha256(path)}",
                        "preview: unavailable for this file type",
                        "FILE_SNAPSHOT_END",
                    ]
                )
            )

    snapshot = "\n\n".join(blocks)
    if len(snapshot) <= max_chars:
        return snapshot
    return snapshot[:max_chars] + "\nFILES_SNAPSHOT_TRUNCATED: increase CODE_FILES_SNAPSHOT_MAX_CHARS to include more content."


def build_executable_r_script(answer_code: str, resolved_files: List[Path]) -> str:
    lines = ["options(warn = 1)"]
    for path in resolved_files:
        if path.suffix.lower() in {".rdata", ".rda"}:
            lines.append(f'if (file.exists("{path.name}")) load("{path.name}")')
    lines.append(answer_code.strip())
    return "\n\n".join(lines) + "\n"


class DockerRunner:
    def __init__(self, image_name: str = "actbench-runtime"):
        load_dotenv(_ROOT / ".env")
        self.image_name = image_name
        self.docker_cmd = os.getenv("DOCKER_PATH", "docker")
        self.docker_env = os.environ.copy()
        docker_config = _ROOT / ".docker"
        docker_config.mkdir(parents=True, exist_ok=True)
        self.docker_env["DOCKER_CONFIG"] = str(docker_config)
        self.build_image()

    def build_image(self) -> None:
        try:
            subprocess.run(
                [self.docker_cmd, "inspect", "--type=image", self.image_name],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=self.docker_env,
            )
            return
        except FileNotFoundError:
            raise RuntimeError(f"Docker executable '{self.docker_cmd}' was not found.")
        except subprocess.CalledProcessError:
            dockerfile = _ROOT / "docker" / "Dockerfile"
            subprocess.run(
                [self.docker_cmd, "build", "-t", self.image_name, "-f", str(dockerfile), "."],
                cwd=str(_ROOT),
                env=self.docker_env,
                check=True,
            )

    def run_r_code(self, code: str, workspace_dir: Path) -> Dict[str, Any]:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        script_path = workspace_dir / "script.R"
        script_path.write_text(code, encoding="utf-8")

        cmd = [
            self.docker_cmd,
            "run",
            "--rm",
            "--network",
            "none",
            "--stop-timeout",
            "30",
            "-v",
            f"{workspace_dir.resolve()}:/workspace_script",
            "-w",
            "/workspace_script",
            self.image_name,
            "Rscript",
            "script.R",
        ]

        try:
            start = time.perf_counter()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
                env=self.docker_env,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
                "execution_time_seconds": round(time.perf_counter() - start, 6),
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Execution timed out.",
                "exit_code": -1,
                "execution_time_seconds": 30.0,
            }


class CodeBenchScorer:
    def __init__(self, benchmark_path: Path, image_name: str = "actbench-runtime"):
        self.benchmark_path = Path(benchmark_path).resolve()
        self.benchmark_dir = self.benchmark_path.parent
        self.runner = DockerRunner(image_name=image_name)

    def _copy_resolved_files(self, resolved_files: List[Path], workspace: Path) -> None:
        for file_path in resolved_files:
            target = workspace / file_path.name
            if not target.exists():
                shutil.copy2(file_path, target)

            if file_path.suffix.lower() in {".rdata", ".rda"}:
                aliases = {
                    file_path.stem + ".RData",
                    file_path.stem + ".Rdata",
                    file_path.stem + ".rdata",
                    file_path.stem + ".rda",
                    file_path.stem + ".Rda",
                }
                for alias in aliases:
                    alias_target = workspace / alias
                    if not alias_target.exists():
                        shutil.copy2(file_path, alias_target)

    def _run_in_workspace(self, script_text: str, resolved_files: List[Path], workspace: Path) -> Dict[str, Any]:
        workspace.mkdir(parents=True, exist_ok=True)
        self._copy_resolved_files(resolved_files, workspace)
        result = self.runner.run_r_code(script_text, workspace_dir=workspace)
        result["script"] = script_text
        return result

    def execute_code(self, question: Dict[str, Any], student_output: str, workspace: Optional[Path] = None) -> Dict[str, Any]:
        resolved_files = resolve_question_files(question, self.benchmark_dir)
        student_code = extract_student_code(student_output)
        student_script = build_executable_r_script(student_code, resolved_files)

        if workspace is not None:
            exec_result = self._run_in_workspace(student_script, resolved_files, workspace)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                exec_result = self._run_in_workspace(student_script, resolved_files, Path(temp_dir))

        exec_result["resolved_files"] = [str(path) for path in resolved_files]
        exec_result["student_code"] = student_code
        return exec_result

    def execute_ground_truth(self, question: Dict[str, Any], workspace: Optional[Path] = None) -> Dict[str, Any]:
        resolved_files = resolve_question_files(question, self.benchmark_dir)
        script_text = build_executable_r_script(question.get("answer_code", ""), resolved_files)

        if workspace is not None:
            exec_result = self._run_in_workspace(script_text, resolved_files, workspace)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                exec_result = self._run_in_workspace(script_text, resolved_files, Path(temp_dir))

        exec_result["resolved_files"] = [str(path) for path in resolved_files]
        exec_result["ground_truth_code"] = question.get("answer_code", "")
        return exec_result

    def score_execution(self, question: Dict[str, Any], exec_result: Dict[str, Any]) -> Dict[str, Any]:
        scoring = question.get("scoring", {})
        if isinstance(scoring, list):
            mode = "numeric"
            references = scoring
        else:
            mode = scoring.get("mode", "numeric") if isinstance(scoring, dict) else "numeric"
            references = scoring.get("reference", []) if isinstance(scoring, dict) else []

        metrics: Dict[str, Any] = {
            "mode": mode,
            "exit_code": exec_result.get("exit_code", -1),
            "stdout": exec_result.get("stdout", ""),
            "stderr": exec_result.get("stderr", ""),
            "execution_time_seconds": exec_result.get("execution_time_seconds"),
            "execution_pass": exec_result.get("exit_code", -1) == 0,
        }

        if exec_result.get("exit_code", -1) != 0:
            metrics["passed"] = False
            metrics["avg_score"] = 0.0
            return metrics

        if mode != "numeric":
            metrics["passed"] = False
            metrics["avg_score"] = 0.0
            metrics["unsupported_mode"] = mode
            return metrics

        if references:
            metrics.update(compare_numeric_reference(exec_result.get("stdout", ""), references, DEFAULT_NUMERIC_RELATIVE_ERROR))
        else:
            metrics.update(
                {
                    "numeric_score": 1.0,
                    "numeric_pass": True,
                    "matched_target_count": 0,
                    "total_target_count": 0,
                    "parsed_numbers": extract_numbers_for_comparison(exec_result.get("stdout", "")),
                    "target_results": [],
                }
            )

        metrics["avg_score"] = 1.0 if metrics.get("numeric_pass", False) else 0.0
        metrics["passed"] = metrics.get("numeric_pass", False)
        return metrics

    def score_response(self, question: Dict[str, Any], student_output: str) -> Dict[str, Any]:
        exec_result = self.execute_code(question, student_output)
        return self.score_execution(question, exec_result)


def write_ground_truth_result(question: Dict[str, Any], benchmark_path: Path, output_dir: Path) -> Dict[str, Any]:
    scorer = CodeBenchScorer(benchmark_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    exec_result = scorer.execute_ground_truth(question, workspace=output_dir)
    score = scorer.score_execution(question, exec_result)
    (output_dir / "script.R").write_text(exec_result.get("script", ""), encoding="utf-8")
    (output_dir / "stdout.txt").write_text(exec_result.get("stdout", ""), encoding="utf-8")
    (output_dir / "stderr.txt").write_text(exec_result.get("stderr", ""), encoding="utf-8")
    (output_dir / "result.json").write_text(json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8")
    return score
