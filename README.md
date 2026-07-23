<div align="center">

# INS-ActBench

### A benchmark for professional actuarial reasoning with large language models

![Tasks](https://img.shields.io/badge/tasks-12%2C050-2448A7)
![Subsets](https://img.shields.io/badge/task_families-3-5B3C88)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Tools](https://img.shields.io/badge/tools-Spreadsheets%20%7C%20R-4E7A3D)

</div>

INS-ActBench is a benchmark for evaluating large language models on actuarial knowledge, case analysis, spreadsheet modeling, and statistical R programming. It contains 12,050 tasks based on public examination and sample materials released by actuarial associations around the world.

Actuarial problems often combine technical knowledge with long documents. INS-ActBench therefore includes conventional question answering alongside case-based multiple selection, spreadsheet workbook completion, and executable R programming.

## Dataset

INS-ActBench has three parts. INS-Act-Know covers the core technical syllabus, INS-Act-Case tests judgment over professional case materials, and INS-Act-Practice evaluates whether a model can complete actuarial work with spreadsheets and R.

| Part | ID | Subject | Task format | Tasks |
|:---|:---:|:---|:---|---:|
| INS-Act-Know | PS | Probability and Statistics | Single-answer MCQ | 2,164 |
| INS-Act-Know | EF | Economics and Finance | Single-answer MCQ | 2,025 |
| INS-Act-Know | AMA | Actuarial Mathematics | Single-answer MCQ | 1,134 |
| INS-Act-Know | LI | Life Insurance | Single-answer MCQ | 1,849 |
| INS-Act-Know | NLI | Non-life Insurance | Single-answer MCQ | 1,168 |
| INS-Act-Know | AMO | Actuarial Modeling | Single-answer MCQ | 1,409 |
| INS-Act-Case | CA | Long-context actuarial cases | Multiple-select | 1,149 |
| INS-Act-Practice | SS | Spreadsheet modeling | Executable Python and workbook | 636 |
| INS-Act-Practice | R | Statistical programming | Executable R | 516 |
| **Total** |  |  |  | **12,050** |

The dataset draws on materials from 16 actuarial associations. For aggregate reporting, they are grouped into SOA, IFoA, and other actuarial sources.

| Source group | Tasks | Share |
|:---|---:|---:|
| SOA | 2,728 | 22.64% |
| IFoA | 2,129 | 17.67% |
| Others | 7,193 | 59.69% |

## Evaluation code

[`bench_runner/run.py`](bench_runner/run.py) is the common entry point for the four task formats. [`bench_runner/score.py`](bench_runner/score.py) summarizes or rescores existing runs. The remaining modules are grouped as follows:

| Path | Contents |
|:---|:---|
| [`bench_runner/`](bench_runner) | Prompts, task runners, scoring, and error analysis |
| [`models/`](models) | Model adapters and configuration factory |
| [`docker/`](docker) | LibreOffice and R execution environment |
| [`config.yaml`](config.yaml) | Model profiles and inference settings |

## Quick start

Python 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Copy [`.env.example`](.env.example) to `.env`, add the credentials required by your model profiles, and review [`config.yaml`](config.yaml). Build the container when running spreadsheet or R tasks:

```bash
docker compose -f docker/docker-compose.yml build
```

Run an evaluation from the repository root:

```bash
python bench_runner/run.py \
  --bench mcq \
  --model DeepSeek-V4-Pro \
  --test-data /path/to/LV1.1.json \
  --limit 10
```
