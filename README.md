<div align="center">

# INS-ActBench

### A benchmark for professional actuarial reasoning with language models

![Tasks](https://img.shields.io/badge/tasks-12%2C050-2448A7)
![Task families](https://img.shields.io/badge/task_families-3-5B3C88)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Tools](https://img.shields.io/badge/tools-Spreadsheets%20%7C%20R-4E7A3D)

</div>

INS-ActBench is a benchmark for evaluating large language models on actuarial knowledge, case analysis, spreadsheet modeling, and statistical programming. It contains 12,050 tasks based on public examination and sample materials released by actuarial organizations around the world.

Actuarial problems often combine technical knowledge with long documents, linked assumptions, calculations, and software tools. INS-ActBench therefore includes conventional question answering alongside case-based multiple selection, workbook completion, and executable R programming.

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

The dataset draws on materials from 16 actuarial associations. Source labels include `SOA`, `IFoA`, `CAA`, `IAI`, `IAJ`, `IAK`, `PAI`, `DAV`, `IBA`, `ASHK`, `CIA`, `ASSA`, `JSCPA`, `AIA`, `CERA`, and `ISOA`. For aggregate reporting, they are grouped into SOA, IFoA, and other actuarial sources.

| Source group | Tasks | Share |
|:---|---:|---:|
| SOA | 2,728 | 22.64% |
| IFoA | 2,129 | 17.67% |
| Others | 7,193 | 59.69% |

In the final dataset, 69.39% of the questions differ from their original form. Adaptations include revised answer options, objective scoring for case questions, shorter case contexts, and numerical targets for practical tasks.

### INS-Act-Know

INS-Act-Know contains 9,749 self-contained multiple-choice questions across six actuarial subject areas. Each question has one correct option and can be scored by exact match.

Questions also carry a `numerical` annotation. The 8,545 numerical questions require formulas, calculations, or interpretation of stated values. The remaining 1,204 questions focus on concepts, definitions, regulation, and professional knowledge. This split allows results to be reported separately for calculation-heavy and conceptual tasks.

### INS-Act-Case

INS-Act-Case contains 1,149 multiple-select questions arranged in 70 case groups. A group may contain background documents, supporting tables, several task stems, and multiple questions that refer to the same case.

Every group has two context versions:

- `case-long` retains the complete case material.
- `case-short` is a compressed version for models with smaller context windows.

Across the 70 groups, the long contexts contain 2,725,625 tokens and the short contexts contain 1,234,691 tokens, an aggregate reduction of 54.70%. Both versions preserve the information required to answer the associated questions. Answers are sets of option letters, such as `BDE`, and scoring requires an exact set match.

### INS-Act-Practice

INS-Act-Practice contains 1,152 tasks that require a model to use an actuarial tool rather than return a short text answer.

The spreadsheet subset has 636 tasks organized around input workbooks. A task specifies the instruction, target answer cells, and any dependency on earlier work in the same workbook. The model writes Python code that reads `input.xlsx`, inserts formulas or values, and saves `output.xlsx`. The completed workbook is recalculated before the target cells are compared with the reference workbook.

The R subset has 516 programming tasks. Records may include data files, a statistical question, and one or more numerical targets. The model writes R code, which runs inside the supplied Docker environment. The scorer parses numerical stdout and checks whether every required target is present within the configured tolerance.

The principal fields used by each subset are:

| Subset | Main fields |
|:---|:---|
| Knowledge | `id`, `source`, `numerical`, `question`, `options`, `answer` |
| Case | `group_id`, `source`, `case-long`, `case-short`, stems, questions, options, answers |
| Spreadsheet | `group_id`, `id`, `source`, `question`, `depends_on`, `template`, `answer_position` |
| R | `group_id`, `id`, `source`, `question`, `files`, `scoring` |

## Evaluation code

[`bench_runner/run.py`](bench_runner/run.py) is the common entry point for the four task formats. [`bench_runner/score.py`](bench_runner/score.py) summarizes or rescores existing runs. Spreadsheet and R answers use a 1% relative numerical tolerance by default, while knowledge and case questions use exact matching.

The remaining modules are grouped as follows:

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

`--bench` accepts `mcq`, `case`, `spreadsheet`, or `code`. Use `--questions` and `--group-ids` to select individual tasks, and `--output-dir` to choose where results are written.

Existing runs can be summarized with:

```bash
python bench_runner/score.py --bench mcq --run-path /path/to/run
```

INS-ActBench is intended for research evaluation. It is not a substitute for professional actuarial review.
