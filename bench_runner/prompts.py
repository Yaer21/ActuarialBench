"""Centralized prompt templates for ActuarialBench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

SYSTEM_PROMPTS: Dict[str, str] = {
    "mcq": (
        "You are an expert actuarial analyst solving the multiple-choice question.\n"
        "Output format rules:\n"
        "- Respond with exactly ONE option letter\n"
        "- Do not output any other text."
    ),
    "case": (
        "You are an expert actuarial analyst solving case-based actuarial multiple-select questions.\n"
        "You will be given case materials and, when relevant, supporting tables that may be useful for solving the question.\n"
        "For each item, use the provided stem as the background and answer the question under that stem.\n"
        "This is a multiple-select question. At least TWO options must be selected.\n"
        "Scoring rule: credit is awarded only if all and only the correct options are selected; any missing option, extra option, incorrect option, or empty response receives zero credit.\n"
        "Output format rules:\n"
        "- Respond with the selected option letters only\n"
        "- Do not output any explanation, reasoning, words, punctuation, or extra text\n"
        "- Write the capital letters together in alphabetical order, for example: AB"
    ),
    "spreadsheet": (
        "You are solving an actuarial spreadsheet task.\n"
        "Write one executable Python code block using openpyxl.\n"
        "Load the input workbook from INPUT_WORKBOOK, or input.xlsx if the variable is missing.\n"
        "Use the workbook snapshot and task text to understand the sheets, cells, and formulas.\n"
        "Fill the final answer cells in ANSWER_POSITION; you may use other existing cells for intermediate calculations.\n"
        "Prefer spreadsheet formulas over hard-coded final values when formulas can be built from the workbook data.\n"
        "Do not rename sheets, create extra sheets, or overwrite unrelated content.\n"
        "Save the completed workbook to OUTPUT_WORKBOOK, or output.xlsx if the variable is missing.\n"
        "Output only the Python code block, with no explanation."
    ),
    "code": (
        "You are solving an actuarial R coding test.\n"
        "Your task is to write accurate R code to compute and solve the given actuarial question.\n\n"
        "CRITICAL Output Format:\n"
        "- Output ONLY valid, executable R code enclosed in a single ```R ... ``` block.\n"
        "- Do NOT output any standard text explanations, preambles, or concluding remarks.\n\n"
        "CRITICAL Code Rules:\n"
        "- Assume a fresh base R session. Do not install external packages.\n"
        "- Recreate any inline vectors/matrices exactly.\n"
        "- If datasets like .RData or .csv are provided, assume they are available in the working directory and load them directly.\n"
        "- ALL final numerical answers, statistics, or metrics MUST be explicitly printed to the console (e.g., by returning the variable name on the last line or calling `print()`) so they can be captured by standard output.\n"
    ),
}

# Shared examples for all LV1/LV2 MCQ runs.
MCQ_FEWSHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "question": "If i = 5%, what is v = (1+i)^(-1)?",
        "options": {
            "A": "0.9524",
            "B": "0.9500",
            "C": "1.0500",
            "D": "0.9048",
            "E": "0.9750",
        },
        "answer": "A",
    },
    {
        "question": "If q_x = 0.02, what is p_x?",
        "options": {
            "A": "0.02",
            "B": "0.98",
            "C": "1.02",
            "D": "0.50",
            "E": "0.00",
        },
        "answer": "B",
    },
]

CASE_FEWSHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "case": "The background of ABC company is ...",
        "table": "The balance sheet of ABC company: ...",
        "stem": "You are hired as a consultant for ABC company, you will ...",
        "question": "Which of the following statements about the consultant are correct?",
        "options": {
            "A": "statement-1",
            "B": "statement-2",
            "C": "statement-3",
            "D": "statement-4",
            "E": "statement-5",
        },
        "answer": "AB",
    },
    {
        "case": "The background of ABC company is ...",
        "table": "The balance sheet of ABC company: ...",
        "stem": "You are hired as a consultant for ABC company, you will ...",
        "question": "You have received five judgements about ABC company from your colleagues, which of the following judgments are more reasonable?",
        "options": {
            "A": "judgement-1",
            "B": "judgement-2",
            "C": "judgement-3",
            "D": "judgement-4",
            "E": "judgement-5",
            "F": "judgement-6",
        },
        "answer": "BDE",
    },
]

TAG_HINTS: Dict[str, str] = {
    "Profit Testing": "Use year-by-year cashflow recursion and reference fixed assumptions with absolute cell references.",
    "Present Value": "Discount factors: v^t = (1+i)^(-t). Present value is the sum of discounted cashflows.",
    "Discounted Mean Term": "DMT is a weighted average term using discounted cashflows.",
    "Multiple Decrements": "Convert independent to dependent decrements using the 1-0.5q adjustment factors.",
    "Index-Linked Bonds": "Apply inflation indexation with the stated time lag; discount post-tax cashflows at the target yield.",
}


@dataclass(frozen=True)
class SpreadsheetPromptContext:
    task_id: str
    question_id: int
    group_id: str
    instruction: str
    background: str
    template_workbook: str
    answer_position: Any
    workbook_snapshot: str
    knowledge_tags: List[str]
    dependency_questions: List[str]


def build_spreadsheet_user_prompt(ctx: SpreadsheetPromptContext) -> str:
    hints = [TAG_HINTS[t] for t in ctx.knowledge_tags if t in TAG_HINTS]
    hint_block = "\n".join(f"- {h}" for h in hints)
    dependency_block = "\n".join(f"- {item}" for item in ctx.dependency_questions)
    question_label = str(ctx.group_id)

    return (
        f"Task ID: {ctx.task_id}\n"
        f"Question: {question_label} [global id={ctx.question_id}]\n\n"
        f"Background:\n{ctx.background}\n\n"
        f"Instruction:\n{ctx.instruction}\n\n"
        f"Files:\n"
        f"- Input workbook file available to your code: input.xlsx\n"
        f"- input.xlsx is selected by the runner: dependency formula workbook when available, otherwise the original template workbook\n"
        f"- Required output workbook file to create: output.xlsx\n"
        f"- Environment variables also available: INPUT_WORKBOOK and OUTPUT_WORKBOOK\n"
        f"- Use Python and openpyxl to read input.xlsx, fill the answer cells, and save output.xlsx\n\n"
        f"Workbook snapshot:\n{ctx.workbook_snapshot}\n\n"
        f"Constraints:\n"
        f"- ANSWER_POSITION: {ctx.answer_position}\n"
        f"- You may read any sheet in the workbook\n"
        f"- Final answers must appear in ANSWER_POSITION because only those cells are scored\n"
        f"- You may use other existing cells for intermediate calculations when helpful\n"
        f"- Use spreadsheet formulas where appropriate\n"
        f"- Do not overwrite unrelated workbook content outside the cells needed for the calculation\n\n"
        + ("Dependency context:\n" + dependency_block + "\n\n" if ctx.dependency_questions else "")
        + ("Helpful hints (from tags):\n" + hint_block + "\n\n" if hints else "")
        + "Output only one Python code block. The code must create output.xlsx."
    )


@dataclass(frozen=True)
class CodePromptContext:
    question: str
    files: List[str]
    knowledge_tags: List[str]
    files_snapshot: str = ""


def build_code_user_prompt(ctx: CodePromptContext) -> str:
    hints = [TAG_HINTS[t] for t in ctx.knowledge_tags if t in TAG_HINTS]
    hint_block = "\n".join(f"- {h}" for h in hints)
    
    file_list = "\n".join(f"- {f}" for f in ctx.files)

    return (
        f"Question:\n{ctx.question}\n\n"
        f"Available Data Files:\n{file_list}\n\n"
        + (f"Files snapshot:\n{ctx.files_snapshot}\n\n" if ctx.files_snapshot else "")
        + ("Helpful hints (from tags):\n" + hint_block + "\n\n" if hints else "")
        + "Please write R code to load the data (if needed) and solve the problem."
    )


@dataclass(frozen=True)
class CasePromptContext:
    case_id: int
    year: str
    season: str
    subject: str
    stem_id: str
    question_id: str
    case_material: str
    table_material: str
    stem_text: str
    question_text: str
    options: Dict[str, str]


def build_case_user_prompt(ctx: CasePromptContext) -> str:
    option_lines = "\n".join(f"{key}. {value}" for key, value in ctx.options.items())
    fewshot_blocks = []
    for idx, example in enumerate(CASE_FEWSHOT_EXAMPLES, start=1):
        example_options = "\n".join(f"{key}. {value}" for key, value in example["options"].items())
        fewshot_blocks.append(
            f"Example {idx}:\n"
            f"Case Material:\n{example['case']}\n\n"
            f"Supporting Table Material:\n{example['table']}\n\n"
            f"Stem:\n{example['stem']}\n\n"
            f"Question:\n{example['question']}\n\n"
            f"Options:\n{example_options}\n"
            f"Answer: {example['answer']}"
        )
    fewshot_text = "\n\n".join(fewshot_blocks)

    return (
        f"Here are some example case-based multiple-select questions and their answers:\n\n"
        f"{fewshot_text}\n\n"
        f"Now solve this question in the same answer format.\n\n"
        f"Case Material:\n{ctx.case_material}\n\n"
        f"Supporting Table Material:\n{ctx.table_material}\n\n"
        f"Stem:\n{ctx.stem_text}\n\n"
        f"Question:\n{ctx.question_text}\n\n"
        f"Options:\n{option_lines}\n"
    )


def guess_task_type(task: Dict[str, Any]) -> str:
    """Best-effort mapping for now; will be expanded later."""

    if task.get("template") is not None:
        return "spreadsheet"
    if task.get("files") and any(f.endswith(".RData") for f in task.get("files", [])):
        return "code"
    return "text"
