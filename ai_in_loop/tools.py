"""Tool definitions for the AI-in-the-Loop application.

This module defines tools that can be used by the LLM via LangGraph's
tool calling capabilities. Tools are defined using LangChain's @tool
decorator for model-agnostic compatibility.
"""

from __future__ import annotations

import ast
import csv
import math
import operator
import re
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any


from data.program_data import get_program_data
from loaders.major_requirements_index import get_url

from typing import Any, TYPE_CHECKING

from langchain_core.tools import tool

import requests
import re
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from .config import Config

# Module-level config reference for search_docs tool
_search_config: Config | None = None


def set_search_config(cfg: Config) -> None:
    """Set config for search_docs tool. Called at startup."""
    global _search_config
    _search_config = cfg


# Safe operators for math expressions
SAFE_OPERATORS: dict[type, Any] = {
    # Arithmetic operators
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    # Comparison operators
    ast.Lt: operator.lt,
    ast.Gt: operator.gt,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.LtE: operator.le,
    ast.GtE: operator.ge,
}

# Safe math functions and constants
SAFE_FUNCTIONS: dict[str, Any] = {
    # Built-in functions
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    # Math module functions
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": math.pow,
    # Combinatorics
    "factorial": math.factorial,
    "comb": math.comb,
    "perm": math.perm,
    # Inverse trigonometric functions
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    # Math constants
    "pi": math.pi,
    "e": math.e,
}


class SafeEvalError(Exception):
    """Raised when expression evaluation fails or is unsafe."""

    pass


def _safe_eval_node(node: ast.AST) -> Any:
    """Recursively evaluate an AST node using only safe operations.

    Args:
        node: An AST node to evaluate

    Returns:
        The computed value

    Raises:
        SafeEvalError: If the expression contains unsupported operations
    """
    # Numbers
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise SafeEvalError(f"Unsupported constant type: {type(node.value).__name__}")

    # Names (variables/constants like pi, e)
    if isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        raise SafeEvalError(f"Unknown variable: {node.id}")

    # Binary operations (a + b, a * b, etc.)
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise SafeEvalError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return SAFE_OPERATORS[op_type](left, right)

    # Unary operations (-x, +x)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise SafeEvalError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval_node(node.operand)
        return SAFE_OPERATORS[op_type](operand)

    # Comparison operations (5 > 3, 2 < 3 < 5, x == y, etc.)
    if isinstance(node, ast.Compare):
        left = _safe_eval_node(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in SAFE_OPERATORS:
                raise SafeEvalError(f"Unsupported comparison operator: {op_type.__name__}")
            right = _safe_eval_node(comparator)
            if not SAFE_OPERATORS[op_type](left, right):
                return False
            left = right  # For chained comparisons like 2 < 3 < 5
        return True

    # Function calls (sqrt(x), sin(x), etc.)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise SafeEvalError("Only simple function calls are supported")
        func_name = node.func.id
        if func_name not in SAFE_FUNCTIONS:
            raise SafeEvalError(f"Unknown function: {func_name}")
        func = SAFE_FUNCTIONS[func_name]
        if not callable(func):
            raise SafeEvalError(f"{func_name} is not a function")
        args = [_safe_eval_node(arg) for arg in node.args]
        return func(*args)

    # Lists/tuples for functions like min, max, sum
    if isinstance(node, ast.List) or isinstance(node, ast.Tuple):
        return [_safe_eval_node(elt) for elt in node.elts]

    raise SafeEvalError(f"Unsupported expression type: {type(node).__name__}")


def safe_eval(expression: str) -> float | int | bool:
    """Safely evaluate a mathematical expression.

    Uses AST parsing to only allow whitelisted operators and functions.
    No arbitrary code execution is possible.

    Args:
        expression: A math expression like "2 + 2" or "sqrt(16) * 3"

    Returns:
        The computed numeric result

    Raises:
        SafeEvalError: If the expression is invalid or contains unsafe operations
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise SafeEvalError(f"Invalid syntax: {e.msg}") from e

    return _safe_eval_node(tree.body)


@tool
def python_calc(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Use this tool to perform arithmetic, comparisons, or other mathematical
    computations.

    Supports:
    - Basic operators: +, -, *, /, **, //, %
    - Comparison operators: <, >, ==, !=, <=, >=
    - Trig functions: sin, cos, tan, asin, acos, atan, atan2
    - Other math: sqrt, log, log10, log2, exp, floor, ceil, pow
    - Combinatorics: factorial, comb, perm
    - Constants: pi, e

    Args:
        expression: A math expression like "2 + 2", "sqrt(16) * 3",
                   or "sin(pi / 2)". Use 'pi' and 'e' for constants.

    Returns:
        The computed result as a string, or an error message if evaluation fails.
    """
    try:
        result = safe_eval(expression)
        # Format nicely: avoid unnecessary decimals for whole numbers
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except SafeEvalError as e:
        return f"Error: {e}"
    except ZeroDivisionError:
        return "Error: Division by zero"
    except OverflowError:
        return "Error: Result too large"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: Unexpected error - {type(e).__name__}: {e}"


@tool
def search_docs(query: str) -> str:
    """Search documents for information relevant to the query.

    Use this tool to find information from documents in the resources/rag directory.

    Args:
        query: The search query describing what information you need.

    Returns:
        Relevant document passages with source info, or a message if none found.
    """
    from .retriever import get_retriever

    # if _search_confix is None, it means that the search system has not been set up yet
    # so instead of giving an exception and crashing, an error string is returned
    if _search_config is None:
        return "Error: Search not configured."

    # create the retriever from the _search_config
    retriever = get_retriever(_search_config)

    # checks if documents exist in the folder, and if not, return an error string
    # so that it doesn't just crash the whole program
    if retriever is None:
        return "No documents available. The resources/rag directory may be empty."

    # sends the input from the user to the retriever, search it, and then return the results
    # .invoke is a way to run something from start to finish
    results = retriever.invoke(query)

    # it tried to search, but did not get any relevent results, so the results variable remains
    # empty
    if not results:
        return "No relevant documents found."

    # this formats all the results into one string, joined by --- (with two empty lines above and below)
    # if the source is known, it puts in the source, and if not, use 'unknown'#
    # doc.page_content is the actual content of the document chunk
    return "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
        for doc in results
    )

DAY_CODES = {"M", "T", "W", "R", "F", "S", "U"}

# subject abbreviations from official course names
SUBJECT_ABBREV: dict[str, str] = {
    "Accounting": "ACCT",
    "American Ethnic Studies": "AES",
    "Applied Learning": "APPL",
    "Art and Art History": "ART",
    "Biology": "BIOL",
    "Business": "BUS",
    "Chemistry": "CHEM",
    "Chinese": "CHIN",
    "Classics": "CLAS",
    "Communication": "COMM",
    "Computer Science": "CSCI",
    "Dance": "DAN",
    "Economics": "ECON",
    "Education": "EDUC",
    "Engineering": "ENGS",
    "English": "ENGL",
    "Environmental Studies": "ENVR",
    "French": "FREN",
    "Gen Education Math & Science": "GEMS",
    "Geological/Environment Science": "GES",
    "German": "GERM",
    "Greek": "GRK",
    "Hebrew": "HEBR",
    "History": "HIST",
    "Interdisciplinary": "IDS",
    "Japanese": "JAPN",
    "Kinesiology": "KIN",
    "Latin": "LATN",
    "Leadership": "LDRS",
    "Linguistics": "LING",
    "Mathematics": "MATH",
    "Math": "MATH",
    "Ministry": "MIN",
    "Music": "MUS",
    "Music Applied Lessons": "MUSA",
    "Music Ensembles": "MUSE",
    "Neuroscience": "NSCI",
    "Nursing": "NURS",
    "Philosophy": "PHIL",
    "Physics": "PHYS",
    "Political Science": "POL",
    "Psychology": "PSY",
    "Religion": "REL",
    "Social Work": "SWK",
    "Sociology": "SOC",
    "Spanish": "SPAN",
    "Theatre": "THEA",
    "Women's and Gender Studies": "WGS",
}

# ------------------
# THIS WHOLE SECTION STANDARDIZES THE USER'S INPUT 
# ------------------

# empty dictionary where random user input courses can be added to map to 
# the right course abbreviation
# this is to make it so engr translates to engs as the official code
# for Hope's engineering courses (example)
# stores all acceptable versions of a subject to map them to the right code
SUBJECT_ALIASES: dict[str, str] = {}

# loop through all the pairs in SUBJECT_ABBREV dictionary
for full_name, code in SUBJECT_ABBREV.items():
    # stores the uppercase of the official name (+ the code) in SUBJECT_ALIASES
    # to make matching easier and case-insensitive
    SUBJECT_ALIASES[full_name.upper()] = code
        
    # remove everything that's not a letter, so spaces, punctuation, etc.
    # catches input that is messy or formatted differently
    SUBJECT_ALIASES[re.sub(r"[^A-Za-z]", "", full_name).upper()] = code

# add in the correct, official codes themselves, so if users just type in the exact
# codes there won't be any unexpected behavior,
# and the tool will still be able to do what it needs to do
for code in SUBJECT_ABBREV.values():
    SUBJECT_ALIASES[code.upper()] = code.upper()

# extra common model/user abbreviations (just in case)
SUBJECT_ALIASES.update({
    "ENGR": "ENGS",
    "COMP": "CSCI",
    "CS": "CSCI",
    "BIO": "BIOL",
    "CHE": "CHEM",
    "PSYC": "PSY",
    "JPN": "JAPN",
    "Orchestra": "MUSE",
    "Band": "MUSE",
})

# take the subject part of the input and changes it to the official code
def _canonicalize_subject_token(token: str) -> str:
    cleaned = str(token).strip().upper()
    
    # remove everything that's not a letter, so spaces, punctuation, etc.
    # this is just in case the cleaned string doesn't find a match in SUBJECT_ALIASES
    compact = re.sub(r"[^A-Za-z]", "", cleaned)

    # this compares the cleaned string or the compact string to keys in SUBJECT_ALIASES
    # to get the right value
    if cleaned in SUBJECT_ALIASES:
        return SUBJECT_ALIASES[cleaned]
    if compact in SUBJECT_ALIASES:
        return SUBJECT_ALIASES[compact]

    # if subject wasn't found both times, then just return the compact string
    return compact

# this function takes the whole course string and converts it to a standard format like "CSCI 195"
def _normalize_requested_course(query: str) -> str:
    q = str(query).strip()

    # this one matches the subject part and course number part
    # \s allows 0 or more spaces between the subject and course number
    match = re.fullmatch(r"([A-Za-z&/\- ]+?)\s*([0-9]+[A-Za-z]?)", q)

    # fallback in case no match
    if not match:
        return q.upper()

    subject_part, number_part = match.groups()

    # convert the subject to the right subject code
    subject_code = _canonicalize_subject_token(subject_part)

    # combines official code with the number 
    # number is just a number so no crazy things need to be done with it
    # like with the course code :')
    return f"{subject_code} {number_part}".upper()

# build the full file path to the schedule CSV
# if the file name, parent folders, or structure changes this part just needs to update accordingly 
def _resource_path(filename: str) -> Path:
    """Return a path inside the repo root resources/schedules directory."""

    # get the absolute path from this file and go up two folders to get the repo root
    repo_root = Path(__file__).resolve().parent.parent

    # join the repo root with resources/schedules and given filename
    return repo_root / "resources" / "schedules" / filename

# ------------------------
# THIS PART DEALS WITH NORMALIZING AND PARSING TIMES!
# ------------------------

# makes all times consistent before converting them to minutes
def _normalize_time_ampm(time_text: str) -> str:
    """Convert '9:30 am' -> '09:30'."""
    dt = datetime.strptime(time_text.strip().lower(), "%I:%M %p")
    return dt.strftime("%H:%M")

# converts a time HH:MM format into minutes after midnight
# so 09:30 becomes 570 minutes
def _time_to_minutes(hhmm: str) -> int:
    hour, minute = map(int, hhmm.split(":"))
    return hour * 60 + minute

# extract the valid day codes from a meeting into a string
# so TR becomes ['T', 'R'] for Tuesday and Thursday
def _parse_days(day_text: str) -> list[str]:
    return [ch for ch in day_text.strip() if ch in DAY_CODES]

# parse the full meeting string into a list of structured meeting dictionaries
def _parse_meeting_times(meeting_text: str) -> list[dict[str, Any]]:
    """
    Parse strings like:
    'TR — 9:30 am-10:50 am — VNZORN 142 | F — 9:30 am-10:20 am — VNZORN 142'
    """
    # if it's empty....
    if not meeting_text:
        return []

    # clean the text and split the meetings by the pipe symbol (|)
    text = str(meeting_text).replace("\xa0", " ").strip()
    parts = [p.strip() for p in text.split("|") if p.strip()]
    meetings: list[dict[str, Any]] = []

    for part in parts:
        # split each meeting into three parts:
        # days, time range, location
        chunks = [c.strip() for c in re.split(r"\s+[—–]\s+", part, maxsplit=2)]
        if len(chunks) < 3:
            continue
        
        # each part respectively stored here 
        day_part, time_part, location = chunks

        # time range to start time and end time
        # accept hyphen, en dash, or em dash in the time range
        match = re.match(r"(.+?)\s*[-–—]\s*(.+)", time_part)
        if not match:
            continue

        # clean ups
        start_raw = match.group(1).strip()
        end_raw = match.group(2).strip()

        # normalize the times and days into a consistent format
        start = _normalize_time_ampm(start_raw)
        end = _normalize_time_ampm(end_raw)
        days = _parse_days(day_part)

        # store each meeting in structured parts so that we can
        # check for conflicts more easily later
        # actually i think this helped speed up the tool!!
        meetings.append(
            {
                "days": days,
                "day_set": set(days),
                "start": start,
                "end": end,
                "start_min": _time_to_minutes(start),
                "end_min": _time_to_minutes(end),
                "location": location,
            }
        )

    return meetings

# safely converts a value into an integer
def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


# opens the csv file for reading as utf-8
def _load_sections_csv(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    # this with makes sure the file is closed when done (no need for f.close)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        # each row becomes a dictionary 
        # dictionary keys come from the header row of the file
        return list(csv.DictReader(f))

# takes the raw row from the CSV file and turns it into a clean dictionary containing
# section info that makes scheduling easier
def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    For scheduling, only timing/identity fields are used to build schedules.
    Status and note are preserved only so the tool can return warnings.
    """
    subject_name = str(row.get("Subject", "")).strip()
    # get the right code here
    subject_code = _canonicalize_subject_token(subject_name)

    course_number_raw = row.get("Course Number", "")
    course_number = str(_safe_int(course_number_raw) or str(course_number_raw).strip())

    # this gets the section number e.g. CSCI 115-01 the -01 part
    section_raw = row.get("Sequence Number", "")
    section_int = _safe_int(section_raw)
    # .zfill pads it with leading 0s so that it's always 2 digits
    section = str(section_int).zfill(2) if section_int is not None else str(section_raw).strip()

    title = str(row.get("Course Title", "")).strip()
    instructor = str(row.get("Instructor", "")).strip()
    meetings = _parse_meeting_times(str(row.get("Meeting Times", "")).strip())

    # waitlist, closed, open
    status = str(row.get("Status", "")).strip()
    # take note of prereqs or permissions if there are any
    note = str(row.get("Note", "")).strip()

    # combine subject code and course number that will be used later for matching 
    course_id = f"{subject_code} {course_number}".strip()

    # this is the cleaned section dictionary containing all the necessary info
    # do it like this so that later on functions can just easily use this rather
    # than having to keep cleaning all the raw CSV values again
    section_data = {
        "subject": subject_code,
        "subject_name": subject_name,
        "course_number": course_number,
        "course_id": course_id,
        "section": section,
        "title": title,
        "instructor": instructor,
        "meetings": meetings,
        "status": status,
        "note": note,
    }

    # compute all the warnings once over here and add to the dictionary
    section_data["warnings"] = _section_warnings(section_data)
    return section_data

# exists so that we don't have to look through every section in the whole CSV file again and again
# without this function, we'd have to keep looping through the sections dictionary to check if the
# key matches the desired course
# with this function, we can just do this: sections_by_course[CSCI 125]/sections_by_course[CSCI125]
# and find all the related sections
def _build_sections_index(sections: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    Build a lookup dictionary so course matching is fast.

    Example keys:
    - 'CSCI 125'
    - 'CSCI125'
    """

    # this dictionary will gradually get filled with course ids, and then their sections as values
    sections_by_course: dict[str, list[dict[str, Any]]] = {}

    # loop through all the sections in the input list
    for section in sections:
        # turns the course code uppercase which makes matching a little easier cuz everything is
        # consistent
        course_key = section["course_id"].upper()
        # this line checks if the course key already exists in the list
        # if no, create a key for it and then allow values which are the sections, to be appended
        # otherwise if it exists, then just append the section to that course key
        sections_by_course.setdefault(course_key, []).append(section)

        # just in case the tool calls the course ids without spaces, e.g. CSCI 125 vs CSCI125
        # this just catches that and still allows the tool to work
        compact_key = course_key.replace(" ", "")
        if compact_key != course_key:
            sections_by_course.setdefault(compact_key, []).append(section)

    return sections_by_course

# this scores schedules to give them a ranking 
def _schedule_score(schedule: list[dict[str, Any]]) -> int:
    score = 0

    for section in schedule:
        for meeting in section["meetings"]:
            # gives schedules that start at 10am or later higher scores
            if meeting["start_min"] >= 10 * 60:
                score += 1

    return score


# looks at the status and notes sections of each course scheduled
# does not prevent any scheduling, but is able to alert the user 
# that a class is closed, waitlisted, or requires special permission to take
def _section_warnings(section: dict[str, Any]) -> list[str]:
    warnings: list[str] = []

    status_text = str(section.get("status", "")).strip()
    note_text = str(section.get("note", "")).strip()
    combined = f"{status_text} {note_text}".lower()

    if "permission" in combined or "consent" in combined:
        warnings.append(
            f"{section['course_id']} section {section['section']} may require instructor/department permission."
        )

    if "waitlist" in combined or "wait list" in combined:
        warnings.append(
            f"{section['course_id']} section {section['section']} appears to be waitlisted."
        )

    if "full" in combined or "closed" in combined:
        warnings.append(
            f"{section['course_id']} section {section['section']} may be full or closed."
        )

    return warnings

# check two single meetings to see if they overlap on the same day at the same time
# is yes, add the conflict, otherwise just return nothing
def _find_meeting_conflict(m1: dict, m2: dict) -> str | None:
    """Checks if two meetings conflict and returns a description if they do."""
    # if they're not on the same day, no chance of conflict anyway
    shared_days = m1["day_set"] & m2["day_set"]
    if not shared_days:
        return None

    if m1["start_min"] < m2["end_min"] and m2["start_min"] < m1["end_min"]:
        days_str = "".join(sorted(shared_days))
        return f"{days_str} ({m1['start']}-{m1['end']} vs {m2['start']}-{m2['end']})"

    return None

# this utilizes the meeting_conflict function above for two sections 
def _get_sections_conflict(sec1: dict, sec2: dict) -> str | None:
    """Returns a string describing the conflict between two sections, if any."""
    # loop through all the meetings in sec1, and compare to all the meetings in sec2
    for m1 in sec1["meetings"]:
        for m2 in sec2["meetings"]:
            reason = _find_meeting_conflict(m1, m2)
            if reason:
                return f"{sec1['course_id']} (Sec {sec1['section']}) and {sec2['course_id']} (Sec {sec2['section']}) overlap on {reason}"
    return None

# takes one full schedule and just adds all the warning messages
# this way it avoids duplicated warnings
def _schedule_warnings(schedule: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    warnings: list[str] = []

    for section in schedule:
        # this adds the warnings that we computed earlier for each section
        # and adds them to the schedule's warnings list
        for warning in section.get("warnings", []):
            # this if statement ensures that no duplicated warnings are added
            if warning not in seen:
                seen.add(warning)
                warnings.append(warning)

    return warnings


# generates a max number of valid schedules
def _generate_valid_schedules(
    sections_by_course: dict[str, list[dict[str, Any]]],
    desired_courses: list[str],
    max_schedules: int,
) -> dict[str, Any]:
    """
    Returns a dict with:
    - 'ok': Boolean
    - 'schedules': List of unique valid schedules
    - 'error': Error message
    - 'conflicts': List of specific overlap descriptions
    """
    grouped: dict[str, list[dict[str, Any]]] = {}

    # match each requested course to its available sections
    for query in desired_courses:
        query_clean = _normalize_requested_course(query)
        query_no_space = query_clean.replace(" ", "")

        matches = sections_by_course.get(query_clean) or sections_by_course.get(query_no_space) or []
        
        # if there are no matches AT ALL, stop schedule generation completely
        if not matches:
            return {
                "ok": False,
                "error": f"Could not find any sections for '{query}'.",
                "schedules": [],
                "conflicts": [],
            }

        course_id = matches[0]["course_id"]

        # dedupe sections for that course just in case
        unique_sections: list[dict[str, Any]] = []
        # keep track of sections that have already been seen
        seen_sections: set[tuple[str, str]] = set()

        for sec in matches:
            sec_key = (sec["course_id"], sec["section"])
            if sec_key not in seen_sections:
                seen_sections.add(sec_key)
                unique_sections.append(sec)

        grouped[course_id] = unique_sections

    # 2. Build combinations
    section_lists = [grouped[cid] for cid in sorted(grouped)]

    valid_schedules: list[list[dict[str, Any]]] = []
    # keep track of schedules that have been added, so they don't get added again
    seen_schedule_keys: set[tuple[tuple[str, str], ...]] = set()
    conflict_examples: set[str] = set()

    # give every combo for every sections
    for combo in product(*section_lists):
        current_schedule = list(combo)
        overlap_found = False

        # loop through every pair of sections in the combos
        for i in range(len(current_schedule)):
            for j in range(i + 1, len(current_schedule)):
                # stop the loop once there's a conflict, cuz that means the schedule won't work anymore
                # and take note of the conflict so that a meaningful message can be returned
                # before adding this, we didn't know what the error was: class doesn't exist, time conflict, something went wrong?
                # we could only guess 
                conflict_desc = _get_sections_conflict(current_schedule[i], current_schedule[j])
                if conflict_desc:
                    conflict_examples.add(conflict_desc)
                    overlap_found = True
                    break
            if overlap_found:
                break
        
        # this is a valid schedule if it reached this point, so add it to the list of schedules
        if not overlap_found:
            schedule_key = tuple(
                (sec["course_id"], sec["section"]) for sec in current_schedule
            )

            # don't add schedules that have already been added
            # before this, it was just maxing out the schedules even though there was only one unique schedules
            # it would duplicate it just to get the max schedules
            # this prevents that, so every schedule added is unique
            if schedule_key not in seen_schedule_keys:
                seen_schedule_keys.add(schedule_key)
                valid_schedules.append(current_schedule)

            # to ensure no more than the max schedules are generated
            if len(valid_schedules) >= max_schedules:   
                break
    
    if valid_schedules:
        valid_schedules.sort(key=_schedule_score, reverse=True)
        return {
            "ok": True,
            "schedules": valid_schedules,
            "error": None,
            "conflicts": [],
        }
    # fallback if no valid schedules were generated
    return {
        "ok": False,
        "schedules": [],
        "error": "No valid schedules found. The requested courses have overlapping times.",
        "conflicts": list(conflict_examples)[:5],
    }


# this just combines all the functions above into one thing before passing it to the 
# "official" scheduling tool
def build_schedule_result(
    desired_courses: list[str],
    max_schedules: int = 5,
) -> dict[str, Any]:
    """
    The main entry point: loads data, generates schedules, and formats the response.
    """
    try:
        # get the path and standardize everything
        csv_path = _resource_path("Course Schedules.csv")
        raw_rows = _load_sections_csv(csv_path)
        normalized_sections = [_normalize_row(row) for row in raw_rows]
        # build the lookup dictionary to enable quick matching
        sections_by_course = _build_sections_index(normalized_sections)

        # stores results of valid schedules
        res = _generate_valid_schedules(sections_by_course, desired_courses, max_schedules)

        formatted_schedules = []
        # this groups related information together, so that it's clear
        # that these particular warnings belong to this particular schedule
        # otherwise it could be a bit ambiguous what warnings belong to which section
        for s in res.get("schedules", []):
            formatted_schedules.append({
                "sections": s,
                "warnings": _schedule_warnings(s),
            })

        return {
            "requested_courses": desired_courses,
            "schedule_count": len(formatted_schedules),
            "schedules": formatted_schedules,
            "error": res.get("error"),
            "conflicts": res.get("conflicts", []),
        }

    except FileNotFoundError:
        return {
            "requested_courses": desired_courses,
            "schedule_count": 0,
            "schedules": [],
            "error": "The Course Schedules.csv file is missing from the resources folder.",
            "conflicts": [],
        }
    except Exception as e:
        return {
            "requested_courses": desired_courses,
            "schedule_count": 0,
            "schedules": [],
            "error": f"An unexpected error occurred: {str(e)}",
            "conflicts": [],
        }


# this is the tool that is actually available to the LLM
@tool
def create_schedule(
    desired_courses: list[str],
    max_schedules: int = 5,
) -> dict[str, Any]:
    """Create up to max_schedules valid, non-overlapping schedules for the requested courses.

    The tool does not exclude sections based on full/waitlist/permission status.
    Those conditions are returned as informational warnings when detected.

    Use this when the user wants an actual class schedule built from course sections.
    Course IDs should look like 'CSCI 125' or 'MATH 131'.

    Args:
        desired_courses: List of desired course IDs.
        max_schedules: Maximum number of schedules to return.

    Returns:
        A dict containing the requested courses, number of schedules found, and the top schedules.
    """
    try:
        if not desired_courses:
            return {
                "requested_courses": [],
                "schedule_count": 0,
                "schedules": [],
                "error": "No desired courses were provided.",
            }

        return build_schedule_result(
            desired_courses=desired_courses,
            max_schedules=max_schedules,
        )
    # just in case cases
    except FileNotFoundError:
        return {
            "requested_courses": desired_courses,
            "schedule_count": 0,
            "schedules": [],
            "error": "Schedule data file not found at resources/schedules/",
        }
    except Exception as e:
        return {
            "requested_courses": desired_courses,
            "schedule_count": 0,
            "schedules": [],
            "error": f"{type(e).__name__}: {e}",
        }
    
def get_major_index():
    """Central place to access major index."""
    return get_program_data()["major"]["index"]

@tool
def major_url_lookup(major: str, concentration: str = None, degree_type: str = None) -> str:
    """Get catalog URL for a major, optional concentration, and optional degree type.
    
    Args:
        major: The major name (e.g., 'biology', 'computer science')
        concentration: Optional concentration (e.g., 'acs' for Chemistry)
        degree_type: Optional degree type - 'ba' or 'bs'. If not specified, prefers bs.
    
    Returns:
        The catalog URL for the requested major/concentration/degree type.
    """
    major_index = get_major_index()

    return get_url(major_index, major, concentration, degree_type)
def scrape_tool(url: str) -> dict:
    """
    Fetches HTML and extracts program details.
    
    Key Logic:
    - User-Agent: Mimics a browser to avoid 403 Forbidden errors.
    - Noise Filter: Ignores text containing 'navigation' or 'social media' 
      to find the real description.
    - Course Regex: Scans paragraphs, lists, and tables for patterns 
      like 'CSCI 101'.
    """
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=10
        )
        response.raise_for_status()
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}

    soup = BeautifulSoup(response.text, "html.parser")

    #Content Extraction
    content = soup.find("div", id="acalog-content") or soup.find("td", class_="block_content") or soup.body

    #Regex to find couse codes
    course_code_pattern = re.compile(r"\b[A-Z]{2,4}\s?[\d]{3,4}[A-Z]?\b")
    
    course_full_pattern = re.compile(r"\b[A-Z]{2,4}\s?[\d]{3,4}[A-Z]?\b[:\-\s\.]+[A-Za-z\s]{5,50}")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown Program"

    description = None
    course_codes = set()
    course_details = []

    #Noise Filter for Description
    noise_words = {"navigation", "social media", "apply", "visit", "directory", "facebook", "twitter", "copyright"}

    #extract courses
    for element in content.find_all(["p", "li", "td", "h2", "h3"]):
        text = element.get_text(" ", strip=True)
        if not text or len(text) < 5:
            continue

        if not description and len(text.split()) > 30:
            if not any(word in text.lower() for word in noise_words):
                description = text

        # Extract Course Codes
        codes = course_code_pattern.findall(text)
        for code in codes:
            course_codes.add(code.strip())

        #Extract Full Course Titles (Optional: for more detail)
        full_matches = course_full_pattern.findall(text)
        for match in full_matches:
            if match.strip() not in course_details:
                course_details.append(match.strip())

    return {
        "title": title,
        "description": description,
        "courses": sorted(list(course_codes)),
        "course_details": course_details 
    }

def program_scraper(url: str) -> str:
    """
    Scrape a college program page using a LangGraph subgraph.
    """
    from .graph import build_scraper_graph

    scraper_graph = build_scraper_graph()
    result = scraper_graph.invoke({"url": url})
    data = result.get("result")

    if not data:
        return "Error: No data returned"

    if "error" in data:
        return f"Error: {data['error']}"

    return str(data)

def get_gen_ed_data():
    """Returns raw gen ed requirements data (also called the "anchor plan")."""
    return get_program_data()["gen_ed"]["raw"] 

@tool
def get_gen_ed_summary() -> str:
    """ Returns a human-readable summary of gen ed requirements (anchor plan).
        Use this when explaining the anchor plan to users."""
    data = get_gen_ed_data()

    total = data.get("total_credits", {})
    min_c = total.get("min", "unknown")
    max_c = total.get("max", "unknown")

    reqs = data.get("requirements", [])

    lines = [f"Gen Ed Credits: {min_c}-{max_c}\n"]

    for r in reqs:
        r_type = r.get("type", "")
        name = r.get("name", "Unnamed Requirement")

        # REQUIRED COURSES
        if r_type == "required":
            codes = [c["code"] for c in r.get("courses", [])]
            lines.append(f"{name} (Required): {', '.join(codes)}")

        # CATEGORY REQUIREMENTS (math/science/etc.)
        elif r_type in ["category", "distribution", "area"]:
            desc = r.get("description", "")
            lines.append(f"{name} ({r_type}): {desc}")

        # GROUP REQUIREMENTS
        elif r_type == "group":
            options = r.get("options", [])
            lines.append(f"{name} (Choose from {len(options)} options)")

        # fallback (don’t lose anything silently)
        else:
            lines.append(f"{name}: {r}")

    return "\n".join(lines)