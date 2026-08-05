"""Answer-checking logic for exam questions."""

from typing import Literal


def check_answer(
    part: Literal["A", "B"], student_answer: str, correct_answer: int | str
) -> bool:
    """Return True if the student's answer matches the correct answer.

    Part A compares an integer option index; Part B allows multiple correct
    values separated by "/" (e.g. "ANS1/ANS2").

    A Part B question with no stored answer matches nothing — a blank reply is
    not a correct one. ``populate_db`` relies on that to load a Question whose
    answer key is missing without ever scoring it right.
    """
    if part == "A":
        return int(student_answer.strip()) == int(correct_answer)
    correct_options = [
        opt.strip() for opt in str(correct_answer).split("/") if opt.strip()
    ]
    return bool(correct_options) and student_answer.strip() in correct_options
