"""Answer-checking logic for exam questions."""

from typing import Literal


def check_answer(
    part: Literal["A", "B"], student_answer: str, correct_answer: int | str
) -> bool:
    """Return True if the student's answer matches the correct answer.

    Part A compares an integer option index; Part B allows multiple correct
    values separated by "/" (e.g. "ANS1/ANS2").
    """
    if part == "A":
        return int(student_answer.strip()) == int(correct_answer)
    correct_options = [opt.strip() for opt in str(correct_answer).split("/")]
    return student_answer.strip() in correct_options
