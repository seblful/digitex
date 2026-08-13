"""Answer-checking logic for exam questions."""

from typing import Literal


def check_answer(
    part: Literal["A", "B"], student_answer: str, correct_answer: int | str | None
) -> bool:
    """Return True if the student's answer matches the correct answer.

    Part A compares an integer option index; Part B allows multiple correct
    values separated by "/" (e.g. "ANS1/ANS2").

    A question with no stored answer key — ``correct_answer`` None — matches
    nothing, in either part. ``populate_db`` loads a Question whose key is
    missing so that its image is servable, and this is what keeps such a
    question from ever being scored right.
    """
    if correct_answer is None:
        return False
    if part == "A":
        return int(student_answer.strip()) == int(correct_answer)
    correct_options = [
        opt.strip() for opt in str(correct_answer).split("/") if opt.strip()
    ]
    return bool(correct_options) and student_answer.strip() in correct_options
