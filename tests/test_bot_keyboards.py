"""Tests for bot keyboard builders."""

from digitex.bot.keyboards import (
    exam_type_kb,
    mode_kb,
    options_kb,
    part_a_kb,
    random_feedback_kb,
    random_part_kb,
    subjects_kb,
    topics_kb,
    years_kb,
)


class TestSubjectsKeyboard:
    def test_builds_keyboard_with_subjects(self) -> None:
        subjects = [(1, "Биология"), (2, "Математика")]
        markup = subjects_kb(subjects)
        buttons = markup.inline_keyboard
        assert len(buttons) == 2
        assert buttons[0][0].text == "Биология"
        assert buttons[0][0].callback_data == "subj:1"
        assert buttons[1][0].text == "Математика"
        assert buttons[1][0].callback_data == "subj:2"

    def test_empty_list_returns_empty_keyboard(self) -> None:
        markup = subjects_kb([])
        assert markup.inline_keyboard == []


class TestYearsKeyboard:
    def test_builds_keyboard_with_years(self) -> None:
        markup = years_kb([2023, 2024, 2025])
        buttons = markup.inline_keyboard
        total = sum(len(row) for row in buttons)
        assert total == 3
        assert buttons[0][0].text == "2023"
        assert buttons[0][0].callback_data == "year:2023"

    def test_empty_years(self) -> None:
        markup = years_kb([])
        assert markup.inline_keyboard == []


class TestOptionsKeyboard:
    def test_builds_keyboard_with_options(self) -> None:
        markup = options_kb([1, 2, 3])
        buttons = markup.inline_keyboard
        total = sum(len(row) for row in buttons)
        assert total == 3
        assert buttons[0][0].callback_data == "opt:1"


class TestPartAKeyboard:
    def test_default_num_options(self) -> None:
        markup = part_a_kb()
        buttons = markup.inline_keyboard
        total = sum(len(row) for row in buttons)
        assert total == 5
        for row in buttons:
            for btn in row:
                assert btn.callback_data.startswith("ans:")

    def test_custom_num_options(self) -> None:
        markup = part_a_kb(3)
        buttons = markup.inline_keyboard
        total = sum(len(row) for row in buttons)
        assert total == 3


class TestModeKeyboard:
    def test_builds_mode_keyboard(self) -> None:
        markup = mode_kb()
        buttons = markup.inline_keyboard
        assert len(buttons) == 3
        assert buttons[0][0].callback_data == "mode:standard"
        assert buttons[1][0].callback_data == "mode:random"
        assert buttons[2][0].callback_data == "mode:topics"


class TestRandomFeedbackKeyboard:
    def test_builds_feedback_keyboard(self) -> None:
        markup = random_feedback_kb()
        buttons = markup.inline_keyboard
        assert len(buttons) == 2
        assert buttons[0][0].callback_data == "random:next"
        assert buttons[1][0].callback_data == "random:finish"


class TestExamTypeKeyboard:
    def test_builds_exam_type_keyboard(self) -> None:
        markup = exam_type_kb()
        buttons = markup.inline_keyboard
        total = sum(len(row) for row in buttons)
        assert total == 2
        assert buttons[0][0].callback_data == "exam_type:CE"
        assert buttons[0][1].callback_data == "exam_type:CT"


class TestRandomPartKeyboard:
    def test_builds_random_part_keyboard(self) -> None:
        markup = random_part_kb()
        buttons = markup.inline_keyboard
        assert len(buttons) == 2
        assert buttons[0][0].callback_data == "random_part:A"
        assert buttons[1][0].callback_data == "random_part:B"


class TestTopicsKeyboard:
    def test_builds_topics_keyboard(self) -> None:
        topics = ["Клетка", "Генетика", "Эволюция"]
        markup = topics_kb(topics)
        buttons = markup.inline_keyboard
        assert len(buttons) == 3
        assert buttons[0][0].text == "Клетка"
        assert buttons[0][0].callback_data == "topic:0"
        assert buttons[2][0].callback_data == "topic:2"

    def test_empty_topics(self) -> None:
        markup = topics_kb([])
        assert markup.inline_keyboard == []
