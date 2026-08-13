"""User-facing message strings."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from digitex.core.domain import ExamType

MSG_REGISTRATION_INFO = (
    "📚 <b>Digitex</b> — подготовка к ЦТ/ЦЭ\n\n"
    "Доступ предоставляется после одобрения администратором."
)
MSG_ASK_NAME = "Пожалуйста, введите ваши <b>имя</b> и <b>фамилию</b>:"
MSG_PENDING = (
    "📋 <b>Статус: на рассмотрении</b>\n"
    "Отправлена: {date}\n\n"
    "Ожидайте подтверждения администратором.\n"
    "После одобрения отправьте /start чтобы начать."
)
MSG_REQUEST_SENT = (
    "✅ Спасибо, {name}! Заявка отправлена.\n\n"
    "📋 <b>Статус: на рассмотрении</b>\n"
    "Отправлена: {date}\n\n"
    "Ожидайте подтверждения администратором.\n"
    "После одобрения отправьте /start чтобы начать."
)
MSG_ADMIN_NEW_REQUEST = (
    "🆕 Новая заявка:\n"
    "Имя: <b>{full_name}</b>\n"
    "Telegram: @{username} (ID: <code>{telegram_id}</code>)"
)
MSG_ADMIN_ONLY = "Только администратор может выполнять это действие."
MSG_APPROVED_USER = "✅ Заявка подтверждена. Отправьте /start для начала."
MSG_REJECTED_USER = "❌ Ваша заявка отклонена."
MSG_APPROVED_ADMIN = "✅ Заявка {full_name} подтверждена."
MSG_REJECTED_ADMIN = "❌ Заявка {full_name} отклонена."

MSG_MODE_SELECT = "Выберите режим тестирования:"
MSG_YEAR_SELECT = "Выберите год:"
MSG_OPTION_SELECT = "Выберите вариант:"
MSG_EXAM_TYPE_SELECT = "Выберите тип экзамена:"
MSG_PART_SELECT = "Выберите часть:"
MSG_TOPIC_SELECT = "Выберите тему:"
MSG_START_TESTING = "Начинаем тестирование!"
MSG_ENTER_ANSWER = "Введите ответ:"

MSG_GREETING = "С возвращением, {name}! 👋\n📚 Выберите предмет для тестирования:"
MSG_NO_YEARS = "Нет доступных лет для этого предмета. Выберите другой предмет:"
MSG_NO_TOPICS = "Нет доступных тем для этого предмета."
MSG_NO_OPTIONS = "Нет доступных вариантов для {exam_type}. Выберите другой год."
MSG_NO_RANDOM_QUESTION = "Не удалось найти случайный вопрос."
MSG_NO_TOPIC_QUESTION = "Не удалось найти вопрос по этой теме."

MSG_CORRECT_ANSWER = "✅ Правильно!"
MSG_WRONG_ANSWER = "❌ Неправильно!\nПравильный ответ: <b>{correct_answer}</b>"
MSG_ANSWER_UNKNOWN = "неизвестен"

MSG_RANDOM_FINISH = (
    "Режим случайных вопросов завершен. Используйте /start для начала заново."
)

MSG_RESULTS_HEADER = "📊 <b>Тестирование завершено</b>"
MSG_RESULTS_SUBJECT = "<b>Предмет:</b> {subject_name}"
MSG_RESULTS_TYPE = "<b>Тип:</b> {exam_type}"
MSG_RESULTS_YEAR = "<b>Год:</b> {year}"
MSG_RESULTS_OPTION = "<b>Вариант:</b> {option_number}"
MSG_RESULTS_SCORE = "<b>Результат:</b> {total_score} из {max_score}"
MSG_RESULTS_PART_A = "├─ Часть А: {part_a_score}"
MSG_RESULTS_PART_B = "└─ Часть Б: {part_b_score}"
MSG_RESULTS_TIME = "<b>Время:</b> {time_spent:.0f} сек"
MSG_RESULTS_ERRORS = "<b>Ошибки:</b>"
MSG_RESULTS_PART_A_H = "<b>Часть А:</b>"
MSG_RESULTS_PART_B_H = "<b>Часть Б:</b>"
MSG_RESULTS_ERROR_ITEM = "  • Вопрос {qnum}: ваш ответ <code>{user_ans}</code>, правильный <code>{correct_ans}</code>"
MSG_RESULTS_RETRY = "Выберите предмет для нового тестирования:"

MSG_EXAM_CE = "ЦЭ"
MSG_EXAM_CT = "ЦТ"

# One label per ExamType, for keyboards and captions alike.
EXAM_LABELS: dict[ExamType, str] = {"CE": MSG_EXAM_CE, "CT": MSG_EXAM_CT}

MSG_KB_STANDARD = "Стандартный режим"
MSG_KB_RANDOM = "Случайные вопросы"
MSG_KB_TOPICS = "Темы"
MSG_KB_PART_A = "Часть A"
MSG_KB_PART_B = "Часть B"
MSG_KB_NEXT = "Следующий вопрос"
MSG_KB_FINISH = "Завершить"
MSG_KB_APPROVE = "✅ Подтвердить"
MSG_KB_REJECT = "❌ Отклонить"

MSG_OPTION_PREFIX = "Вариант"

MSG_RANDOM_ORIGIN = (
    "<tg-spoiler>{exam_label} {year} год, вариант {option_number}</tg-spoiler>"
)
MSG_RANDOM_TOPIC = "Тема: {topic_name}\n{origin}"

MSG_HELP = (
    "📚 <b>Digitex Telegram Bot</b>\n\n"
    "Бот для подготовки к ЦТ/ЦЭ.\n\n"
    "<b>Режимы:</b>\n"
    "• <b>Стандартный</b> — полный тест по году и варианту\n"
    "• <b>Случайные вопросы</b> — случайные вопросы по типу экзамена\n"
    "• <b>Темы</b> — вопросы по конкретной теме\n\n"
    "<b>Команды:</b>\n"
    "/start — выбрать предмет\n"
    "/help — помощь"
)

CMD_START_DESC = "Начать / выбрать предмет"
CMD_HELP_DESC = "Помощь"


def format_answer(answer: int | str | None) -> str:
    """Render a stored answer key for display.

    A question whose year shipped without an answer key has none to name, and
    both the random-mode verdict and the results screen have to say so — so the
    wording for that case lives here once.
    """
    return MSG_ANSWER_UNKNOWN if answer is None else str(answer)
