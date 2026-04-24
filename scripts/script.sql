-- Subjects
CREATE TABLE IF NOT EXISTS subjects (
    subject_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

-- Books (one per subject per year)
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL,
    year_value INTEGER NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
    UNIQUE (subject_id, year_value)
);

-- Options / variants (up to 10 per book)
CREATE TABLE IF NOT EXISTS options (
    option_id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL,
    option_number INTEGER NOT NULL CHECK (option_number BETWEEN 1 AND 10),
    FOREIGN KEY (book_id) REFERENCES books(book_id),
    UNIQUE (book_id, option_number)
);

-- Questions — part and number mirror filesystem: {option}/{part}/{number}.jpg
CREATE TABLE IF NOT EXISTS questions (
    question_id INTEGER PRIMARY KEY,
    option_id INTEGER NOT NULL,
    part TEXT NOT NULL CHECK (part IN ('A', 'B')),
    question_number INTEGER NOT NULL,
    FOREIGN KEY (option_id) REFERENCES options(option_id),
    UNIQUE (option_id, part, question_number)
);

-- Correct answer for Part A (1–5, matches extracted answer digit)
CREATE TABLE IF NOT EXISTS part_a_answers (
    question_id INTEGER PRIMARY KEY,
    correct_order INTEGER NOT NULL CHECK (correct_order BETWEEN 1 AND 5),
    FOREIGN KEY (question_id) REFERENCES questions(question_id)
);

-- Correct answer for Part B (text, e.g. "ВЕРНАДСКИЙ", "А4Б2В1Г3")
CREATE TABLE IF NOT EXISTS part_b_answers (
    question_id INTEGER PRIMARY KEY,
    answer_text TEXT NOT NULL,
    FOREIGN KEY (question_id) REFERENCES questions(question_id)
);

-- Images (one per question in current extraction, structure allows more)
CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL,
    image_data BLOB NOT NULL,
    telegram_file_id TEXT,              -- cached after first Telegram upload
    image_order INTEGER NOT NULL,
    FOREIGN KEY (question_id) REFERENCES questions(question_id),
    UNIQUE (question_id, image_order)
);

-- Students (Telegram users)
CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY,
    telegram_id INTEGER NOT NULL UNIQUE,
    name TEXT NOT NULL,
    username TEXT,                      -- Telegram @username, can change
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Active or completed test sessions
CREATE TABLE IF NOT EXISTS test_sessions (
    session_id INTEGER PRIMARY KEY,
    student_id INTEGER NOT NULL,
    book_id INTEGER NOT NULL,
    option_number INTEGER NOT NULL CHECK (option_number BETWEEN 1 AND 10),
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,              -- NULL while session is in progress
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (book_id) REFERENCES books(book_id)
);

-- Per-question answers recorded during a session
CREATE TABLE IF NOT EXISTS session_answers (
    answer_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    question_id INTEGER NOT NULL,
    student_answer TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL CHECK (is_correct IN (0, 1)),
    time_spent REAL NOT NULL,           -- seconds
    answered_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES test_sessions(session_id),
    FOREIGN KEY (question_id) REFERENCES questions(question_id),
    UNIQUE (session_id, question_id)
);
