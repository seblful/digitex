-- Subjects
CREATE TABLE subjects (
    subject_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

-- Books (one per subject per year)
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL,
    year_value INTEGER NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
    UNIQUE (subject_id, year_value)
);

-- Options / variants (up to 10 per book)
CREATE TABLE options (
    option_id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL,
    option_number INTEGER NOT NULL CHECK (option_number BETWEEN 1 AND 10),
    FOREIGN KEY (book_id) REFERENCES books(book_id),
    UNIQUE (book_id, option_number)
);

-- Parts (A and B for each option)
CREATE TABLE parts (
    part_id INTEGER PRIMARY KEY,
    option_id INTEGER NOT NULL,
    part_type TEXT NOT NULL CHECK (part_type IN ('A', 'B')),
    FOREIGN KEY (option_id) REFERENCES options(option_id),
    UNIQUE (option_id, part_type)
);

-- Questions (image-only, no text)
CREATE TABLE questions (
    question_id INTEGER PRIMARY KEY,
    part_id INTEGER NOT NULL,
    question_number INTEGER NOT NULL,
    specification TEXT,
    FOREIGN KEY (part_id) REFERENCES parts(part_id),
    UNIQUE (part_id, question_number)
);

-- Correct answer for Part A
CREATE TABLE part_a_answers (
    question_id INTEGER PRIMARY KEY,
    correct_order INTEGER NOT NULL CHECK (correct_order BETWEEN 1 AND 5),
    FOREIGN KEY (question_id) REFERENCES questions(question_id)
);

-- Correct answer for Part B (exactly one per question)
CREATE TABLE part_b_answers (
    question_id INTEGER PRIMARY KEY,
    answer_text TEXT NOT NULL,
    FOREIGN KEY (question_id) REFERENCES questions(question_id)
);

-- Images (question images and table images)
CREATE TABLE images (
    image_id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL,
    image_data BLOB NOT NULL,
    telegram_file_id TEXT,              -- cached after first Telegram upload
    is_table BOOLEAN NOT NULL DEFAULT 0 CHECK (is_table IN (0, 1)),
    image_order INTEGER NOT NULL,
    FOREIGN KEY (question_id) REFERENCES questions(question_id),
    UNIQUE (question_id, image_order)
);

-- Students (Telegram users)
CREATE TABLE students (
    student_id INTEGER PRIMARY KEY,
    telegram_id INTEGER NOT NULL UNIQUE,
    name TEXT NOT NULL,
    username TEXT,                      -- Telegram @username, can change
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Active or completed test sessions
CREATE TABLE test_sessions (
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
CREATE TABLE session_answers (
    answer_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    question_id INTEGER NOT NULL,
    student_answer TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL CHECK (is_correct IN (0, 1)),
    time_spent REAL NOT NULL,           -- seconds
    answered_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES test_sessions(session_id),
    FOREIGN KEY (question_id) REFERENCES questions(question_id),
    UNIQUE (session_id, question_id)    -- one answer per question per session
);
