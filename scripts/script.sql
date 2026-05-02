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
    a_num_options INTEGER NOT NULL DEFAULT 5,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
    UNIQUE (subject_id, year_value)
);

-- Options / variants (up to 10 per book)
CREATE TABLE IF NOT EXISTS options (
    option_id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL,
    option_number INTEGER NOT NULL CHECK (option_number BETWEEN 1 AND 10),
    exam_type TEXT NOT NULL DEFAULT 'CT' CHECK (exam_type IN ('CE', 'CT')),
    FOREIGN KEY (book_id) REFERENCES books(book_id),
    UNIQUE (book_id, option_number)
);

-- Part A questions (multiple choice, answer is 1..a_num_options)
CREATE TABLE IF NOT EXISTS part_a_questions (
    question_id INTEGER PRIMARY KEY,
    option_id INTEGER NOT NULL,
    question_number INTEGER NOT NULL,
    answer INTEGER NOT NULL CHECK (answer BETWEEN 1 AND 5),
    FOREIGN KEY (option_id) REFERENCES options(option_id),
    UNIQUE (option_id, question_number)
);

-- Part B questions (free-form text answer)
CREATE TABLE IF NOT EXISTS part_b_questions (
    question_id INTEGER PRIMARY KEY,
    option_id INTEGER NOT NULL,
    question_number INTEGER NOT NULL,
    answer TEXT NOT NULL,
    FOREIGN KEY (option_id) REFERENCES options(option_id),
    UNIQUE (option_id, question_number)
);

-- Images (one per question in current extraction, structure allows more)
CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL,
    part TEXT NOT NULL,
    image_data BLOB NOT NULL,
    telegram_file_id TEXT,
    UNIQUE (question_id, part)
);

-- Students (Telegram users)
CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY,
    telegram_id INTEGER NOT NULL UNIQUE,
    name TEXT NOT NULL,
    username TEXT,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Active or completed test sessions
CREATE TABLE IF NOT EXISTS test_sessions (
    session_id INTEGER PRIMARY KEY,
    student_id INTEGER NOT NULL,
    option_id INTEGER NOT NULL,
    started_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME,
    FOREIGN KEY (student_id) REFERENCES students(student_id),
    FOREIGN KEY (option_id) REFERENCES options(option_id)
);

-- Topic-to-question mappings (loaded from topic_to_year.json)
CREATE TABLE IF NOT EXISTS question_topics (
    question_id INTEGER NOT NULL,
    part TEXT NOT NULL CHECK (part IN ('A', 'B')),
    topic_name TEXT NOT NULL,
    PRIMARY KEY (question_id, part, topic_name)
);

-- Per-question answers recorded during a session
CREATE TABLE IF NOT EXISTS session_answers (
    answer_id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL,
    question_id INTEGER NOT NULL,
    student_answer TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL CHECK (is_correct IN (0, 1)),
    time_spent REAL NOT NULL,
    answered_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES test_sessions(session_id),
    UNIQUE (session_id, question_id)
);

-- Authorized users (registration / approval flow)
CREATE TABLE IF NOT EXISTS authorized_users (
    telegram_id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    telegram_username TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    handled_at DATETIME,
    handled_by INTEGER
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_books_subject_year ON books(subject_id, year_value);
CREATE INDEX IF NOT EXISTS idx_options_book_exam ON options(book_id, exam_type);
CREATE INDEX IF NOT EXISTS idx_options_exam_type ON options(exam_type);
CREATE INDEX IF NOT EXISTS idx_question_topics_topic ON question_topics(topic_name);
CREATE INDEX IF NOT EXISTS idx_session_answers_session ON session_answers(session_id);
CREATE INDEX IF NOT EXISTS idx_session_answers_correct ON session_answers(session_id, is_correct);
