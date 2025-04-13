-- Subjects table
CREATE TABLE subjects (
    subject_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

-- Years table
CREATE TABLE years (
    year_id INTEGER PRIMARY KEY,
    subject_id INTEGER NOT NULL,
    year_value INTEGER NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id),
    UNIQUE (subject_id, year_value)
);

-- Types table (2 types per year)
CREATE TABLE types (
    type_id INTEGER PRIMARY KEY,
    year_id INTEGER NOT NULL,
    type_number INTEGER CHECK (type_number IN (1, 2)),
    FOREIGN KEY (year_id) REFERENCES years(year_id),
    UNIQUE (year_id, type_number)
);

-- Options table (up to 10 per type)
CREATE TABLE options (
    option_id INTEGER PRIMARY KEY,
    type_id INTEGER NOT NULL,
    option_number INTEGER CHECK (option_number BETWEEN 1 AND 10),
    FOREIGN KEY (type_id) REFERENCES types(type_id),
    UNIQUE (type_id, option_number)
);

-- Parts table (A and B for each option)
CREATE TABLE parts (
    part_id INTEGER PRIMARY KEY,
    option_id INTEGER NOT NULL,
    part_type TEXT CHECK (part_type IN ('A', 'B')),
    FOREIGN KEY (option_id) REFERENCES options(option_id),
    UNIQUE (option_id, part_type)
);

-- Questions table
CREATE TABLE questions (
    question_id INTEGER PRIMARY KEY,
    part_id INTEGER NOT NULL,
    question_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    specification TEXT,
    FOREIGN KEY (part_id) REFERENCES parts(part_id),
    UNIQUE (part_id, question_number)
);

-- Question options (for part A)
CREATE TABLE question_options (
    question_option_id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL,
    option_text TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL,
    display_order INTEGER NOT NULL,
    FOREIGN KEY (question_id) REFERENCES questions(question_id),
    UNIQUE (question_id, display_order)
);

-- Answers (for part B)
CREATE TABLE answers (
    answer_id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL,
    answer_text TEXT NOT NULL,
    FOREIGN KEY (question_id) REFERENCES questions(question_id)
);

-- Images table
CREATE TABLE images (
    image_id INTEGER PRIMARY KEY,
    question_id INTEGER NOT NULL,
    image_data BLOB NOT NULL,      -- Changed from image_path to image_data
    is_table BOOLEAN NOT NULL,     -- Changed from image_type to is_table
    image_order INTEGER NOT NULL,
    FOREIGN KEY (question_id) REFERENCES questions(question_id),
    UNIQUE (question_id, image_order),
    CHECK (is_table IN (0, 1))     -- Ensures boolean values (0 = common, 1 = table)
);