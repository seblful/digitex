-- Subjects table to store different subjects
CREATE TABLE subjects (
    subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_name TEXT NOT NULL
);

-- Collections (types) table - 2 types per subject
CREATE TABLE collections (
    collection_id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL,
    collection_name TEXT NOT NULL,
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

-- Years table for different years in collections
CREATE TABLE years (
    year_id INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id INTEGER NOT NULL,
    year_number INTEGER NOT NULL,
    FOREIGN KEY (collection_id) REFERENCES collections(collection_id)
);

-- Options table for different options in years
CREATE TABLE options (
    option_id INTEGER PRIMARY KEY AUTOINCREMENT,
    year_id INTEGER NOT NULL,
    option_number INTEGER NOT NULL,
    FOREIGN KEY (year_id) REFERENCES years(year_id)
);

-- Parts table - 2 parts per option
CREATE TABLE parts (
    part_id INTEGER PRIMARY KEY AUTOINCREMENT,
    option_id INTEGER NOT NULL,
    part_number INTEGER NOT NULL,
    answer_type TEXT NOT NULL CHECK (answer_type IN ('multiple_choice', 'single_answer')),
    FOREIGN KEY (option_id) REFERENCES options(option_id)
);

-- Questions table
CREATE TABLE questions (
    question_id INTEGER PRIMARY KEY AUTOINCREMENT,
    part_id INTEGER NOT NULL,
    question_number INTEGER NOT NULL,
    question_text TEXT NOT NULL,
    specification TEXT,
    single_answer TEXT, -- For questions with single correct answer
    FOREIGN KEY (part_id) REFERENCES parts(part_id)
);

-- Image types table
CREATE TABLE image_types (
    type_id INTEGER PRIMARY KEY AUTOINCREMENT,
    type_name TEXT NOT NULL UNIQUE
);

-- Images table
CREATE TABLE images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL,
    type_id INTEGER NOT NULL,
    image_data BLOB NOT NULL,
    image_order INTEGER NOT NULL, -- To maintain order of multiple images
    FOREIGN KEY (question_id) REFERENCES questions(question_id),
    FOREIGN KEY (type_id) REFERENCES image_types(type_id)
);

-- Question options table (for multiple choice questions)
CREATE TABLE question_options (
    option_id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL,
    option_text TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL DEFAULT 0,
    FOREIGN KEY (question_id) REFERENCES questions(question_id)
);

-- Create indexes for better performance
CREATE INDEX idx_collections_subject ON collections(subject_id);
CREATE INDEX idx_years_collection ON years(collection_id);
CREATE INDEX idx_options_year ON options(year_id);
CREATE INDEX idx_parts_option ON parts(option_id);
CREATE INDEX idx_questions_part ON questions(part_id);
CREATE INDEX idx_images_question ON images(question_id);
CREATE INDEX idx_images_type ON images(type_id);
CREATE INDEX idx_question_options_question ON question_options(question_id);

-- Add unique constraints
CREATE UNIQUE INDEX idx_unique_collection 
ON collections(subject_id, collection_name);

CREATE UNIQUE INDEX idx_unique_year 
ON years(collection_id, year_number);

CREATE UNIQUE INDEX idx_unique_option 
ON options(year_id, option_number);

CREATE UNIQUE INDEX idx_unique_part 
ON parts(option_id, part_number);

CREATE UNIQUE INDEX idx_unique_question 
ON questions(part_id, question_number);

CREATE UNIQUE INDEX idx_unique_image_order
ON images(question_id, type_id, image_order);

-- Insert some common image types
INSERT INTO image_types (type_name) VALUES 
('question_image'),
('solution_image'),
('table_image');