import sqlite3

DATABASE_PATH = "data/tests.db"
SQL_SCRIPT_PATH = "script.sql"


def main():
    with open(SQL_SCRIPT_PATH, 'r') as file:
        creation_queries = file.read()

    connection = sqlite3.connect(DATABASE_PATH)
    cursor = connection.cursor()
    cursor.executescript(creation_queries)
    connection.commit()
    connection.close()


if __name__ == "__main__":
    main()
