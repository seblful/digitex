import sqlite3

from modules.config import get_settings

SQL_SCRIPT_PATH = "script.sql"


def main():
    settings = get_settings()
    with open(SQL_SCRIPT_PATH, 'r') as file:
        creation_queries = file.read()

    connection = sqlite3.connect(settings.database.path)
    cursor = connection.cursor()
    cursor.executescript(creation_queries)
    connection.commit()
    connection.close()


if __name__ == "__main__":
    main()
