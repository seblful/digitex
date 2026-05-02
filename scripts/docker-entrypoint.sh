#!/bin/bash
set -e

DB_FILE="${DB_PATH:-/app/data/production.db}"

if [ ! -f "$DB_FILE" ]; then
    echo "Database not found at $DB_FILE"
    mkdir -p "$(dirname "$DB_FILE")"

    if [ -f /app/seed/seed.db ]; then
        echo "Seeding from /app/seed/seed.db ($(du -h /app/seed/seed.db | cut -f1)) ..."
        cp /app/seed/seed.db "$DB_FILE"
        echo "Database seeded."
    else
        echo "No seed found. Creating schema..."
        python -c "
import sqlite3
conn = sqlite3.connect('$DB_FILE')
conn.executescript(open('/app/scripts/script.sql').read())
conn.commit()
conn.close()
"
        echo "Schema initialized (empty tables)."
    fi
fi

exec python -m digitex.cli.bot
