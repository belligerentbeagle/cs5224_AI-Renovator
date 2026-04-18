#!/usr/bin/env python3
"""Apply the PostgreSQL schema in schema.sql to the configured database.

This is a lightweight alternative to invoking `psql` directly, which is handy
for local machines that do not have the PostgreSQL CLI installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

REQUIRED_TABLES = {
    "users",
    "products",
    "projects",
    "photos",
    "design_generations",
    "generation_products",
}


def _existing_tables(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            select table_name
            from information_schema.tables
            where table_schema = 'public'
            """
        )
        return {row[0] for row in cur.fetchall()}


def main() -> int:
    load_dotenv(".env")

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL is not set in .env")
        return 1

    schema_path = Path("schema.sql")
    if not schema_path.exists():
        print("ERROR: schema.sql not found")
        return 1

    sql = schema_path.read_text()

    try:
        with psycopg2.connect(database_url) as conn:
            existing = _existing_tables(conn)
            if REQUIRED_TABLES.issubset(existing):
                print("Schema already present; nothing to apply.")
                return 0
            with conn.cursor() as cur:
                cur.execute(sql)
        print("Schema applied successfully.")
        return 0
    except Exception as exc:
        print(f"ERROR applying schema: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
