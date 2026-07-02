import sqlite3
import json
import os
import gzip
import sys
from pathlib import Path
from typing import Iterator, Optional


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = ROOT_DIR / "redrob_data_sample.db"
DEFAULT_INPUT_PATH = ROOT_DIR / "sample_candidates.json"
BATCH_SIZE = 2000


def setup_database(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create a SQLite database and the base candidates table."""
    db_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            name TEXT,
            headline TEXT,
            summary TEXT,
            location TEXT,
            country TEXT,
            years_of_experience REAL,
            current_title TEXT,
            current_company TEXT,
            current_company_size TEXT,
            current_industry TEXT,
            career_history_count INTEGER,
            education_count INTEGER,
            skill_count INTEGER,
            notice_period_days INTEGER,
            recruiter_response_rate REAL,
            last_active_date TEXT,
            profile_completeness_score REAL,
            github_activity_score REAL,
            saved_by_recruiters_30d INTEGER,
            interview_completion_rate REAL,
            offer_acceptance_rate REAL,
            verified_email INTEGER,
            verified_phone INTEGER,
            linkedin_connected INTEGER,
            open_to_work_flag INTEGER,
            willing_to_relocate INTEGER,
            profile_json TEXT,
            career_history_json TEXT,
            education_json TEXT,
            skills_json TEXT,
            languages_json TEXT,
            redrob_signals_json TEXT,
            full_record_json TEXT
        )
        """
    )
    conn.commit()
    return conn


def iter_candidate_records(input_path: Path) -> Iterator[dict]:
    """Yield one candidate at a time from .jsonl, .jsonl.gz, or .json files."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix == ".gz":
        with gzip.open(input_path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if line:
                    yield json.loads(line)
    elif input_path.suffix == ".json":
        with open(input_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            if isinstance(payload, list):
                for item in payload:
                    yield item
            else:
                yield payload
    else:
        with open(input_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if line:
                    yield json.loads(line)


def extract_candidate_fields(candidate: dict) -> tuple:
    """Flatten the nested JSON into a simple tuple for SQLite insertion."""
    profile = candidate.get("profile") or {}
    signals = candidate.get("redrob_signals") or {}
    career_history = candidate.get("career_history") or []
    education = candidate.get("education") or []
    skills = candidate.get("skills") or []
    languages = candidate.get("languages") or []

    return (
        candidate.get("candidate_id"),
        profile.get("anonymized_name"),
        profile.get("headline"),
        profile.get("summary"),
        profile.get("location"),
        profile.get("country"),
        profile.get("years_of_experience"),
        profile.get("current_title"),
        profile.get("current_company"),
        profile.get("current_company_size"),
        profile.get("current_industry"),
        len(career_history),
        len(education),
        len(skills),
        signals.get("notice_period_days"),
        signals.get("recruiter_response_rate"),
        signals.get("last_active_date"),
        signals.get("profile_completeness_score"),
        signals.get("github_activity_score"),
        signals.get("saved_by_recruiters_30d"),
        signals.get("interview_completion_rate"),
        signals.get("offer_acceptance_rate"),
        int(signals.get("verified_email") is True),
        int(signals.get("verified_phone") is True),
        int(signals.get("linkedin_connected") is True),
        int(signals.get("open_to_work_flag") is True),
        int(signals.get("willing_to_relocate") is True),
        json.dumps(profile, ensure_ascii=False),
        json.dumps(career_history, ensure_ascii=False),
        json.dumps(education, ensure_ascii=False),
        json.dumps(skills, ensure_ascii=False),
        json.dumps(languages, ensure_ascii=False),
        json.dumps(signals, ensure_ascii=False),
        json.dumps(candidate, ensure_ascii=False),
    )


def ingest_data(conn: sqlite3.Connection, input_path: Optional[Path] = None, batch_size: int = BATCH_SIZE) -> int:
    """Stream a JSONL file into SQLite in batches to keep memory usage low."""
    input_path = input_path or DEFAULT_INPUT_PATH
    input_path = Path(input_path)

    cursor = conn.cursor()
    batch = []
    inserted_count = 0

    for candidate in iter_candidate_records(input_path):
        batch.append(extract_candidate_fields(candidate))
        if len(batch) >= batch_size:
            cursor.executemany(
                """
                INSERT OR REPLACE INTO candidates (
                    candidate_id, name, headline, summary, location, country,
                    years_of_experience, current_title, current_company,
                    current_company_size, current_industry, career_history_count,
                    education_count, skill_count, notice_period_days,
                    recruiter_response_rate, last_active_date,
                    profile_completeness_score, github_activity_score,
                    saved_by_recruiters_30d, interview_completion_rate,
                    offer_acceptance_rate, verified_email, verified_phone,
                    linkedin_connected, open_to_work_flag, willing_to_relocate,
                    profile_json, career_history_json, education_json,
                    skills_json, languages_json, redrob_signals_json,
                    full_record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch,
            )
            conn.commit()
            inserted_count += len(batch)
            batch.clear()

    if batch:
        cursor.executemany(
            """
            INSERT OR REPLACE INTO candidates (
                candidate_id, name, headline, summary, location, country,
                years_of_experience, current_title, current_company,
                current_company_size, current_industry, career_history_count,
                education_count, skill_count, notice_period_days,
                recruiter_response_rate, last_active_date,
                profile_completeness_score, github_activity_score,
                saved_by_recruiters_30d, interview_completion_rate,
                offer_acceptance_rate, verified_email, verified_phone,
                linkedin_connected, open_to_work_flag, willing_to_relocate,
                profile_json, career_history_json, education_json,
                skills_json, languages_json, redrob_signals_json,
                full_record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            batch,
        )
        conn.commit()
        inserted_count += len(batch)

    print(f"Successfully inserted {inserted_count} candidates into {DEFAULT_DB_PATH}")
    return inserted_count


def verify_data(conn: sqlite3.Connection, limit: int = 5) -> None:
    """Run a small verification query against the database."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT candidate_id, current_title, recruiter_response_rate, last_active_date
        FROM candidates
        ORDER BY recruiter_response_rate DESC, years_of_experience DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = cursor.fetchall()
    print("\nSample rows from database:")
    for row in rows:
        print(row)


if __name__ == "__main__":
    print("Starting Day 1 data ingestion...")

    input_path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    db_path_arg = sys.argv[2] if len(sys.argv) > 2 else None

    if input_path_arg:
        input_path = Path(input_path_arg)
        if not input_path.is_absolute():
            input_path = (ROOT_DIR / input_path).resolve()
    else:
        input_path = ROOT_DIR / "candidates.jsonl"
        if not input_path.exists():
            input_path = ROOT_DIR / "sample_candidates.json"

    if db_path_arg:
        db_path = Path(db_path_arg)
        if not db_path.is_absolute():
            db_path = (ROOT_DIR / db_path).resolve()
    else:
        db_path = DEFAULT_DB_PATH

    db_connection = setup_database(db_path)
    count = ingest_data(db_connection, input_path=input_path)
    verify_data(db_connection)
    db_connection.close()
    print("\nProcess complete.")