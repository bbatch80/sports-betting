"""
SQL validation for LLM-generated queries.

Defense-in-depth: validates that generated SQL is read-only before execution.
This is one of several layers — the database transaction is also set to READ ONLY.
"""

import re
from typing import Tuple


# Keywords that indicate mutation — reject if found
_MUTATION_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXEC|EXECUTE|GRANT|REVOKE|"
    r"MERGE|REPLACE|CALL|COPY|LOAD|VACUUM|REINDEX|CLUSTER|LOCK|UNLOCK|SET\s+ROLE|"
    r"BEGIN|COMMIT|ROLLBACK|SAVEPOINT)\b",
    re.IGNORECASE,
)

# SQL comments — block as injection vectors
_SQL_COMMENTS = re.compile(r"(--|/\*)")

# Multi-statement detection — semicolon followed by another statement
_MULTI_STATEMENT = re.compile(r";\s*\S")

# Valid statement starts
_VALID_STARTS = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)

# Existing LIMIT clause
_HAS_LIMIT = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)


def validate_sql(sql: str) -> Tuple[bool, str]:
    """
    Validate that SQL is safe to execute (SELECT-only).

    Returns:
        (is_valid, error_message) — error_message is empty string if valid.
    """
    if not sql or not sql.strip():
        return False, "Empty SQL query"

    stripped = sql.strip().rstrip(";").strip()

    if not _VALID_STARTS.match(stripped):
        return False, "Query must start with SELECT or WITH"

    if _MUTATION_KEYWORDS.search(stripped):
        return False, "Query contains disallowed keywords"

    if _SQL_COMMENTS.search(stripped):
        return False, "Query contains SQL comments"

    if _MULTI_STATEMENT.search(stripped):
        return False, "Multiple statements not allowed"

    return True, ""


def enforce_row_limit(sql: str, max_rows: int = 50) -> str:
    """Add LIMIT clause if missing."""
    stripped = sql.strip().rstrip(";").strip()

    if not _HAS_LIMIT.search(stripped):
        stripped = f"{stripped}\nLIMIT {max_rows}"

    return stripped
