"""
LLM client for text-to-SQL pipeline via AWS Bedrock.

Uses the Bedrock `converse` API which is model-agnostic — same request format
works for Claude, Llama, Mistral, etc. Swap models by changing LLM_MODEL_ID env var.
"""

import os
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.ai.sql_guard import validate_sql, enforce_row_limit
from src.ai.schema_prompt import SYSTEM_PROMPT_SQL, SYSTEM_PROMPT_ANSWER

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """
    Model-agnostic config loaded from environment variables.

    Environment variables:
        LLM_MODEL_ID: Bedrock model ID (default: Claude 3.5 Haiku)
        AWS_REGION: AWS region for Bedrock (default: us-east-1)
    """
    model_id: str = ""
    aws_region: str = ""
    max_tokens_sql: int = 500
    max_tokens_answer: int = 1500
    max_result_rows: int = 50

    def __post_init__(self):
        if not self.model_id:
            self.model_id = os.getenv(
                "LLM_MODEL_ID",
                "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            )
        if not self.aws_region:
            self.aws_region = os.getenv("AWS_REGION", "us-east-1")

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create config from environment variables."""
        return cls()


@dataclass
class QueryResult:
    """Result of a natural language query."""
    answer: str
    sql: Optional[str] = None
    dataframe: Optional[pd.DataFrame] = None
    error: Optional[str] = None


def _get_bedrock_client(config: LLMConfig):
    """Lazy-create Bedrock client."""
    import boto3
    return boto3.client(
        "bedrock-runtime",
        region_name=config.aws_region,
    )


def _call_llm(
    client,
    model_id: str,
    system_prompt: str,
    messages: list,
    max_tokens: int,
) -> str:
    """
    Call Bedrock's converse API.

    Uses `converse` (not `invoke_model`) because it's model-agnostic —
    same request format works for Claude, Llama, Mistral. Swap model by
    changing LLM_MODEL_ID env var, no code changes needed.
    """
    response = client.converse(
        modelId=model_id,
        system=[{"text": system_prompt}],
        messages=messages,
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0.0},
    )
    return response["output"]["message"]["content"][0]["text"]


def _build_messages(
    question: str,
    chat_history: Optional[List[dict]] = None,
) -> list:
    """Build message list with optional chat history for follow-up context."""
    messages = []

    if chat_history:
        for entry in chat_history[-6:]:  # Keep last 3 exchanges (6 messages)
            messages.append({"role": "user", "content": [{"text": entry["question"]}]})
            sql_text = entry.get("sql") or "No query generated"
            messages.append({"role": "assistant", "content": [{"text": sql_text}]})

    messages.append({"role": "user", "content": [{"text": question}]})
    return messages


def _extract_sql(raw: str) -> str:
    """Extract SQL from LLM response, stripping any markdown fences."""
    # Strip markdown code fences if present
    match = re.search(r"```(?:sql)?\s*(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw.strip()


def ask_question(
    question: str,
    engine: Engine,
    config: Optional[LLMConfig] = None,
    chat_history: Optional[List[dict]] = None,
) -> QueryResult:
    """
    Main entry point: natural language question → answer.

    Pipeline:
    1. question + schema → LLM → SQL
    2. validate SQL (SELECT-only, row limits)
    3. execute SQL in read-only transaction
    4. results + question → LLM → natural language answer
    """
    if config is None:
        config = LLMConfig.from_env()

    try:
        client = _get_bedrock_client(config)
    except Exception:
        logger.exception("Failed to create Bedrock client")
        return QueryResult(
            answer="AI Research is not available right now. Please check that AWS credentials are configured.",
            error="bedrock_client_error",
        )

    # Step 1: Generate SQL
    try:
        messages = _build_messages(question, chat_history)
        raw_sql = _call_llm(
            client, config.model_id, SYSTEM_PROMPT_SQL, messages, config.max_tokens_sql
        )
        sql = _extract_sql(raw_sql)
    except Exception:
        logger.exception("LLM SQL generation failed")
        return QueryResult(
            answer="I had trouble understanding that question. Could you try rephrasing it?",
            error="sql_generation_error",
        )

    # Step 2: Validate SQL
    is_valid, err_msg = validate_sql(sql)
    if not is_valid:
        logger.warning("SQL validation failed: %s | SQL: %s", err_msg, sql)
        return QueryResult(
            answer="I couldn't generate a safe query for that question. Try asking about game results, team records, or betting trends.",
            sql=sql,
            error=f"validation_error: {err_msg}",
        )

    sql = enforce_row_limit(sql, config.max_result_rows)

    # Step 3: Execute SQL in read-only transaction
    try:
        with engine.connect() as conn:
            # PostgreSQL: set transaction to read-only for extra safety
            try:
                conn.execute(text("SET TRANSACTION READ ONLY"))
            except Exception:
                pass  # Not all connections support SET TRANSACTION

            df = pd.read_sql(text(sql), conn)
    except Exception:
        logger.exception("SQL execution failed: %s", sql)
        return QueryResult(
            answer="I found a query but couldn't get results from the database. Try rephrasing your question.",
            sql=sql,
            error="execution_error",
        )

    if df.empty:
        return QueryResult(
            answer="I didn't find any matching data. Try broadening your search — for example, check the team name spelling or try a different date range.",
            sql=sql,
            dataframe=df,
        )

    # Step 4: Format results into natural language
    try:
        results_text = df.to_string(index=False, max_rows=config.max_result_rows)
        answer_messages = [
            {
                "role": "user",
                "content": [{"text": f"Question: {question}\n\nResults:\n{results_text}"}],
            }
        ]
        answer = _call_llm(
            client, config.model_id, SYSTEM_PROMPT_ANSWER, answer_messages, config.max_tokens_answer
        )
    except Exception:
        logger.exception("LLM answer formatting failed")
        # Fall back to showing raw data
        answer = f"Here are the results (I couldn't format a natural language summary):\n\n{df.to_string(index=False)}"

    return QueryResult(answer=answer, sql=sql, dataframe=df)
