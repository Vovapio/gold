#!/usr/bin/env python3
"""Telegram relationship analyzer.

Parses Telegram HTML exports, computes communication statistics, summarizes large
conversations with OpenAI models, and writes a structured report.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup
from dateutil import parser as date_parser
from openai import OpenAI
from textblob import TextBlob

LOGGER = logging.getLogger("telegram_relationship_analyzer")


@dataclass
class Message:
    """Represents one parsed Telegram message."""

    message_id: int
    sender: str
    timestamp: datetime
    text: str
    message_length: int
    reply_to_message_id: int | None
    media_type: str | None
    is_service: bool


class TelegramParser:
    """Parse Telegram chat HTML exports and persist normalized messages."""

    def __init__(self, export_dir: Path, db_path: Path, workers: int = 1) -> None:
        self.export_dir = export_dir
        self.db_path = db_path
        self.workers = max(1, workers)

    def validate_input(self) -> list[Path]:
        """Validate export directory and return chat*.html files."""
        if not self.export_dir.exists() or not self.export_dir.is_dir():
            raise FileNotFoundError(f"Export directory does not exist: {self.export_dir}")

        html_files = sorted(self.export_dir.glob("chat*.html"))
        if not html_files:
            raise FileNotFoundError(
                f"No Telegram chat HTML files found in {self.export_dir}. Expected chat*.html"
            )
        return html_files

    def initialize_db(self) -> None:
        """Create messages table and indexes if needed."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY,
                    sender TEXT,
                    timestamp TEXT,
                    text TEXT,
                    message_length INTEGER,
                    reply_to_message_id INTEGER,
                    media_type TEXT,
                    is_service INTEGER DEFAULT 0
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_sender ON messages(sender)")

    def parse(self, resume: bool = True) -> int:
        """Parse all chat HTML files and write data into SQLite.

        Returns the number of inserted messages.
        """
        html_files = self.validate_input()
        self.initialize_db()

        inserted = 0
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            existing_max_id = conn.execute("SELECT COALESCE(MAX(message_id), 0) FROM messages").fetchone()[0]
            for file_path in html_files:
                LOGGER.info("Parsing %s", file_path.name)
                try:
                    messages = self._parse_html_file(file_path)
                    if resume:
                        messages = [m for m in messages if m.message_id > existing_max_id]
                    inserted += self._insert_messages(conn, messages)
                    conn.commit()
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception("Failed to parse %s: %s", file_path, exc)
        LOGGER.info("Inserted %d messages", inserted)
        return inserted

    def _parse_html_file(self, file_path: Path) -> list[Message]:
        """Extract messages from one Telegram HTML file."""
        soup = BeautifulSoup(file_path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        messages: list[Message] = []

        for message_div in soup.select("div.message.default"):
            try:
                m = self._extract_message(message_div)
                if m:
                    messages.append(m)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Skipping malformed message in %s: %s", file_path.name, exc)
        return messages

    def _extract_message(self, message_div: Any) -> Message | None:
        """Extract a Message object, skipping service and joined messages."""
        if "service" in message_div.get("class", []):
            return None

        message_id_raw = message_div.get("id", "")
        match = re.search(r"message(\d+)", message_id_raw)
        if not match:
            LOGGER.warning("Missing message id, skipping message")
            return None
        message_id = int(match.group(1))

        from_name = message_div.select_one("div.from_name")
        sender = from_name.get_text(strip=True) if from_name else "Unknown"

        date_elem = message_div.select_one("div.date")
        if not date_elem or not date_elem.get("title"):
            LOGGER.warning("Message %s missing date, skipping", message_id)
            return None
        timestamp = date_parser.parse(date_elem["title"])

        # Skip joined messages and service entries represented in body text.
        text_elem = message_div.select_one("div.text")
        text = ""
        if text_elem:
            text = html.unescape(text_elem.get_text(" ", strip=True))
            text = re.sub(r"\s+", " ", text).strip()

        lowered = text.lower()
        if any(token in lowered for token in ("joined the group", "joined telegram", "added ")):
            return None

        media_type = None
        if message_div.select_one("a.photo_wrap"):
            media_type = "photo"
        elif message_div.select_one("video") or message_div.select_one("a.video_file"):
            media_type = "video"
        elif message_div.select_one("audio") or message_div.select_one("a.audio_file"):
            media_type = "audio"
        elif message_div.select_one("a.document"):
            media_type = "document"
        elif message_div.select_one("a.sticker"):
            media_type = "sticker"

        reply_to_id = None
        reply_anchor = message_div.select_one("a.reply_to")
        if reply_anchor and reply_anchor.get("href"):
            reply_match = re.search(r"go_to_message(\d+)", reply_anchor["href"])
            if reply_match:
                reply_to_id = int(reply_match.group(1))

        return Message(
            message_id=message_id,
            sender=sender,
            timestamp=timestamp,
            text=text,
            message_length=len(text),
            reply_to_message_id=reply_to_id,
            media_type=media_type,
            is_service=False,
        )

    def _insert_messages(self, conn: sqlite3.Connection, messages: Sequence[Message]) -> int:
        """Insert parsed messages into SQLite with upsert behavior."""
        if not messages:
            return 0

        rows = [
            (
                m.message_id,
                m.sender,
                m.timestamp.isoformat(),
                m.text,
                m.message_length,
                m.reply_to_message_id,
                m.media_type,
                int(m.is_service),
            )
            for m in messages
        ]
        conn.executemany(
            """
            INSERT OR IGNORE INTO messages
            (message_id, sender, timestamp, text, message_length, reply_to_message_id, media_type, is_service)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        return conn.total_changes


class DataPreprocessor:
    """Load normalized data and compute participant/message-level features."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def load_messages(self) -> pd.DataFrame:
        """Load message table as DataFrame sorted by time."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM messages ORDER BY timestamp ASC", conn)
        if df.empty:
            raise ValueError("No messages available for analysis")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"])
        df["text"] = df["text"].fillna("")
        df["sender"] = df["sender"].fillna("Unknown")
        df["message_length"] = df["message_length"].fillna(0).astype(int)
        return df

    def enrich_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute words, sentiment and hourly buckets."""
        enriched = df.copy()
        enriched["word_count"] = enriched["text"].str.split().str.len().fillna(0).astype(int)
        enriched["hour"] = enriched["timestamp"].dt.hour
        enriched["day"] = enriched["timestamp"].dt.date

        sentiments = enriched["text"].apply(self._sentiment_and_emotion)
        enriched["sentiment_polarity"] = sentiments.apply(lambda d: d[0])
        enriched["emotion"] = sentiments.apply(lambda d: d[1])
        return enriched

    @staticmethod
    def _sentiment_and_emotion(text: str) -> tuple[float, str]:
        """Infer sentiment polarity and rough emotion label."""
        if not text.strip():
            return 0.0, "neutral"
        polarity = float(TextBlob(text).sentiment.polarity)
        if polarity >= 0.25:
            emotion = "positive"
        elif polarity <= -0.25:
            emotion = "negative"
        else:
            emotion = "neutral"
        return polarity, emotion

    def per_participant_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate participant-level statistics."""
        total_messages = len(df)

        grouped = (
            df.groupby("sender", as_index=False)
            .agg(
                messages=("message_id", "count"),
                avg_message_length=("message_length", "mean"),
                total_words=("word_count", "sum"),
                avg_sentiment=("sentiment_polarity", "mean"),
            )
            .sort_values("messages", ascending=False)
        )
        grouped["message_ratio"] = grouped["messages"] / max(total_messages, 1)
        return grouped

    def hourly_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Message distribution by sender and hour."""
        dist = (
            df.groupby(["sender", "hour"]).size().rename("count").reset_index().sort_values(["sender", "hour"])
        )
        return dist

    def response_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate response times based on sender switches in chronological order."""
        local = df[["message_id", "sender", "timestamp"]].copy().sort_values("timestamp")
        local["prev_sender"] = local["sender"].shift(1)
        local["prev_timestamp"] = local["timestamp"].shift(1)
        local["response_seconds"] = (
            local["timestamp"] - local["prev_timestamp"]
        ).dt.total_seconds()
        local = local[(local["sender"] != local["prev_sender"]) & (local["response_seconds"] >= 0)]
        return (
            local.groupby("sender", as_index=False)["response_seconds"]
            .mean()
            .rename(columns={"response_seconds": "avg_response_seconds"})
        )


class Summarizer:
    """Chunk, summarize, and recursively summarize with OpenAI models."""

    def __init__(
        self,
        api_key: str,
        relationship_type: str,
        model: str = "gpt-4o-mini",
        chunk_messages: int = 1000,
        overlap: int = 50,
        max_retries: int = 5,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.relationship_type = relationship_type
        self.model = model
        self.chunk_messages = chunk_messages
        self.overlap = overlap
        self.max_retries = max_retries

    def build_chunks(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """Split conversation in chronological chunks with overlap."""
        chunks: list[pd.DataFrame] = []
        start = 0
        n = len(df)
        while start < n:
            end = min(start + self.chunk_messages, n)
            chunks.append(df.iloc[start:end])
            if end >= n:
                break
            start = max(0, end - self.overlap)
        return chunks

    def summarize_chunk(self, chunk_df: pd.DataFrame, chunk_idx: int) -> str:
        """Summarize one chunk with retries and exponential backoff."""
        sample_rows = []
        for _, row in chunk_df.iterrows():
            line = f"[{row['timestamp']}] {row['sender']}: {row['text'][:400]}"
            sample_rows.append(line)
        prompt_body = "\n".join(sample_rows)

        system_msg = (
            "You summarize Telegram chats for relationship analysis. "
            f"Focus on {self.relationship_type} dynamics, major topics, sentiment trends, "
            "notable interactions, and communication shifts."
        )
        user_msg = (
            f"Chunk {chunk_idx}: Summarize key developments. Highlight positive/negative patterns, "
            "conflict markers, support, humor, apology, gratitude, and cooperation.\n\n"
            f"Messages:\n{prompt_body}"
        )
        return self._chat_with_retry(system_msg, user_msg)

    def recursive_summary(self, chunk_summaries: Sequence[str]) -> str:
        """Produce one high-level summary from chunk summaries."""
        combined = "\n\n".join(f"Chunk {i+1}: {s}" for i, s in enumerate(chunk_summaries))
        system_msg = (
            "You are an expert communication analyst. Return a concise but specific relationship analysis "
            f"for a {self.relationship_type} context. Cover communication patterns, emotional trends, "
            "four horsemen (criticism, contempt, defensiveness, stonewalling), and positive behaviors."
        )
        user_msg = (
            "Create an overall summary from these chunk summaries. Include caveats where evidence is weak.\n\n"
            f"{combined}"
        )
        return self._chat_with_retry(system_msg, user_msg)

    def _chat_with_retry(self, system_msg: str, user_msg: str) -> str:
        """Call OpenAI Chat Completions with retry/backoff."""
        wait = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                if attempt == self.max_retries:
                    raise
                LOGGER.warning("OpenAI call failed (attempt %d/%d): %s", attempt, self.max_retries, exc)
                time.sleep(wait)
                wait *= 2
        return ""


class Analyzer:
    """Combine statistics and summaries into structured insights."""

    POSITIVE_MARKERS = ("thanks", "thank you", "sorry", "love", "great", "haha", "lol", "appreciate")
    NEGATIVE_MARKERS = ("always", "never", "whatever", "fine", "leave me", "stupid", "ignore")

    def __init__(self, relationship_type: str) -> None:
        self.relationship_type = relationship_type

    def analyze(self, enriched_df: pd.DataFrame, participant_stats: pd.DataFrame, response_df: pd.DataFrame) -> dict[str, Any]:
        """Generate quantitative insights and behavior markers."""
        total_messages = len(enriched_df)
        date_min = enriched_df["timestamp"].min()
        date_max = enriched_df["timestamp"].max()

        pos_count = self._marker_count(enriched_df["text"], self.POSITIVE_MARKERS)
        neg_count = self._marker_count(enriched_df["text"], self.NEGATIVE_MARKERS)

        by_month = (
            enriched_df.assign(month=enriched_df["timestamp"].dt.to_period("M").astype(str))
            .groupby("month")
            .size()
            .rename("messages")
            .reset_index()
            .to_dict(orient="records")
        )

        sufficiency_warnings: list[str] = []
        if total_messages < 200:
            sufficiency_warnings.append(
                "Fewer than 200 messages were analyzed; conclusions are likely noisy."
            )
        if enriched_df["sender"].nunique() < 2:
            sufficiency_warnings.append(
                "Only one participant detected; relationship inference may be unreliable."
            )

        avg_sentiment = float(enriched_df["sentiment_polarity"].mean()) if total_messages else 0.0

        insights = {
            "relationship_type": self.relationship_type,
            "coverage": {
                "message_count": total_messages,
                "participants": int(enriched_df["sender"].nunique()),
                "time_range": {
                    "start": date_min.isoformat() if pd.notna(date_min) else None,
                    "end": date_max.isoformat() if pd.notna(date_max) else None,
                },
            },
            "communication_volume_balance": participant_stats.to_dict(orient="records"),
            "response_times": response_df.to_dict(orient="records"),
            "activity_over_time": by_month,
            "sentiment": {
                "overall_average_polarity": avg_sentiment,
                "emotion_distribution": enriched_df["emotion"].value_counts().to_dict(),
            },
            "behavior_markers": {
                "positive_markers": pos_count,
                "negative_markers": neg_count,
                "positive_to_negative_ratio": (pos_count / neg_count) if neg_count else math.inf,
            },
            "inferred_traits": self._infer_traits(avg_sentiment, pos_count, neg_count),
            "sufficiency_warnings": sufficiency_warnings,
        }
        return insights

    @staticmethod
    def _marker_count(text_series: pd.Series, markers: Sequence[str]) -> int:
        pattern = "|".join(re.escape(marker) for marker in markers)
        return int(text_series.str.lower().str.count(pattern).sum())

    @staticmethod
    def _infer_traits(avg_sentiment: float, pos_count: int, neg_count: int) -> dict[str, str]:
        traits: dict[str, str] = {}
        if avg_sentiment > 0.2 and pos_count >= neg_count:
            traits["tone"] = "Generally warm/supportive communication style"
        elif avg_sentiment < -0.15 or neg_count > pos_count * 1.5:
            traits["tone"] = "Frequent strain/conflict indicators"
        else:
            traits["tone"] = "Mixed tone with balanced positive and negative signals"

        if pos_count > neg_count * 1.5:
            traits["attachment_style_hint"] = "Likely secure/affirming interaction patterns"
        elif neg_count > pos_count * 1.5:
            traits["attachment_style_hint"] = "Possible anxious/avoidant stress responses"
        else:
            traits["attachment_style_hint"] = "No clear attachment-style signal"
        return traits


class Reporter:
    """Generate report artifacts (JSON/text and optional charts)."""

    def __init__(self, output_path: Path, charts_dir: Path | None = None) -> None:
        self.output_path = output_path
        self.charts_dir = charts_dir

    def save_report(
        self,
        overall_summary: str,
        participant_stats: pd.DataFrame,
        hourly_dist: pd.DataFrame,
        insights: dict[str, Any],
    ) -> Path:
        """Persist report with summary, stats table and recommendations."""
        chart_paths = self._render_charts(hourly_dist) if self.charts_dir else []

        payload = {
            "high_level_summary": overall_summary,
            "participant_stats": participant_stats.to_dict(orient="records"),
            "insights": insights,
            "detected_behaviors": {
                "positive": [
                    "Supportive statements",
                    "Humor markers",
                    "Apologies and gratitude",
                    "Shared-goal language",
                ],
                "negative": [
                    "Criticism",
                    "Contempt",
                    "Defensiveness",
                    "Stonewalling",
                ],
            },
            "suggestions": self._suggestions(insights),
            "chart_paths": [str(p) for p in chart_paths],
        }

        if self.output_path.suffix.lower() == ".txt":
            self.output_path.write_text(self._to_text(payload), encoding="utf-8")
        else:
            self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.output_path

    def _render_charts(self, hourly_dist: pd.DataFrame) -> list[Path]:
        if hourly_dist.empty or self.charts_dir is None:
            return []
        self.charts_dir.mkdir(parents=True, exist_ok=True)

        pivot = hourly_dist.pivot(index="hour", columns="sender", values="count").fillna(0)
        fig, ax = plt.subplots(figsize=(10, 5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Message distribution by hour")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Messages")
        fig.tight_layout()

        chart_path = self.charts_dir / "hourly_distribution.png"
        fig.savefig(chart_path)
        plt.close(fig)
        return [chart_path]

    @staticmethod
    def _to_text(payload: dict[str, Any]) -> str:
        return (
            "=== High-level Summary ===\n"
            f"{payload['high_level_summary']}\n\n"
            "=== Participant Stats ===\n"
            f"{json.dumps(payload['participant_stats'], indent=2)}\n\n"
            "=== Insights ===\n"
            f"{json.dumps(payload['insights'], indent=2)}\n\n"
            "=== Suggestions ===\n"
            + "\n".join(f"- {s}" for s in payload["suggestions"])
        )

    @staticmethod
    def _suggestions(insights: dict[str, Any]) -> list[str]:
        suggestions = [
            "Set regular check-ins during peak active hours to improve responsiveness.",
            "Use explicit appreciation and repair attempts after tense exchanges.",
            "When conflict appears, pause and restate goals before continuing.",
        ]
        warnings = insights.get("sufficiency_warnings") or []
        if warnings:
            suggestions.append("Collect more longitudinal data before making major decisions.")
        return suggestions


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Telegram chat export for relationship insights")
    parser.add_argument("--export-dir", required=True, type=Path, help="Telegram export directory")
    parser.add_argument(
        "--relationship-type",
        required=True,
        choices=["friendship", "romantic", "professional"],
        help="Relationship framing for analysis",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output report path (.json or .txt)")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (fallback to OPENAI_API_KEY)")
    parser.add_argument("--db-path", type=Path, default=Path("analysis_messages.sqlite"), help="SQLite cache path")
    parser.add_argument("--resume", action="store_true", help="Resume by reusing cached SQLite messages")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Messages per GPT chunk")
    parser.add_argument("--overlap", type=int, default=50, help="Chunk overlap")
    parser.add_argument("--charts-dir", type=Path, default=None, help="Optional directory for chart images")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        LOGGER.error("OpenAI API key missing. Pass --api-key or set OPENAI_API_KEY.")
        return 2

    try:
        parser = TelegramParser(args.export_dir, args.db_path)
        parser.parse(resume=args.resume)

        preprocessor = DataPreprocessor(args.db_path)
        df = preprocessor.load_messages()
        enriched_df = preprocessor.enrich_messages(df)

        participant_stats = preprocessor.per_participant_stats(enriched_df)
        hourly_dist = preprocessor.hourly_distribution(enriched_df)
        response_df = preprocessor.response_times(enriched_df)

        summarizer = Summarizer(
            api_key=api_key,
            relationship_type=args.relationship_type,
            chunk_messages=args.chunk_size,
            overlap=args.overlap,
        )

        chunks = summarizer.build_chunks(enriched_df)
        chunk_summaries: list[str] = []
        for idx, chunk in enumerate(chunks, start=1):
            LOGGER.info("Summarizing chunk %d/%d", idx, len(chunks))
            chunk_summaries.append(summarizer.summarize_chunk(chunk, idx))
            time.sleep(0.2)  # gentle pacing for rate limits

        overall_summary = summarizer.recursive_summary(chunk_summaries)

        analyzer = Analyzer(args.relationship_type)
        insights = analyzer.analyze(enriched_df, participant_stats, response_df)

        reporter = Reporter(output_path=args.output, charts_dir=args.charts_dir)
        report_path = reporter.save_report(overall_summary, participant_stats, hourly_dist, insights)

        LOGGER.info("Report written to %s", report_path)
        return 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Analysis failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
