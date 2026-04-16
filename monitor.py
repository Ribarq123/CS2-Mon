#!/usr/bin/env python3
"""
monitor.py

A Python 3.12+ command-line monitoring application that:
- reads system and metric definitions from a JSON config file
- collects supported system metrics with psutil
- stores all collected data in a local SQLite database
- displays stored measurements over a selected time period
- calculates summary statistics
- optionally resets the database
- optionally forwards collected data to a Flask API

Usage examples:
    python monitor.py collect --config config.json
    python monitor.py show --db monitoring.db
    python monitor.py show --db monitoring.db --start "2026-04-15 00:00:00" --end "2026-04-16 23:59:59"
    python monitor.py stats --db monitoring.db --metric cpu_usage
    python monitor.py reset-db --db monitoring.db --confirm
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import requests


DEFAULT_DB_PATH = "monitoring.db"
DEFAULT_LOG_DIR = "logs"
DEFAULT_LOG_FILE = "logs/monitor.log"
DEFAULT_API_TIMEOUT = 10


@dataclass
class Measurement:
    timestamp: str
    source: str
    hostname: str
    ip_address: str
    metric_name: str
    metric_value: float | None
    status: str
    error_message: str | None = None


def setup_logging(log_file: str = DEFAULT_LOG_FILE) -> None:
    """Configure logging to file and console."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate JSON config."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if "system" not in config:
        raise ValueError("Config must contain a 'system' section.")
    if "metrics" not in config:
        raise ValueError("Config must contain a 'metrics' section.")

    system = config["system"]
    if "hostname" not in system or "ip_address" not in system:
        raise ValueError("System config must include 'hostname' and 'ip_address'.")

    return config


def get_timestamp() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def ensure_database(db_path: str) -> None:
    """Create the SQLite database and table if they do not exist."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                hostname TEXT NOT NULL,
                ip_address TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                status TEXT NOT NULL,
                error_message TEXT
            )
            """
        )
        conn.commit()


def insert_measurement(db_path: str, measurement: Measurement) -> None:
    """Insert one measurement into SQLite."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO measurements (
                timestamp,
                source,
                hostname,
                ip_address,
                metric_name,
                metric_value,
                status,
                error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                measurement.timestamp,
                measurement.source,
                measurement.hostname,
                measurement.ip_address,
                measurement.metric_name,
                measurement.metric_value,
                measurement.status,
                measurement.error_message,
            ),
        )
        conn.commit()


def parse_datetime(value: str | None) -> str | None:
    """Parse user-supplied datetime and return normalized string."""
    if value is None:
        return None

    normalized = value.strip().replace("T", " ")
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.isoformat(sep=" ")
    except ValueError as exc:
        raise ValueError(
            f"Invalid datetime format: {value}. "
            "Use 'YYYY-MM-DD HH:MM:SS' or ISO format."
        ) from exc


def build_where_clause(
    start: str | None = None,
    end: str | None = None,
    metric: str | None = None,
    hostname: str | None = None,
) -> tuple[str, list[Any]]:
    """Build WHERE clause for query filters."""
    clauses: list[str] = []
    params: list[Any] = []

    if start:
        clauses.append("timestamp >= ?")
        params.append(start)

    if end:
        clauses.append("timestamp <= ?")
        params.append(end)

    if metric:
        clauses.append("metric_name = ?")
        params.append(metric)

    if hostname:
        clauses.append("hostname = ?")
        params.append(hostname)

    if not clauses:
        return "", params

    return "WHERE " + " AND ".join(clauses), params


def fetch_measurements(
    db_path: str,
    start: str | None = None,
    end: str | None = None,
    metric: str | None = None,
    hostname: str | None = None,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """Fetch measurements from SQLite."""
    where_clause, params = build_where_clause(start, end, metric, hostname)

    query = f"""
        SELECT
            id,
            timestamp,
            source,
            hostname,
            ip_address,
            metric_name,
            metric_value,
            status,
            error_message
        FROM measurements
        {where_clause}
        ORDER BY timestamp DESC, id DESC
    """

    if limit:
        query += " LIMIT ?"
        params.append(limit)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    return rows


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a simple formatted table."""
    if not rows:
        print("No rows found.")
        return

    widths = [len(header) for header in headers]

    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    header_line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    separator = "-+-".join("-" * widths[i] for i in range(len(headers)))

    print(header_line)
    print(separator)

    for row in rows:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def evaluate_status(
    metric_name: str,
    metric_value: float | None,
    thresholds: dict[str, Any] | None,
) -> str:
    """Evaluate status based on optional thresholds."""
    if metric_value is None:
        return "error"

    if not thresholds or metric_name not in thresholds:
        return "ok"

    metric_thresholds = thresholds[metric_name]
    warning = metric_thresholds.get("warning")
    critical = metric_thresholds.get("critical")

    if critical is not None and metric_value >= critical:
        return "critical"

    if warning is not None and metric_value >= warning:
        return "warning"

    return "ok"


def get_disk_usage_percent(path: str) -> float:
    """Return disk usage percentage for a given path."""
    return round(psutil.disk_usage(path).percent, 2)


def collect_metric_value(metric_name: str, config: dict[str, Any]) -> float:
    """Collect a supported metric value."""
    custom_disk_metrics = config.get("custom_disk_metrics", {})

    metric_collectors: dict[str, Any] = {
        "cpu_usage": lambda: round(psutil.cpu_percent(interval=1), 2),
        "memory_usage": lambda: round(psutil.virtual_memory().percent, 2),
        "disk_root_usage": lambda: get_disk_usage_percent("/"),
        "disk_home_usage": lambda: get_disk_usage_percent("/home"),
        "disk_data_usage": lambda: get_disk_usage_percent("/data"),
    }

    if metric_name in metric_collectors:
        return metric_collectors[metric_name]()

    if metric_name in custom_disk_metrics:
        return get_disk_usage_percent(custom_disk_metrics[metric_name])

    raise ValueError(f"Unsupported metric: {metric_name}")


def send_to_api(measurement: Measurement, api_config: dict[str, Any]) -> bool:
    """Send one measurement to the Flask API."""
    if not api_config.get("enabled", False):
        return False

    url = api_config.get("url")
    api_key = api_config.get("api_key")

    if not url or not api_key:
        logging.warning("API sending enabled, but URL or API key is missing.")
        return False

    payload = {
        "source": measurement.source,
        "hostname": measurement.hostname,
        "metric_name": measurement.metric_name,
        "metric_value": measurement.metric_value,
        "status": measurement.status,
    }

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }

    try:
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=api_config.get("timeout", DEFAULT_API_TIMEOUT),
        )
        response.raise_for_status()
        logging.info(
            "API send successful for metric '%s' with status %s.",
            measurement.metric_name,
            response.status_code,
        )
        return True
    except requests.RequestException as exc:
        logging.error(
            "API send failed for metric '%s': %s",
            measurement.metric_name,
            exc,
        )
        return False


def run_collect(config_path: str, db_path: str) -> None:
    """Collect metrics, store locally, and optionally forward to API."""
    config = load_config(config_path)
    ensure_database(db_path)

    system = config["system"]
    metrics = config["metrics"]
    thresholds = config.get("thresholds", {})
    api_config = config.get("api", {})

    source = system.get("source", "onprem")
    hostname = system["hostname"]
    ip_address = system["ip_address"]

    logging.info("Starting collection for host '%s' (%s).", hostname, ip_address)

    for metric_name in metrics:
        timestamp = get_timestamp()

        try:
            value = collect_metric_value(metric_name, config)
            status = evaluate_status(metric_name, value, thresholds)

            measurement = Measurement(
                timestamp=timestamp,
                source=source,
                hostname=hostname,
                ip_address=ip_address,
                metric_name=metric_name,
                metric_value=value,
                status=status,
                error_message=None,
            )

            insert_measurement(db_path, measurement)
            logging.info(
                "Stored metric '%s' with value %s and status '%s'.",
                metric_name,
                value,
                status,
            )

            if api_config.get("enabled", False):
                send_to_api(measurement, api_config)

        except Exception as exc:  # noqa: BLE001
            measurement = Measurement(
                timestamp=timestamp,
                source=source,
                hostname=hostname,
                ip_address=ip_address,
                metric_name=metric_name,
                metric_value=None,
                status="error",
                error_message=str(exc),
            )

            insert_measurement(db_path, measurement)
            logging.error(
                "Failed to collect metric '%s': %s",
                metric_name,
                exc,
            )


def run_show(
    db_path: str,
    start: str | None,
    end: str | None,
    metric: str | None,
    hostname: str | None,
    limit: int | None,
) -> None:
    """Display stored measurements."""
    ensure_database(db_path)

    rows = fetch_measurements(
        db_path=db_path,
        start=parse_datetime(start),
        end=parse_datetime(end),
        metric=metric,
        hostname=hostname,
        limit=limit,
    )

    formatted_rows: list[list[str]] = []
    for row in rows:
        value = "" if row["metric_value"] is None else f"{row['metric_value']:.2f}"
        error = row["error_message"] or ""
        formatted_rows.append(
            [
                str(row["id"]),
                row["timestamp"],
                row["source"],
                row["hostname"],
                row["ip_address"],
                row["metric_name"],
                value,
                row["status"],
                error,
            ]
        )

    headers = [
        "id",
        "timestamp",
        "source",
        "hostname",
        "ip_address",
        "metric",
        "value",
        "status",
        "error",
    ]
    print_table(headers, formatted_rows)


def run_stats(
    db_path: str,
    start: str | None,
    end: str | None,
    metric: str | None,
    hostname: str | None,
) -> None:
    """Calculate summary statistics for numeric measurements."""
    ensure_database(db_path)

    where_clause, params = build_where_clause(
        start=parse_datetime(start),
        end=parse_datetime(end),
        metric=metric,
        hostname=hostname,
    )

    query = f"""
        SELECT
            metric_name,
            COUNT(metric_value) AS count_values,
            AVG(metric_value) AS avg_value,
            MIN(metric_value) AS min_value,
            MAX(metric_value) AS max_value
        FROM measurements
        {where_clause}
        AND metric_value IS NOT NULL
    """ if where_clause else """
        SELECT
            metric_name,
            COUNT(metric_value) AS count_values,
            AVG(metric_value) AS avg_value,
            MIN(metric_value) AS min_value,
            MAX(metric_value) AS max_value
        FROM measurements
        WHERE metric_value IS NOT NULL
    """

    query += " GROUP BY metric_name ORDER BY metric_name"

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    formatted_rows: list[list[str]] = []
    for row in rows:
        formatted_rows.append(
            [
                row["metric_name"],
                str(row["count_values"]),
                f"{row['avg_value']:.2f}",
                f"{row['min_value']:.2f}",
                f"{row['max_value']:.2f}",
            ]
        )

    headers = ["metric", "count", "average", "minimum", "maximum"]
    print_table(headers, formatted_rows)


def run_reset_db(db_path: str, confirm: bool) -> None:
    """Delete all rows from the database only when explicitly confirmed."""
    if not confirm:
        raise ValueError("reset-db requires --confirm to avoid accidental data loss.")

    ensure_database(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM measurements")
        conn.commit()

    logging.info("Database '%s' was reset successfully.", db_path)
    print(f"Database '{db_path}' reset successfully.")


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Python monitoring application with SQLite storage and optional API forwarding."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect configured metrics, store them locally, and optionally send them to the API.",
    )
    collect_parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file.",
    )
    collect_parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH}).",
    )

    show_parser = subparsers.add_parser(
        "show",
        help="Display stored measurements.",
    )
    show_parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH}).",
    )
    show_parser.add_argument("--start", help="Start datetime filter.")
    show_parser.add_argument("--end", help="End datetime filter.")
    show_parser.add_argument("--metric", help="Filter by metric name.")
    show_parser.add_argument("--hostname", help="Filter by hostname.")
    show_parser.add_argument("--limit", type=int, help="Limit result count.")

    stats_parser = subparsers.add_parser(
        "stats",
        help="Show summary statistics for stored measurements.",
    )
    stats_parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH}).",
    )
    stats_parser.add_argument("--start", help="Start datetime filter.")
    stats_parser.add_argument("--end", help="End datetime filter.")
    stats_parser.add_argument("--metric", help="Filter by metric name.")
    stats_parser.add_argument("--hostname", help="Filter by hostname.")

    reset_parser = subparsers.add_parser(
        "reset-db",
        help="Clear the measurements table.",
    )
    reset_parser.add_argument(
        "--db",
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH}).",
    )
    reset_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required flag to confirm database reset.",
    )

    return parser


def main() -> None:
    """Application entry point."""
    setup_logging()
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "collect":
            run_collect(config_path=args.config, db_path=args.db)
        elif args.command == "show":
            run_show(
                db_path=args.db,
                start=args.start,
                end=args.end,
                metric=args.metric,
                hostname=args.hostname,
                limit=args.limit,
            )
        elif args.command == "stats":
            run_stats(
                db_path=args.db,
                start=args.start,
                end=args.end,
                metric=args.metric,
                hostname=args.hostname,
            )
        elif args.command == "reset-db":
            run_reset_db(db_path=args.db, confirm=args.confirm)
        else:
            parser.print_help()
    except Exception as exc:  # noqa: BLE001
        logging.error("Application error: %s", exc)
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()