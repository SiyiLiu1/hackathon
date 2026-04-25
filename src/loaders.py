"""Data loaders and joins for Track 1 datasets.

This module loads the core Track 1 tables from parquet, CSV, or SQLite
sources, applies required column renames, validates expected schema, and
builds the canonical ``referrals_enriched`` DataFrame used by downstream
analysis.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd

LOGGER = logging.getLogger(__name__)
DEFAULT_TRACK1_RAW_DIR = Path(__file__).resolve().parents[1] / "tracks" / "referral-care-coordination" / "data" / "raw"
DEFAULT_TRACK1_SAMPLE_DIR = (
    Path(__file__).resolve().parents[1] / "tracks" / "referral-care-coordination" / "data" / "sample"
)


class SchemaValidationError(ValueError):
    """Base exception for table schema validation failures."""


class MissingRequiredTableError(SchemaValidationError):
    """Raised when one or more required tables are not provided."""


class MissingRequiredColumnError(SchemaValidationError):
    """Raised when one or more required columns are missing."""


TRACK1_TABLE_SPECS: Dict[str, dict[str, list[str] | str]] = {
    "orgs": {"table": "organizations", "aliases": ["organizations", "orgs"]},
    "clients": {"table": "clients", "aliases": ["clients"]},
    "referrals": {"table": "referrals", "aliases": ["referrals"]},
    "encounters": {
        "table": "service_encounters",
        "aliases": ["service_encounters", "encounters"],
    },
    "consent": {
        "table": "consent_records",
        "aliases": ["consent_records", "consent"],
    },
    "dsa": {
        "table": "data_sharing_agreements",
        "aliases": ["data_sharing_agreements", "dsa"],
    },
    "duplicate_flags": {
        "table": "duplicate_flags",
        "aliases": ["duplicate_flags"],
    },
}


REQUIRED_RENAMES: Dict[str, str] = {
    "assessment_total_score": "vi_spdat_score",
    "indigenous_identity": "indigenous_flag",
    "current_sleeping_location": "address_line1",
    "referral_reason": "reason_code",
    "encounter_start": "occurred_at",
    "effective_date": "effective_at",
    "expiry_date": "expires_at",
    "client_id_primary": "client_id_a",
    "client_id_secondary": "client_id_b",
}

NULL_LIKE_TEXT = {"", "null", "none", "nan", "na", "n/a", "n\\a"}


REQUIRED_COLUMNS: Dict[str, set[str]] = {
    "orgs": {"org_id", "org_name", "org_type"},
    "clients": {
        "client_id",
        "first_name",
        "last_name",
        "dob",
        "primary_org_id",
        "current_consent_id",
    },
    "referrals": {
        "referral_id",
        "client_id",
        "referring_org_id",
        "receiving_org_id",
        "consent_record_id",
        "status",
        "submitted_at",
        "reason_code",
    },
    "encounters": {"encounter_id", "client_id", "org_id", "occurred_at"},
    "consent": {
        "consent_id",
        "client_id",
        "collecting_org_id",
        "status",
        "sharing_scope_type",
        "effective_at",
        "expires_at",
    },
    # Keep DSA validation compatible with generated and fixture schemas:
    # type/dsa_type is intentionally optional.
    "dsa": {"dsa_id", "effective_at", "expires_at"},
    "duplicate_flags": {"client_id_a", "client_id_b", "review_status"},
}


def get_required_columns() -> Dict[str, set[str]]:
    """Return the required schema by table key.

    Returns:
        A dictionary mapping table key to a set of required column names.
    """
    return {table: set(columns) for table, columns in REQUIRED_COLUMNS.items()}


def validate_required_columns(
    tables: Mapping[str, pd.DataFrame], required_columns: Mapping[str, set[str]] | None = None
) -> None:
    """Validate that all loaded tables include required columns.

    Args:
        tables: Mapping of table key to loaded DataFrame.
        required_columns: Optional schema override. Defaults to
            :func:`get_required_columns`.

    Raises:
        MissingRequiredTableError: If a required table is missing from ``tables``.
        MissingRequiredColumnError: If one or more required columns are missing.
    """
    required = required_columns or get_required_columns()

    missing_tables = [table for table in required if table not in tables]
    if missing_tables:
        message = (
            "Schema validation failed. Missing required table(s): "
            + ", ".join(sorted(missing_tables))
            + ". Ensure all Track 1 parquet files are loaded."
        )
        LOGGER.error(message)
        raise MissingRequiredTableError(message)

    errors: list[tuple[str, list[str]]] = []
    for table, expected in required.items():
        missing_cols = sorted(expected - set(tables[table].columns))
        if missing_cols:
            errors.append((table, missing_cols))

    if errors:
        detail_lines = []
        for table, missing_cols in errors:
            detail_lines.append(
                f"Table '{table}' is missing required column(s): {', '.join(missing_cols)}"
            )
        message = (
            "Schema validation failed.\n"
            + "\n".join(detail_lines)
            + "\nCheck source parquet schema and required renames."
        )
        LOGGER.error(message)
        raise MissingRequiredColumnError(message)
    LOGGER.debug("Schema validation passed for tables: %s", ", ".join(sorted(required)))


def _apply_required_renames(df: pd.DataFrame) -> pd.DataFrame:
    """Apply project-required normalization renames when source columns exist."""
    rename_map = {src: dst for src, dst in REQUIRED_RENAMES.items() if src in df.columns}
    if not rename_map:
        return df
    LOGGER.debug(
        "Applying required renames: %s",
        ", ".join(f"{src}->{dst}" for src, dst in sorted(rename_map.items())),
    )
    return df.rename(columns=rename_map, errors="raise")


def _normalize_text_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common text null markers to pandas missing values."""
    if df.empty:
        return df
    out = df.copy()
    object_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        text = out[col].astype("string").str.strip()
        out[col] = text.mask(text.str.lower().isin(NULL_LIKE_TEXT), pd.NA)
    return out


def _load_from_sqlite(db_path: Path, table_name: str) -> pd.DataFrame:
    """Load a table from SQLite if present."""
    with sqlite3.connect(str(db_path)) as conn:
        available = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
        names = set(available["name"].astype(str).tolist())
        if table_name not in names:
            raise FileNotFoundError(f"SQLite table not found: {table_name} in {db_path}")
        return pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)


def _read_table_file(path: Path) -> pd.DataFrame:
    """Read a parquet/CSV file by extension."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file format: {path}")


def _iter_candidate_files(base_dir: Path, aliases: list[str]) -> list[Path]:
    """Return likely table files ordered by preferred format."""
    candidates: list[Path] = []
    for alias in aliases:
        for variant in (alias, f"{alias}_sample"):
            for ext in (".parquet", ".csv"):
                candidate = base_dir / f"{variant}{ext}"
                if candidate.exists():
                    candidates.append(candidate)
    return candidates


def _load_table_from_source(source: Path, table_key: str) -> pd.DataFrame:
    """Load one logical table from directory or SQLite source."""
    spec = TRACK1_TABLE_SPECS[table_key]
    table_name = str(spec["table"])
    aliases = list(spec["aliases"])

    if source.is_file() and source.suffix.lower() in {".sqlite", ".db"}:
        LOGGER.info("Loading table '%s' from sqlite %s", table_key, source)
        df = _load_from_sqlite(source, table_name)
        return _normalize_text_nulls(_apply_required_renames(df))

    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Data source directory does not exist: {source}")

    for file_path in _iter_candidate_files(source, aliases):
        LOGGER.info("Loading table '%s' from %s", table_key, file_path)
        df = _read_table_file(file_path)
        return _normalize_text_nulls(_apply_required_renames(df))

    sqlite_candidates = list(source.glob("*.sqlite")) + list(source.glob("*.db"))
    for db_path in sqlite_candidates:
        try:
            LOGGER.info("Trying table '%s' from sqlite %s", table_key, db_path)
            df = _load_from_sqlite(db_path, table_name)
            return _normalize_text_nulls(_apply_required_renames(df))
        except FileNotFoundError:
            continue

    aliases_text = ", ".join(sorted(set(aliases)))
    raise FileNotFoundError(
        f"Could not locate source for table '{table_key}' in {source}. "
        f"Tried aliases: {aliases_text} with parquet/csv/sqlite."
    )


def _resolve_track1_source(raw_dir: str | Path | None) -> Path:
    """Resolve data source path from user path then project fallbacks."""
    candidate_paths: list[Path] = []
    if raw_dir:
        candidate_paths.append(Path(raw_dir).expanduser())
    candidate_paths.extend([DEFAULT_TRACK1_RAW_DIR, DEFAULT_TRACK1_SAMPLE_DIR])

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidate_paths:
        resolved = candidate.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(resolved)

    for candidate in deduped:
        if candidate.is_file() and candidate.suffix.lower() in {".sqlite", ".db"}:
            print(f"[Track1] Resolved data path: {candidate}")
            LOGGER.info("Resolved Track 1 sqlite source: %s", candidate)
            return candidate
        if candidate.is_dir():
            org_candidates = _iter_candidate_files(
                candidate,
                list(TRACK1_TABLE_SPECS["orgs"]["aliases"]),
            )
            if org_candidates:
                print(f"[Track1] Resolved data path: {candidate}")
                LOGGER.info("Resolved Track 1 directory source: %s", candidate)
                return candidate
            if list(candidate.glob("*.sqlite")) or list(candidate.glob("*.db")):
                print(f"[Track1] Resolved data path: {candidate}")
                LOGGER.info("Resolved Track 1 sqlite directory source: %s", candidate)
                return candidate

    preferred = deduped[0]
    print(f"[Track1] Resolved data path: {preferred}")
    raise FileNotFoundError(
        "No Track 1 dataset found. Checked provided path and fallbacks for parquet/csv/sqlite: "
        + ", ".join(str(path) for path in deduped)
    )


def load_track1_data(raw_dir: str | Path = "data/raw") -> Dict[str, pd.DataFrame]:
    """Load Track 1 data from parquet, CSV, or SQLite.

    This loads:
    ``orgs, clients, referrals, encounters, consent, dsa, duplicate_flags``.
    Required renames are applied immediately after read.

    Additionally, this computes:
    ``duplicate_flags.is_true_duplicate = review_status in ['confirmed_duplicate', 'merged']``.

    Args:
        raw_dir: Folder containing Track 1 parquet files.

    Returns:
        Dictionary keyed by table name with pandas DataFrames.

    Raises:
        FileNotFoundError: If source or required table cannot be loaded.
        ValueError: If required columns are missing after renaming.
    """
    source = _resolve_track1_source(raw_dir)
    LOGGER.info("Loading Track 1 data from %s", source)
    tables = {key: _load_table_from_source(source, key) for key in TRACK1_TABLE_SPECS}

    dup_df = tables["duplicate_flags"].copy()
    status_norm = dup_df["review_status"].astype("string").str.lower().str.strip()
    dup_df["is_true_duplicate"] = status_norm.isin({"confirmed_duplicate", "merged"})
    tables["duplicate_flags"] = dup_df
    LOGGER.debug(
        "Derived 'duplicate_flags.is_true_duplicate' using review_status in "
        "['confirmed_duplicate', 'merged']"
    )

    validate_required_columns(tables)
    LOGGER.info("Track 1 data load complete")
    return tables


def build_referrals_enriched(tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the canonical referrals enrichment join.

    Join shape:
    ``referrals + client_slim + referring_org + receiving_org + current_consent``.

    Args:
        tables: Mapping returned by :func:`load_track1_data`.

    Returns:
        Enriched referrals DataFrame.

    Raises:
        KeyError: If required table keys are missing.
        ValueError: If required join columns are missing.
    """
    validate_required_columns(tables)

    referrals = tables["referrals"].copy()
    clients = tables["clients"].copy()
    orgs = tables["orgs"].copy()
    consent = tables["consent"].copy()

    client_slim_cols = [
        "client_id",
        "primary_org_id",
        "current_consent_id",
        "first_name",
        "last_name",
        "dob",
        "vi_spdat_score",
        "indigenous_flag",
        "address_line1",
        "ocap_protected",
        "default_sharing_scope",
    ]
    available_client_cols = [col for col in client_slim_cols if col in clients.columns]
    missing_client_slim_cols = sorted(set(client_slim_cols) - set(available_client_cols))
    if missing_client_slim_cols:
        LOGGER.warning(
            "Optional client_slim columns missing in 'clients': %s",
            ", ".join(missing_client_slim_cols),
        )
    client_slim = clients[available_client_cols].copy()

    referring_org = orgs.add_prefix("referring_org_")
    receiving_org = orgs.add_prefix("receiving_org_")

    current_consent_cols = [
        "consent_id",
        "status",
        "sharing_scope_type",
        "purpose_codes",
        "effective_at",
        "expires_at",
        "withdrawal_date",
        "dsa_id",
    ]
    available_consent_cols = [col for col in current_consent_cols if col in consent.columns]
    missing_current_consent_cols = sorted(set(current_consent_cols) - set(available_consent_cols))
    if missing_current_consent_cols:
        LOGGER.warning(
            "Optional current_consent columns missing in 'consent': %s",
            ", ".join(missing_current_consent_cols),
        )
    current_consent = consent[available_consent_cols].copy()
    current_consent = current_consent.rename(columns=lambda c: f"current_consent_{c}")

    LOGGER.info("Building referrals_enriched via canonical joins")
    referrals_enriched = referrals.merge(client_slim, on="client_id", how="left", validate="m:1")
    referrals_enriched = referrals_enriched.merge(
        referring_org,
        left_on="referring_org_id",
        right_on="referring_org_org_id",
        how="left",
        validate="m:1",
    )
    referrals_enriched = referrals_enriched.merge(
        receiving_org,
        left_on="receiving_org_id",
        right_on="receiving_org_org_id",
        how="left",
        validate="m:1",
    )
    referrals_enriched = referrals_enriched.merge(
        current_consent,
        left_on="consent_record_id",
        right_on="current_consent_consent_id",
        how="left",
        validate="m:1",
    )

    LOGGER.info(
        "Built referrals_enriched with %s rows and %s columns",
        len(referrals_enriched),
        len(referrals_enriched.columns),
    )
    return referrals_enriched
