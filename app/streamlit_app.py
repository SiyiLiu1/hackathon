"""CareMatch Safe Streamlit landing page for Track 1.

Run with:
    streamlit run app/streamlit_app.py

This app follows the same shell pattern as ``shared/app/streamlit_app.py``
but is specialized for Track 1 Inter-Org Referral & Care Coordination.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.loaders import build_referrals_enriched, load_track1_data  # noqa: E402

DEFAULT_TRACK1_RAW_DIR = str(_ROOT / "tracks" / "referral-care-coordination" / "data" / "raw")
TRACK1_LABEL = "Track 1 — Referral & Care Coordination"
DEFAULT_TRACK1_RAW_DIR = os.environ.get("TRACK1_DATA_DIR", DEFAULT_TRACK1_RAW_DIR)


@st.cache_data(show_spinner=False)
def load_shared_track1_data(raw_dir: str) -> Dict[str, pd.DataFrame]:
    """Load Track 1 tables once and cache for page navigation."""
    tables = load_track1_data(raw_dir=raw_dir)
    tables["referrals_enriched"] = build_referrals_enriched(tables)
    return tables


NOT_AVAILABLE = "Not available from dataset"


def _metric_entry(
    value: int | float | str,
    source_table: str,
    filter_logic: str,
    required_columns: list[str],
) -> dict[str, Any]:
    """Build a metric record including provenance metadata."""
    return {
        "value": value,
        "source_table": source_table,
        "filter_logic": filter_logic,
        "required_columns": required_columns,
    }


def _table_has_columns(data: Dict[str, pd.DataFrame], table: str, required_columns: list[str]) -> bool:
    """Return True when table exists and contains all required columns."""
    if table not in data:
        return False
    return set(required_columns).issubset(set(data[table].columns))


def _format_metric_value(value: int | float | str) -> str:
    """Format metric values for Streamlit metric widgets."""
    if isinstance(value, (int, float)):
        return f"{int(value):,}" if float(value).is_integer() else f"{value:,.2f}"
    return str(value)


def _kpi(
    label: str,
    value: int | float | str,
    help_text: str | None = None,
    help: str | None = None,
) -> None:
    """Render KPI metric with the same tooltip behavior as shared app."""
    tooltip = help_text if help_text is not None else help
    st.metric(label=label, value=_format_metric_value(value), help=tooltip)


def _numeric_metric_value(metrics: dict[str, dict[str, Any]], metric_name: str) -> int | None:
    """Return int value for metric if numeric, otherwise None."""
    value = metrics.get(metric_name, {}).get("value")
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _consent_issue_count(consent_df: pd.DataFrame) -> int | None:
    """Compute consent issue count from real consent columns only."""
    available_any = False
    issue_mask = pd.Series(False, index=consent_df.index)
    now = pd.Timestamp.utcnow().tz_localize(None)

    if "status" in consent_df.columns:
        available_any = True
        status_norm = consent_df["status"].astype(str).str.lower().str.strip()
        issue_mask = issue_mask | status_norm.isin({"withdrawn", "expired"}) | status_norm.str.contains("invalid")
    if "withdrawal_date" in consent_df.columns:
        available_any = True
        issue_mask = issue_mask | pd.to_datetime(consent_df["withdrawal_date"], errors="coerce").notna()
    if "expires_at" in consent_df.columns:
        available_any = True
        expires_at = pd.to_datetime(consent_df["expires_at"], errors="coerce", utc=True).dt.tz_localize(None)
        issue_mask = issue_mask | (expires_at.notna() & (expires_at < now))
    if "purpose_codes" in consent_df.columns:
        available_any = True
        purpose_text = consent_df["purpose_codes"].astype(str).str.strip().str.lower()
        issue_mask = issue_mask | consent_df["purpose_codes"].isna() | purpose_text.isin({"", "[]", "none", "nan"})
    if "sharing_scope_type" in consent_df.columns:
        available_any = True
        scope_norm = consent_df["sharing_scope_type"].astype(str).str.lower().str.strip()
        issue_mask = issue_mask | scope_norm.isin({"no_sharing", "single_agency_only", "anonymous_only"})

    if not available_any:
        return None
    return int(issue_mask.sum())


def compute_metrics(data: Dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Compute dashboard metrics with strict provenance."""
    metrics: dict[str, dict[str, Any]] = {}

    core_table_metrics = [
        ("clients", "clients"),
        ("referrals_total", "referrals"),
        ("service_encounters", "encounters"),
        ("consent_records", "consent"),
        ("duplicate_flags_total", "duplicate_flags"),
    ]
    for metric_name, table_name in core_table_metrics:
        if table_name not in data:
            metrics[metric_name] = _metric_entry(
                NOT_AVAILABLE,
                table_name,
                f"Table '{table_name}' is missing.",
                [],
            )
            continue
        table_df = data[table_name]
        metrics[metric_name] = _metric_entry(
            int(len(table_df)),
            table_name,
            "Row count of loaded table.",
            [],
        )

    if _table_has_columns(data, "referrals", ["status"]):
        referral_status = data["referrals"]["status"].astype(str).str.lower().str.strip()
        pending_count = int(referral_status.isin({"pending", "submitted", "in_review", "queued"}).sum())
        metrics["pending_referrals"] = _metric_entry(
            pending_count,
            "referrals",
            "COUNT(*) WHERE status IN ('pending','submitted','in_review','queued').",
            ["status"],
        )
    else:
        metrics["pending_referrals"] = _metric_entry(
            NOT_AVAILABLE,
            "referrals",
            "Required column 'status' not found.",
            ["status"],
        )

    if "consent" in data:
        consent_issue_count = _consent_issue_count(data["consent"])
        if consent_issue_count is None:
            metrics["consent_issues"] = _metric_entry(
                NOT_AVAILABLE,
                "consent",
                "None of status, withdrawal_date, expires_at, purpose_codes, sharing_scope_type columns exist.",
                ["status", "withdrawal_date", "expires_at", "purpose_codes", "sharing_scope_type"],
            )
        else:
            metrics["consent_issues"] = _metric_entry(
                consent_issue_count,
                "consent",
                (
                    "COUNT(*) WHERE status in {'withdrawn','expired'} OR status contains 'invalid' "
                    "OR withdrawal_date IS NOT NULL OR expires_at < now OR purpose_codes missing/empty "
                    "OR sharing_scope_type in {'no_sharing','single_agency_only','anonymous_only'}."
                ),
                ["status", "withdrawal_date", "expires_at", "purpose_codes", "sharing_scope_type"],
            )
    else:
        metrics["consent_issues"] = _metric_entry(
            NOT_AVAILABLE,
            "consent",
            "Table 'consent' is missing.",
            ["status", "withdrawal_date", "expires_at", "purpose_codes", "sharing_scope_type"],
        )

    if "duplicate_flags" in data:
        dup = data["duplicate_flags"]
        if "is_true_duplicate" in dup.columns:
            true_positive = int(dup["is_true_duplicate"].fillna(False).astype(bool).sum())
            true_positive_logic = "COUNT(*) WHERE is_true_duplicate = TRUE."
            true_positive_required = ["is_true_duplicate"]
        elif "review_status" in dup.columns:
            review_status = dup["review_status"].astype(str).str.lower().str.strip()
            true_positive = int(review_status.isin({"confirmed_duplicate", "merged"}).sum())
            true_positive_logic = "COUNT(*) WHERE review_status IN ('confirmed_duplicate','merged')."
            true_positive_required = ["review_status"]
        else:
            true_positive = NOT_AVAILABLE
            true_positive_logic = "Required columns 'is_true_duplicate' or 'review_status' not found."
            true_positive_required = ["is_true_duplicate", "review_status"]

        decoy_value: int | str = NOT_AVAILABLE
        decoy_logic = "No decoy/false-positive label column found in duplicate_flags dataset."
        decoy_required = ["review_status or decoy label column"]
        if "review_status" in dup.columns:
            review_status = dup["review_status"].astype(str).str.lower().str.strip()
            decoy_matches = review_status.isin({"decoy_false_positive", "false_positive", "decoy"})
            if decoy_matches.any():
                decoy_value = int(decoy_matches.sum())
                decoy_logic = "COUNT(*) WHERE review_status IN ('decoy_false_positive','false_positive','decoy')."
                decoy_required = ["review_status"]

        metrics["duplicate_flags_true_positive"] = _metric_entry(
            true_positive,
            "duplicate_flags",
            true_positive_logic,
            true_positive_required,
        )
        metrics["duplicate_flags_decoy_false_positive"] = _metric_entry(
            decoy_value,
            "duplicate_flags",
            decoy_logic,
            decoy_required,
        )
    else:
        metrics["duplicate_flags_true_positive"] = _metric_entry(
            NOT_AVAILABLE,
            "duplicate_flags",
            "Table 'duplicate_flags' is missing.",
            ["is_true_duplicate", "review_status"],
        )
        metrics["duplicate_flags_decoy_false_positive"] = _metric_entry(
            NOT_AVAILABLE,
            "duplicate_flags",
            "Table 'duplicate_flags' is missing.",
            ["review_status or decoy label column"],
        )

    if _table_has_columns(data, "referrals_enriched", ["current_consent_status", "current_consent_expires_at", "ocap_protected"]):
        enriched = data["referrals_enriched"].copy()
        status_norm = enriched["current_consent_status"].astype(str).str.lower().str.strip()
        expires_at = pd.to_datetime(enriched["current_consent_expires_at"], errors="coerce", utc=True).dt.tz_localize(None)
        ocap_protected = enriched["ocap_protected"].fillna(False).astype(bool)
        now = pd.Timestamp.utcnow().tz_localize(None)
        valid_consent = status_norm.isin({"active", "valid"}) & (expires_at.isna() | (expires_at >= now))

        metrics["safe_to_proceed"] = _metric_entry(
            int((valid_consent & ~ocap_protected).sum()),
            "referrals_enriched",
            "COUNT(*) WHERE consent status in {'active','valid'} AND (expires_at IS NULL OR expires_at >= now) AND ocap_protected = FALSE.",
            ["current_consent_status", "current_consent_expires_at", "ocap_protected"],
        )
        metrics["blocked_by_consent"] = _metric_entry(
            int((~valid_consent).sum()),
            "referrals_enriched",
            "COUNT(*) WHERE consent status not in {'active','valid'} OR expires_at < now.",
            ["current_consent_status", "current_consent_expires_at"],
        )
        metrics["ocap_review_required"] = _metric_entry(
            int(ocap_protected.sum()),
            "referrals_enriched",
            "COUNT(*) WHERE ocap_protected = TRUE.",
            ["ocap_protected"],
        )
    else:
        missing_cols = ["current_consent_status", "current_consent_expires_at", "ocap_protected"]
        metrics["safe_to_proceed"] = _metric_entry(
            NOT_AVAILABLE,
            "referrals_enriched",
            "Required columns missing.",
            missing_cols,
        )
        metrics["blocked_by_consent"] = _metric_entry(
            NOT_AVAILABLE,
            "referrals_enriched",
            "Required columns missing.",
            missing_cols,
        )
        metrics["ocap_review_required"] = _metric_entry(
            NOT_AVAILABLE,
            "referrals_enriched",
            "Required columns missing.",
            missing_cols,
        )

    return metrics


def _table_inventory_rows(tables: Dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Build inventory rows for loaded table diagnostics."""
    rows: list[dict[str, Any]] = []
    for table_name, table_df in tables.items():
        rows.append(
            {
                "table_name": table_name,
                "row_count": int(len(table_df)),
                "column_count": int(len(table_df.columns)),
                "columns": ", ".join(table_df.columns.astype(str).tolist()),
            }
        )
    return rows


def _print_table_inventory(tables: Dict[str, pd.DataFrame]) -> None:
    """Print table names, row counts, and column names for auditability."""
    print("[Track1] Loaded table inventory")
    for table_name, table_df in tables.items():
        print(f"[Track1] table={table_name} rows={len(table_df)} cols={len(table_df.columns)}")
        print(f"[Track1] columns={', '.join(table_df.columns.astype(str).tolist())}")


def _sanity_warning_message(metric_name: str, actual: int, expected: int) -> str | None:
    """Return warning message when a key table count is outside expected range."""
    tolerance = 0.35
    if expected <= 0:
        return None
    pct_delta = abs(actual - expected) / expected
    if pct_delta <= tolerance:
        return None
    return (
        f"{metric_name} is {actual:,}, which is outside the expected around-{expected:,} range "
        f"(threshold +/- {int(tolerance * 100)}%)."
    )


def main() -> None:
    """Render app shell and expose page navigation."""
    st.set_page_config(
        page_title="Home",
        page_icon=":house:",
        layout="wide",
    )

    st.title("CareMatch Safe — Referral Coordination Workspace")
    st.caption(
        "Synthetic HIFIS-inspired data covering organizations, clients, referrals, service encounters, consent, "
        "data sharing agreements, and duplicate flags."
    )
    st.markdown(
        """
        <style>
            .card {
                border: 1px solid #e6e9ef;
                border-radius: 0.75rem;
                padding: 0.9rem 1rem 0.85rem 1rem;
                background-color: #ffffff;
                box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
            }
            .card h4 {
                margin: 0 0 0.35rem 0;
                font-size: 1rem;
                font-weight: 600;
            }
            .metric-lg {
                margin: 0.05rem 0 0.35rem 0;
                font-size: 2rem;
                line-height: 1;
                font-weight: 700;
                color: #111827;
            }
            .card-desc {
                margin: 0;
                color: #475467;
                font-size: 0.9rem;
                min-height: 2.5rem;
            }
            .chip-row {
                margin-top: 0.45rem;
            }
            .chip {
                display: inline-block;
                border-radius: 999px;
                padding: 0.12rem 0.48rem;
                margin: 0 0.35rem 0.35rem 0;
                font-size: 0.72rem;
                font-weight: 600;
                border: 1px solid transparent;
            }
            .chip-urgent {
                color: #b42318;
                background: #fef3f2;
                border-color: #fecdca;
            }
            .chip-pending {
                color: #175cd3;
                background: #eff8ff;
                border-color: #b2ddff;
            }
            .chip-consent {
                color: #b54708;
                background: #fffaeb;
                border-color: #fedf89;
            }
            .chip-neutral {
                color: #344054;
                background: #f2f4f7;
                border-color: #d0d5dd;
            }
            .priority-card {
                border: 1px solid #e4e7ec;
                border-radius: 0.75rem;
                padding: 0.65rem 0.8rem;
                background: #fcfcfd;
                font-size: 0.9rem;
                font-weight: 600;
                color: #1d2939;
            }
            .workflow-step {
                border-left: 3px solid #d0d5dd;
                padding: 0.2rem 0 0.2rem 0.75rem;
                margin-bottom: 0.45rem;
                color: #344054;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "track1_raw_dir" not in st.session_state:
        st.session_state["track1_raw_dir"] = DEFAULT_TRACK1_RAW_DIR

    with st.sidebar:
        st.markdown("## CareMatch Safe")
        st.caption("Policy-first referral coordination")
        st.markdown("### Workspace")
        raw_dir = st.text_input("Track 1 raw data directory", key="track1_raw_dir")
        st.caption("Local demo dataset")

    try:
        tables = load_shared_track1_data(raw_dir)
    except Exception as exc:
        st.error(f"Failed to load shared data: {exc}")
        return

    st.session_state["shared_track1_tables"] = tables

    _print_table_inventory(tables)
    metrics = compute_metrics(tables)
    inventory_df = pd.DataFrame(_table_inventory_rows(tables))

    expected_ranges = {
        "clients": 800,
        "referrals_total": 3000,
        "service_encounters": 10000,
        "consent_records": 5000,
        "duplicate_flags_total": 500,
    }
    sanity_warnings: list[str] = []
    for metric_name, expected_value in expected_ranges.items():
        actual_value = _numeric_metric_value(metrics, metric_name)
        if actual_value is None:
            continue
        warning = _sanity_warning_message(metric_name, actual_value, expected_value)
        if warning:
            sanity_warnings.append(warning)

    if sanity_warnings:
        st.warning("Dataset sanity checks found unexpected volume:")
        for warning in sanity_warnings:
            st.caption(f"- {warning}")
    st.caption("Decision Support Dashboard")

    clients = tables["clients"]
    referrals = tables["referrals"]
    encounters = tables["encounters"]
    consent = tables["consent"]
    # Match shared/app behavior: duplicate-pair KPI uses the dup_flags table.
    # Our loader key is `duplicate_flags`, so we support both names.
    dup_flags = tables.get("dup_flags", tables["duplicate_flags"])
    chronic_rate = (
        (clients["chronic_homeless_flag"] == True).mean() * 100  # noqa: E712
        if "chronic_homeless_flag" in clients.columns
        else 0.0
    )
    active_consent_pct = (consent["status"] == "active").mean() * 100 if "status" in consent.columns else 0.0

    metric_cards = [
        ("Clients", len(clients), "Unique rows in clients"),
        ("Referrals", len(referrals), "Total referral transactions"),
        ("Encounters", len(encounters), "Service encounters recorded"),
        ("Chronic rate", f"{chronic_rate:.1f}%", "Share of clients flagged chronic"),
        ("Duplicate pairs", len(dup_flags), f"Active consents: {active_consent_pct:.1f}%"),
        (
            "Consent issues",
            metrics["consent_issues"]["value"],
            "Consent exceptions computed from withdrawn, expired, invalid, missing purpose, or sharing-restricted consent rows.",
        ),
        (
            "Pending referrals",
            metrics["pending_referrals"]["value"],
            "COUNT(*) where referrals.status is pending/submitted/in_review/queued.",
        ),
    ]
    metrics_per_row = 4
    for row_start in range(0, len(metric_cards), metrics_per_row):
        row_cols = st.columns(metrics_per_row)
        row_metrics = metric_cards[row_start : row_start + metrics_per_row]
        for idx, col in enumerate(row_cols):
            with col:
                if idx < len(row_metrics):
                    label, value, help_text = row_metrics[idx]
                    _kpi(label, value, help_text=help_text)
                else:
                    st.empty()
    st.markdown("---")
    st.subheader("Next steps")
    nav_a, nav_b, nav_c = st.columns(3)
    with nav_a:
        st.markdown("**Safe Client View** — client-level policy decision and restrictions.")
        st.page_link("pages/1_safe_client_view.py", label="Open Safe Client View", icon="🔒")
    with nav_b:
        st.markdown("**Referral Matching** — eligible organizations and policy-safe routing.")
        st.page_link("pages/2_referral_matching.py", label="Open Referral Matching", icon="🤝")
    with nav_c:
        st.markdown("**Ops Dashboard** — operational and policy risk visibility.")
        st.page_link("pages/3_ops_dashboard.py", label="Open Ops Dashboard", icon="📊")

    st.markdown("---")
    st.markdown("### Today’s priorities")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown('<div class="priority-card">Review duplicate flags</div>', unsafe_allow_html=True)
    with t2:
        st.markdown('<div class="priority-card">Resolve consent issues</div>', unsafe_allow_html=True)
    with t3:
        st.markdown('<div class="priority-card">Process pending referrals</div>', unsafe_allow_html=True)

    st.markdown("## Task Queues")
    q1, q2, q3, q4 = st.columns(4)

    with q1:
        with st.container(border=False):
            st.markdown(
                f"""
                <div class="card">
                    <h4>🚩 Duplicate flags</h4>
                    <p class="metric-lg">{_format_metric_value(metrics['duplicate_flags_total']['value'])}</p>
                    <p class="card-desc">Records flagged by duplicate detection from Track 1 duplicate_flags data.</p>
                    <div class="chip-row">
                        <span class="chip chip-urgent">data quality</span>
                        <span class="chip chip-urgent">review</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("Review duplicate flags", key="btn_review_duplicate_flags", use_container_width=True):
            st.switch_page("pages/1_safe_client_view.py")

    with q2:
        with st.container(border=False):
            st.markdown(
                f"""
                <div class="card">
                    <h4>📥 Pending referrals</h4>
                    <p class="metric-lg">{_format_metric_value(metrics['pending_referrals']['value'])}</p>
                    <p class="card-desc">Referrals awaiting triage, matching, or next-step handoff to service teams.</p>
                    <div class="chip-row">
                        <span class="chip chip-pending">in progress</span>
                        <span class="chip chip-pending">blue queue</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("Open referral queue", key="btn_open_pending_referrals", use_container_width=True):
            st.switch_page("pages/2_referral_matching.py")

    with q3:
        with st.container(border=False):
            st.markdown(
                f"""
                <div class="card">
                    <h4>⚠️ Consent issues</h4>
                    <p class="metric-lg">{_format_metric_value(metrics['consent_issues']['value'])}</p>
                    <p class="card-desc">Cases that need consent verification before information can be shared safely.</p>
                    <div class="chip-row">
                        <span class="chip chip-consent">needs review</span>
                        <span class="chip chip-consent">yellow queue</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("Resolve consent issues", key="btn_resolve_consent_issues", use_container_width=True):
            st.switch_page("pages/3_ops_dashboard.py")

    with q4:
        with st.container(border=False):
            st.markdown(
                f"""
                <div class="card">
                    <h4>👥 Clients</h4>
                    <p class="metric-lg">{_format_metric_value(metrics['clients']['value'])}</p>
                    <p class="card-desc">Current caseload under active coordination and ongoing service follow-up.</p>
                    <div class="chip-row">
                        <span class="chip chip-neutral">caseload</span>
                        <span class="chip chip-neutral">active</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if st.button("View caseload", key="btn_open_client_caseload", use_container_width=True):
            st.switch_page("pages/1_safe_client_view.py")

    st.markdown("### Suggested workflow")
    wf1, wf2 = st.columns(2)
    with wf1:
        st.markdown('<div class="workflow-step"><strong>Step 1:</strong> Open consent issues</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-step"><strong>Step 2:</strong> Review safe client view</div>', unsafe_allow_html=True)
    with wf2:
        st.markdown('<div class="workflow-step"><strong>Step 3:</strong> Match eligible referrals</div>', unsafe_allow_html=True)
        st.markdown('<div class="workflow-step"><strong>Step 4:</strong> Escalate blocked / OCAP cases</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Policy Status")
    p1, p2, p3, p4 = st.columns([1, 1, 1, 1.5])
    p1.metric(
        "Safe-to-proceed cases",
        _format_metric_value(metrics["safe_to_proceed"]["value"]),
        help="Cases where policy checks support continuing the referral workflow.",
    )
    p2.metric(
        "Blocked by consent",
        _format_metric_value(metrics["blocked_by_consent"]["value"]),
        delta_color="inverse",
        help="Cases paused because consent requirements are not currently satisfied.",
    )
    p3.metric(
        "OCAP review required",
        _format_metric_value(metrics["ocap_review_required"]["value"]),
        delta_color="inverse",
        help="Cases that require OCAP-sensitive data governance review before sharing.",
    )
    with p4:
        st.caption("Policy checks include consent state, scope restrictions, and OCAP-sensitive data handling.")
        st.page_link(
            "pages/3_ops_dashboard.py",
            label="View Policy Decisions",
            icon=":material/gavel:",
        )

    with st.expander("Metric provenance", expanded=False):
        st.caption("Loaded table names, row counts, and column names")
        st.dataframe(inventory_df, use_container_width=True, hide_index=True)

        provenance_rows = []
        for metric_name in sorted(metrics):
            metric = metrics[metric_name]
            provenance_rows.append(
                {
                    "metric": metric_name,
                    "value": metric["value"],
                    "source_table": metric["source_table"],
                    "filter_logic": metric["filter_logic"],
                    "required_columns": ", ".join(metric["required_columns"]),
                }
            )
        st.caption("Exact metric calculation provenance")
        st.dataframe(pd.DataFrame(provenance_rows), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
