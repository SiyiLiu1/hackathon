"""Compact operations dashboard for consent gaps and referral bottlenecks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.streamlit_app import (  # noqa: E402
    DEFAULT_TRACK1_RAW_DIR,
    load_shared_track1_data,
)
from src.policy_engine import evaluate_access  # noqa: E402


def _get_tables() -> dict[str, pd.DataFrame]:
    """Return cached shared data tables for Streamlit pages."""
    if "shared_track1_tables" in st.session_state:
        return st.session_state["shared_track1_tables"]
    raw_dir = st.session_state.get("track1_raw_dir", DEFAULT_TRACK1_RAW_DIR)
    tables = load_shared_track1_data(raw_dir)
    st.session_state["shared_track1_tables"] = tables
    return tables


def _to_naive_datetime(series: pd.Series) -> pd.Series:
    """Convert datetime-like series to tz-naive pandas datetime."""
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt.dt.tz_localize(None)


def _consent_gap_overview(encounters: pd.DataFrame, consent: pd.DataFrame) -> dict[str, float]:
    """Build encounter-level consent-gap KPIs."""
    if encounters.empty or "client_id" not in encounters.columns:
        return {
            "encounters_total": 0,
            "encounters_with_active_consent": 0,
            "encounters_with_gap": 0,
            "gap_rate_pct": 0.0,
        }

    now = pd.Timestamp.utcnow().tz_convert(None)
    consent_df = consent.copy()
    if "client_id" not in consent_df.columns:
        consent_df = pd.DataFrame(columns=["client_id"])
    if "status" not in consent_df.columns:
        consent_df["status"] = ""

    consent_df["status_norm"] = consent_df["status"].astype(str).str.lower().str.strip()
    consent_df["effective_at_dt"] = _to_naive_datetime(
        consent_df["effective_at"] if "effective_at" in consent_df.columns else pd.Series(pd.NaT, index=consent_df.index)
    )
    consent_df["expires_at_dt"] = _to_naive_datetime(
        consent_df["expires_at"] if "expires_at" in consent_df.columns else pd.Series(pd.NaT, index=consent_df.index)
    )
    consent_df["withdrawal_date_dt"] = _to_naive_datetime(
        consent_df["withdrawal_date"] if "withdrawal_date" in consent_df.columns else pd.Series(pd.NaT, index=consent_df.index)
    )

    active_like = consent_df["status_norm"].isin({"active", "valid"})
    not_expired = consent_df["expires_at_dt"].isna() | (consent_df["expires_at_dt"] >= now)
    started = consent_df["effective_at_dt"].isna() | (consent_df["effective_at_dt"] <= now)
    not_withdrawn = consent_df["withdrawal_date_dt"].isna() | (consent_df["withdrawal_date_dt"] > now)
    active_consents = consent_df[active_like & not_expired & started & not_withdrawn].copy()

    active_clients = set(active_consents["client_id"].astype(str).dropna().tolist())
    encounter_clients = encounters["client_id"].astype(str)
    has_active = encounter_clients.isin(active_clients)

    total = int(len(encounters))
    with_active = int(has_active.sum())
    gaps = total - with_active
    gap_rate = (gaps / total * 100.0) if total else 0.0
    return {
        "encounters_total": total,
        "encounters_with_active_consent": with_active,
        "encounters_with_gap": gaps,
        "gap_rate_pct": gap_rate,
    }


def _status_mix(referrals: pd.DataFrame) -> pd.DataFrame:
    """Aggregate referral status mix for compact chart."""
    if referrals.empty or "status" not in referrals.columns:
        return pd.DataFrame(columns=["status", "count"])
    grouped = (
        referrals.assign(status=referrals["status"].fillna("unknown").astype(str).str.lower().str.strip())
        .groupby("status", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    grouped["status"] = grouped["status"].str.replace("_", " ").str.title()
    return grouped


def _consent_risk_counts(consent: pd.DataFrame, clients: pd.DataFrame) -> dict[str, int]:
    """Aggregate consent and governance risk counts."""
    consent_df = consent.copy()
    clients_df = clients.copy()

    if "status" not in consent_df.columns:
        consent_df["status"] = ""
    if "sharing_scope_type" not in consent_df.columns:
        consent_df["sharing_scope_type"] = ""

    status_norm = consent_df["status"].astype(str).str.lower().str.strip()
    scope_norm = consent_df["sharing_scope_type"].astype(str).str.lower().str.strip()

    ocap_count = 0
    if "ocap_protected" in clients_df.columns:
        ocap_count = int(clients_df["ocap_protected"].fillna(False).astype(bool).sum())

    return {
        "expired": int(status_norm.eq("expired").sum()),
        "withdrawn": int(status_norm.eq("withdrawn").sum()),
        "pending": int(status_norm.eq("pending").sum()),
        "single_agency_only": int(scope_norm.eq("single_agency_only").sum()),
        "ocap_protected": ocap_count,
    }


def _duplicate_summary(duplicate_flags: pd.DataFrame) -> dict[str, int]:
    """Build duplicate true-positive vs decoy summary."""
    dup = duplicate_flags.copy()
    if "review_status" not in dup.columns:
        return {"true_positive": 0, "decoy": 0, "total": int(len(dup))}
    status_norm = dup["review_status"].astype(str).str.lower().str.strip()
    true_positive = int(status_norm.isin({"confirmed_duplicate", "merged"}).sum())
    decoy = int(status_norm.isin({"decoy_false_positive", "false_positive", "decoy", "not_duplicate"}).sum())
    return {"true_positive": true_positive, "decoy": decoy, "total": int(len(dup))}


def _manual_review_queue(tables: dict[str, pd.DataFrame], max_rows: int = 100) -> pd.DataFrame:
    """Build cases requiring manual review from policy decisions."""
    clients = tables.get("clients", pd.DataFrame())
    if clients.empty or "client_id" not in clients.columns:
        return pd.DataFrame(columns=["client_id", "decision", "issue", "recommended_action"])
    rows: list[dict[str, str]] = []
    for client_id in clients["client_id"].astype(str).dropna().tolist():
        access = evaluate_access(client_id=client_id, intended_action="referral_matching", data=tables)
        decision = str(access.get("decision", "blocked"))
        if decision == "ready":
            continue
        issues = access.get("issues", [])
        first_issue = issues[0] if issues else {}
        rows.append(
            {
                "client_id": client_id,
                "decision": decision.title(),
                "issue": str(first_issue.get("title", "Policy review needed")),
                "recommended_action": str(first_issue.get("recommended_action", "Manual review")),
            }
        )
        if len(rows) >= max_rows:
            break
    return pd.DataFrame(rows)


def _median_time_to_decision_hours(referrals: pd.DataFrame) -> float:
    """Compute median referral decision time in hours."""
    needed = {"submitted_at", "decision_at"}
    if referrals.empty or not needed.issubset(referrals.columns):
        return 0.0

    df = referrals.copy()
    df["submitted_at_dt"] = _to_naive_datetime(df["submitted_at"])
    df["decision_at_dt"] = _to_naive_datetime(df["decision_at"])
    df["decision_hours"] = (df["decision_at_dt"] - df["submitted_at_dt"]).dt.total_seconds() / 3600.0
    valid = df["decision_hours"].dropna()
    if valid.empty:
        return 0.0
    return float(valid.median())


def _pair_bottlenecks(referrals: pd.DataFrame, orgs: pd.DataFrame) -> pd.DataFrame:
    """Top referring->receiving pairs ranked by decline/no-show volume."""
    needed = {"referring_org_id", "receiving_org_id", "status"}
    if referrals.empty or not needed.issubset(referrals.columns):
        return pd.DataFrame(
            columns=[
                "pair",
                "total_referrals",
                "decline_or_no_show",
                "decline_or_no_show_rate_pct",
            ]
        )

    df = referrals.copy()
    df["status_norm"] = df["status"].fillna("").astype(str).str.lower().str.strip()
    negative_status = {"declined", "no_show", "no-show", "noshow", "cancelled", "canceled"}
    df["is_decline_or_no_show"] = df["status_norm"].isin(negative_status).astype(int)

    pairs = (
        df.groupby(["referring_org_id", "receiving_org_id"], dropna=False)
        .agg(
            total_referrals=("status_norm", "size"),
            decline_or_no_show=("is_decline_or_no_show", "sum"),
        )
        .reset_index()
    )
    pairs["decline_or_no_show_rate_pct"] = (
        pairs["decline_or_no_show"] / pairs["total_referrals"].clip(lower=1) * 100.0
    )

    org_names = orgs[["org_id", "org_name"]].copy() if {"org_id", "org_name"}.issubset(orgs.columns) else pd.DataFrame()
    if not org_names.empty:
        pairs = pairs.merge(
            org_names.rename(columns={"org_id": "referring_org_id", "org_name": "referring_org_name"}),
            on="referring_org_id",
            how="left",
        )
        pairs = pairs.merge(
            org_names.rename(columns={"org_id": "receiving_org_id", "org_name": "receiving_org_name"}),
            on="receiving_org_id",
            how="left",
        )
        pairs["pair"] = (
            pairs["referring_org_name"].fillna(pairs["referring_org_id"].astype(str))
            + " -> "
            + pairs["receiving_org_name"].fillna(pairs["receiving_org_id"].astype(str))
        )
    else:
        pairs["pair"] = pairs["referring_org_id"].astype(str) + " -> " + pairs["receiving_org_id"].astype(str)

    pairs = pairs.sort_values(
        ["decline_or_no_show", "decline_or_no_show_rate_pct", "total_referrals"],
        ascending=[False, False, False],
    ).head(8)
    return pairs


def main() -> None:
    """Render compact judge-friendly operations dashboard."""
    st.set_page_config(page_title="Ops Dashboard", page_icon=":bar_chart:", layout="wide")

    if "track1_raw_dir" not in st.session_state:
        st.session_state["track1_raw_dir"] = DEFAULT_TRACK1_RAW_DIR

    with st.sidebar:
        st.markdown("## CareMatch Safe")
        st.caption("Policy-first referral coordination")
        st.markdown("### Workspace")
        st.text_input("Track 1 raw data directory", key="track1_raw_dir")
        st.caption("Local demo dataset")

    st.title("Operations Dashboard")
    st.caption("Compact view for consent gaps and referral bottlenecks.")

    try:
        tables = _get_tables()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    encounters = tables["encounters"]
    consent = tables["consent"]
    referrals = tables["referrals"]
    orgs = tables["orgs"]
    clients = tables["clients"]
    duplicate_flags = tables["duplicate_flags"]

    st.warning("Aggregated operations metrics only. No raw cross-agency client-level details are shown here.")

    gap = _consent_gap_overview(encounters, consent)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Encounters", f"{gap['encounters_total']:,}")
    s2.metric("Active consent", f"{gap['encounters_with_active_consent']:,}")
    s3.metric("Consent gaps", f"{gap['encounters_with_gap']:,}")
    s4.metric("Gap rate", f"{gap['gap_rate_pct']:.1f}%")

    st.divider()
    st.subheader("Referral Bottlenecks")

    left, right = st.columns([1.15, 0.85])
    with left:
        st.caption("Status mix")
        status_mix = _status_mix(referrals)
        if status_mix.empty:
            st.info("Status mix unavailable.")
        else:
            fig_status = px.bar(
                status_mix,
                x="status",
                y="count",
                labels={"status": "", "count": "Referrals"},
                height=260,
            )
            fig_status.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(fig_status, use_container_width=True)

    with right:
        st.caption("Median time to decision")
        median_hours = _median_time_to_decision_hours(referrals)
        st.metric("Median decision time", f"{median_hours:.1f} h")
        st.caption("Based on referrals with both submission and decision timestamps.")

    st.caption("Top referring -> receiving pairs by decline or no-show")
    pair_bottlenecks = _pair_bottlenecks(referrals, orgs)
    if pair_bottlenecks.empty:
        st.info("Pair bottleneck view unavailable.")
    else:
        fig_pairs = px.bar(
            pair_bottlenecks.sort_values("decline_or_no_show", ascending=True),
            x="decline_or_no_show",
            y="pair",
            orientation="h",
            labels={"decline_or_no_show": "Decline/No-show count", "pair": ""},
            height=320,
        )
        fig_pairs.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig_pairs, use_container_width=True)

        compact_table = pair_bottlenecks[
            ["pair", "total_referrals", "decline_or_no_show", "decline_or_no_show_rate_pct"]
        ].copy()
        compact_table["decline_or_no_show_rate_pct"] = compact_table["decline_or_no_show_rate_pct"].round(1)
        compact_table = compact_table.rename(
            columns={
                "pair": "Referring -> Receiving",
                "total_referrals": "Total referrals",
                "decline_or_no_show": "Decline/No-show",
                "decline_or_no_show_rate_pct": "Rate %",
            }
        )
        st.dataframe(compact_table, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Policy & Governance Risk")

    consent_risks = _consent_risk_counts(consent, clients)
    duplicates = _duplicate_summary(duplicate_flags)
    review_queue = _manual_review_queue(tables=tables, max_rows=200)

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Expired consent", f"{consent_risks['expired']:,}")
    r2.metric("Withdrawn consent", f"{consent_risks['withdrawn']:,}")
    r3.metric("Pending consent", f"{consent_risks['pending']:,}")
    r4.metric("Single-agency only", f"{consent_risks['single_agency_only']:,}")
    r5.metric("OCAP-protected", f"{consent_risks['ocap_protected']:,}")

    d1, d2, d3 = st.columns(3)
    d1.metric("Duplicate flags total", f"{duplicates['total']:,}")
    d2.metric("True positives", f"{duplicates['true_positive']:,}")
    d3.metric("Decoy / not duplicate", f"{duplicates['decoy']:,}")

    blocked_count = int((review_queue["decision"].str.lower() == "blocked").sum()) if not review_queue.empty else 0
    limited_count = int((review_queue["decision"].str.lower() == "limited").sum()) if not review_queue.empty else 0
    st.warning(
        f"Red-flag policy issues detected: {blocked_count:,} blocked and {limited_count:,} limited-access cases "
        "in the current review queue sample."
    )

    st.subheader("Cases requiring manual review")
    if review_queue.empty:
        st.success("No manual review cases detected in current data.")
    else:
        st.dataframe(review_queue, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
