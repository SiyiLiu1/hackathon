"""Safe Client View page.

Policy is evaluated before any cross-agency fields are rendered.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from dataclasses import asdict

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.loaders import build_referrals_enriched, load_track1_data  # noqa: E402
from src.policy_engine import PolicyDecision, evaluate_access, evaluate_policy  # noqa: E402

DEFAULT_TRACK1_RAW_DIR = str(_ROOT / "tracks" / "referral-care-coordination" / "data" / "raw")

RESTRICTION_CATEGORIES: list[tuple[str, str]] = [
    ("personal_details", "Personal details"),
    ("contact_information", "Contact information"),
    ("assessment_risk", "Assessment and risk information"),
    ("referral_service_history", "Referral and service history"),
    ("org_care_coordination", "Organization and care coordination details"),
]


def _get_tables() -> dict[str, pd.DataFrame]:
    """Fetch cached tables from session or load from configured directory."""
    if "shared_track1_tables" in st.session_state:
        return st.session_state["shared_track1_tables"]
    raw_dir = st.session_state.get("track1_raw_dir", DEFAULT_TRACK1_RAW_DIR)
    tables = _load_shared_track1_data(raw_dir)
    st.session_state["shared_track1_tables"] = tables
    return tables


@st.cache_data(show_spinner=False)
def _load_shared_track1_data(raw_dir: str) -> dict[str, pd.DataFrame]:
    """Load Track 1 tables once and cache for page navigation."""
    tables = load_track1_data(raw_dir=raw_dir)
    tables["referrals_enriched"] = build_referrals_enriched(tables)
    return tables


def _safe_payload(row: pd.Series, decision: PolicyDecision) -> dict[str, Any]:
    """Build row payload that includes only policy-allowed fields."""
    row_dict = row.to_dict()
    return {field: row_dict.get(field) for field in decision.allowed_fields if field in row_dict}


def _issue_icon(issue: dict[str, str]) -> str:
    """Choose icon by severity for policy issue card."""
    severity = issue.get("severity", "info")
    if severity == "error":
        return "🛑"
    if severity == "warning":
        return "🟠"
    return "ℹ️"


def _render_action_bar(access_decision: str, issues: list[dict[str, str]]) -> None:
    """Render top decision/action bar with aggregated policy issues."""
    with st.container(border=True):
        if access_decision == "blocked":
            st.error("### Action required: Cannot proceed")
            actions = ["Request updated consent", "Escalate to OCAP/privacy reviewer", "Pause cross-agency sharing"]
        elif access_decision == "limited":
            st.warning(
                "### Limited access\n\nSome information is hidden, but a limited workflow may still be available."
            )
            actions = [
                "Continue with limited information",
                "Request broader consent",
                "Manual review",
            ]
        else:
            st.success(
                "### Ready to proceed\n\nThis client record is available under current consent and sharing rules."
            )
            actions = ["Create referral", "View client summary", "Coordinate with receiving organization"]

        if issues:
            st.markdown("Active policy issues:")
            severity_class = {"error": "issue-error", "warning": "issue-warning", "info": "issue-info"}
            for issue in issues:
                css_class = severity_class.get(issue.get("severity", "info"), "issue-info")
                title = issue.get("title", "Policy issue")
                explanation = issue.get("explanation", "Policy check requires review.")
                action = issue.get("recommended_action", "Escalate to privacy/admin review.")
                icon = _issue_icon(issue)
                st.markdown(
                    (
                        f'<div class="issue-card {css_class}">'
                        f'<div class="issue-title">{icon} {title}</div>'
                        f'<div class="issue-explanation">{explanation}</div>'
                        f'<div class="issue-action"><strong>Recommended action:</strong> {action}</div>'
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

        st.markdown("**Primary actions**")
        action_cols = st.columns(3)
        for idx, action_text in enumerate(actions):
            with action_cols[idx]:
                st.info(action_text)


def _render_next_actions(access_decision: str, decision: PolicyDecision, issues: list[dict[str, str]]) -> None:
    """Render caseworker-oriented next actions."""
    if issues:
        actions = list(dict.fromkeys([issue.get("recommended_action", "") for issue in issues if issue.get("recommended_action")]))
    elif access_decision == "ready":
        actions = [
            "Create cross-agency referral",
            "Review care history",
            "Coordinate with receiving organization",
        ]
    elif access_decision == "limited":
        actions = [
            "Review safe summary",
            "Continue within allowed scope",
            "Request broader consent before cross-agency referral",
        ]
    else:
        actions = [
            "Do not share cross-agency information",
            "Resolve consent or governance issue first",
            "Escalate to privacy/admin review",
        ]

    st.subheader("What you can do next")
    if len(actions) <= 3:
        cards = st.columns(max(1, len(actions)))
        for idx, action_text in enumerate(actions):
            with cards[idx]:
                with st.container(border=True):
                    st.markdown(action_text)
    else:
        with st.container(border=True):
            for action_text in actions:
                st.markdown(f"- {action_text}")



def _pretty_field_name(name: str) -> str:
    """Convert field keys to readable labels."""
    return name.replace("current_consent_", "consent: ").replace("_", " ").strip().title()


def _friendly_label(name: str) -> str:
    """Convert field names into caseworker-friendly labels."""
    mapping = {
        "client_id": "Client ID",
        "primary_org_id": "Primary agency",
        "vi_spdat_score": "Assessment score",
        "assessment_date": "Last assessment",
        "status": "Current status",
        "submitted_at": "Referral submitted",
        "current_consent_status": "Consent status",
        "current_consent_sharing_scope_type": "Sharing scope",
    }
    return mapping.get(name, _pretty_field_name(name))


def _format_value(value: Any) -> str:
    """Format values for caseworker readability."""
    if value is None or pd.isna(value):
        return "Not recorded"
    ts = pd.to_datetime(value, errors="coerce")
    if not pd.isna(ts):
        return ts.strftime("%b %d, %Y")
    return str(value)


def _format_snapshot_value(value: Any) -> str:
    """Format and humanize values shown in the client snapshot."""
    base = _format_value(value)
    mapping = {
        "single_agency_only": "Single-agency only",
        "all_dsa_agencies": "All partner agencies",
        "accepted": "Accepted",
        "active": "Active",
        "pending": "Pending",
    }
    return mapping.get(base.strip().lower(), base)


def _render_snapshot_item(label: str, value: Any) -> None:
    """Render one snapshot item with constrained typography."""
    display_value = _format_snapshot_value(value)
    st.markdown(f'<div class="snapshot-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="snapshot-value">{display_value}</div>', unsafe_allow_html=True)


def _risk_level(client_row: pd.Series) -> str:
    """Compute a simple risk-level label from assessment score."""
    score = pd.to_numeric(client_row.get("vi_spdat_score"), errors="coerce")
    if pd.isna(score):
        return "Not assessed"
    if score >= 10:
        return "High"
    if score >= 5:
        return "Moderate"
    return "Low"


def _restriction_group_for_field(field_name: str) -> str:
    """Map a field name to caseworker-readable restriction categories."""
    name = str(field_name).lower()

    if any(token in name for token in ("client_id", "first_name", "last_name", "middle_name", "alias", "dob")):
        return "personal_details"
    if any(token in name for token in ("address", "city", "province", "postal", "phone", "email", "location")):
        return "contact_information"
    if any(token in name for token in ("assessment", "vi_spdat", "acuity", "risk", "health", "mental", "substance", "physical", "developmental")):
        return "assessment_risk"
    if any(token in name for token in ("referral", "encounter", "status", "submitted_at", "decision_at", "service")):
        return "referral_service_history"
    return "org_care_coordination"


def _collect_restriction_categories(fields: list[str]) -> set[str]:
    """Collect restriction categories represented by field names."""
    labels: set[str] = set()
    for field in fields:
        labels.add(_restriction_group_for_field(field))
    return labels


def _ordered_restriction_labels(category_keys: set[str]) -> list[str]:
    """Convert category keys to display labels in canonical order."""
    return [label for key, label in RESTRICTION_CATEGORIES if key in category_keys]


def _render_information_restrictions(decision: PolicyDecision) -> None:
    """Render collapsed human-readable restrictions summary."""
    with st.expander("🔒 Information restrictions", expanded=False):
        if decision.view_status == "full":
            st.success("No restrictions are currently applied in this view.")
            return

        if decision.view_status == "blocked":
            category_keys = {key for key, _ in RESTRICTION_CATEGORIES}
        else:
            category_keys = _collect_restriction_categories(decision.redacted_fields)

        labels = _ordered_restriction_labels(category_keys)
        if not labels:
            st.info("Restriction categories are not available for this record.")
            return

        st.markdown("The following information categories are restricted:")
        for label in labels:
            st.markdown(f"- {label}")
        st.caption("Technical field-level restrictions are available in Technical details for admins.")


def _render_admin_technical_details(decision: PolicyDecision) -> None:
    """Render admin-only technical details in collapsed expander."""
    with st.expander("Technical details for admins", expanded=False):
        st.json(asdict(decision))
        st.markdown("**reasons**")
        st.write(decision.reasons)
        st.markdown("**flags**")
        st.write(decision.flags)
        st.markdown("**allowed_fields**")
        st.write(decision.allowed_fields)
        st.markdown("**redacted_fields**")
        st.write(decision.redacted_fields)


def _select_latest_referral(client_referrals: pd.DataFrame) -> pd.Series | None:
    """Pick latest referral row for policy context."""
    if client_referrals.empty:
        return None
    rows = client_referrals.copy()
    if "submitted_at" in rows.columns:
        rows["submitted_at_dt"] = pd.to_datetime(rows["submitted_at"], errors="coerce")
        rows = rows.sort_values("submitted_at_dt", ascending=False)
    return rows.iloc[0]


def _build_policy_record(
    client_row: pd.Series, current_consent_row: pd.Series | None, referral_row: pd.Series | None
) -> dict[str, Any]:
    """Build a policy context record without rendering raw values."""
    record = {}
    record.update(client_row.to_dict())
    if current_consent_row is not None:
        consent_payload = current_consent_row.to_dict()
        for key, value in consent_payload.items():
            record[f"current_consent_{key}"] = value
    if referral_row is not None:
        record.update(referral_row.to_dict())
    else:
        # Required referral-like fields for policy evaluation fallback.
        record.setdefault("referral_id", "N/A")
        record.setdefault("referring_org_id", client_row.get("primary_org_id"))
        record.setdefault("receiving_org_id", client_row.get("primary_org_id"))
        record.setdefault("consent_record_id", client_row.get("current_consent_id"))
    return record


def main() -> None:
    """Render Safe Client View."""
    st.set_page_config(page_title="Safe Client View", page_icon=":lock:", layout="wide")
    if "track1_raw_dir" not in st.session_state:
        st.session_state["track1_raw_dir"] = DEFAULT_TRACK1_RAW_DIR
    with st.sidebar:
        st.markdown("## CareMatch Safe")
        st.caption("Policy-first referral coordination")
        st.markdown("### Workspace")
        st.text_input("Track 1 raw data directory", key="track1_raw_dir")
        st.caption("Local demo dataset")
    st.title("Safe Client View")
    st.caption("Caseworker workflow with policy-aware access controls.")
    st.markdown(
        """
        <style>
        .issue-card {
            border: 1px solid #e5e7eb;
            border-left-width: 4px;
            border-radius: 0.5rem;
            padding: 0.65rem 0.75rem;
            margin: 0.5rem 0;
            background: #ffffff;
        }
        .issue-error {
            border-left-color: #dc2626;
            background: #fef2f2;
        }
        .issue-warning {
            border-left-color: #d97706;
            background: #fffbeb;
        }
        .issue-info {
            border-left-color: #2563eb;
            background: #eff6ff;
        }
        .issue-title {
            font-size: 1rem;
            font-weight: 600;
            color: #111827;
            margin-bottom: 0.2rem;
        }
        .issue-explanation {
            font-size: 0.92rem;
            color: #374151;
        }
        .issue-action {
            font-size: 0.92rem;
            color: #1f2937;
            margin-top: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        tables = _get_tables()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    clients = tables["clients"]
    consent = tables["consent"]
    referrals_enriched = tables["referrals_enriched"]
    encounters = tables["encounters"]

    if "client_id" not in clients.columns:
        st.error("Missing required column in clients: client_id")
        return

    client_ids = sorted(clients["client_id"].astype(str).dropna().unique().tolist())
    if not client_ids:
        st.info("No clients available.")
        return

    selected_client_id = st.selectbox("Choose client_id", options=client_ids, index=0)

    # Prepare policy context first; do not render raw client row before this point.
    client_row = clients[clients["client_id"].astype(str) == str(selected_client_id)].iloc[0]

    current_consent_row = None
    current_consent_id = client_row.get("current_consent_id")
    if current_consent_id is not None and "consent_id" in consent.columns:
        m = consent[consent["consent_id"].astype(str) == str(current_consent_id)]
        if not m.empty:
            current_consent_row = m.iloc[0]

    client_referrals = referrals_enriched[
        referrals_enriched["client_id"].astype(str) == str(selected_client_id)
    ].copy()
    latest_referral_row = _select_latest_referral(client_referrals)

    policy_record = _build_policy_record(client_row, current_consent_row, latest_referral_row)
    decision = evaluate_policy(
        policy_record,
        is_multi_agency_view=True,
        action="view",
        all_fields=policy_record.keys(),
    )
    access_eval = evaluate_access(
        client_id=str(selected_client_id),
        intended_action="view",
        data=tables,
    )
    access_decision = str(access_eval.get("decision", "blocked"))
    issues = access_eval.get("issues", [])

    latest_status = latest_referral_row.get("status") if latest_referral_row is not None else "No active referral"
    _render_action_bar(access_decision, issues)

    consent_status = "Not recorded"
    sharing_scope = "Not recorded"
    if current_consent_row is not None:
        consent_status = _format_value(current_consent_row.get("status"))
        sharing_scope = _format_value(current_consent_row.get("sharing_scope_type"))
    else:
        consent_status = _format_value(client_row.get("current_consent_status"))
        sharing_scope = _format_value(client_row.get("current_consent_sharing_scope_type"))

    st.subheader("Client snapshot")
    st.markdown(
        """
        <style>
        .snapshot-value {
            font-size: 22px;
            font-weight: 500;
            line-height: 1.25;
        }
        .snapshot-label {
            font-size: 14px;
            color: #6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        row_a = st.columns(3)
        with row_a[0]:
            _render_snapshot_item("Client ID", client_row.get("client_id"))
        with row_a[1]:
            _render_snapshot_item("Risk level", _risk_level(client_row))
        with row_a[2]:
            _render_snapshot_item("Current status", latest_status)

        row_b = st.columns(3)
        with row_b[0]:
            _render_snapshot_item("Last assessment", client_row.get("assessment_date"))
        with row_b[1]:
            _render_snapshot_item("Consent status", consent_status)
        with row_b[2]:
            _render_snapshot_item("Sharing scope", sharing_scope)

    st.caption(f"Legal basis: {access_eval.get('legal_basis', 'Not recorded')}")

    _render_next_actions(access_decision, decision, issues)
    _render_information_restrictions(decision)

    with st.expander("Client details", expanded=False):
        if decision.view_status == "blocked":
            st.info("Client details are unavailable in this cross-agency view.")
        else:
            safe_client = _safe_payload(client_row, decision)
            if safe_client:
                safe_client_rows = [
                    {"Field": _friendly_label(field), "Value": _format_value(value)}
                    for field, value in safe_client.items()
                ]
                st.dataframe(pd.DataFrame(safe_client_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No client fields are available in this view.")

    with st.expander("Consent details", expanded=False):
        if current_consent_row is None:
            st.info("No current consent record found for this client.")
        elif decision.view_status == "blocked":
            st.info("Consent details are unavailable in this cross-agency view.")
        else:
            safe_consent = _safe_payload(
                pd.Series({f"current_consent_{k}": v for k, v in current_consent_row.to_dict().items()}),
                decision,
            )
            if safe_consent:
                safe_consent_rows = [
                    {"Field": _friendly_label(field), "Value": _format_value(value)}
                    for field, value in safe_consent.items()
                ]
                st.dataframe(pd.DataFrame(safe_consent_rows), use_container_width=True, hide_index=True)
            else:
                st.info("Consent details are restricted by current policy rules.")

    with st.expander("Referral history", expanded=False):
        if client_referrals.empty:
            st.info("No referrals found for this client.")
        else:
            referral_summary = (
                client_referrals.groupby("status", dropna=False)
                .size()
                .rename("count")
                .reset_index()
                .sort_values("count", ascending=False)
            )
            st.dataframe(referral_summary, use_container_width=True, hide_index=True)

    with st.expander("Service history", expanded=False):
        if "client_id" not in encounters.columns:
            st.info("Service history unavailable: encounters are missing client_id.")
        else:
            client_encounters = encounters[encounters["client_id"].astype(str) == str(selected_client_id)].copy()
            if client_encounters.empty:
                st.info("No service encounters found for this client.")
            else:
                encounter_summary = (
                    client_encounters.groupby("encounter_type", dropna=False)
                    .size()
                    .rename("count")
                    .reset_index()
                    .sort_values("count", ascending=False)
                )
                st.dataframe(encounter_summary, use_container_width=True, hide_index=True)

    _render_admin_technical_details(decision)


if __name__ == "__main__":
    main()
