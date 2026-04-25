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

from app.streamlit_app import DEFAULT_TRACK1_RAW_DIR, load_shared_track1_data  # noqa: E402
from src.policy_engine import PolicyDecision, evaluate_policy  # noqa: E402

CATEGORIES: list[tuple[str, str]] = [
    ("basic_info", "Basic client information (name, client ID)"),
    ("contact", "Contact and location information"),
    ("consent", "Consent and sharing status"),
    ("assessment", "Assessment and risk information"),
    ("services", "Referral and service history"),
    ("organization", "Organization and care coordination details"),
]


def _get_tables() -> dict[str, pd.DataFrame]:
    """Fetch cached tables from session or load from configured directory."""
    if "shared_track1_tables" in st.session_state:
        return st.session_state["shared_track1_tables"]
    raw_dir = st.session_state.get("track1_raw_dir", DEFAULT_TRACK1_RAW_DIR)
    tables = load_shared_track1_data(raw_dir)
    st.session_state["shared_track1_tables"] = tables
    return tables


def _safe_payload(row: pd.Series, decision: PolicyDecision) -> dict[str, Any]:
    """Build row payload that includes only policy-allowed fields."""
    row_dict = row.to_dict()
    return {field: row_dict.get(field) for field in decision.allowed_fields if field in row_dict}


def render_policy_status(decision: PolicyDecision) -> None:
    """Render policy-specific status messaging separate from field visibility."""
    flags = decision.flags or {}

    if flags.get("ocap_restriction"):
        st.warning("🟣 OCAP review required")
        st.write(
            "This client has Indigenous data-governance protections. "
            "Cross-agency sharing requires explicit review before information can be shared."
        )
        st.markdown("**Recommended action:** Manual OCAP / privacy review")
    elif flags.get("expired_consent"):
        st.error("🔴 Consent update needed")
        st.write(
            "This client record cannot be shown in a cross-agency view because the current consent is expired."
        )
        st.markdown("**Recommended action:** Request updated consent")
    elif flags.get("scope_mismatch"):
        st.warning("🟡 Cross-agency sharing not allowed")
        st.write(
            "Current consent only allows single-agency use, so this record cannot be shown in a cross-agency view."
        )
        st.markdown("**Recommended action:** Request cross-agency consent or use single-agency workflow")
    elif flags.get("missing_consent"):
        st.error("🔴 Consent record missing")
        st.write("No active consent record was found for this client.")
        st.markdown("**Recommended action:** Verify consent before sharing information")
    elif decision.view_status == "full":
        st.success("🟢 Safe to view client record")
        st.write("This client record is available under the current consent and sharing rules.")
    elif decision.view_status == "redacted":
        st.warning("🟡 Limited view")
        st.write("Some client information is hidden because of consent or privacy rules.")
    else:
        st.error("🔴 Access restricted")
        st.write("This client record cannot be shown in a cross-agency view right now.")
        st.markdown("**Recommended action:** Manual privacy review")


def _render_blocked_next_steps(decision: PolicyDecision) -> None:
    """Render blocked-state next steps with policy-specific guidance."""
    st.markdown("#### Suggested next steps")
    if decision.flags.get("ocap_restriction"):
        st.markdown("- Do not share cross-agency information automatically")
        st.markdown("- Confirm OCAP governing scope")
        st.markdown("- Escalate to OCAP / privacy reviewer")
        st.markdown("- Confirm whether explicit cross-agency consent exists")
    elif decision.flags.get("expired_consent"):
        st.markdown("- Request updated consent")
        st.markdown("- Confirm whether a newer consent record exists")
        st.markdown("- Escalate to privacy/admin review")
    elif decision.flags.get("scope_mismatch"):
        st.markdown("- Use single-agency workflow if appropriate")
        st.markdown("- Request cross-agency consent")
        st.markdown("- Manual privacy review")
    elif decision.flags.get("missing_consent"):
        st.markdown("- Verify consent record")
        st.markdown("- Ask client to provide consent")
        st.markdown("- Manual privacy review")
    else:
        st.markdown("- Check consent status")
        st.markdown("- Request updated consent")
        st.markdown("- Use single-agency view if appropriate")
        st.markdown("- Escalate to privacy/admin review")


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


def _category_for_field(field_name: str) -> str:
    """Map a field name to one of the shared client-view categories."""
    name = str(field_name).lower()

    if any(token in name for token in ("client_id", "first_name", "last_name", "middle_name", "alias", "dob")):
        return "basic_info"
    if any(token in name for token in ("address", "city", "province", "postal", "phone", "email", "location")):
        return "contact"
    if "consent" in name or any(token in name for token in ("sharing_scope", "legal_basis", "purpose_codes", "dsa_")):
        return "consent"
    if any(token in name for token in ("assessment", "vi_spdat", "acuity", "risk", "health", "mental", "substance", "physical", "developmental")):
        return "assessment"
    if any(token in name for token in ("referral", "encounter", "status", "submitted_at", "decision_at", "service")):
        return "services"
    return "organization"


def _collect_category_labels(fields: list[str]) -> set[str]:
    """Collect category keys represented by a list of field names."""
    labels: set[str] = set()
    for field in fields:
        labels.add(_category_for_field(field))
    return labels


def _ordered_category_labels(category_keys: set[str]) -> list[str]:
    """Convert category keys to display labels in canonical order."""
    return [label for key, label in CATEGORIES if key in category_keys]


def render_visibility_columns(
    allowed_categories: set[str],
    hidden_categories: set[str],
    *,
    allowed_note: str | None = None,
    empty_allowed_text: str = "No information is available in this view.",
    empty_hidden_text: str = "No information is hidden under current consent.",
) -> None:
    """Render standardized two-column visibility summary."""
    allowed_labels = _ordered_category_labels(allowed_categories)
    hidden_labels = _ordered_category_labels(hidden_categories)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### What you can see")
        if allowed_note:
            st.caption(allowed_note)
        if not allowed_labels:
            st.markdown(f"- {empty_allowed_text}")
        else:
            for label in allowed_labels:
                st.markdown(f"- {label}")
    with col_b:
        st.markdown("#### Information we can't show right now")
        if not hidden_labels:
            st.markdown(f"- {empty_hidden_text}")
        else:
            for label in hidden_labels:
                st.markdown(f"- {label}")


def _render_visible_hidden_summary(decision: PolicyDecision) -> None:
    """Resolve policy results into standardized category visibility columns."""
    all_category_keys = {key for key, _ in CATEGORIES}

    if decision.view_status == "full":
        allowed_categories = set(all_category_keys)
        hidden_categories: set[str] = set()
        render_visibility_columns(allowed_categories, hidden_categories)
        return

    if decision.view_status == "blocked":
        allowed_categories = set()
        hidden_categories = set(all_category_keys)
        render_visibility_columns(
            allowed_categories,
            hidden_categories,
            empty_allowed_text="No client details are available in this cross-agency view.",
        )
        return

    # For redacted/limited views, categorize fields into mutually exclusive groups.
    allowed_categories = _collect_category_labels(decision.allowed_fields) & all_category_keys
    redacted_categories = _collect_category_labels(decision.redacted_fields) & all_category_keys
    partial_categories = allowed_categories & redacted_categories
    visible_categories = allowed_categories - redacted_categories
    hidden_categories = redacted_categories - allowed_categories
    shown_categories = visible_categories | partial_categories

    render_visibility_columns(
        shown_categories,
        hidden_categories,
        allowed_note="Some details in these areas may be hidden due to consent or privacy rules.",
        empty_allowed_text="No category is available in this limited view.",
        empty_hidden_text="No additional information is hidden beyond the limited fields shown.",
    )


def _render_admin_technical_details(decision: PolicyDecision) -> None:
    """Render admin-only technical details in collapsed expander."""
    with st.expander("🔧 Show technical details", expanded=False):
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
    st.title("Safe Client View")
    st.caption("Governance-first view: policy is checked before cross-agency client details are shown.")

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

    render_policy_status(decision)
    _render_visible_hidden_summary(decision)

    latest_status = latest_referral_row.get("status") if latest_referral_row is not None else "No active referral"
    st.subheader("Client Summary")
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown(f"- **Client ID:** {_format_value(client_row.get('client_id'))}")
        st.markdown(f"- **Risk level:** {_risk_level(client_row)}")
    with c_right:
        st.markdown(f"- **Current status:** {_format_value(latest_status)}")
        st.markdown(f"- **Last assessment:** {_format_value(client_row.get('assessment_date'))}")

    if decision.view_status == "blocked":
        st.subheader("Client details")
        st.info("No client details are shown because policy blocked this cross-agency view.")
        _render_blocked_next_steps(decision)
        _render_admin_technical_details(decision)
        return

    st.subheader("Client details")
    safe_client = _safe_payload(client_row, decision)
    if safe_client:
        safe_client_rows = [
            {"Field": _friendly_label(field), "Value": _format_value(value)}
            for field, value in safe_client.items()
        ]
        st.dataframe(pd.DataFrame(safe_client_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No client fields are available in this view.")

    st.subheader("Consent summary")
    if current_consent_row is None:
        st.info("No current consent record found for this client.")
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
            st.info("Consent details are hidden by current policy rules.")

    if decision.view_status == "redacted":
        st.caption("Hidden fields are summarized above under 'Information we can't show right now'.")

    st.subheader("Referral summary")
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

    st.subheader("Encounter summary")
    if "client_id" not in encounters.columns:
        st.info("Encounter summary unavailable: encounters missing client_id.")
        _render_admin_technical_details(decision)
        return

    client_encounters = encounters[encounters["client_id"].astype(str) == str(selected_client_id)].copy()
    if client_encounters.empty:
        st.info("No encounters found for this client.")
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
