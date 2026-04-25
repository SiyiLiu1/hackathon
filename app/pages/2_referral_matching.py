"""Referral matching page with policy-first organization ranking."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.streamlit_app import (  # noqa: E402
    DEFAULT_TRACK1_RAW_DIR,
    load_shared_track1_data,
)
from src.policy_engine import PolicyDecision, evaluate_access, evaluate_policy  # noqa: E402
from src.referral_matching import (  # noqa: E402
    ACTION_BLOCKED,
    ACTION_MANUAL_REVIEW,
    ACTION_REFER_WITH_REDACTION,
    ACTION_SAFE_TO_REFER,
    assign_referral_action,
    rank_receiving_orgs,
)


def _get_tables() -> dict[str, pd.DataFrame]:
    """Return cached shared data tables for Streamlit pages."""
    if "shared_track1_tables" in st.session_state:
        return st.session_state["shared_track1_tables"]
    raw_dir = st.session_state.get("track1_raw_dir", DEFAULT_TRACK1_RAW_DIR)
    tables = load_shared_track1_data(raw_dir)
    st.session_state["shared_track1_tables"] = tables
    return tables


def _latest_referral_for_client(client_referrals: pd.DataFrame) -> pd.Series | None:
    """Pick latest referral row for selected client."""
    if client_referrals.empty:
        return None
    rows = client_referrals.copy()
    if "submitted_at" in rows.columns:
        rows["submitted_at_dt"] = pd.to_datetime(rows["submitted_at"], errors="coerce")
        rows = rows.sort_values("submitted_at_dt", ascending=False)
    return rows.iloc[0]


def _current_consent_for_client(client_row: pd.Series, consent_df: pd.DataFrame) -> pd.Series | None:
    """Resolve client's current consent row."""
    if "consent_id" not in consent_df.columns:
        return None
    current_consent_id = client_row.get("current_consent_id")
    if pd.isna(current_consent_id):
        return None
    matches = consent_df[consent_df["consent_id"].astype(str) == str(current_consent_id)]
    if matches.empty:
        return None
    return matches.iloc[0]


def _build_policy_record(
    client_row: pd.Series,
    current_consent_row: pd.Series | None,
    latest_referral_row: pd.Series | None,
) -> dict[str, Any]:
    """Build policy context mapping without exposing fields in UI."""
    record: dict[str, Any] = dict(client_row.to_dict())
    if current_consent_row is not None:
        for key, value in current_consent_row.to_dict().items():
            record[f"current_consent_{key}"] = value
    if latest_referral_row is not None:
        record.update(latest_referral_row.to_dict())
    else:
        # Fallback referral-like keys required by policy and matcher context.
        record.setdefault("referral_id", "N/A")
        record.setdefault("referring_org_id", client_row.get("primary_org_id"))
        record.setdefault("receiving_org_id", client_row.get("primary_org_id"))
        record.setdefault("consent_record_id", client_row.get("current_consent_id"))
        record.setdefault("status", "pending")
    return record


def _to_float(raw_value: str | None) -> float | None:
    """Parse an optional float value safely."""
    if raw_value is None:
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value):
        return None
    return value


def _parse_ranking_explanation(raw_explanation: str) -> dict[str, float]:
    """Parse raw key=value explanation text into scores."""
    parsed: dict[str, float] = {}
    if not raw_explanation:
        return parsed
    for chunk in raw_explanation.split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        score = _to_float(value.strip())
        if score is None:
            continue
        parsed[key.strip()] = score
    return parsed


def _human_match_reasons(raw_explanation: str) -> list[str]:
    """Convert technical score components to plain-language reasons."""
    parsed = _parse_ranking_explanation(raw_explanation)
    reasons: list[str] = []

    acceptance = parsed.get("acceptance_rate")
    if acceptance is not None and acceptance >= 0.75:
        reasons.append("good acceptance likelihood")

    speed = parsed.get("decision_speed_score")
    if speed is not None and speed >= 0.60:
        reasons.append("moderate response speed")

    taxonomy = parsed.get("taxonomy_fit")
    if taxonomy is not None:
        if taxonomy >= 0.5:
            reasons.append("strong service fit")
        elif taxonomy > 0:
            reasons.append("partial service fit")

    acuity = parsed.get("acuity_fit")
    if acuity is not None and acuity >= 0.5:
        reasons.append("fits client acuity level")

    return reasons


def _reasons_sentence(raw_explanation: str) -> str:
    """Build one human-readable sentence from ranked explanation."""
    reasons = _human_match_reasons(raw_explanation)
    if not reasons:
        return "Requires manual review due to limited matching context."
    if len(reasons) == 1:
        return reasons[0].capitalize() + "."
    if len(reasons) == 2:
        return f"{reasons[0].capitalize()} and {reasons[1]}."
    return f"{', '.join(reasons[:-1]).capitalize()}, and {reasons[-1]}."


def _technical_scoring_text(raw_explanation: str) -> str:
    """Format score components without exposing raw key names."""
    parsed = _parse_ranking_explanation(raw_explanation)
    if not parsed:
        return "Scoring details unavailable."

    parts: list[str] = []
    if "acceptance_rate" in parsed:
        parts.append(f"Acceptance likelihood {parsed['acceptance_rate']:.2f}")
    if "decision_speed_score" in parsed:
        parts.append(f"Response speed {parsed['decision_speed_score']:.2f}")
    if "taxonomy_fit" in parsed:
        parts.append(f"Service fit {parsed['taxonomy_fit']:.2f}")
    if "acuity_fit" in parsed:
        parts.append(f"Acuity fit {parsed['acuity_fit']:.2f}")
    if "stability_capacity" in parsed:
        parts.append(f"Stability and capacity {parsed['stability_capacity']:.2f}")
    if not parts:
        return "Scoring details unavailable."
    return "; ".join(parts)


def _match_level(score: float) -> str:
    """Map score into High/Medium/Low tier."""
    if score >= 0.75:
        return "High"
    if score >= 0.5:
        return "Medium"
    return "Low"


def _action_text(action_label: str) -> str:
    """Map technical actions to caseworker-friendly action labels."""
    action_map = {
        ACTION_SAFE_TO_REFER: "Start referral",
        ACTION_REFER_WITH_REDACTION: "Start referral with redaction",
        ACTION_BLOCKED: "Blocked until review",
        ACTION_MANUAL_REVIEW: "Manual review required",
    }
    return action_map.get(action_label, "Manual review required")


def _scope_tokens(value: Any) -> set[str]:
    """Parse comma/semicolon-delimited scope org list."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return set()
    text = str(value).replace(",", ";")
    return {token.strip() for token in text.split(";") if token.strip()}


def _blocked_matches(
    policy_record: dict[str, Any],
    orgs: pd.DataFrame,
    access_decision: str,
    issues: list[dict[str, str]],
) -> pd.DataFrame:
    """Build blocked organization rows with safe, plain-language reasons."""
    if "org_id" not in orgs.columns or "org_name" not in orgs.columns:
        return pd.DataFrame(columns=["Organization", "Policy status", "Reason"])
    blocked_reason = "Blocked by consent or governance policy."
    if issues:
        blocked_reason = "; ".join(
            sorted({str(issue.get("title", "Policy check")) for issue in issues if issue.get("title")})
        )

    out = orgs[["org_id", "org_name"]].copy().rename(columns={"org_id": "Organization ID", "org_name": "Organization"})
    out["Policy status"] = "Blocked"
    out["Reason"] = blocked_reason

    scope_type = _pretty_text(policy_record.get("current_consent_sharing_scope_type")).lower().replace(" ", "_")
    allowed_scope_orgs = _scope_tokens(policy_record.get("current_consent_sharing_scope_agency_ids"))
    if scope_type in {"limited_agencies", "named_agencies"} and allowed_scope_orgs:
        out["Policy status"] = out["Organization ID"].apply(
            lambda org_id: "Eligible" if str(org_id) in allowed_scope_orgs else "Blocked"
        )
        out["Reason"] = out["Organization ID"].apply(
            lambda org_id: "Within consent scope" if str(org_id) in allowed_scope_orgs else "Outside consent scope"
        )
    elif scope_type == "single_agency_only":
        primary_org_id = str(policy_record.get("primary_org_id", "")).strip()
        out["Policy status"] = out["Organization ID"].apply(
            lambda org_id: "Eligible" if str(org_id) == primary_org_id else "Blocked"
        )
        out["Reason"] = out["Organization ID"].apply(
            lambda org_id: "Current agency only" if str(org_id) == primary_org_id else "Single-agency consent restriction"
        )
    elif access_decision == "blocked":
        out["Policy status"] = "Blocked"

    return out[["Organization", "Policy status", "Reason"]]


def _referral_readiness_status(decision: PolicyDecision, final_action: str) -> tuple[str, str, list[str], str]:
    """Return readiness label, plain-language explanation, steps, and primary action label."""
    flags = decision.flags or {}
    if flags.get("ocap_restriction", False):
        return (
            "OCAP review required",
            "Referral sharing is restricted by OCAP rules until a permitted sharing path is confirmed.",
            [
                "Confirm OCAP governance conditions with the appropriate reviewer",
                "Verify whether the current consent scope supports cross-agency referral",
                "Document the review outcome before routing",
            ],
            "Start OCAP review",
        )
    if decision.view_status == "blocked":
        return (
            "Consent update needed",
            "Referral cannot be sent yet because consent status or sharing scope does not allow this referral.",
            [
                "Confirm client consent is current and active",
                "Update sharing scope if this referral requires cross-agency sharing",
                "Re-check referral readiness after consent updates",
            ],
            "Resolve consent issue",
        )
    if final_action == ACTION_MANUAL_REVIEW:
        return (
            "Manual review required",
            "No eligible receiving organizations could be routed automatically for this service need.",
            [
                "Review service need and referral context with a supervisor",
                "Identify potential receiving organizations through manual coordination",
                "Document why manual routing was selected",
            ],
            "Start manual review",
        )
    if final_action == ACTION_REFER_WITH_REDACTION or decision.view_status == "redacted":
        return (
            "Send with redaction",
            "Referral can proceed with redaction using only policy-approved information.",
            [
                "Review what must be protected before sending",
                "Confirm only allowed information is included",
                "Document that this referral was sent with redaction",
            ],
            "Start referral with redaction",
        )
    return (
        "Ready to send",
        "Referral can be sent under the current consent and sharing rules.",
        [
            "Confirm service fit before sending",
            "Send referral to an eligible receiving organization",
            "Document referral rationale in the case record",
        ],
        "Start referral",
    )


def _acuity_level(value: Any) -> str:
    """Map acuity score to low/moderate/high."""
    score = _to_float(str(value) if value is not None else None)
    if score is None:
        return "Not available"
    if score >= 9:
        return f"High ({score:.1f})"
    if score >= 4:
        return f"Moderate ({score:.1f})"
    return f"Low ({score:.1f})"


def _pretty_text(value: Any, default: str = "Not available") -> str:
    """Return a readable label for optional text fields."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    text = str(value).strip()
    if not text:
        return default
    return text.replace("_", " ").title()


def _infer_service_role(org_name: str, org_type: Any, taxonomy_code: Any) -> str:
    """Infer likely service role from org metadata."""
    haystack = " ".join([str(org_name or ""), str(org_type or ""), str(taxonomy_code or "")]).lower()
    if any(token in haystack for token in ("housing", "shelter", "transitional", "rent")):
        return "Housing / shelter support"
    if any(token in haystack for token in ("food", "meal", "nutrition", "basic needs", "pantry")):
        return "Food and basic needs"
    if any(token in haystack for token in ("legal", "justice", "advocacy", "law")):
        return "Legal aid"
    if any(token in haystack for token in ("mental", "recovery", "addiction", "substance", "therapy")):
        return "Mental health / recovery"
    return "Outreach / community support"


def _capacity_text(total_slots: Any, occupied_slots: Any, waitlist_flag: Any) -> str:
    """Build capacity/waitlist summary text."""
    waitlist = bool(waitlist_flag) if waitlist_flag is not None and not pd.isna(waitlist_flag) else False
    if waitlist:
        return "Waitlist in place"
    total = _to_float(str(total_slots) if total_slots is not None else None)
    occupied = _to_float(str(occupied_slots) if occupied_slots is not None else None)
    if total is None or occupied is None:
        return "Capacity details not available"
    available = max(0, int(round(total - occupied)))
    return f"{available} open of {int(round(total))} slots"


def _prepare_ranked_display(ranked: pd.DataFrame) -> pd.DataFrame:
    """Prepare compact ranked table for caseworker review."""
    display = ranked.copy()
    display.insert(0, "rank", range(1, len(display) + 1))
    display["score_pct"] = (display["score"] * 100.0).round(1)
    display["match_confidence"] = display.apply(
        lambda row: f"{row['score_pct']:.1f}% ({_match_level(float(row['score']))})", axis=1
    )
    display["likely_service_role"] = "Service support aligned to current referral needs"
    display["why_this_may_help"] = display["explanation"].apply(_reasons_sentence)
    display["recommended_action"] = display["action_label"].apply(_action_text)
    display = display.rename(
        columns={
            "rank": "Rank",
            "receiving_org_name": "Organization",
            "likely_service_role": "Likely service role",
            "match_confidence": "Match confidence",
            "why_this_may_help": "Why this may help",
            "recommended_action": "Recommended action",
        }
    )
    return display[
        ["Rank", "Organization", "Likely service role", "Match confidence", "Why this may help", "Recommended action"]
    ]


def _render_readiness_card(status: str, message: str) -> None:
    """Render step 1 readiness status card."""
    if status in {"Consent update needed", "OCAP review required"}:
        st.error(f"**{status}**")
    elif status in {"Send with redaction", "Manual review required"}:
        st.warning(f"**{status}**")
    else:
        st.success(f"**{status}**")
    st.text(message)


def _render_redaction_notice() -> None:
    """Render caseworker-friendly redaction notice for limited referrals."""
    st.subheader("What will be protected")
    st.text(
        "The referral package will exclude sensitive fields and only include information "
        "allowed by the current policy decision."
    )
    st.text("- Personal details")
    st.text("- Contact information")
    st.text("- Health and assessment details")
    st.text("- Governance-sensitive fields")
    st.caption("What can be included")
    st.text("- Service need and referral context")
    st.text("- Policy-approved case details")
    st.text("- Information allowed under current sharing rules")


def _render_referral_need(policy_record: dict[str, Any]) -> None:
    """Render step 2 referral need context from latest referral/client data."""
    service_need = (
        policy_record.get("referral_type")
        or policy_record.get("reason_code")
        or policy_record.get("service_need")
        or "Not specified"
    )
    consent_status = _pretty_text(policy_record.get("current_consent_status"))
    sharing_scope = _pretty_text(policy_record.get("current_consent_sharing_scope_type"))
    acuity = _acuity_level(policy_record.get("vi_spdat_score"))

    c1, c2 = st.columns(2)
    c1.caption("Service need")
    c1.write(_pretty_text(service_need))
    c1.caption("Risk / acuity level")
    c1.write(acuity)
    c2.caption("Consent status")
    c2.write(consent_status)
    c2.caption("Sharing scope")
    c2.write(sharing_scope)


def _render_grouped_orgs(
    ranked: pd.DataFrame,
    orgs: pd.DataFrame,
) -> None:
    """Render step 3 eligible receiving organizations grouped by service role."""
    org_meta = orgs.copy().rename(columns={"org_id": "receiving_org_id"})
    merge_cols = [
        "receiving_org_id",
        "org_type",
        "service_taxonomy_code",
        "capacity_total_slots",
        "capacity_occupied_slots",
        "waitlist_flag",
    ]
    available_cols = [col for col in merge_cols if col in org_meta.columns]
    with_meta = ranked.merge(org_meta[available_cols], on="receiving_org_id", how="left")
    with_meta["likely_service_role"] = with_meta.apply(
        lambda row: _infer_service_role(
            str(row.get("receiving_org_name", "")),
            row.get("org_type"),
            row.get("service_taxonomy_code"),
        ),
        axis=1,
    )
    with_meta["match_confidence"] = with_meta["score"].apply(
        lambda score: f"{float(score) * 100.0:.1f}% ({_match_level(float(score))})"
    )
    with_meta["why_this_may_help"] = with_meta["explanation"].apply(_reasons_sentence)
    with_meta["capacity_waitlist"] = with_meta.apply(
        lambda row: _capacity_text(
            row.get("capacity_total_slots"),
            row.get("capacity_occupied_slots"),
            row.get("waitlist_flag"),
        ),
        axis=1,
    )
    with_meta["recommended_action"] = with_meta["action_label"].apply(_action_text)

    st.subheader("Eligible receiving organizations for this service need")
    st.caption("These organizations may support different needs. Review fit before sending a referral.")

    suggested = with_meta.iloc[0]
    st.subheader("Suggested next referral")
    st.text(
        "This is the strongest match based on current data, but other organizations may be "
        "appropriate for different client needs."
    )
    lead_a, lead_b, lead_c = st.columns(3)
    lead_a.caption("Organization")
    lead_a.write(str(suggested["receiving_org_name"]))
    lead_b.caption("Service role")
    lead_b.write(str(suggested["likely_service_role"]))
    lead_c.caption("Match confidence")
    lead_c.write(str(suggested["match_confidence"]))
    st.caption("Why it may be appropriate")
    st.write(str(suggested["why_this_may_help"]))
    st.caption("Capacity / waitlist")
    st.write(str(suggested["capacity_waitlist"]))
    st.caption("Recommended action")
    st.write(str(suggested["recommended_action"]))
    st.divider()

    for role, group in with_meta.groupby("likely_service_role", sort=True):
        st.subheader(str(role))
        display = group.copy()
        display.insert(0, "Rank", range(1, len(display) + 1))
        display = display.rename(
            columns={
                "receiving_org_name": "Organization",
                "likely_service_role": "Service role",
                "match_confidence": "Match confidence",
                "why_this_may_help": "Why it may be appropriate",
                "capacity_waitlist": "Capacity / waitlist",
                "recommended_action": "Recommended action",
            }
        )
        st.dataframe(
            display[
                [
                    "Rank",
                    "Organization",
                    "Service role",
                    "Match confidence",
                    "Why it may be appropriate",
                    "Capacity / waitlist",
                    "Recommended action",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )


def _render_technical_scoring_details(ranked: pd.DataFrame | None) -> None:
    """Show raw ranking explanations in a collapsed technical section."""
    with st.expander("Technical scoring details", expanded=False):
        st.caption("Raw ranking explanation")
        if ranked is None or ranked.empty:
            st.text("No ranked organizations available.")
        else:
            details = ranked[["receiving_org_name", "explanation"]].copy()
            details["explanation"] = details["explanation"].astype(str).apply(_technical_scoring_text)
            details = details.rename(
                columns={
                    "receiving_org_name": "Organization",
                    "explanation": "Scoring details",
                }
            )
            st.dataframe(details, use_container_width=True, hide_index=True)


def main() -> None:
    """Render referral matching page."""
    st.set_page_config(page_title="Referral Matching", page_icon=":handshake:", layout="wide")

    if "track1_raw_dir" not in st.session_state:
        st.session_state["track1_raw_dir"] = DEFAULT_TRACK1_RAW_DIR

    with st.sidebar:
        st.subheader("CareMatch Safe")
        st.caption("Policy-first referral coordination")
        st.subheader("Workspace")
        st.text_input("Track 1 raw data directory", key="track1_raw_dir")
        st.caption("Local demo dataset")

    st.title("Referral Readiness & Routing")
    st.caption(
        "Check whether a referral can be sent, then identify appropriate receiving organizations by service need."
    )

    try:
        tables = _get_tables()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        return

    clients = tables["clients"]
    consent = tables["consent"]
    referrals_enriched = tables["referrals_enriched"]
    orgs = tables["orgs"]
    encounters = tables["encounters"]

    if "client_id" not in clients.columns:
        st.error("Missing required clients column: client_id")
        return

    client_ids = sorted(clients["client_id"].astype(str).dropna().unique().tolist())
    if not client_ids:
        st.info("No clients available for referral matching.")
        return

    selected_client_id = st.selectbox("Select client", options=client_ids)

    # Policy context is assembled before exposing any cross-agency referral data.
    client_row = clients[clients["client_id"].astype(str) == str(selected_client_id)].iloc[0]
    client_referrals = referrals_enriched[
        referrals_enriched["client_id"].astype(str) == str(selected_client_id)
    ].copy()
    latest_referral = _latest_referral_for_client(client_referrals)
    current_consent = _current_consent_for_client(client_row, consent)
    policy_record = _build_policy_record(client_row, current_consent, latest_referral)

    decision = evaluate_policy(
        policy_record,
        is_multi_agency_view=True,
        action="view",
        all_fields=policy_record.keys(),
    )
    access_eval = evaluate_access(
        client_id=str(selected_client_id),
        intended_action="referral_matching",
        data=tables,
    )
    access_decision = str(access_eval.get("decision", "blocked"))
    issues = access_eval.get("issues", [])

    ranked = rank_receiving_orgs(
        client_row=policy_record,
        policy_decision=decision,
        orgs_df=orgs,
        referrals_enriched_df=referrals_enriched,
        encounters_df=encounters,
    )
    final_action = assign_referral_action(decision, ranked)
    status, status_message, readiness_steps, primary_action_label = _referral_readiness_status(decision, final_action)

    st.subheader("Step 1: Referral readiness")
    _render_readiness_card(status, status_message)
    st.divider()

    st.subheader("Step 2: Referral need")
    _render_referral_need(policy_record)
    st.divider()

    blocked_statuses = {"Consent update needed", "OCAP review required"}
    if status in blocked_statuses or access_decision == "blocked":
        st.error("Referral cannot be sent yet")
        st.caption("Next steps")
        for step in readiness_steps:
            st.text(f"- {step}")
        blocked = _blocked_matches(policy_record=policy_record, orgs=orgs, access_decision=access_decision, issues=issues)
        if not blocked.empty:
            st.subheader("Blocked receiving organizations")
            st.caption("Organizations are shown without restricted client details.")
            st.dataframe(blocked, use_container_width=True, hide_index=True)
        st.divider()
        st.subheader("Step 4: Caseworker checklist")
        st.checkbox("Confirm client consent is current")
        st.checkbox("Confirm sharing scope allows this referral")
        st.checkbox("Review whether redaction is required")
        st.checkbox("Confirm receiving organization fits the service need")
        st.checkbox("Document referral rationale")
        st.divider()
        st.subheader("Primary action")
        st.button(primary_action_label, use_container_width=True, key="btn_primary_action_blocked")
        _render_technical_scoring_details(ranked=ranked)
        return

    st.subheader("Step 3: Matching receiving organizations")
    if status == "Send with redaction":
        st.warning("Referral can proceed with redaction")
        _render_redaction_notice()
        st.caption("Eligible organizations for limited referral")

    blocked = _blocked_matches(policy_record=policy_record, orgs=orgs, access_decision=access_decision, issues=issues)
    if ranked.empty:
        st.warning("No eligible receiving organizations are currently available for this service need.")
    else:
        _render_grouped_orgs(ranked, orgs)
        if not blocked.empty and (blocked["Policy status"] == "Blocked").any():
            with st.expander("Show blocked organizations and reasons", expanded=False):
                st.dataframe(blocked[blocked["Policy status"] == "Blocked"], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Step 4: Caseworker checklist")
    st.checkbox("Confirm client consent is current")
    st.checkbox("Confirm sharing scope allows this referral")
    st.checkbox("Review whether redaction is required")
    st.checkbox("Confirm receiving organization fits the service need")
    st.checkbox("Document referral rationale")

    st.divider()
    st.subheader("Primary action")
    st.button(primary_action_label, use_container_width=True, key="btn_primary_action")

    _render_technical_scoring_details(ranked=ranked)


if __name__ == "__main__":
    main()
