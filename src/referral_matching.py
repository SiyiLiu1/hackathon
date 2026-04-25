"""Explainable referral matching and ranking for receiving organizations.

This module ranks receiving organizations only after policy evaluation permits
cross-agency recommendation.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import pandas as pd


ACTION_SAFE_TO_REFER = "safe_to_refer"
ACTION_REFER_WITH_REDACTION = "refer_with_redaction"
ACTION_MANUAL_REVIEW = "manual_review_required"
ACTION_BLOCKED = "blocked_by_policy"

VALID_ACTIONS = {
    ACTION_SAFE_TO_REFER,
    ACTION_REFER_WITH_REDACTION,
    ACTION_MANUAL_REVIEW,
    ACTION_BLOCKED,
}


def _require_columns(df: pd.DataFrame, table_name: str, required: Iterable[str]) -> None:
    """Fail fast when required columns are missing."""
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(
            f"Table '{table_name}' is missing required column(s): {', '.join(missing)}"
        )


def _to_naive_timestamp(value: Any) -> pd.Timestamp | None:
    """Convert scalar timestamp to tz-naive."""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    return ts


def _to_datetime(series: pd.Series) -> pd.Series:
    """Convert a series to tz-naive pandas datetime with coercion."""
    dt = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    return dt


def _as_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float with safe fallback."""
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_text(value: Any) -> str:
    """Normalize arbitrary value to stripped lowercase text."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _policy_field(policy_decision: Any, field: str, default: Any = None) -> Any:
    """Read a policy decision field from dataclass-like or dict-like object."""
    if hasattr(policy_decision, field):
        return getattr(policy_decision, field)
    if isinstance(policy_decision, Mapping):
        return policy_decision.get(field, default)
    return default


def _taxonomy_fit_score(referral_type: str, org_taxonomy: str, org_type: str) -> float:
    """Score service taxonomy fit in [0, 1] with explicit rules."""
    rt = _as_text(referral_type)
    tax = _as_text(org_taxonomy)
    ot = _as_text(org_type)

    if not rt:
        return 0.5
    if rt == tax:
        return 1.0
    if rt in tax or tax in rt:
        return 0.8
    if rt == ot:
        return 0.7
    if rt in ot or ot in rt:
        return 0.6
    return 0.2


def _acuity_fit_score(client_vi_spdat: float, org_median_vi_spdat: float | None) -> float:
    """Score acuity fit in [0, 1] based on distance from org historical median."""
    client_score = _as_float(client_vi_spdat, default=0.0)
    if org_median_vi_spdat is None or pd.isna(org_median_vi_spdat):
        return 0.5
    distance = abs(client_score - _as_float(org_median_vi_spdat, default=client_score))
    # Explicit formula: linear decay over 12 points.
    return max(0.0, 1.0 - (distance / 12.0))


def _capacity_proxy_score(total_slots: Any, occupied_slots: Any, waitlist_flag: Any) -> float:
    """Simple capacity/stability proxy in [0, 1]."""
    total = _as_float(total_slots, default=0.0)
    occupied = _as_float(occupied_slots, default=0.0)
    waitlist = bool(waitlist_flag) if waitlist_flag is not None and not pd.isna(waitlist_flag) else False

    if total <= 0:
        base = 0.5
    else:
        availability = max(0.0, min(1.0, (total - occupied) / total))
        base = 0.2 + 0.8 * availability
    if waitlist:
        base -= 0.2
    return max(0.0, min(1.0, base))


def compute_org_historical_metrics(
    referrals_enriched_df: pd.DataFrame,
    encounters_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute explainable historical org metrics used for ranking.

    Output columns:
      - receiving_org_id
      - historical_acceptance_rate
      - median_decision_speed_hours
      - org_median_vi_spdat
      - recent_positive_outcome_rate
    """
    _require_columns(
        referrals_enriched_df,
        "referrals_enriched",
        ["receiving_org_id", "status", "submitted_at"],
    )
    _require_columns(encounters_df, "encounters", ["org_id", "outcome_flag"])

    ref = referrals_enriched_df.copy()
    ref["status_norm"] = ref["status"].map(_as_text)
    ref["submitted_at_dt"] = _to_datetime(ref["submitted_at"])
    if "decision_at" in ref.columns:
        ref["decision_at_dt"] = _to_datetime(ref["decision_at"])
    else:
        ref["decision_at_dt"] = pd.NaT

    decided = ref[ref["status_norm"].isin(["accepted", "declined"])].copy()
    acceptance = (
        decided.assign(is_accepted=(decided["status_norm"] == "accepted").astype(float))
        .groupby("receiving_org_id", as_index=False)["is_accepted"]
        .mean()
        .rename(columns={"is_accepted": "historical_acceptance_rate"})
    )

    decided["decision_speed_hours"] = (
        (decided["decision_at_dt"] - decided["submitted_at_dt"]).dt.total_seconds() / 3600.0
    )
    speed = (
        decided.dropna(subset=["decision_speed_hours"])
        .groupby("receiving_org_id", as_index=False)["decision_speed_hours"]
        .median()
        .rename(columns={"decision_speed_hours": "median_decision_speed_hours"})
    )

    if "vi_spdat_score" in ref.columns:
        vi = (
            ref.dropna(subset=["vi_spdat_score"])
            .groupby("receiving_org_id", as_index=False)["vi_spdat_score"]
            .median()
            .rename(columns={"vi_spdat_score": "org_median_vi_spdat"})
        )
    else:
        vi = pd.DataFrame(columns=["receiving_org_id", "org_median_vi_spdat"])

    enc = encounters_df.copy()
    if "occurred_at" in enc.columns:
        enc["occurred_at_dt"] = _to_datetime(enc["occurred_at"])
    else:
        enc["occurred_at_dt"] = pd.NaT
    if enc["occurred_at_dt"].notna().any():
        now_ts = _to_naive_timestamp(pd.Timestamp.utcnow())
        cutoff = (now_ts or pd.Timestamp.now()) - pd.Timedelta(days=90)
        recent = enc[(enc["occurred_at_dt"].isna()) | (enc["occurred_at_dt"] >= cutoff)].copy()
    else:
        recent = enc.copy()

    recent["is_positive"] = recent["outcome_flag"].map(_as_text).eq("positive").astype(float)
    outcome = (
        recent.groupby("org_id", as_index=False)["is_positive"]
        .mean()
        .rename(columns={"org_id": "receiving_org_id", "is_positive": "recent_positive_outcome_rate"})
    )

    metrics = acceptance.merge(speed, on="receiving_org_id", how="outer")
    metrics = metrics.merge(vi, on="receiving_org_id", how="outer")
    metrics = metrics.merge(outcome, on="receiving_org_id", how="outer")

    metrics["historical_acceptance_rate"] = metrics["historical_acceptance_rate"].fillna(0.5)
    metrics["median_decision_speed_hours"] = metrics["median_decision_speed_hours"].fillna(72.0)
    metrics["org_median_vi_spdat"] = metrics["org_median_vi_spdat"].fillna(pd.NA)
    metrics["recent_positive_outcome_rate"] = metrics["recent_positive_outcome_rate"].fillna(0.5)

    return metrics


def build_org_feature_frame(
    orgs_df: pd.DataFrame,
    referrals_enriched_df: pd.DataFrame,
    encounters_df: pd.DataFrame,
    clients_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build full organization feature frame for referral ranking."""
    _require_columns(orgs_df, "orgs", ["org_id", "org_name"])
    _require_columns(
        referrals_enriched_df,
        "referrals_enriched",
        ["receiving_org_id", "status", "submitted_at"],
    )
    _require_columns(encounters_df, "encounters", ["org_id", "outcome_flag"])

    metrics = compute_org_historical_metrics(referrals_enriched_df, encounters_df)

    org_features = orgs_df.copy().rename(
        columns={
            "org_id": "receiving_org_id",
            "org_name": "receiving_org_name",
            "service_taxonomy_code": "org_service_taxonomy_code",
            "org_type": "org_type",
        }
    )
    org_features = org_features.merge(metrics, on="receiving_org_id", how="left")

    # Fill defaults explicitly for deterministic scoring behavior.
    org_features["historical_acceptance_rate"] = org_features["historical_acceptance_rate"].fillna(0.5)
    org_features["median_decision_speed_hours"] = org_features["median_decision_speed_hours"].fillna(72.0)
    org_features["recent_positive_outcome_rate"] = org_features["recent_positive_outcome_rate"].fillna(0.5)
    if "org_median_vi_spdat" not in org_features.columns:
        org_features["org_median_vi_spdat"] = pd.NA

    if "capacity_total_slots" not in org_features.columns:
        org_features["capacity_total_slots"] = pd.NA
    if "capacity_occupied_slots" not in org_features.columns:
        org_features["capacity_occupied_slots"] = pd.NA
    if "waitlist_flag" not in org_features.columns:
        org_features["waitlist_flag"] = False
    if "org_service_taxonomy_code" not in org_features.columns:
        org_features["org_service_taxonomy_code"] = ""
    if "org_type" not in org_features.columns:
        org_features["org_type"] = ""

    return org_features


def rank_receiving_orgs(
    client_row: Mapping[str, Any],
    policy_decision: Any,
    orgs_df: pd.DataFrame,
    referrals_enriched_df: pd.DataFrame,
    encounters_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rank eligible receiving organizations after policy evaluation.

    The rank score is an explicit weighted formula:
      score =
        0.30 * taxonomy_fit +
        0.25 * historical_acceptance_rate +
        0.20 * decision_speed_score +
        0.15 * acuity_fit +
        0.10 * stability_capacity_score
    """
    allowed = bool(_policy_field(policy_decision, "allowed", False))
    flags = _policy_field(policy_decision, "flags", {}) or {}
    view_status = str(_policy_field(policy_decision, "view_status", "blocked"))

    cross_agency_blocked = (
        (not allowed)
        or bool(flags.get("ocap_restriction", False))
        or bool(flags.get("consent_gap", False))
        or bool(flags.get("scope_mismatch", False))
    )
    if cross_agency_blocked:
        return pd.DataFrame(
            columns=[
                "receiving_org_id",
                "receiving_org_name",
                "score",
                "explanation",
                "action_label",
            ]
        )

    org_features = build_org_feature_frame(orgs_df, referrals_enriched_df, encounters_df)

    referring_org_id = client_row.get("referring_org_id")
    if referring_org_id is not None:
        org_features = org_features[org_features["receiving_org_id"] != referring_org_id].copy()

    # Optional consent scope filtering for named agencies.
    scope_type = _as_text(client_row.get("current_consent_sharing_scope_type", client_row.get("sharing_scope_type")))
    allowed_orgs_raw = client_row.get(
        "current_consent_sharing_scope_agency_ids", client_row.get("sharing_scope_agency_ids")
    )
    if scope_type == "named_agencies" and allowed_orgs_raw is not None and not pd.isna(allowed_orgs_raw):
        allowed_orgs = {token.strip() for token in str(allowed_orgs_raw).split(";") if token.strip()}
        org_features = org_features[org_features["receiving_org_id"].isin(allowed_orgs)].copy()

    referral_type = str(client_row.get("referral_type", client_row.get("reason_code", "")))
    client_vi_spdat = _as_float(client_row.get("vi_spdat_score"), default=0.0)

    # Explicit component formulas.
    org_features["taxonomy_fit"] = org_features.apply(
        lambda r: _taxonomy_fit_score(referral_type, r["org_service_taxonomy_code"], r["org_type"]), axis=1
    )
    org_features["decision_speed_score"] = (
        1.0 - (org_features["median_decision_speed_hours"].clip(lower=0.0, upper=168.0) / 168.0)
    ).clip(lower=0.0, upper=1.0)
    org_features["acuity_fit"] = org_features["org_median_vi_spdat"].apply(
        lambda org_med: _acuity_fit_score(client_vi_spdat, org_med)
    )
    org_features["capacity_proxy"] = org_features.apply(
        lambda r: _capacity_proxy_score(
            r.get("capacity_total_slots"),
            r.get("capacity_occupied_slots"),
            r.get("waitlist_flag"),
        ),
        axis=1,
    )
    org_features["stability_capacity_score"] = (
        0.7 * org_features["recent_positive_outcome_rate"] + 0.3 * org_features["capacity_proxy"]
    ).clip(lower=0.0, upper=1.0)

    org_features["score"] = (
        0.30 * org_features["taxonomy_fit"]
        + 0.25 * org_features["historical_acceptance_rate"]
        + 0.20 * org_features["decision_speed_score"]
        + 0.15 * org_features["acuity_fit"]
        + 0.10 * org_features["stability_capacity_score"]
    ).round(4)

    org_features["explanation"] = org_features.apply(
        lambda r: (
            "taxonomy_fit="
            + f"{r['taxonomy_fit']:.2f}"
            + "; acceptance_rate="
            + f"{r['historical_acceptance_rate']:.2f}"
            + "; decision_speed_score="
            + f"{r['decision_speed_score']:.2f}"
            + "; acuity_fit="
            + f"{r['acuity_fit']:.2f}"
            + "; stability_capacity="
            + f"{r['stability_capacity_score']:.2f}"
        ),
        axis=1,
    )

    ranked = org_features.sort_values(
        by=["score", "historical_acceptance_rate", "receiving_org_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    action_label = ACTION_REFER_WITH_REDACTION if view_status == "redacted" else ACTION_SAFE_TO_REFER
    ranked["action_label"] = action_label

    return ranked[
        [
            "receiving_org_id",
            "receiving_org_name",
            "score",
            "explanation",
            "action_label",
        ]
    ]


def assign_referral_action(policy_decision: Any, ranked_df: pd.DataFrame) -> str:
    """Assign final referral action from policy and ranking outputs."""
    allowed = bool(_policy_field(policy_decision, "allowed", False))
    view_status = str(_policy_field(policy_decision, "view_status", "blocked"))
    flags = _policy_field(policy_decision, "flags", {}) or {}

    if (
        (not allowed)
        or bool(flags.get("ocap_restriction", False))
        or bool(flags.get("consent_gap", False))
        or bool(flags.get("scope_mismatch", False))
    ):
        return ACTION_BLOCKED

    if ranked_df is None or ranked_df.empty:
        return ACTION_MANUAL_REVIEW

    if view_status == "redacted" or bool(flags.get("governance_gap", False)):
        return ACTION_REFER_WITH_REDACTION

    return ACTION_SAFE_TO_REFER
