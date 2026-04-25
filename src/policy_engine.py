"""Governance policy engine for Track 1 referral data.

This module enforces project privacy/governance redlines before data is used in
UI/API/export/analytics workflows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)


class PolicySchemaError(ValueError):
    """Base exception for policy input schema validation failures."""


class MissingPolicyTableError(PolicySchemaError):
    """Raised when a required table is missing for policy evaluation."""


class MissingPolicyColumnError(PolicySchemaError):
    """Raised when required policy columns are missing."""


@dataclass(frozen=True)
class PolicyDecision:
    """Decision contract for policy evaluation."""

    allowed: bool
    view_status: str
    reasons: list[str]
    flags: dict[str, bool]
    allowed_fields: list[str]
    redacted_fields: list[str]


DUPLICATE_DECISIONS = {"auto_merge_eligible", "review_required", "blocked_by_policy"}

SENSITIVE_FIELDS = {
    "first_name",
    "last_name",
    "middle_name",
    "aliases",
    "dob",
    "address_line1",
    "primary_language",
    "mental_health_flag",
    "substance_use_flag",
    "physical_health_flag",
    "developmental_flag",
    "ocap_governing_nation",
    "ocap_data_use_conditions",
}

MULTI_AGENCY_SCOPES = {"cluster", "ca_table", "all", "multi_agency"}
SINGLE_AGENCY_SCOPES = {"single_agency_only", "org", "single_org"}
GOVERNMENT_LEGAL_BASES = {"foippa", "government_sharing", "government-sharing", "public_body"}
NO_SHARING_SCOPES = {"no_sharing", "none", "blocked"}

ACTIVE_CONSENT_STATUSES = {"active", "valid"}
PENDING_CONSENT_STATUSES = {"pending"}
BLOCKED_CONSENT_STATUSES = {"expired", "withdrawn", "superseded"}

CROSS_AGENCY_ACTIONS = {
    "view",
    "api",
    "export",
    "analytics",
    "create_referral",
    "coordinate_with_receiving_org",
    "referral_matching",
}

POLICY_REQUIRED_COLUMNS: dict[str, set[str]] = {
    "referrals_enriched": {
        "referral_id",
        "client_id",
        "referring_org_id",
        "receiving_org_id",
        "ocap_protected",
    }
}

POLICY_OPTIONAL_COLUMNS = {
    "current_consent_status",
    "current_consent_withdrawal_date",
    "current_consent_expires_at",
    "current_consent_effective_at",
    "current_consent_sharing_scope_type",
    "current_consent_sharing_scope_agency_ids",
    "current_consent_legal_basis",
    "current_consent_purpose_codes",
    "current_consent_consent_id",
}
SAFE_BASE_FIELDS = {
    "client_id",
    "primary_org_id",
    "current_consent_id",
    "status",
    "referral_id",
    "referring_org_id",
    "receiving_org_id",
    "assessment_acuity_level",
    "vi_spdat_score",
}


def get_required_columns() -> dict[str, set[str]]:
    """Return required columns for policy evaluation inputs."""
    return {table: set(columns) for table, columns in POLICY_REQUIRED_COLUMNS.items()}


def validate_required_columns(
    tables: Mapping[str, pd.DataFrame], required_columns: Mapping[str, set[str]] | None = None
) -> None:
    """Validate policy-engine table presence and required columns.

    Args:
        tables: Mapping of input table names to DataFrames.
        required_columns: Optional override for required columns.

    Raises:
        MissingPolicyTableError: If required table is absent.
        MissingPolicyColumnError: If required columns are missing.
    """
    required = required_columns or get_required_columns()
    missing_tables = sorted(table for table in required if table not in tables)
    if missing_tables:
        message = "Policy schema validation failed. Missing table(s): " + ", ".join(missing_tables)
        LOGGER.error(message)
        raise MissingPolicyTableError(message)

    errors: list[str] = []
    for table, cols in required.items():
        missing_cols = sorted(cols - set(tables[table].columns))
        if missing_cols:
            errors.append(f"Table '{table}' missing required column(s): {', '.join(missing_cols)}")
    if errors:
        message = "Policy schema validation failed.\n" + "\n".join(errors)
        LOGGER.error(message)
        raise MissingPolicyColumnError(message)


def _to_naive_timestamp(value: Any) -> pd.Timestamp | None:
    """Parse timestamp and normalize to tz-naive for safe comparisons."""
    if value is None or pd.isna(value):
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert(None)
    return ts


def _is_empty_text(value: Any) -> bool:
    """Return True when a text-like value is null/blank."""
    return value is None or pd.isna(value) or str(value).strip() == ""


def _get_first_present(row: Mapping[str, Any], candidates: Sequence[str]) -> Any:
    """Get first non-missing key from row mapping."""
    for name in candidates:
        if name in row:
            return row[name]
    return None


def _normalize_scope(scope_value: Any) -> str:
    """Normalize consent scope string for policy checks."""
    if _is_empty_text(scope_value):
        return ""
    return str(scope_value).strip().lower()


def _normalize_action(action: str) -> str:
    """Normalize intended action keyword."""
    if _is_empty_text(action):
        return "view"
    return str(action).strip().lower()


def _is_scope_allowed_for_org(
    scope: str, context_org_id: str | None, allowed_org_ids_raw: Any
) -> tuple[bool, bool]:
    """Check whether scope permits use for the request org.

    Returns:
        Tuple of (scope_allowed, is_scope_mismatch).
    """
    if scope in {"all_dsa_agencies", "all_partner_agencies"}:
        return True, False
    if scope in {"limited_agencies", "named_agencies"}:
        if _is_empty_text(allowed_org_ids_raw) or _is_empty_text(context_org_id):
            return True, False
        allowed = {
            token.strip()
            for token in str(allowed_org_ids_raw).replace(",", ";").split(";")
            if token.strip()
        }
        return context_org_id in allowed, context_org_id not in allowed
    if scope in MULTI_AGENCY_SCOPES:
        return True, False
    if scope in SINGLE_AGENCY_SCOPES or scope in NO_SHARING_SCOPES:
        return False, True
    return False, True


def _build_issue(code: str, severity: str, title: str, explanation: str, action: str) -> dict[str, str]:
    """Build one policy issue card."""
    return {
        "code": code,
        "severity": severity,
        "title": title,
        "explanation": explanation,
        "recommended_action": action,
    }


def _decision_from_issues(issues: list[dict[str, str]]) -> str:
    """Map issue severities into ready/limited/blocked."""
    severities = {issue["severity"] for issue in issues}
    if "error" in severities:
        return "blocked"
    if "warning" in severities:
        return "limited"
    return "ready"


def _build_policy_record_from_tables(client_id: str, data: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Assemble a policy input record for one client from loaded tables."""
    clients = data.get("clients", pd.DataFrame())
    consent = data.get("consent", pd.DataFrame())
    referrals = data.get("referrals_enriched", pd.DataFrame())
    if "client_id" not in clients.columns:
        return {}
    client_rows = clients[clients["client_id"].astype(str) == str(client_id)]
    if client_rows.empty:
        return {}

    client_row = client_rows.iloc[0]
    record = dict(client_row.to_dict())
    current_consent_id = client_row.get("current_consent_id")
    if current_consent_id is not None and "consent_id" in consent.columns:
        consent_rows = consent[consent["consent_id"].astype(str) == str(current_consent_id)]
        if not consent_rows.empty:
            for key, value in consent_rows.iloc[0].to_dict().items():
                record[f"current_consent_{key}"] = value

    if "client_id" in referrals.columns:
        referral_rows = referrals[referrals["client_id"].astype(str) == str(client_id)].copy()
        if not referral_rows.empty:
            if "submitted_at" in referral_rows.columns:
                referral_rows["submitted_at_dt"] = pd.to_datetime(referral_rows["submitted_at"], errors="coerce")
                referral_rows = referral_rows.sort_values("submitted_at_dt", ascending=False)
            record.update(referral_rows.iloc[0].to_dict())

    record.setdefault("referral_id", "N/A")
    record.setdefault("referring_org_id", client_row.get("primary_org_id"))
    record.setdefault("receiving_org_id", client_row.get("primary_org_id"))
    record.setdefault("consent_record_id", current_consent_id)
    return record


def evaluate_access(
    client_id: str,
    intended_action: str,
    data: Mapping[str, pd.DataFrame],
    *,
    context_org_id: str | None = None,
    reference_time: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Evaluate client access decision contract for Streamlit pages.

    Returns a dict with:
      decision, issues, allowed_fields, hidden_fields, recommended_actions,
      legal_basis, flags.
    """
    record = _build_policy_record_from_tables(client_id, data)
    if not record:
        return {
            "decision": "blocked",
            "issues": [
                _build_issue(
                    "client_not_found",
                    "error",
                    "Client not found",
                    "The selected client is not available in the loaded dataset.",
                    "Confirm client ID and refresh the dataset source.",
                )
            ],
            "allowed_fields": ["client_id"],
            "hidden_fields": [],
            "recommended_actions": ["Confirm client ID and refresh the dataset source."],
            "legal_basis": "Not recorded",
            "flags": {"consent_gap": True, "ocap_restriction": False, "scope_mismatch": False},
        }

    decision = evaluate_policy(
        record,
        reference_time=reference_time,
        context_org_id=context_org_id,
        is_multi_agency_view=_normalize_action(intended_action) in CROSS_AGENCY_ACTIONS,
        action=_normalize_action(intended_action),
        all_fields=record.keys(),
    )

    issues: list[dict[str, str]] = []
    flags = decision.flags
    if flags.get("missing_consent", False):
        issues.append(
            _build_issue(
                "consent_missing",
                "error",
                "Consent record missing",
                "Cross-agency sharing is blocked because no current consent record was found.",
                "Capture or confirm consent before proceeding.",
            )
        )
    if flags.get("withdrawn_consent", False):
        issues.append(
            _build_issue(
                "consent_withdrawn",
                "error",
                "Consent withdrawn",
                "Cross-agency sharing is blocked because consent has been withdrawn.",
                "Stop sharing and request updated consent from the client.",
            )
        )
    if flags.get("expired_consent", False):
        issues.append(
            _build_issue(
                "consent_expired",
                "error",
                "Consent expired",
                "Cross-agency sharing is blocked because consent has expired.",
                "Request updated consent from the client before proceeding.",
            )
        )
    if flags.get("scope_mismatch", False):
        severity = "warning" if decision.view_status != "blocked" else "error"
        issues.append(
            _build_issue(
                "scope_restriction",
                severity,
                "Consent scope restriction",
                "Current consent does not permit this cross-agency sharing scope.",
                "Use a single-agency workflow or request broader consent.",
            )
        )
    if flags.get("ocap_restriction", False):
        severity = "error" if decision.view_status == "blocked" else "warning"
        issues.append(
            _build_issue(
                "ocap_restriction",
                severity,
                "OCAP restriction in effect",
                "Indigenous data-governance protections require additional review.",
                "Escalate to OCAP/privacy reviewer and confirm allowed scope.",
            )
        )
    if flags.get("governance_gap", False):
        issues.append(
            _build_issue(
                "legal_basis_gap",
                "warning",
                "Legal basis review needed",
                "The legal basis or purpose-of-use details are incomplete.",
                "Confirm legal basis and purpose before sharing additional details.",
            )
        )

    access_decision = _decision_from_issues(issues)
    if not issues and decision.view_status == "redacted":
        access_decision = "limited"

    all_fields = list(record.keys())
    allowed_fields = list(dict.fromkeys(decision.allowed_fields))
    hidden_fields = [field for field in all_fields if field not in set(allowed_fields)]
    recommended_actions = list(
        dict.fromkeys(issue["recommended_action"] for issue in issues if issue.get("recommended_action"))
    )
    legal_basis_raw = _get_first_present(record, ("current_consent_legal_basis", "legal_basis"))
    legal_basis = "Not recorded" if _is_empty_text(legal_basis_raw) else str(legal_basis_raw)
    return {
        "decision": access_decision,
        "issues": issues,
        "allowed_fields": allowed_fields,
        "hidden_fields": hidden_fields,
        "recommended_actions": recommended_actions,
        "legal_basis": legal_basis,
        "flags": flags,
    }


def evaluate_policy(
    record: Mapping[str, Any],
    *,
    reference_time: pd.Timestamp | None = None,
    context_org_id: str | None = None,
    is_multi_agency_view: bool = True,
    action: str = "view",
    all_fields: Iterable[str] | None = None,
) -> PolicyDecision:
    """Evaluate a single enriched referral row against governance redlines.

    Args:
        record: One row from `referrals_enriched` (dict-like).
        reference_time: Evaluation timestamp. Defaults to current UTC time.
        context_org_id: Requesting org id for scope checks.
        is_multi_agency_view: Whether data will be shown/used cross-org.
        action: Intended action (`view`, `api`, `export`, `analytics`, `auto_merge`).
        all_fields: Optional explicit list of record fields to evaluate for redaction.

    Returns:
        A PolicyDecision following the required project contract.
    """
    ref_time = _to_naive_timestamp(reference_time) or _to_naive_timestamp(pd.Timestamp.utcnow())
    action_norm = _normalize_action(action)
    is_cross_agency_action = is_multi_agency_view or action_norm in CROSS_AGENCY_ACTIONS
    reason_list: list[str] = []
    scope_hard_block = False
    flags = {
        "consent_gap": False,
        "governance_gap": False,
        "ocap_restriction": False,
        "scope_mismatch": False,
        "withdrawn_consent": False,
        "expired_consent": False,
        "missing_consent": False,
    }

    consent_status = _get_first_present(record, ("current_consent_status", "status"))
    consent_id = _get_first_present(record, ("current_consent_consent_id", "consent_record_id"))
    sharing_scope_type = _normalize_scope(
        _get_first_present(record, ("current_consent_sharing_scope_type", "sharing_scope_type"))
    )
    allowed_scope_orgs = _get_first_present(
        record, ("current_consent_sharing_scope_agency_ids", "sharing_scope_agency_ids")
    )
    legal_basis = _get_first_present(record, ("current_consent_legal_basis", "legal_basis"))
    purpose_codes = _get_first_present(record, ("current_consent_purpose_codes", "purpose_codes"))
    effective_at = _to_naive_timestamp(_get_first_present(record, ("current_consent_effective_at", "effective_at")))
    expires_at = _to_naive_timestamp(_get_first_present(record, ("current_consent_expires_at", "expires_at")))
    withdrawal_date = _to_naive_timestamp(
        _get_first_present(record, ("current_consent_withdrawal_date", "withdrawal_date"))
    )
    ocap_protected = bool(_get_first_present(record, ("ocap_protected",)))

    if _is_empty_text(consent_status) or _is_empty_text(consent_id):
        flags["missing_consent"] = True
        reason_list.append("Missing consent record.")

    status_norm = "" if _is_empty_text(consent_status) else str(consent_status).strip().lower()
    if status_norm in BLOCKED_CONSENT_STATUSES and status_norm == "withdrawn" and (
        withdrawal_date is None or ref_time >= withdrawal_date
    ):
        flags["withdrawn_consent"] = True
        reason_list.append("Consent is withdrawn at reference time.")
    if status_norm in {"expired", "superseded"}:
        flags["expired_consent"] = True
        reason_list.append(f"Consent status '{status_norm}' blocks cross-agency sharing.")
    if status_norm in PENDING_CONSENT_STATUSES:
        flags["scope_mismatch"] = True
        reason_list.append("Consent is pending and requires verification before sharing.")

    if expires_at is not None and ref_time > expires_at:
        flags["expired_consent"] = True
        reason_list.append("Consent is expired at reference time.")

    if effective_at is not None and ref_time < effective_at:
        flags["consent_gap"] = True
        reason_list.append("Consent is not yet effective at reference time.")

    if is_cross_agency_action and sharing_scope_type in SINGLE_AGENCY_SCOPES:
        flags["scope_mismatch"] = True
        reason_list.append("Consent scope is single-agency only; multi-agency use blocked.")
    if is_cross_agency_action and sharing_scope_type in NO_SHARING_SCOPES:
        flags["scope_mismatch"] = True
        scope_hard_block = True
        reason_list.append("Consent scope explicitly blocks sharing.")

    if is_cross_agency_action and sharing_scope_type in (
        MULTI_AGENCY_SCOPES | {"named_agencies", "limited_agencies", "all_dsa_agencies", "all_partner_agencies"}
    ):
        scope_allowed, scope_mismatch = _is_scope_allowed_for_org(
            sharing_scope_type, context_org_id, allowed_scope_orgs
        )
        if not scope_allowed and scope_mismatch:
            flags["scope_mismatch"] = True
            reason_list.append("Requesting org is outside consent sharing scope.")

    legal_basis_norm = "" if _is_empty_text(legal_basis) else str(legal_basis).strip().lower()
    ocap_approved_scope = sharing_scope_type in {
        "all_dsa_agencies",
        "all_partner_agencies",
        "limited_agencies",
        "named_agencies",
    }
    ocap_approved_basis = ("ocap" in legal_basis_norm) or ("indigenous" in legal_basis_norm)
    if ocap_protected and is_cross_agency_action:
        if not (ocap_approved_scope or ocap_approved_basis):
            flags["ocap_restriction"] = True
            reason_list.append("OCAP-protected client requires explicit approved sharing scope.")
        else:
            flags["governance_gap"] = True
            reason_list.append("OCAP-protected client requires reviewer confirmation before sharing.")

    if legal_basis_norm in GOVERNMENT_LEGAL_BASES and _is_empty_text(purpose_codes):
        flags["governance_gap"] = True
        reason_list.append("Government/FOIPPA sharing record has empty purpose codes.")

    if flags["missing_consent"] or flags["expired_consent"] or flags["withdrawn_consent"] or scope_hard_block:
        flags["consent_gap"] = True

    hard_block = flags["consent_gap"] or flags["ocap_restriction"]
    has_governance_redaction = flags["governance_gap"] and not hard_block
    has_scope_limited_redaction = flags["scope_mismatch"] and not hard_block

    all_field_list = list(all_fields) if all_fields is not None else list(record.keys())
    sensitive = sorted(field for field in all_field_list if field in SENSITIVE_FIELDS)

    if hard_block:
        allowed = False
        view_status = "blocked"
        allowed_fields = [field for field in all_field_list if field in SAFE_BASE_FIELDS]
        redacted_fields = sorted(set(all_field_list) - set(allowed_fields))
    elif has_governance_redaction or has_scope_limited_redaction:
        allowed = True
        view_status = "redacted"
        redacted_fields = sensitive
        allowed_fields = [field for field in all_field_list if field not in set(redacted_fields)]
    else:
        allowed = True
        view_status = "full"
        allowed_fields = all_field_list
        redacted_fields = []

    if action == "auto_merge" and (not allowed or flags["ocap_restriction"] or flags["consent_gap"]):
        # Explicit safety guard for merge operations.
        allowed = False
        view_status = "blocked"
        if "Auto-merge blocked by policy restrictions." not in reason_list:
            reason_list.append("Auto-merge blocked by policy restrictions.")
        redacted_fields = sorted(set(all_field_list))
        allowed_fields = []

    return PolicyDecision(
        allowed=allowed,
        view_status=view_status,
        reasons=reason_list,
        flags=flags,
        allowed_fields=allowed_fields,
        redacted_fields=redacted_fields,
    )


def build_policy_view(
    referrals_enriched: pd.DataFrame,
    *,
    reference_time: pd.Timestamp | None = None,
    context_org_id: str | None = None,
    is_multi_agency_view: bool = True,
    action: str = "view",
) -> pd.DataFrame:
    """Apply policy evaluation to each row of referrals_enriched.

    Args:
        referrals_enriched: Enriched referrals table.
        reference_time: Evaluation timestamp.
        context_org_id: Requesting org id.
        is_multi_agency_view: Whether request is cross-org.
        action: Intended action context.

    Returns:
        DataFrame with appended policy decision columns.
    """
    tables = {"referrals_enriched": referrals_enriched}
    validate_required_columns(tables)

    missing_optional = sorted(POLICY_OPTIONAL_COLUMNS - set(referrals_enriched.columns))
    if missing_optional:
        LOGGER.warning(
            "Policy evaluation continuing with missing optional column(s): %s",
            ", ".join(missing_optional),
        )

    rows: list[dict[str, Any]] = []
    for _, row in referrals_enriched.iterrows():
        row_dict = row.to_dict()
        decision = evaluate_policy(
            row_dict,
            reference_time=reference_time,
            context_org_id=context_org_id,
            is_multi_agency_view=is_multi_agency_view,
            action=action,
            all_fields=referrals_enriched.columns,
        )
        out = dict(row_dict)
        out["policy_allowed"] = decision.allowed
        out["policy_view_status"] = decision.view_status
        out["policy_reasons"] = decision.reasons
        out["policy_flags"] = decision.flags
        out["policy_allowed_fields"] = decision.allowed_fields
        out["policy_redacted_fields"] = decision.redacted_fields
        rows.append(out)

    return pd.DataFrame(rows)


def route_duplicate_decision(
    duplicate_row: Mapping[str, Any],
    policy_decision: PolicyDecision,
    *,
    auto_merge_threshold: float = 0.9,
) -> str:
    """Route duplicate decision per policy redlines.

    Returns one of:
      - ``auto_merge_eligible``
      - ``review_required``
      - ``blocked_by_policy``
    """
    restricted = (
        (not policy_decision.allowed)
        or policy_decision.flags.get("ocap_restriction", False)
        or policy_decision.flags.get("consent_gap", False)
    )
    if restricted:
        return "blocked_by_policy"

    is_true_duplicate = bool(duplicate_row.get("is_true_duplicate", False))
    match_score = duplicate_row.get("match_score", 0.0)
    try:
        match_score_f = float(match_score)
    except (TypeError, ValueError):
        match_score_f = 0.0

    if is_true_duplicate and match_score_f >= auto_merge_threshold:
        return "auto_merge_eligible"
    return "review_required"


def assert_valid_duplicate_decision(decision: str) -> None:
    """Ensure duplicate decision is within the allowed routing contract."""
    if decision not in DUPLICATE_DECISIONS:
        raise ValueError(
            f"Invalid duplicate decision '{decision}'. "
            f"Expected one of: {', '.join(sorted(DUPLICATE_DECISIONS))}."
        )
