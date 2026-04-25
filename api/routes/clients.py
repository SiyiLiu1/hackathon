"""Client-facing safe-view routes."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.loaders import build_referrals_enriched, load_track1_data
from src.policy_engine import PolicyDecision, evaluate_policy

router = APIRouter(prefix="/clients", tags=["clients"])

DEFAULT_TRACK1_RAW_DIR = str(
    Path(__file__).resolve().parents[2] / "tracks" / "referral-care-coordination" / "data" / "raw"
)


class PolicyMetadata(BaseModel):
    """Policy metadata returned with safe payload."""

    allowed: bool
    view_status: str
    reasons: list[str] = Field(default_factory=list)
    flags: dict[str, bool] = Field(default_factory=dict)
    allowed_fields: list[str] = Field(default_factory=list)
    redacted_fields: list[str] = Field(default_factory=list)


class SafeClientViewResponse(BaseModel):
    """Safe-view response for one client."""

    client_id: str
    safe_payload: dict[str, Any] = Field(default_factory=dict)
    policy: PolicyMetadata


def _series_to_dict(series: pd.Series | None) -> dict[str, Any]:
    """Convert optional Series to dict."""
    if series is None:
        return {}
    return series.to_dict()


@lru_cache(maxsize=4)
def _load_shared_tables(raw_dir: str) -> dict[str, pd.DataFrame]:
    """Load/cached Track 1 data and enriched join."""
    tables = load_track1_data(raw_dir=raw_dir)
    tables["referrals_enriched"] = build_referrals_enriched(tables)
    return tables


def _latest_referral(client_referrals: pd.DataFrame) -> pd.Series | None:
    """Return latest referral row for a client."""
    if client_referrals.empty:
        return None
    rows = client_referrals.copy()
    if "submitted_at" in rows.columns:
        rows["submitted_at_dt"] = pd.to_datetime(rows["submitted_at"], errors="coerce")
        rows = rows.sort_values("submitted_at_dt", ascending=False)
    return rows.iloc[0]


def _build_policy_record(
    client_row: pd.Series, consent_row: pd.Series | None, referral_row: pd.Series | None
) -> dict[str, Any]:
    """Build policy context record."""
    record: dict[str, Any] = {}
    record.update(client_row.to_dict())
    if consent_row is not None:
        for key, value in consent_row.to_dict().items():
            record[f"current_consent_{key}"] = value
    if referral_row is not None:
        record.update(referral_row.to_dict())
    else:
        record.setdefault("referral_id", "N/A")
        record.setdefault("referring_org_id", client_row.get("primary_org_id"))
        record.setdefault("receiving_org_id", client_row.get("primary_org_id"))
        record.setdefault("consent_record_id", client_row.get("current_consent_id"))
    return record


def _to_policy_metadata(decision: PolicyDecision) -> PolicyMetadata:
    """Convert PolicyDecision object to response model."""
    return PolicyMetadata(
        allowed=decision.allowed,
        view_status=decision.view_status,
        reasons=decision.reasons,
        flags=decision.flags,
        allowed_fields=decision.allowed_fields,
        redacted_fields=decision.redacted_fields,
    )


@router.get("/{client_id}/safe-view", response_model=SafeClientViewResponse)
def get_client_safe_view(
    client_id: str,
    raw_dir: str = Query(default=DEFAULT_TRACK1_RAW_DIR, description="Track 1 raw parquet directory."),
) -> SafeClientViewResponse:
    """Return policy-approved safe payload for one client.

    Conservative-by-default behavior:
      - If consent/policy context is incomplete, return blocked/redacted-safe metadata and minimal payload.
      - Never return raw fields that rely on frontend-side hiding.
    """
    try:
        tables = _load_shared_tables(raw_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load data: {exc}") from exc

    clients = tables["clients"]
    consent = tables["consent"]
    referrals_enriched = tables["referrals_enriched"]

    client_matches = clients[clients["client_id"].astype(str) == str(client_id)]
    if client_matches.empty:
        raise HTTPException(status_code=404, detail=f"Client not found: {client_id}")
    client_row = client_matches.iloc[0]

    consent_row = None
    current_consent_id = client_row.get("current_consent_id")
    if current_consent_id is not None and "consent_id" in consent.columns:
        consent_matches = consent[consent["consent_id"].astype(str) == str(current_consent_id)]
        if not consent_matches.empty:
            consent_row = consent_matches.iloc[0]

    client_referrals = referrals_enriched[referrals_enriched["client_id"].astype(str) == str(client_id)].copy()
    referral_row = _latest_referral(client_referrals)
    policy_record = _build_policy_record(client_row, consent_row, referral_row)

    try:
        decision = evaluate_policy(
            policy_record,
            is_multi_agency_view=True,
            action="api",
            all_fields=policy_record.keys(),
        )
    except Exception:
        # Conservative fallback for incomplete policy context.
        fallback_decision = PolicyDecision(
            allowed=False,
            view_status="blocked",
            reasons=["Policy context incomplete. Returning conservative safe payload."],
            flags={
                "consent_gap": True,
                "governance_gap": False,
                "ocap_restriction": False,
                "scope_mismatch": False,
                "withdrawn_consent": False,
                "expired_consent": False,
                "missing_consent": True,
            },
            allowed_fields=["client_id"],
            redacted_fields=sorted(set(policy_record.keys()) - {"client_id"}),
        )
        return SafeClientViewResponse(
            client_id=str(client_id),
            safe_payload={"client_id": str(client_id)},
            policy=_to_policy_metadata(fallback_decision),
        )

    client_payload = _series_to_dict(client_row)
    consent_payload = {f"current_consent_{k}": v for k, v in _series_to_dict(consent_row).items()}
    merged_payload = {}
    merged_payload.update(client_payload)
    merged_payload.update(consent_payload)

    safe_payload = {
        field: merged_payload.get(field)
        for field in decision.allowed_fields
        if field in merged_payload
    }
    safe_payload["client_id"] = str(client_id)

    return SafeClientViewResponse(
        client_id=str(client_id),
        safe_payload=safe_payload,
        policy=_to_policy_metadata(decision),
    )

