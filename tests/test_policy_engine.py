from __future__ import annotations

import pandas as pd

from src.policy_engine import evaluate_access, evaluate_policy


def _tables(consent_status: str, scope: str, ocap: bool = False) -> dict[str, pd.DataFrame]:
    clients = pd.DataFrame(
        [
            {
                "client_id": "C1",
                "primary_org_id": "ORG-A",
                "current_consent_id": "CONS-1",
                "ocap_protected": ocap,
                "first_name": "Ana",
                "last_name": "Lee",
            }
        ]
    )
    consent = pd.DataFrame(
        [
            {
                "consent_id": "CONS-1",
                "client_id": "C1",
                "status": consent_status,
                "sharing_scope_type": scope,
                "legal_basis": "pipa",
                "effective_at": "2025-01-01",
                "expires_at": "2030-01-01",
            }
        ]
    )
    referrals_enriched = pd.DataFrame(
        [
            {
                "referral_id": "R1",
                "client_id": "C1",
                "referring_org_id": "ORG-A",
                "receiving_org_id": "ORG-B",
                "consent_record_id": "CONS-1",
                "status": "submitted",
                "submitted_at": "2026-01-01",
            }
        ]
    )
    return {"clients": clients, "consent": consent, "referrals_enriched": referrals_enriched}


def test_evaluate_access_ready_for_active_broad_scope() -> None:
    access = evaluate_access("C1", "view", _tables(consent_status="active", scope="all_dsa_agencies"))
    assert access["decision"] == "ready"
    assert access["issues"] == []


def test_evaluate_access_limited_for_single_agency_scope() -> None:
    access = evaluate_access("C1", "view", _tables(consent_status="active", scope="single_agency_only"))
    assert access["decision"] == "limited"
    assert any(issue["code"] == "scope_restriction" for issue in access["issues"])


def test_evaluate_access_blocked_for_withdrawn_consent() -> None:
    access = evaluate_access("C1", "view", _tables(consent_status="withdrawn", scope="all_dsa_agencies"))
    assert access["decision"] == "blocked"
    assert any(issue["code"] == "consent_withdrawn" for issue in access["issues"])


def test_evaluate_policy_blocked_keeps_minimum_safe_fields() -> None:
    decision = evaluate_policy(
        {
            "client_id": "C1",
            "first_name": "Ana",
            "last_name": "Lee",
            "ocap_protected": False,
            "current_consent_status": "withdrawn",
            "current_consent_consent_id": "CONS-1",
            "referral_id": "R1",
            "referring_org_id": "ORG-A",
            "receiving_org_id": "ORG-B",
        },
        action="view",
        is_multi_agency_view=True,
        all_fields=[
            "client_id",
            "first_name",
            "last_name",
            "referral_id",
            "referring_org_id",
            "receiving_org_id",
        ],
    )
    assert decision.view_status == "blocked"
    assert "client_id" in decision.allowed_fields
    assert "first_name" in decision.redacted_fields
