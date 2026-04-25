from __future__ import annotations

import pandas as pd

from src.loaders import build_referrals_enriched, get_required_columns, validate_required_columns


def _core_tables() -> dict[str, pd.DataFrame]:
    orgs = pd.DataFrame(
        [
            {"org_id": "ORG-A", "org_name": "Org A", "org_type": "shelter"},
            {"org_id": "ORG-B", "org_name": "Org B", "org_type": "health"},
        ]
    )
    clients = pd.DataFrame(
        [
            {
                "client_id": "C1",
                "first_name": "Ana",
                "last_name": "Lee",
                "dob": "1990-01-01",
                "primary_org_id": "ORG-A",
                "current_consent_id": "CONS-1",
                "vi_spdat_score": 8,
                "indigenous_flag": "none",
                "address_line1": "x",
                "ocap_protected": False,
                "default_sharing_scope": "cluster",
            }
        ]
    )
    referrals = pd.DataFrame(
        [
            {
                "referral_id": "R1",
                "client_id": "C1",
                "referring_org_id": "ORG-A",
                "receiving_org_id": "ORG-B",
                "consent_record_id": "CONS-1",
                "status": "submitted",
                "submitted_at": "2026-01-01",
                "reason_code": "housing",
            }
        ]
    )
    encounters = pd.DataFrame(
        [{"encounter_id": "E1", "client_id": "C1", "org_id": "ORG-A", "occurred_at": "2026-01-01"}]
    )
    consent = pd.DataFrame(
        [
            {
                "consent_id": "CONS-1",
                "client_id": "C1",
                "collecting_org_id": "ORG-A",
                "status": "active",
                "sharing_scope_type": "cluster",
                "effective_at": "2025-01-01",
                "expires_at": "2027-01-01",
                "purpose_codes": "service_delivery",
                "withdrawal_date": None,
                "dsa_id": "DSA-1",
            }
        ]
    )
    dsa = pd.DataFrame(
        [{"dsa_id": "DSA-1", "type": "multi_party", "effective_at": "2025-01-01", "expires_at": "2027-01-01"}]
    )
    duplicate_flags = pd.DataFrame([{"client_id_a": "C1", "client_id_b": "C2", "review_status": "pending"}])
    return {
        "orgs": orgs,
        "clients": clients,
        "referrals": referrals,
        "encounters": encounters,
        "consent": consent,
        "dsa": dsa,
        "duplicate_flags": duplicate_flags,
    }


def test_required_columns_validate_for_core_tables() -> None:
    tables = _core_tables()
    validate_required_columns(tables, get_required_columns())


def test_referrals_enriched_builds_on_small_fixture() -> None:
    tables = _core_tables()
    enriched = build_referrals_enriched(tables)
    assert not enriched.empty
    assert {"referral_id", "client_id", "referring_org_id", "receiving_org_id"}.issubset(enriched.columns)
