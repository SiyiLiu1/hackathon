from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

app = pytest.importorskip("api.main").app
clients_route = pytest.importorskip("api.routes.clients")


def _tables_for_api() -> dict[str, pd.DataFrame]:
    clients = pd.DataFrame(
        [
            {
                "client_id": "C-ALLOW",
                "first_name": "Ana",
                "last_name": "Lee",
                "ocap_protected": False,
                "primary_org_id": "ORG-A",
                "current_consent_id": "CONS-A",
            },
            {
                "client_id": "C-RESTRICT",
                "first_name": "Bob",
                "last_name": "Ray",
                "ocap_protected": True,
                "primary_org_id": "ORG-A",
                "current_consent_id": "CONS-R",
            },
        ]
    )
    consent = pd.DataFrame(
        [
            {
                "consent_id": "CONS-A",
                "client_id": "C-ALLOW",
                "status": "active",
                "sharing_scope_type": "cluster",
                "effective_at": "2025-01-01",
                "expires_at": "2027-01-01",
                "withdrawal_date": None,
                "legal_basis": "consent",
                "purpose_codes": "service_delivery",
            },
            {
                "consent_id": "CONS-R",
                "client_id": "C-RESTRICT",
                "status": "active",
                "sharing_scope_type": "single_agency_only",
                "effective_at": "2025-01-01",
                "expires_at": "2027-01-01",
                "withdrawal_date": None,
                "legal_basis": "consent",
                "purpose_codes": "service_delivery",
            },
        ]
    )
    referrals_enriched = pd.DataFrame(
        [
            {
                "referral_id": "R1",
                "client_id": "C-ALLOW",
                "referring_org_id": "ORG-A",
                "receiving_org_id": "ORG-B",
                "consent_record_id": "CONS-A",
                "submitted_at": "2026-01-01",
            },
            {
                "referral_id": "R2",
                "client_id": "C-RESTRICT",
                "referring_org_id": "ORG-A",
                "receiving_org_id": "ORG-B",
                "consent_record_id": "CONS-R",
                "submitted_at": "2026-01-01",
            },
        ]
    )
    return {"clients": clients, "consent": consent, "referrals_enriched": referrals_enriched}


def test_safe_view_returns_404_for_missing_client(monkeypatch) -> None:
    monkeypatch.setattr(clients_route, "_load_shared_tables", lambda raw_dir: _tables_for_api())
    client = TestClient(app)
    resp = client.get("/clients/NOPE/safe-view")
    assert resp.status_code == 404


def test_safe_view_redacts_restricted_fields(monkeypatch) -> None:
    monkeypatch.setattr(clients_route, "_load_shared_tables", lambda raw_dir: _tables_for_api())
    client = TestClient(app)
    resp = client.get("/clients/C-RESTRICT/safe-view")
    body = resp.json()
    assert resp.status_code == 200
    assert body["policy"]["view_status"] == "blocked"
    assert "first_name" not in body["safe_payload"]
    assert body["safe_payload"]["client_id"] == "C-RESTRICT"


def test_safe_view_returns_policy_metadata(monkeypatch) -> None:
    monkeypatch.setattr(clients_route, "_load_shared_tables", lambda raw_dir: _tables_for_api())
    client = TestClient(app)
    resp = client.get("/clients/C-ALLOW/safe-view")
    body = resp.json()
    assert resp.status_code == 200
    assert "policy" in body
    assert {"allowed", "view_status", "reasons", "flags", "allowed_fields", "redacted_fields"}.issubset(
        body["policy"].keys()
    )
