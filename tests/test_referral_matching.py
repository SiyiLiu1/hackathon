from __future__ import annotations

import pandas as pd

from src.policy_engine import PolicyDecision
from src.referral_matching import rank_receiving_orgs


def _orgs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "org_id": "ORG-A",
                "org_name": "Org A",
                "org_type": "housing",
                "service_taxonomy_code": "housing",
                "capacity_total_slots": 10,
                "capacity_occupied_slots": 5,
                "waitlist_flag": False,
            },
            {
                "org_id": "ORG-B",
                "org_name": "Org B",
                "org_type": "housing",
                "service_taxonomy_code": "housing",
                "capacity_total_slots": 8,
                "capacity_occupied_slots": 7,
                "waitlist_flag": True,
            },
        ]
    )


def _referrals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "receiving_org_id": "ORG-A",
                "status": "accepted",
                "submitted_at": "2026-01-01",
                "decision_at": "2026-01-02",
                "vi_spdat_score": 7,
            },
            {
                "receiving_org_id": "ORG-B",
                "status": "declined",
                "submitted_at": "2026-01-01",
                "decision_at": "2026-01-04",
                "vi_spdat_score": 9,
            },
        ]
    )


def _encounters() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"org_id": "ORG-A", "outcome_flag": "positive", "occurred_at": "2026-01-05"},
            {"org_id": "ORG-B", "outcome_flag": "negative", "occurred_at": "2026-01-05"},
        ]
    )


def test_matcher_not_ranked_when_policy_blocks() -> None:
    decision = PolicyDecision(
        allowed=False,
        view_status="blocked",
        reasons=["blocked"],
        flags={"consent_gap": True, "ocap_restriction": False, "scope_mismatch": False},
        allowed_fields=[],
        redacted_fields=[],
    )
    out = rank_receiving_orgs(
        client_row={"client_id": "C1", "referral_type": "housing"},
        policy_decision=decision,
        orgs_df=_orgs(),
        referrals_enriched_df=_referrals(),
        encounters_df=_encounters(),
    )
    assert out.empty


def test_matcher_returns_action_label_and_explanation() -> None:
    decision = PolicyDecision(
        allowed=True,
        view_status="redacted",
        reasons=[],
        flags={"consent_gap": False, "ocap_restriction": False, "scope_mismatch": False},
        allowed_fields=[],
        redacted_fields=[],
    )
    out = rank_receiving_orgs(
        client_row={"client_id": "C1", "referral_type": "housing", "vi_spdat_score": 8},
        policy_decision=decision,
        orgs_df=_orgs(),
        referrals_enriched_df=_referrals(),
        encounters_df=_encounters(),
    )
    assert not out.empty
    assert {"action_label", "explanation", "score"}.issubset(out.columns)
    assert out["action_label"].iloc[0] == "refer_with_redaction"
