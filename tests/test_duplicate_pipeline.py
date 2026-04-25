from __future__ import annotations

import pandas as pd

from src.duplicate_pipeline import (
    assign_duplicate_decision,
    evaluate_duplicates,
    generate_candidate_pairs,
    score_duplicate_pairs,
)


def test_policy_restricted_clients_are_not_auto_merged() -> None:
    clients = pd.DataFrame(
        [
            {
                "client_id": "C1",
                "first_name": "Ana",
                "last_name": "Lee",
                "dob": "1990-01-01",
                "ocap_protected": True,
                "consent_coverage_level": "explicit",
                "default_sharing_scope": "cluster",
            },
            {
                "client_id": "C2",
                "first_name": "Ana",
                "last_name": "Lee",
                "dob": "1990-01-01",
                "ocap_protected": False,
                "consent_coverage_level": "explicit",
                "default_sharing_scope": "cluster",
            },
        ]
    )
    pairs = generate_candidate_pairs(clients)
    features = pd.DataFrame(
        [
            {
                "client_id_a": "C1",
                "client_id_b": "C2",
                "first_name_similarity": 1.0,
                "last_name_similarity": 1.0,
                "full_name_similarity": 1.0,
                "dob_exact_match": 1.0,
                "dob_year_match": 1.0,
                "alias_similarity": 0.0,
                "address_similarity": 0.0,
                "phone_exact_match": 0.0,
                "same_primary_org": 0.0,
            }
        ]
    )
    scored = score_duplicate_pairs(features)
    decided = assign_duplicate_decision(scored, clients_df=clients)
    assert not pairs.empty
    assert decided["decision_label"].iloc[0] == "blocked_by_policy"


def test_evaluation_returns_precision_recall_f1_keys() -> None:
    scored = pd.DataFrame(
        [
            {"client_id_a": "C1", "client_id_b": "C2", "match_score": 0.9},
            {"client_id_a": "C3", "client_id_b": "C4", "match_score": 0.1},
        ]
    )
    dup_flags = pd.DataFrame(
        [
            {
                "client_id_primary": "C1",
                "client_id_secondary": "C2",
                "review_status": "confirmed_duplicate",
            },
            {
                "client_id_primary": "C3",
                "client_id_secondary": "C4",
                "review_status": "not_duplicate",
            },
        ]
    )
    metrics = evaluate_duplicates(scored, dup_flags)
    assert {"precision", "recall", "f1"}.issubset(metrics.keys())
