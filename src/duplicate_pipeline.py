"""Duplicate triage pipeline for Track 1 client records.

This module provides explainable duplicate candidate generation, feature
engineering, heuristic scoring, policy-safe decision assignment, and metric
evaluation against ground-truth duplicate flags.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

import pandas as pd


DECISION_AUTO_MERGE = "auto_merge_eligible"
DECISION_REVIEW = "review_required"
DECISION_BLOCKED = "blocked_by_policy"

VALID_DECISIONS = {DECISION_AUTO_MERGE, DECISION_REVIEW, DECISION_BLOCKED}

# Explicit thresholds for easy debugging/tuning.
AUTO_MERGE_THRESHOLD = 0.88
REVIEW_THRESHOLD = 0.62


def _require_columns(df: pd.DataFrame, table_name: str, required: list[str]) -> None:
    """Validate required columns for a DataFrame."""
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Table '{table_name}' is missing required column(s): {', '.join(missing)}")


def _is_truthy(value: Any) -> bool:
    """Interpret booleans robustly, including string forms."""
    if value is None or pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return _norm_text(value) in {"1", "true", "t", "yes", "y"}


def _norm_text(value: Any) -> str:
    """Normalize to lowercase stripped text."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def _norm_date(value: Any) -> str:
    """Normalize date-like value to YYYY-MM-DD when possible."""
    if value is None or pd.isna(value):
        return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y-%m-%d")


def _norm_phone(value: Any) -> str:
    """Keep digits only for phone comparison."""
    if value is None or pd.isna(value):
        return ""
    return "".join(ch for ch in str(value) if ch.isdigit())


def soundex(name: str) -> str:
    """Compute Soundex code for a name."""
    text = _norm_text(name)
    if not text:
        return "0000"

    first = text[0].upper()
    mapping = {
        **{c: "1" for c in "bfpv"},
        **{c: "2" for c in "cgjkqsxz"},
        **{c: "3" for c in "dt"},
        "l": "4",
        **{c: "5" for c in "mn"},
        "r": "6",
    }

    digits: list[str] = []
    prev = mapping.get(text[0], "")
    for ch in text[1:]:
        code = mapping.get(ch, "")
        if code != prev and code != "":
            digits.append(code)
        prev = code

    result = (first + "".join(digits) + "000")[:4]
    return result


def string_similarity(a: Any, b: Any) -> float:
    """Return similarity in [0, 1] using SequenceMatcher."""
    left = _norm_text(a)
    right = _norm_text(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def generate_candidate_pairs(clients_df: pd.DataFrame) -> pd.DataFrame:
    """Generate candidate duplicate pairs using blocking rules.

    Blocking keys:
      - soundex(last_name) + dob
      - soundex(last_name) + first initial
      - normalized phone (if present)
    """
    required = {"client_id", "first_name", "last_name", "dob"}
    missing = sorted(required - set(clients_df.columns))
    if missing:
        raise ValueError(
            "Table 'clients' is missing required column(s): " + ", ".join(missing)
        )

    c = clients_df.copy()
    c["first_name_n"] = c["first_name"].map(_norm_text)
    c["last_name_n"] = c["last_name"].map(_norm_text)
    c["dob_n"] = c["dob"].map(_norm_date)
    c["ln_sdx"] = c["last_name_n"].map(soundex)
    c["fi"] = c["first_name_n"].str[:1].fillna("")
    if "phone" in c.columns:
        c["phone_n"] = c["phone"].map(_norm_phone)
    else:
        c["phone_n"] = ""

    pairs: set[tuple[str, str]] = set()

    def _collect(block: pd.DataFrame) -> None:
        ids = [str(v) for v in block["client_id"].tolist() if not pd.isna(v)]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = sorted((ids[i], ids[j]))
                if a != b:
                    pairs.add((a, b))

    for _, block in c.groupby(["ln_sdx", "dob_n"], dropna=False):
        if len(block) > 1:
            _collect(block)
    for _, block in c.groupby(["ln_sdx", "fi"], dropna=False):
        if len(block) > 1:
            _collect(block)
    for _, block in c[c["phone_n"] != ""].groupby("phone_n", dropna=False):
        if len(block) > 1:
            _collect(block)

    out = pd.DataFrame(sorted(pairs), columns=["client_id_a", "client_id_b"])
    return out


def engineer_duplicate_features(
    clients_df: pd.DataFrame, candidate_pairs_df: pd.DataFrame
) -> pd.DataFrame:
    """Engineer pairwise duplicate features for each candidate pair."""
    required_clients = {"client_id", "first_name", "last_name", "dob"}
    missing_clients = sorted(required_clients - set(clients_df.columns))
    if missing_clients:
        raise ValueError(
            "Table 'clients' is missing required column(s): " + ", ".join(missing_clients)
        )
    required_pairs = {"client_id_a", "client_id_b"}
    missing_pairs = sorted(required_pairs - set(candidate_pairs_df.columns))
    if missing_pairs:
        raise ValueError(
            "Table 'candidate_pairs' is missing required column(s): "
            + ", ".join(missing_pairs)
        )

    base = clients_df.copy().set_index("client_id", drop=False)

    rows: list[dict[str, Any]] = []
    for _, pair in candidate_pairs_df.iterrows():
        ida = pair["client_id_a"]
        idb = pair["client_id_b"]
        if ida not in base.index or idb not in base.index:
            continue
        a = base.loc[ida]
        b = base.loc[idb]

        first_sim = string_similarity(a.get("first_name"), b.get("first_name"))
        last_sim = string_similarity(a.get("last_name"), b.get("last_name"))
        full_sim = string_similarity(
            f"{a.get('first_name', '')} {a.get('last_name', '')}",
            f"{b.get('first_name', '')} {b.get('last_name', '')}",
        )
        dob_match = float(_norm_date(a.get("dob")) == _norm_date(b.get("dob")) and _norm_date(a.get("dob")) != "")
        dob_year_match = 0.0
        da = pd.to_datetime(a.get("dob"), errors="coerce")
        db = pd.to_datetime(b.get("dob"), errors="coerce")
        if not pd.isna(da) and not pd.isna(db):
            dob_year_match = float(da.year == db.year)

        alias_sim = 0.0
        if "aliases" in base.columns:
            alias_sim = string_similarity(a.get("aliases"), b.get("aliases"))

        address_sim = 0.0
        address_col = None
        if "address_line1" in base.columns:
            address_col = "address_line1"
        elif "current_sleeping_location" in base.columns:
            address_col = "current_sleeping_location"
        if address_col:
            address_sim = string_similarity(a.get(address_col), b.get(address_col))

        phone_match = 0.0
        if "phone" in base.columns:
            pa = _norm_phone(a.get("phone"))
            pb = _norm_phone(b.get("phone"))
            phone_match = float(pa != "" and pa == pb)

        same_primary_org = 0.0
        if "primary_org_id" in base.columns:
            same_primary_org = float(
                _norm_text(a.get("primary_org_id")) != ""
                and _norm_text(a.get("primary_org_id")) == _norm_text(b.get("primary_org_id"))
            )

        rows.append(
            {
                "client_id_a": ida,
                "client_id_b": idb,
                "first_name_similarity": first_sim,
                "last_name_similarity": last_sim,
                "full_name_similarity": full_sim,
                "dob_exact_match": dob_match,
                "dob_year_match": dob_year_match,
                "alias_similarity": alias_sim,
                "address_similarity": address_sim,
                "phone_exact_match": phone_match,
                "same_primary_org": same_primary_org,
            }
        )

    return pd.DataFrame(rows)


def score_duplicate_pairs(
    features_df: pd.DataFrame, dup_flags_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Score candidate pairs using explicit weighted rules."""
    required = {
        "client_id_a",
        "client_id_b",
        "first_name_similarity",
        "last_name_similarity",
        "full_name_similarity",
        "dob_exact_match",
        "dob_year_match",
        "alias_similarity",
        "address_similarity",
        "phone_exact_match",
        "same_primary_org",
    }
    missing = sorted(required - set(features_df.columns))
    if missing:
        raise ValueError(
            "Table 'features' is missing required column(s): " + ", ".join(missing)
        )

    s = features_df.copy()
    # Explicit weighted sum formula.
    s["match_score"] = (
        0.22 * s["full_name_similarity"]
        + 0.16 * s["last_name_similarity"]
        + 0.12 * s["first_name_similarity"]
        + 0.22 * s["dob_exact_match"]
        + 0.06 * s["dob_year_match"]
        + 0.08 * s["alias_similarity"]
        + 0.07 * s["address_similarity"]
        + 0.05 * s["phone_exact_match"]
        + 0.02 * s["same_primary_org"]
    ).clip(lower=0.0, upper=1.0)

    def _explain(row: pd.Series) -> str:
        reasons: list[str] = []
        if row["dob_exact_match"] >= 1.0:
            reasons.append("exact_dob")
        elif row["dob_year_match"] >= 1.0:
            reasons.append("dob_year_match")
        if row["last_name_similarity"] >= 0.92:
            reasons.append("very_close_last_name")
        if row["first_name_similarity"] >= 0.90:
            reasons.append("very_close_first_name")
        if row["alias_similarity"] >= 0.80:
            reasons.append("alias_overlap")
        if row["address_similarity"] >= 0.80:
            reasons.append("address_overlap")
        if row["phone_exact_match"] >= 1.0:
            reasons.append("same_phone")
        if not reasons:
            reasons.append("weak_signals_review")
        return ";".join(reasons)

    s["explanation"] = s.apply(_explain, axis=1)

    if dup_flags_df is not None and len(dup_flags_df) > 0:
        if {"client_id_a", "client_id_b"}.issubset(dup_flags_df.columns):
            truth = dup_flags_df[["client_id_a", "client_id_b"]].copy()
        elif {"client_id_primary", "client_id_secondary"}.issubset(dup_flags_df.columns):
            truth = dup_flags_df.rename(
                columns={"client_id_primary": "client_id_a", "client_id_secondary": "client_id_b"}
            )[["client_id_a", "client_id_b"]].copy()
        else:
            truth = pd.DataFrame(columns=["client_id_a", "client_id_b"])

        if "review_status" in dup_flags_df.columns and len(truth) > 0:
            m = dup_flags_df.copy()
            if {"client_id_primary", "client_id_secondary"}.issubset(m.columns):
                m = m.rename(
                    columns={"client_id_primary": "client_id_a", "client_id_secondary": "client_id_b"}
                )
            m["review_status_n"] = m["review_status"].map(_norm_text)
            m["is_true_duplicate"] = m["review_status_n"].isin(["confirmed_duplicate", "merged"])
            truth = m[["client_id_a", "client_id_b", "is_true_duplicate"]].copy()
            truth["client_id_a"] = truth["client_id_a"].astype(str)
            truth["client_id_b"] = truth["client_id_b"].astype(str)
            truth[["client_id_a", "client_id_b"]] = truth.apply(
                lambda r: pd.Series(sorted([r["client_id_a"], r["client_id_b"]])), axis=1
            )
            truth = truth.drop_duplicates(subset=["client_id_a", "client_id_b"], keep="last")
            s = s.merge(truth, on=["client_id_a", "client_id_b"], how="left")
        else:
            s["is_true_duplicate"] = pd.NA
    else:
        s["is_true_duplicate"] = pd.NA

    return s.sort_values("match_score", ascending=False).reset_index(drop=True)


def assign_duplicate_decision(
    scored_df: pd.DataFrame, clients_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Assign duplicate decisions with policy-safe blocking rules.

    Rule order:
      1) Block when either record is restricted.
      2) Auto-merge when score >= AUTO_MERGE_THRESHOLD.
      3) Review when score >= REVIEW_THRESHOLD.
      4) Review for low-score tail (never silent-drop in this pipeline).
    """
    required = {"client_id_a", "client_id_b", "match_score"}
    missing = sorted(required - set(scored_df.columns))
    if missing:
        raise ValueError(
            "Table 'scored' is missing required column(s): " + ", ".join(missing)
        )

    out = scored_df.copy()

    out["is_policy_restricted"] = False
    out["restriction_reason"] = ""

    if clients_df is not None and len(clients_df) > 0:
        _require_columns(clients_df, "clients", ["client_id"])
        c = clients_df.copy()
        c["client_id"] = c["client_id"].astype(str)
        c = c.set_index("client_id", drop=False)

        def _restricted(client_id: str) -> tuple[bool, str]:
            if client_id not in c.index:
                return False, ""
            row = c.loc[client_id]
            reasons: list[str] = []
            if "ocap_protected" in c.columns and _is_truthy(row.get("ocap_protected")):
                reasons.append("ocap_protected")

            if "consent_coverage_level" in c.columns:
                ccl = _norm_text(row.get("consent_coverage_level"))
                if ccl in {"none", "revoked", "withdrawn"}:
                    reasons.append("consent_restricted")

            # Support scope restrictions whether modeled as default scope
            # or explicit consent scope columns.
            if "default_sharing_scope" in c.columns:
                scope = _norm_text(row.get("default_sharing_scope"))
                if scope in {"none", "single_agency_only", "org", "single_org"}:
                    reasons.append("scope_restricted")
            if "sharing_scope_type" in c.columns:
                scope = _norm_text(row.get("sharing_scope_type"))
                if scope in {"single_agency_only", "org", "single_org", "none"}:
                    reasons.append("scope_restricted")
            if "current_consent_sharing_scope_type" in c.columns:
                scope = _norm_text(row.get("current_consent_sharing_scope_type"))
                if scope in {"single_agency_only", "org", "single_org", "none"}:
                    reasons.append("scope_restricted")

            # Support consent status modeled in several possible columns.
            for status_col in ("consent_status", "current_consent_status", "status"):
                if status_col in c.columns:
                    status_v = _norm_text(row.get(status_col))
                    if status_v in {"withdrawn", "expired", "revoked"}:
                        reasons.append("consent_restricted")
                        break

            # Expired consent date guards when expiry columns are present.
            for expiry_col in ("consent_expires_at", "current_consent_expires_at", "expires_at", "expiry_date"):
                if expiry_col in c.columns:
                    expiry_ts = pd.to_datetime(row.get(expiry_col), errors="coerce")
                    if not pd.isna(expiry_ts) and pd.Timestamp.utcnow() > expiry_ts:
                        reasons.append("consent_restricted")
                        break

            # Generic policy flags if already propagated to client rows.
            for policy_col, reason in (
                ("is_policy_restricted", "policy_restricted"),
                ("policy_restricted", "policy_restricted"),
                ("consent_gap", "consent_gap"),
            ):
                if policy_col in c.columns and _is_truthy(row.get(policy_col)):
                    reasons.append(reason)

            return (len(reasons) > 0), ";".join(reasons)

        restricted_flags: list[bool] = []
        restricted_reasons: list[str] = []
        for _, row in out.iterrows():
            ra, rea = _restricted(str(row["client_id_a"]))
            rb, reb = _restricted(str(row["client_id_b"]))
            restricted = ra or rb
            reason = ";".join([part for part in [rea, reb] if part != ""])
            restricted_flags.append(restricted)
            restricted_reasons.append(reason)
        out["is_policy_restricted"] = restricted_flags
        out["restriction_reason"] = restricted_reasons

    out["decision_label"] = DECISION_REVIEW
    out.loc[out["match_score"] >= REVIEW_THRESHOLD, "decision_label"] = DECISION_REVIEW
    out.loc[out["match_score"] >= AUTO_MERGE_THRESHOLD, "decision_label"] = DECISION_AUTO_MERGE
    out.loc[out["is_policy_restricted"], "decision_label"] = DECISION_BLOCKED

    out["decision_explanation"] = out.apply(
        lambda r: (
            "blocked_by_policy:" + r["restriction_reason"]
            if r["decision_label"] == DECISION_BLOCKED
            else (
                f"score>={AUTO_MERGE_THRESHOLD:.2f}"
                if r["decision_label"] == DECISION_AUTO_MERGE
                else f"score<{AUTO_MERGE_THRESHOLD:.2f}; review_required"
            )
        ),
        axis=1,
    )

    bad = set(out["decision_label"].unique()) - VALID_DECISIONS
    if bad:
        raise ValueError(
            "Invalid decision labels produced: " + ", ".join(sorted(str(v) for v in bad))
        )

    return out


def evaluate_duplicates(scored_df: pd.DataFrame, dup_flags_df: pd.DataFrame) -> dict:
    """Evaluate duplicate predictions vs duplicate_flags truth using precision/recall/F1."""
    required_scored = {"client_id_a", "client_id_b", "match_score"}
    missing_scored = sorted(required_scored - set(scored_df.columns))
    if missing_scored:
        raise ValueError(
            "Table 'scored' is missing required column(s): " + ", ".join(missing_scored)
        )

    if {"client_id_a", "client_id_b", "is_true_duplicate"}.issubset(dup_flags_df.columns):
        truth = dup_flags_df[["client_id_a", "client_id_b", "is_true_duplicate"]].copy()
    elif {"client_id_primary", "client_id_secondary", "review_status"}.issubset(dup_flags_df.columns):
        truth = dup_flags_df.rename(
            columns={"client_id_primary": "client_id_a", "client_id_secondary": "client_id_b"}
        )[["client_id_a", "client_id_b", "review_status"]].copy()
        truth["is_true_duplicate"] = truth["review_status"].map(_norm_text).isin(
            ["confirmed_duplicate", "merged"]
        )
        truth = truth.drop(columns=["review_status"])
    else:
        raise ValueError(
            "Table 'dup_flags' must include either "
            "('client_id_a','client_id_b','is_true_duplicate') or "
            "('client_id_primary','client_id_secondary','review_status')."
        )

    pred = scored_df[["client_id_a", "client_id_b", "match_score"]].copy()
    pred["predicted_duplicate"] = pred["match_score"] >= REVIEW_THRESHOLD

    for df in (truth, pred):
        df["client_id_a"] = df["client_id_a"].astype(str)
        df["client_id_b"] = df["client_id_b"].astype(str)
        df[["client_id_a", "client_id_b"]] = df.apply(
            lambda r: pd.Series(sorted([r["client_id_a"], r["client_id_b"]])),
            axis=1,
        )

    merged = pred.merge(
        truth[["client_id_a", "client_id_b", "is_true_duplicate"]],
        on=["client_id_a", "client_id_b"],
        how="inner",
    )

    tp = int(((merged["predicted_duplicate"]) & (merged["is_true_duplicate"])).sum())
    fp = int(((merged["predicted_duplicate"]) & (~merged["is_true_duplicate"])).sum())
    fn = int(((~merged["predicted_duplicate"]) & (merged["is_true_duplicate"])).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "n_scored_pairs": int(len(scored_df)),
        "n_eval_pairs": int(len(merged)),
        "review_threshold": REVIEW_THRESHOLD,
        "auto_merge_threshold": AUTO_MERGE_THRESHOLD,
    }
