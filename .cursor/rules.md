This repository builds CareMatch Safe, a governance-aware referral matching platform for caseworkers.

Primary objective:
- Help a caseworker safely coordinate and refer a client to the right receiving organization.
- Enforce privacy, consent, and OCAP before any cross-agency data is shown or any recommendation is generated.

Product definition:
- This is not a public client portal.
- This is not a generic CRM.
- This is not a raw data browser.
- This is not a notebook-only analytics submission.
- This is a policy-enforcing coordination tool for caseworkers and coordinators.

Non-negotiable operating order:
1. Evaluate policy.
2. Build safe/redacted payload.
3. Rank eligible receiving organizations.
4. Return action label and explanation.
5. Log reasons and flags.

Track and data assumptions:
- Track 1: Inter-Org Referral & Care Coordination.
- Data may come from parquet files and/or SQLite views.
- Never hard-code dataset row counts.
- Treat record counts in docs as descriptive, not guaranteed.

Hard rules to enforce everywhere:
- If client.ocap_protected == True, do not expose cross-agency client details unless policy and consent explicitly allow it.
- If consent.status is withdrawn, no new data use is allowed after withdrawal_date.
- If consent.status is expired, or expires_at is before the reference time, treat it as not active.
- If consent is missing, treat it as a consent gap and default to restriction.
- If sharing_scope_type == "single_agency_only", the client must not appear in a multi-agency view.
- If a FOIPPA-related record has empty or missing purpose_codes, flag a governance gap.
- If any policy restriction is uncertain, default to conservative behavior: restrict, redact, or require manual review.
- Duplicate detection must never auto-merge restricted or OCAP-sensitive records.
- No UI page, API route, analytics module, or match engine may bypass the policy layer.

Required action labels:
- safe_to_refer
- refer_with_redaction
- manual_review_required
- blocked_by_policy

Required duplicate decisions:
- auto_merge_eligible
- review_required
- blocked_by_policy

Implementation requirements:
- Use Python.
- Separate data loading, policy logic, referral ranking, duplicate triage, API, and UI into different modules.
- Put core privacy/governance logic in src/, not only inside Streamlit pages.
- Prefer pure functions and small modules.
- Use dataclasses or pydantic models for structured outputs.
- Add docstrings and type hints.
- Fail loudly on missing required columns or malformed timestamps.

Loader requirements:
- Load orgs, clients, referrals, encounters, consent, dsa, duplicate_flags from the track data directory.
- Apply the documented renames used by the quickstart notebook.
- Build referrals_enriched from referrals + clients + referring org + receiving org + current consent.
- Validate required columns before downstream use.

Policy engine requirements:
- Return structured decisions, not booleans only.
- Include: allowed, view_status, reasons, flags, allowed_fields, redacted_fields.
- Expose helper functions for:
  - client-view policy
  - consent-gap detection
  - governance-flag detection
  - safe payload construction
- Default to redaction when uncertain.

Referral matching requirements:
- Never recommend a receiving organization before policy evaluation.
- Only rank organizations that the client may lawfully be considered for in the current context.
- Use explainable factors such as taxonomy fit, historical acceptance, speed, and acuity fit.
- Return an action label and explanation with every ranking result.

Duplicate requirements:
- Score candidate duplicate pairs against duplicate_flags.
- Measure precision, recall, and F1.
- Do not auto-merge restricted clients.
- Keep scoring separate from merge authorization.

API requirements:
- Prefer a safe-view endpoint such as GET /clients/{id}/safe-view.
- The API must not return raw restricted fields and expect the frontend to hide them.
- Include policy metadata in responses.

UI requirements:
- Streamlit pages must show policy warning banners clearly.
- Restricted or redacted records must explain why.
- Keep UI simple, operational, and demo-friendly.
- English locale should follow en-CA conventions.

Testing requirements:
- Add pytest coverage for schema validation, policy decisions, safe payloads, matcher behavior, duplicate routing, and API responses.
- Include synthetic fixtures for withdrawn consent, expired consent, missing consent, single_agency_only, FOIPPA missing purpose_codes, and ocap_protected.

Do not:
- Do not expose OCAP-protected data in cross-agency views by default.
- Do not silently treat missing consent as valid.
- Do not silently fill legal/governance fields in ways that change meaning.
- Do not auto-merge all likely duplicates.
- Do not put critical privacy rules only in frontend code.
- Do not claim model improvements without measured results.

Coding style:
- Concise, explicit, modular, readable.
- Prefer deterministic logic over hidden magic.
- Output code that is runnable with minimal edits.
