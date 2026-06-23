# ============================================================
# resume_after_credit_crash.py
# Resumes parsing from CREDIT_PROFILE onward — skips PROFILE
# and FINANCIALS entirely since those already succeeded and
# are already saved. Makes ZERO redundant LLM calls.
#
# Run this INSTEAD of run_parser_agent() when you crash partway
# through parse_annual_report() and don't want to re-burn keys
# on steps that already completed.
# ============================================================

import time
from parser_agents import (
    extract_text_from_pdf,
    extract_json_from_text,
    CREDIT_PROMPT,
    COLLATERAL_PROMPT,
    AUDIT_PROMPT,
    RISK_PROMPT,
    extract_text_from_xlsx,
)
from db import (
    save_credit_profile,
    save_collateral_info,
    save_audit_findings,
    save_risk_ratings,
)
from config import DOC_PATHS

COMPANY_NAME = "Abha Power and Steel Limited"  # change if needed


def resume_credit_and_collateral():
    """Re-reads the annual report PDF text (no LLM cost) and runs
    ONLY the CREDIT and COLLATERAL extraction steps. PROFILE and
    FINANCIALS are NOT touched — already saved, not re-parsed."""
    print("[Resume] Reading Annual Report text (no LLM call)...")
    text = extract_text_from_pdf(DOC_PATHS["annual_report"])

    print("\n[Resume] === Extracting: CREDIT PROFILE ===")
    credit = extract_json_from_text(text, CREDIT_PROMPT, merge_mode="list")
    if credit:
        save_credit_profile(credit)
    print(f"[Resume] Credit profile saved: {len(credit)} instrument(s).")
    time.sleep(3)

    print("\n[Resume] === Extracting: COLLATERAL ===")
    collateral = extract_json_from_text(text, COLLATERAL_PROMPT, merge_mode="list")
    if collateral:
        save_collateral_info(collateral)
    print(f"[Resume] Collateral saved: {len(collateral)} item(s).")


def resume_audit_report():
    """Only run this if parse_audit_report() never ran yet."""
    print("\n[Resume] === Reading Audit Report ===")
    text = extract_text_from_pdf(DOC_PATHS["audit_report"])
    audit = extract_json_from_text(text, AUDIT_PROMPT, merge_mode="dict")
    time.sleep(3)
    save_audit_findings(audit)
    print("[Resume] Audit findings saved.")


def resume_save_risk():
    """Only run this if parse_save_risk_file() never ran yet."""
    print("\n[Resume] === Reading SAVE Risk File ===")
    text = extract_text_from_xlsx(DOC_PATHS["save_risk"])
    risk = extract_json_from_text(text, RISK_PROMPT, merge_mode="dict")
    time.sleep(3)
    save_risk_ratings(risk)
    print("[Resume] Risk ratings saved.")


if __name__ == "__main__":
    # Step 1: finish what crashed (credit_profile + collateral)
    resume_credit_and_collateral()

    # Step 2 & 3: uncomment ONLY if these haven't run yet for this company.
    # If they already ran successfully before the crash, leave commented
    # out — running them again wastes calls for no new data.
    # resume_audit_report()
    # resume_save_risk()

    print("\n[Resume] Done. borrower_profile and financials were never touched.")
