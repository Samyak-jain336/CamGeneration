import json

from db import (
    get_borrower_profile,
    get_risk_ratings,
    get_financials,
    get_audit_findings,
    get_credit_profile,
    get_collateral_info,
    save_inconsistencies
)
from section_generation_final import call_sga_llm


SAFE_SAVE_RATINGS = ["SB-1", "SB-1+"]

BAD_AUDIT_OPINIONS = [
    "qualified",
    "adverse",
    "disclaimer",
    "disclaimer of opinion"
]

class InconsistencyAgent:

    def detect(self, company_name: str):

        borrower = get_borrower_profile(company_name) or {}
        risk = get_risk_ratings(company_name) or {}
        financials = get_financials(company_name) or []
        audit = get_audit_findings(company_name) or {}

        inconsistencies = []

        # =====================================================
        # COMPANY NAME CHECK
        # =====================================================

        borrower_name = borrower.get("company_name")
        risk_name = risk.get("company_name")

        if (
            borrower_name
            and risk_name
            and str(borrower_name).strip().lower()
            != str(risk_name).strip().lower()
        ):
            inconsistencies.append(
                {
                    "company_name": company_name,
                    "field_name": "company_name",
                    "source_1": "borrower_profile",
                    "value_1": borrower_name,
                    "source_2": "save_risk",
                    "value_2": risk_name,
                    "severity": "critical"
                }
            )

        # Flag company_name as missing only if absent from ALL
        # three sources that carry it (borrower, risk, financials)
        financials_name = financials[-1].get("company_name") if financials else None
        credit = get_credit_profile(company_name) or []
        collateral = get_collateral_info(company_name) or []
        credit_name = credit[0].get("company_name") if credit else None
        collateral_name = collateral[0].get("company_name") if collateral else None

        if (
            not borrower_name and not risk_name and not financials_name
            and not credit_name and not collateral_name
        ):
            inconsistencies.append(
                {
                    "company_name": company_name,
                    "field_name": "company_name",
                    "source_1": "borrower_profile",
                    "value_1": None,
                    "source_2": "save_risk",
                    "value_2": None,
                    "severity": "critical",
                    "issue": "missing_in_all_sources"
                }
            )

        # =====================================================
        # AUDIT VS SAVE RATING CHECK
        # =====================================================

        audit_opinion = audit.get("opinion")
        save_rating = risk.get("save_rating")

        if (
            audit_opinion
            and save_rating
            and str(audit_opinion).strip().lower() in BAD_AUDIT_OPINIONS
            and str(save_rating).strip().upper() in SAFE_SAVE_RATINGS
        ):
            inconsistencies.append(
                {
                    "company_name": company_name,
                    "field_name": "audit_vs_risk_rating",
                    "source_1": "audit_report",
                    "value_1": audit_opinion,
                    "source_2": "save_risk",
                    "value_2": save_rating,
                    "severity": "critical"
                }
            )

        # Flag audit_opinion / save_rating as missing only if
        # absent from both their respective sources entirely
        if not audit_opinion and not save_rating:
            inconsistencies.append(
                {
                    "company_name": company_name,
                    "field_name": "audit_vs_risk_rating",
                    "source_1": "audit_report",
                    "value_1": None,
                    "source_2": "save_risk",
                    "value_2": None,
                    "severity": "critical",
                    "issue": "missing_in_all_sources"
                }
            )
        
        # =====================================================
        # FISCAL YEAR ALIGNMENT CHECK
        # =====================================================

        latest_financials_fy = financials[-1].get("fiscal_year") if financials else None
        audit_year = audit.get("audit_year")
        assessment_date = risk.get("assessment_date")

        def _year_digits(value):
            if not value:
                return None
            digits = "".join(ch for ch in str(value) if ch.isdigit())
            return digits[-4:] if len(digits) >= 4 else None

        fin_year_digits = _year_digits(latest_financials_fy)
        audit_year_digits = _year_digits(audit_year)

        if fin_year_digits and audit_year_digits and fin_year_digits != audit_year_digits:
            inconsistencies.append(
                {
                    "company_name": company_name,
                    "field_name": "fiscal_year_alignment",
                    "source_1": "financials",
                    "value_1": latest_financials_fy,
                    "source_2": "audit_report",
                    "value_2": audit_year,
                    "severity": "minor"
                }
            )

        # =====================================================
        # LLM FALLBACK — only runs if the 3 rule-based checks
        # above found nothing. Costs one LLM call, only on
        # otherwise-empty results.
        # =====================================================

        if not inconsistencies:
            llm_results = self._detect_with_llm(
                company_name, borrower, risk, financials, audit, credit, collateral
            )
            inconsistencies.extend(llm_results)

        # =====================================================
        # SAVE RESULTS
        # =====================================================

        if inconsistencies:
            save_inconsistencies(inconsistencies)

        return inconsistencies

    def _detect_with_llm(
        self, company_name, borrower, risk, financials, audit, credit, collateral
    ):
        """Fallback consistency check using an LLM, only called when the
        rule-based checks above return nothing. Uses the same Groq→OpenAI
        fallback chain as the SGAs (call_sga_llm), so no new client setup
        is needed here."""

        prompt = f"""You are reviewing four independent data sources for the same
borrower, extracted from separate source documents (Annual Report, Audit
Report, SAVE Risk Assessment, Credit Profile). Identify any factual conflicts
between them — for example mismatched dates, names, financial figures, or
ratings that cannot both be true at once.

Only report a conflict if it is clearly supported by the data below. Do not
invent a conflict that isn't present. If you find none, return an empty list.

BORROWER PROFILE:
{json.dumps(borrower, default=str, indent=2)}

SAVE RISK RATING:
{json.dumps(risk, default=str, indent=2)}

FINANCIALS (all fiscal years):
{json.dumps(financials, default=str, indent=2)}

AUDIT FINDINGS:
{json.dumps(audit, default=str, indent=2)}

CREDIT PROFILE:
{json.dumps(credit, default=str, indent=2)}

COLLATERAL INFO:
{json.dumps(collateral, default=str, indent=2)}

Respond with ONLY a JSON array, no other text, no markdown fences. Each
element must have exactly this shape:
{{"field_name": "...", "source_1": "...", "value_1": "...", "source_2": "...", "value_2": "...", "severity": "critical" or "minor"}}

If there are no conflicts, respond with exactly: []
"""

        try:
            raw_response = call_sga_llm(prompt)
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:]
            parsed = json.loads(cleaned)
        except Exception as e:
            print(f"[InconsistencyAgent] LLM fallback failed or returned bad JSON: {e}")
            return []

        results = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            results.append({
                "company_name": company_name,
                "field_name": item.get("field_name", "llm_detected"),
                "source_1": item.get("source_1", "unknown"),
                "value_1": item.get("value_1"),
                "source_2": item.get("source_2", "unknown"),
                "value_2": item.get("value_2"),
                "severity": item.get("severity") if item.get("severity") in ("critical", "minor") else "minor"
            })
        return results


if __name__ == "__main__":

    import sys

    agent = InconsistencyAgent()

    company = sys.argv[1] if len(sys.argv) > 1 else "Arjun Textiles Limited"

    result = agent.detect(company)

    print("\nDetected Inconsistencies:\n")

    for item in result:
        print(item)
