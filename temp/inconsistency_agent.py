from db import (
    get_borrower_profile,
    get_risk_ratings,
    get_financials,
    get_audit_findings,
    save_inconsistencies
)


class InconsistencyAgent:

    def detect(self, company_name: str):

        borrower = get_borrower_profile(company_name)
        risk = get_risk_ratings(company_name)
        financials = get_financials(company_name)
        audit = get_audit_findings(company_name)

        inconsistencies = []

        # =====================================================
        # COMPANY NAME CHECK
        # =====================================================

        borrower_name = borrower.get("company_name")
        risk_name = risk.get("company_name")

        if (
            borrower_name
            and risk_name
            and borrower_name.strip().lower()
            != risk_name.strip().lower()
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

        # =====================================================
        # AUDIT VS SAVE RATING CHECK
        # =====================================================

        audit_opinion = audit.get("opinion")
        save_rating = risk.get("save_rating")

        bad_opinions = [
            "qualified",
            "adverse",
            "disclaimer"
        ]

        if (
            audit_opinion
            and save_rating
            and audit_opinion.lower() in bad_opinions
            and "SB-1" in str(save_rating)
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

        # =====================================================
        # SAVE RESULTS
        # =====================================================

        if inconsistencies:
            save_inconsistencies(inconsistencies)

        return inconsistencies


if __name__ == "__main__":

    agent = InconsistencyAgent()

    result = agent.detect(
        "Arjun Textiles Limited"
    )

    print("\nDetected Inconsistencies:\n")

    for item in result:
        print(item)