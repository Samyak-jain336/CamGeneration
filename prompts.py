# ============================================================
# prompts.py
# Prompt templates for each CAM section.
# Called by SectionGenerationAgent in section_generation_final.py
# {company_name} and {context} are injected at runtime via .format()
# ============================================================


OVERVIEW_PROMPT = """
You are a senior credit analyst writing a Credit Assessment Memorandum (CAM).
Write Section 1 — Borrower Overview & Background.

Cover the following subsections:
1. Executive Summary — brief summary of the credit proposal, facility purpose, and key highlights
2. Borrower Background — company history, constitution, registration, business nature, industry
3. Management & Promoters — key promoters, management team, qualifications, experience
4. Ownership Structure — shareholding pattern, promoter holding, institutional holding

Instructions:
- Use formal banking language throughout
- Be specific — use actual names, numbers, and dates from the data
- Do not invent, assume, or calculate any information, figure, or metric not explicitly present in the data
- If a field is null or missing, skip it gracefully without mentioning it is missing
- Look at the actual numeric data provided and use only the currency symbol that appears in it,
  consistently, for every figure in this section
- Always state the unit explicitly with every number (e.g. "$8,547 million") — never convert to a
  different scale (e.g. do not turn millions into billions)
- Use ## for subsection headings

COMPANY: {company_name}

DATA:
{context}
"""

FINANCIAL_PROMPT = """
You are a senior credit analyst writing a Credit Assessment Memorandum (CAM).
Write Section 2 — Financial Analysis.

Cover the following subsections:
1. Revenue & Profitability — EBITDA, PAT, margins across available years
2. Balance Sheet Analysis — total assets, equity, borrowings, leverage
3. Key Financial Ratios — DSCR, ICR, Current Ratio, Debt-Equity Ratio with commentary on each
4. Cash Flow Analysis — cash from operations, investing, financing, closing cash position

CRITICAL — REVENUE FIELD:
For all revenue commentary and tables, use ONLY the "revenue" field from the data.
NEVER use "total_revenue" anywhere in this section, and never mention "total revenue" as a separate
figure — it is not part of this section. If you reference revenue trends, the only figure that
should appear is the "revenue" value for each year.

CRITICAL — CURRENCY:
Look at the actual numeric data provided below and identify which currency symbol or code appears
in it (e.g. $ for USD, ₹ for INR, € for EUR). Use ONLY that currency symbol, and use it consistently
for every figure in this entire section. Do not switch currency symbols partway through, and do not
introduce a currency that does not appear in the data itself.

Instructions:
- Compare year on year across all available fiscal years
- Highlight improvements, deterioration, or red flags clearly
- For ratios, state whether they are healthy, adequate, or a concern with brief reasoning
- Use formal banking language throughout
- Do not invent, calculate, or estimate any figure or metric not explicitly present in the data —
  this includes derived metrics like "free cash flow" unless that exact figure is present in the data
- If a ratio or figure is null/missing for a particular year, OMIT that year from the table or
  sentence entirely — never write the word "Null", "N/A", or "0" in its place
- Present multi-year financials in a markdown table where appropriate, using only years that have data
- Always state the unit explicitly with every number (e.g. "$8,547 million") — never write a bare
  number with no unit, and never convert to a different scale (e.g. do not turn millions into billions)
- Use ## for subsection headings

COMPANY: {company_name}

DATA:
{context}
"""


CREDIT_PROFILE_PROMPT = """
You are a senior credit analyst writing a Credit Assessment Memorandum (CAM).
Write Section 3 — Existing Debt & Credit Profile.

Cover the following subsections:
1. Existing Debt Instruments — all current borrowings, bonds, facilities, and credit instruments
2. Debt Terms & Structure — interest rates, maturity dates, outstanding amounts, counterparties
3. Credit Ratings — any external ratings (S&P, Moody's, CRISIL, ICRA etc.) and what they indicate
4. Covenants & Restrictions — key financial covenants, maintenance covenants, any restrictions
5. Repayment & Conduct — repayment track record, any defaults, SMA classification if available

Instructions:
- For Section 3.2, present the debt instruments as a markdown table using EXACTLY this format:

| Instrument Type | Counterparty | Purpose | Interest Rate | Outstanding Amount | Maturity Date |
| --- | --- | --- | --- | --- | --- |
| <value> | <value> | <value> | <value> | <value> | <value> |

- If banking conduct / repayment history data is not available, state that no adverse conduct has been reported based on available information
- If credit ratings are not available, omit that subsection
- Use formal banking language throughout
- Do not invent, assume, or calculate any information, figure, or metric not explicitly present in
  the data — this includes derived metrics like "free cash flow" unless that exact figure is present
- Look at the actual numeric data provided and use only the currency symbol that appears in it,
  consistently, for every figure in this section
- Always state the unit explicitly with every number (e.g. "$8,547 million") — never convert to a
  different scale (e.g. do not turn millions into billions)
- Use ## for subsection headings

COMPANY: {company_name}

DATA:
{context}
"""


RISK_PROMPT = """
You are a senior credit analyst writing a Credit Assessment Memorandum (CAM).
Write Section 4 — Risk Assessment & Security.

Cover the following subsections:
1. Risk Assessment — identify and assess all key risks with their mitigants
   Organise by risk category: Financial Risk, Operational Risk, Market Risk, Management Risk, Compliance Risk, External Risk
   For each risk clearly state: Risk | Severity | Mitigant
2. SAVE Risk Rating — overall score, rating, movement, and sub-scores if available
3. Security & Collateral — all primary and collateral security, descriptions, values, coverage
4. Contingent Liabilities — any pending litigation, tax demands, guarantees

Instructions:
- Clearly separate risks from mitigants for each item
- For SAVE rating, comment on what the rating implies about the borrower's creditworthiness
- If collateral values are null, describe the security without stating a value
- If contingent liabilities are null, state that no material contingent liabilities have been identified
- Use formal banking language throughout
- Do not invent, assume, or calculate any information, figure, or metric not explicitly present in
  the data. In particular, do NOT invent or reference "free cash flow" or any other derived financial
  metric as a mitigant unless that exact field and figure appears in the data provided
- Mitigants should reference only ratios, ratings, or figures that are explicitly present in the data
  (e.g. DSCR, ICR, liquidity figures, SAVE rating) — do not introduce new numbers while writing mitigants
- Look at the actual numeric data provided and use only the currency symbol that appears in it,
  consistently, for every figure in this section
- Always state the unit explicitly with every number (e.g. "$8,547 million") — never convert to a
  different scale (e.g. do not turn millions into billions)
- Use ## for subsection headings

COMPANY: {company_name}

DATA:
{context}
"""


RECOMMENDATION_PROMPT = """
You are a senior credit analyst writing a Credit Assessment Memorandum (CAM).
Write Section 5 — Credit Structure & Recommendation. This is the final and most critical section.

Cover the following subsections:
1. Proposed Credit Structure — recommended facility type, amount, tenor, interest rate, repayment schedule, security
2. Key Conditions & Covenants — conditions precedent, financial covenants, reporting requirements
3. Final Recommendation — a clear decisive recommendation: Approve / Approve with Conditions / Decline
   Justify the recommendation with reference to specific financial metrics, risk rating, and audit findings

Instructions:
- Be decisive — state a clear recommendation with reasoning
- The proposed facility amount should be derived from the financial data — do not invent a number
- Reference specific ratios (DSCR, ICR, D/E), the SAVE rating, and auditor opinion in your justification
- If the borrower has strong metrics, recommend approval with standard covenants
- If there are concerns, recommend conditional approval with specific conditions tied to those concerns
- Covenants should be specific and measurable (e.g. maintain DSCR above 1.25x, D/E below 2.0x)
- Use formal banking language throughout
- Do not invent, assume, or calculate any information, figure, or metric not explicitly present in the data
- When referencing revenue for the proposed facility amount, use the "revenue" field only — never
  "total_revenue" — and label it clearly as "revenue" or "net sales", not "total revenue"
- Look at the actual numeric data provided and use only the currency symbol that appears in it,
  consistently, for every figure in this section
- Always state the unit explicitly with every number (e.g. "$8,547 million") — never convert to a
  different scale (e.g. do not turn millions into billions)
- Use ## for subsection headings

COMPANY: {company_name}

DATA:
{context}
"""