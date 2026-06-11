# ============================================================
# parser_agent.py
# Reads 3 input documents → extracts structured data → stores
# into MySQL via db.py
#
# Uses Claude to extract structured JSON from PDF text.
# ============================================================

import json
import time
import pdfplumber
import openpyxl
from groq import Groq
 
from config import GROQ_API_KEY, LLM_MODEL, DOC_PATHS
from db import (
    save_borrower_profile, save_financials, save_audit_findings,
    save_risk_ratings, save_banking_history, save_collateral_info, get_connection
)

client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# PDF TEXT EXTRACTOR
# ============================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def extract_text_from_xlsx(xlsx_path: str) -> str:
    """
    Reads ALL sheets from an Excel workbook.
    Returns all content as plain text — one sheet at a time.
    No PDF conversion needed.
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    all_text = ""
 
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        all_text += f"\n\n=== SHEET: {sheet_name} ===\n"
 
        for row in ws.iter_rows(values_only=True):
            # Skip completely empty rows
            if any(cell is not None for cell in row):
                row_text = " | ".join(
                    str(cell) if cell is not None else ""
                    for cell in row
                )
                all_text += row_text + "\n"
 
    return all_text.strip()


# ============================================================
# LLM EXTRACTION HELPER
# ============================================================

def extract_json_from_text(text: str, prompt: str) -> dict:
    """
    Sends PDF text + extraction prompt to Claude.
    Returns parsed JSON dict.
    """
    message = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=4096,
        messages=[
            {
                "role": "system",
                "content": "You are a financial document parser. Always respond with valid JSON only. No explanation, no markdown fences, no extra text."
            },
            {
                "role": "user",
                "content": f"{prompt}\n\nDOCUMENT TEXT:\n{text[:12000]}"
            }
        ]
    )
    raw = message.choices[0].message.content.strip()

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    return json.loads(raw)


# ============================================================
# INDIVIDUAL PARSERS — one per document
# ============================================================

def parse_annual_report(company_name: str):
    """
    Reads annual_report.pdf → extracts:
    - borrower_profile
    - financials (3 years)
    - banking_history (borrowings table)
    - collateral_info
    """
    print("[Parser] Reading Annual Report...")
    text = extract_text_from_pdf(DOC_PATHS["annual_report"])

    # -- Borrower Profile ------------------------------------
    profile_prompt = """
    Extract the following from this Annual Report as a JSON object.
    Return ONLY valid JSON, no explanation, no markdown fences.
    {
      "company_name": "...",
      "cin": "...",
      "constitution": "...",
      "incorporation_year": 2004,
      "registered_office": "...",
      "business_nature": "...",
      "promoters": [{"name": "...", "designation": "..."}],
      "key_management": [{"name": "...", "designation": "...", "qualification": "...", "experience": "..."}],
      "shareholding": {"promoter_pct": 62.50, "public_inst_pct": 14.00, "public_non_inst_pct": 23.50}
    }
    """
    profile = extract_json_from_text(text, profile_prompt)
    time.sleep(3)
    save_borrower_profile(profile)
    print("[Parser] Borrower profile saved.")

    # -- Financials ------------------------------------------
    financials_prompt = """
    Extract 3 years of financials from this Annual Report as a JSON array.
    Return ONLY valid JSON, no explanation, no markdown fences.
    Each element should be:
    {
      "company_name": "...",
      "fiscal_year": "FY24",
      "revenue": 18742.36,
      "other_income": 214.80,
      "total_revenue": 18957.16,
      "cost_of_materials": 11628.46,
      "employee_expense": 1893.20,
      "finance_costs": 824.36,
      "depreciation": 612.44,
      "other_expenses": 1974.18,
      "pbt": 2024.52,
      "tax": 516.24,
      "pat": 1508.28,
      "ebitda": 3461.32,
      "ebitda_margin": 18.47,
      "pat_margin": 8.05,
      "current_ratio": 1.42,
      "quick_ratio": 0.84,
      "debt_equity_ratio": 0.86,
      "icr": 3.46,
      "dscr": 1.84,
      "roe": 22.4,
      "roce": 19.8,
      "asset_turnover": 1.10,
      "inventory_days": 71,
      "debtor_days": 55,
      "working_capital_days": 94,
      "total_assets": 17013.20,
      "total_equity": 7442.36,
      "total_borrowings": 5770.80,
      "cash_from_operations": 2678.32,
      "cash_from_investing": -665.80,
      "cash_from_financing": -1595.16,
      "closing_cash": 701.96
    }
    Return as a JSON array with 3 elements (FY22, FY23, FY24).
    """
    financials = extract_json_from_text(text, financials_prompt)
    time.sleep(3)
    if isinstance(financials, list):
        save_financials(financials)
    else:
        save_financials([financials])
    print("[Parser] Financials saved.")

    # -- Banking History (Borrowings table) ------------------
    banking_prompt = """
    Extract the borrowings/loan schedule table from this Annual Report as a JSON array.
    Return ONLY valid JSON, no explanation, no markdown fences.
    Each element:
    {
      "company_name": "...",
      "lender": "State Bank of India",
      "facility_type": "Term Loan",
      "purpose": "Machinery Upgrade",
      "interest_rate": 9.25,
      "outstanding_amt": 1842.40,
      "fiscal_year": "FY24"
    }
    """
    banking = extract_json_from_text(text, banking_prompt)
    time.sleep(3)
    if isinstance(banking, list):
        save_banking_history(banking)
    print("[Parser] Banking history saved.")

    # -- Collateral ------------------------------------------
    collateral_prompt = """
    Extract collateral and security information from this Annual Report as a JSON array.
    Return ONLY valid JSON, no explanation, no markdown fences.
    Each element:
    {
      "company_name": "...",
      "security_type": "Primary / Collateral",
      "description": "First charge on current assets",
      "value_lakhs": 6462.00,
      "remarks": "Stock + Debtors"
    }
    """
    collateral = extract_json_from_text(text, collateral_prompt)
    time.sleep(3)
    if isinstance(collateral, list):
        save_collateral_info(collateral)
    print("[Parser] Collateral info saved.")


def parse_audit_report(company_name: str):
    """
    Reads audit_report.pdf → extracts:
    - audit_findings (KAMs, CARO, observations, summary)
    """
    print("[Parser] Reading Audit Report...")
    text = extract_text_from_pdf(DOC_PATHS["audit_report"])

    audit_prompt = """
    Extract audit findings from this Audit Report as a JSON object.
    Return ONLY valid JSON, no explanation, no markdown fences.
    {
      "company_name": "...",
      "audit_year": "FY24",
      "auditor_name": "M/s. Kapoor & Associates",
      "firm_reg_no": "001284C",
      "opinion": "Clean / Qualified / Adverse",
      "key_audit_matters": [
        {"matter": "Revenue Recognition", "description": "...", "how_addressed": "..."}
      ],
      "caro_findings": [
        {"para": "3(i)(a)", "matter": "PPE", "finding": "..."}
      ],
      "observations": [
        {"obs_no": "O-01", "observation": "...", "risk_level": "Medium", "management_response": "..."}
      ],
      "summary_assessment": {
        "financial_integrity": "Clean",
        "internal_controls": "Adequate",
        "going_concern": "No Issues",
        "fraud": "None Detected"
      },
      "contingent_liabilities": [
        {"nature": "Income Tax Dispute", "amount_lakhs": 84.20, "remarks": "..."}
      ]
    }
    """
    audit = extract_json_from_text(text, audit_prompt)
    time.sleep(3)
    save_audit_findings(audit)
    print("[Parser] Audit findings saved.")


def parse_save_risk_file(company_name: str):
    """
    Reads save_risk_file.pdf → extracts:
    - risk_ratings (SAVE score, flags, triggers)
    Note: If you have the xlsx, convert to PDF first,
    or swap pdfplumber for openpyxl here.
    """
    print("[Parser] Reading SAVE Risk File...")
    text = extract_text_from_xlsx(DOC_PATHS["save_risk"])

    risk_prompt = """
    Extract SAVE risk rating data from this document as a JSON object.
    Return ONLY valid JSON, no explanation, no markdown fences.
    {
      "company_name": "...",
      "assessment_date": "01-Jun-2024",
      "save_score": 75.7,
      "save_rating": "SB-3",
      "previous_rating": "SB-4",
      "rating_movement": "Upgrade",
      "solvency_score": 23.4,
      "asset_score": 19.5,
      "viability_score": 18.0,
      "external_score": 14.8,
      "risk_flags": [
        {"flag_id": "F-01", "description": "...", "category": "Financial", "status": "Monitored"}
      ],
      "watch_list_triggers": [
        {"trigger_id": "T-01", "condition": "...", "consequence": "..."}
      ]
    }
    """
    risk = extract_json_from_text(text, risk_prompt)
    time.sleep(3)
    save_risk_ratings(risk)
    print("[Parser] Risk ratings saved.")


# ============================================================
# MAIN PARSER AGENT FUNCTION
# Called by orchestrator in cam_langgraph.py
# ============================================================

def run_parser_agent(company_name: str):
    """
    Runs all 3 parsers in sequence.
    Returns True on success, raises Exception on failure.
    """
    print(f"\n[Parser Agent] Starting ingestion for: {company_name}")
    conn = get_connection()
    cursor = conn.cursor()
    for table in ["borrower_profile", "financials", "audit_findings",
                  "risk_ratings", "banking_history", "collateral_info"]:
        cursor.execute(f"TRUNCATE TABLE {table}")
    conn.commit()
    cursor.close()
    conn.close()
    print("[Parser] All tables cleaned.")
    parse_annual_report(company_name)
    parse_audit_report(company_name)
    parse_save_risk_file(company_name)
    print(f"[Parser Agent] All documents parsed and stored in MySQL.\n")
    return True

if __name__ == "__main__":
    run_parser_agent("Arjun Textiles Limited")