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

#from openai import OpenAI
from groq import Groq
from openai import OpenAI
import requests


from config import (
    GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, GROQ_API_KEY_4, GROQ_API_KEY_5, GROQ_API_KEY_6, GROQ_MODEL,
    OLLAMA_MODEL, OLLAMA_URL,
    OPENAI_API_KEY, OPENAI_MODEL,
    DOC_PATHS
)


from db import (
    save_borrower_profile, save_financials, save_audit_findings,
    save_risk_ratings, save_collateral_info, get_connection, save_credit_profile
)


groq_client_1 = Groq(api_key=GROQ_API_KEY)
groq_client_2 = Groq(api_key=GROQ_API_KEY_2)
groq_client_3 = Groq(api_key=GROQ_API_KEY_3)
groq_client_4 = Groq(api_key=GROQ_API_KEY_4)
groq_client_5 = Groq(api_key=GROQ_API_KEY_5)
groq_client_6 = Groq(api_key=GROQ_API_KEY_6)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = "You are a financial document parser. Always respond with valid JSON only. No explanation, no markdown fences, no extra text. If a field is not found in this chunk, return null for that field — never guess or fabricate values."


def fix_math_expressions_in_json(raw: str) -> str:
    """
    Safety net for when the LLM writes arithmetic expressions instead of
    final numbers as JSON values, e.g. "total_revenue": 8547 + 62
    or "ebitda_margin": (1977 + 412) / 8547

    Finds patterns like   : <number/expression with + - * / and parens>,
    and evaluates them down to a plain number before json.loads() runs.
    Only matches simple numeric expressions — never touches strings,
    since those are already inside quotes and untouched by this regex.
    """
    import re

    # Matches:  "key": <expr ending in , or } or newline>
    # where <expr> contains digits, ., +, -, *, /, (, ), spaces — but
    # is NOT already a clean plain number or a quoted string.
    pattern = re.compile(
        r'(:\s*)([\(\)\d\.\s\+\-\*\/]+)([,\}\n])'
    )

    def try_eval(match):
        prefix, expr, suffix = match.groups()
        stripped = expr.strip()

        # If it's already a plain number, leave it alone
        if re.fullmatch(r'-?\d+(\.\d+)?', stripped):
            return match.group(0)

        # Must contain at least one operator to be worth evaluating
        if not re.search(r'[\+\-\*\/]', stripped):
            return match.group(0)

        try:
            value = eval(stripped, {"__builtins__": {}}, {})
            if isinstance(value, (int, float)):
                return f"{prefix}{value}{suffix}"
        except Exception:
            pass

        return match.group(0)

    return pattern.sub(try_eval, raw)

def call_groq_1(prompt: str) -> str:
    response = groq_client_1.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ]
    )
    return response.choices[0].message.content


def call_groq_2(prompt: str) -> str:
    response = groq_client_2.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ]
    )
    return response.choices[0].message.content


def call_groq_3(prompt: str) -> str:
    response = groq_client_3.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ]
    )
    return response.choices[0].message.content


def call_groq_4(prompt: str) -> str:
    response = groq_client_4.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ]
    )
    return response.choices[0].message.content


def call_groq_5(prompt: str) -> str:
    response = groq_client_5.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ]
    )
    return response.choices[0].message.content


def call_groq_6(prompt: str) -> str:
    response = groq_client_6.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ]
    )
    return response.choices[0].message.content

def call_openai(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ]
    )
    return response.choices[0].message.content


def call_ollama(prompt: str) -> str:
    """
    Calls local Ollama instance.
    Make sure Ollama is running: ollama serve
    And model is pulled: ollama pull llama3.1:8b
    """
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    response = requests.post(
        OLLAMA_URL,
        json={
            "model":  OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 4096
            }
        },
        timeout=300   # Ollama can be slow on large chunks — 5 min timeout
    )
    response.raise_for_status()
    return response.json()["response"]


def call_parser_llm(prompt: str) -> str:
    """
    Tries providers in order:
    Groq 1 → Groq 2 → Groq 3 → Groq 4 → Groq 5 → Groq 6 → Ollama (local fallback)
    Falls back on rate limit or error.
    """
    providers = [
        {"name": "Groq Account 1", "fn": call_groq_1},
        {"name": "Groq Account 2", "fn": call_groq_2},
        {"name": "Groq Account 3", "fn": call_groq_3},
        {"name": "Groq Account 4", "fn": call_groq_4},
        {"name": "Groq Account 5", "fn": call_groq_5},
        {"name": "Groq Account 6", "fn": call_groq_6},
        {"name": "OpenAI",         "fn": call_openai},
        {"name": "Ollama (local)", "fn": call_ollama},
    ]

    last_error = None

    for provider in providers:
        try:
            print(f"[Parser] Trying {provider['name']}...")
            result = provider["fn"](prompt)
            print(f"[Parser] {provider['name']} succeeded.")
            return result

        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in
                   ["rate limit", "rate_limit", "429", "quota",
                    "too many requests", "resource_exhausted", "exceeded"]):
                print(f"[Parser] {provider['name']} rate limit — trying next...")
                last_error = e
                time.sleep(5)
                continue
            else:
                print(f"[Parser] {provider['name']} error: {str(e)} — trying next...")
                last_error = e
                continue

    raise Exception(f"All parser providers failed. Last error: {last_error}")

# ============================================================
# PDF TEXT EXTRACTOR
# ============================================================

def detect_page_layout(page) -> str:
    """
    Detects the layout of a PDF page by analysing where text blocks sit.
    Returns:
      'single'   — one column, normal full page
      'two_col'  — two side-by-side pages/columns
      'blank'    — no meaningful text found
    """
    width  = page.width
    height = page.height

    # Extract word-level bounding boxes
    words = page.extract_words()

    if not words:
        return "blank"

    # Find the x-midpoint of each word
    left_words  = [w for w in words if float(w["x0"]) < width / 2]
    right_words = [w for w in words if float(w["x0"]) >= width / 2]

    left_count  = len(left_words)
    right_count = len(right_words)
    total_count = len(words)

    if total_count == 0:
        return "blank"

    left_ratio  = left_count  / total_count
    right_ratio = right_count / total_count

    # If both sides have at least 25% of total words → two column layout
    if left_ratio >= 0.25 and right_ratio >= 0.25:
        # Check for a natural gap in the middle (column gutter)
        all_x = sorted([float(w["x1"]) for w in left_words] +
                       [float(w["x0"]) for w in right_words])
        mid_gap = width / 2
        gap_words = [w for w in words
                     if float(w["x1"]) > mid_gap * 0.85
                     and float(w["x0"]) < mid_gap * 1.15]

        # If very few words cross the midpoint → clean column separation
        if len(gap_words) / total_count < 0.05:
            return "two_col"

    return "single"


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file using pdfplumber.
    Auto-detects layout per page:
      - Single page layout  → extracts as one block
      - Two-column layout   → splits into left and right halves separately
      - Blank page          → skipped
    Labels each physical page clearly so the LLM never mixes data across pages.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            width  = page.width
            height = page.height
            layout = detect_page_layout(page)

            if layout == "blank":
                # Skip blank pages entirely
                continue

            elif layout == "two_col":
                # Two pages side by side — extract each half independently
                left_bbox  = (0,         0, width / 2, height)
                right_bbox = (width / 2, 0, width,     height)

                left_text  = page.crop(left_bbox).extract_text()
                right_text = page.crop(right_bbox).extract_text()

                if left_text and left_text.strip():
                    text += f"\n\n--- PAGE {page_num + 1}A ---\n"
                    text += left_text.strip()

                if right_text and right_text.strip():
                    text += f"\n\n--- PAGE {page_num + 1}B ---\n"
                    text += right_text.strip()

            else:
                # Normal single page
                full_text = page.extract_text()
                if full_text and full_text.strip():
                    text += f"\n\n--- PAGE {page_num + 1} ---\n"
                    text += full_text.strip()

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

def extract_json_from_text(text: str, prompt: str, merge_mode: str = "dict") -> dict | list:
    """
    Splits full document text into overlapping chunks of 12000 chars.
    Sends each chunk to LLM separately, then merges results.
    merge_mode = "dict"  → merges all chunk results into one dict (for single-object extractions)
    merge_mode = "list"  → concatenates all chunk results into one list (for array extractions)
    """
    CHUNK_SIZE = 12000
    OVERLAP    = 500

    # Build chunks with overlap so nothing is missed at boundaries
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - OVERLAP

    print(f"[Parser] Document split into {len(chunks)} chunk(s).")

    merged_dict = {}
    merged_list = []
    field_votes = {}

    for i, chunk in enumerate(chunks):
        print(f"[Parser] Processing chunk {i+1}/{len(chunks)}...")
        try:
            full_prompt = f"{prompt}\n\nDOCUMENT CHUNK {i+1} OF {len(chunks)}:\n{chunk}"
            raw = call_parser_llm(full_prompt).strip()
            print(f"\n{'='*60}\nRAW LLM OUTPUT — Chunk {i+1}\n{'='*60}")
            print(raw)
            print('='*60)

            # Strip markdown fences if present
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            raw = fix_math_expressions_in_json(raw)

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                # Try adding missing opening braces for array elements
                import re
                raw = re.sub(r',\s*\n(\s*"[a-z_]+")', r',\n    {\1', raw)
                parsed = json.loads(raw)

            if merge_mode == "list":
                if isinstance(parsed, list):
                    merged_list.extend(parsed)
            else:
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if v is None or v == "" or v == [] or v == {}:
                            continue  # don't let an empty answer compete
                        field_votes.setdefault(k, []).append(v)

            time.sleep(3)

        except Exception as e:
            print(f"[Parser] Chunk {i+1} failed: {e} — skipping.")
            time.sleep(3)
            continue

    if merge_mode == "dict":
        merged_dict = resolve_field_votes(field_votes)

    return merged_list if merge_mode == "list" else merged_dict


def resolve_field_votes(field_votes: dict) -> dict:
    """
    Resolves competing chunk answers for the same field.
    Rule: most common non-empty answer wins. On a tie, the LAST
    chunk's answer wins (later chunks often see summary/total tables
    that earlier chunks haven't reached yet).
    """
    resolved = {}
    for field, votes in field_votes.items():
        if not votes:
            continue

        # Lists/dicts aren't hashable for counting — just take the
        # most recently seen one (can't meaningfully "vote" on these)
        if isinstance(votes[0], (list, dict)):
            resolved[field] = votes[-1]
            continue

        counts = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        max_count = max(counts.values())
        tied = [v for v in votes if counts[v] == max_count]
        resolved[field] = tied[-1]  # last among the tied/leading answers

    return resolved


def extract_combined_json_from_text(text: str, combined_prompt: str) -> dict:
    """
    Like extract_json_from_text, but sends ONE combined prompt per chunk
    instead of one prompt per field-group. Cuts LLM calls by 4x for the
    annual report parse (1 call per chunk instead of 4).

    Expects the LLM to return a single JSON object shaped like:
    {
      "profile": {...} or null,
      "financials": [...] or [],
      "credit": [...] or [],
      "collateral": [...] or []
    }

    Returns a dict with keys: profile (dict), financials (list),
    credit (list), collateral (list) — merged across all chunks.
    """
    CHUNK_SIZE = 12000
    OVERLAP    = 500

    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - OVERLAP

    print(f"[Parser] Document split into {len(chunks)} chunk(s). (combined prompt mode)")

    merged_profile     = {}
    profile_votes      = {}
    merged_financials  = []
    merged_credit      = []
    merged_collateral  = []

    for i, chunk in enumerate(chunks):
        print(f"[Parser] Processing chunk {i+1}/{len(chunks)} (combined)...")
        try:
            full_prompt = f"{combined_prompt}\n\nDOCUMENT CHUNK {i+1} OF {len(chunks)}:\n{chunk}"
            raw = call_parser_llm(full_prompt).strip()
            print(f"\n{'='*60}\nRAW LLM OUTPUT — Chunk {i+1}\n{'='*60}")
            print(raw)
            print('='*60)

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            raw = fix_math_expressions_in_json(raw)

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                import re
                raw = re.sub(r',\s*\n(\s*"[a-z_]+")', r',\n    {\1', raw)
                parsed = json.loads(raw)

            if not isinstance(parsed, dict):
                print(f"[Parser] Chunk {i+1} did not return a dict — skipping.")
                continue

            # -- profile: collect votes, resolve by majority after all chunks
            chunk_profile = parsed.get("profile")
            if isinstance(chunk_profile, dict):
                for k, v in chunk_profile.items():
                    if v is None or v == "" or v == [] or v == {}:
                        continue
                    profile_votes.setdefault(k, []).append(v)

            # -- financials / credit / collateral: concatenate lists
            chunk_financials = parsed.get("financials")
            if isinstance(chunk_financials, list):
                merged_financials.extend(chunk_financials)

            chunk_credit = parsed.get("credit")
            if isinstance(chunk_credit, list):
                merged_credit.extend(chunk_credit)

            chunk_collateral = parsed.get("collateral")
            if isinstance(chunk_collateral, list):
                merged_collateral.extend(chunk_collateral)

            time.sleep(3)

        except Exception as e:
            print(f"[Parser] Chunk {i+1} failed: {e} — skipping.")
            time.sleep(3)
            continue

    merged_profile = resolve_field_votes(profile_votes)

    return {
        "profile": merged_profile,
        "financials": merged_financials,
        "credit": merged_credit,
        "collateral": merged_collateral,
    }



def parse_annual_report(company_name: str):
    """
    Reads annual_report.pdf → extracts:
    - borrower_profile
    - financials (3 years)
    - banking_history (borrowings table)
    - collateral_info

    Uses ONE combined prompt per chunk (instead of 4 separate prompts)
    so each chunk only gets sent to the LLM once. Cuts LLM calls 4x.
    """
    print("[Parser] Reading Annual Report...")
    text = extract_text_from_pdf(DOC_PATHS["annual_report"])

    combined_prompt = """
Extract FOUR separate categories of information from this financial document chunk and return as ONE JSON object with exactly these four top-level keys: "profile", "financials", "credit", "collateral".

Return ONLY valid JSON, no explanation, no markdown fences.
If a field isn't present in this chunk, return null for that field — never guess or fabricate values.
CRITICAL: Every numeric value must be a plain final number (e.g. 8609, 41.5, -867). NEVER write arithmetic expressions as values (e.g. do NOT write "8547 + 62" or "(1977 + 412) / 8547" — calculate the result yourself and write only the final number).
The document may be an Indian Annual Report, a US SEC 10-K, or any other corporate filing.
Extract by meaning, not by label — labels vary across document types and jurisdictions.

{
  "profile": {
    "company_name": "extract company name",
    "cin": "extract registration or identification number (CIN, EIN, etc.) or null",
    "constitution": "extract entity type eg Private Limited, Public Limited, Corporation, LLC, or null",
    "incorporation_year": "extract as integer or null",
    "registered_office": "extract registered or principal office address or null",
    "business_nature": "extract what the company does — products, services, industry, or null",
    "promoters": [{"name": "extract name", "designation": "extract role"}],
    "key_management": [{"name": "extract name", "designation": "extract role", "qualification": "extract if available else null", "experience": "extract if available else null"}],
    "shareholding": {"promoter_pct": "extract promoter or founder ownership % as number or null", "public_inst_pct": "extract institutional ownership % as number or null", "public_non_inst_pct": "extract retail or non-institutional ownership % as number or null"}
  },
  "financials": [
    {
      "company_name": "extract company name",
      "fiscal_year": "extract fiscal year label eg FY24 or 2024",
      "revenue": "extract as number in document units",
      "other_income": "extract as number or null",
      "total_revenue": "extract as number or null",
      "cost_of_materials": "extract cost of goods sold or materials consumed as number or null",
      "employee_expense": "extract employee or staff costs as number or null",
      "finance_costs": "extract interest expense as number or null",
      "depreciation": "extract depreciation and amortisation as number or null",
      "other_expenses": "extract other operating expenses as number or null",
      "pbt": "extract profit or income before tax as number or null",
      "tax": "extract tax expense as number or null",
      "pat": "extract net income or profit after tax as number or null",
      "ebitda": "extract or calculate as number or null",
      "ebitda_margin": "extract or calculate as percentage or null",
      "pat_margin": "extract or calculate as percentage or null",
      "current_ratio": "extract or calculate as number or null",
      "quick_ratio": "extract or calculate as number or null",
      "debt_equity_ratio": "extract or calculate as number or null",
      "icr": "extract or calculate as number or null",
      "dscr": "extract or calculate as number or null",
      "roe": "extract return on equity as percentage or null",
      "roce": "extract return on capital employed as percentage or null",
      "asset_turnover": "extract or calculate as number or null",
      "inventory_days": "extract or calculate as integer or null",
      "debtor_days": "extract or calculate as integer or null",
      "working_capital_days": "extract or calculate as integer or null",
      "total_assets": "extract as number or null",
      "total_equity": "extract total shareholders equity as number or null",
      "total_borrowings": "extract total debt including short and long term as number or null",
      "cash_from_operations": "extract as number or null",
      "cash_from_investing": "extract as number or null",
      "cash_from_financing": "extract as number or null",
      "closing_cash": "extract closing cash and cash equivalents as number or null"
    }
  ],
  "credit": [
    {
      "company_name": "extract company name",
      "instrument_type": "extract type of debt instrument (Term Loan, Bond, Debenture, Revolving Credit Facility, etc.)",
      "counterparty": "extract lender, bank, or bondholder name if available else null",
      "purpose": "extract stated purpose of the borrowing if available else null",
      "interest_rate": "extract as number or null",
      "outstanding_amt": "extract outstanding balance as number or null",
      "maturity_date": "extract maturity or repayment date as string or null",
      "credit_rating": "extract external credit rating if available else null",
      "covenants": "extract key covenant terms as text if available else null",
      "fiscal_year": "extract fiscal year this data relates to"
    }
  ],
  "collateral": [
    {
      "company_name": "extract company name",
      "security_type": "extract type of security or collateral (Mortgage, Hypothecation, Lien, Pledge, etc.)",
      "description": "extract description of the asset or security",
      "value_lakhs": "extract value as number in document units or null",
      "remarks": "extract any more notes or terms or null"
    }
  ]
}

If this chunk contains no relevant data for a category, return an empty value for that key
("profile": {}, or "financials": [], "credit": [], "collateral": [] as appropriate) —
do not omit the key itself. Extract everything available for all four categories from this chunk.
"""

    result = extract_combined_json_from_text(text, combined_prompt)

    profile = result["profile"]
    save_borrower_profile(profile)
    print("[Parser] Borrower profile saved.")

    financials = result["financials"]
    if isinstance(financials, list) and financials:
        save_financials(financials)
    print("[Parser] Financials saved.")

    credit = result["credit"]
    if isinstance(credit, list) and credit:
        save_credit_profile(credit)
    print("[Parser] Credit profile saved.")

    collateral = result["collateral"]
    if isinstance(collateral, list) and collateral:
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
        "company_name": "extract from document",
        "audit_year": "extract fiscal year",
        "auditor_name": "extract auditor firm name",
        "firm_reg_no": "extract firm registration number",
        "opinion": "extract opinion eg Clean or Qualified or Adverse",
        "key_audit_matters": [
          {"matter": "extract matter name", "description": "extract description", "how_addressed": "extract how addressed"}
        ],
        "caro_findings": [
          {"para": "extract para number", "matter": "extract matter", "finding": "extract finding"}
        ],
        "observations": [
          {"obs_no": "extract observation number", "observation": "extract observation text", "risk_level": "extract risk level", "management_response": "extract response"}
        ],
        "summary_assessment": {
          "financial_integrity": "extract assessment",
          "internal_controls": "extract assessment",
          "going_concern": "extract assessment",
          "fraud": "extract assessment"
        },
        "contingent_liabilities": [
          {"nature": "extract nature", "amount_lakhs": extract_number, "remarks": "extract remarks"}
        ]
      }
      Extract all findings, observations and contingent liabilities from the document.
    """
    audit = extract_json_from_text(text, audit_prompt, merge_mode="dict")
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
If a field isn't found in this chunk, return null for that field — never guess or fabricate a rating label or trigger that isn't explicitly present.

The document contains a composite weighted risk scorecard with several named risk dimensions
(e.g. "Financial Strength", "Profitability & Growth", "Cash Flow Adequacy", "Asset Quality",
"Industry / Market Risk", "Management Quality", "Compliance & Legal", "ESG / Sustainability" —
exact names and number of dimensions vary by document, extract whatever dimensions actually appear).
Each dimension typically has a weight (%), a raw score (often out of 10), and a weighted score.

The document also contains an overall composite score and a rating SCALE that maps score ranges
to rating labels (e.g. "8.0–10.0 = LOW RISK"). Find which range the composite score falls into
and use THAT exact label as save_rating — do not invent a different label format (e.g. do not
write "SB-1" or similar unless that exact label literally appears in the rating scale table).

A "watch list trigger" or flagged item is something the document itself explicitly marks as a
concern, breach, or watch status (e.g. a ratio that fails its own stated benchmark, or a row
explicitly labeled "WATCH" or similar). Do NOT infer or fabricate a trigger from a metric that
the document does not itself flag as a concern.

{
  "company_name": "extract from document",
  "assessment_date": "extract assessment date",
  "composite_score": "extract overall composite/weighted score as number or null",
  "save_rating": "extract the EXACT rating label from the document's own rating scale that matches the composite score, or null",
  "previous_rating": "extract previous rating if explicitly stated or null",
  "rating_movement": "extract movement eg Upgrade or Downgrade or Stable, only if explicitly stated, else null",
  "risk_dimensions": [
    {"dimension": "extract dimension name exactly as labeled", "weight_pct": "extract weight as number or null", "raw_score": "extract raw score as number or null", "weighted_score": "extract weighted score as number or null", "commentary": "extract commentary or remarks if present, else null"}
  ],
  "risk_flags": [
    {"flag_id": "extract flag id or number or null", "description": "extract description", "category": "extract category", "status": "extract mitigation status or null"}
  ],
  "watch_list_triggers": [
    {"trigger_id": "extract a short label for what is being flagged", "condition": "extract the exact metric/condition that triggered the flag, as stated in the document", "consequence": "extract the document's own remark/consequence for this flag, do not invent one"}
  ]
}
Extract every risk dimension row in the composite scorecard, every risk flag in the risk matrix, and only watch triggers that the document itself explicitly marks as a concern.
"""
    risk = extract_json_from_text(text, risk_prompt, merge_mode="dict")

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
                  "risk_ratings", "credit_profile", "collateral_info"]:
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
    run_parser_agent("Passionfruit Inc.")