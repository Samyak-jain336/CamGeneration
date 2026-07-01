# ============================================================
# parser_agents.py
# Reads 3 input documents → extracts structured data → stores
# into MySQL via db.py
#
# Uses separate focused prompts per extraction category for
# maximum accuracy. Each category gets full LLM attention and
# full token budget per chunk.
# ============================================================

import json
import re
import time
import pdfplumber
import openpyxl
import requests

from groq import Groq
from openai import OpenAI
from google import genai


from config import (
    GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3,
    GROQ_API_KEY_4, GROQ_API_KEY_5, GROQ_API_KEY_6,
    GROQ_MODEL,
    OLLAMA_MODEL, OLLAMA_URL,
    OPENAI_API_KEY, OPENAI_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    DOC_PATHS
)

from db import (
    save_borrower_profile, save_financials, save_audit_findings,
    save_risk_ratings, save_collateral_info, save_credit_profile,
    save_raw_document, get_connection
)


# ============================================================
# LLM CLIENTS
# ============================================================

groq_client_1 = Groq(api_key=GROQ_API_KEY)
groq_client_2 = Groq(api_key=GROQ_API_KEY_2)
groq_client_3 = Groq(api_key=GROQ_API_KEY_3)
groq_client_4 = Groq(api_key=GROQ_API_KEY_4)
groq_client_5 = Groq(api_key=GROQ_API_KEY_5)
groq_client_6 = Groq(api_key=GROQ_API_KEY_6)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


SYSTEM_PROMPT = (
    "You are a financial document parser. Always respond with valid JSON only. "
    "No explanation, no markdown fences, no extra text. "
    "If a field is not found in this chunk, return null for that field — "
    "never guess or fabricate values."
)


# ============================================================
# MATH EXPRESSION SAFETY NET
# ============================================================

def fix_math_expressions_in_json(raw: str) -> str:
    pattern = re.compile(
        r'("\w+"\s*:\s*)([\(\)\d\.\s\+\-\*\/]+)([,\}\n])'
    )
    def try_eval(match):
        prefix, expr, suffix = match.groups()
        stripped = expr.strip()
        if re.fullmatch(r'-?\d+(\.\d+)?', stripped):
            return match.group(0)
        # Only treat as arithmetic if there's a clearly-spaced operator,
        # e.g. "8547 + 62" — never a tight hyphen like "1998-99" or "13-08-2024"
        if not re.search(r'\d\s[\+\-\*\/]\s\d', stripped):
            return match.group(0)
        try:
            value = eval(stripped, {"__builtins__": {}}, {})
            if isinstance(value, (int, float)):
                return f"{prefix}{value}{suffix}"
        except Exception:
            pass
        return match.group(0)
    return pattern.sub(try_eval, raw)


# ============================================================
# LLM PROVIDER FUNCTIONS
# ============================================================

def call_gemini(prompt: str) -> str:
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
        config={"temperature": 0, "response_mime_type": "application/json"}
    )
    return response.text

def call_groq_1(prompt: str) -> str:
    response = groq_client_1.chat.completions.create(
        model=GROQ_MODEL, max_tokens=4096,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def call_groq_2(prompt: str) -> str:
    response = groq_client_2.chat.completions.create(
        model=GROQ_MODEL, max_tokens=4096,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def call_groq_3(prompt: str) -> str:
    response = groq_client_3.chat.completions.create(
        model=GROQ_MODEL, max_tokens=4096,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def call_groq_4(prompt: str) -> str:
    response = groq_client_4.chat.completions.create(
        model=GROQ_MODEL, max_tokens=4096,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def call_groq_5(prompt: str) -> str:
    response = groq_client_5.chat.completions.create(
        model=GROQ_MODEL, max_tokens=4096,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def call_groq_6(prompt: str) -> str:
    response = groq_client_6.chat.completions.create(
        model=GROQ_MODEL, max_tokens=4096,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def call_openai(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL, max_tokens=4096,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def call_ollama(prompt: str) -> str:
    """Calls local Ollama instance. Requires: ollama serve + ollama pull llama3.1:8b"""
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_predict": 4096}
        },
        timeout=300
    )
    response.raise_for_status()
    return response.json()["response"]


# ============================================================
# PROVIDER FALLBACK CHAIN
# ============================================================

GEMINI_MAX_RETRIES = 5

def call_parser_llm(prompt: str) -> str:
    """
    Gemini-only. Retries on failure (e.g. 503 UNAVAILABLE, rate limit)
    up to GEMINI_MAX_RETRIES times with increasing backoff.
    Raises after exhausting retries — caller (extract_json_from_text)
    catches this per-chunk and skips that chunk.
    """
    last_error = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            print(f"  [LLM] Trying Gemini (attempt {attempt}/{GEMINI_MAX_RETRIES})...")
            result = call_gemini(prompt)
            print(f"  [LLM] Gemini succeeded.")
            return result
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in
                   ["503", "unavailable", "overloaded"]):
                print(f"  [LLM] Gemini unavailable (attempt {attempt}/{GEMINI_MAX_RETRIES})...")
            elif any(kw in error_str for kw in
                   ["rate limit", "rate_limit", "429", "quota",
                    "too many requests", "resource_exhausted", "exceeded"]):
                print(f"  [LLM] Gemini rate limited (attempt {attempt}/{GEMINI_MAX_RETRIES})...")
            else:
                print(f"  [LLM] Gemini error (attempt {attempt}/{GEMINI_MAX_RETRIES}): {e}")
            last_error = e
            if attempt < GEMINI_MAX_RETRIES:
                backoff = 5 * attempt
                print(f"  [LLM] Retrying in {backoff}s...")
                time.sleep(backoff)
            continue

    raise Exception(f"Gemini failed after {GEMINI_MAX_RETRIES} attempts. Last error: {last_error}")


# ============================================================
# PDF TEXT EXTRACTOR (with layout detection)
# ============================================================

def detect_page_layout(page) -> str:
    """Detects single-column, two-column, or blank page layout."""
    width = page.width
    words = page.extract_words()

    if not words:
        return "blank"

    total_count = len(words)
    left_count = sum(1 for w in words if float(w["x0"]) < width / 2)
    right_count = total_count - left_count

    if total_count == 0:
        return "blank"

    left_ratio = left_count / total_count
    right_ratio = right_count / total_count

    if left_ratio >= 0.25 and right_ratio >= 0.25:
        left_words = [w for w in words if float(w["x0"]) < width / 2]
        right_words = [w for w in words if float(w["x0"]) >= width / 2]
        mid_gap = width / 2
        gap_words = [w for w in words
                     if float(w["x1"]) > mid_gap * 0.85
                     and float(w["x0"]) < mid_gap * 1.15]
        if len(gap_words) / total_count < 0.05:
            return "two_col"

    return "single"


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF. Auto-detects layout per page:
    single → one block, two-column → left/right halves, blank → skipped.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            width = page.width
            height = page.height
            layout = detect_page_layout(page)

            if layout == "blank":
                continue
            elif layout == "two_col":
                left_bbox = (0, 0, width / 2, height)
                right_bbox = (width / 2, 0, width, height)
                left_text = page.crop(left_bbox).extract_text()
                right_text = page.crop(right_bbox).extract_text()
                if left_text and left_text.strip():
                    text += f"\n\n--- PAGE {page_num + 1}A ---\n{left_text.strip()}"
                if right_text and right_text.strip():
                    text += f"\n\n--- PAGE {page_num + 1}B ---\n{right_text.strip()}"
            else:
                full_text = page.extract_text()
                if full_text and full_text.strip():
                    text += f"\n\n--- PAGE {page_num + 1} ---\n{full_text.strip()}"

    return text.strip()


# ============================================================
# EXCEL TEXT EXTRACTOR
# ============================================================

def extract_text_from_xlsx(xlsx_path: str) -> str:
    """Reads ALL sheets from an Excel workbook as plain text."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    all_text = ""
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        all_text += f"\n\n=== SHEET: {sheet_name} ===\n"
        for row in ws.iter_rows(values_only=True):
            if any(cell is not None for cell in row):
                row_text = " | ".join(
                    str(cell) if cell is not None else "" for cell in row
                )
                all_text += row_text + "\n"
    return all_text.strip()


# ============================================================
# CHUNKING + LLM EXTRACTION
# ============================================================

CHUNK_SIZE = 90000
OVERLAP = 2000

def build_chunks(text: str) -> list:
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - OVERLAP
    return chunks


def clean_llm_json(raw: str) -> str:
    """Strips markdown fences and fixes math expressions."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    raw = fix_math_expressions_in_json(raw)
    return raw


def parse_json_safe(raw: str):
    """Attempts json.loads with a fallback for common LLM formatting errors."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try adding missing opening braces for array elements
        raw = re.sub(r',\s*\n(\s*"[a-z_]+")', r',\n    {\1', raw)
        return json.loads(raw)


def resolve_field_votes(field_votes: dict) -> dict:
    """
    Resolves competing chunk answers for the same field.
    - Lists: pick the LONGEST list (most complete extraction)
    - Dicts: pick the one with the most non-null keys
    - Scalars: most common non-empty answer wins; ties → last chunk wins
    """
    resolved = {}
    for field, votes in field_votes.items():
        if not votes:
            continue

        # Lists: pick the longest (most complete extraction)
        if isinstance(votes[0], list):
            resolved[field] = max(votes, key=len)
            continue

        # Dicts: pick the one with the most non-null populated keys
        if isinstance(votes[0], dict):
            resolved[field] = max(
                votes,
                key=lambda d: sum(
                    1 for v in d.values()
                    if v is not None and v != "" and v != [] and v != {}
                )
            )
            continue

        # Scalars: majority vote, ties → last chunk wins
        counts = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        max_count = max(counts.values())
        tied = [v for v in votes if counts[v] == max_count]
        resolved[field] = tied[-1]

    return resolved


def extract_json_from_text(text: str, prompt: str, merge_mode: str = "dict"):
    """
    Splits document into overlapping chunks, sends each to LLM with
    the given prompt, then merges results.
    merge_mode = "dict"  → merges into one dict (single-object extractions)
    merge_mode = "list"  → concatenates into one list (array extractions)
    """
    chunks = build_chunks(text)
    print(f"[Parser] Document split into {len(chunks)} chunk(s).")

    merged_list = []
    field_votes = {}

    for i, chunk in enumerate(chunks):
        print(f"[Parser] Processing chunk {i + 1}/{len(chunks)}...")
        try:
            full_prompt = f"{prompt}\n\nDOCUMENT CHUNK {i + 1} OF {len(chunks)}:\n{chunk}"
            raw = call_parser_llm(full_prompt)
            raw = clean_llm_json(raw)

            print(f"\n{'=' * 60}\nRAW LLM OUTPUT — Chunk {i + 1}\n{'=' * 60}")
            print(raw[:2000] + ("..." if len(raw) > 2000 else ""))
            print("=" * 60)

            parsed = parse_json_safe(raw)

            if merge_mode == "list":
                if isinstance(parsed, list):
                    merged_list.extend(parsed)
            else:
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if v is None or v == "" or v == [] or v == {}:
                            continue
                        field_votes.setdefault(k, []).append(v)

            time.sleep(3)

        except Exception as e:
            print(f"[Parser] Chunk {i + 1} failed: {e} — skipping.")
            time.sleep(3)
            continue

    if merge_mode == "dict":
        return resolve_field_votes(field_votes)
    return merged_list


# ============================================================
# FINANCIALS DEDUPLICATION
# ============================================================

def deduplicate_list_rows(rows: list, key_fields: list, require_key: bool = False) -> list:
    """
    Generic dedup for merge_mode='list' extractions split across
    overlapping chunks. Same key_fields → same row; keep the one
    with the most non-null fields.

    require_key=True skips rows where every key field is empty/falsy
    (financials rows with no fiscal_year have nothing to key on).
    """
    best = {}
    for row in rows:
        key = tuple(row.get(f) for f in key_fields)
        if require_key and not any(key):
            continue
        non_nulls = sum(1 for v in row.values() if v not in (None, "", "null"))
        if key not in best or non_nulls > best[key]["score"]:
            best[key] = {"row": row, "score": non_nulls}
    deduped = [entry["row"] for entry in best.values()]
    print(f"[Parser] Deduped on {key_fields}: {len(rows)} → {len(deduped)}")
    return deduped


PROMOTER_KW = ["promoter"]
INSTITUTIONAL_KW = ["bank", "institution", "fii", "dii", "mutual fund",
                     "insurance", "body corporate", "bodies corporate"]

def bucket_shareholding(categories: list) -> dict:
    """
    Buckets raw shareholding categories (any number, any labels) into
    promoter / institutional / non-institutional. Every category lands
    in exactly one bucket — the three numbers always sum to ~100%.
    """
    promoter = inst = non_inst = 0.0
    for c in categories:
        label = (c.get("category") or "").lower()
        pct = c.get("pct") or 0
        if any(k in label for k in PROMOTER_KW):
            promoter += pct
        elif any(k in label for k in INSTITUTIONAL_KW):
            inst += pct
        else:
            non_inst += pct
    return {"promoter_pct": promoter, "public_inst_pct": inst, "public_non_inst_pct": non_inst}

# ============================================================
# PROMPT TEMPLATES — one per extraction category
# ============================================================

PROFILE_PROMPT = """
Extract company profile information from this financial document.
Return ONLY valid JSON object, no explanation, no markdown fences.
If a field isn't found in this chunk, return null — never guess or fabricate.
The document may be an Indian Annual Report, a US SEC 10-K, or any other corporate filing.
Extract by meaning, not by label — labels vary across jurisdictions.

{
  "company_name": "extract the full legal company name", keep under 255 char (if already less make no change if somehow greater than 255, use shortform ONLY ONLY in that case)
  "cin": "extract registration or identification number (CIN, EIN, File No., etc.) or null",
  "constitution": "extract entity type eg Private Limited, Public Limited, Corporation, LLC, or null",
  "incorporation_year": "extract as integer or null",
  "registered_office": "extract registered or principal office full address or null",
  "business_nature": "extract what the company does — products, services, industry description, or null",
  "promoters": [{"name": "extract name", "designation": "extract role"}],
  "key_management": [{"name": "extract name", "designation": "extract role", "qualification": "extract if available else null", "experience": "extract years or description if available else null"}],
  "shareholding_categories": [
    {"category": "extract exact category label as printed, e.g. 'Promoters Holding', 'Other Bodies Corporate', 'Non Resident Indians'", "pct": "extract percentage as number"}
  ]
}

IMPORTANT: Extract ALL executives, directors, and officers listed in the document — not just the CEO.
If there is a table of directors or executive officers, extract every person from it.
"""

FINANCIALS_PROMPT = """
Extract financial data from this financial document.
Return ONLY a valid JSON array, no explanation, no markdown fences.
If a field isn't found in this chunk, return null — never guess or fabricate.
CRITICAL: Every numeric value must be a plain final number (e.g. 8547, 41.5, -867).
NEVER write arithmetic expressions as values (e.g. do NOT write "8547 + 62").

MULTI-YEAR RULE (MANDATORY):
If this chunk contains a multi-year financial statement (e.g. columns for FY2023, FY2024, FY2025),
you MUST return ONE SEPARATE JSON object for EACH fiscal year in the array.
For example, if the chunk shows 3 years, return 3 objects — one per year.
NEVER collapse multiple years into one object. NEVER return only the latest year.
If only one year is present in this chunk, return one object.

REVENUE vs TOTAL REVENUE (MANDATORY — READ CAREFULLY):
- "revenue" = the net sales / turnover / revenue from operations line item ONLY, copied exactly as printed in the document. Do not add, estimate, or round anything onto it.
- "other_income" = any separate "other income" or "other income/(expense)" line item, copied exactly as printed. If no such line exists in this chunk, set it to null.
- "total_revenue" = revenue + other_income, calculated by you ONLY when both values are present and explicitly stated as separate lines in the document. If other_income is null, set total_revenue equal to revenue (do not invent a different number).
- NEVER estimate, round, average, or approximate any of these three values. Every number must be a number you can point to as literally printed in the document text for this fiscal year.
- Worked pattern: if the document shows "Total net sales: X" and "Other income, net: Y" as two distinct lines, then revenue=X, other_income=Y, total_revenue=X+Y. If the document shows only ONE combined revenue line with no separate other-income breakdown, set revenue equal to that single line and leave other_income null and total_revenue equal to revenue.

[
  {
    "company_name": "extract company name",
    "currency_unit": "extract the currency and scale used in this document, e.g. USD millions, INR Lakhs, INR Crores, EUR thousands — state exactly as the document presents it",
    "fiscal_year": "extract fiscal year label eg FY2025 or 2024 — each object MUST have a DIFFERENT year", pls keep it under 6-7 char,
    "revenue": "extract NET SALES only as number (exclude other income)",
    "other_income": "extract other income as number or null",
    "total_revenue": "extract or calculate net sales + other income as number or null",
    "cost_of_materials": "extract cost of goods sold or cost of sales as number or null",
    "employee_expense": "extract employee or staff costs as number or null",
    "finance_costs": "extract interest expense or finance costs as number or null",
    "depreciation": "extract depreciation and amortisation as number or null",
    "other_expenses": "extract other operating expenses as number or null",
    "pbt": "extract profit or income before tax as number or null",
    "tax": "extract tax provision or tax expense as number or null",
    "pat": "extract net income or profit after tax as number or null",
    "ebitda": "extract EBITDA as number or null — if not stated, calculate as operating income + depreciation",
    "ebitda_margin": "extract or calculate EBITDA / revenue as percentage or null",
    "pat_margin": "extract or calculate net income / revenue as percentage or null",
    "current_ratio": "extract as number or null",
    "quick_ratio": "extract as number or null",
    "debt_equity_ratio": "extract as number or null",
    "icr": "extract interest coverage ratio as number or null",
    "dscr": "extract debt service coverage ratio as number or null",
    "roe": "extract return on equity as percentage or null",
    "roce": "extract return on capital employed as percentage or null",
    "asset_turnover": "extract or calculate as number or null",
    "inventory_days": "extract or calculate as integer or null",
    "debtor_days": "extract or calculate as integer or null",
    "working_capital_days": "extract or calculate as integer or null",
    "total_assets": "extract total assets as number or null",
    "total_equity": "extract total shareholders equity as number or null",
    "total_borrowings": "extract total debt including short and long term as number or null",
    "cash_from_operations": "extract cash from operating activities as number or null",
    "cash_from_investing": "extract cash from investing activities as number or null",
    "cash_from_financing": "extract cash from financing activities as number or null",
    "closing_cash": "extract closing cash and cash equivalents as number or null"
  }
]
"""

CREDIT_PROMPT = """
Extract debt and credit profile information from this financial document.
Return ONLY a valid JSON array, no explanation, no markdown fences.
If a field isn't found in this chunk, return null — never guess or fabricate.
Extract EVERY debt instrument mentioned: term loans, bonds, debentures, credit facilities, notes, etc.
ONLY extract a row if the source text explicitly identifies it as a loan, borrowing,
credit facility, deposit, debenture, or bond — typically with stated terms (security,
tenor, interest rate) in a dedicated "Borrowings" or "Loans" note.

Do NOT extract amounts from a generic "outstanding balance" / "amounts due to/from
related parties" table unless that specific row explicitly calls it a loan or advance.
If the nature of a related-party balance is ambiguous (could be a payable, lease, or
trade balance rather than a loan), set instrument_type to "Related Party Balance —
Unclassified" instead of labeling it a loan.

[
  {
    "company_name": "extract company name",
    "instrument_type": "extract type of debt instrument (Term Loan, Senior Unsecured Notes, Bond, Debenture, Revolving Credit Facility, etc.)",
    "counterparty": "extract lender, bank, or bondholder name if available else null",
    "purpose": "extract stated purpose of the borrowing if available else null",
    "interest_rate": "extract as a plain numeric value with no percent sign and no units, e.g. 9.25 not '9.25%' or '9.25 percent'",
    "outstanding_amt": "extract outstanding balance as number or null",
    "maturity_date": "extract maturity or repayment date as string or null",
    "credit_rating": "extract external credit rating (S&P, Moody's, CRISIL, ICRA etc.) if available else null",
    "covenants": "extract key covenant terms as text if available else null",
    "fiscal_year": "extract fiscal year this data relates to"
  }
]
"""

COLLATERAL_PROMPT = """
Extract collateral and security information from this financial document.
Return ONLY a valid JSON array, no explanation, no markdown fences.
If a field isn't found in this chunk, return null — never guess or fabricate.
If no collateral or security is mentioned in this chunk, return an empty array: []

[
  {
    "company_name": "extract company name",
    "security_type": "extract type of security or collateral (Mortgage, Hypothecation, Lien, Pledge, Negative Pledge, etc.)", keep as small as possible.
    "description": "extract description of the asset or security",
    "value_lakhs": "extract value as number in document units or null",
    "remarks": "extract any additional notes or terms or null"
  }
]
"""

AUDIT_PROMPT = """
Extract audit findings from this Audit Report as a JSON object.
Return ONLY valid JSON, no explanation, no markdown fences.
If a field isn't found in this chunk, return null — never guess or fabricate.

IMPORTANT — Opinion field:
Use the EXACT term from the document. Common values:
- "Unqualified" (also called "Clean" opinion — use "Unqualified")
- "Qualified"
- "Adverse"
- "Disclaimer of Opinion"

{
  "company_name": "extract from document",
  "audit_year": "extract fiscal year",
  "auditor_name": "extract auditor firm name",
  "firm_reg_no": "extract firm registration number or PCAOB ID or null",
  "opinion": "extract opinion as Unqualified or Qualified or Adverse or Disclaimer",
  "key_audit_matters": [
    {"matter": "extract matter name", "description": "extract description", "how_addressed": "extract how addressed"}
  ],
  "caro_findings": [
    {"para": "extract para number", "matter": "extract matter", "finding": "extract finding"}
  ],
  "observations": [
    {"obs_no": "extract observation number or null", "observation": "extract observation text", "risk_level": "extract risk level or null", "management_response": "extract response or null"}
  ],
  "summary_assessment": {
    "financial_integrity": "extract assessment or null",
    "internal_controls": "extract assessment or null",
    "going_concern": "extract assessment or null",
    "fraud": "extract assessment or null"
  },
  "contingent_liabilities": [
    {"nature": "extract nature", "amount_lakhs": "extract amount as number or null", "remarks": "extract remarks or null"}
  ]
}
Extract all key audit matters, findings, observations and contingent liabilities from the document.
If the document uses the term Critical Audit Matters (CAM/KAM), extract those as key_audit_matters.
"""

AUDIT_CATCHALL_PROMPT = """
Extract anything important in this chunk that is NOT already covered by:
auditor name, opinion type, key audit matters, CARO findings, observations,
or contingent liabilities.

This includes things like: management's qualitative remarks, "advantages" or
"strengths" sections, internal control commentary, going-concern discussion
not already captured, or any other narrative content that seems relevant to
a credit analyst but doesn't fit a named field.

Return ONLY a valid JSON array, no explanation, no markdown fences.
If nothing in this chunk fits, return an empty array: []

[
  {"topic": "short label for what this is, e.g. 'Advantages'", "content": "the actual text/summary"}
]
"""

AUDIT_COVERAGE_CHECK_PROMPT = """
You are reviewing an audit report for completeness of extraction.

Below is a list of everything ALREADY extracted from this document into
structured fields. Read the full audit report text further below, and
identify any audit-relevant content — findings, observations, qualitative
remarks, sections like "advantages", management commentary, going-concern
notes, anything a credit analyst would care about — that is NOT represented,
even approximately, in the list already extracted.

Return ONLY a valid JSON array, no explanation, no markdown fences.
If everything relevant is already covered, return an empty array: []

[
  {"topic": "short label", "content": "the missed text/summary"}
]
"""

RISK_PROMPT = """
Extract SAVE risk rating data from this document as a JSON object.
Return ONLY valid JSON, no explanation, no markdown fences.
If a field isn't found in this chunk, return null — never guess or fabricate a rating label or trigger.

The document contains a composite weighted risk scorecard with several named risk dimensions
(e.g. "Financial Strength", "Profitability & Growth", "Cash Flow Adequacy", "Asset Quality",
"Industry / Market Risk", "Management Quality", "Compliance & Legal", "ESG / Sustainability" —
exact names and number of dimensions vary by document, extract whatever actually appears).
Each dimension typically has a weight (%), a raw score (often out of 10), and a weighted score.

The document also contains an overall composite score and a rating SCALE that maps score ranges
to rating labels (e.g. "8.0-10.0 = LOW RISK"). Find which range the composite score falls into
and use THAT exact label as save_rating — do not invent a different label format (e.g. do not
write "SB-1" or similar unless that exact label literally appears in the rating scale table).

A "watch list trigger" is something the document itself explicitly marks as a concern, breach,
or watch status. Do NOT infer or fabricate a trigger from a metric that the document does not
itself flag as a concern.

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
Extract every risk dimension row in the composite scorecard, every risk flag in the risk matrix,
and only watch triggers that the document itself explicitly marks as a concern.
"""


# ============================================================
# INDIVIDUAL DOCUMENT PARSERS
# ============================================================

def parse_annual_report(company_name: str):
    """
    Reads annual_report.pdf → extracts 4 categories using
    SEPARATE focused prompts (one per category). Each category
    gets full LLM attention and full token budget per chunk.
    """
    print("[Parser] Reading Annual Report...")
    text = extract_text_from_pdf(DOC_PATHS["annual_report"])

    # ── 1. PROFILE ──────────────────────────────────────────
    print("\n[Parser] === Extracting: PROFILE ===")
    profile = extract_json_from_text(text, PROFILE_PROMPT, merge_mode="dict")
    if "shareholding_categories" in profile:
        profile["shareholding"] = bucket_shareholding(profile["shareholding_categories"])
    save_borrower_profile(profile)
    print("[Parser] Borrower profile saved.")
    time.sleep(3)

    # ── 2. FINANCIALS ───────────────────────────────────────
    print("\n[Parser] === Extracting: FINANCIALS ===")
    financials_raw = extract_json_from_text(text, FINANCIALS_PROMPT, merge_mode="list")
    financials = deduplicate_list_rows(financials_raw, ["fiscal_year"], require_key=True)
    if financials:
        save_financials(financials)
    print(f"[Parser] Financials saved: {len(financials)} year(s).")
    time.sleep(3)

    # ── 3. CREDIT PROFILE ───────────────────────────────────
    print("\n[Parser] === Extracting: CREDIT PROFILE ===")
    credit_raw = extract_json_from_text(text, CREDIT_PROMPT, merge_mode="list")
    credit = deduplicate_list_rows(credit_raw, ["instrument_type", "counterparty", "outstanding_amt", "fiscal_year"])
    if credit:
        save_credit_profile(credit)
    print(f"[Parser] Credit profile saved: {len(credit)} instrument(s).")
    time.sleep(3)

    # ── 4. COLLATERAL ───────────────────────────────────────
    print("\n[Parser] === Extracting: COLLATERAL ===")
    collateral_raw = extract_json_from_text(text, COLLATERAL_PROMPT, merge_mode="list")
    collateral = deduplicate_list_rows(collateral_raw, ["security_type", "description", "value_lakhs"])
    if collateral:
        save_collateral_info(collateral)
    print(f"[Parser] Collateral saved: {len(collateral)} item(s).")


def parse_audit_report(company_name: str):
    """Reads audit_report.pdf → extracts audit_findings."""
    print("\n[Parser] === Reading Audit Report ===")
    text = extract_text_from_pdf(DOC_PATHS["audit_report"])

    save_raw_document(company_name, None, text)

    audit = extract_json_from_text(text, AUDIT_PROMPT, merge_mode="dict")
    time.sleep(3)

    print("[Parser] === Extracting: AUDIT CATCH-ALL ===")
    other_findings = extract_json_from_text(text, AUDIT_CATCHALL_PROMPT, merge_mode="list")
    time.sleep(3)

    print("[Parser] === Running coverage check ===")
    already_extracted_json = json.dumps({
        "key_audit_matters": audit.get("key_audit_matters", []),
        "caro_findings": audit.get("caro_findings", []),
        "observations": audit.get("observations", []),
        "contingent_liabilities": audit.get("contingent_liabilities", []),
        "other_findings": other_findings
    }, default=str)

    coverage_prompt = (
        AUDIT_COVERAGE_CHECK_PROMPT
        + "\n\nALREADY EXTRACTED:\n" + already_extracted_json
        + "\n\nFULL AUDIT REPORT TEXT:\n" + text
    )
    try:
        raw = call_parser_llm(coverage_prompt)
        raw = clean_llm_json(raw)
        gaps = parse_json_safe(raw)
        if isinstance(gaps, list) and gaps:
            print(f"[Parser] Coverage check found {len(gaps)} gap(s).")
            other_findings.extend(gaps)
    except Exception as e:
        print(f"[Parser] Coverage check failed: {e} — skipping.")

    audit["other_findings"] = other_findings
    time.sleep(3)

    save_audit_findings(audit)
    print(f"[Parser] Audit findings saved. ({len(other_findings)} total catch-all item(s))")


def parse_save_risk_file(company_name: str):
    """Reads SAVE Risk Excel → extracts risk_ratings."""
    print("\n[Parser] === Reading SAVE Risk File ===")
    text = extract_text_from_xlsx(DOC_PATHS["save_risk"])
    risk = extract_json_from_text(text, RISK_PROMPT, merge_mode="dict")
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

    # Clean all tables before re-parsing
    conn = get_connection()
    cursor = conn.cursor()
    for table in ["borrower_profile", "financials", "audit_findings",
                  "risk_ratings", "credit_profile", "collateral_info",
                  "raw_documents", "inconsistencies"]:
        cursor.execute(f"TRUNCATE TABLE {table}")
    conn.commit()
    cursor.close()
    conn.close()
    print("[Parser] All tables cleaned.")

    parse_annual_report(company_name)
    parse_audit_report(company_name)
    parse_save_risk_file(company_name)

    print(f"\n[Parser Agent] All documents parsed and stored in MySQL.\n")
    return True


if __name__ == "__main__":
    run_parser_agent("Durlax Top Surface")
