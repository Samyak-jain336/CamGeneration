from PIL import GimpPaletteFile
import os
import time
from typing import TypedDict, Dict, List
from dotenv import load_dotenv
from typing import Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI



from config import GROQ_API_KEY, LLM_MODEL, OUTPUT_CAM_PATH, GEMINI_API_KEY, LLM_MODEL1
from db import (
    setup_schema, init_queue, mark_task,
    increment_retry, get_task_status,
    get_borrower_profile, get_financials,
    get_audit_findings, get_risk_ratings,
    get_banking_history, get_collateral_info,
    get_connection
)
from temp.parser_agents import run_parser_agent

load_dotenv()

#LLM Setup
llm = ChatGroq(model_name=LLM_MODEL, api_key=GROQ_API_KEY, max_tokens=2048)
'''llm = ChatOllama(
    model="llama3.2",
    temperature=0
)'''


#States

class CAMState(TypedDict):

    input_query:        str
    sections:           Annotated[Dict[str, str], lambda x, y: {**x, **y}]
    errors:             Annotated[Dict[str, str], lambda x, y: {**x, **y}]
    retry_count:        Annotated[Dict[str, int], lambda x, y: {**x, **y}]
    final_cam:          str

    task_queue:         List[str]
    completed_tasks:    Annotated[List[str], operator.add]
    current_task:       Annotated[str, lambda x, y: y]

    company_name:       str

# task execution order
TASK_ORDER = [
    "parse",
    "overview",
    "financial",
    "industry",
    "risk",
    "recommendation"
]

#Orchestrator 
def orchestrator(state: CAMState) -> dict:

    print("\n")
    print("[Orchestrator] Pipeline starting...")
    print("\n")

    setup_schema()

    init_queue(TASK_ORDER)
    print("\n")
    print(f"[Orchestrator] Queue built: {TASK_ORDER}")
    print("\n")

    return {
        "sections":         {},
        "errors":           {},
        "retry_count": {
            "overview":         0,
            "financial":        0,
            "industry":         0,
            "risk":             0,
            "recommendation":   0
        },
        "task_queue":       TASK_ORDER,
        "completed_tasks":  [],
        "current_task":     "parse",
        "company_name":     "",
        "final_cam":        ""
    }

# Parser Node
def parser_node(state: CAMState) -> dict:
    print("\n[Parser Node] Checking if data already exists...")
    mark_task("parse", "in_progress")
    retries = state["retry_count"].get("parse", 0)

    try:
        # check if data already exists
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT COUNT(*) as cnt FROM borrower_profile")
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row["cnt"] > 0:
            print("[Parser Node] Data already in DB — skipping parsing.")
        else:
            print("[Parser Node] No data found — running parser...")
            run_parser_agent("temp")

        # fresh connection to get company name
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT company_name FROM borrower_profile ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        company = row["company_name"] if row else "Unknown"

        mark_task("parse", "done")
        print(f"[Parser Node] Done. Company: {company}")
        print(f"[Queue] parse done")

        return {
            "company_name":     company,
            "completed_tasks":  ["parse"],
            "current_task":     "overview"
        }

    except Exception as e:
        print(f"[Error] Parser failed: {str(e)}")
        print(f"[Retry] Attempt {retries + 1}/3 for parse...")

        mark_task("parse", "failed", str(e))
        increment_retry("parse")

        new_retries = dict(state["retry_count"])
        new_retries["parse"] = retries + 1

        return {
            "errors":       {"parse": str(e)},
            "retry_count":  new_retries
        }

# COMMON RUNNER
def run_section(state: CAMState, section: str, prompt: str) -> dict:

    retries = state["retry_count"].get(section, 0)
    mark_task(section, "in_progress")
    print(f"\n[{section.upper()} Agent] Running... (attempt {retries + 1}/3)")

    try:
        time.sleep(3)
        res = llm.invoke(prompt)
        mark_task(section, "done")

        completed = state.get("completed_tasks", []) + [section]
        print(f"[Queue] {section} done")

        return {
            "sections":        {**state.get("sections", {}), section: res.content},
            "completed_tasks": completed,
            "current_task":    section
        }

    except Exception as e:
        print(f"[Error] {section} failed: {str(e)}")
        print(f"[Retry] Attempt {retries + 1}/3 for {section}...")

        mark_task(section, "failed", str(e))
        increment_retry(section)

        new_retries = dict(state["retry_count"])
        new_retries[section] = retries + 1

        return {
            "errors":      {**state.get("errors", {}), section: str(e)},
            "retry_count": new_retries
        }

# OVERVIEW AGENT
def overview_agent(state: CAMState) -> dict:
    company  = state["company_name"]
    profile  = get_borrower_profile(company)

    prompt = f"""
You are a senior credit analyst writing a Credit Assessment Memo (CAM).
Write Section 1 covering:
1. Executive Summary
2. Borrower Background
3. Purpose of Facility

Use ONLY this data. Be specific. Formal banking language. Figures in Rs Lakhs. State only what is in the data provided. Do not invent stuff.

Company:         {profile.get('company_name')}
CIN:             {profile.get('cin')}
Constitution:    {profile.get('constitution')}
Incorporated:    {profile.get('incorporation_year')}
Office:          {profile.get('registered_office')}
Business:        {profile.get('business_nature')}
Promoters:       {profile.get('promoters')}
Key Management:  {profile.get('key_management')}
Shareholding:    {profile.get('shareholding')}
Query:           {state['input_query']}
"""
    return run_section(state, "overview", prompt)

# FINANCIAL AGENT
def financial_agent(state: CAMState) -> dict:
    company    = state["company_name"]
    financials = get_financials(company)

    fin_text = ""
    for f in financials:
        fin_text += f"""
--- {f.get('fiscal_year')} ---
Revenue: {f.get('revenue')} | PAT: {f.get('pat')} | EBITDA: {f.get('ebitda')} ({f.get('ebitda_margin')}%)
DSCR: {f.get('dscr')} | ICR: {f.get('icr')} | D/E: {f.get('debt_equity_ratio')}
Current Ratio: {f.get('current_ratio')} | Total Assets: {f.get('total_assets')}
Cash from Ops: {f.get('cash_from_operations')} | Closing Cash: {f.get('closing_cash')}
"""

    prompt = f"""
You are a senior credit analyst writing a Credit Assessment Memo (CAM).
Write Section 2 covering:
1. Financial Performance — revenue trends, margins (3 years)
2. Balance Sheet Analysis — assets, debt, equity
3. Key Ratios — DSCR, ICR, Current Ratio, D/E with commentary
4. Cash Flow Analysis

Compare year on year. Highlight improvements or red flags.
Formal banking language. Figures in Rs Lakhs. State only what is in the data provided. Do not invent any thing.

FINANCIALS:
{fin_text}
"""
    return run_section(state, "financial", prompt)

# INDUSTRY AGENT
def industry_agent(state: CAMState) -> dict:
    company = state["company_name"]
    banking = get_banking_history(company)
    audit   = get_audit_findings(company)

    banking_text = ""
    for b in banking:
        banking_text += f"\n{b.get('lender')} | {b.get('facility_type')} | {b.get('purpose')} | Rate: {b.get('interest_rate')}% | O/S: Rs{b.get('outstanding_amt')}L"

    obs_text = "\n".join([
        f"- [{o.get('risk_level')}] {o.get('observation')} -> {o.get('management_response')}"
        for o in audit.get("observations", [])
    ])

    prompt = f"""
You are a senior credit analyst writing a Credit Assessment Memo (CAM).
Write Section 3 covering:
1. Banking Conduct — track record, any defaults or SMA. State only what is in the data provided. Do not invent any SMA history.
2. Existing Credit Facilities — all loans, rates, outstanding amounts
3. Repayment Track Record

Be specific with numbers. Formal banking language. Figures in Rs Lakhs.

IMPORTANT: For section 3.2, you MUST present the credit facilities as a markdown
table using EXACTLY this pipe-delimited format with a separator row:

| Lender | Facility Type | Purpose | Interest Rate | Outstanding Amount (Rs Lakhs) |
| --- | --- | --- | --- | --- |
| <value> | <value> | <value> | <value> | <value> |

Do NOT use bullet points or prose for the facilities list. Use only the table format.

BANKING FACILITIES:
{banking_text}

AUDIT OBSERVATIONS:
{obs_text}

AUDITOR OPINION: {audit.get('opinion')}
GOING CONCERN:   {audit.get('summary_assessment', {}).get('going_concern')}
"""
    return run_section(state, "industry", prompt)

# RISK AGENT
def risk_agent(state: CAMState) -> dict:
    company    = state["company_name"]
    risk       = get_risk_ratings(company)
    collateral = get_collateral_info(company)
    audit      = get_audit_findings(company)

    flags_text = "\n".join([
        f"- [{f.get('category')} | {f.get('status')}] {f.get('description')}"
        for f in risk.get("risk_flags", [])
    ])

    collateral_text = "\n".join([
        f"- {c.get('security_type')}: {c.get('description')} | Value: Rs{c.get('value_lakhs')}L"
        for c in collateral
    ])

    cont_text = "\n".join([
        f"- {c.get('nature')}: Rs{c.get('amount_lakhs')}L — {c.get('remarks')}"
        for c in audit.get("contingent_liabilities", [])
    ])

    prompt = f"""
You are a senior credit analyst writing a Credit Assessment Memo (CAM).
Write Section 4 covering:
1. Risk Assessment — financial, operational, management risks and mitigants
2. Security and Collateral — primary and collateral security, coverage ratio
3. Contingent Liabilities

Clearly separate risks from mitigants. Formal banking language. Figures in Rs Lakhs. State only what is in the data provided. Do not invent anything.

SAVE RISK RATING:
Score: {risk.get('save_score')} | Rating: {risk.get('save_rating')} ({risk.get('rating_movement')} from {risk.get('previous_rating')})
Solvency: {risk.get('solvency_score')}/30 | Asset: {risk.get('asset_score')}/25
Viability: {risk.get('viability_score')}/25 | External: {risk.get('external_score')}/20

RISK FLAGS:
{flags_text}

COLLATERAL:
{collateral_text}

CONTINGENT LIABILITIES:
{cont_text}
"""
    return run_section(state, "risk", prompt)

# RECOMMENDATION AGENT
def recommendation_agent(state: CAMState) -> dict:
    company    = state["company_name"]
    profile    = get_borrower_profile(company)
    financials = get_financials(company)
    risk       = get_risk_ratings(company)
    banking    = get_banking_history(company)
    audit      = get_audit_findings(company)

    latest           = financials[-1] if financials else {}
    total_outstanding = sum(b.get("outstanding_amt", 0) or 0 for b in banking)

    revenue        = latest.get("revenue") or 0
    suggested_loan = round(float(revenue) * 0.25, 2)

    prompt = f"""
You are a senior credit analyst writing a Credit Assessment Memo (CAM).
Write Section 5 — the final section covering:
1. Proposed Credit Structure — facility type, amount, tenor, rate, repayment, covenants
2. Final Recommendation — Approve / Decline / Approve with conditions
3. Key Conditions and Covenants

Be decisive. State specific numbers. Formal banking language. Figures in Rs Lakhs. State only what is in the data provided. Do not invent stuff.

SUMMARY:
Company:              {profile.get('company_name')}
SAVE Rating:          {risk.get('save_rating')} ({risk.get('save_score')}/100)
Latest Revenue:       Rs{latest.get('revenue')}L ({latest.get('fiscal_year')})
PAT:                  Rs{latest.get('pat')}L | Margin: {latest.get('pat_margin')}%
DSCR:                 {latest.get('dscr')}x | ICR: {latest.get('icr')}x
D/E:                  {latest.get('debt_equity_ratio')}x
Total Borrowings:     Rs{round(total_outstanding, 2)}L
Auditor Opinion:      {audit.get('opinion')}
Going Concern:        {audit.get('summary_assessment', {}).get('going_concern')}
Fraud:                {audit.get('summary_assessment', {}).get('fraud')}

PREVIOUS SECTIONS:
Overview:      {state['sections'].get('overview', '')[:300]}...
Financial:     {state['sections'].get('financial', '')[:300]}...
Industry:      {state['sections'].get('industry', '')[:300]}...
Risk:          {state['sections'].get('risk', '')[:300]}...

Suggested Facility Amount: Rs {suggested_loan} Lakhs
(calculated as 25% of latest annual revenue of Rs {revenue} Lakhs)

Query: {state['input_query']}
"""
    return run_section(state, "recommendation", prompt)

# RETRY HANDLER
def retry_failed_sections(state: CAMState) -> dict:

    failed      = state.get("errors", {})
    new_sections = dict(state.get("sections", {}))
    new_errors   = {}
    new_retries  = dict(state["retry_count"])
    completed    = list(state.get("completed_tasks", []))

    if not failed:
        print("\n[Retry] No errors found. Skipping retry.")
        return {
            "sections":        new_sections,
            "errors":          {},
            "retry_count":     new_retries,
            "completed_tasks": completed
        }

    for section, error_msg in failed.items():

        current_retries = new_retries.get(section, 0)

        if current_retries >= 3:
            print(f"[Error] {section} exceeded max retries (3). Skipping.")
            new_errors[section] = f"Max retries exceeded. Last error: {error_msg}"
            continue

        print(f"\n[Error] {section} failed with: {error_msg}")
        print(f"[Retry] Attempt {current_retries + 1}/3 for {section}...")

        try:
            time.sleep(3)
            res = llm.invoke(
                f"You are a senior credit analyst. "
                f"Retry writing the {section} section of a Credit Assessment Memo "
                f"for {state['company_name']}. "
                f"Previous attempt failed with: {error_msg}. "
                f"Write a complete professional {section} section."
            )
            new_sections[section]  = res.content
            new_retries[section]   = current_retries + 1
            completed.append(section)
            mark_task(section, "done")
            print(f"[Retry] {section} succeeded on attempt {current_retries + 1}.")

        except Exception as e:
            new_retries[section] = current_retries + 1
            new_errors[section]  = str(e)
            mark_task(section, "failed", str(e))
            print(f"[Error] {section} failed again: {str(e)}")

    return {
        "sections":        new_sections,
        "errors":          new_errors,
        "retry_count":     new_retries,
        "completed_tasks": completed
    }


# COMBINE CAM
# -----------------------------------------------
# COMBINE CAM
# -----------------------------------------------
def combine_cam(state: CAMState) -> dict:

    s       = state["sections"]
    company = state.get("company_name", "")

    print("\n" + "="*50)
    print("[Combine] Assembling final CAM...")
    print("="*50)

    for section in ["overview", "financial", "industry", "risk", "recommendation"]:
        status = "✓" if section in s else "✗ missing"
        print(f"  {section:<20} {status}")

    # assemble text for terminal
    cam = f"""
{"="*60}
         CREDIT ASSESSMENT MEMORANDUM
{"="*60}
Company: {company}

{"-"*60}
SECTION 1 — BORROWER OVERVIEW & BACKGROUND
{s.get("overview", "[Not generated]")}

{"-"*60}
SECTION 2 — FINANCIAL ANALYSIS
{s.get("financial", "[Not generated]")}

{"-"*60}
SECTION 3 — BANKING & CREDIT HISTORY
{s.get("industry", "[Not generated]")}

{"-"*60}
SECTION 4 — RISK ASSESSMENT & SECURITY
{s.get("risk", "[Not generated]")}

{"-"*60}
SECTION 5 — CREDIT STRUCTURE & RECOMMENDATION
{s.get("recommendation", "[Not generated]")}

{"="*60}
"""
    print("\n" + cam)

    # save as .docx
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor, Inches, Cm
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        import os

        os.makedirs("outputs", exist_ok=True)
        doc = Document()

        # ── Page Setup ──────────────────────────────────────
        section_obj = doc.sections[0]
        section_obj.top_margin    = Cm(2.5)
        section_obj.bottom_margin = Cm(2.5)
        section_obj.left_margin   = Cm(3)
        section_obj.right_margin  = Cm(2.5)

        # ── Header ──────────────────────────────────────────
        header = section_obj.header
        header_para = header.paragraphs[0]
        header_para.text = f"CREDIT ASSESSMENT MEMORANDUM  |  {company}  |  CONFIDENTIAL"
        header_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        header_run = header_para.runs[0]
        header_run.font.size = Pt(8)
        header_run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

        # ── Footer with page numbers ─────────────────────────
        footer = section_obj.footer
        footer_para = footer.paragraphs[0]
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        footer_run = footer_para.add_run()
        footer_run.font.size = Pt(8)
        footer_run.font.color.rgb = RGBColor(0x60, 0x60, 0x60)

        # add page number field
        fldChar1 = OxmlElement('w:fldChar')
        fldChar1.set(qn('w:fldCharType'), 'begin')
        instrText = OxmlElement('w:instrText')
        instrText.text = 'PAGE'
        fldChar2 = OxmlElement('w:fldChar')
        fldChar2.set(qn('w:fldCharType'), 'end')
        footer_run._r.append(fldChar1)
        footer_run._r.append(instrText)
        footer_run._r.append(fldChar2)

        # ── Title Page ──────────────────────────────────────
        title_para = doc.add_paragraph()
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        title_para.paragraph_format.space_before = Pt(60)
        title_run = title_para.add_run("CREDIT ASSESSMENT MEMORANDUM")
        title_run.bold = True
        title_run.font.size = Pt(20)
        title_run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)

        doc.add_paragraph("")

        company_para = doc.add_paragraph()
        company_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        company_run = company_para.add_run(company)
        company_run.bold = True
        company_run.font.size = Pt(14)
        company_run.font.color.rgb = RGBColor(0x33, 0x66, 0x99)

        doc.add_paragraph("")

        conf_para = doc.add_paragraph()
        conf_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        conf_run = conf_para.add_run("CONFIDENTIAL — FOR INTERNAL USE ONLY")
        conf_run.font.size = Pt(9)
        conf_run.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
        conf_run.bold = True

        doc.add_page_break()

        # ── Helper — parse and write content ────────────────
        def write_content(doc, content):
            import re
            from docx.enum.table import WD_TABLE_ALIGNMENT

            lines = content.split("\n")
            idx = 0

            while idx < len(lines):
                line = lines[idx].strip()

                # ── Markdown table detection ──────────────────
                if line.startswith("|") and line.endswith("|"):
                    table_lines = []
                    while idx < len(lines) and lines[idx].strip().startswith("|"):
                        table_lines.append(lines[idx].strip())
                        idx += 1

                    # drop separator rows like | --- | :---: |
                    data_rows = [
                        r for r in table_lines
                        if not re.match(r'^\|[\s\-\|:]+\|$', r)
                    ]
                    if not data_rows:
                        continue

                    parsed = []
                    for row in data_rows:
                        cells = [c.strip() for c in row.strip("|").split("|")]
                        parsed.append(cells)

                    num_cols = max(len(r) for r in parsed)
                    tbl = doc.add_table(rows=len(parsed), cols=num_cols)
                    tbl.style = "Table Grid"
                    tbl.alignment = WD_TABLE_ALIGNMENT.LEFT

                    for ri, row_data in enumerate(parsed):
                        row_obj = tbl.rows[ri]
                        for ci in range(num_cols):
                            cell = row_obj.cells[ci]
                            text = row_data[ci] if ci < len(row_data) else ""
                            text = text.replace("**", "")
                            para = cell.paragraphs[0]
                            run = para.add_run(text)
                            run.font.size = Pt(9.5)

                            if ri == 0:
                                # Header row — dark blue bg, white bold text
                                run.bold = True
                                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                                tcPr = cell._tc.get_or_add_tcPr()
                                shd = OxmlElement('w:shd')
                                shd.set(qn('w:val'), 'clear')
                                shd.set(qn('w:color'), 'auto')
                                shd.set(qn('w:fill'), '003366')
                                tcPr.append(shd)
                            else:
                                run.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
                                # Alternate row shading
                                if ri % 2 == 0:
                                    tcPr = cell._tc.get_or_add_tcPr()
                                    shd = OxmlElement('w:shd')
                                    shd.set(qn('w:val'), 'clear')
                                    shd.set(qn('w:color'), 'auto')
                                    shd.set(qn('w:fill'), 'EAF0F8')
                                    tcPr.append(shd)

                    doc.add_paragraph("")
                    continue

                # ── All existing line-level handlers ──────────
                idx += 1

                if not line:
                    doc.add_paragraph("")
                    continue

                # bullet points — * or -
                if line.startswith("* ") or line.startswith("- ") or line.startswith("+ "):
                    line = line[2:]
                    para = doc.add_paragraph(style="List Bullet")
                    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    parts = line.split("**")
                    for i, part in enumerate(parts):
                        if not part:
                            continue
                        run = para.add_run(part)
                        run.font.size = Pt(10.5)
                        if i % 2 == 1:
                            run.bold = True
                    continue

                # sub bullets — tab indented
                if line.startswith("\t* ") or line.startswith("\t+ "):
                    line = line[3:]
                    para = doc.add_paragraph(style="List Bullet 2")
                    run = para.add_run(line)
                    run.font.size = Pt(10.5)
                    continue

                # handle ### subheadings
                if line.startswith("### "):
                    line = line[4:]
                    para = doc.add_paragraph()
                    run = para.add_run(line)
                    run.bold = True
                    run.font.size = Pt(11)
                    run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
                    continue

                # handle ## subheadings
                if line.startswith("## "):
                    line = line[3:]
                    para = doc.add_paragraph()
                    run = para.add_run(line)
                    run.bold = True
                    run.font.size = Pt(12)
                    run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
                    continue

                # numbered list
                if len(line) > 2 and line[0].isdigit() and line[1] in ".):":
                    para = doc.add_paragraph(style="List Number")
                    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    parts = line.split("**")
                    for i, part in enumerate(parts):
                        if not part:
                            continue
                        run = para.add_run(part)
                        run.font.size = Pt(10.5)
                        if i % 2 == 1:
                            run.bold = True
                    continue

                # section subheadings — lines ending with :
                if line.endswith(":") and len(line) < 60 and "**" not in line:
                    para = doc.add_paragraph()
                    run = para.add_run(line)
                    run.bold = True
                    run.font.size = Pt(11)
                    run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
                    continue

                # normal paragraph with bold parsing
                para = doc.add_paragraph()
                para.paragraph_format.space_after = Pt(4)
                para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                parts = line.split("**")
                for i, part in enumerate(parts):
                    if not part:
                        continue
                    run = para.add_run(part)
                    run.font.size = Pt(10.5)
                    if i % 2 == 1:
                        run.bold = True

        # ── Write Sections ───────────────────────────────────
        section_map = {
            "SECTION 1 — BORROWER OVERVIEW & BACKGROUND":    s.get("overview",       "[Not generated]"),
            "SECTION 2 — FINANCIAL ANALYSIS":                s.get("financial",      "[Not generated]"),
            "SECTION 3 — BANKING & CREDIT HISTORY":          s.get("industry",       "[Not generated]"),
            "SECTION 4 — RISK ASSESSMENT & SECURITY":        s.get("risk",           "[Not generated]"),
            "SECTION 5 — CREDIT STRUCTURE & RECOMMENDATION": s.get("recommendation", "[Not generated]"),
        }

        for heading, content in section_map.items():

            # section heading
            h = doc.add_heading(heading, level=1)
            h.paragraph_format.space_before = Pt(16)
            h.paragraph_format.space_after  = Pt(8)
            for run in h.runs:
                run.font.color.rgb = RGBColor(0x00, 0x33, 0x66)
                run.font.size      = Pt(13)
                run.bold           = True

            # horizontal line under heading
            p = doc.add_paragraph()
            pPr = p._p.get_or_add_pPr()
            pBdr = OxmlElement('w:pBdr')
            bottom = OxmlElement('w:bottom')
            bottom.set(qn('w:val'), 'single')
            bottom.set(qn('w:sz'), '6')
            bottom.set(qn('w:space'), '1')
            bottom.set(qn('w:color'), '003366')
            pBdr.append(bottom)
            pPr.append(pBdr)

            # write content
            write_content(doc, content)
            doc.add_paragraph("")

        # Save to a temp file first — avoids hang if the output
        # file is currently open in Word or a PDF viewer
        import tempfile, shutil
        tmp_path = OUTPUT_CAM_PATH + ".tmp"
        doc.save(tmp_path)
        shutil.move(tmp_path, OUTPUT_CAM_PATH)
        print(f"\n[Combine] CAM saved to: {OUTPUT_CAM_PATH}")

    except Exception as e:
        print(f"[Combine] Could not save .docx: {e}")

    return {"final_cam": cam}


# BUILD GRAPH
graph = StateGraph(CAMState)

# nodes
graph.add_node("orchestrator",   orchestrator)
graph.add_node("parser",         parser_node)
graph.add_node("overview",       overview_agent)
graph.add_node("financial",      financial_agent)
graph.add_node("industry",       industry_agent)
graph.add_node("risk",           risk_agent)
graph.add_node("recommendation", recommendation_agent)
graph.add_node("retry",          retry_failed_sections)
graph.add_node("combine",        combine_cam)

# edges
# edges
graph.add_edge(START,            "orchestrator")
graph.add_edge("orchestrator",   "parser")

# parser → overview + financial in parallel
graph.add_edge("parser",         "overview")
graph.add_edge("parser",         "financial")

# overview → industry
graph.add_edge("overview",       "industry")

# financial + industry → risk
graph.add_edge("financial",      "risk")
graph.add_edge("industry",       "risk")

# risk → recommendation
graph.add_edge("risk",           "recommendation")

# only recommendation feeds retry — it's the last SGA
graph.add_edge("recommendation", "retry")

# retry → combine → END
graph.add_edge("retry",          "combine")
graph.add_edge("combine",        END)

app = graph.compile()

if __name__ == "__main__":

    result = app.invoke({
        "input_query":      "Arjun Textiles Limited — Working Capital Enhancement",
        "sections":         {},
        "errors":           {},
        "retry_count": {
            "parse":            0,
            "overview":         0,
            "financial":        0,
            "industry":         0,
            "risk":             0,
            "recommendation":   0
        },
        "final_cam":        "",
        "company_name":     "",
        "task_queue":       [],
        "completed_tasks":  [],
        "current_task":     ""
    })

    print("\n[Pipeline] Complete!")

    # draw pipeline diagram as PNG
    import os
    os.makedirs("outputs", exist_ok=True)

    print("\n[Mermaid] Drawing pipeline diagram...")
    app.get_graph().draw_mermaid_png(
        output_file_path="outputs/pipeline_diagram.png"
    )
    print("[Mermaid] Saved to: outputs/pipeline_diagram.png")