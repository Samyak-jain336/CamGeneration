from PIL import GimpPaletteFile
import os
import time
from typing import TypedDict, Dict, List
from dotenv import load_dotenv
from typing import Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from config import OUTPUT_CAM_PATH
from db import (
    setup_schema, init_queue, mark_task,
    increment_retry, get_task_status,
    get_borrower_profile, get_financials,
    get_audit_findings, get_risk_ratings,
    get_credit_profile, get_collateral_info,
    get_connection
)
from parser_agents import run_parser_agent
from section_generation_final import SectionGenerationAgent

from inconsistency_agent import InconsistencyAgent
from reconciliation import run_reconciliation
from db import save_reconciliation_results

inconsistency_agent = InconsistencyAgent()

sga = SectionGenerationAgent()

load_dotenv()
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING")

#LLM Setup
#llm = ChatGroq(model_name=LLM_MODEL, api_key=GROQ_API_KEY, max_tokens=2048)
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
    inconsistencies:    Annotated[List[dict], operator.add]

# task execution order
TASK_ORDER = [
    "parse",
    "overview",
    "financial",
    "credit_profile",
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
        "inconsistencies": [],
        "retry_count": {
            "overview":         0,
            "financial":        0,
            "credit_profile":   0,
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

        # Deterministic reconciliation — pure Python, no LLM call.
        # Runs once, right after parsing, before any SGA touches the data.
        print("[Parser Node] Running deterministic reconciliation...")
        financials_rows = get_financials(company)
        recon_results = run_reconciliation(company, financials_rows)
        save_reconciliation_results(recon_results)
        mismatches = [r for r in recon_results if r["status"] == "mismatch"]
        print(f"[Parser Node] Reconciliation done. {len(mismatches)} mismatch(es) found.")

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

# OVERVIEW AGENT
def overview_agent(state: CAMState) -> dict:
    retries = state["retry_count"].get("overview", 0)
    mark_task("overview", "in_progress")
    print(f"\n[OVERVIEW Agent] Running... (attempt {retries + 1}/3)")
    try:
        content = sga.generate(state["company_name"], "generate_overview")
        mark_task("overview", "done")
        print("[Queue] overview done")
        return {
            "sections":        {**state.get("sections", {}), "overview": content},
            "completed_tasks": state.get("completed_tasks", []) + ["overview"],
            "current_task":    "overview"
        }
    except Exception as e:
        mark_task("overview", "failed", str(e))
        increment_retry("overview")
        new_retries = dict(state["retry_count"])
        new_retries["overview"] = retries + 1
        return {"errors": {**state.get("errors", {}), "overview": str(e)}, "retry_count": new_retries}


# FINANCIAL AGENT
def financial_agent(state: CAMState) -> dict:
    retries = state["retry_count"].get("financial", 0)
    mark_task("financial", "in_progress")
    print(f"\n[FINANCIAL Agent] Running... (attempt {retries + 1}/3)")
    try:
        content = sga.generate(state["company_name"], "generate_financial")
        mark_task("financial", "done")
        print("[Queue] financial done")
        return {
            "sections":        {**state.get("sections", {}), "financial": content},
            "completed_tasks": state.get("completed_tasks", []) + ["financial"],
            "current_task":    "financial"
        }
    except Exception as e:
        mark_task("financial", "failed", str(e))
        increment_retry("financial")
        new_retries = dict(state["retry_count"])
        new_retries["financial"] = retries + 1
        return {"errors": {**state.get("errors", {}), "financial": str(e)}, "retry_count": new_retries}


# CREDIT PROFILE AGENT
def credit_profile_agent(state: CAMState) -> dict:
    retries = state["retry_count"].get("credit_profile", 0)
    mark_task("credit_profile", "in_progress")
    print(f"\n[CREDIT PROFILE Agent] Running... (attempt {retries + 1}/3)")
    try:
        content = sga.generate(state["company_name"], "generate_credit_profile")
        mark_task("credit_profile", "done")
        print("[Queue] credit_profile done")
        return {
            "sections":        {**state.get("sections", {}), "credit_profile": content},
            "completed_tasks": state.get("completed_tasks", []) + ["credit_profile"],
            "current_task":    "credit_profile"
        }
    except Exception as e:
        mark_task("credit_profile", "failed", str(e))
        increment_retry("credit_profile")
        new_retries = dict(state["retry_count"])
        new_retries["credit_profile"] = retries + 1
        return {"errors": {**state.get("errors", {}), "credit_profile": str(e)}, "retry_count": new_retries}


# RISK AGENT
def risk_agent(state: CAMState) -> dict:
    retries = state["retry_count"].get("risk", 0)
    mark_task("risk", "in_progress")
    print(f"\n[RISK Agent] Running... (attempt {retries + 1}/3)")
    try:
        content = sga.generate(state["company_name"], "generate_risk")
        mark_task("risk", "done")
        print("[Queue] risk done")
        return {
            "sections":        {**state.get("sections", {}), "risk": content},
            "completed_tasks": state.get("completed_tasks", []) + ["risk"],
            "current_task":    "risk"
        }
    except Exception as e:
        mark_task("risk", "failed", str(e))
        increment_retry("risk")
        new_retries = dict(state["retry_count"])
        new_retries["risk"] = retries + 1
        return {"errors": {**state.get("errors", {}), "risk": str(e)}, "retry_count": new_retries}


# RECOMMENDATION AGENT
def recommendation_agent(state: CAMState) -> dict:
    retries = state["retry_count"].get("recommendation", 0)
    mark_task("recommendation", "in_progress")
    print(f"\n[RECOMMENDATION Agent] Running... (attempt {retries + 1}/3)")
    try:
        content = sga.generate(state["company_name"], "generate_recommendation")
        mark_task("recommendation", "done")
        print("[Queue] recommendation done")
        return {
            "sections":        {**state.get("sections", {}), "recommendation": content},
            "completed_tasks": state.get("completed_tasks", []) + ["recommendation"],
            "current_task":    "recommendation"
        }
    except Exception as e:
        mark_task("recommendation", "failed", str(e))
        increment_retry("recommendation")
        new_retries = dict(state["retry_count"])
        new_retries["recommendation"] = retries + 1
        return {"errors": {**state.get("errors", {}), "recommendation": str(e)}, "retry_count": new_retries}

def inconsistency_node(state: CAMState) -> dict:

    print("\n[INCONSISTENCY Agent] Running...")

    try:

        inconsistencies = inconsistency_agent.detect(
            state["company_name"]
        )

        print(
            f"[INCONSISTENCY Agent] Found {len(inconsistencies)} inconsistencies"
        )

        return {
            "inconsistencies": inconsistencies,
            "completed_tasks": ["inconsistency"],
            "current_task": "inconsistency"
        }

    except Exception as e:

        print(f"[INCONSISTENCY Agent] Failed: {e}")

        return {
            "errors": {
                **state.get("errors", {}),
                "inconsistency": str(e)
            }
        }

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
            task_type = f"generate_{section}"
            content = sga.generate(state["company_name"], task_type)
            new_sections[section]  = content
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

    import json

    s = state["sections"]
    company = state.get("company_name", "")

    inconsistencies = state.get(
        "inconsistencies",
        []
    )

    print("\n" + "="*50)
    print("[Combine] Assembling final CAM...")
    print("="*50)

    for section in ["overview", "financial", "credit_profile", "risk", "recommendation"]:
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
SECTION 3 — EXISTING DEBT & CREDIT PROFILE
{s.get("credit_profile", "[Not generated]")}

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
    "SECTION 1 — BORROWER OVERVIEW & BACKGROUND":
        s.get("overview", "[Not generated]"),

    "SECTION 2 — FINANCIAL ANALYSIS":
        s.get("financial", "[Not generated]"),

    "SECTION 3 — EXISTING DEBT & CREDIT PROFILE":
        s.get("credit_profile", "[Not generated]"),

    "SECTION 4 — RISK ASSESSMENT & SECURITY":
        s.get("risk", "[Not generated]"),

    "SECTION 5 — DATA CONSISTENCY REVIEW":
        ("No material inconsistencies were detected across the Annual Report, "
         "Audit Report, and SAVE Risk Assessment for this borrower."
         if not inconsistencies else
         "\n\n".join(
             f"- **{i.get('field_name', 'Unknown field')}**: "
             f"{i.get('source_1','Source 1')} reports \"{i.get('value_1','')}\" "
             f"while {i.get('source_2','Source 2')} reports \"{i.get('value_2','')}\" "
             f"(severity: {i.get('severity','minor')})"
             for i in inconsistencies
         )),

    "SECTION 6 — CREDIT STRUCTURE & RECOMMENDATION":
        s.get("recommendation", "[Not generated]")
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
graph.add_node("credit_profile", credit_profile_agent)
graph.add_node("risk",           risk_agent)
graph.add_node("recommendation", recommendation_agent)
graph.add_node("inconsistency", inconsistency_node)
graph.add_node("retry", retry_failed_sections)
graph.add_node("combine",        combine_cam)

# edges
graph.add_edge(START,            "orchestrator")
graph.add_edge("orchestrator",   "parser")

# parser → overview + financial in parallel
graph.add_edge("parser",         "overview")
graph.add_edge("parser",         "financial")

# overview → credit_profile (renamed from industry)
graph.add_edge("overview",       "credit_profile")

# financial + credit_profile → risk
graph.add_edge("financial",      "risk")
graph.add_edge("credit_profile", "risk")

# risk → recommendation
graph.add_edge("risk",           "recommendation")

# only recommendation feeds retry — it's the last SGA
graph.add_edge(
    "recommendation",
    "inconsistency"
)

graph.add_edge(
    "inconsistency",
    "retry"
)

# retry → combine → END
graph.add_edge("retry",          "combine")
graph.add_edge("combine",        END)
app = graph.compile()

if __name__ == "__main__":

    result = app.invoke({
        "input_query":      "ARJUN TEXTILES LIMITED",
        "sections":         {},
        "errors":           {},
        "inconsistencies": [],
        "retry_count": {
            "parse":            0,
            "overview":         0,
            "financial":        0,
            "credit_profile":   0,
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
