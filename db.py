import json
import re
import mysql.connector
from mysql.connector import Error
from config import DB_CONFIG
from dotenv import load_dotenv
load_dotenv()

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)



SCHEMA_SQL = [

    # -- Borrower Profile ------------------------------------
    """
    CREATE TABLE IF NOT EXISTS borrower_profile (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        company_name    VARCHAR(255),
        cin             VARCHAR(50),
        constitution    VARCHAR(255),
        incorporation_year INT,
        registered_office TEXT,
        business_nature TEXT,
        promoters       JSON,
        key_management  JSON,
        shareholding    JSON
    )
    """,

    # -- Financials (one row per year) -----------------------
    """
    CREATE TABLE IF NOT EXISTS financials (
        id                      INT AUTO_INCREMENT PRIMARY KEY,
        company_name            VARCHAR(255),
        fiscal_year             VARCHAR(10),
        currency_unit           VARCHAR(50),
        revenue                 DECIMAL(15,2),
        other_income            DECIMAL(15,2),
        total_revenue           DECIMAL(15,2),
        cost_of_materials       DECIMAL(15,2),
        employee_expense        DECIMAL(15,2),
        finance_costs           DECIMAL(15,2),
        depreciation            DECIMAL(15,2),
        other_expenses          DECIMAL(15,2),
        pbt                     DECIMAL(15,2),
        tax                     DECIMAL(15,2),
        pat                     DECIMAL(15,2),
        ebitda                  DECIMAL(15,2),
        ebitda_margin           DECIMAL(6,2),
        pat_margin              DECIMAL(6,2),
        current_ratio           DECIMAL(6,2),
        quick_ratio             DECIMAL(6,2),
        debt_equity_ratio       DECIMAL(6,2),
        icr                     DECIMAL(6,2),
        dscr                    DECIMAL(6,2),
        roe                     DECIMAL(6,2),
        roce                    DECIMAL(6,2),
        asset_turnover          DECIMAL(6,2),
        inventory_days          INT,
        debtor_days             INT,
        working_capital_days    INT,
        total_assets            DECIMAL(15,2),
        total_equity            DECIMAL(15,2),
        total_borrowings        DECIMAL(15,2),
        cash_from_operations    DECIMAL(15,2),
        cash_from_investing     DECIMAL(15,2),
        cash_from_financing     DECIMAL(15,2),
        closing_cash            DECIMAL(15,2)
    )
    """,

    # -- Audit Findings --------------------------------------
    """
    CREATE TABLE IF NOT EXISTS audit_findings (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        company_name    VARCHAR(255),
        audit_year      VARCHAR(10),
        auditor_name    VARCHAR(255),
        firm_reg_no     VARCHAR(50),
        opinion         VARCHAR(50),
        key_audit_matters   JSON,
        caro_findings       JSON,
        observations        JSON,
        summary_assessment  JSON,
        contingent_liabilities JSON,
        other_findings      JSON
    )
    """,

    # -- Raw Documents (audit report only — safety net) ------
    """
    CREATE TABLE IF NOT EXISTS raw_documents (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        company_name    VARCHAR(255),
        fiscal_year     VARCHAR(10),
        raw_text        LONGTEXT
    )
    """,

    # -- Risk Ratings (from SAVE Risk file) ------------------
    """
    CREATE TABLE IF NOT EXISTS risk_ratings (
        id                  INT AUTO_INCREMENT PRIMARY KEY,
        company_name        VARCHAR(255),
        assessment_date     VARCHAR(20),
        composite_score     DECIMAL(5,2),
        save_rating         VARCHAR(50),
        previous_rating     VARCHAR(50),
        rating_movement     VARCHAR(50),
        risk_dimensions     JSON,
        risk_flags          JSON,
        watch_list_triggers JSON
    )   
    """,

    # -- Credit Profile (replaces banking_history) -----------
    """
    CREATE TABLE IF NOT EXISTS credit_profile (
        id                  INT AUTO_INCREMENT PRIMARY KEY,
        company_name        VARCHAR(255),
        instrument_type     VARCHAR(100),
        counterparty        VARCHAR(255),
        purpose             VARCHAR(255),
        interest_rate       DECIMAL(5,2),
        outstanding_amt     DECIMAL(15,2),
        maturity_date       VARCHAR(50),
        credit_rating       VARCHAR(50),
        covenants           TEXT,
        fiscal_year         VARCHAR(10)
    )
    """,

    # -- Collateral Info -------------------------------------
    """
    CREATE TABLE IF NOT EXISTS collateral_info (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        company_name    VARCHAR(255),
        security_type   VARCHAR(255),
        description     TEXT,
        value_lakhs     DECIMAL(15,2),
        remarks         TEXT
    )
    """,

    # -- Inconsistencies -------------------------------------
    """
    CREATE TABLE IF NOT EXISTS inconsistencies (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        company_name    VARCHAR(255),
        field_name      VARCHAR(255),
        source_1        VARCHAR(100),
        value_1         TEXT,
        source_2        VARCHAR(100),
        value_2         TEXT,
        severity        ENUM('critical', 'minor')
    )
    """,

    # -- Reconciliation Results (deterministic, no LLM involved) ----
    """
    CREATE TABLE IF NOT EXISTS reconciliation_results (
        id              INT AUTO_INCREMENT PRIMARY KEY,
        company_name    VARCHAR(255),
        fiscal_year     VARCHAR(10),
        metric_name     VARCHAR(100),
        computed_value  DECIMAL(15,4),
        stored_value    DECIMAL(15,4),
        delta_pct       DECIMAL(8,4),
        status          ENUM('match', 'mismatch', 'no_data', 'unverifiable', 'computed')
    )
    """,

    # -- Task Queue ------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS task_queue (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        task_name   VARCHAR(100),
        status      ENUM('pending','in_progress','done','failed') DEFAULT 'pending',
        retries     INT DEFAULT 0,
        error_msg   TEXT
    )
    """
]


def setup_schema():
    """Creates all tables if they don't exist yet."""
    conn = get_connection()
    cursor = conn.cursor()
    for sql in SCHEMA_SQL:
        cursor.execute(sql)
    conn.commit()
    cursor.close()
    conn.close()
    print("[DB] Schema ready.")


def drop_and_recreate():
    """Wipes all tables and recreates them — use during dev only."""
    tables = [
        "task_queue", "reconciliation_results", "inconsistencies",
        "collateral_info", "credit_profile", "raw_documents",
        "risk_ratings", "audit_findings", "financials", "borrower_profile"
    ]
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
    for t in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {t}")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
    conn.commit()
    cursor.close()
    conn.close()
    setup_schema()
    print("[DB] Tables dropped and recreated.")


# ============================================================
# QUEUE HELPERS
# ============================================================

def init_queue(tasks: list):
    """Inserts initial task list into task_queue table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM task_queue")
    for task in tasks:
        cursor.execute(
            "INSERT INTO task_queue (task_name, status, retries) VALUES (%s, 'pending', 0)",
            (task,)
        )
    conn.commit()
    cursor.close()
    conn.close()


def mark_task(task_name: str, status: str, error_msg: str = None):
    """Updates a task's status in the queue."""
    conn = get_connection()
    cursor = conn.cursor()
    if error_msg:
        cursor.execute(
            "UPDATE task_queue SET status=%s, error_msg=%s WHERE task_name=%s",
            (status, error_msg, task_name)
        )
    else:
        cursor.execute(
            "UPDATE task_queue SET status=%s WHERE task_name=%s",
            (status, task_name)
        )
    conn.commit()
    cursor.close()
    conn.close()


def increment_retry(task_name: str):
    """Bumps retry count for a task."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE task_queue SET retries = retries + 1 WHERE task_name = %s",
        (task_name,)
    )
    conn.commit()
    cursor.close()
    conn.close()


def get_task_status(task_name: str) -> dict:
    """Returns status + retry count for a task."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT status, retries, error_msg FROM task_queue WHERE task_name = %s",
        (task_name,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row or {}


def all_tasks_done(tasks: list) -> bool:
    """Returns True only if every task in the list is marked done."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    fmt = ",".join(["%s"] * len(tasks))
    cursor.execute(
        f"SELECT COUNT(*) as cnt FROM task_queue WHERE task_name IN ({fmt}) AND status != 'done'",
        tasks
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row["cnt"] == 0


# ============================================================
# WRITE HELPERS
# ============================================================

def save_borrower_profile(data: dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO borrower_profile
        (company_name, cin, constitution, incorporation_year,
         registered_office, business_nature, promoters,
         key_management, shareholding)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        data.get("company_name"), data.get("cin"), data.get("constitution"),
        data.get("incorporation_year"), data.get("registered_office"),
        data.get("business_nature"),
        json.dumps(data.get("promoters", [])),
        json.dumps(data.get("key_management", [])),
        json.dumps(data.get("shareholding", {}))
    ))
    conn.commit()
    cursor.close()
    conn.close()

def normalize_fiscal_year(fy: str) -> str:
    """
    Normalizes fiscal year labels to a consistent format.
    Examples:
      "FY2025"  → "FY2025"
      "2025"    → "FY2025"
      "FY24"    → "FY2024"
      "FY22"    → "FY2022"
    """
    if not fy:
        return fy
    fy = str(fy).strip()
    # Strip FY prefix if present to get the raw number
    raw = fy.upper().lstrip("FY").strip()
    # Handle 2-digit years like 22, 23, 24
    if len(raw) == 2 and raw.isdigit():
        raw = "20" + raw
    return "FY" + raw


def save_financials(rows: list):
    """rows: list of dicts, one per fiscal year."""
    conn = get_connection()
    cursor = conn.cursor()
    for r in rows:
        r["fiscal_year"] = normalize_fiscal_year(r.get("fiscal_year", ""))
        # Skip if this company+fiscal_year already has a fuller row
        cursor.execute(
            "SELECT id FROM financials WHERE company_name=%s AND fiscal_year=%s",
            (r.get("company_name"), r.get("fiscal_year"))
        )
        existing = cursor.fetchone()
        if existing:
            # Only replace if new row has more non-null values
            cursor.execute("SELECT * FROM financials WHERE id=%s", (existing[0],))
            old = cursor.fetchone()
            old_nulls = sum(1 for v in old if v is None)
            new_nulls = sum(1 for v in r.values() if v is None)
            if new_nulls >= old_nulls:
                continue  # existing row is better or equal, skip
            cursor.execute("DELETE FROM financials WHERE id=%s", (existing[0],))
        cursor.execute("""
            INSERT INTO financials
            (company_name, fiscal_year, currency_unit, revenue, other_income, total_revenue,
             cost_of_materials, employee_expense, finance_costs, depreciation,
             other_expenses, pbt, tax, pat, ebitda, ebitda_margin, pat_margin,
             current_ratio, quick_ratio, debt_equity_ratio, icr, dscr,
             roe, roce, asset_turnover, inventory_days, debtor_days,
             working_capital_days, total_assets, total_equity, total_borrowings,
             cash_from_operations, cash_from_investing, cash_from_financing, closing_cash)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            r.get("company_name"), r.get("fiscal_year"), r.get("currency_unit"),
            r.get("revenue"), r.get("other_income"), r.get("total_revenue"),
            r.get("cost_of_materials"), r.get("employee_expense"),
            r.get("finance_costs"), r.get("depreciation"), r.get("other_expenses"),
            r.get("pbt"), r.get("tax"), r.get("pat"),
            r.get("ebitda"), r.get("ebitda_margin"), r.get("pat_margin"),
            r.get("current_ratio"), r.get("quick_ratio"), r.get("debt_equity_ratio"),
            r.get("icr"), r.get("dscr"), r.get("roe"), r.get("roce"),
            r.get("asset_turnover"), r.get("inventory_days"), r.get("debtor_days"),
            r.get("working_capital_days"), r.get("total_assets"), r.get("total_equity"),
            r.get("total_borrowings"), r.get("cash_from_operations"),
            r.get("cash_from_investing"), r.get("cash_from_financing"),
            r.get("closing_cash")
        ))
    conn.commit()
    cursor.close()
    conn.close()


def save_audit_findings(data: dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO audit_findings
        (company_name, audit_year, auditor_name, firm_reg_no, opinion,
         key_audit_matters, caro_findings, observations,
         summary_assessment, contingent_liabilities, other_findings)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        data.get("company_name"), data.get("audit_year"),
        data.get("auditor_name"), data.get("firm_reg_no"), data.get("opinion"),
        json.dumps(data.get("key_audit_matters", [])),
        json.dumps(data.get("caro_findings", [])),
        json.dumps(data.get("observations", [])),
        json.dumps(data.get("summary_assessment", {})),
        json.dumps(data.get("contingent_liabilities", [])),
        json.dumps(data.get("other_findings", []))
    ))
    conn.commit()
    cursor.close()
    conn.close()

def save_risk_ratings(data: dict):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO risk_ratings
        (company_name, assessment_date, composite_score, save_rating,
         previous_rating, rating_movement,
         risk_dimensions, risk_flags, watch_list_triggers)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        data.get("company_name"), data.get("assessment_date"),
        data.get("composite_score"), data.get("save_rating"),
        data.get("previous_rating"), data.get("rating_movement"),
        json.dumps(data.get("risk_dimensions", [])),
        json.dumps(data.get("risk_flags", [])),
        json.dumps(data.get("watch_list_triggers", []))
    ))
    conn.commit()
    cursor.close()
    conn.close()


def _clean_numeric(value):
    """Strips %, currency symbols, commas, and whitespace from a value
    that's meant to be numeric but may have arrived as a formatted string
    (e.g. '9.25%', '₹1,842.40', '1,842.40 lakhs')."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip()
    s = re.sub(r"[^\d.\-]", "", s)
    if s in ("", "-", "."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def save_credit_profile(rows: list):
    conn = get_connection()
    cursor = conn.cursor()
    for r in rows:
        cursor.execute("""
            INSERT INTO credit_profile
            (company_name, instrument_type, counterparty, purpose,
             interest_rate, outstanding_amt, maturity_date,
             credit_rating, covenants, fiscal_year)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            r.get("company_name"), r.get("instrument_type"), r.get("counterparty"),
            r.get("purpose"), _clean_numeric(r.get("interest_rate")),
            _clean_numeric(r.get("outstanding_amt")),
            r.get("maturity_date"), r.get("credit_rating"),
            r.get("covenants"), r.get("fiscal_year")
        ))
    conn.commit()
    cursor.close()
    conn.close()


def save_collateral_info(rows: list):
    conn = get_connection()
    cursor = conn.cursor()
    for r in rows:
        cursor.execute("""
            INSERT INTO collateral_info
            (company_name, security_type, description, value_lakhs, remarks)
            VALUES (%s,%s,%s,%s,%s)
        """, (
            r.get("company_name"), r.get("security_type"),
            r.get("description"), r.get("value_lakhs"), r.get("remarks")
        ))
    conn.commit()
    cursor.close()
    conn.close()

def save_raw_document(company_name: str, fiscal_year: str, raw_text: str):
    """Stores the full unprocessed audit report text, before LLM extraction
    runs. Safety net — if structured extraction misses something (e.g. an
    'advantages' section), the source text is still recoverable here."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO raw_documents (company_name, fiscal_year, raw_text)
        VALUES (%s,%s,%s)
    """, (company_name, fiscal_year, raw_text))
    conn.commit()
    cursor.close()
    conn.close()


def get_raw_document(company_name: str) -> str:
    """Returns the raw audit report text for this company."""
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT raw_text FROM raw_documents WHERE company_name=%s ORDER BY id DESC LIMIT 1",
        (company_name,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    return row["raw_text"] if row else ""


# ============================================================
# READ HELPERS — used by SGAs to query DB
# ============================================================

def get_borrower_profile(company_name: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM borrower_profile WHERE company_name = %s ORDER BY id DESC LIMIT 1",
        (company_name,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row:
        for f in ["promoters", "key_management", "shareholding"]:
            if row.get(f):
                row[f] = json.loads(row[f])
    return row or {}


def get_financials(company_name: str) -> list:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM financials WHERE company_name = %s ORDER BY fiscal_year ASC",
        (company_name,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def get_audit_findings(company_name: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM audit_findings WHERE company_name = %s ORDER BY id DESC LIMIT 1",
        (company_name,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row:
        for f in ["key_audit_matters","caro_findings","observations","summary_assessment","contingent_liabilities","other_findings"]:
            if row.get(f):
                row[f] = json.loads(row[f])
    return row or {}


def get_risk_ratings(company_name: str) -> dict:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM risk_ratings WHERE company_name = %s ORDER BY id DESC LIMIT 1",
        (company_name,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row:
        for f in ["risk_dimensions", "risk_flags", "watch_list_triggers"]:
            if row.get(f):
                row[f] = json.loads(row[f])
    return row or {}

def get_credit_profile(company_name: str) -> list:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM credit_profile WHERE company_name = %s ORDER BY fiscal_year DESC",
        (company_name,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def get_collateral_info(company_name: str) -> list:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM collateral_info WHERE company_name = %s",
        (company_name,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows

def save_inconsistencies(rows: list):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM inconsistencies WHERE company_name = %s", (rows[0].get("company_name"),))
    for r in rows:
        cursor.execute("""
            INSERT INTO inconsistencies
            (company_name, field_name, source_1, value_1, source_2, value_2, severity)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (
            r.get("company_name"), r.get("field_name"),
            r.get("source_1"), r.get("value_1"),
            r.get("source_2"), r.get("value_2"),
            r.get("severity")
        ))
    conn.commit()
    cursor.close()
    conn.close()


def get_inconsistencies(company_name: str) -> list:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM inconsistencies WHERE company_name = %s ORDER BY severity ASC",
        (company_name,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def save_reconciliation_results(rows: list):
    """Wipes prior results for this company, then inserts fresh ones.
    Called once per run from parser_node, right after financials are parsed —
    no LLM involved anywhere in this path."""
    if not rows:
        return
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM reconciliation_results WHERE company_name = %s",
        (rows[0].get("company_name"),)
    )
    for r in rows:
        cursor.execute("""
            INSERT INTO reconciliation_results
            (company_name, fiscal_year, metric_name, computed_value,
             stored_value, delta_pct, status)
            VALUES (%s,%s,%s,%s,%s,%s,%s)
        """, (
            r.get("company_name"), r.get("fiscal_year"), r.get("metric_name"),
            r.get("computed_value"), r.get("stored_value"),
            r.get("delta_pct"), r.get("status")
        ))
    conn.commit()
    cursor.close()
    conn.close()


def get_reconciliation_results(company_name: str) -> list:
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM reconciliation_results WHERE company_name = %s ORDER BY fiscal_year ASC, metric_name ASC",
        (company_name,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


if __name__ == "__main__":
    print("Setting up database schema...")
    setup_schema()
    print("Done! Check MySQL Workbench — all tables should now appear in cam_db.")