# ============================================================
# config.py
# Central config — fill in your MySQL credentials here
# ============================================================
import os
from dotenv import load_dotenv
load_dotenv()
# -- MySQL Connection ----------------------------------------
DB_CONFIG = {
    "host":     "localhost",
    "port":     3306,
    "user":     "root",
    "password": os.getenv("DB_PASSWORD"),
    "database": "cam_db"
}

# -- LLM Providers -------------------------------------------
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2")
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3")
GROQ_API_KEY_4 = os.getenv("GROQ_API_KEY_4")
GROQ_API_KEY_5 = os.getenv("GROQ_API_KEY_5")
GROQ_API_KEY_6 = os.getenv("GROQ_API_KEY_6")
GROQ_MODEL     = "llama-3.3-70b-versatile"
LLM_MODEL = GROQ_MODEL

OLLAMA_MODEL   = "llama3.1:8b"   # must be pulled locally via: ollama pull llama3.1:8b
OLLAMA_URL     = "http://localhost:11434/api/generate"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL   = "gpt-4o-mini"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = "gemini-2.5-flash"

# -- Input Document Paths ------------------------------------
'''DOC_PATHS = {
    "annual_report": "inputs1/Passionfruit_Inc_Annual_Report_10K_FY2025.pdf",
    "audit_report":  "inputs1/Passionfruit_Inc_Audit_Report_FY2025.pdf",
    "save_risk":     "inputs1/Passionfruit_Inc_SAVE_Risk_FY2025.xlsx"  
}
'''
'''DOC_PATHS = {
    "annual_report": "NewInputs/annualReport.pdf",
    "audit_report":  "NewInputs/auditReport.pdf",
    "save_risk":     "NewInputs/saveRisk.xlsx"  
}'''
'''
DOC_PATHS = {
    "annual_report": "Sri_Ramakrishna_Mills_Inputs/annual-report-for-the-year-2023-24.pdf",
    "audit_report":  "Sri_Ramakrishna_Mills_Inputs/AuditReport.pdf",
    "save_risk":     "Sri_Ramakrishna_Mills_Inputs/saveRisk_SriRamakrishnaMills.xlsx"  
}'''

DOC_PATHS = {
    "annual_report": "durlax/Annual_Report_2024-25.pdf",
    "audit_report":  "durlax/Audit_Report_2024-25.pdf",
    "save_risk":     "durlax/Durlax_Top_Surface_SAVE_Risk.xlsx"  
}

'''
DOC_PATHS = {
    "annual_report": "inputs/annual_report.pdf",
    "audit_report":  "inputs/audit_report.pdf",
    "save_risk":     "inputs/save_risk_file.xlsx"  
}'''

# -- Output --------------------------------------------------
OUTPUT_CAM_PATH = "outputs/durlax_cam.docx"

# -- Queue / Retry -------------------------------------------
MAX_RETRIES = 3