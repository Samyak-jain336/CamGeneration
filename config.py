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

# -- Anthropic -----------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
LLM_MODEL    = "llama-3.3-70b-versatile"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LLM_MODEL1      = "gemini-2.0-flash"   # free tier, fast

# -- Input Document Paths ------------------------------------
DOC_PATHS = {
    "annual_report": "inputs/annual_report.pdf",
    "audit_report":  "inputs/audit_report.pdf",
    "save_risk":     "inputs/save_risk_file.xlsx"  
}

# -- Output --------------------------------------------------
OUTPUT_CAM_PATH = "outputs/final_cam.docx"

# -- Queue / Retry -------------------------------------------
MAX_RETRIES = 3