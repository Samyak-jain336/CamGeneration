from groq import Groq
from openai import OpenAI
from google import genai
import time

from config import (
    GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3,
    GROQ_API_KEY_4, GROQ_API_KEY_5, GROQ_API_KEY_6,
    LLM_MODEL, OUTPUT_CAM_PATH,
    OPENAI_API_KEY, OPENAI_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL
)

from prompts import (
    OVERVIEW_PROMPT,
    FINANCIAL_PROMPT,
    CREDIT_PROFILE_PROMPT,
    RISK_PROMPT,
    RECOMMENDATION_PROMPT
)

from db import (
    get_borrower_profile,
    get_financials,
    get_audit_findings,
    get_risk_ratings,
    get_credit_profile,
    get_collateral_info
)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client_1 = Groq(api_key=GROQ_API_KEY)
groq_client_2 = Groq(api_key=GROQ_API_KEY_2)
groq_client_3 = Groq(api_key=GROQ_API_KEY_3)
groq_client_4 = Groq(api_key=GROQ_API_KEY_4)
groq_client_5 = Groq(api_key=GROQ_API_KEY_5)
groq_client_6 = Groq(api_key=GROQ_API_KEY_6)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def _call_gemini(prompt: str) -> str:
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={"temperature": 0.2}
    )
    return response.text

def _call_groq(client, prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content


def _call_openai(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content


GEMINI_MAX_RETRIES = 5

def call_sga_llm(prompt: str) -> str:
    """
    Gemini-only. Retries on failure (e.g. 503 UNAVAILABLE, rate limit)
    up to GEMINI_MAX_RETRIES times with increasing backoff.
    """
    last_error = None
    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            print(f"  [SGA] Trying Gemini (attempt {attempt}/{GEMINI_MAX_RETRIES})...")
            result = _call_gemini(prompt)
            print(f"  [SGA] Gemini succeeded.")
            return result
        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in
                   ["503", "unavailable", "overloaded"]):
                print(f"  [SGA] Gemini unavailable (attempt {attempt}/{GEMINI_MAX_RETRIES})...")
            elif any(kw in error_str for kw in
                   ["rate limit", "rate_limit", "429", "quota",
                    "too many requests", "resource_exhausted", "exceeded"]):
                print(f"  [SGA] Gemini rate limited (attempt {attempt}/{GEMINI_MAX_RETRIES})...")
            else:
                print(f"  [SGA] Gemini error (attempt {attempt}/{GEMINI_MAX_RETRIES}): {e}")
            last_error = e
            if attempt < GEMINI_MAX_RETRIES:
                backoff = 5 * attempt
                print(f"  [SGA] Retrying in {backoff}s...")
                time.sleep(backoff)
            continue

    raise Exception(f"Gemini failed after {GEMINI_MAX_RETRIES} attempts. Last error: {last_error}")

class SectionGenerationAgent:

    def __init__(self):
        pass

    # Each task_type only receives the tables it actually needs.
    # This mirrors the parser's per-category prompt split: less
    # irrelevant data per call means the LLM focuses fully on what
    # matters for that specific section instead of sifting through
    # six tables' worth of unrelated fields.
    CONTEXT_MAP = {
        "generate_overview": ["borrower_profile"],
        "generate_financial": ["financials"],
        "generate_credit_profile": ["credit_profile", "financials"],
        "generate_risk": ["risk_ratings", "collateral_info", "audit_findings"],
        "generate_recommendation": ["financials", "risk_ratings", "credit_profile", "audit_findings"],
    }

    TABLE_FETCHERS = {
        "borrower_profile": get_borrower_profile,
        "financials": get_financials,
        "audit_findings": get_audit_findings,
        "risk_ratings": get_risk_ratings,
        "credit_profile": get_credit_profile,
        "collateral_info": get_collateral_info,
    }

    def build_context(self, company_name: str, task_type: str):
        needed_tables = self.CONTEXT_MAP.get(task_type, list(self.TABLE_FETCHERS.keys()))

        context = {
            table_name: self.TABLE_FETCHERS[table_name](company_name)
            for table_name in needed_tables
        }

        return context

    def generate(
        self,
        company_name: str,
        task_type: str
    ):

        context = self.build_context(company_name, task_type)

        prompt_map = {
            "generate_overview": OVERVIEW_PROMPT,
            "generate_financial": FINANCIAL_PROMPT,
            "generate_credit_profile": CREDIT_PROFILE_PROMPT,
            "generate_risk": RISK_PROMPT,
            "generate_recommendation": RECOMMENDATION_PROMPT
        }

        if task_type not in prompt_map:
            raise ValueError(f"Unsupported task type: {task_type}")

        prompt = prompt_map[task_type].format(
            company_name=company_name,
            context=context
        )

        return call_sga_llm(prompt)