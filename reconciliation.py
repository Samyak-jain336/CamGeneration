# reconciliation.py
#
# Deterministic, LLM-free reconciliation layer.
# Runs after all parser agents finish, before SGAs run.
# No prompts, no LLM client, no fallback chain — pure arithmetic.
#
# Placeholder field names below are marked with TODO — swap in your
# real db.py column names before running.

from typing import Optional

TOLERANCE_PCT = 0.02  # 2% — treat anything inside this band as a match


def _safe_div(numerator, denominator) -> Optional[float]:
    """Returns None instead of raising on missing/zero data, so a gap
    in source data produces status='no_data' rather than a crash."""
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _compare(computed: Optional[float], stored: Optional[float]) -> dict:
    if computed is None or stored is None:
        return {"status": "no_data", "computed_value": computed, "stored_value": stored, "delta_pct": None}
    if stored == 0:
        delta_pct = None
    else:
        delta_pct = abs(computed - stored) / abs(stored)
    status = "match" if (delta_pct is not None and delta_pct <= TOLERANCE_PCT) else "mismatch"
    return {"status": status, "computed_value": computed, "stored_value": stored, "delta_pct": delta_pct}


def reconcile_ratios(fy_row: dict) -> list[dict]:
    """fy_row = one row from `financials` for a single fiscal year.
    TODO: confirm these keys exist in your financials table — if your
    schema only stores the ratio itself (no raw components), this
    check degrades gracefully to status='no_data', which is fine."""
    results = []

    current_ratio_computed = _safe_div(
        fy_row.get("current_assets"), fy_row.get("current_liabilities")
    )
    results.append({
        "metric_name": "current_ratio",
        **_compare(current_ratio_computed, fy_row.get("current_ratio")),
    })

    de_ratio_computed = _safe_div(
        fy_row.get("total_debt"), fy_row.get("total_equity")
    )
    results.append({
        "metric_name": "debt_equity_ratio",
        **_compare(de_ratio_computed, fy_row.get("debt_equity_ratio")),
    })

    icr_computed = _safe_div(
        fy_row.get("ebit"), fy_row.get("interest_expense")
    )
    results.append({
        "metric_name": "interest_coverage_ratio",
        **_compare(icr_computed, fy_row.get("icr")),
    })

    # EBITDA margin / PAT margin internal tie-out (uses revenue you already trust)
    ebitda_margin_computed = _safe_div(fy_row.get("ebitda"), fy_row.get("revenue"))
    results.append({
        "metric_name": "ebitda_margin",
        **_compare(ebitda_margin_computed, fy_row.get("ebitda_margin")),
    })

    pat_margin_computed = _safe_div(fy_row.get("pat"), fy_row.get("revenue"))
    results.append({
        "metric_name": "pat_margin",
        **_compare(pat_margin_computed, fy_row.get("pat_margin")),
    })

    return results


def reconcile_yoy(financials: list[dict]) -> list[dict]:
    """financials = full list of fiscal-year rows for one company,
    sorted ascending by fiscal year. Computes YoY deltas in code so
    the SGA never has to do subtraction/division in its own head."""
    results = []
    sorted_rows = sorted(financials, key=lambda r: r.get("fiscal_year", ""))

    for prev_row, curr_row in zip(sorted_rows, sorted_rows[1:]):
        for metric in ("revenue", "ebitda", "pat"):
            prev_val = prev_row.get(metric)
            curr_val = curr_row.get(metric)
            yoy_pct = None
            if prev_val not in (None, 0) and curr_val is not None:
                yoy_pct = (curr_val - prev_val) / prev_val

            results.append({
                "fiscal_year": curr_row.get("fiscal_year"),
                "metric_name": f"{metric}_yoy_growth",
                "computed_value": yoy_pct,
                "stored_value": None,   # nothing to compare against — this IS the source of truth
                "delta_pct": None,
                "status": "computed" if yoy_pct is not None else "no_data",
            })

    return results


def run_reconciliation(company_name: str, financials: list[dict]) -> list[dict]:
    """Entry point called once from the graph, after parsing, before SGAs.
    Returns a flat list of row-dicts ready for a save_reconciliation_results()
    call in db.py — schema: company_name, fiscal_year, metric_name,
    computed_value, stored_value, delta_pct, status."""
    all_results = []

    for fy_row in financials:
        for r in reconcile_ratios(fy_row):
            r["company_name"] = company_name
            r["fiscal_year"] = fy_row.get("fiscal_year")
            all_results.append(r)

    for r in reconcile_yoy(financials):
        r["company_name"] = company_name
        all_results.append(r)

    return all_results
