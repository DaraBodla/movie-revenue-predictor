
"""Lightweight data validation script (DeepChecks-like) without external dependency.

This keeps the container runnable even if the 'deepchecks' library is not installed.
It performs basic schema and sanity checks and writes a JSON report to /app/reports.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

import config

REQUIRED_COLUMNS = ['budget','popularity','runtime','vote_average','vote_count','revenue']


def validate(df: pd.DataFrame) -> dict:
    report = {
        "timestamp": datetime.now().isoformat(),
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "checks": {},
        "status": "pass"
    }

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report["checks"]["required_columns"] = {"missing": missing, "pass": len(missing) == 0}
    if missing:
        report["status"] = "fail"
        return report

    # Basic numeric sanity
    def frac(cond):
        return float(cond.mean()) if len(df) else 0.0

    report["checks"]["non_negative_budget"] = {"pass": bool((df["budget"] >= 0).all()), "fraction_negative": frac(df["budget"] < 0)}
    report["checks"]["positive_runtime"] = {"pass": bool((df["runtime"] > 0).all()), "fraction_non_positive": frac(df["runtime"] <= 0)}
    report["checks"]["vote_average_range"] = {"pass": bool(((df["vote_average"] >= 0) & (df["vote_average"] <= 10)).all()),
                                              "fraction_out_of_range": frac((df["vote_average"] < 0) | (df["vote_average"] > 10))}
    report["checks"]["positive_revenue"] = {"pass": bool((df["revenue"] > 0).all()), "fraction_non_positive": frac(df["revenue"] <= 0)}

    # If any critical check fails, mark fail
    critical = ["non_negative_budget", "positive_runtime", "vote_average_range", "positive_revenue"]
    if any(not report["checks"][k]["pass"] for k in critical):
        report["status"] = "fail"

    return report


def main():
    raw_path = config.RAW_DATA_PATH
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}. Place movies_clean.csv into data/raw/.")

    df = pd.read_csv(raw_path)
    report = validate(df)

    out = Path(config.REPORTS_DIR) / "validation_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validation report written to {out}")
    print(f"Validation status: {report['status']}")


if __name__ == "__main__":
    main()
