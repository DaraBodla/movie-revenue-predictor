from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Tuple

@dataclass
class ValidationResult:
    success: bool
    details: Dict[str, Any]

class DataValidator:
    """Lightweight validation (Great Expectations-style checks without the dependency)."""

    def validate_raw_data(self, df: pd.DataFrame) -> ValidationResult:
        required = ['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'vote_count', 'release_month']
        missing = [c for c in required if c not in df.columns]
        details: Dict[str, Any] = {
            "missing_columns": missing,
            "row_count": int(len(df)),
            "checks": {}
        }
        if missing:
            return ValidationResult(False, details)

        # Type coercion checks
        for col in required:
            details["checks"][f"{col}_non_null"] = int(df[col].notna().sum())

        # Range checks (based on business rules suggested in IMPROVEMENT_RECOMMENDATIONS)
        def between(col, lo=None, hi=None):
            s = pd.to_numeric(df[col], errors='coerce')
            ok = s.notna()
            if lo is not None: ok &= s >= lo
            if hi is not None: ok &= s <= hi
            return int(ok.sum()), int(len(s) - ok.sum())

        details["checks"]["budget_between"] = dict(zip(["pass","fail"], between("budget", 0, None)))
        details["checks"]["runtime_between"] = dict(zip(["pass","fail"], between("runtime", 30, 300)))
        details["checks"]["vote_average_between"] = dict(zip(["pass","fail"], between("vote_average", 0, 10)))
        details["checks"]["release_month_between"] = dict(zip(["pass","fail"], between("release_month", 1, 12)))
        details["checks"]["revenue_positive"] = {"pass": int((df["revenue"] > 0).sum()), "fail": int((df["revenue"] <= 0).sum())}

        # Optional uniqueness check if id exists
        if "id" in df.columns:
            details["checks"]["id_unique"] = {"pass": int(df["id"].nunique()), "total": int(len(df))}
            if df["id"].nunique() != len(df):
                return ValidationResult(False, details)

        # Fail if too many invalid rows in critical checks
        if details["checks"]["runtime_between"]["fail"] > 0:
            return ValidationResult(False, details)
        if details["checks"]["vote_average_between"]["fail"] > 0:
            return ValidationResult(False, details)
        if details["checks"]["release_month_between"]["fail"] > 0:
            return ValidationResult(False, details)
        if details["checks"]["revenue_positive"]["fail"] > 0:
            return ValidationResult(False, details)

        return ValidationResult(True, details)
