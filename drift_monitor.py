import os
import json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import config

REPORT_DIR = os.path.join("reports", "drift")
os.makedirs(REPORT_DIR, exist_ok=True)

CURRENT_PATH = os.path.join("data", "current", "live_inputs.csv")
REFERENCE_PATH = config.PROCESSED_DATA_PATH  # your training/reference distribution

FEATURES = [
    "budget", "popularity", "runtime", "vote_average", "vote_count", "release_month"
]

def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 50 or len(actual) < 50:
        return float("nan")

    quantiles = np.quantile(expected, np.linspace(0, 1, buckets + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    e_counts, _ = np.histogram(expected, bins=quantiles)
    a_counts, _ = np.histogram(actual, bins=quantiles)

    e_perc = e_counts / max(e_counts.sum(), 1)
    a_perc = a_counts / max(a_counts.sum(), 1)

    eps = 1e-6
    e_perc = np.clip(e_perc, eps, 1)
    a_perc = np.clip(a_perc, eps, 1)

    return float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))

def run():
    if not os.path.exists(CURRENT_PATH):
        raise FileNotFoundError(
            f"No live inputs found at {CURRENT_PATH}. "
            "Make at least a few /predict calls first."
        )

    ref = pd.read_csv(REFERENCE_PATH)
    cur = pd.read_csv(CURRENT_PATH)

    # keep only features that exist in both
    feats = [f for f in FEATURES if f in ref.columns and f in cur.columns]
    if not feats:
        raise ValueError("No common drift features found between reference and current datasets.")

    results = {
        "reference_path": REFERENCE_PATH,
        "current_path": CURRENT_PATH,
        "reference_rows": int(len(ref)),
        "current_rows": int(len(cur)),
        "features_checked": feats,
        "feature_drift": {},
        "overall_drift_detected": False,
    }

    # thresholds (simple + defensible)
    KS_P_THRESHOLD = 0.05
    PSI_THRESHOLD = 0.20  # >0.2 = moderate drift commonly used

    drift_flags = []

    for f in feats:
        r = pd.to_numeric(ref[f], errors="coerce").to_numpy()
        c = pd.to_numeric(cur[f], errors="coerce").to_numpy()

        # KS test
        ks_stat, ks_p = ks_2samp(r[~np.isnan(r)], c[~np.isnan(c)]) if len(c[~np.isnan(c)]) > 1 else (float("nan"), float("nan"))
        psi_val = psi(r, c)

        drift_detected = False
        if np.isfinite(ks_p) and ks_p < KS_P_THRESHOLD:
            drift_detected = True
        if np.isfinite(psi_val) and psi_val >= PSI_THRESHOLD:
            drift_detected = True

        drift_flags.append(drift_detected)

        results["feature_drift"][f] = {
            "ks_stat": float(ks_stat) if np.isfinite(ks_stat) else None,
            "ks_p_value": float(ks_p) if np.isfinite(ks_p) else None,
            "psi": float(psi_val) if np.isfinite(psi_val) else None,
            "drift_detected": bool(drift_detected),
            "ref_mean": float(np.nanmean(r)),
            "cur_mean": float(np.nanmean(c)),
        }

    results["overall_drift_detected"] = bool(any(drift_flags))

    # write JSON
    json_path = os.path.join(REPORT_DIR, "drift_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # simple HTML report
    html_path = os.path.join(REPORT_DIR, "drift_report.html")
    rows = []
    for f, d in results["feature_drift"].items():
        rows.append(
            f"<tr><td>{f}</td><td>{d['ks_p_value']}</td><td>{d['psi']}</td>"
            f"<td>{d['ref_mean']:.4f}</td><td>{d['cur_mean']:.4f}</td>"
            f"<td>{'YES' if d['drift_detected'] else 'NO'}</td></tr>"
        )
    html = f"""
    <html><head><meta charset="utf-8"><title>Drift Report</title></head>
    <body>
      <h2>Data Drift Report</h2>
      <p><b>Overall drift detected:</b> {results['overall_drift_detected']}</p>
      <p><b>Reference rows:</b> {results['reference_rows']} | <b>Current rows:</b> {results['current_rows']}</p>
      <table border="1" cellpadding="6" cellspacing="0">
        <tr><th>Feature</th><th>KS p-value</th><th>PSI</th><th>Ref mean</th><th>Cur mean</th><th>Drift?</th></tr>
        {''.join(rows)}
      </table>
    </body></html>
    """
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print("✅ Drift report generated:")
    print(" -", json_path)
    print(" -", html_path)

if __name__ == "___main__":
    run()