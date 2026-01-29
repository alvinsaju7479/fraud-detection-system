from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import PROCESSED_DIR, REPORTS_DIR, TARGET_COL
from src.utils.logger import get_logger

logger = get_logger("drift")


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index (PSI)
    Rule of thumb:
      < 0.10  : no significant drift
      0.10-0.25: moderate drift
      > 0.25  : significant drift
    """
    expected = expected.astype(float)
    actual = actual.astype(float)

    # remove nan/inf
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 2 or len(actual) < 2:
        return float("nan")

    # quantile bins based on expected distribution (train)
    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(expected, quantiles))
    if len(cut_points) <= 2:
        return float("nan")

    # histogram counts
    e_cnt, _ = np.histogram(expected, bins=cut_points)
    a_cnt, _ = np.histogram(actual, bins=cut_points)

    e_pct = e_cnt / max(e_cnt.sum(), 1)
    a_pct = a_cnt / max(a_cnt.sum(), 1)

    # avoid log(0)
    eps = 1e-6
    e_pct = np.clip(e_pct, eps, 1)
    a_pct = np.clip(a_pct, eps, 1)

    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def _drift_label(psi_value: float) -> str:
    if not np.isfinite(psi_value):
        return "N/A"
    if psi_value < 0.10:
        return "Low"
    if psi_value < 0.25:
        return "Medium"
    return "High"


def build_report(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    features = [c for c in train_df.columns if c != TARGET_COL]
    rows = []

    for col in features:
        tr = train_df[col].to_numpy()
        te = test_df[col].to_numpy()

        psi_val = _psi(tr, te, bins=10)

        # KS test (distribution difference). For huge samples p-value can be tiny.
        tr_clean = tr[np.isfinite(tr)]
        te_clean = te[np.isfinite(te)]
        if len(tr_clean) > 1 and len(te_clean) > 1:
            ks = ks_2samp(tr_clean, te_clean)
            ks_stat = float(ks.statistic)
            ks_p = float(ks.pvalue)
        else:
            ks_stat, ks_p = float("nan"), float("nan")

        rows.append({
            "feature": col,
            "psi": psi_val,
            "psi_level": _drift_label(psi_val),
            "ks_stat": ks_stat,
            "ks_pvalue": ks_p,
            "train_mean": float(np.nanmean(tr)),
            "test_mean": float(np.nanmean(te)),
            "train_std": float(np.nanstd(tr)),
            "test_std": float(np.nanstd(te)),
        })

    report_df = pd.DataFrame(rows).sort_values(["psi"], ascending=False)
    return report_df


def save_html(report_df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # simple HTML styling
    def color_level(level: str) -> str:
        if level == "High":
            return "background:#ffcccc;"
        if level == "Medium":
            return "background:#fff2cc;"
        if level == "Low":
            return "background:#d9ead3;"
        return ""

    styled_rows = []
    for _, r in report_df.iterrows():
        style = color_level(r["psi_level"])
        styled_rows.append(
            f"<tr style='{style}'>"
            f"<td>{r['feature']}</td>"
            f"<td>{r['psi']:.4f}</td>"
            f"<td>{r['psi_level']}</td>"
            f"<td>{r['ks_stat']:.4f}</td>"
            f"<td>{r['ks_pvalue']:.2e}</td>"
            f"<td>{r['train_mean']:.4f}</td>"
            f"<td>{r['test_mean']:.4f}</td>"
            f"<td>{r['train_std']:.4f}</td>"
            f"<td>{r['test_std']:.4f}</td>"
            f"</tr>"
        )

    high = int((report_df["psi_level"] == "High").sum())
    med = int((report_df["psi_level"] == "Medium").sum())
    low = int((report_df["psi_level"] == "Low").sum())

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Data Drift Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 6px; }}
    .meta {{ margin-bottom: 18px; color:#333; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; font-size: 13px; }}
    th {{ background: #f2f2f2; text-align: left; }}
    .badge {{ display:inline-block; padding:4px 10px; border-radius: 999px; margin-right:8px; }}
    .low {{ background:#d9ead3; }}
    .med {{ background:#fff2cc; }}
    .high {{ background:#ffcccc; }}
    code {{ background:#f6f8fa; padding:2px 6px; border-radius:6px; }}
  </style>
</head>
<body>
  <h1>Data Drift Report (PSI + KS Test)</h1>
  <div class="meta">
    <div>
      <span class="badge low">Low: {low}</span>
      <span class="badge med">Medium: {med}</span>
      <span class="badge high">High: {high}</span>
    </div>
    <p>
      PSI interpretation: <code>&lt;0.10</code> Low, <code>0.10–0.25</code> Medium, <code>&gt;0.25</code> High drift.
      KS p-value can be very small for large datasets; use PSI as primary signal.
    </p>
  </div>

  <table>
    <thead>
      <tr>
        <th>Feature</th>
        <th>PSI</th>
        <th>PSI Level</th>
        <th>KS Stat</th>
        <th>KS p-value</th>
        <th>Train Mean</th>
        <th>Test Mean</th>
        <th>Train Std</th>
        <th>Test Std</th>
      </tr>
    </thead>
    <tbody>
      {''.join(styled_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main():
    train_path = PROCESSED_DIR / "train.parquet"
    test_path = PROCESSED_DIR / "test.parquet"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run: python -m src.data.make_dataset and python -m src.data.split first")

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    report_df = build_report(train, test)

    out_dir = REPORTS_DIR / "evidently"   # keep same folder name
    out_file = out_dir / "drift_report.html"
    save_html(report_df, out_file)

    logger.info(f"Saved drift report -> {out_file}")
    logger.info(f"Top drift features:\n{report_df[['feature','psi','psi_level']].head(8).to_string(index=False)}")


if __name__ == "__main__":
    main()
