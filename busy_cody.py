# demand_minimal.py
import math
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Cutoffs (Syntetos & Boylan) ---
ADI_CUTOFF = 1.32
P_CUTOFF = 1.0 / ADI_CUTOFF        # ≈ 0.757576
CV2_CUTOFF = 0.49

def _choose_method(p: float, cv2: float, p_cut: float = P_CUTOFF, cv2_cut: float = CV2_CUTOFF) -> Tuple[str, str]:
    """Return (Category, Suggested method) using p and CV² cutoffs."""
    if pd.isna(p) or pd.isna(cv2):
        return "Insufficient data", ""
    if p <= 0:
        return "No demand", ""
    if p >= p_cut and cv2 <= cv2_cut:
        return "Smooth", "SES"
    if p >= p_cut and cv2 > cv2_cut:
        return "Erratic", "SES"
    if p < p_cut and cv2 <= cv2_cut:
        return "Intermittent", "Croston / SBA"
    return "Lumpy", "SBA"

def run_minimal(
    file_path: str,
    sheet_name: str = "classification",
    output_excel: str = "results_minimal.xlsx",
    output_plot: str = "classification_grid_p.png",
):
    # ----- Load -----
    xls = pd.ExcelFile(file_path)
    try:
        df = xls.parse(sheet_name)
    except ValueError:
        df = xls.parse(xls.sheet_names[0])

    # ----- Parse headers & periods -----
    date_cols = list(df.columns[1:])
    parsed_dates = pd.to_datetime(date_cols, errors="coerce")
    n_periods = int(parsed_dates.notna().sum()) or len(date_cols)

    # ----- Build taille / frequence & collect non-zero values -----
    combined_rows = []
    per_product_values = {}
    max_len = 0

    for _, row in df.iterrows():
        product = str(row.iloc[0])
        numeric = pd.to_numeric(row.iloc[1:], errors="coerce").fillna(0).values

        nz = numeric != 0
        vals = numeric[nz].tolist()

        # inter-arrival days from column dates
        arr_dates = parsed_dates[nz]
        if vals and arr_dates.notna().all():
            inter = pd.Series(arr_dates).diff().dropna().dt.days.tolist()
            inter_arrivals = [1] + inter
        else:
            inter_arrivals = []

        max_len = max(max_len, len(vals), len(inter_arrivals))
        combined_rows.append((product, vals, inter_arrivals))
        per_product_values[product] = vals

    # taille / frequence table
    final_rows = []
    for product, pv, ia in combined_rows:
        pv = list(pv) + [""] * (max_len - len(pv))
        ia = list(ia) + [""] * (max_len - len(ia))
        final_rows.append([product, "taille"] + pv)
        final_rows.append(["", "frequence"] + ia)
    combined_df = pd.DataFrame(final_rows, columns=["Product", "Type"] + list(range(max_len)))

    # ----- Table 1: moyenne, ecart-type, CV² (non-zero only) -----
    stats_rows = []
    for product, vals in per_product_values.items():
        if vals:
            s = pd.Series(vals, dtype="float64")
            mean = s.mean()
            std = s.std(ddof=1)          # sample std (Excel STDEV.S)
            cv2 = (std / mean) ** 2 if mean != 0 else np.nan
        else:
            mean = std = cv2 = np.nan
        stats_rows.append([product, mean, std, cv2])

    stats_df = (
        pd.DataFrame(stats_rows, columns=["Produit", "moyenne", "ecart-type", "CV^2"])
        .set_index("Produit")
        .sort_index()
    )

    # ----- Table 2: N périodes, N fréquence, p -----
    counts_rows = []
    for product, vals in per_product_values.items():
        n_freq = len(vals)
        p = (n_freq / n_periods) if n_periods else np.nan
        counts_rows.append([product, n_periods, n_freq, p])

    counts_df = (
        pd.DataFrame(counts_rows, columns=["Produit", "N périodes", "N fréquence", "p"])
        .set_index("Produit")
        .sort_index()
    )

    # ----- Methods table (what to use per product) -----
    methods_df = stats_df.join(counts_df, how="outer")
    meth = methods_df.apply(lambda r: _choose_method(r["p"], r["CV^2"]), axis=1, result_type="expand")
    methods_df["Category"] = meth[0]
    methods_df["Suggested"] = meth[1]
    methods_df = methods_df[["CV^2", "p", "Category", "Suggested"]]

    # ----- Plot: p (x) vs CV² (y) with cutoffs & labels -----
    fig, ax = plt.subplots(figsize=(8, 6))
    x = methods_df["p"].clip(lower=0, upper=1)
    y = methods_df["CV^2"]

    ax.scatter(x, y)
    for label, xi, yi in zip(methods_df.index, x, y):
        if pd.notna(xi) and pd.notna(yi):
            ax.annotate(str(label), (xi, yi), textcoords="offset points", xytext=(5, 5))

    ax.axvline(P_CUTOFF, linestyle="--")
    ax.axhline(CV2_CUTOFF, linestyle="--")
    ax.set_xlabel("p (share of non-zero periods)")
    ax.set_xlim(0, 1)
    ax.set_ylabel("CV^2")
    ax.set_title("Demand classification (p vs CV^2) — Syntetos & Boylan")
    plt.tight_layout()
    fig.savefig(output_plot, dpi=150)
    plt.close(fig)

    # ----- Write Excel: Results sheet + Methods sheet -----
    with pd.ExcelWriter(output_excel) as writer:
        # Results sheet: Table 1, blank, Table 2, blank, Combined
        sheet = "Results"
        stats_df.reset_index().to_excel(writer, index=False, sheet_name=sheet, startrow=0, startcol=0)
        r2 = len(stats_df) + 3
        counts_df.reset_index().to_excel(writer, index=False, sheet_name=sheet, startrow=r2, startcol=0)
        r3 = r2 + len(counts_df) + 3
        combined_df.to_excel(writer, index=False, sheet_name=sheet, startrow=r3, startcol=0)

        # Methods sheet: per-product coordinates + chosen method
        methods_df.reset_index().to_excel(writer, index=False, sheet_name="Methods")

    print(f"Saved: {output_excel}")
    print(f"Saved plot: {output_plot}")

if __name__ == "__main__":
    # Example usage
    run_minimal(
        file_path="articles.xlsx",         # input workbook
        sheet_name="classification",       # change if needed
        output_excel="results_minimal.xlsx",
        output_plot="classification_grid_p.png",
    )
