# app.py
import io
import math
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# --------------------------------------------------
# Defaults
# --------------------------------------------------
ADI_CUTOFF_DEFAULT = 1.32
CV2_CUTOFF_DEFAULT = 0.49

st.set_page_config(page_title="Demand Classification — Syntetos & Boylan", layout="wide")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def classify_demand(adi: float, cv2: float, adi_cut: float, cv2_cut: float):
    """Return (Category, Suggested) using Syntetos & Boylan quadrants."""
    if pd.isna(cv2) or pd.isna(adi):
        return "Insufficient data", ""
    if math.isinf(adi) or adi == 0:
        return "No demand", ""
    if adi <= adi_cut and cv2 <= cv2_cut:
        return "Smooth", "SES"
    if adi <= adi_cut and cv2 > cv2_cut:
        return "Erratic", "SES"
    if adi > adi_cut and cv2 <= cv2_cut:
        return "Intermittent", "Croston / SBA"
    return "Lumpy", "SBA"


def compute_all(
    df: pd.DataFrame, adi_cut: float, cv2_cut: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    df: first column = product, remaining columns = dates (headers) with numeric quantities.
    Returns: combined_df, stats_df, counts_df, class_df (with both CV² variants & categories)
    """
    # Parse date headers
    date_cols = list(df.columns[1:])
    parsed_dates = pd.to_datetime(date_cols, errors="coerce")
    n_periods = int(parsed_dates.notna().sum()) or len(date_cols)

    # Build Combined & collect per-product values
    combined_rows = []
    per_product_values = {}
    max_len = 0

    for _, row in df.iterrows():
        product = str(row.iloc[0])
        numeric = pd.to_numeric(row.iloc[1:], errors="coerce").fillna(0).values

        nz = numeric != 0
        vals = numeric[nz].tolist()

        arr_dates = parsed_dates[nz]
        if vals and arr_dates.notna().all():
            inter = pd.Series(arr_dates).diff().dropna().dt.days.tolist()
            inter_arrivals = [1] + inter
        else:
            inter_arrivals = []

        max_len = max(max_len, len(vals), len(inter_arrivals))
        combined_rows.append((product, vals, inter_arrivals))
        per_product_values[product] = vals

    # Combined table (two rows per product)
    final_rows = []
    for product, pv, ia in combined_rows:
        pv = list(pv) + [""] * (max_len - len(pv))
        ia = list(ia) + [""] * (max_len - len(ia))
        final_rows.append([product, "taille"] + pv)
        final_rows.append(["", "frequence"] + ia)
    combined_df = pd.DataFrame(final_rows, columns=["Product", "Type"] + list(range(max_len)))

    # --- Summary 1 (stats) with BOTH CV² definitions
    stats_rows = []
    for product, vals in per_product_values.items():
        if vals:
            s = pd.Series(vals, dtype="float64")
            mean = s.mean()
            std = s.std(ddof=1)  # sample std (Excel STDEV.S)
            ssum = s.sum()
            cv2_sb = (std / mean) ** 2 if mean != 0 else np.nan        # S&B (recommended)
            cv2_legacy = (std / ssum) ** 2 if ssum != 0 else np.nan     # Legacy (your sheet)
        else:
            mean = std = ssum = cv2_sb = cv2_legacy = np.nan
        stats_rows.append([product, mean, std, ssum, cv2_sb, cv2_legacy])

    stats_df = (
        pd.DataFrame(
            stats_rows,
            columns=[
                "Produit",
                "moyenne_non_zero",   # TRUE mean of non-zero demands
                "ecart-type",
                "somme_non_zero",     # to mirror your earlier table
                "CV^2_S&B",           # (σ/μ)^2  <-- used for S&B thresholds
                "CV^2_legacy",        # (σ/Σx)^2 <-- what produced tiny values
            ],
        )
        .set_index("Produit")
        .sort_index()
    )

    # --- Summary 2 (counts)
    counts_rows = []
    for product, vals in per_product_values.items():
        n_freq = len(vals)
        P = (n_freq / n_periods) if n_periods else np.nan
        counts_rows.append([product, n_periods, n_freq, P])
    counts_df = (
        pd.DataFrame(counts_rows, columns=["Produit", "N périodes", "N fréquence", "P"])
        .set_index("Produit")
        .sort_index()
    )

    # --- Classification dataframe
    class_df = stats_df.join(counts_df, how="outer")
    class_df["ADI"] = class_df.apply(
        lambda r: (r["N périodes"] / r["N fréquence"])
        if pd.notna(r["N fréquence"]) and r["N fréquence"] not in (0, None) and r["N fréquence"] != 0
        else np.inf,
        axis=1,
    )

    # Categories for BOTH CV² variants
    res_sb = class_df.apply(lambda r: classify_demand(r["ADI"], r["CV^2_S&B"], adi_cut, cv2_cut), axis=1, result_type="expand")
    class_df["Category_S&B"] = res_sb[0]
    class_df["Suggested_S&B"] = res_sb[1]

    res_legacy = class_df.apply(lambda r: classify_demand(r["ADI"], r["CV^2_legacy"], adi_cut, cv2_cut), axis=1, result_type="expand")
    class_df["Category_Legacy"] = res_legacy[0]
    class_df["Suggested_Legacy"] = res_legacy[1]

    return combined_df, stats_df, counts_df, class_df


def make_plot(class_df: pd.DataFrame, adi_cut: float, cv2_cut: float, cv2_col: str):
    """Return Matplotlib figure for ADI vs chosen CV² column."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = class_df["ADI"].replace(np.inf, np.nan)
    y = class_df[cv2_col]

    ax.scatter(x, y)
    for label, xi, yi in zip(class_df.index, x, y):
        if pd.notna(xi) and pd.notna(yi):
            ax.annotate(str(label), (xi, yi), textcoords="offset points", xytext=(5, 5))

    ax.axvline(adi_cut, linestyle="--")
    ax.axhline(cv2_cut, linestyle="--")
    ax.set_xlabel("ADI (Average inter-demand interval)")
    ax.set_ylabel(f"{cv2_col} value")
    ax.set_title("Syntetos & Boylan Demand Classification")
    fig.tight_layout()
    return fig


def excel_bytes(
    combined_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    class_df: pd.DataFrame,
    selected_cols: Tuple[str, str],
    one_sheet: bool = True,
) -> io.BytesIO:
    """Build an Excel in memory. selected_cols=(Category_selected, Suggested_selected)."""
    sel_cat_col, sel_sug_col = selected_cols
    export_class = class_df.copy()
    export_class.insert(0, "Category_Selected", export_class.pop(sel_cat_col))
    export_class.insert(1, "Suggested_Selected", export_class.pop(sel_sug_col))

    buf = io.BytesIO()
    for engine in ("openpyxl", "xlsxwriter", None):
        try:
            writer = pd.ExcelWriter(buf, engine=engine) if engine else pd.ExcelWriter(buf)
            with writer:
                if one_sheet:
                    sheet = "Results"
                    export_class.reset_index().to_excel(writer, index=False, sheet_name=sheet, startrow=0, startcol=0)
                    r2 = len(export_class) + 3
                    combined_df.to_excel(writer, index=False, sheet_name=sheet, startrow=r2, startcol=0)
                else:
                    combined_df.to_excel(writer, index=False, sheet_name="Combined")
                    stats_df.reset_index().to_excel(writer, index=False, sheet_name="Summary", startrow=0)
                    start_row_2 = len(stats_df) + 3
                    counts_df.reset_index().to_excel(writer, index=False, sheet_name="Summary", startrow=start_row_2)
                    export_class.reset_index().to_excel(writer, index=False, sheet_name="Classification")
            break
        except ModuleNotFoundError:
            buf = io.BytesIO()
            continue
    buf.seek(0)
    return buf

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Demand Classification — Syntetos & Boylan")

with st.sidebar:
    st.header("Settings")
    adi_cut = st.number_input("ADI cut-off", value=float(ADI_CUTOFF_DEFAULT), format="%.4f")
    cv2_cut = st.number_input("CV² cut-off", value=float(CV2_CUTOFF_DEFAULT), format="%.4f")
    cv2_choice = st.radio(
        "CV² definition for plot & 'selected' category",
        options=("S&B (σ/μ)^2 — recommended", "Legacy (σ/Σx)^2 — for comparison"),
        index=0,
    )
    cv2_col = "CV^2_S&B" if cv2_choice.startswith("S&B") else "CV^2_legacy"
    cat_col = "Category_S&B" if cv2_col == "CV^2_S&B" else "Category_Legacy"
    sug_col = "Suggested_S&B" if cv2_col == "CV^2_S&B" else "Suggested_Legacy"
    one_sheet = st.checkbox("Write everything on ONE sheet", value=True)

uploaded = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
sheet_name = None

if uploaded is not None:
    try:
        xls = pd.ExcelFile(uploaded)
        default = "classification"
        options = xls.sheet_names
        # try to preselect "classification" if present (case-insensitive)
        idx = 0
        for i, s in enumerate(options):
            if s.lower() == default:
                idx = i
                break
        sheet_name = st.selectbox("Choose sheet", options=options, index=idx)
    except Exception as e:
        st.error(f"Could not read the workbook: {e}")

if uploaded is not None and sheet_name is not None:
    try:
        df_raw = pd.read_excel(uploaded, sheet_name=sheet_name)
        st.success(f"Loaded sheet: {sheet_name}")

        with st.expander("Preview input data"):
            st.dataframe(df_raw.head(20), use_container_width=True)

        combined_df, stats_df, counts_df, class_df = compute_all(df_raw, adi_cut=adi_cut, cv2_cut=cv2_cut)

        # Selected columns (based on CV2 choice)
        class_df["Category_Selected"] = class_df[cat_col]
        class_df["Suggested_Selected"] = class_df[sug_col]

        st.subheader("Classification (includes both CV² versions)")
        st.caption("Columns 'Category_Selected' and 'Suggested_Selected' reflect your CV² choice in the sidebar.")
        st.dataframe(class_df.reset_index(), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Table 1 — Stats & CV²**")
            st.dataframe(stats_df.reset_index(), use_container_width=True)
        with c2:
            st.markdown("**Table 2 — N périodes / N fréquence / P**")
            st.dataframe(counts_df.reset_index(), use_container_width=True)

        st.subheader("Combined table (taille / frequence)")
        st.dataframe(combined_df, use_container_width=True)

        st.subheader(f"Syntetos & Boylan grid (ADI vs {cv2_col})")
        fig = make_plot(class_df, adi_cut=adi_cut, cv2_cut=cv2_cut, cv2_col=cv2_col)
        st.pyplot(fig, use_container_width=True)

        # Downloads
        st.markdown("### Downloads")
        excel_buf = excel_bytes(
            combined_df=combined_df,
            stats_df=stats_df,
            counts_df=counts_df,
            class_df=class_df,
            selected_cols=("Category_Selected", "Suggested_Selected"),
            one_sheet=one_sheet,
        )
        st.download_button(
            "Download Excel",
            data=excel_buf,
            file_name="results_with_classification.xlsx" if one_sheet else "results_multi_sheet.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", bbox_inches="tight")
        png_buf.seek(0)
        st.download_button("Download plot (PNG)", data=png_buf, file_name="classification_grid.png", mime="image/png")

        st.caption(
            "Notes: CV²_S&B uses mean of non-zero demands (recommended). "
            "CV²_legacy uses sum of non-zero demands (for comparison). "
            "ADI = N périodes / N fréquence (∞ if no demand)."
        )

    except Exception as e:
        st.error(f"Processing failed: {e}")
else:
    st.info(
        "Upload an Excel file to get started. The first column must be the product name; "
        "remaining columns should be date headers with numeric quantities (0 means no purchase)."
    )
