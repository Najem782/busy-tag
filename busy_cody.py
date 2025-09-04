import io
import math
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ------------------ Defaults (you can change in the UI) ------------------
ADI_CUTOFF_DEFAULT = 1.32
CV2_CUTOFF_DEFAULT = 0.49

st.set_page_config(page_title="Demand Classification (Syntetos & Boylan)", layout="wide")

# ------------------ Core computations ------------------
def compute_all(
    df: pd.DataFrame,
    adi_cut: float = ADI_CUTOFF_DEFAULT,
    cv2_cut: float = CV2_CUTOFF_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    df: first column is product, remaining columns are dates (headers) with numeric quantities.
    Returns: combined_df, stats_df, counts_df, class_df
    """

    # Parse date headers (columns after the first)
    date_cols = list(df.columns[1:])
    parsed_dates = pd.to_datetime(date_cols, errors="coerce")
    n_periods = int(parsed_dates.notna().sum()) or len(date_cols)

    combined_rows = []
    per_product_values = {}
    max_len = 0

    for _, row in df.iterrows():
        product = str(row.iloc[0])
        numeric = pd.to_numeric(row.iloc[1:], errors="coerce").fillna(0).values

        nz = numeric != 0
        purchase_values = numeric[nz].tolist()

        arrival_dates = parsed_dates[nz]
        if purchase_values and arrival_dates.notna().all():
            inter = pd.Series(arrival_dates).diff().dropna().dt.days.tolist()
            inter_arrivals = [1] + inter  # convention: first interval = 1
        else:
            inter_arrivals = []

        max_len = max(max_len, len(purchase_values), len(inter_arrivals))
        combined_rows.append((product, purchase_values, inter_arrivals))
        per_product_values[product] = purchase_values

    # Combined (two rows per product)
    final_rows = []
    for product, pv, ia in combined_rows:
        pv = list(pv) + [""] * (max_len - len(pv))
        ia = list(ia) + [""] * (max_len - len(ia))
        final_rows.append([product, "taille"] + pv)
        final_rows.append(["", "frequence"] + ia)
    combined_df = pd.DataFrame(final_rows, columns=["Product", "Type"] + list(range(max_len)))

    # Summary 1: mean, std, CV^2
    stats_rows = []
    for product, vals in per_product_values.items():
        if vals:
            s = pd.Series(vals, dtype="float64")
            mean = s.mean()
            std = s.std(ddof=1)  # sample std (Excel STDEV.S)
            cv2 = (std / mean) ** 2 if mean != 0 else np.nan
        else:
            mean = std = cv2 = np.nan
        stats_rows.append([product, mean, std, cv2])
    stats_df = (
        pd.DataFrame(stats_rows, columns=["Produit", "moyenne", "ecart-type", "CV^2"])
        .set_index("Produit")
        .sort_index()
    )

    # Summary 2: counts and P
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

    # Classification table (ADI & category)
    class_df = stats_df.join(counts_df, how="outer")
    class_df["ADI"] = class_df.apply(
        lambda r: (r["N périodes"] / r["N fréquence"])
        if pd.notna(r["N fréquence"]) and r["N fréquence"] not in (0, None)
        else np.inf,
        axis=1,
    )

    def classify(adi, cv2, adi_cut, cv2_cut):
        if pd.isna(cv2) or pd.isna(adi):
            return "Insufficient data", ""
        if math.isinf(adi):
            return "No demand", ""
        if adi <= adi_cut and cv2 <= cv2_cut:
            return "Smooth", "SES"
        if adi <= adi_cut and cv2 > cv2_cut:
            return "Erratic", "SES"
        if adi > adi_cut and cv2 <= cv2_cut:
            return "Intermittent", "Croston / SBA"
        return "Lumpy", "SBA"

    res = class_df.apply(lambda r: classify(r["ADI"], r["CV^2"], adi_cut, cv2_cut), axis=1, result_type="expand")
    class_df["Category"] = res[0]
    class_df["Suggested"] = res[1]

    return combined_df, stats_df, counts_df, class_df


def make_plot(class_df: pd.DataFrame, adi_cut: float, cv2_cut: float):
    """Return a Matplotlib Figure with the Syntetos & Boylan grid."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = class_df["ADI"].replace(np.inf, np.nan)
    y = class_df["CV^2"]

    ax.scatter(x, y)
    for label, xi, yi in zip(class_df.index, x, y):
        if pd.notna(xi) and pd.notna(yi):
            ax.annotate(str(label), (xi, yi), textcoords="offset points", xytext=(5, 5))

    ax.axvline(adi_cut, linestyle="--")
    ax.axhline(cv2_cut, linestyle="--")
    ax.set_xlabel("ADI (Average inter-demand interval)")
    ax.set_ylabel("CV^2 (coefficient of variation squared)")
    ax.set_title("Syntetos & Boylan Demand Classification")
    fig.tight_layout()
    return fig


def excel_bytes(
    combined_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    class_df: pd.DataFrame,
    one_sheet: bool = True,
) -> io.BytesIO:
    """Build an Excel file in-memory and return a BytesIO."""
    buf = io.BytesIO()
    # Try openpyxl -> xlsxwriter -> default
    for engine in ("openpyxl", "xlsxwriter", None):
        try:
            writer = pd.ExcelWriter(buf, engine=engine) if engine else pd.ExcelWriter(buf)
            with writer:
                if one_sheet:
                    sheet = "Results"
                    class_df.reset_index().to_excel(writer, index=False, sheet_name=sheet, startrow=0, startcol=0)
                    r2 = len(class_df) + 3
                    combined_df.to_excel(writer, index=False, sheet_name=sheet, startrow=r2, startcol=0)
                else:
                    combined_df.to_excel(writer, index=False, sheet_name="Combined")
                    stats_df.reset_index().to_excel(writer, index=False, sheet_name="Summary", startrow=0, startcol=0)
                    start_row_2 = len(stats_df) + 3
                    counts_df.reset_index().to_excel(writer, index=False, sheet_name="Summary", startrow=start_row_2, startcol=0)
                    class_df.reset_index().to_excel(writer, index=False, sheet_name="Classification")
            break
        except ModuleNotFoundError:
            buf = io.BytesIO()
            continue
    buf.seek(0)
    return buf


# ------------------ Streamlit UI ------------------
st.title("Demand Classification — Syntetos & Boylan")

with st.sidebar:
    st.header("Settings")
    adi_cut = st.number_input("ADI cut-off", value=float(ADI_CUTOFF_DEFAULT), format="%.4f")
    cv2_cut = st.number_input("CV² cut-off", value=float(CV2_CUTOFF_DEFAULT), format="%.4f")
    one_sheet = st.checkbox("Write everything on ONE sheet", value=True)
    st.caption("By default, results go to one 'Results' sheet (classification on top, combined below).")

uploaded = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

sheet_name = None
if uploaded is not None:
    try:
        xls = pd.ExcelFile(uploaded)
        # Try to default to a sheet named 'classification' if present
        default_sheet = "classification" if "classification" in [s.lower() for s in xls.sheet_names] else xls.sheet_names[0]
        sheet_name = st.selectbox("Choose sheet", options=xls.sheet_names, index=xls.sheet_names.index(default_sheet if default_sheet in xls.sheet_names else xls.sheet_names[0]))
    except Exception as e:
        st.error(f"Could not read the workbook: {e}")

if uploaded is not None and sheet_name is not None:
    try:
        df_raw = pd.read_excel(uploaded, sheet_name=sheet_name)
        st.success(f"Loaded sheet: {sheet_name}")

        # Show a peek of the raw data
        with st.expander("Preview input data"):
            st.dataframe(df_raw.head(20), use_container_width=True)

        combined_df, stats_df, counts_df, class_df = compute_all(df_raw, adi_cut=adi_cut, cv2_cut=cv2_cut)

        # Display classification & suggestions
        st.subheader("Classification (CV², ADI, Category, Suggested)")
        st.dataframe(class_df.reset_index(), use_container_width=True)

        # Two summary tables
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Table 1 — moyenne / écart-type / CV²**")
            st.dataframe(stats_df.reset_index(), use_container_width=True)
        with c2:
            st.markdown("**Table 2 — N périodes / N fréquence / P**")
            st.dataframe(counts_df.reset_index(), use_container_width=True)

        # Combined table
        st.subheader("Combined table (taille / frequence)")
        st.dataframe(combined_df, use_container_width=True)

        # Plot
        st.subheader("Syntetos & Boylan grid (ADI vs CV²)")
        fig = make_plot(class_df, adi_cut=adi_cut, cv2_cut=cv2_cut)
        st.pyplot(fig, use_container_width=True)

        # Downloads
        st.markdown("### Downloads")
        excel_buf = excel_bytes(combined_df, stats_df, counts_df, class_df, one_sheet=one_sheet)
        st.download_button(
            "Download Excel",
            data=excel_buf,
            file_name="results_with_classification.xlsx" if one_sheet else "results_multi_sheet.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # Download plot PNG
        png_buf = io.BytesIO()
        fig.savefig(png_buf, format="png", bbox_inches="tight")
        png_buf.seek(0)
        st.download_button("Download plot (PNG)", data=png_buf, file_name="classification_grid.png", mime="image/png")

        st.caption("Notes: CV² uses sample standard deviation (STDEV.S). ADI = N périodes / N fréquence (∞ if no demand).")
    except Exception as e:
        st.error(f"Processing failed: {e}")
else:
    st.info("Upload an Excel file to get started. The first column must be the product name; remaining columns should be date headers with numeric quantities (0 means no purchase).")
