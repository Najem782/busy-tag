import streamlit as st
import pandas as pd
import os

# Set up page configuration
st.set_config = st.set_page_config(page_title="Restaurant Manager", layout="wide", page_icon="🍽️")

st.title("🍽️ Restaurant Management Dashboard")
st.markdown("---")

# File paths for cloud persistence
EXPENSE_FILE = "expense_store.csv"
INVENTORY_FILE = "inventory_store.csv"
CASH_FILE = "cash_store.csv"

# --- INVENTORY TABLE SETUP ---
INV_COLUMNS = ["rassyoun", "kbir", "sghiir", "hout"]

# The 8 rows the user types into
INPUT_ROWS = [
    "Beyet (1)",
    "Achat",
    "Vente (1)",
    "Reste (1)",
    "Reste (2)",
    "Plus",
    "Vente (2)",
    "Beyet (2)"
]

# --- EXPENSES SETUP ---
FIXED_ITEMS = [
    "Frite", "Gazouz", "Fham", "Khobz", "Aymen", 
    "Attar", "Khadema", "Khodhra", "mazraa", "karim", "tassliih"
]

# --- PERSISTENCE LAYER ---
def load_expense_data():
    if os.path.exists(EXPENSE_FILE):
        try:
            df = pd.read_csv(EXPENSE_FILE)
            if "Expense Item" in df.columns and "Amount" in df.columns:
                default_df = pd.DataFrame({"Expense Item": FIXED_ITEMS, "Amount": [0.0] * len(FIXED_ITEMS)})
                merged = pd.merge(default_df, df, on="Expense Item", how="left", suffixes=("_def", ""))
                merged["Amount"] = merged["Amount"].fillna(0.0)
                return merged[["Expense Item", "Amount"]]
        except: pass
    return pd.DataFrame({"Expense Item": FIXED_ITEMS, "Amount": [0.0] * len(FIXED_ITEMS)})

def load_inventory_data():
    if os.path.exists(INVENTORY_FILE):
        try:
            df = pd.read_csv(INVENTORY_FILE, index_col=0)
            if len(df) == len(INPUT_ROWS):
                return df.reindex(columns=INV_COLUMNS, fill_value=0.0)
        except: pass
    return pd.DataFrame(0.0, index=INPUT_ROWS, columns=INV_COLUMNS)

def load_cash_left():
    if os.path.exists(CASH_FILE):
        try:
            with open(CASH_FILE, "r") as f:
                return float(f.read().strip())
        except: pass
    return 0.0

def save_csv(df, filepath):
    df.to_csv(filepath, index=True if "Expense Item" not in df.columns else False)

def save_cash_left(value):
    with open(CASH_FILE, "w") as f:
        f.write(str(value))

# Initialize states safely
if "left_table_data" not in st.session_state:
    st.session_state.left_table_data = load_expense_data()
if "inventory_data" not in st.session_state:
    st.session_state.inventory_data = load_inventory_data()
if "cash_left" not in st.session_state:
    st.session_state.cash_left = load_cash_left()


# ==========================================
# SECTION 1: HSEEB DJEJ (Stock Balance Ledger)
# ==========================================
st.subheader("📊 Hseeb Djej")
st.caption("Enter your inventory logs below.")

# Render editable grid for raw inputs only (avoids the styling glitch entirely)
edited_inv_df = st.data_editor(
    st.session_state.inventory_data,
    use_container_width=True,
    key="inventory_editor"
)

# Save inventory inputs if changed
if not edited_inv_df.equals(st.session_state.inventory_data):
    st.session_state.inventory_data = edited_inv_df
    save_csv(edited_inv_df, INVENTORY_FILE)
    st.rerun()

# --- CALCULATIONS & HIGH-VISIBILITY DISPLAY ---
st.markdown("#### 🔍 Calculated Differences")

# Calculate the results dynamically in the background
diff1_results = {}
diff2_results = {}

for col in INV_COLUMNS:
    # Flipped Loop 1: Différence = Vente + Reste - Beyet - Achat
    diff1_results[col] = (edited_inv_df.at["Vente (1)", col] + 
                          edited_inv_df.at["Reste (1)", col] - 
                          edited_inv_df.at["Beyet (1)", col] - 
                          edited_inv_df.at["Achat", col])
    
    # Flipped Loop 2: Différence = Vente + Beyet - Reste - Plus
    diff2_results[col] = (edited_inv_df.at["Vente (2)", col] + 
                          edited_inv_df.at["Beyet (2)", col] - 
                          edited_inv_df.at["Reste (2)", col] - 
                          edited_inv_df.at["Plus", col])

# Display the calculated outputs using clean visual cards
col1, col2, col3, col4 = st.columns(4)
columns_mapped = [col1, col2, col3, col4]

for index, name in enumerate(INV_COLUMNS):
    with columns_mapped[index]:
        st.markdown(f"**{name.upper()}**")
        st.markdown(
            f"""
            <div style="background-color: #ffe0b2; padding: 8px; border-radius: 4px; border-left: 4px solid #f57c00; margin-bottom: 5px;">
                <span style="font-size: 12px; color: #e65100; font-weight: bold;">Différence (1):</span><br>
                <span style="font-size: 18px; color: #e65100; font-weight: bold;">{diff1_results[name]:,.2f}</span>
            </div>
            <div style="background-color: #fff3e0; padding: 8px; border-radius: 4px; border-left: 4px solid #ffb74d;">
                <span style="font-size: 12px; color: #f57c00; font-weight: bold;">Différence (2):</span><br>
                <span style="font-size: 18px; color: #f57c00; font-weight: bold;">{diff2_results[name]:,.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("---")

# ==========================================
# SECTION 2: CHARGE & FLOUSS
# ==========================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📋 Charge")
    st.caption("Modify amounts directly in the grid cell rows.")
    
    edited_left_df = st.data_editor(
        st.session_state.left_left_data if "left_left_data" in st.session_state else st.session_state.left_table_data,
        use_container_width=True,
        hide_index=True,
        disabled=["Expense Item"],
        key="left_editor"
    )
    
    if not edited_left_df.equals(st.session_state.left_table_data):
        st.session_state.left_table_data = edited_left_df
        save_csv(edited_left_df, EXPENSE_FILE)
        st.rerun()
        
    total_expenses = edited_left_df["Amount"].sum()
    st.markdown(
        f"""
        <div style="background-color: #f0f2f6; padding: 12px; border-radius: 6px; border-left: 5px solid #ff4b4b; margin-top: 10px;">
            <h4 style="margin: 0; color: #31333F; display: flex; justify-content: space-between;">
                <span>Total Charge:</span>
                <span>{total_expenses:,.3f} DT</span>
            </h4>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col_right:
    st.subheader("💰 Flouss")
    st.caption("Input the remaining physical cash stack.")
    
    cash_left_input = st.number_input(
        "Flouss (Money Left):", 
        min_value=0.0, 
        value=st.session_state.cash_left, 
        step=5.0, 
        format="%.3f"
    )
    
    if cash_left_input != st.session_state.cash_left:
        st.session_state.cash_left = cash_left_input
        save_cash_left(cash_left_input)
        st.rerun()
    
    # Recette = Total Charge + Flouss
    total_revenue = total_expenses + cash_left_input
    
    st.markdown(
        f"""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 6px; border-left: 5px solid #2e7d32; margin-top: 25px;">
            <h3 style="margin: 0; color: #1b5e20; display: flex; justify-content: space-between;">
                <span>Recette Totale:</span>
                <span>{total_revenue:,.3f} DT</span>
            </h3>
            <p style="margin: 5px 0 0 0; font-size: 13px; color: #2e7d32; font-style: italic;">
                Calculation: Total Charge ({total_expenses:,.3f}) + Flouss ({cash_left_input:,.3f})
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
