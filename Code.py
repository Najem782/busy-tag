import streamlit as st
import pandas as pd
import os

# Set up page configuration
st.set_page_config(page_title="Restaurant Manager", layout="wide", page_icon="🍽️")

st.title("🍽️ Restaurant Management Dashboard")
st.markdown("Enter your daily metrics below. All differences, totals, and revenue update instantly.")
st.markdown("---")

# File paths for cloud persistence
EXPENSE_FILE = "expense_store.csv"
INVENTORY_FILE = "inventory_store.csv"
CASH_FILE = "cash_store.csv"

# --- INVENTORY TABLE SETUP ---
INV_COLUMNS = ["rassyoun", "kbir", "sghiir", "hout"]

# 10 exact rows structured sequentially based on your input loop
INV_ROWS = [
    "Beyet (1)",
    "Achat",
    "Vente (1)",
    "Reste (1)",
    "Différence (1)",  # Formula 1 goes here
    "Reste (2)",
    "Plus",
    "Vente (2)",
    "Beyet (2)",
    "Différence (2)"   # Formula 2 goes here
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
            if len(df) == len(INV_ROWS):
                return df.reindex(columns=INV_COLUMNS, fill_value=0.0)
        except: pass
    return pd.DataFrame(0.0, index=INV_ROWS, columns=INV_COLUMNS)

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

# Initialize Session States safely
if "left_table_data" not in st.session_state:
    st.session_state.left_table_data = load_expense_data()
if "inventory_data" not in st.session_state:
    st.session_state.inventory_data = load_inventory_data()
if "cash_left" not in st.session_state:
    st.session_state.cash_left = load_cash_left()


# ==========================================
# SECTION 1: THE 10-ROW BALANCE LEDGER
# ==========================================
st.subheader("📊 Stock & Balance Tracker")
st.caption("Fill in the fields. Row calculations update in real time.")

working_inv_df = st.session_state.inventory_data.copy()

# Lock the formulation rows so users can't accidentally type over them
edited_inv_df = st.data_editor(
    working_inv_df,
    use_container_width=True,
    disabled=["Différence (1)", "Différence (2)"],
    key="inventory_editor"
)

# Run math calculations over the 4 custom columns
for col in INV_COLUMNS:
    try:
        # Loop 1 Math: Différence = Beyet + Achat - Vente - Reste
        diff1 = (edited_inv_df.at["Beyet (1)", col] + 
                 edited_inv_df.at["Achat", col] - 
                 edited_inv_df.at["Vente (1)", col] - 
                 edited_inv_df.at["Reste (1)", col])
        edited_inv_df.at["Différence (1)", col] = diff1
        
        # Loop 2 Math: Différence = Reste + Plus - Vente - Beyet
        diff2 = (edited_inv_df.at["Reste (2)", col] + 
                 edited_inv_df.at["Plus", col] - 
                 edited_inv_df.at["Vente (2)", col] - 
                 edited_inv_df.at["Beyet (2)", col])
        edited_inv_df.at["Différence (2)", col] = diff2
    except KeyError:
        pass

# Force saving and rerun if states alter
if not edited_inv_df.equals(st.session_state.inventory_data):
    st.session_state.inventory_data = edited_inv_df
    save_csv(edited_inv_df, INVENTORY_FILE)
    st.rerun()

st.markdown("---")

# ==========================================
# SECTION 2: OPERATIONS & EXPENSES
# ==========================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📋 Core Tracked Expenses")
    st.caption("Modify amounts directly in the grid cell rows.")
    
    edited_left_df = st.data_editor(
        st.session_state.left_table_data,
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
                <span>Total Expenses:</span>
                <span>{total_expenses:,.3f} DT</span>
            </h4>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col_right:
    st.subheader("💰 Revenue & Cash Tracking (Recette)")
    st.caption("Log your remaining cash below to calculate total daily revenue.")
    
    # Static operational metric from previous update
    st.metric(label="Other Operations Fixed Total", value="60.000 DT")
    
    # Input field for Money Left / Argent Restant
    cash_left_input = st.number_input(
        "Argent Restant (Money Left in Register):", 
        min_value=0.0, 
        value=st.session_state.cash_left, 
        step=5.0, 
        format="%.3f"
    )
    
    # If the user edits the input box, save it to session state and disk
    if cash_left_input != st.session_state.cash_left:
        st.session_state.cash_left = cash_left_input
        save_cash_left(cash_left_input)
        st.rerun()
    
    # Calculate Total Revenue (Recette Totale)
    # Recette = Expenses + Cash Left + 60 DT Fixed
    total_revenue = total_expenses + cash_left_input + 60.0
    
    st.markdown(
        f"""
        <div style="background-color: #e8f5e9; padding: 15px; border-radius: 6px; border-left: 5px solid #2e7d32; margin-top: 25px;">
            <h3 style="margin: 0; color: #1b5e20; display: flex; justify-content: space-between;">
                <span>Recette Totale (Whole Revenue):</span>
                <span>{total_revenue:,.3f} DT</span>
            </h3>
            <p style="margin: 5px 0 0 0; font-size: 13px; color: #2e7d32; font-style: italic;">
                Calculation: Total Expenses ({total_expenses:,.3f}) + Argent Restant ({cash_left_input:,.3f}) + Fixed Ops (60.000)
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
