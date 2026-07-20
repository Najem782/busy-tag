import streamlit as st
import pandas as pd
import os

# Set up page configuration
st.set_page_config(page_title="Restaurant Expense Manager", layout="wide", page_icon="🍽️")

st.title("🍽️ Restaurant Expense Manager")
st.markdown("Enter your daily operational figures. Calculations update automatically.")
st.markdown("---")

# File paths for cloud persistence
EXPENSE_FILE = "expense_store.csv"
INVENTORY_FILE = "inventory_store.csv"

# Core setups from previous configurations
FIXED_ITEMS = [
    "Frite", "Gazouz", "Fham", "Khobz", "Aymen", 
    "Attar", "Khadema", "Khodhra", "mazraa", "karim"
]

# --- INVENTORY DATA CONFIGURATION ---
# Columns: rassyoun, kbir, sghiir, hout, [blank] -> named "Other"
INV_COLUMNS = ["rassyoun", "kbir", "sghiir", "hout", "Other"]
# 11 Rows required by user
INV_ROWS = [
    "beyet", "achat", "vente", "reste", "différence",
    "reste plus", "vente beyet", "difference"
]
# Fill out remaining rows up to 11 total with custom empty placeholders if needed
while len(INV_ROWS) < 11:
    INV_ROWS.append(f"Row {len(INV_ROWS) + 1}")

# --- PERSISTENCE LAYER FUNCTIONS ---
def load_expense_data():
    if os.path.exists(EXPENSE_FILE):
        try:
            df = pd.read_csv(EXPENSE_FILE)
            if "Expense Item" in df.columns and "Amount" in df.columns:
                return df
        except: pass
    return pd.DataFrame({"Expense Item": FIXED_ITEMS, "Amount": [0.0] * 10})

def load_inventory_data():
    if os.path.exists(INVENTORY_FILE):
        try:
            df = pd.read_csv(INVENTORY_FILE, index_col=0)
            if list(df.columns) == INV_COLUMNS and len(df) == 11:
                return df
        except: pass
    # Default matrix with 11 rows and 5 columns initialized to 0.0
    return pd.DataFrame(0.0, index=INV_ROWS, columns=INV_COLUMNS)

def save_csv(df, filepath):
    df.to_csv(filepath, index=True if "Expense Item" not in df.columns else False)

# Initialize Session States safely
if "left_table_data" not in st.session_state:
    st.session_state.left_table_data = load_expense_data()
if "inventory_data" not in st.session_state:
    st.session_state.inventory_data = load_inventory_data()


# ==========================================
# NEW SECTION: 5-COLUMN BY 11-ROW INVENTORY
# ==========================================
st.subheader("📊 Stock & Sales Balance Ledger")
st.caption("Edit values directly in the grid. Rows labeled 'différence' and 'difference' calculate automatically.")

# Prepare dataframe for editing (make a copy to alter programmatically)
working_inv_df = st.session_state.inventory_data.copy()

# Render editable grid for inventory tracker
# We disable the formula rows so users don't overwrite calculation paths manually
edited_inv_df = st.data_editor(
    working_inv_df,
    use_container_width=True,
    disabled=["différence", "difference"], # Locks formula rows from manual overrides
    key="inventory_editor"
)

# Apply Formulas to the user data matrix dynamically across all 5 columns
for col in INV_COLUMNS:
    try:
        # First Half Formula: Différence = beyet + achat - vente - reste
        diff1 = (edited_inv_df.at["beyet", col] + 
                 edited_inv_df.at["achat", col] - 
                 edited_inv_df.at["vente", col] - 
                 edited_inv_df.at["reste", col])
        edited_inv_df.at["différence", col] = diff1
        
        # Second Half Formula: Difference = reste plus - vente beyet (Interpreted cleanly from user metrics)
        # Formula: Difference = (reste + plus) - vente - beyet
        # Maps "reste plus" cell + "Row 7" (vente beyet item indicators) to handle the equation
        r_plus = edited_inv_df.at["reste plus", col]
        v_beyet = edited_inv_df.at["vente beyet", col]
        
        diff2 = r_plus - v_beyet - edited_inv_df.at["vente", col] - edited_inv_df.at["beyet", col]
        edited_inv_df.at["difference", col] = diff2
    except KeyError:
        pass # Catch safety triggers if tracking items skew row labels

# Save Inventory Changes if edits exist
if not edited_inv_df.equals(st.session_state.inventory_data):
    st.session_state.inventory_data = edited_inv_df
    save_csv(edited_inv_df, INVENTORY_FILE)
    st.rerun()

st.markdown("---")

# ==========================================
# PREVIOUS OPERATIONS SECTION: EXPENSES
# ==========================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📋 Core Tracked Expenses")
    st.caption("Double-click any cell in the 'Amount' column to update entries.")
    
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
    st.subheader("📝 General / Miscellaneous Operations")
    st.caption("A flexible scratchpad layout to list external or fluctuating daily costs.")
    
    if "right_table_data" not in st.session_state:
        st.session_state.right_table_data = pd.DataFrame(
            [
                {"Category/Item": "Rent & Utilities", "Amount": 0.0, "Notes": ""},
                {"Category/Item": "Equipment Repair", "Amount": 0.0, "Notes": ""},
            ]
        )
        
    edited_right_df = st.data_editor(
        st.session_state.right_table_data,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        key="right_editor"
    )
    st.session_state.right_table_data = edited_right_df
    
    right_total = edited_right_df["Amount"].sum() if not edited_right_df.empty else 0.0
    st.metric(label="Other Operations Total", value=f"{right_total:,.3f} DT")
