import streamlit as st
import pandas as pd
import os

# Set up page configuration
st.set_page_config(page_title="Restaurant Expense Manager", layout="wide", page_icon="🍽️")

st.title("🍽️ Restaurant Expense Manager")
st.markdown("Enter your daily expenses below. Changes are saved automatically.")
st.markdown("---")

# File path for cloud persistence
DATA_FILE = "expense_store.csv"

# 10 core items requested by user
FIXED_ITEMS = [
    "Frite", "Gazouz", "Fham", "Khobz", "Aymen", 
    "Attar", "Khadema", "Khodhra", "mazraa", "karim"
]

# --- DATA PERSISTENCE LAYER ---
def load_saved_data():
    """Loads existing data from CSV or initializes default states if file doesn't exist."""
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            # Ensure it contains the expected columns
            if "Expense Item" in df.columns and "Amount" in df.columns:
                return df
        except Exception:
            pass
            
    # Default initial fallback dataframe matching exactly your requirements
    return pd.DataFrame({
        "Expense Item": FIXED_ITEMS,
        "Amount": [0.0] * 10
    })

def save_data(df):
    """Saves the current dataframe state to CSV file."""
    df.to_csv(DATA_FILE, index=False)

# Initialize data state safely
if "left_table_data" not in st.session_state:
    st.session_state.left_table_data = load_saved_data()


# --- INTERFACE LAYOUT ---
col_left, col_right = st.columns(2)

# --- LEFT TABLE (Core Tracked Expenses) ---
with col_left:
    st.subheader("📋 Core Tracked Expenses")
    st.caption("Double-click any cell in the 'Amount' column to update entries.")
    
    # Display the editable interface
    edited_left_df = st.data_editor(
        st.session_state.left_table_data,
        use_container_width=True,
        hide_index=True,
        disabled=["Expense Item"],  # Locks the item names from being edited
        key="left_editor"
    )
    
    # Check if user changed values; if so, update state and auto-save to disk
    if not edited_left_df.equals(st.session_state.left_table_data):
        st.session_state.left_table_data = edited_left_df
        save_data(edited_left_df)
        st.rerun()
        
    # Calculate aggregate expenses
    total_expenses = edited_left_df["Amount"].sum()
    
    # Custom styled row block to isolate the final calculation display nicely
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

# --- RIGHT TABLE (General / Other Expenses) ---
with col_right:
    st.subheader("📝 General / Miscellaneous Operations")
    st.caption("A flexible scratchpad layout to list external or fluctuating daily costs.")
    
    if "right_table_data" not in st.session_state:
        st.session_state.right_table_data = pd.DataFrame(
            [
                {"Category/Item": "Rent & Utilities", "Amount": 0.0, "Notes": ""},
                {"Category/Item": "Equipment Repair", "Amount": 0.0, "Notes": ""},
                {"Category/Item": "Marketing", "Amount": 0.0, "Notes": ""},
            ]
        )
        
    edited_right_df = st.data_editor(
        st.session_state.right_table_data,
        use_container_width=True,
        num_rows="dynamic",  # Allows adding or deleting rows cleanly on the fly
        hide_index=True,
        key="right_editor"
    )
    st.session_state.right_table_data = edited_right_df
    
    right_total = edited_right_df["Amount"].sum() if not edited_right_df.empty else 0.0
    st.metric(label="Other Operations Total", value=f"{right_total:,.3f} DT")

st.markdown("---")
