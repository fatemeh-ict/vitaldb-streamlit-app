import streamlit as st
import pandas as pd

# ------------------------------
# Load Data
@st.cache_data

def load_data():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    df_labs = pd.read_csv("https://api.vitaldb.net/labs")
    return df_cases, df_trks, df_labs

# ------------------------------
# Filtering Functions
def filter_cases(df_cases, df_trks, ane_types, required_signals, exclude_drugs):
    # Filter by anesthesia type
    filtered_cases = df_cases[df_cases['ane_type'].isin(ane_types)]
    valid_caseids = set(filtered_cases['caseid'])

    # Filter by required signals
    if required_signals:
        for signal in required_signals:
            cases_with_signal = set(df_trks[df_trks['tname'] == signal]['caseid'])
            valid_caseids = valid_caseids.intersection(cases_with_signal)

    # Exclude by intraoperative drugs
    if exclude_drugs:
        drug_cols = [col for col in exclude_drugs if col in filtered_cases.columns]
        filtered_cases = filtered_cases[~filtered_cases[drug_cols].gt(0).any(axis=1)]
        valid_caseids = valid_caseids.intersection(set(filtered_cases['caseid']))

    return list(valid_caseids)

# ------------------------------
# Streamlit App
st.title("VitalDB Dataset Dynamic Filtering Platform")

st.write("Select the filtering criteria exactly as designed in your original code.")

# Load datasets
df_cases, df_trks, df_labs = load_data()

# Sidebar selections
st.sidebar.header("Filtering Options")

# Anesthesia Type
ane_types_all = df_cases['ane_type'].dropna().unique().tolist()
selected_ane_types = st.sidebar.multiselect("Select Anesthesia Types", ane_types_all, default=["General"])

# Required Signals
available_signals = df_trks['tname'].dropna().unique().tolist()
selected_signals = st.sidebar.multiselect("Select Required Signals", available_signals, default=[
    "Solar8000/NIBP_DBP",
    "Solar8000/NIBP_SBP",
    "BIS/BIS",
    "Orchestra/PPF20_CE",
    "Orchestra/RFTN20_CE",
    "Orchestra/PPF20_RATE",
    "Orchestra/RFTN20_RATE"
])

# Drugs to Exclude
potential_drugs = ['intraop_mdz', 'intraop_ftn', 'intraop_epi', 'intraop_phe', 'intraop_eph']
existing_drug_cols = [col for col in potential_drugs if col in df_cases.columns]
selected_drugs = st.sidebar.multiselect("Select Drugs to Exclude", existing_drug_cols, default=existing_drug_cols)

# Apply Filter Button
if st.sidebar.button("Apply Filtering"):
    # Apply filtering based on selections
    valid_caseids = filter_cases(df_cases, df_trks, selected_ane_types, selected_signals, selected_drugs)

    if valid_caseids:
        df_cases_filtered = df_cases[df_cases['caseid'].isin(valid_caseids)]
        df_trks_filtered = df_trks[df_trks['caseid'].isin(valid_caseids)]
        df_labs_filtered = df_labs[df_labs['caseid'].isin(valid_caseids)]

        st.success(f"Filtering completed. {len(valid_caseids)} valid cases found.")

        # Show basic stats
        st.subheader("Filtered Cases Summary")
        st.write(f"Number of Cases: {len(df_cases_filtered)}")
        st.write(f"Average Age: {df_cases_filtered['age'].mean():.2f}")
        st.write(f"Average BMI: {df_cases_filtered['bmi'].mean():.2f}")

        # Download buttons
        csv_cases = df_cases_filtered.to_csv(index=False).encode('utf-8')
        csv_trks = df_trks_filtered.to_csv(index=False).encode('utf-8')
        csv_labs = df_labs_filtered.to_csv(index=False).encode('utf-8')

        st.download_button("Download Filtered Cases", csv_cases, "filtered_cases.csv", "text/csv")
        st.download_button("Download Filtered Tracks", csv_trks, "filtered_trks.csv", "text/csv")
        st.download_button("Download Filtered Labs", csv_labs, "filtered_labs.csv", "text/csv")

        # Optional: show the first few rows
        if st.checkbox("Show Filtered Cases Table"):
            st.dataframe(df_cases_filtered.head())

    else:
        st.warning("No cases found matching the selected criteria.")

# ------------------------------
# End of App
