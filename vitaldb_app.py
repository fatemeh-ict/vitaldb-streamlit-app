import streamlit as st
import pandas as pd
import vitaldb

# ------------------------ Page Setup ------------------------
st.set_page_config(page_title="VitalDB Signal Pipeline", layout="wide")
st.title("VitalDB Interactive Signal Processing Pipeline")

# ------------------------ Tabs ------------------------
tabs = st.tabs(["1. Select Cases", "2. Signal Analysis", "3. Interpolation", "4. Evaluation & Plots"])

# ------------------------ Tab 1: Case Selection ------------------------
with tabs[0]:
    st.subheader("Step 1: Select Anesthesia Type and Filter Drugs")

    ane_type = st.selectbox("Select Anesthesia Type", ["General", "Spinal", "Epidural", "MAC"])

    all_drugs = ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"]
    removed_drugs = st.multiselect("Select drugs to exclude (intraoperative boluses)", all_drugs, default=all_drugs)

    st.markdown("---")
    st.subheader("Automatic Variable Groups")
    st.markdown("Two required variable groups are hardcoded for now. These will be used to select valid case IDs.")

    required_variables_1 = [
        "Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
        "Orchestra/PPF20_CE", "Orchestra/RFTN20_CE", "Orchestra/PPF20_RATE",
        "Orchestra/RFTN20_RATE"
    ]

    required_variables_2 = [
        "Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
        "Orchestra/PPF20_CE", "Orchestra/RFTN50_CE", "Orchestra/PPF20_RATE",
        "Orchestra/RFTN50_RATE"
    ]

    # Load data once
    @st.cache_data
    def load_vitaldb_data():
        return (
            pd.read_csv("https://api.vitaldb.net/cases"),
            pd.read_csv("https://api.vitaldb.net/trks"),
            pd.read_csv("https://api.vitaldb.net/labs")
        )

    df_cases, df_trks, df_labs = load_vitaldb_data()

    # Filter by anesthesia type
    df_cases = df_cases[df_cases['ane_type'] == ane_type].copy()

    # Drug removal logic
    if removed_drugs:
        df_cases = df_cases[~df_cases[removed_drugs].gt(0).any(axis=1)]

    # Function to get case IDs containing required variables
    def get_valid_case_ids(required_variables):
        valid_ids = set(df_cases['caseid'])
        for var in required_variables:
            case_ids_with_var = set(df_trks[df_trks['tname'] == var]['caseid'])
            valid_ids &= case_ids_with_var
        return valid_ids

    valid_ids_1 = get_valid_case_ids(required_variables_1)
    valid_ids_2 = get_valid_case_ids(required_variables_2)
    final_valid_ids = sorted(list(valid_ids_1.union(valid_ids_2)))

    # Filter full dataframes
    df_cases_filtered = df_cases[df_cases['caseid'].isin(final_valid_ids)].copy()
    df_trks_filtered = df_trks[df_trks['caseid'].isin(final_valid_ids)].copy()
    df_labs_filtered = df_labs[df_labs['caseid'].isin(final_valid_ids)].copy()

    st.success(f"Total valid case IDs: {len(final_valid_ids)}")

    with st.expander("📊 Show filtered dataset dimensions"):
        st.write(f"df_cases_filtered: {df_cases_filtered.shape}")
        st.write(f"df_trks_filtered: {df_trks_filtered.shape}")
        st.write(f"df_labs_filtered: {df_labs_filtered.shape}")

    st.download_button(
        "Download Filtered Case Data",
        df_cases_filtered.to_csv(index=False).encode('utf-8'),
        file_name="filtered_cases.csv",
        mime="text/csv"
    )

    # Save filtered data to session state for next steps
    st.session_state.valid_ids = final_valid_ids
    st.session_state.df_cases_filtered = df_cases_filtered
    st.session_state.df_trks_filtered = df_trks_filtered
    st.session_state.df_labs_filtered = df_labs_filtered
    st.session_state.required_variables = list(set(required_variables_1 + required_variables_2))
