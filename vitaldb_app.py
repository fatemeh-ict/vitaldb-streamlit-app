import streamlit as st
import pandas as pd
from pipeline_modules import CaseSelector

# Load data from API
@st.cache_data

def load_data():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    df_labs = pd.read_csv("https://api.vitaldb.net/labs")
    return df_cases, df_trks, df_labs

df_cases, df_trks, df_labs = load_data()

st.title("VitalDB Signal Preprocessing Tool")

# Tab 1 - Data Filtering
with st.expander("1. Data Filtering", expanded=True):
    ane_type = st.selectbox("Select Anesthesia Type:", df_cases["ane_type"].dropna().unique())

    intraoperative_boluses = [
        "intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"
    ]
    removed_boluses = st.multiselect("Remove cases with these drugs:", options=intraoperative_boluses, default=intraoperative_boluses)

    required_variables_1 = [
        "Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
        "Orchestra/PPF20_CE", "Orchestra/RFTN20_CE",
        "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"
    ]

    required_variables_2 = [
        "Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
        "Orchestra/PPF20_CE", "Orchestra/RFTN50_CE",
        "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"
    ]

    if st.button("Apply Filters"):
        selector1 = CaseSelector(df_cases, df_trks, ane_type=ane_type, required_variables=required_variables_1, intraoperative_boluses=removed_boluses)
        valid_ids_1 = set(selector1.select_valid_cases())

        selector2 = CaseSelector(df_cases, df_trks, ane_type=ane_type, required_variables=required_variables_2, intraoperative_boluses=removed_boluses)
        valid_ids_2 = set(selector2.select_valid_cases())

        valid_ids = sorted(list(valid_ids_1.union(valid_ids_2)))

        df_cases_filtered = df_cases[df_cases['caseid'].isin(valid_ids)].copy()
        df_trks_filtered = df_trks[df_trks['caseid'].isin(valid_ids)].copy()
        df_labs_filtered = df_labs[df_labs['caseid'].isin(valid_ids)].copy()

        st.session_state["valid_ids"] = valid_ids
        st.session_state["df_cases_filtered"] = df_cases_filtered
        st.session_state["df_trks_filtered"] = df_trks_filtered
        st.session_state["df_labs_filtered"] = df_labs_filtered

        st.success(f"✅ Filtered {len(valid_ids)} valid case IDs.")

        st.write("Filtered `df_cases` size:", df_cases_filtered.shape)
        st.write("Filtered `df_trks` size:", df_trks_filtered.shape)
        st.write("Filtered `df_labs` size:", df_labs_filtered.shape)

        st.download_button("Download Filtered Cases CSV", df_cases_filtered.to_csv(index=False), "filtered_cases.csv")
        st.download_button("Download Filtered Trks CSV", df_trks_filtered.to_csv(index=False), "filtered_trks.csv")
        st.download_button("Download Filtered Labs CSV", df_labs_filtered.to_csv(index=False), "filtered_labs.csv")
