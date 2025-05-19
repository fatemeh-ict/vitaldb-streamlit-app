import streamlit as st
import pandas as pd
import numpy as np
import vitaldb
from io import BytesIO

# ------------------ UTILS -------------------
def load_data():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    df_labs = pd.read_csv("https://api.vitaldb.net/labs")
    return df_cases, df_trks, df_labs

# ------------------ PAGE -------------------
st.set_page_config(page_title="VitalDB Streamlit", layout="wide")
st.title("VitalDB Signal Preprocessing Tool")

# ------------------ LOAD DATA -------------------
@st.cache_data
def cached_data():
    return load_data()

df_cases, df_trks, df_labs = cached_data()

# ------------------ SESSION INIT -------------------
if "valid_ids" not in st.session_state:
    st.session_state.valid_ids = []
    st.session_state.df_cases_filtered = pd.DataFrame()
    st.session_state.df_trks_filtered = pd.DataFrame()
    st.session_state.df_labs_filtered = pd.DataFrame()

# ------------------ TAB 1: FILTER -------------------
with st.expander("1. Filter Cases", expanded=True):
    ane_type = st.selectbox("Select Anesthesia Type:", df_cases["ane_type"].dropna().unique())
    intraoperative_boluses = ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"]
    removed_boluses = st.multiselect("Remove cases with these drugs:", options=intraoperative_boluses, default=intraoperative_boluses)

    required_variables_1 = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS", "Orchestra/PPF20_CE", "Orchestra/RFTN20_CE", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"]
    required_variables_2 = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS", "Orchestra/PPF20_CE", "Orchestra/RFTN50_CE", "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"]

    if st.button("Apply Filters"):
        def get_valid_ids(df_cases, df_trks, ane_type, variables, removed_boluses):
            filtered = df_cases[df_cases['ane_type'] == ane_type].copy()
            valid_ids = set(filtered['caseid'])
            for var in variables:
                valid_ids &= set(df_trks[df_trks['tname'] == var]['caseid'])
            if removed_boluses:
                valid_boluses = [col for col in removed_boluses if col in filtered.columns]
                filtered = filtered[~filtered[valid_boluses].gt(0).any(axis=1)]
                valid_ids &= set(filtered['caseid'])
            return valid_ids

        ids1 = get_valid_ids(df_cases, df_trks, ane_type, required_variables_1, removed_boluses)
        ids2 = get_valid_ids(df_cases, df_trks, ane_type, required_variables_2, removed_boluses)
        valid_ids = sorted(list(ids1.union(ids2)))

        df_cases_filtered = df_cases[df_cases['caseid'].isin(valid_ids)].copy()
        df_trks_filtered = df_trks[df_trks['caseid'].isin(valid_ids)].copy()
        df_labs_filtered = df_labs[df_labs['caseid'].isin(valid_ids)].copy()

        st.session_state.valid_ids = valid_ids
        st.session_state.df_cases_filtered = df_cases_filtered
        st.session_state.df_trks_filtered = df_trks_filtered
        st.session_state.df_labs_filtered = df_labs_filtered

        st.success(f"✅ Filtered {len(valid_ids)} valid case IDs.")
        st.write("Filtered `df_cases` size:", df_cases_filtered.shape)
        st.write("Filtered `df_trks` size:", df_trks_filtered.shape)
        st.write("Filtered `df_labs` size:", df_labs_filtered.shape)

        # DOWNLOADS
        st.download_button("Download Filtered Cases CSV", df_cases_filtered.to_csv(index=False), "filtered_cases.csv")
        st.download_button("Download Filtered Trks CSV", df_trks_filtered.to_csv(index=False), "filtered_trks.csv")
        st.download_button("Download Filtered Labs CSV", df_labs_filtered.to_csv(index=False), "filtered_labs.csv")

# ------------------ TAB 2: ANALYZE -------------------
with st.expander("2. Signal Analysis", expanded=True):
    if st.session_state.valid_ids:
        selected_ids = st.multiselect("Select Case IDs to Analyze:", st.session_state.valid_ids, default=st.session_state.valid_ids[:2])

        if st.button("Run Analysis"):
            for cid in selected_ids:
                st.markdown(f"#### Case {cid}")
                try:
                    variables = ["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "Orchestra/RFTN50_RATE"]
                    data = vitaldb.load_case(cid, variables, interval=1)

                    st.write(f"Loaded signal shape: {data.shape}")

                    df = pd.DataFrame(data, columns=variables)
                    df["Time"] = np.arange(len(df))

                    for var in variables:
                        st.line_chart(df[["Time", var]].set_index("Time"))

                except Exception as e:
                    st.error(f"❌ Error loading case {cid}: {e}")
    else:
        st.warning("⚠️ Please apply filters first to select valid case IDs.")
