
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import seaborn as sns
import os

# ==========================
# Data Classes
# ==========================

class CaseSelector:
    def __init__(self, df_cases, df_trks, ane_type="General", required_variables=None, intraoperative_boluses=None):
        self.df_cases = df_cases.copy()
        self.df_trks = df_trks.copy()
        self.ane_type = ane_type
        self.required_variables = required_variables or []
        self.intraoperative_boluses = intraoperative_boluses or []

    def select_valid_cases(self):
        df_cases_filtered = self.df_cases[self.df_cases['ane_type'] == self.ane_type].copy()
        valid_case_ids = set(df_cases_filtered['caseid'])

        for var in self.required_variables:
            trk_cases = set(self.df_trks[self.df_trks['tname'] == var]['caseid'])
            valid_case_ids &= trk_cases

        if self.intraoperative_boluses:
            valid_boluses = [col for col in self.intraoperative_boluses if col in df_cases_filtered.columns]
            df_cases_filtered = df_cases_filtered[~df_cases_filtered[valid_boluses].gt(0).any(axis=1)]
            valid_case_ids &= set(df_cases_filtered['caseid'])

        return sorted(list(valid_case_ids))

# ==========================
# Streamlit GUI
# ==========================

st.set_page_config(layout="wide")
st.title("🩺 VitalDB Streamlit Analyzer")

tabs = st.tabs(["1️⃣ Select Cases", "2️⃣ Signal Quality", "3️⃣ Interpolation", "4️⃣ Evaluation", "5️⃣ Export"])

with tabs[0]:
    st.header("Step 1: Select Valid Cases")

    ane_type = st.selectbox("Anesthesia Type", ["General", "Spinal", "Regional"])
    intraoperative_boluses = st.multiselect("Exclude Boluses", ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"])

    selected_group = st.radio("Signal Group", ["Group 1 (RFTN20)", "Group 2 (RFTN50)", "Both Groups"])

    required_variables_1 = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
                            "Orchestra/PPF20_CE", "Orchestra/RFTN20_CE", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"]

    required_variables_2 = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
                            "Orchestra/PPF20_CE", "Orchestra/RFTN50_CE", "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"]

    if st.button("📥 Load and Filter Cases"):
        st.session_state['df_cases'] = pd.read_csv("https://api.vitaldb.net/cases")
        st.session_state['df_trks'] = pd.read_csv("https://api.vitaldb.net/trks")
        st.session_state['df_labs'] = pd.read_csv("https://api.vitaldb.net/labs")

        selector1 = CaseSelector(st.session_state['df_cases'], st.session_state['df_trks'], ane_type=ane_type,
                                 required_variables=required_variables_1, intraoperative_boluses=intraoperative_boluses)
        selector2 = CaseSelector(st.session_state['df_cases'], st.session_state['df_trks'], ane_type=ane_type,
                                 required_variables=required_variables_2, intraoperative_boluses=intraoperative_boluses)

        ids1 = set(selector1.select_valid_cases())
        ids2 = set(selector2.select_valid_cases())

        if selected_group == "Group 1 (RFTN20)":
            final_ids = ids1
        elif selected_group == "Group 2 (RFTN50)":
            final_ids = ids2
        else:
            final_ids = ids1.union(ids2)

        st.session_state['valid_ids'] = sorted(list(final_ids))
        st.session_state['df_cases_filtered'] = st.session_state['df_cases'][st.session_state['df_cases']['caseid'].isin(final_ids)]

        st.success(f"{len(st.session_state['valid_ids'])} valid case(s) selected.")
        st.dataframe(st.session_state['df_cases_filtered'].head(10))
