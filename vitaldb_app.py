import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vitaldb

# ------------------------------
st.set_page_config(layout="wide")
st.title("VitalDB Dataset Filtering & Signal Analyzer")

# ------------------------------
@st.cache_data

def load_data():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    df_labs = pd.read_csv("https://api.vitaldb.net/labs")
    return df_cases, df_trks, df_labs

# ------------------------------
def flexible_case_selection(df_cases, df_trks, ane_types, exclude_drugs):
    group1 = [
        "Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
        "Orchestra/PPF20_CE", "Orchestra/RFTN20_CE",
        "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"
    ]
    group2 = [
        "Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
        "Orchestra/PPF20_CE", "Orchestra/RFTN50_CE",
        "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"
    ]
    filtered_cases = df_cases[df_cases['ane_type'].isin(ane_types)]
    valid_caseids = set(filtered_cases['caseid'])

    def match_group(group):
        ids = set(filtered_cases['caseid'])
        for sig in group:
            sig_cases = set(df_trks[df_trks['tname'] == sig]['caseid'])
            ids = ids.intersection(sig_cases)
        return ids

    ids_group1 = match_group(group1)
    ids_group2 = match_group(group2)
    valid_caseids = ids_group1.union(ids_group2)

    if exclude_drugs:
        drug_cols = [col for col in exclude_drugs if col in filtered_cases.columns]
        filtered_cases = filtered_cases[~filtered_cases[drug_cols].gt(0).any(axis=1)]
        valid_caseids = valid_caseids.intersection(set(filtered_cases['caseid']))

    return list(valid_caseids)

# ------------------------------
class VitalDBAnalyzer:
    def __init__(self, caseid, data, variable_names, thresholds, global_medians, global_mads):
        self.caseid = caseid
        self.data = data
        self.variable_names = variable_names
        self.thresholds = thresholds
        self.global_medians = global_medians
        self.global_mads = global_mads
        self.issues = {var: {'outlier': [], 'outlier_values': []} for var in variable_names}

    def check_outliers(self):
        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i]
            original = signal.copy()
            if var in self.global_medians:
                median = self.global_medians[var]
                mad = self.global_mads[var] or 1e-6
                outlier_idx = np.where(np.abs(signal - median) > 3.5 * mad)[0]
                self.issues[var]['outlier'] = outlier_idx.tolist()
                self.issues[var]['outlier_values'] = original[outlier_idx].tolist()
                self.data[outlier_idx, i] = np.nan

    def plot_issues(self):
        df = pd.DataFrame(self.data, columns=self.variable_names)
        df['time'] = np.arange(len(df))
        fig = make_subplots(rows=len(self.variable_names), cols=1, shared_xaxes=True,
                            subplot_titles=self.variable_names)
        for i, var in enumerate(self.variable_names):
            fig.add_trace(go.Scatter(x=df['time'], y=df[var], mode='lines', name=var), row=i+1, col=1)
            if self.issues[var]['outlier']:
                x = [df['time'][j] for j in self.issues[var]['outlier']]
                y = self.issues[var]['outlier_values']
                fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f"Outliers: {var}",
                                         marker=dict(color='red', size=6)), row=i+1, col=1)
        fig.update_layout(height=250 * len(self.variable_names), title=f"Case {self.caseid} Signal Quality")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------
df_cases, df_trks, df_labs = load_data()

# Sidebar
st.sidebar.header("Step 1: Filter Dataset")
ane_types_all = df_cases['ane_type'].dropna().unique().tolist()
selected_ane_types = st.sidebar.multiselect("Anesthesia Types", ane_types_all, default=["General"])

potential_drugs = ['intraop_mdz', 'intraop_ftn', 'intraop_epi', 'intraop_phe', 'intraop_eph']
existing_drugs = [col for col in potential_drugs if col in df_cases.columns]
selected_drugs = st.sidebar.multiselect("Exclude Cases with These Drugs", existing_drugs, default=existing_drugs)

if st.sidebar.button("Apply Filters"):
    valid_caseids = flexible_case_selection(df_cases, df_trks, selected_ane_types, selected_drugs)
    st.session_state['valid_caseids'] = valid_caseids
    st.success(f"Filtering complete: {len(valid_caseids)} valid cases.")

# Signal analysis
if 'valid_caseids' in st.session_state:
    st.sidebar.header("Step 2: Analyze Signals")
    variable_names = ["Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "BIS/BIS",
                      "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP"]
    thresholds = {"outlier": 3.5}
    case_to_analyze = st.sidebar.selectbox("Choose Case ID to Analyze", st.session_state['valid_caseids'])

    def compute_global_stats(caseids):
        medians = {}
        mads = {}
        for var in variable_names:
            all_vals = []
            for cid in caseids[:10]:
                try:
                    data = vitaldb.load_case(cid, variable_names)
                    if data is not None:
                        sig = data[:, variable_names.index(var)]
                        all_vals.extend(sig[~np.isnan(sig)])
                except:
                    continue
            med = np.median(all_vals)
            mad = np.median(np.abs(all_vals - med)) or 1e-6
            medians[var] = med
            mads[var] = mad
        return medians, mads

    if st.sidebar.button("Run Analysis"):
        with st.spinner("Loading and analyzing case..."):
            data = vitaldb.load_case(case_to_analyze, variable_names)
            medians, mads = compute_global_stats(st.session_state['valid_caseids'])
            analyzer = VitalDBAnalyzer(case_to_analyze, data.copy(), variable_names, thresholds, medians, mads)
            analyzer.check_outliers()
            analyzer.plot_issues()
            st.success("Done.")
