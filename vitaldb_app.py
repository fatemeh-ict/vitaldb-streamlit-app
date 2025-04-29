# VitalDB Streamlit Analyzer - Final Professional Version

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vitaldb

st.set_page_config(layout="wide")
st.title("VitalDB Analyzer with Outlier & NaN Detection")

# -------------------- Load Data --------------------
@st.cache_data
def load_data():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    df_labs = pd.read_csv("https://api.vitaldb.net/labs")
    return df_cases, df_trks, df_labs

# -------------------- Filtering --------------------
def flexible_case_selection(df_cases, df_trks, ane_types, exclude_drugs):
    group1 = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
              "Orchestra/PPF20_CE", "Orchestra/RFTN20_CE",
              "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"]
    group2 = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
              "Orchestra/PPF20_CE", "Orchestra/RFTN50_CE",
              "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"]

    filtered_cases = df_cases[df_cases['ane_type'].isin(ane_types)]
    valid_caseids = set(filtered_cases['caseid'])

    def match_group(group):
        ids = set(filtered_cases['caseid'])
        for sig in group:
            ids &= set(df_trks[df_trks['tname'] == sig]['caseid'])
        return ids

    valid_caseids = match_group(group1).union(match_group(group2))

    if exclude_drugs:
        drug_cols = [col for col in exclude_drugs if col in filtered_cases.columns]
        filtered_cases = filtered_cases[~filtered_cases[drug_cols].gt(0).any(axis=1)]
        valid_caseids &= set(filtered_cases['caseid'])

    return list(valid_caseids)

# -------------------- Analyzer Class --------------------
class VitalDBAnalyzer:
    def __init__(self, caseid, data, variable_names, thresholds, global_medians, global_mads):
        self.caseid = caseid
        self.data = data
        self.variable_names = variable_names
        self.thresholds = thresholds
        self.global_medians = global_medians
        self.global_mads = global_mads
        self.issues = {var: {'outlier': [], 'outlier_values': [], 'nan': []} for var in variable_names}
        self.warnings = []

    def _loop_signals(self):
        for i, var in enumerate(self.variable_names):
            yield i, var, self.data[:, i]

    def check_missing_data(self):
        for i, var, signal in self._loop_signals():
            nan_idx = np.where(np.isnan(signal))[0]
            self.issues[var]['nan'] = nan_idx.tolist()
            if len(nan_idx) / len(signal) > self.thresholds['missing']:
                self.warnings.append(f"[{self.caseid}] [{var}] > {self.thresholds['missing']*100:.0f}% missing data.")

    def check_outliers_custom(self):
        for i, var, signal in self._loop_signals():
            original = signal.copy()
            if "RATE" in var:
                idx = np.where(signal < 0)[0]
            elif "BIS" in var:
                idx = np.where((signal <= 0) | (signal > 100))[0]
            elif "NIBP" in var:
                idx = list(np.where(signal <= 0)[0])
                if var in self.global_medians:
                    median = self.global_medians[var]
                    mad = self.global_mads[var] or 1e-6
                    idx.extend(np.where(np.abs(signal - median) > 3.5 * mad)[0].tolist())
                idx = sorted(set(idx))
            else:
                if var in self.global_medians:
                    median = self.global_medians[var]
                    mad = self.global_mads[var] or 1e-6
                    idx = np.where(np.abs(signal - median) > 3.5 * mad)[0]
                else:
                    continue
            self.issues[var]['outlier'] = idx
            self.issues[var]['outlier_values'] = original[idx].tolist()
            signal[idx] = np.nan
            if len(idx):
                self.warnings.append(f"[{self.caseid}] [{var}] {len(idx)} outliers replaced with NaN.")

    def plot_issues(self):
        df = pd.DataFrame(self.data, columns=self.variable_names)
        df['time'] = np.arange(len(df))
        fig = make_subplots(rows=len(self.variable_names), cols=1, shared_xaxes=True,
                            subplot_titles=self.variable_names)
        for i, var in enumerate(self.variable_names):
            row = i + 1
            signal = df[var]
            mode = 'markers+lines' if np.count_nonzero(~np.isnan(signal)) > 20 else 'markers'
            fig.add_trace(go.Scatter(x=df['time'], y=signal, mode=mode, name=var), row=row, col=1)
            if self.issues[var]['outlier']:
                x = [df['time'][j] for j in self.issues[var]['outlier']]
                y = self.issues[var]['outlier_values']
                fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name=f"Outliers {var}",
                                         marker=dict(color='red', size=6, symbol='star')), row=row, col=1)
            if self.issues[var]['nan']:
                x_nan = [df['time'][j] for j in self.issues[var]['nan']]
                y_nan = [signal.min() - 5] * len(x_nan)
                fig.add_trace(go.Scatter(x=x_nan, y=y_nan, mode='markers', name=f"NaNs {var}",
                                         marker=dict(color='gray', size=5, symbol='line-ns-open')), row=row, col=1)
        fig.update_layout(height=300 * len(self.variable_names), title=f"Signal Issues - Case {self.caseid}")
        st.plotly_chart(fig, use_container_width=True)

# -------------------- Stats --------------------
@st.cache_data
def compute_global_stats(caseids, variable_names):
    medians, mads = {}, {}
    for var in variable_names:
        values = []
        for cid in caseids:
            try:
                data = vitaldb.load_case(cid, variable_names)
                if data is not None:
                    values.extend(data[:, variable_names.index(var)][~np.isnan(data[:, variable_names.index(var)])])
            except:
                continue
        if values:
            med = np.median(values)
            mad = np.median(np.abs(values - med)) or 1e-6
            medians[var] = med
            mads[var] = mad
    return medians, mads

# -------------------- Streamlit UI --------------------
df_cases, df_trks, df_labs = load_data()
filter_tab, analysis_tab = st.tabs(["Filter & Download", "Signal Analysis"])

with filter_tab:
    st.subheader("Step 1: Filter Dataset")
    ane_types_all = df_cases['ane_type'].dropna().unique().tolist()
    selected_ane_types = st.multiselect("Anesthesia Types", ane_types_all, default=["General"])
    drugs = ['intraop_mdz', 'intraop_ftn', 'intraop_epi', 'intraop_phe', 'intraop_eph']
    existing_drugs = [d for d in drugs if d in df_cases.columns]
    selected_drugs = st.multiselect("Drugs to Exclude", existing_drugs, default=existing_drugs)

    if st.button("Apply Filtering"):
        valid_caseids = flexible_case_selection(df_cases, df_trks, selected_ane_types, selected_drugs)
        if valid_caseids:
            df_cases_filtered = df_cases[df_cases['caseid'].isin(valid_caseids)]
            df_trks_filtered = df_trks[df_trks['caseid'].isin(valid_caseids)]
            df_labs_filtered = df_labs[df_labs['caseid'].isin(valid_caseids)]

            st.success(f"{len(valid_caseids)} valid cases found.")
            st.download_button("Download Filtered Cases", df_cases_filtered.to_csv(index=False), "filtered_cases.csv")
            st.download_button("Download Filtered Tracks", df_trks_filtered.to_csv(index=False), "filtered_trks.csv")
            st.download_button("Download Filtered Labs", df_labs_filtered.to_csv(index=False), "filtered_labs.csv")

            st.session_state['filtered_ids'] = valid_caseids
            if st.checkbox("Show Filtered Table"):
                st.dataframe(df_cases_filtered.head())
        else:
            st.warning("No valid cases found.")

with analysis_tab:
    st.subheader("Step 2: Analyze Case")
    variable_names = ["Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "BIS/BIS",
                      "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP"]
    filtered_ids = st.session_state.get("filtered_ids", [])
    selected_caseid = st.selectbox("Select Case ID", filtered_ids if filtered_ids else ["No Data"])

    if st.button("Run Analysis") and selected_caseid != "No Data":
        with st.spinner("Analyzing..."):
            data = vitaldb.load_case(selected_caseid, variable_names)
            thresholds = {"missing": 0.05}
            global_medians, global_mads = compute_global_stats(filtered_ids, variable_names)
            analyzer = VitalDBAnalyzer(selected_caseid, data.copy(), variable_names,
                                       thresholds, global_medians, global_mads)
            analyzer.check_missing_data()
            analyzer.check_outliers_custom()
            analyzer.plot_issues()
            for w in analyzer.warnings:
                st.warning(w)
