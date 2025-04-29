import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vitaldb

st.set_page_config(layout="wide")
st.title("VitalDB Analyzer")

@st.cache_data
def load_data():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    df_labs = pd.read_csv("https://api.vitaldb.net/labs")
    return df_cases, df_trks, df_labs

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

class VitalDBAnalyzer:
    def __init__(self, caseid, data, variable_names, thresholds, global_medians, global_mads):
        self.caseid = caseid
        self.data = data
        self.variable_names = variable_names
        self.thresholds = thresholds
        self.global_medians = global_medians
        self.global_mads = global_mads
        self.issues = {var: {'outlier': [], 'outlier_values': [], 'nan': []} for var in variable_names}

    def check_outliers_and_nan(self):
        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i]
            original = signal.copy()
            nan_idx = np.where(np.isnan(signal))[0]
            self.issues[var]['nan'] = nan_idx.tolist()
            if var in self.global_medians:
                median = self.global_medians[var]
                mad = self.global_mads[var] or 1e-6
                outlier_idx = np.where(np.abs(signal - median) > 3.5 * mad)[0]
                self.issues[var]['outlier'] = outlier_idx.tolist()
                self.issues[var]['outlier_values'] = original[outlier_idx].tolist()
                signal[outlier_idx] = np.nan

    def plot_issues(self):
        df = pd.DataFrame(self.data, columns=self.variable_names)
        df['time'] = np.arange(len(df))
        fig = make_subplots(rows=len(self.variable_names), cols=1, shared_xaxes=True,
                            vertical_spacing=0.05, subplot_titles=self.variable_names)
        shown_legend = {"outlier": False, "nan": False}

        for i, var in enumerate(self.variable_names):
            signal = df[var].copy()
            time = df['time']
            row = i + 1

            if np.count_nonzero(~np.isnan(signal)) < 5:
                st.warning(f"Skipping {var} – not enough valid data.")
                continue

            fig.add_trace(go.Scatter(x=time, y=signal, mode='lines+markers', name=var), row=row, col=1)

            if self.issues[var]['outlier']:
                outlier_x = [time[idx] for idx in self.issues[var]['outlier']]
                outlier_y = self.issues[var]['outlier_values']
                fig.add_trace(go.Scatter(
                    x=outlier_x, y=outlier_y, mode='markers',
                    name="Outliers" if not shown_legend['outlier'] else None,
                    showlegend=not shown_legend['outlier'],
                    marker=dict(color='purple', size=6, symbol='star')
                ), row=row, col=1)
                shown_legend['outlier'] = True

            if self.issues[var]['nan']:
                nan_x = [time[idx] for idx in self.issues[var]['nan']]
                nan_y = [signal.min() - 5] * len(nan_x)
                fig.add_trace(go.Scatter(
                    x=nan_x, y=nan_y, mode='markers',
                    name="NaN" if not shown_legend["nan"] else None,
                    showlegend=not shown_legend["nan"],
                    marker=dict(color='gray', size=5, symbol='line-ns-open')
                ), row=row, col=1)
                shown_legend["nan"] = True

            fig.update_yaxes(title_text=var, row=row, col=1)
            fig.update_xaxes(title_text='Time (s)', row=row, col=1)

        fig.update_layout(title=f"Signal with Issues - Case {self.caseid}", height=250 * len(self.variable_names))
        st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def compute_global_stats(caseids, variable_names):
    medians = {}
    mads = {}
    for var in variable_names:
        all_vals = []
        for cid in caseids:
            try:
                data = vitaldb.load_case(cid, variable_names)
                if data is not None:
                    sig = data[:, variable_names.index(var)]
                    all_vals.extend(sig[~np.isnan(sig)])
            except:
                continue
        if all_vals:
            med = np.median(all_vals)
            mad = np.median(np.abs(all_vals - med)) or 1e-6
            medians[var] = med
            mads[var] = mad
    return medians, mads

# -------------------------
df_cases, df_trks, df_labs = load_data()

# Tabs
filter_tab, analysis_tab = st.tabs([" Filter & Download", " Signal Analysis"])

with filter_tab:
    st.subheader("Step 1: Filter Dataset")
    ane_types_all = df_cases['ane_type'].dropna().unique().tolist()
    selected_ane_types = st.multiselect("Select Anesthesia Types", ane_types_all, default=["General"])
    potential_drugs = ['intraop_mdz', 'intraop_ftn', 'intraop_epi', 'intraop_phe', 'intraop_eph']
    existing_drug_cols = [col for col in potential_drugs if col in df_cases.columns]
    selected_drugs = st.multiselect("Select Drugs to Exclude", existing_drug_cols, default=existing_drug_cols)

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

            if st.checkbox("Show Filtered Cases Table"):
                st.dataframe(df_cases_filtered.head())
        else:
            st.warning("No cases found matching criteria.")

with analysis_tab:
    st.subheader("Step 2: Signal Quality Check")
    variable_names = ["Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "BIS/BIS",
                      "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP"]
    selected_caseids = df_cases[df_cases['ane_type'] == 'General']['caseid'].unique().tolist()
    selected_caseid = st.selectbox("Select Case ID", selected_caseids)

    if st.button("Analyze Selected Case"):
        with st.spinner("Analyzing..."):
            data = vitaldb.load_case(selected_caseid, variable_names)
            global_medians, global_mads = compute_global_stats(selected_caseids[:10], variable_names)
            analyzer = VitalDBAnalyzer(selected_caseid, data.copy(), variable_names,
                                       thresholds={}, global_medians=global_medians, global_mads=global_mads)
            analyzer.check_outliers_and_nan()
            analyzer.plot_issues()
