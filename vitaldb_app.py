import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import vitaldb
import os

# ------------------------- Config -------------------------
st.set_page_config(page_title="VitalDB Analyzer Tabs", layout="wide")
st.title("🧠 VitalDB Multi-Tab Signal Analysis")

# ------------------------- Load Data -------------------------
@st.cache_data(show_spinner=False)
def load_metadata():
    return (
        pd.read_csv("https://api.vitaldb.net/cases"),
        pd.read_csv("https://api.vitaldb.net/trks")
    )

df_cases, df_trks = load_metadata()

# ------------------------- Classes -------------------------
class CaseSelector:
    def __init__(self, df_cases, df_trks, ane_type, required_variables, intraoperative_boluses):
        self.df_cases = df_cases.copy()
        self.df_trks = df_trks.copy()
        self.ane_type = ane_type
        self.required_variables = required_variables
        self.intraoperative_boluses = intraoperative_boluses

    def select_valid_cases(self):
        df_filtered = self.df_cases[self.df_cases['ane_type'] == self.ane_type].copy()
        valid_ids = set(df_filtered['caseid'])

        for var in self.required_variables:
            trk_cases = set(self.df_trks[self.df_trks['tname'] == var]['caseid'])
            valid_ids &= trk_cases

        if self.intraoperative_boluses:
            valid_boluses = [col for col in self.intraoperative_boluses if col in df_filtered.columns]
            df_filtered = df_filtered[~df_filtered[valid_boluses].gt(0).any(axis=1)]
            valid_ids &= set(df_filtered['caseid'])

        return sorted(list(valid_ids)), df_filtered

class SignalAnalyzer:
    def __init__(self, caseid, data, variable_names, global_medians, global_mads):
        self.caseid = caseid
        self.data = data
        self.variable_names = variable_names
        self.global_medians = global_medians
        self.global_mads = global_mads
        self.issues = {var: {'nan': [], 'outlier': [], 'jump': []} for var in variable_names}

    def analyze(self):
        for i, var in enumerate(self.variable_names):
            sig = self.data[:, i]
            self.issues[var]['nan'] = np.where(np.isnan(sig))[0].tolist()
            median = self.global_medians[var]
            mad = self.global_mads[var] or 1e-6
            outliers = np.where(np.abs(sig - median) > 3.5 * mad)[0]
            self.issues[var]['outlier'] = outliers.tolist()
            diffs = np.diff(sig)
            self.issues[var]['jump'] = np.where(np.abs(diffs - np.median(diffs)) > 3.5 * mad)[0].tolist()
        return self.issues

class SignalProcessor:
    def __init__(self, data):
        self.data = data.copy()

    def interpolate(self):
        x = np.arange(self.data.shape[0])
        for i in range(self.data.shape[1]):
            sig = self.data[:, i]
            if np.isnan(sig).sum():
                valid = ~np.isnan(sig)
                if valid.sum() > 1:
                    f = interp1d(x[valid], sig[valid], kind='linear', fill_value='extrapolate')
                    self.data[:, i] = f(x)
        return self.data

class Evaluator:
    def __init__(self, raw_data, imputed_data, variable_names):
        self.raw_df = pd.DataFrame(raw_data, columns=variable_names)
        self.imp_df = pd.DataFrame(imputed_data, columns=variable_names)

    def compute_stats(self):
        result = []
        for col in self.raw_df.columns:
            result.append({
                'Variable': col,
                'NaNs Before': self.raw_df[col].isna().sum(),
                'NaNs After': self.imp_df[col].isna().sum(),
                'Mean Before': self.raw_df[col].mean(),
                'Mean After': self.imp_df[col].mean(),
                'Std Before': self.raw_df[col].std(),
                'Std After': self.imp_df[col].std(),
            })
        return pd.DataFrame(result)

# ------------------------- TABS -------------------------
tabs = st.tabs(["📂 Case Selection", "📈 Signal Analysis", "⚙️ Interpolation", "📊 Statistics"])

with tabs[0]:
    st.header("Step 1: Select Case Filters")
    ane_type = st.selectbox("Anesthesia Type:", sorted(df_cases['ane_type'].dropna().unique()))
    bolus_vars = ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe"]
    selected_bolus = st.multiselect("Exclude Cases Using Drugs:", bolus_vars)

    all_signals = sorted(set(df_trks['tname'].dropna().unique()))
    selected_signals = st.multiselect("Required Signals:", all_signals[:100], default=["BIS/BIS"])
    case_limit = st.radio("Number of Cases to Analyze:", [10, "All"], horizontal=True)

    selector = CaseSelector(df_cases, df_trks, ane_type, selected_signals, selected_bolus)
    valid_case_ids, filtered_cases = selector.select_valid_cases()

    st.success(f"✅ {len(valid_case_ids)} valid cases selected.")
    st.download_button("📥 Download Valid Case IDs", pd.DataFrame(valid_case_ids, columns=["caseid"]).to_csv(index=False), file_name="valid_case_ids.csv")

# Process only if valid
if valid_case_ids:
    if case_limit == 10:
        valid_case_ids = valid_case_ids[:10]

    all_data = [vitaldb.load_case(cid, selected_signals, interval=1) for cid in valid_case_ids]
    min_len = min([d.shape[0] for d in all_data])
    merged_data = np.concatenate([d[:min_len, :] for d in all_data], axis=0)

    global_medians = {var: np.nanmedian(merged_data[:, i]) for i, var in enumerate(selected_signals)}
    global_mads = {var: np.median(np.abs(merged_data[:, i] - global_medians[var])) or 1e-6 for i, var in enumerate(selected_signals)}

    with tabs[1]:
        st.header("Signal Analysis")
        analyzer = SignalAnalyzer("merged", merged_data, selected_signals, global_medians, global_mads)
        issues = analyzer.analyze()
        st.json(issues)

    with tabs[2]:
        st.header("Interpolation Results")
        processor = SignalProcessor(merged_data)
        interpolated = processor.interpolate()

        idx = st.selectbox("Select Signal for Plotting:", selected_signals)
        idx_num = selected_signals.index(idx)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(merged_data[:, idx_num], label="Raw")
        ax.plot(interpolated[:, idx_num], label="Interpolated", linestyle="--")
        ax.set_title(f"{idx} - Raw vs Interpolated")
        ax.legend()
        st.pyplot(fig)

    with tabs[3]:
        st.header("Statistical Summary")
        evaluator = Evaluator(merged_data, interpolated, selected_signals)
        stats_df = evaluator.compute_stats()
        st.dataframe(stats_df, use_container_width=True)
else:
    st.warning("No valid data to analyze.")
