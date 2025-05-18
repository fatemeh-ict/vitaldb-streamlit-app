# vitaldb_streamlit_app.py - Multi-tab Streamlit Version with Case Export and Global Stats on Sample
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import vitaldb
import seaborn as sns
import os
import io

# ========== Classes ==========

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
        return sorted(list(valid_case_ids)), df_cases_filtered[df_cases_filtered['caseid'].isin(valid_case_ids)]

class SignalAnalyzer:
    def __init__(self, caseid, data, variable_names, global_medians=None, global_mads=None):
        self.caseid = caseid
        self.data = data
        self.variable_names = variable_names
        self.global_medians = global_medians or {}
        self.global_mads = global_mads or {}
        self.issues = {var: {'nan': [], 'outlier': [], 'jump': []} for var in variable_names}

    def analyze(self):
        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i]
            nan_idx = np.where(np.isnan(signal))[0]
            self.issues[var]['nan'] = nan_idx.tolist()
            mad = self.global_mads.get(var, 1e-6)
            median = self.global_medians.get(var, 0)
            outliers = np.where(np.abs(signal - median) > 3.5 * mad)[0]
            self.issues[var]['outlier'] = outliers.tolist()
            diffs = np.diff(signal)
            jump_idx = np.where(np.abs(diffs - np.median(diffs)) > 3.5 * mad)[0]
            self.issues[var]['jump'] = jump_idx.tolist()
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
        self.imputed_df = pd.DataFrame(imputed_data, columns=variable_names)
        self.variable_names = variable_names

    def compute_stats(self):
        stats = []
        for var in self.variable_names:
            raw = self.raw_df[var]
            imp = self.imputed_df[var]
            stats.append({
                'Variable': var,
                'Mean Before': raw.mean(),
                'Mean After': imp.mean(),
                'Median Before': raw.median(),
                'Median After': imp.median(),
                'Std Before': raw.std(),
                'Std After': imp.std(),
                'NaNs Before': raw.isna().sum(),
                'NaNs After': imp.isna().sum()
            })
        return pd.DataFrame(stats)

class StatisticsPlotter:
    def compare_stats(self, stats_df):
        fig, axs = plt.subplots(3, 1, figsize=(12, 10))
        x = np.arange(len(stats_df))
        axs[0].bar(x - 0.2, stats_df['Mean Before'], 0.4, label='Before')
        axs[0].bar(x + 0.2, stats_df['Mean After'], 0.4, label='After')
        axs[0].set_title("Mean Comparison")
        axs[0].set_xticks(x)
        axs[0].set_xticklabels(stats_df['Variable'], rotation=45)
        axs[0].legend()

        axs[1].bar(x - 0.2, stats_df['Median Before'], 0.4)
        axs[1].bar(x + 0.2, stats_df['Median After'], 0.4)
        axs[1].set_title("Median Comparison")
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(stats_df['Variable'], rotation=45)

        axs[2].bar(x - 0.2, stats_df['Std Before'], 0.4)
        axs[2].bar(x + 0.2, stats_df['Std After'], 0.4)
        axs[2].set_title("Std Deviation Comparison")
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(stats_df['Variable'], rotation=45)

        plt.tight_layout()
        return fig

# ========== App Setup ==========
st.set_page_config(page_title="VitalDB Analyzer Tabs", layout="wide")
st.title("🧠 VitalDB Multi-Tab Signal Analysis")

def load_metadata():
    return (
        pd.read_csv("https://api.vitaldb.net/cases"),
        pd.read_csv("https://api.vitaldb.net/trks")
    )

df_cases, df_trks = load_metadata()
group1 = ["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"]
group2 = ["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"]
drug_vars = ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe"]

# ========== Tabs ==========
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Case Selection", "📊 Signal Analysis", "🛠 Interpolation", "📈 Statistics"])

with tab1:
    st.subheader("Step 1: Select Case Filters")
    selected_ane = st.selectbox("Anesthesia Type:", sorted(df_cases['ane_type'].dropna().unique()))
    selected_drugs = st.multiselect("Exclude Cases Using Drugs:", drug_vars)
    selected_signals = st.multiselect("Required Signals:", sorted(set(group1 + group2)), default=group1)
    limit = st.radio("Number of Cases to Analyze:", [10, "All"], horizontal=True)

    selector = CaseSelector(df_cases, df_trks, ane_type=selected_ane, required_variables=selected_signals, intraoperative_boluses=selected_drugs)
    valid_ids, df_valid = selector.select_valid_cases()
    st.success(f"✅ {len(valid_ids)} valid cases selected.")
    st.download_button("📥 Download Filtered Case IDs", pd.DataFrame({"caseid": valid_ids}).to_csv(index=False), file_name="valid_cases.csv")

with tab2:
    st.subheader("Step 2: Analyze Signal Quality")
    if not valid_ids:
        st.warning("No valid cases. Adjust selection in Tab 1.")
        st.stop()
    sample_ids = valid_ids[:10] if limit == 10 else valid_ids
    all_data = [vitaldb.load_case(cid, selected_signals, interval=1) for cid in sample_ids]
    min_len = min(len(d) for d in all_data)
    merged_data = np.concatenate([d[:min_len] for d in all_data], axis=0)
    global_medians = {sig: np.nanmedian(merged_data[:, i]) for i, sig in enumerate(selected_signals)}
    global_mads = {sig: np.median(np.abs(merged_data[:, i] - global_medians[sig])) or 1e-6 for i, sig in enumerate(selected_signals)}
    analyzer = SignalAnalyzer("Merged", merged_data, selected_signals, global_medians, global_mads)
    issues = analyzer.analyze()
    st.json(issues)

with tab3:
    st.subheader("Step 3: Interpolate Missing Data and Evaluate")
    processor = SignalProcessor(merged_data)
    interpolated = processor.interpolate()
    evaluator = Evaluator(merged_data, interpolated, selected_signals)
    stats_df = evaluator.compute_stats()
    st.dataframe(stats_df, use_container_width=True)

with tab4:
    st.subheader("Step 4: Plot and Compare Statistics")
    plotter = StatisticsPlotter()
    st.pyplot(plotter.compare_stats(stats_df))
    selected_plot = st.selectbox("Choose Signal for Comparison Plot:", selected_signals)
    idx = selected_signals.index(selected_plot)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(merged_data[:, idx], label="Raw", alpha=0.5)
    ax.plot(interpolated[:, idx], label="Interpolated", linestyle="--")
    ax.set_title(f"{selected_plot} Before vs After Interpolation")
    ax.legend()
    st.pyplot(fig)
    st.success("✅ Pipeline Completed")
