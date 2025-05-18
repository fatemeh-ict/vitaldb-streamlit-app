import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vitaldb
from scipy.interpolate import interp1d
from collections import defaultdict

st.set_page_config(page_title="VitalDB Analyzer", layout="wide")
st.title("🧠 VitalDB Signal Analysis Pipeline")

@st.cache_data(show_spinner=False)
def load_metadata():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    return df_cases, df_trks

df_cases, df_trks = load_metadata()

group1 = ["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"]
group2 = ["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"]
all_signals = list(set(group1 + group2))

# Filtering helper
def filter_cases(df_trks, signals):
    valid = set(df_trks['caseid'].unique())
    for s in signals:
        ids = set(df_trks[df_trks['tname'] == s]['caseid'])
        valid &= ids
    return valid

# Analyzer
class SignalAnalyzer:
    def __init__(self, data, variables, global_medians, global_mads):
        self.data = data
        self.variables = variables
        self.global_medians = global_medians
        self.global_mads = global_mads
        self.issues = defaultdict(dict)

    def analyze(self):
        for i, var in enumerate(self.variables):
            x = self.data[:, i]
            self.issues[var]['NaNs'] = int(np.isnan(x).sum())
            diff = np.diff(x)
            jump = np.where(np.abs(diff - np.median(diff)) > 3.5 * self.global_mads[var])[0]
            self.issues[var]['Jumps'] = int(len(jump))
            out = np.where(np.abs(x - self.global_medians[var]) > 3.5 * self.global_mads[var])[0]
            self.issues[var]['Outliers'] = int(len(out))
        return self.issues

# Processor
class SignalProcessor:
    def __init__(self, data):
        self.data = data

    def interpolate(self):
        x = np.arange(self.data.shape[0])
        new_data = self.data.copy()
        for i in range(new_data.shape[1]):
            sig = new_data[:, i]
            if np.isnan(sig).sum():
                valid = ~np.isnan(sig)
                f = interp1d(x[valid], sig[valid], kind='linear', fill_value='extrapolate')
                new_data[:, i] = f(x)
        return new_data

# Evaluator
class Evaluator:
    def __init__(self, raw, interpolated, variables):
        self.raw = raw
        self.interpolated = interpolated
        self.variables = variables

    def compute(self):
        summary = []
        for i, var in enumerate(self.variables):
            summary.append({
                'Signal': var,
                'Mean Before': np.nanmean(self.raw[:, i]),
                'Mean After': np.nanmean(self.interpolated[:, i]),
                'Median Before': np.nanmedian(self.raw[:, i]),
                'Median After': np.nanmedian(self.interpolated[:, i]),
                'NaNs Before': int(np.isnan(self.raw[:, i]).sum()),
                'NaNs After': int(np.isnan(self.interpolated[:, i]).sum()),
                'Std Before': np.nanstd(self.raw[:, i]),
                'Std After': np.std(self.interpolated[:, i])
            })
        return pd.DataFrame(summary)

# TABS
tabs = st.tabs(["🔎 Case Selection", "📊 Signal Quality", "🛠 Interpolation", "📈 Dataset Stats"])

with tabs[0]:
    st.header("1️⃣ Select and Filter Cases")
    drug_vars = ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe"]
    selected_signals = st.multiselect("Select signals:", all_signals, default=all_signals)
    ane_filter = st.multiselect("Filter by anesthesia type (ane_type):", sorted(df_cases['ane_type'].dropna().unique()))
    exclude_drugs = st.multiselect("Exclude if these drugs used:", drug_vars)
    limit_choice = st.radio("Run on:", [10, "All"], index=0)

    ids1 = filter_cases(df_trks, group1)
    ids2 = filter_cases(df_trks, group2)
    union_ids = list(ids1.union(ids2))
    df_filtered = df_cases[df_cases['caseid'].isin(union_ids)]
    if ane_filter:
        df_filtered = df_filtered[df_filtered['ane_type'].isin(ane_filter)]
    if exclude_drugs:
        df_filtered = df_filtered[df_filtered[exclude_drugs].sum(axis=1) == 0]
    final_ids = df_filtered['caseid'].tolist()
    if limit_choice == 10:
        final_ids = final_ids[:10]
    st.success(f"✅ {len(final_ids)} cases selected.")

with tabs[1]:
    st.header("2️⃣ Signal Quality Analysis")
    if not final_ids:
        st.warning("No cases available.")
        st.stop()

    data_all = [vitaldb.load_case(cid, selected_signals, interval=1) for cid in final_ids]
    min_len = min(d.shape[0] for d in data_all)
    data_trimmed = np.concatenate([d[:min_len, :] for d in data_all], axis=0)
    global_medians = {sig: np.median(data_trimmed[:, i][~np.isnan(data_trimmed[:, i])]) for i, sig in enumerate(selected_signals)}
    global_mads = {sig: np.median(np.abs(data_trimmed[:, i][~np.isnan(data_trimmed[:, i])] - global_medians[sig])) or 1e-6 for i, sig in enumerate(selected_signals)}

    analyzer = SignalAnalyzer(data_trimmed, selected_signals, global_medians, global_mads)
    result = analyzer.analyze()
    st.json(result)

    show_case = st.selectbox("Plot one case ID:", final_ids)
    case_data = vitaldb.load_case(show_case, selected_signals, interval=1)
    for i, sig in enumerate(selected_signals):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(case_data[:, i], label=sig)
        ax.set_title(f"{sig} - Raw")
        ax.legend()
        st.pyplot(fig)

with tabs[2]:
    st.header("3️⃣ Interpolation & Comparison")
    processor = SignalProcessor(data_trimmed)
    interpolated = processor.interpolate()

    evaluator = Evaluator(data_trimmed, interpolated, selected_signals)
    df_eval = evaluator.compute()
    st.dataframe(df_eval, use_container_width=True)

    compare_sig = st.selectbox("Compare Before/After:", selected_signals)
    idx = selected_signals.index(compare_sig)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(data_trimmed[:, idx], label="Raw")
    ax.plot(interpolated[:, idx], label="Interpolated", linestyle="--")
    ax.legend()
    ax.set_title(f"{compare_sig} - Comparison")
    st.pyplot(fig)

with tabs[3]:
    st.header("4️⃣ Dataset Overview")
    df_used = df_cases[df_cases['caseid'].isin(final_ids)]
    st.subheader("Numerical Description")
    st.dataframe(df_used.select_dtypes(include=np.number).describe().T)

    st.subheader("Categorical Frequencies")
    for cat in ['sex', 'ane_type', 'department', 'position']:
        if cat in df_used:
            st.markdown(f"**{cat}**")
            st.dataframe(df_used[cat].value_counts())
