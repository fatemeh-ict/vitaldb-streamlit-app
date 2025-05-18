import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vitaldb
from scipy.interpolate import interp1d
from collections import defaultdict

# ---------------------------
st.set_page_config(page_title="VitalDB Full Analyzer", layout="wide")
st.title("🧠 Full VitalDB Signal Pipeline")

# ---------------------------
@st.cache_data(show_spinner=False)
def load_metadata():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    return df_cases, df_trks

df_cases, df_trks = load_metadata()

# ---------------------------
# Group Definitions
group1 = ["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"]
group2 = ["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"]
all_signals = list(set(group1 + group2))

# ---------------------------
def filter_cases(df_trks, signals):
    valid = set(df_trks['caseid'].unique())
    for s in signals:
        ids = set(df_trks[df_trks['tname'] == s]['caseid'])
        valid &= ids
    return valid

# ---------------------------
# Classes
class SignalAnalyzer:
    def __init__(self, data, variable_names, global_medians, global_mads):
        self.data = data
        self.variable_names = variable_names
        self.global_medians = global_medians
        self.global_mads = global_mads
        self.issues = defaultdict(dict)

    def analyze(self):
        for i, var in enumerate(self.variable_names):
            x = self.data[:, i]
            nan_idx = np.isnan(x)
            self.issues[var]['nan_count'] = nan_idx.sum()

            diff = np.diff(x)
            mad_val = self.global_mads[var] or 1e-6
            jump_idx = np.where(np.abs(diff - np.median(diff)) > 3.5 * mad_val)[0]
            self.issues[var]['jump_count'] = len(jump_idx)

            if "NIBP" in var:
                out_idx = np.where((x <= 0) | (np.abs(x - self.global_medians[var]) > 3.5 * mad_val))[0]
            else:
                out_idx = np.where(np.abs(x - self.global_medians[var]) > 3.5 * mad_val)[0]
            self.issues[var]['outlier_count'] = len(out_idx)

        return self.issues

class SignalProcessor:
    def __init__(self, data):
        self.raw = data

    def interpolate(self):
        x = np.arange(self.raw.shape[0])
        data_interp = self.raw.copy()
        for i in range(data_interp.shape[1]):
            sig = data_interp[:, i]
            if np.isnan(sig).sum():
                valid = ~np.isnan(sig)
                if valid.sum() >= 2:
                    f = interp1d(x[valid], sig[valid], kind='linear', fill_value='extrapolate')
                    data_interp[:, i] = f(x)
        return data_interp

class Evaluator:
    def __init__(self, raw, imputed, variables):
        self.raw = raw
        self.imputed = imputed
        self.variables = variables

    def compute(self):
        result = []
        for i, var in enumerate(self.variables):
            result.append({
                'Signal': var,
                'Mean Before': np.nanmean(self.raw[:, i]),
                'Mean After': np.nanmean(self.imputed[:, i]),
                'Median Before': np.nanmedian(self.raw[:, i]),
                'Median After': np.median(self.imputed[:, i]),
                'NaNs Before': np.isnan(self.raw[:, i]).sum(),
                'NaNs After': np.isnan(self.imputed[:, i]).sum(),
                'Std Before': np.nanstd(self.raw[:, i]),
                'Std After': np.std(self.imputed[:, i])
            })
        return pd.DataFrame(result)

# ---------------------------
tabs = st.tabs(["🧬 Select & Filter", "🔍 Signal Quality", "🛠 Interpolation", "📊 Dataset Stats"])

with tabs[0]:
    st.header("Step 1: Select Variables and Filter")
    drug_vars = ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe"]
    selected_signals = st.multiselect("Select signals (intersection of group 1 & 2):", all_signals, default=all_signals)
    selected_optype = st.multiselect("Select operation type:", sorted(df_cases['optype'].dropna().unique()))
    excluded_drugs = st.multiselect("Exclude cases with these drugs:", drug_vars)
    case_limit = st.radio("Number of cases to use:", [10, "All"], index=0)

    ids1 = filter_cases(df_trks, group1)
    ids2 = filter_cases(df_trks, group2)
    all_valid_ids = list(ids1.union(ids2))
    df_valid = df_cases[df_cases['caseid'].isin(all_valid_ids)]
    if selected_optype:
        df_valid = df_valid[df_valid['optype'].isin(selected_optype)]
    if excluded_drugs:
        df_valid = df_valid[df_valid[excluded_drugs].sum(axis=1) == 0]

    final_ids = df_valid['caseid'].tolist()
    if case_limit == 10:
        final_ids = final_ids[:10]

    st.success(f"Selected {len(final_ids)} cases.")

with tabs[1]:
    st.header("Step 2: Signal Quality Analysis")
    if not final_ids:
        st.warning("No valid case IDs selected.")
        st.stop()

    all_data = [vitaldb.load_case(cid, selected_signals, interval=1) for cid in final_ids]
    min_len = min(d.shape[0] for d in all_data)
    trimmed = np.concatenate([d[:min_len, :] for d in all_data], axis=0)

    global_medians = {sig: np.median(trimmed[:, i][~np.isnan(trimmed[:, i])]) for i, sig in enumerate(selected_signals)}
    global_mads = {sig: np.median(np.abs(trimmed[:, i][~np.isnan(trimmed[:, i])] - global_medians[sig])) or 1e-6 for i, sig in enumerate(selected_signals)}

    analyzer = SignalAnalyzer(trimmed, selected_signals, global_medians, global_mads)
    issue_stats = analyzer.analyze()

    selected_case = st.selectbox("Choose a case ID to plot:", final_ids)
    data_plot = vitaldb.load_case(selected_case, selected_signals, interval=1)
    for i, sig in enumerate(selected_signals):
        fig, ax = plt.subplots(figsize=(10, 2.5))
        ax.plot(data_plot[:, i], label=sig)
        ax.set_title(f"{sig} - Raw")
        ax.legend()
        st.pyplot(fig)
    st.json(issue_stats)

with tabs[2]:
    st.header("Step 3: Interpolation & Alignment")
    processor = SignalProcessor(trimmed)
    interpolated = processor.interpolate()

    evaluator = Evaluator(trimmed, interpolated, selected_signals)
    df_eval = evaluator.compute()
    st.dataframe(df_eval, use_container_width=True)

    chosen = st.selectbox("Choose a signal to plot before/after:", selected_signals)
    idx = selected_signals.index(chosen)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(trimmed[:, idx], label='Raw', alpha=0.6)
    ax.plot(interpolated[:, idx], label='Interpolated', linestyle='--')
    ax.set_title(f"{chosen} - Before vs After")
    ax.legend()
    st.pyplot(fig)

with tabs[3]:
    st.header("Step 4: Dataset Summary")
    df_used = df_cases[df_cases['caseid'].isin(final_ids)]
    st.subheader("Numerical Stats")
    st.dataframe(df_used.select_dtypes(include=np.number).describe().T)
    st.subheader("Categorical Counts")
    for col in ['sex', 'optype', 'department', 'approach', 'position']:
        st.markdown(f"**{col}**")
        st.dataframe(df_used[col].value_counts())
