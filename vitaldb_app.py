import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vitaldb
from scipy.interpolate import interp1d

st.set_page_config(layout="wide")
st.title("VitalDB Analyzer with Outlier, NaN, and Interpolation Analysis")

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

# -------------------- Interpolation --------------------
def detect_gaps(signal):
    gaps = []
    is_gap = False
    start_idx = None
    for idx, value in enumerate(signal):
        if np.isnan(value):
            if not is_gap:
                is_gap = True
                start_idx = idx
        else:
            if is_gap:
                gaps.append({'start': start_idx, 'length': idx - start_idx})
                is_gap = False
    if is_gap:
        gaps.append({'start': start_idx, 'length': len(signal) - start_idx})
    return gaps

def classify_gaps(gaps):
    if not gaps:
        return []
    lengths = np.array([gap['length'] for gap in gaps])
    median_len = np.median(lengths)
    mad_len = np.median(np.abs(lengths - median_len)) or 1e-6
    for gap in gaps:
        gap['type'] = 'long' if gap['length'] > median_len + 3.5 * mad_len else 'short'
    return gaps

def select_interpolation_method(signal, global_std=10):
    clean = signal[~np.isnan(signal)]
    if len(clean) < 5:
        return 'linear'
    std = np.std(clean)
    if std < global_std * 0.7:
        return 'linear'
    elif std < global_std * 1.3:
        return 'cubic'
    else:
        return 'slinear'

def impute_signal(signal, gaps_classified, global_std):
    signal = signal.copy()
    x = np.arange(len(signal))
    for gap in gaps_classified:
        start, end = gap['start'], gap['start'] + gap['length']
        if gap['type'] == 'short':
            valid_idx = ~np.isnan(signal)
            if valid_idx.sum() < 2:
                continue
            method = select_interpolation_method(signal, global_std)
            f = interp1d(x[valid_idx], signal[valid_idx], kind=method, fill_value='extrapolate')
            signal[start:end] = f(x[start:end])
    return signal

def process_case_interpolation(caseid, variable_names, global_std):
    data = vitaldb.load_case(caseid, variable_names)
    if data is None:
        return None, None
    df = pd.DataFrame(data, columns=variable_names)
    df_interp = df.copy()
    for var in variable_names:
        sig = df[var].values
        gaps = detect_gaps(sig)
        classified = classify_gaps(gaps)
        df_interp[var] = impute_signal(sig, classified, global_std)
    return df, df_interp

def plot_before_after(df_raw, df_interp, variable_names, caseid):
    fig = make_subplots(rows=len(variable_names), cols=1, shared_xaxes=True, subplot_titles=variable_names)
    for i, var in enumerate(variable_names):
        x = np.arange(len(df_raw))
        fig.add_trace(go.Scatter(x=x, y=df_raw[var], mode='lines+markers', name=f"Raw {var}"), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df_interp[var], mode='lines', name=f"Imputed {var}"), row=i+1, col=1)
    fig.update_layout(height=300 * len(variable_names), title=f"Case {caseid} - Raw vs Imputed")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Streamlit UI --------------------
df_cases, df_trks, df_labs = load_data()
filter_tab, analysis_tab, interp_tab = st.tabs(["Filter & Download", "Signal Analysis", "Interpolation"])

# -------------------- Filter Tab --------------------
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
            st.session_state['filtered_ids'] = valid_caseids
            st.success(f"{len(valid_caseids)} valid cases found.")
        else:
            st.warning("No valid cases found.")

# -------------------- Signal Analysis Tab --------------------
with analysis_tab:
    st.subheader("Step 2: Analyze Case")
    if 'filtered_ids' not in st.session_state or not st.session_state['filtered_ids']:
        st.warning("Please filter the cases in Step 1 before analysis.")
    else:
        variable_names = ["Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "BIS/BIS",
                          "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP"]
        filtered_ids = st.session_state['filtered_ids']
        selected_caseid = st.selectbox("Select Case ID", filtered_ids)
        if st.button("Run Analysis"):
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

# -------------------- Interpolation Tab --------------------
with interp_tab:
    st.subheader("Step 3: Interpolation")
    if 'filtered_ids' not in st.session_state or not st.session_state['filtered_ids']:
        st.warning("Please filter the cases in Step 1 before interpolation.")
    else:
        variable_names = ["Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "BIS/BIS",
                          "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP"]
        filtered_ids = st.session_state['filtered_ids']
        selected_caseid = st.selectbox("Select Case ID for Interpolation", filtered_ids)
        if st.button("Run Interpolation"):
            with st.spinner("Interpolating and plotting..."):
                global_std = 10
                df_raw, df_interp = process_case_interpolation(selected_caseid, variable_names, global_std)
                if df_raw is not None:
                    plot_before_after(df_raw, df_interp, variable_names, selected_caseid)
                    st.success("Interpolation complete.")
                else:
                    st.error("Failed to load the selected case.")
