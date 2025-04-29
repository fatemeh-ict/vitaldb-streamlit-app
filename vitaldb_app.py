# VitalDB Streamlit Analyzer - Final Professional Version with Interpolation Tab

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

# -------------------- Interpolation Utilities --------------------
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
    classified = []
    for gap in gaps:
        gap['type'] = 'long' if gap['length'] > median_len + 3.5 * mad_len else 'short'
        classified.append(gap)
    return classified

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

def align_signals_by_start(df, variable_names):
    start_indices = [df[var].first_valid_index() for var in variable_names if df[var].first_valid_index() is not None]
    if not start_indices:
        return df
    min_start = max(start_indices)
    return df.iloc[min_start:].reset_index(drop=True)

def impute_signal(signal, gaps_classified, global_std, action_long='leave'):
    signal = signal.copy()
    x = np.arange(len(signal))
    for gap in gaps_classified:
        start, end = gap['start'], gap['start'] + gap['length']
        if gap['type'] == 'short':
            valid_idx = (~np.isnan(signal))
            if valid_idx.sum() < 2:
                continue
            method_short = select_interpolation_method(signal, global_std)
            f = interp1d(x[valid_idx], signal[valid_idx], kind=method_short, fill_value='extrapolate')
            signal[start:end] = f(x[start:end])
        elif gap['type'] == 'long':
            if action_long == 'zero':
                signal[start:end] = 0
            elif action_long == 'nan':
                signal[start:end] = np.nan
            elif action_long == 'leave':
                pass
    return signal

def process_case_interpolation(caseid, variable_names, global_std, interval=1):
    raw_data = vitaldb.load_case(caseid, variable_names, interval=interval)
    if raw_data is None:
        return None, None
    df_signal = pd.DataFrame(raw_data, columns=variable_names)
    df_signal = align_signals_by_start(df_signal, variable_names)
    df_imputed = df_signal.copy()
    for var in variable_names:
        sig = df_signal[var].values
        gaps = detect_gaps(sig)
        classified = classify_gaps(gaps)
        df_imputed[var] = impute_signal(sig, classified, global_std)
    return df_signal, df_imputed

def plot_before_after(df_raw, df_imputed, variable_names, caseid):
    fig = make_subplots(rows=len(variable_names), cols=1, shared_xaxes=True, subplot_titles=variable_names)
    for i, var in enumerate(variable_names):
        x = np.arange(len(df_raw))
        fig.add_trace(go.Scatter(x=x, y=df_raw[var], mode='lines+markers', name=f"Raw {var}"), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df_imputed[var], mode='lines', name=f"Imputed {var}"), row=i+1, col=1)
    fig.update_layout(height=300 * len(variable_names), title=f"Case {caseid} - Raw vs Imputed")
    st.plotly_chart(fig, use_container_width=True)

# -------------------- UI --------------------
df_cases, df_trks, df_labs = load_data()
filter_tab, analysis_tab, interp_tab = st.tabs(["Filter & Download", "Signal Analysis", "Interpolation"])

# Filter Tab (unchanged)
# Analysis Tab (unchanged)

with interp_tab:
    st.subheader("Step 3: Interpolation Analysis")
    variable_names = ["Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "BIS/BIS",
                      "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP"]
    filtered_ids = st.session_state.get("filtered_ids", [])
    selected_caseid = st.selectbox("Select Case ID for Interpolation", filtered_ids if filtered_ids else ["No Data"])

    if st.button("Run Interpolation") and selected_caseid != "No Data":
        with st.spinner("Interpolating and Plotting..."):
            global_std = 10
            df_raw, df_imputed = process_case_interpolation(selected_caseid, variable_names, global_std)
            if df_raw is not None and df_imputed is not None:
                plot_before_after(df_raw, df_imputed, variable_names, selected_caseid)
                st.success("Interpolation completed successfully.")
            else:
                st.error("Failed to load or process the case.")
