# VitalDB Professional Streamlit GUI - Analyzer & Interpolator

import streamlit as st
import pandas as pd
import numpy as np
import vitaldb
from scipy.interpolate import interp1d
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("💉 VitalDB Professional Analyzer & Interpolator")

# ------------------ Load Data ------------------
@st.cache_data
def load_vitaldb():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    df_labs = pd.read_csv("https://api.vitaldb.net/labs")
    return df_cases, df_trks, df_labs

df_cases, df_trks, df_labs = load_vitaldb()

# ------------------ UI Inputs ------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    ane_type = st.selectbox("Anesthesia Type", df_cases['ane_type'].dropna().unique())
    remove_drugs = st.multiselect("Exclude Cases with These Drugs", 
                                   ['intraop_mdz', 'intraop_ftn', 'intraop_epi', 'intraop_phe', 'intraop_eph'])

    group_choice = st.radio("Select Variable Group", ["Group 1 (RFTN20)", "Group 2 (RFTN50)"])

    if group_choice == "Group 1 (RFTN20)":
        variable_names = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
                          "Orchestra/PPF20_CE", "Orchestra/RFTN20_CE", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"]
    else:
        variable_names = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
                          "Orchestra/PPF20_CE", "Orchestra/RFTN50_CE", "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"]

    gap_action = st.selectbox("Long Gap Handling", ["leave", "nan", "zero"])
    interp_method = st.selectbox("Interpolation Method", ["auto", "linear", "cubic", "slinear"])

# ------------------ Case Filtering ------------------
def select_valid_cases():
    df_filtered = df_cases[df_cases['ane_type'] == ane_type].copy()
    case_ids = set(df_filtered['caseid'])
    for var in variable_names:
        case_ids &= set(df_trks[df_trks['tname'] == var]['caseid'])
    if remove_drugs:
        cols = [col for col in remove_drugs if col in df_filtered.columns]
        df_filtered = df_filtered[~df_filtered[cols].gt(0).any(axis=1)]
        case_ids &= set(df_filtered['caseid'])
    return sorted(case_ids)

valid_ids = select_valid_cases()
st.success(f"✅ {len(valid_ids)} valid case(s) found.")

selected_caseid = st.selectbox("📦 Choose a CaseID to Analyze", valid_ids)

# ------------------ Analysis & Interpolation ------------------
def compute_stats(data, variable_names):
    medians, mads = {}, {}
    for i, var in enumerate(variable_names):
        sig = data[:, i]
        clean = sig[~np.isnan(sig)]
        medians[var] = np.median(clean)
        mads[var] = np.median(np.abs(clean - medians[var])) or 1e-6
    return medians, mads

def analyze_and_plot(caseid, variable_names):
    data = vitaldb.load_case(caseid, variable_names)
    df = pd.DataFrame(data, columns=variable_names)
    df['time'] = np.arange(len(df))

    medians, mads = compute_stats(data, variable_names)
    fig = make_subplots(rows=len(variable_names), cols=1, shared_xaxes=True, subplot_titles=variable_names)

    for i, var in enumerate(variable_names):
        sig = df[var]
        time = df['time']
        nan_idx = np.where(np.isnan(sig))[0]
        out_idx = np.where(np.abs(sig - medians[var]) > 3.5 * mads[var])[0]

        fig.add_trace(go.Scatter(x=time, y=sig, mode='lines', name=var), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=time[nan_idx], y=[sig.min()-5]*len(nan_idx), mode='markers',
                                 marker=dict(color='gray'), name=f'NaN - {var}'), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=time[out_idx], y=sig[out_idx], mode='markers',
                                 marker=dict(color='red', symbol='star', size=7), name=f'Outliers - {var}'), row=i+1, col=1)

    fig.update_layout(height=300*len(variable_names), title=f"Signal Quality - Case {caseid}")
    return df, fig

def interpolate(df, variable_names):
    global_std = {v: df[v].std() for v in variable_names}
    df_interp = df.copy()
    for var in variable_names:
        sig = df[var].values
        x = np.arange(len(sig))
        valid = ~np.isnan(sig)
        if valid.sum() < 2:
            continue
        if interp_method == "auto":
            std = np.std(sig[valid])
            if std < global_std[var]*0.7:
                method = "linear"
            elif std < global_std[var]*1.3:
                method = "cubic"
            else:
                method = "slinear"
        else:
            method = interp_method
        f = interp1d(x[valid], sig[valid], kind=method, fill_value="extrapolate")
        sig_filled = sig.copy()
        gap = np.isnan(sig)
        sig_filled[gap] = f(x[gap]) if gap_action != "leave" else sig[gap]
        df_interp[var] = sig_filled
    return df_interp

# ------------------ Run Analysis ------------------
if st.button("🔍 Run Analysis"):
    with st.spinner("Running full analysis and interpolation..."):
        df_raw, fig_raw = analyze_and_plot(selected_caseid, variable_names)
        st.plotly_chart(fig_raw, use_container_width=True)

        df_interp = interpolate(df_raw.copy(), variable_names)
        fig_interp = make_subplots(rows=len(variable_names), cols=1, shared_xaxes=True, subplot_titles=variable_names)

        for i, var in enumerate(variable_names):
            fig_interp.add_trace(go.Scatter(x=df_interp['time'], y=df_raw[var], mode='lines', name=f"Raw {var}"), row=i+1, col=1)
            fig_interp.add_trace(go.Scatter(x=df_interp['time'], y=df_interp[var], mode='lines', name=f"Imputed {var}"), row=i+1, col=1)

        fig_interp.update_layout(height=300*len(variable_names), title=f"🔁 Interpolation - Case {selected_caseid}")
        st.plotly_chart(fig_interp, use_container_width=True)

        st.download_button("📁 Download Interpolated CSV", df_interp.to_csv(index=False), file_name=f"case_{selected_caseid}_interpolated.csv")
