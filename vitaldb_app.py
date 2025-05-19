# streamlit_vitaldb_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import vitaldb
from io import BytesIO
from scipy.interpolate import interp1d

# ------------------- Step 1: Filter Cases -------------------
st.title("VitalDB Signal Processing App")
st.sidebar.header("Case Filtering Options")

with st.sidebar.expander("1. Filter Settings"):
    ane_type = st.selectbox("Anesthesia Type", ["General", "Spinal", "Epidural"])
    required_vars_group1 = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
                            "Orchestra/PPF20_CE", "Orchestra/RFTN20_CE", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"]
    required_vars_group2 = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
                            "Orchestra/PPF20_CE", "Orchestra/RFTN50_CE", "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"]
    boluses_to_exclude = st.multiselect("Exclude if intraoperative boluses present:",
                                        ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"])

@st.cache_data(show_spinner=True)
def load_data():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    return df_cases, df_trks

df_cases, df_trks = load_data()

def filter_cases(df_cases, df_trks, group1_vars, group2_vars, boluses, ane_type):
    df = df_cases[df_cases['ane_type'] == ane_type].copy()
    ids1 = set(df[df_trks['tname'].isin(group1_vars)]['caseid'])
    ids2 = set(df[df_trks['tname'].isin(group2_vars)]['caseid'])
    combined_ids = sorted(list(ids1.union(ids2)))
    df_filtered = df[df['caseid'].isin(combined_ids)].copy()
    if boluses:
        df_filtered = df_filtered[~df_filtered[boluses].gt(0).any(axis=1)]
    return df_filtered, combined_ids

filtered_cases, valid_ids = filter_cases(df_cases, df_trks, required_vars_group1, required_vars_group2, boluses_to_exclude, ane_type)
df_trks_filtered = df_trks[df_trks['caseid'].isin(valid_ids)].copy()

st.subheader("Filtered Dataset Overview")
st.write(f"Total valid cases: {len(valid_ids)}")
st.write("Filtered cases preview:")
st.dataframe(filtered_cases.head())

# Download filtered data
csv = filtered_cases.to_csv(index=False).encode('utf-8')
st.download_button("Download Filtered Cases CSV", data=csv, file_name='filtered_cases.csv')

# ------------------- Step 2: Analyze Signals -------------------
st.subheader("2. Analyze Signal Quality")
selected_id = st.selectbox("Choose Case ID to Analyze", valid_ids[:20])
variables = ["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "Orchestra/RFTN50_RATE"]

@st.cache_data(show_spinner=True)
def load_case_data(caseid):
    return vitaldb.load_case(caseid, variables, interval=1)

data = load_case_data(selected_id)
st.write(f"Signal shape: {data.shape}")

fig, axs = plt.subplots(len(variables), 1, figsize=(10, len(variables)*2), sharex=True)
for i, var in enumerate(variables):
    axs[i].plot(data[:, i], label=var)
    axs[i].legend()
    axs[i].grid(True)
st.pyplot(fig)

# ------------------- Step 3: Interpolation -------------------
st.subheader("3. Interpolation Preview")
def linear_interpolate(signal):
    x = np.arange(len(signal))
    valid = ~np.isnan(signal)
    if valid.sum() < 2:
        return signal
    f = interp1d(x[valid], signal[valid], kind='linear', fill_value='extrapolate')
    return f(x)

data_interp = np.copy(data)
for i in range(data.shape[1]):
    sig = data[:, i]
    sig[sig == 0] = np.nan  # handle BIS=0 issue
    data_interp[:, i] = linear_interpolate(sig)

fig2, axs2 = plt.subplots(len(variables), 1, figsize=(10, len(variables)*2), sharex=True)
for i, var in enumerate(variables):
    axs2[i].plot(data[:, i], 'o-', alpha=0.5, label='Raw')
    axs2[i].plot(data_interp[:, i], '-', label='Interpolated')
    axs2[i].legend()
    axs2[i].grid(True)
st.pyplot(fig2)

# ------------------- Step 4: Stats Comparison -------------------
st.subheader("4. Summary Statistics")
def compute_stats(raw, interp):
    rows = []
    for i, var in enumerate(variables):
        raw_sig = raw[:, i]
        interp_sig = interp[:, i]
        rows.append({
            'variable': var,
            'mean_raw': np.nanmean(raw_sig),
            'mean_interp': np.nanmean(interp_sig),
            'std_raw': np.nanstd(raw_sig),
            'std_interp': np.nanstd(interp_sig),
            'nan_raw': np.isnan(raw_sig).sum(),
            'nan_interp': np.isnan(interp_sig).sum()
        })
    return pd.DataFrame(rows)

stats_df = compute_stats(data, data_interp)
st.dataframe(stats_df)

# ------------------- Step 5: Final Export -------------------
st.subheader("5. Download Interpolated Data")
raw_csv = pd.DataFrame(data, columns=variables).to_csv(index=False).encode('utf-8')
interp_csv = pd.DataFrame(data_interp, columns=variables).to_csv(index=False).encode('utf-8')
st.download_button("Download Raw Signals", raw_csv, file_name="raw_signals.csv")
st.download_button("Download Interpolated Signals", interp_csv, file_name="interpolated_signals.csv")
