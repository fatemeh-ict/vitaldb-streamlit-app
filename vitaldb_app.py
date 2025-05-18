import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vitaldb

# ------------------------------
# Settings
st.set_page_config(page_title="VitalDB BIS/BIS Analysis", layout="centered")
st.title("📊 BIS/BIS Analysis with Global Median")

# ------------------------------
# Configuration
variables = [
    "BIS/BIS",
    "Solar8000/NIBP_SBP",
    "Solar8000/NIBP_DBP",
    "Orchestra/PPF20_RATE",
    "Orchestra/RFTN20_RATE",
    "Orchestra/RFTN50_RATE"
]

# Load metadata
with st.spinner("Loading metadata from VitalDB..."):
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")

# Filter first 10 valid case IDs
def select_valid_case_ids(df_cases, df_trks, required_vars):
    valid_ids = set(df_cases['caseid'])
    for var in required_vars:
        case_ids_with_var = set(df_trks[df_trks['tname'] == var]['caseid'])
        valid_ids &= case_ids_with_var
    return sorted(list(valid_ids))[:10]

case_ids = select_valid_case_ids(df_cases, df_trks, variables)

# Load all data for selected cases
all_data = []
for cid in case_ids:
    data = vitaldb.load_case(cid, variables, interval=1)
    all_data.append(data)

# Trim to shortest length and concatenate
min_len = min(d.shape[0] for d in all_data)
trimmed_data = np.concatenate([d[:min_len, :] for d in all_data], axis=0)

# Compute global medians and MADs
global_medians = {}
global_mads = {}
for i, var in enumerate(variables):
    sig = trimmed_data[:, i]
    sig = sig[~np.isnan(sig)]
    median = np.median(sig)
    mad = np.median(np.abs(sig - median)) or 1e-6
    global_medians[var] = median
    global_mads[var] = mad

# Show data overview
st.subheader("📋 Global Statistics")
st.write(f"Number of cases: {len(case_ids)}")
st.write(f"Trimmed data shape: {trimmed_data.shape}")
st.write(f"Global median for BIS/BIS: {global_medians['BIS/BIS']:.2f}")

# Plot histogram for BIS/BIS
bis_index = variables.index("BIS/BIS")
bis_data = trimmed_data[:, bis_index]
bis_data = bis_data[~np.isnan(bis_data)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(bis_data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax.axvline(global_medians['BIS/BIS'], color='red', linestyle='--', linewidth=2, label=f"Global Median = {global_medians['BIS/BIS']:.2f}")
ax.set_xlabel("BIS/BIS values")
ax.set_ylabel("Frequency")
ax.set_title("Histogram of BIS/BIS with Global Median")
ax.legend()
st.pyplot(fig)

# Summary stats
st.subheader("📊 Summary Statistics")
st.write({
    "Mean": round(np.mean(bis_data), 2),
    "Median": round(np.median(bis_data), 2),
    "Std Dev": round(np.std(bis_data), 2),
    "NaNs": int(len(trimmed_data[:, bis_index]) - len(bis_data))
})
