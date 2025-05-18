import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vitaldb

# ------------------------------
# Settings
st.set_page_config(page_title="VitalDB Pipeline Analyzer", layout="centered")
st.title("🧠 VitalDB Signal Analysis Pipeline")

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

# ------------------------------
# Classes
class PipelineRunner:
    def __init__(self, case_ids, variables):
        self.case_ids = case_ids
        self.variables = variables
        self.global_medians = {}
        self.global_mads = {}
        self.results = []

    def run(self):
        all_data = []
        for cid in self.case_ids:
            try:
                data = vitaldb.load_case(cid, self.variables, interval=1)
                if data is not None and data.shape[0] > 0:
                    all_data.append(data)
            except:
                continue

        if not all_data:
            return None, None, 0

        min_len = min(d.shape[0] for d in all_data)
        trimmed_data = np.concatenate([d[:min_len, :] for d in all_data], axis=0)

        for i, var in enumerate(self.variables):
            sig = trimmed_data[:, i]
            sig = sig[~np.isnan(sig)]
            median = np.median(sig)
            mad = np.median(np.abs(sig - median)) or 1e-6
            self.global_medians[var] = median
            self.global_mads[var] = mad

        return trimmed_data, self.global_medians, len(all_data)

# ------------------------------
# Helper function to filter valid cases
def select_valid_case_ids(df_cases, df_trks, required_vars):
    valid_ids = set(df_cases['caseid'])
    for var in required_vars:
        case_ids_with_var = set(df_trks[df_trks['tname'] == var]['caseid'])
        valid_ids &= case_ids_with_var
    return sorted(list(valid_ids))[:10]

# ------------------------------
# Load metadata
with st.spinner("Loading metadata from VitalDB..."):
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")

case_ids = select_valid_case_ids(df_cases, df_trks, variables)

st.subheader("🔧 Running Pipeline on Selected Cases")
st.write(f"Using {len(case_ids)} valid case IDs.")

if not case_ids:
    st.error("No valid case IDs found. Try reducing the number of required signals.")
    st.stop()

runner = PipelineRunner(case_ids, variables)
trimmed_data, global_medians, used_cases = runner.run()

if trimmed_data is not None:
    st.success(f"✅ Loaded and analyzed {used_cases} cases")

    st.subheader("📋 Global Statistics")
    st.write(f"Trimmed data shape: {trimmed_data.shape}")
    st.write(f"Global median for BIS/BIS: {global_medians['BIS/BIS']:.2f}")

    bis_index = variables.index("BIS/BIS")
    bis_data = trimmed_data[:, bis_index]
    bis_data = bis_data[~np.isnan(bis_data)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(bis_data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(global_medians['BIS/BIS'], color='red', linestyle='--', linewidth=2,
               label=f"Global Median = {global_medians['BIS/BIS']:.2f}")
    ax.set_xlabel("BIS/BIS values")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of BIS/BIS with Global Median")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 Summary Statistics")
    st.write({
        "Mean": round(np.mean(bis_data), 2),
        "Median": round(np.median(bis_data), 2),
        "Std Dev": round(np.std(bis_data), 2),
        "NaNs": int(len(trimmed_data[:, bis_index]) - len(bis_data))
    })
else:
    st.error("No valid data found.")
