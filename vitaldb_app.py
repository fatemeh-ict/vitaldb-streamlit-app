import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vitaldb
from scipy.interpolate import interp1d

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
class SignalProcessor:
    def __init__(self, data):
        self.data = data.copy()

    def interpolate_nans(self):
        x = np.arange(self.data.shape[0])
        for i in range(self.data.shape[1]):
            signal = self.data[:, i]
            if np.isnan(signal).sum() > 0:
                mask = ~np.isnan(signal)
                try:
                    interp_func = interp1d(x[mask], signal[mask], kind='linear', fill_value="extrapolate")
                    self.data[:, i] = interp_func(x)
                except Exception as e:
                    st.warning(f"Interpolation failed for variable index {i}: {e}")
        return self.data

class Evaluator:
    def __init__(self, raw_data, imputed_data):
        self.raw_data = raw_data
        self.imputed_data = imputed_data

    def summary(self, var_index):
        raw = self.raw_data[:, var_index]
        imp = self.imputed_data[:, var_index]
        return {
            "Mean (raw)": round(np.nanmean(raw), 2),
            "Mean (imputed)": round(np.nanmean(imp), 2),
            "NaNs (before)": int(np.isnan(raw).sum()),
            "NaNs (after)": int(np.isnan(imp).sum())
        }

class PipelineRunner:
    def __init__(self, case_ids, variables):
        self.case_ids = case_ids
        self.variables = variables

    def run(self):
        all_data = []
        for cid in self.case_ids:
            try:
                data = vitaldb.load_case(cid, self.variables, interval=1)
                if data is not None and isinstance(data, np.ndarray) and data.shape[0] > 0:
                    all_data.append(data)
            except Exception as e:
                st.warning(f"Skipping case {cid} due to error: {e}")
                continue

        if not all_data:
            return None, 0

        min_len = min(d.shape[0] for d in all_data)
        trimmed_data = np.concatenate([d[:min_len, :] for d in all_data], axis=0)
        return trimmed_data, len(all_data)

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
try:
    with st.spinner("Loading metadata from VitalDB..."):
        df_cases = pd.read_csv("https://api.vitaldb.net/cases")
        df_trks = pd.read_csv("https://api.vitaldb.net/trks")
except Exception as e:
    st.error(f"Failed to load metadata: {e}")
    st.stop()

case_ids = select_valid_case_ids(df_cases, df_trks, variables)

st.subheader("🔧 Running Pipeline on Selected Cases")
st.write(f"Using {len(case_ids)} valid case IDs.")

if not case_ids:
    st.error("No valid case IDs found. Try reducing the number of required signals.")
    st.stop()

pipeline = PipelineRunner(case_ids, variables)
data, used = pipeline.run()

if data is None:
    st.error("No valid data found.")
    st.stop()

st.success(f"✅ Loaded {used} cases, shape: {data.shape}")

# Process signals
processor = SignalProcessor(data)
imputed = processor.interpolate_nans()

# Plot
bis_index = variables.index("BIS/BIS")
bis_data = imputed[:, bis_index]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(bis_data, label="BIS/BIS", color="skyblue")
ax.set_title("Interpolated BIS/BIS Signal")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Value")
ax.legend()
st.pyplot(fig)

# Evaluation
evaluator = Evaluator(raw_data=data, imputed_data=imputed)
st.subheader("📊 BIS/BIS Summary")
st.write(evaluator.summary(bis_index))
