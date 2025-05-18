import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vitaldb
from scipy.interpolate import interp1d

# ------------------------------
# Setup
st.set_page_config(page_title="VitalDB Pipeline Analyzer", layout="centered")
st.title("🧠 VitalDB Signal Analysis Pipeline")

# ------------------------------
# Configuration (فقط سیگنال‌های رایج)
variables = [
    "BIS/BIS",
    "Solar8000/NIBP_SBP",
    "Solar8000/NIBP_DBP",
    "Orchestra/PPF20_RATE"
]

# ------------------------------
# Helper Class to Select Valid Case IDs
def select_valid_case_ids(df_cases, df_trks, required_vars):
    valid_ids = set(df_cases['caseid'])
    for var in required_vars:
        case_ids_with_var = set(df_trks[df_trks['tname'] == var]['caseid'])
        valid_ids &= case_ids_with_var
    return sorted(list(valid_ids))[:10]

# ------------------------------
# Signal Processor
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
                    f = interp1d(x[mask], signal[mask], kind='linear', fill_value='extrapolate')
                    self.data[:, i] = f(x)
                except Exception:
                    continue
        return self.data

# ------------------------------
# Evaluator
class Evaluator:
    def __init__(self, raw_data, imputed_data):
        self.raw = raw_data
        self.imputed = imputed_data

    def summary(self, i):
        return {
            "Mean (raw)": round(np.nanmean(self.raw[:, i]), 2),
            "Mean (clean)": round(np.nanmean(self.imputed[:, i]), 2),
            "NaNs before": int(np.isnan(self.raw[:, i]).sum()),
            "NaNs after": int(np.isnan(self.imputed[:, i]).sum())
        }

# ------------------------------
# PipelineRunner
class PipelineRunner:
    def __init__(self, case_ids, variables):
        self.case_ids = case_ids
        self.variables = variables

    def run(self):
        all_data = []
        for cid in self.case_ids:
            try:
                data = vitaldb.load_case(cid, self.variables, interval=1)
                if isinstance(data, np.ndarray) and data.shape[0] > 0:
                    all_data.append(data)
            except:
                continue

        if not all_data:
            return None, 0

        min_len = min(d.shape[0] for d in all_data)
        trimmed_data = np.concatenate([d[:min_len, :] for d in all_data], axis=0)
        return trimmed_data, len(all_data)

# ------------------------------
# Load Metadata
try:
    with st.spinner("📦 Loading metadata..."):
        df_cases = pd.read_csv("https://api.vitaldb.net/cases")
        df_trks = pd.read_csv("https://api.vitaldb.net/trks")
except Exception as e:
    st.error(f"❌ Failed to load metadata: {e}")
    st.stop()

# ------------------------------
# Case Selection
case_ids = select_valid_case_ids(df_cases, df_trks, variables)
st.subheader("📁 Case Selection")
st.write(f"Found {len(case_ids)} valid case IDs.")

if not case_ids:
    st.error("⚠️ No valid cases. Try fewer variables.")
    st.stop()

# ------------------------------
# Run Pipeline
pipeline = PipelineRunner(case_ids, variables)
raw_data, used = pipeline.run()

if raw_data is None:
    st.error("❌ No data available after loading cases.")
    st.stop()

st.success(f"✅ Loaded data from {used} cases. Shape: {raw_data.shape}")

# ------------------------------
# Interpolate
processor = SignalProcessor(raw_data)
imputed_data = processor.interpolate_nans()

# ------------------------------
# Plot BIS/BIS
bis_index = variables.index("BIS/BIS")
bis_signal = imputed_data[:, bis_index]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(bis_signal, label="Interpolated BIS/BIS", color="blue", alpha=0.7)
ax.set_title("BIS/BIS Signal (Interpolated)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Value")
ax.legend()
st.pyplot(fig)

# ------------------------------
# Evaluation
evaluator = Evaluator(raw_data, imputed_data)
st.subheader("📊 Signal Summary")
st.write(evaluator.summary(bis_index))
