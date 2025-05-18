import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vitaldb
from scipy.interpolate import interp1d

# ------------------------------
# Page setup
st.set_page_config(page_title="VitalDB Dual Group Analyzer", layout="centered")
st.title("🧠 VitalDB Dual Group Signal Analysis")

# ------------------------------
# Define groups
group1_signals = [
    "BIS/BIS",
    "Solar8000/NIBP_SBP",
    "Solar8000/NIBP_DBP",
    "Orchestra/PPF20_RATE",
    "Orchestra/RFTN20_RATE"
]
group2_signals = [
    "BIS/BIS",
    "Solar8000/NIBP_SBP",
    "Solar8000/NIBP_DBP",
    "Orchestra/PPF20_RATE",
    "Orchestra/RFTN50_RATE"
]

# ------------------------------
# Load metadata
@st.cache_data(show_spinner=False)
def load_metadata():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    return df_cases, df_trks

df_cases, df_trks = load_metadata()

# ------------------------------
# Function to select case IDs based on signals
def filter_cases(signals):
    valid_ids = set(df_cases['caseid'])
    for sig in signals:
        trk_ids = set(df_trks[df_trks['tname'] == sig]['caseid'])
        valid_ids &= trk_ids
    return valid_ids

# ------------------------------
# Merge case IDs from both groups
valid_ids_1 = filter_cases(group1_signals)
valid_ids_2 = filter_cases(group2_signals)
merged_ids = sorted(list(valid_ids_1.union(valid_ids_2)))[:10]  # Limit to 10

if not merged_ids:
    st.error("❌ No valid case IDs found using either group.")
    st.stop()

# ------------------------------
# Let user choose signals for analysis
all_signals = list(set(group1_signals + group2_signals))
selected_signals = st.multiselect("📌 Choose signals for analysis:", all_signals, default=["BIS/BIS"])

if not selected_signals:
    st.warning("Please select at least one signal.")
    st.stop()

# ------------------------------
# Classes
class SignalProcessor:
    def __init__(self, data):
        self.data = data.copy()

    def interpolate_nans(self):
        x = np.arange(self.data.shape[0])
        for i in range(self.data.shape[1]):
            sig = self.data[:, i]
            if np.isnan(sig).sum():
                mask = ~np.isnan(sig)
                try:
                    f = interp1d(x[mask], sig[mask], kind='linear', fill_value='extrapolate')
                    self.data[:, i] = f(x)
                except:
                    continue
        return self.data

class Evaluator:
    def __init__(self, raw, imputed):
        self.raw = raw
        self.imputed = imputed

    def summary(self, i):
        return {
            "Mean (raw)": round(np.nanmean(self.raw[:, i]), 2),
            "Mean (imputed)": round(np.nanmean(self.imputed[:, i]), 2),
            "NaNs before": int(np.isnan(self.raw[:, i]).sum()),
            "NaNs after": int(np.isnan(self.imputed[:, i]).sum())
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
                if isinstance(data, np.ndarray) and data.shape[0] > 0:
                    all_data.append(data)
            except:
                continue
        if not all_data:
            return None, 0
        min_len = min(d.shape[0] for d in all_data)
        trimmed = np.concatenate([d[:min_len, :] for d in all_data], axis=0)
        return trimmed, len(all_data)

# ------------------------------
# Run pipeline
st.subheader("🚀 Running pipeline...")
st.write(f"Using {len(merged_ids)} merged case IDs.")

runner = PipelineRunner(merged_ids, selected_signals)
data_raw, used = runner.run()

if data_raw is None:
    st.error("No data loaded.")
    st.stop()

processor = SignalProcessor(data_raw)
data_clean = processor.interpolate_nans()
evaluator = Evaluator(data_raw, data_clean)

st.success(f"✅ Loaded and processed {used} cases. Shape: {data_clean.shape}")

# ------------------------------
# Show summary and plot
for i, var in enumerate(selected_signals):
    st.markdown(f"### 📈 {var}")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(data_clean[:, i], label=var)
    ax.set_title(f"Interpolated Signal - {var}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.write(evaluator.summary(i))
