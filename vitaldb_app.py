import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vitaldb
from scipy.interpolate import interp1d

st.set_page_config(page_title="VitalDB Analyzer", layout="wide")
st.title("💉 VitalDB Case Filtering and Signal Analysis")

# Load metadata
@st.cache_data(show_spinner=False)
def load_metadata():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    return df_cases, df_trks

df_cases, df_trks = load_metadata()

# Define signal groups
group1_signals = [
    "BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"
]
group2_signals = [
    "BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"
]
all_signals = list(set(group1_signals + group2_signals))

# Helper function to filter by signals
def filter_cases(df_trks, signals):
    valid_ids = set(df_trks['caseid'].unique())
    for sig in signals:
        ids = set(df_trks[df_trks['tname'] == sig]['caseid'])
        valid_ids &= ids
    return valid_ids

# Tabs setup
tabs = st.tabs(["🔍 Case Selection", "⚙️ Signal Preprocessing", "📈 Visualization", "📤 Export"])

# ------------------------
# TAB 1: Case Selection
with tabs[0]:
    st.header("🔍 Filter Cases by Signal, Drugs, and Surgery Type")
    selected_signals = st.multiselect("Select signals:", all_signals, default=["BIS/BIS"])
    selected_optype = st.multiselect("Filter by operation type:", sorted(df_cases['optype'].dropna().unique()))
    drug_vars = ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe"]
    drugs_to_exclude = st.multiselect("Exclude cases with intraoperative drugs:", drug_vars)

    valid_ids_1 = filter_cases(df_trks, group1_signals)
    valid_ids_2 = filter_cases(df_trks, group2_signals)
    merged_ids = sorted(list(valid_ids_1.union(valid_ids_2)))

    df_filtered = df_cases[df_cases['caseid'].isin(merged_ids)].copy()
    if selected_optype:
        df_filtered = df_filtered[df_filtered['optype'].isin(selected_optype)]
    if drugs_to_exclude:
        df_filtered = df_filtered[df_filtered[drugs_to_exclude].sum(axis=1) == 0]

    case_ids = df_filtered['caseid'].tolist()[:10]
    st.success(f"✅ {len(case_ids)} cases selected for analysis.")

# ------------------------
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

# ------------------------
# TAB 2: Signal Preprocessing
with tabs[1]:
    st.header("⚙️ Signal Preprocessing")
    if not case_ids:
        st.warning("No cases selected.")
        st.stop()

    runner = PipelineRunner(case_ids, selected_signals)
    raw_data, used = runner.run()
    if raw_data is None:
        st.error("No data loaded.")
        st.stop()

    processor = SignalProcessor(raw_data)
    imputed_data = processor.interpolate_nans()
    evaluator = Evaluator(raw_data, imputed_data)
    st.success(f"✅ {used} cases processed. Shape: {imputed_data.shape}")

# ------------------------
# TAB 3: Visualization
with tabs[2]:
    st.header("📈 Signal Visualization and Summary")
    if 'imputed_data' not in locals():
        st.warning("Preprocess signals first.")
        st.stop()
    for i, var in enumerate(selected_signals):
        st.markdown(f"### {var}")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(imputed_data[:, i], label=var)
        ax.set_title(f"{var} - Interpolated")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        st.write(evaluator.summary(i))

# ------------------------
# TAB 4: Export
with tabs[3]:
    st.header("📤 Export Cleaned Data")
    if 'imputed_data' in locals():
        df_out = pd.DataFrame(imputed_data, columns=selected_signals)
        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "cleaned_signals.csv", "text/csv")
    else:
        st.info("No data available for export.")
