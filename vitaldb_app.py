import streamlit as st
import pandas as pd
import numpy as np
import vitaldb
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ------------------ CONFIG -------------------
st.set_page_config(page_title="VitalDB Streamlit", layout="wide")
st.title("VitalDB Signal Preprocessing Tool")

# ------------------ DATA LOADING -------------------
@st.cache_data
def load_data():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    df_labs = pd.read_csv("https://api.vitaldb.net/labs")
    return df_cases, df_trks, df_labs

df_cases, df_trks, df_labs = load_data()

# ------------------ GLOBALS -------------------
if "valid_ids" not in st.session_state:
    st.session_state.valid_ids = []
    st.session_state.df_cases_filtered = pd.DataFrame()

# ------------------ GLOBAL STATS FUNCTION -------------------
def compute_global_stats(case_ids, variables):
    all_data_list = []
    for cid in case_ids:
        try:
            data = vitaldb.load_case(cid, variables, interval=1)
            all_data_list.append(data)
        except:
            continue

    if not all_data_list:
        return {}, {}

    min_len = min(d.shape[0] for d in all_data_list)
    trimmed_data = np.concatenate([d[:min_len, :] for d in all_data_list], axis=0)
    global_medians = {}
    global_mads = {}
    for i, var in enumerate(variables):
        sig = trimmed_data[:, i]
        sig = sig[~np.isnan(sig)]
        global_medians[var] = np.median(sig)
        global_mads[var] = np.median(np.abs(sig - global_medians[var])) or 1e-6

    return global_medians, global_mads

# ------------------ CUSTOM SIGNAL ANALYZER -------------------
class SignalAnalyzer:
    def __init__(self, caseid, data, variable_names, global_medians, global_mads):
        self.caseid = caseid
        self.data = data
        self.variable_names = variable_names
        self.global_medians = global_medians
        self.global_mads = global_mads
        self.issues = {var: {'nan': [], 'outlier': [], 'jump': [], 'signal': data[:, i]} for i, var in enumerate(variable_names)}

    def analyze(self):
        results = []
        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i].copy()
            if "BIS" in var:
                signal[signal == 0] = np.nan

            nan_idx = np.where(np.isnan(signal))[0]
            median = self.global_medians.get(var, 0)
            mad = self.global_mads.get(var, 1e-6)
            outlier_mask = np.abs(signal - median) > 3.5 * mad
            jump_mask = np.abs(np.diff(signal, prepend=signal[0])) > 3.5 * mad

            self.issues[var]['nan'] = nan_idx
            self.issues[var]['outlier'] = np.where(outlier_mask)[0]
            self.issues[var]['jump'] = np.where(jump_mask)[0]

            results.append({
                "variable": var,
                "nan_count": len(nan_idx),
                "nan_pct": len(nan_idx) / len(signal) * 100,
                "outliers": outlier_mask.sum(),
                "jumps": jump_mask.sum(),
            })

        return pd.DataFrame(results)

    def plot(self):
        fig = make_subplots(rows=len(self.variable_names), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            subplot_titles=self.variable_names)
        for i, var in enumerate(self.variable_names):
            sig = self.issues[var]['signal']
            time = np.arange(len(sig))
            fig.add_trace(go.Scatter(x=time, y=sig, mode='lines', name=var), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=time[self.issues[var]['nan']], y=[sig.min()-1]*len(self.issues[var]['nan']),
                                     mode='markers', marker=dict(color='gray'), name='NaN'), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=time[self.issues[var]['outlier']], y=sig[self.issues[var]['outlier']],
                                     mode='markers', marker=dict(color='red'), name='Outlier'), row=i+1, col=1)
            fig.add_trace(go.Scatter(x=time[self.issues[var]['jump']], y=sig[self.issues[var]['jump']],
                                     mode='markers', marker=dict(color='orange'), name='Jump'), row=i+1, col=1)

        fig.update_layout(height=300 * len(self.variable_names), title_text=f"Signal Plots - Case {self.caseid}")
        return fig

# ------------------ UI - FILTER -------------------
with st.expander("1. Filter Cases", expanded=True):
    ane_type = st.selectbox("Select Anesthesia Type:", df_cases["ane_type"].dropna().unique())
    intraoperative_boluses = ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"]
    removed_boluses = st.multiselect("Remove cases with these drugs:", options=intraoperative_boluses, default=intraoperative_boluses)

    required_variables_1 = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS", "Orchestra/PPF20_CE", "Orchestra/RFTN20_CE", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"]
    required_variables_2 = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS", "Orchestra/PPF20_CE", "Orchestra/RFTN50_CE", "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"]

    if st.button("Apply Filters"):
        def get_valid_ids(df_cases, df_trks, ane_type, variables, removed_boluses):
            filtered = df_cases[df_cases['ane_type'] == ane_type].copy()
            valid_ids = set(filtered['caseid'])
            for var in variables:
                valid_ids &= set(df_trks[df_trks['tname'] == var]['caseid'])
            if removed_boluses:
                valid_boluses = [col for col in removed_boluses if col in filtered.columns]
                filtered = filtered[~filtered[valid_boluses].gt(0).any(axis=1)]
                valid_ids &= set(filtered['caseid'])
            return valid_ids

        ids1 = get_valid_ids(df_cases, df_trks, ane_type, required_variables_1, removed_boluses)
        ids2 = get_valid_ids(df_cases, df_trks, ane_type, required_variables_2, removed_boluses)
        valid_ids = sorted(list(ids1.union(ids2)))

        st.session_state.valid_ids = valid_ids
        st.session_state.df_cases_filtered = df_cases[df_cases['caseid'].isin(valid_ids)]

        st.success(f"✅ Filtered {len(valid_ids)} valid case IDs.")
        st.write(st.session_state.df_cases_filtered)

# ------------------ UI - ANALYZE -------------------
with st.expander("2. Analyze Signals", expanded=False):
    if st.session_state.valid_ids:
        selected_ids = st.multiselect("Select Case IDs to Analyze", st.session_state.valid_ids, default=st.session_state.valid_ids[:3])
        signal_list = ["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "Orchestra/RFTN50_RATE"]

        if st.button("Analyze Selected Signals"):
            global_medians, global_mads = compute_global_stats(selected_ids, signal_list)

            for cid in selected_ids:
                try:
                    data = vitaldb.load_case(cid, signal_list, interval=1)
                    analyzer = SignalAnalyzer(cid, data, signal_list, global_medians, global_mads)
                    result_df = analyzer.analyze()

                    st.subheader(f"Case {cid} Summary")
                    st.dataframe(result_df)
                    st.plotly_chart(analyzer.plot(), use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Failed to process case {cid}: {e}")
    else:
        st.info("Please filter valid case IDs first in Tab 1.")
