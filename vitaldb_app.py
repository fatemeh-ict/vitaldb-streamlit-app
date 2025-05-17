import streamlit as st
import pandas as pd
import numpy as np
from vitaldb import load_case
import plotly.graph_objects as go

# =============================
# Class: CaseSelector
# =============================
class CaseSelector:
    def __init__(self, df_cases, df_trks, ane_type="General", required_variables=None, intraoperative_boluses=None):
        self.df_cases = df_cases.copy()
        self.df_trks = df_trks.copy()
        self.ane_type = ane_type
        self.required_variables = required_variables or []
        self.intraoperative_boluses = intraoperative_boluses or []

    def select_valid_cases(self):
        df_cases_filtered = self.df_cases[self.df_cases['ane_type'] == self.ane_type].copy()
        valid_case_ids = set(df_cases_filtered['caseid'])

        for var in self.required_variables:
            trk_cases = set(self.df_trks[self.df_trks['tname'] == var]['caseid'])
            valid_case_ids &= trk_cases

        if self.intraoperative_boluses:
            valid_boluses = [col for col in self.intraoperative_boluses if col in df_cases_filtered.columns]
            df_cases_filtered = df_cases_filtered[~df_cases_filtered[valid_boluses].gt(0).any(axis=1)]
            valid_case_ids &= set(df_cases_filtered['caseid'])

        return sorted(list(valid_case_ids))

# =============================
# Class: SignalAnalyzer
# =============================
class SignalAnalyzer:
    def __init__(self, caseid, data, variable_names, thresholds=None, plot=True, global_medians=None, global_mads=None):
        self.caseid = caseid
        self.data = data
        self.variable_names = variable_names
        self.thresholds = thresholds or {"missing": 0.05, "gap": 30, "jump": 100}
        self.plot_enabled = plot
        self.global_medians = global_medians or {}
        self.global_mads = global_mads or {}
        self.issues = {var: {'nan': [], 'gap': [], 'outlier': [], 'outlier_values': [], 'jump': []} for var in variable_names}

    def analyze(self):
        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i]
            original = signal.copy()

            # NaN detection
            nan_idx = np.where(np.isnan(signal))[0]
            self.issues[var]['nan'] = nan_idx.tolist()

            # Gap detection
            gap_list = []
            is_gap = False
            start_idx = 0
            for idx, val in enumerate(signal):
                if np.isnan(val):
                    if not is_gap:
                        is_gap = True
                        start_idx = idx
                else:
                    if is_gap:
                        gap_len = idx - start_idx
                        gap_list.append({"start": start_idx, "length": gap_len})
                        is_gap = False
            if is_gap:
                gap_len = len(signal) - start_idx
                gap_list.append({"start": start_idx, "length": gap_len})
            self.issues[var]['gap'] = gap_list

            # Outlier detection
            outliers = []
            if "RATE" in var:
                outliers = np.where(signal < 0)[0].tolist()
            elif "BIS" in var:
                outliers = np.where((signal <= 0) | (signal > 100))[0].tolist()
            elif "NIBP" in var:
                outliers = np.where(signal <= 0)[0].tolist()
            elif var in self.global_medians and var in self.global_mads:
                mad = self.global_mads[var] or 1e-6
                median = self.global_medians[var]
                mad_idx = np.where(np.abs(signal - median) > 3.5 * mad)[0].tolist()
                outliers.extend(mad_idx)

            self.issues[var]['outlier'] = sorted(set(outliers))
            self.issues[var]['outlier_values'] = original[self.issues[var]['outlier']].tolist()

            # Jump detection
            diffs = np.diff(signal)
            if len(diffs) > 0:
                median_diff = np.median(diffs)
                mad_diff = np.median(np.abs(diffs - median_diff)) or 1e-6
                jump_idx = np.where(np.abs(diffs - median_diff) > 3.5 * mad_diff)[0]
                self.issues[var]['jump'] = jump_idx.tolist()

    def plot(self, signal_units=None):
        import plotly.subplots as sp
        fig = sp.make_subplots(rows=len(self.variable_names), cols=1, shared_xaxes=True,
                               subplot_titles=self.variable_names, vertical_spacing=0.05)
        time = np.arange(self.data.shape[0])

        for i, var in enumerate(self.variable_names):
            row = i + 1
            signal = self.data[:, i]
            fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name=var), row=row, col=1)
            if self.issues[var]['outlier']:
                idx = self.issues[var]['outlier']
                fig.add_trace(go.Scatter(x=time[idx], y=signal[idx], mode='markers',
                                         marker=dict(color='red', size=6), name=f"{var} Outliers"), row=row, col=1)
        fig.update_layout(height=300 * len(self.variable_names), title=f"Case {self.caseid} Signal Plots")
        return fig

# =============================
# Streamlit App Logic
# =============================
st.sidebar.header("📁 Case Configuration")
df_cases = pd.read_csv("https://api.vitaldb.net/cases")
df_trks = pd.read_csv("https://api.vitaldb.net/trks")

signal_options = sorted(df_trks['tname'].value_counts().index.tolist())
selected_signals = st.sidebar.multiselect("Select Signals", signal_options[:100],
                                          default=["BIS/BIS", "Solar8000/NIBP_SBP"])
selected_drugs = st.sidebar.multiselect("Exclude cases with drugs", ["intraop_mdz", "intraop_ftn"])

if st.sidebar.button("🔍 Filter Cases"):
    selector = CaseSelector(df_cases, df_trks, required_variables=selected_signals,
                            intraoperative_boluses=selected_drugs)
    valid_ids = selector.select_valid_cases()
    st.session_state.valid_ids = valid_ids
    st.success(f"✅ Found {len(valid_ids)} valid case IDs")

if 'valid_ids' in st.session_state:
    selected_cases = st.multiselect("Select Case IDs to Analyze", st.session_state.valid_ids[:50])
    plot_cases = st.multiselect("Select Case IDs to Plot", selected_cases)

    if st.button("🚀 Run Analysis"):
        for cid in selected_cases:
            try:
                data = load_case(cid, selected_signals, interval=1)
                medians = {v: np.nanmedian(data[:, i]) for i, v in enumerate(selected_signals)}
                mads = {v: np.nanmedian(np.abs(data[:, i] - medians[v])) or 1e-6 for i, v in enumerate(selected_signals)}

                analyzer = SignalAnalyzer(caseid=cid, data=data, variable_names=selected_signals,
                                          global_medians=medians, global_mads=mads, plot=False)
                analyzer.analyze()

                st.subheader(f"📋 Stats for Case {cid}")
                for var in selected_signals:
                    st.write(f"**{var}**: NaNs={len(analyzer.issues[var]['nan'])}, "
                             f"Outliers={len(analyzer.issues[var]['outlier'])}, "
                             f"Jumps={len(analyzer.issues[var]['jump'])}")

                if cid in plot_cases:
                    fig = analyzer.plot()
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error processing case {cid}: {e}")
