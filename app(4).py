
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import seaborn as sns
import os
import vitaldb

# ==========================
# CaseSelector Class
# ==========================
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

# Initialize Streamlit
st.set_page_config(layout="wide")
st.title("🩺 VitalDB Streamlit Analyzer")

tabs = st.tabs(["1️⃣ Select Cases", "2️⃣ Signal Quality", "3️⃣ Interpolation", "4️⃣ Evaluation", "5️⃣ Export"])

with tabs[0]:
    st.header("Step 1: Select Valid Cases")

    ane_type = st.selectbox("Anesthesia Type", ["General", "Spinal", "Regional"])
    intraoperative_boluses = st.multiselect("Exclude Boluses", ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"])

    selected_group = st.radio("Signal Group", ["Group 1 (RFTN20)", "Group 2 (RFTN50)", "Both Groups"])

    required_variables_1 = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
                            "Orchestra/PPF20_CE", "Orchestra/RFTN20_CE", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"]

    required_variables_2 = ["Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
                            "Orchestra/PPF20_CE", "Orchestra/RFTN50_CE", "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"]

    if st.button("📥 Load and Filter Cases"):
        with st.spinner("Loading data from VitalDB..."):
            df_cases = pd.read_csv("https://api.vitaldb.net/cases")
            df_trks = pd.read_csv("https://api.vitaldb.net/trks")
            df_labs = pd.read_csv("https://api.vitaldb.net/labs")

            selector1 = CaseSelector(df_cases, df_trks, ane_type=ane_type,
                                     required_variables=required_variables_1, intraoperative_boluses=intraoperative_boluses)
            selector2 = CaseSelector(df_cases, df_trks, ane_type=ane_type,
                                     required_variables=required_variables_2, intraoperative_boluses=intraoperative_boluses)

            ids1 = set(selector1.select_valid_cases())
            ids2 = set(selector2.select_valid_cases())

            if selected_group == "Group 1 (RFTN20)":
                final_ids = ids1
                selected_vars = required_variables_1
            elif selected_group == "Group 2 (RFTN50)":
                final_ids = ids2
                selected_vars = required_variables_2
            else:
                final_ids = ids1.union(ids2)
                selected_vars = list(set(required_variables_1 + required_variables_2))

            st.session_state["case_ids"] = sorted(list(final_ids))
            st.session_state["df_cases_filtered"] = df_cases[df_cases['caseid'].isin(final_ids)]
            st.session_state["df_trks_filtered"] = df_trks[df_trks['caseid'].isin(final_ids)]
            st.session_state["df_labs_filtered"] = df_labs[df_labs['caseid'].isin(final_ids)]
            st.session_state["selected_vars"] = selected_vars

        st.success(f"{len(st.session_state['case_ids'])} valid case(s) selected.")
        st.dataframe(st.session_state['df_cases_filtered'].head(10))

        st.download_button("⬇️ Download Filtered Cases", st.session_state['df_cases_filtered'].to_csv(index=False), "filtered_cases.csv")
        st.download_button("⬇️ Download Filtered Tracks", st.session_state['df_trks_filtered'].to_csv(index=False), "filtered_trks.csv")
        st.download_button("⬇️ Download Filtered Labs", st.session_state['df_labs_filtered'].to_csv(index=False), "filtered_labs.csv")


# ==========================
# SignalAnalyzer Class
# ==========================
class SignalAnalyzer:
    def __init__(self, caseid, data, variable_names):
        self.caseid = caseid
        self.data = data
        self.variable_names = variable_names
        self.issues = {var: {'nan': 0, 'gap': 0, 'outlier': 0, 'jump': 0} for var in variable_names}

    def analyze(self):
        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i]
            self.issues[var]['nan'] = int(np.isnan(signal).sum())
            self.issues[var]['gap'] = int(np.sum(np.diff(np.where(np.isnan(signal), 1, 0)) > 1))
            self.issues[var]['outlier'] = int((signal < 0).sum())
            diffs = np.diff(signal)
            median_diff = np.median(diffs)
            mad_diff = np.median(np.abs(diffs - median_diff)) or 1e-6
            jump_idx = np.where(np.abs(diffs - median_diff) > 3.5 * mad_diff)[0]
            self.issues[var]['jump'] = int(len(jump_idx))
        return self.issues

# ==========================
# SignalProcessor Class
# ==========================
class SignalProcessor:
    def __init__(self, data, variable_names):
        self.data = data.copy()
        self.variable_names = variable_names

    def interpolate(self):
        x = np.arange(self.data.shape[0])
        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i]
            if np.isnan(signal).sum() > 0:
                valid_mask = ~np.isnan(signal)
                if valid_mask.sum() > 1:
                    f = interp1d(x[valid_mask], signal[valid_mask], kind='linear', fill_value='extrapolate')
                    self.data[:, i] = f(x)
        return self.data

# ==========================
# Evaluator Class
# ==========================
class Evaluator:
    def __init__(self, raw_data, imputed_data, variable_names):
        self.raw = pd.DataFrame(raw_data, columns=variable_names)
        self.imputed = pd.DataFrame(imputed_data, columns=variable_names)
        self.variable_names = variable_names

    def compute(self):
        rows = []
        for var in self.variable_names:
            rows.append({
                'variable': var,
                'mean_before': self.raw[var].mean(),
                'mean_after': self.imputed[var].mean(),
                'nan_before': self.raw[var].isna().sum(),
                'nan_after': self.imputed[var].isna().sum(),
            })
        return pd.DataFrame(rows)

# ==========================
# Tab 2: Signal Quality
# ==========================
with tabs[1]:
    st.header("Step 2: Signal Quality Analysis")
    if 'case_ids' in st.session_state:
        selected_case = st.selectbox("Select a case to analyze", st.session_state['case_ids'])
        variables = st.session_state["selected_vars"]
        data = vitaldb.load_case(selected_case, variables, interval=1)
        analyzer = SignalAnalyzer(caseid=selected_case, data=data, variable_names=variables)
        results = analyzer.analyze()

        for var in variables:
            st.subheader(f"🔍 {var}")
            st.write(results[var])
        st.session_state["raw_data"] = data
    else:
        st.warning("Please select cases first.")

# ==========================
# Tab 3: Interpolation
# ==========================
with tabs[2]:
    st.header("Step 3: Interpolate Missing Data")
    if 'raw_data' in st.session_state:
        processor = SignalProcessor(data=st.session_state['raw_data'], variable_names=st.session_state['selected_vars'])
        interpolated = processor.interpolate()
        st.session_state["interpolated"] = interpolated
        st.success("Interpolation completed.")
        st.line_chart(interpolated)
    else:
        st.warning("Analyze a case first.")

# ==========================
# Tab 4: Evaluation
# ==========================
with tabs[3]:
    st.header("Step 4: Evaluation Before and After Imputation")
    if 'interpolated' in st.session_state and 'raw_data' in st.session_state:
        evaluator = Evaluator(
            raw_data=st.session_state['raw_data'],
            imputed_data=st.session_state['interpolated'],
            variable_names=st.session_state['selected_vars']
        )
        df_eval = evaluator.compute()
        st.dataframe(df_eval)
        st.session_state['df_eval'] = df_eval
    else:
        st.warning("Run interpolation first.")

# ==========================
# Tab 5: Export Results
# ==========================
with tabs[4]:
    st.header("Step 5: Download Evaluation Results")
    if 'df_eval' in st.session_state:
        csv = st.session_state['df_eval'].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Evaluation CSV", csv, "evaluation_results.csv", "text/csv")
    else:
        st.info("No results to export.")


# ==========================
# Updated SignalAnalyzer Class (from Colab logic)
# ==========================
class SignalAnalyzer:
    def __init__(self, caseid, data, variable_names):
        self.caseid = caseid
        self.data = data
        self.variable_names = variable_names
        self.issues = {var: {'nan': [], 'gap': [], 'outlier': [], 'jump': []} for var in variable_names}

    def analyze(self):
        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i]
            nan_idx = np.where(np.isnan(signal))[0]
            self.issues[var]['nan'] = nan_idx.tolist()

            # Gap detection (list of dicts with start and length)
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

            # Outlier detection (based on MAD or hard rules)
            outliers = []
            if "RATE" in var:
                outliers = np.where(signal < 0)[0].tolist()
            elif "BIS" in var:
                outliers = np.where((signal <= 0) | (signal > 100))[0].tolist()
            elif "NIBP" in var:
                outliers = np.where(signal <= 0)[0].tolist()
                sig = signal[~np.isnan(signal)]
                if len(sig) > 0:
                    med = np.median(sig)
                    mad = np.median(np.abs(sig - med)) or 1e-6
                    mad_idx = np.where(np.abs(signal - med) > 3.5 * mad)[0].tolist()
                    outliers.extend(mad_idx)
            self.issues[var]['outlier'] = sorted(set(outliers))

            # Jump detection (using MAD on diff)
            diffs = np.diff(signal)
            if len(diffs) > 0:
                median_diff = np.median(diffs)
                mad_diff = np.median(np.abs(diffs - median_diff)) or 1e-6
                jump_idx = np.where(np.abs(diffs - median_diff) > 3.5 * mad_diff)[0]
                self.issues[var]['jump'] = jump_idx.tolist()
        return self.issues

# ==========================
# Tab 2: Updated Signal Quality with Plotly
# ==========================
with tabs[1]:
    st.header("Step 2: Signal Quality Analysis (Real + Plot)")

    target_vars = ["BIS/BIS", "Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP",
                   "Orchestra/RFTN50_RATE", "Orchestra/RFTN20_RATE", "Orchestra/PPF20_RATE"]

    if 'case_ids' in st.session_state:
        selected_case = st.selectbox("Select a case to analyze", st.session_state['case_ids'], key="select_case_tab2")
        data = vitaldb.load_case(selected_case, target_vars, interval=1)
        analyzer = SignalAnalyzer(caseid=selected_case, data=data, variable_names=target_vars)
        results = analyzer.analyze()
        df = pd.DataFrame(data, columns=target_vars)
        df["time"] = np.arange(len(df))

        fig = make_subplots(rows=len(target_vars), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            subplot_titles=target_vars)

        for i, var in enumerate(target_vars):
            row = i + 1
            signal = df[var]
            time = df["time"]

            # Plot signal line
            fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name=var), row=row, col=1)

            # Plot NaNs
            if results[var]['nan']:
                fig.add_trace(go.Scatter(x=time[results[var]['nan']],
                                         y=[signal.min() - 5] * len(results[var]['nan']),
                                         mode='markers', marker=dict(color='gray', symbol='line-ns-open', size=6),
                                         name='NaNs', showlegend=(i==0)), row=row, col=1)

            # Plot Outliers
            if results[var]['outlier']:
                fig.add_trace(go.Scatter(x=time[results[var]['outlier']],
                                         y=signal[results[var]['outlier']],
                                         mode='markers', marker=dict(color='purple', symbol='star', size=7),
                                         name='Outliers', showlegend=(i==0)), row=row, col=1)

            # Plot Jumps
            if results[var]['jump']:
                fig.add_trace(go.Scatter(x=time[results[var]['jump']],
                                         y=signal[results[var]['jump']],
                                         mode='markers', marker=dict(color='orange', symbol='x', size=7),
                                         name='Jumps', showlegend=(i==0)), row=row, col=1)

            # Plot Gaps
            for gap in results[var]['gap']:
                start = gap['start']
                end = gap['start'] + gap['length']
                fig.add_shape(type="rect",
                              x0=time[start], x1=time[end - 1],
                              y0=signal.min(), y1=signal.max(),
                              fillcolor="red", opacity=0.15,
                              line=dict(width=0), row=row, col=1)

        fig.update_layout(height=250 * len(target_vars), title=f"Signal Quality - Case {selected_case}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select cases in Tab 1 first.")
