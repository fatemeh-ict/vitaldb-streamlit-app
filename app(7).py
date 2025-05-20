
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import vitaldb

# ==========================
# Class: CaseSelector
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

# ==========================
# Class: SignalAnalyzer (Colab logic)
# ==========================
class SignalAnalyzer:
    def __init__(self, caseid, data, variable_names, global_medians=None, global_mads=None):
        self.caseid = caseid
        self.data = data
        self.variable_names = variable_names
        self.global_medians = global_medians or {}
        self.global_mads = global_mads or {}
        self.issues = {
            var: {
                'nan': [],
                'gap': [],
                'classified_gaps': [],
                'outlier': [],
                'outlier_values': [],
                'jump': []
            } for var in variable_names
        }

    def analyze(self):
        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i]
            original = signal.copy()
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
                        gap_list.append({"start": start_idx, "length": idx - start_idx})
                        is_gap = False
            if is_gap:
                gap_list.append({"start": start_idx, "length": len(signal) - start_idx})
            self.issues[var]['gap'] = gap_list

            # Classify gaps
            if gap_list:
                lengths = np.array([g["length"] for g in gap_list])
                median_gap = np.median(lengths)
                mad_gap = np.median(np.abs(lengths - median_gap)) or 1e-6
                for g in gap_list:
                    g["type"] = "long" if g["length"] > median_gap + 3.5 * mad_gap else "short"
                self.issues[var]['classified_gaps'] = gap_list

            # Outlier detection
            outliers = []
            if "RATE" in var:
                outliers = np.where(signal < 0)[0].tolist()
            elif "BIS" in var:
                outliers = np.where((signal <= 0) | (signal > 100))[0].tolist()
            elif "NIBP" in var:
                invalid = np.where(signal <= 0)[0].tolist()
                outliers.extend(invalid)
                if var in self.global_medians:
                    median = self.global_medians[var]
                    mad = self.global_mads[var]
                    mad_idx = np.where(np.abs(signal - median) > 3.5 * mad)[0].tolist()
                    outliers.extend(mad_idx)
            else:
                if var in self.global_medians:
                    median = self.global_medians[var]
                    mad = self.global_mads[var]
                    mad_idx = np.where(np.abs(signal - median) > 3.5 * mad)[0].tolist()
                    outliers.extend(mad_idx)
            outliers = sorted(set(outliers))
            self.issues[var]['outlier'] = outliers
            self.issues[var]['outlier_values'] = original[outliers].tolist()

            # Jump detection
            diffs = np.diff(signal)
            if len(diffs):
                median_diff = np.median(diffs)
                mad_diff = np.median(np.abs(diffs - median_diff)) or 1e-6
                jump_idx = np.where(np.abs(diffs - median_diff) > 3.5 * mad_diff)[0]
                self.issues[var]['jump'] = jump_idx.tolist()
        return self.issues

# ==========================
# Streamlit App
# ==========================
st.set_page_config(layout="wide")
st.title("🩺 VitalDB Analyzer with Accurate Signal Quality")

tabs = st.tabs(["1️⃣ Select Cases", "2️⃣ Analyze Signals"])

with tabs[0]:
    st.header("Step 1: Select Valid Cases")
    ane_type = st.selectbox("Anesthesia Type", ["General", "Spinal", "Regional"])
    intraoperative_boluses = st.multiselect("Exclude Boluses", ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"])

    required_variables = [
        "BIS/BIS",
        "Solar8000/NIBP_DBP",
        "Solar8000/NIBP_SBP",
        "Orchestra/RFTN50_RATE",
        "Orchestra/RFTN20_RATE",
        "Orchestra/PPF20_RATE"
    ]

    if st.button("📥 Load and Filter Cases"):
        df_cases = pd.read_csv("https://api.vitaldb.net/cases")
        df_trks = pd.read_csv("https://api.vitaldb.net/trks")

        selector = CaseSelector(df_cases, df_trks, ane_type=ane_type,
                                required_variables=required_variables,
                                intraoperative_boluses=intraoperative_boluses)
        valid_ids = selector.select_valid_cases()
        st.session_state["case_ids"] = valid_ids
        st.success(f"{len(valid_ids)} valid case(s) found.")
        st.dataframe(df_cases[df_cases['caseid'].isin(valid_ids)].head(10))

with tabs[1]:
    st.header("Step 2: Analyze Signal Quality (Accurate)")
    variables = [
        "BIS/BIS",
        "Solar8000/NIBP_DBP",
        "Solar8000/NIBP_SBP",
        "Orchestra/RFTN50_RATE",
        "Orchestra/RFTN20_RATE",
        "Orchestra/PPF20_RATE"
    ]

    if 'case_ids' in st.session_state:
        selected_case = st.selectbox("Select a case to analyze", st.session_state["case_ids"])
        data = vitaldb.load_case(selected_case, variables, interval=1)

        global_medians = {}
        global_mads = {}
        for i, var in enumerate(variables):
            sig = data[:, i]
            sig = sig[~np.isnan(sig)]
            global_medians[var] = np.median(sig)
            global_mads[var] = np.median(np.abs(sig - global_medians[var])) or 1e-6

        analyzer = SignalAnalyzer(selected_case, data, variables, global_medians, global_mads)
        results = analyzer.analyze()
        df = pd.DataFrame(data, columns=variables)
        df["time"] = np.arange(len(df))

        fig = make_subplots(rows=len(variables), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            subplot_titles=variables)

        for i, var in enumerate(variables):
            row = i + 1
            signal = df[var]
            time = df["time"]

            st.subheader(f"🔍 {var}")
            st.markdown(
                f"**NaNs:** {len(results[var]['nan'])} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Gaps:** {len(results[var]['gap'])} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Outliers:** {len(results[var]['outlier'])} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Jumps:** {len(results[var]['jump'])}"
            )

            fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name=var), row=row, col=1)

            if results[var]['nan']:
                fig.add_trace(go.Scatter(x=time[results[var]['nan']],
                                         y=[signal.min() - 5] * len(results[var]['nan']),
                                         mode='markers', marker=dict(color='gray', symbol='line-ns-open', size=6),
                                         name='NaNs', showlegend=(i == 0)), row=row, col=1)

            if results[var]['outlier']:
                fig.add_trace(go.Scatter(x=time[results[var]['outlier']],
                                         y=signal[results[var]['outlier']],
                                         mode='markers', marker=dict(color='purple', symbol='star', size=7),
                                         name='Outliers', showlegend=(i == 0)), row=row, col=1)

            if results[var]['jump']:
                fig.add_trace(go.Scatter(x=time[results[var]['jump']],
                                         y=signal[results[var]['jump']],
                                         mode='markers', marker=dict(color='orange', symbol='x', size=7),
                                         name='Jumps', showlegend=(i == 0)), row=row, col=1)

            for gap in results[var]['gap']:
                start = gap['start']
                end = gap['start'] + gap['length']
                fig.add_shape(type="rect",
                              x0=time[start], x1=time[end - 1],
                              y0=signal.min(), y1=signal.max(),
                              fillcolor="red", opacity=0.15,
                              line=dict(width=0), row=row, col=1)

        fig.update_layout(height=300 * len(variables), title=f"Signal Quality - Case {selected_case}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select valid cases first.")
