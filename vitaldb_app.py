# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from vitaldb import load_case
from pathlib import Path
from io import BytesIO

# your_modules.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.interpolate import interp1d
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import vitaldb

class CaseSelector:
    def __init__(self, df_cases, df_trks, ane_type="General", required_variables=None, intraoperative_boluses=None, optype_filter=None):
        self.df_cases = df_cases.copy()
        self.df_trks = df_trks.copy()
        self.ane_type = ane_type
        self.required_variables = required_variables or []
        self.intraoperative_boluses = intraoperative_boluses or []
        self.optype_filter = optype_filter or []

    def select_valid_cases(self):
        df_filtered = self.df_cases.copy()
        if self.ane_type:
            df_filtered = df_filtered[df_filtered['ane_type'] == self.ane_type]

        if self.optype_filter:
            df_filtered = df_filtered[df_filtered['optype'].isin(self.optype_filter)]

        valid_case_ids = set(df_filtered['caseid'])
        for var in self.required_variables:
            trk_cases = set(self.df_trks[self.df_trks['tname'] == var]['caseid'])
            valid_case_ids &= trk_cases

        if self.intraoperative_boluses:
            valid_boluses = [col for col in self.intraoperative_boluses if col in df_filtered.columns]
            df_filtered = df_filtered[~df_filtered[valid_boluses].gt(0).any(axis=1)]
            valid_case_ids &= set(df_filtered['caseid'])

        return sorted(list(valid_case_ids)), df_filtered[df_filtered['caseid'].isin(valid_case_ids)]

class SignalAnalyzer:
    def __init__(self, caseid, data, variable_names, thresholds=None, global_medians=None, global_mads=None):
        self.caseid = caseid
        self.data = data
        self.variable_names = variable_names
        self.thresholds = thresholds or {"missing": 0.05, "gap": 30, "jump": 100}
        self.global_medians = global_medians or {}
        self.global_mads = global_mads or {}
        self.issues = {var: {'nan': [], 'gap': [], 'classified_gaps': [], 'outlier': [], 'outlier_values': [], 'jump': []} for var in variable_names}
        self.warnings = []

    def analyze(self):
        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i]
            original = signal.copy()
            nan_idx = np.where(np.isnan(signal))[0]
            self.issues[var]['nan'] = nan_idx.tolist()
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
            if gap_list:
                lengths = np.array([g["length"] for g in gap_list])
                median_gap = np.median(lengths)
                mad_gap = np.median(np.abs(lengths - median_gap)) or 1e-6
                for g in gap_list:
                    g["type"] = "long" if g["length"] > median_gap + 3.5 * mad_gap else "short"
                self.issues[var]['classified_gaps'] = gap_list
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
            diffs = np.diff(signal)
            if len(diffs) > 0:
                median_diff = np.median(diffs)
                mad_diff = np.median(np.abs(diffs - median_diff)) or 1e-6
                jump_idx = np.where(np.abs(diffs - median_diff) > 3.5 * mad_diff)[0]
                self.issues[var]['jump'] = jump_idx.tolist()

    def plot(self):
        df = pd.DataFrame(self.data, columns=self.variable_names)
        df["time"] = np.arange(len(df))
        fig = make_subplots(rows=len(self.variable_names), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=self.variable_names)
        for i, var in enumerate(self.variable_names):
            row = i + 1
            signal = df[var]
            time = df["time"]
            fig.add_trace(go.Scatter(x=time, y=signal, mode='lines+markers', name=var), row=row, col=1)
            if self.issues[var]['outlier']:
                idx = self.issues[var]['outlier']
                val = self.issues[var]['outlier_values']
                fig.add_trace(go.Scatter(x=time[idx], y=val, mode='markers',
                                         marker=dict(color='purple', size=6, symbol='star'),
                                         name="Outliers"), row=row, col=1)
        fig.update_xaxes(title_text="Time (s)")
        fig.update_layout(title=f"Signal Diagnostics - Case {self.caseid}", height=300 * len(self.variable_names))
        return fig

class SignalProcessor:
    def __init__(self, data, issues, variable_names,
                 gap_strategy='interpolate_short', long_gap_strategy='leave',
                 interp_method='auto', global_std_dict=None):
        self.data = data.copy()
        self.issues = issues
        self.variable_names = variable_names
        self.gap_strategy = gap_strategy
        self.long_gap_strategy = long_gap_strategy
        self.interp_method = interp_method
        self.global_std_dict = global_std_dict or {}

    def select_interp_method(self, signal, varname):
        if self.interp_method != 'auto':
            return self.interp_method
        signal_clean = signal[~np.isnan(signal)]
        if len(signal_clean) < 5:
            return 'linear'
        std = np.std(signal_clean)
        global_std = self.global_std_dict.get(varname, 10)
        if std < global_std * 0.7:
            return 'linear'
        elif std < global_std * 1.3:
            return 'cubic'
        else:
            return 'slinear'

    def align_signals_soft(self):
        aligned_signals = []
        for i in range(self.data.shape[1]):
            signal = self.data[:, i]
            start_idx = np.argmax(~np.isnan(signal))
            aligned_signals.append(signal[start_idx:])
        min_len = min(len(sig) for sig in aligned_signals)
        self.data = np.column_stack([sig[:min_len] for sig in aligned_signals])

    def interpolate_short_gaps(self):
        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i]
            gaps = self.issues[var].get('classified_gaps', [])
            x = np.arange(len(signal))
            valid_mask = ~np.isnan(signal)
            if valid_mask.sum() < 2:
                continue
            method = self.select_interp_method(signal, var)
            f = interp1d(x[valid_mask], signal[valid_mask], kind=method, fill_value='extrapolate')
            for gap in gaps:
                start = gap['start']
                end = start + gap['length']
                if gap['type'] == 'short' or start == 0 or end == len(signal):
                    signal[start:end] = f(x[start:end])
                elif gap['type'] == 'long':
                    if self.long_gap_strategy == 'zero':
                        signal[start:end] = 0
                    elif self.long_gap_strategy == 'nan':
                        signal[start:end] = np.nan
            self.data[:, i] = signal

    def process(self):
        if self.gap_strategy == 'interpolate_short':
            self.interpolate_short_gaps()
        self.align_signals_soft()
        return self.data

class Evaluator:
    def __init__(self, raw_data, imputed_data, variable_names):
        self.raw_df = pd.DataFrame(raw_data, columns=variable_names)
        self.imputed_df = pd.DataFrame(imputed_data, columns=variable_names)
        self.variable_names = variable_names

    def compute_stats(self, raw_length):
        stats_list = []
        for var in self.variable_names:
            raw = self.raw_df[var]
            imp = self.imputed_df[var]
            stats_list.append({
                'variable': var,
                'mean_before': raw.dropna().mean(),
                'median_before': raw.dropna().median(),
                'std_before': raw.dropna().std(),
                'nan_before': raw.isna().sum(),
                'mean_after': imp.mean(),
                'median_after': imp.median(),
                'std_after': imp.std(),
                'nan_after': imp.isna().sum()
            })
        return pd.DataFrame(stats_list)

class StatisticsPlotter:
    def __init__(self, output_folder="plots_statistics"):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def plot_case_summary(self, df_stats, max_cases=None):
        caseids = df_stats['caseid'].unique()
        if max_cases:
            caseids = caseids[:max_cases]
        for caseid in caseids:
            df_case = df_stats[df_stats['caseid'] == caseid]
            variables = df_case['variable'].values
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            axes[0].bar(variables, df_case['mean_before'], label='Before')
            axes[0].bar(variables, df_case['mean_after'], label='After')
            axes[0].set_title(f"Mean - Case {caseid}"); axes[0].legend()
            axes[1].bar(variables, df_case['median_before'], label='Before')
            axes[1].bar(variables, df_case['median_after'], label='After')
            axes[1].set_title(f"Median - Case {caseid}"); axes[1].legend()
            axes[2].bar(variables, df_case['std_before'], label='Before')
            axes[2].bar(variables, df_case['std_after'], label='After')
            axes[2].set_title(f"STD - Case {caseid}"); axes[2].legend()
            plt.tight_layout()
            plt.savefig(f"{self.output_folder}/case_{caseid}_stats.png")
            plt.close()

class PipelineRunner:
    def __init__(self, case_ids, variables, drug_vars=None, df_cases=None, df_cases_filtered=None):
        self.case_ids = case_ids
        self.variables = variables
        self.drug_vars = drug_vars or []
        self.df_cases = df_cases
        self.df_cases_filtered = df_cases_filtered
        self.results = []

    def run(self):
        for cid in self.case_ids:
            try:
                data = vitaldb.load_case(cid, self.variables, interval=1)
                medians = {v: np.nanmedian(data[:, i]) for i, v in enumerate(self.variables)}
                mads = {v: np.nanmedian(np.abs(data[:, i] - medians[v])) or 1e-6 for i, v in enumerate(self.variables)}
                analyzer = SignalAnalyzer(cid, data, self.variables, global_medians=medians, global_mads=mads)
                analyzer.analyze()
                processor = SignalProcessor(analyzer.data, analyzer.issues, self.variables)
                clean_data = processor.process()
                evaluator = Evaluator(data, clean_data, self.variables)
                stats = evaluator.compute_stats(raw_length=data.shape[0])
                stats['caseid'] = cid
                self.results.append(stats)
                plotter = StatisticsPlotter()
                plotter.plot_case_summary(stats)
            except Exception as e:
                print(f"Error processing case {cid}: {e}")

    def get_summary(self):
        return pd.concat(self.results, ignore_index=True)

st.set_page_config(page_title="VitalDB Streamlit Analyzer", layout="wide")
st.title("📊 VitalDB Case Analysis App")

@st.cache_data
def load_metadata():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    df_labs = pd.read_csv("https://api.vitaldb.net/labs")
    return df_cases, df_trks, df_labs

# df_cases, df_trks, df_labs = load_metadata()

signal_options = sorted(df_trks['tname'].value_counts().index.tolist())
default_signals = ["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"]
valid_defaults = [sig for sig in default_signals if sig in signal_options]

st.sidebar.header("🧪 Filter Configuration")
selected_signals = st.sidebar.multiselect("Select Required Signals", signal_options, default=valid_defaults)

drug_vars = ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"]
selected_drugs = st.sidebar.multiselect("Exclude Cases With Drugs", drug_vars, default=["intraop_mdz", "intraop_ftn"])

ane_types = sorted(df_cases['ane_type'].dropna().unique())
selected_ane_types = st.sidebar.multiselect("Select Surgery Types (ane_types)", ane_types)

if st.sidebar.button("🔍 Filter Valid Cases"):
    selector = CaseSelector(df_cases, df_trks, required_variables=selected_signals,
                            intraoperative_boluses=selected_drugs, optype_filter=selected_ane_types)
    valid_ids, filtered_df = selector.select_valid_cases()
    st.session_state.valid_ids = valid_ids
    st.session_state.filtered_df = filtered_df
    st.success(f"✅ {len(valid_ids)} valid cases found.")

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Filtered Cases", csv, "filtered_cases.csv", "text/csv")

if "valid_ids" in st.session_state:
    selected_cases = st.multiselect("🧪 Select Case IDs to Run Full Pipeline", st.session_state.valid_ids[:100])

    if st.button("🚀 Run Pipeline on Selected Cases"):
        from your_modules import PipelineRunner

        runner = PipelineRunner(
            case_ids=selected_cases,
            variables=selected_signals,
            drug_vars=selected_drugs,
            df_cases=df_cases,
            df_cases_filtered=st.session_state.filtered_df
        )
        runner.run()
        df_summary = runner.get_summary()

        st.subheader("📈 Summary Table")
        st.dataframe(df_summary)

        csv = df_summary.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Pipeline Summary", csv, "pipeline_summary.csv", "text/csv")

        st.success("✅ Pipeline run complete. Check charts in the output folder.")
