import streamlit as st
import pandas as pd
import numpy as np
import vitaldb

# modules/selector.py

import pandas as pd

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
# modules/analyzer.py

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class SignalAnalyzer:
    def __init__(self, caseid, data, variable_names, thresholds=None, plot=True, global_medians=None, global_mads=None):
        self.caseid = caseid
        self.data = data
        self.variable_names = variable_names
        self.thresholds = thresholds or {"missing": 0.05, "gap": 30, "jump": 100}
        self.plot_enabled = plot
        self.global_medians = global_medians or {}
        self.global_mads = global_mads or {}
        self.issues = {
            var: {'nan': [], 'gap': [], 'classified_gaps': [], 'outlier': [], 'outlier_values': [], 'jump': []}
            for var in variable_names
        }
        self.warnings = []

    def analyze(self):
        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i]
            original = signal.copy()
            nan_idx = np.where(np.isnan(signal))[0]
            self.issues[var]['nan'] = nan_idx.tolist()

            # Gap Detection
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

            # Classify gaps
            if gap_list:
                lengths = np.array([g["length"] for g in gap_list])
                median_gap = np.median(lengths)
                mad_gap = np.median(np.abs(lengths - median_gap)) or 1e-6
                for g in gap_list:
                    g["type"] = "long" if g["length"] > median_gap + 3.5 * mad_gap else "short"
                self.issues[var]['classified_gaps'] = gap_list

            # Outlier Detection
            outliers = []
            if "RATE" in var:
                outliers = np.where(signal < 0)[0].tolist()
            elif "BIS" in var:
                outliers = np.where((signal <= 0) | (signal > 100))[0].tolist()
            elif "NIBP" in var:
                invalid_idx = np.where(signal <= 0)[0].tolist()
                outliers.extend(invalid_idx)
            elif var in self.global_medians and var in self.global_mads:
                mad = self.global_mads[var] or 1e-6
                median = self.global_medians[var]
                mad_idx = np.where(np.abs(signal - median) > 3.5 * mad)[0].tolist()
                outliers.extend(mad_idx)

            self.issues[var]['outlier'] = sorted(set(outliers))
            self.issues[var]['outlier_values'] = original[outliers].tolist()

            # Jump Detection
            diffs = np.diff(signal)
            if len(diffs) > 0:
                median_diff = np.median(diffs)
                mad_diff = np.median(np.abs(diffs - median_diff)) or 1e-6
                jump_idx = np.where(np.abs(diffs - median_diff) > 3.5 * mad_diff)[0]
                self.issues[var]['jump'] = jump_idx.tolist()

    def plot(self, signal_units=None):
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

        fig.update_layout(title=f"Signal Diagnostics - Case {self.caseid}", height=300 * len(self.variable_names))
        return fig
# modules/processor.py

import numpy as np
from scipy.interpolate import interp1d

class SignalProcessor:
    def __init__(self, data, issues, variable_names,
                 gap_strategy='interpolate_short',
                 long_gap_strategy='leave',
                 interp_method='auto',
                 global_std_dict=None):
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
        starts = []

        for i in range(self.data.shape[1]):
            signal = self.data[:, i]
            start_idx = np.argmax(~np.isnan(signal))
            starts.append(start_idx)
            aligned_signals.append(signal[start_idx:])

        min_len = min(len(sig) for sig in aligned_signals)
        for i in range(len(aligned_signals)):
            aligned_signals[i] = aligned_signals[i][:min_len]

        self.data = np.column_stack(aligned_signals)

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
                if gap['type'] == 'short':
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
# modules/evaluator.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

        stats_df = pd.DataFrame(stats_list)
        length_info = {
            'original_length': raw_length,
            'final_length': self.imputed_df.shape[0],
            'trimmed_samples': raw_length - self.imputed_df.shape[0],
            'percentage_trimmed': round((raw_length - self.imputed_df.shape[0]) / raw_length * 100, 2)
        }

        return stats_df, pd.DataFrame([length_info])

    def plot_comparison(self, max_points=1000):
        fig, axes = plt.subplots(len(self.variable_names), 1, figsize=(14, 3 * len(self.variable_names)), sharex=True)
        if len(self.variable_names) == 1:
            axes = [axes]
        for i, var in enumerate(self.variable_names):
            ax = axes[i]
            raw = self.raw_df[var].values[:max_points]
            imp = self.imputed_df[var].values[:max_points]
            x = np.arange(len(raw))
            ax.plot(x, raw, 'o-', label='Raw', alpha=0.5)
            ax.plot(x, imp, '-', label='Imputed', linewidth=2)
            ax.set_title(var)
            ax.legend()
            ax.grid(True)
        plt.xlabel("Time (samples)")
        plt.tight_layout()
        plt.show()



st.set_page_config(layout="wide", page_title="VitalDB Analyzer")

# Load metadata once and cache it
@st.cache_data
def load_metadata():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    df_labs = pd.read_csv("https://api.vitaldb.net/labs")
    return df_cases, df_trks, df_labs

df_cases, df_trks, df_labs = load_metadata()

st.title("🩺 VitalDB Signal Analyzer")

# --- Sidebar: Variable and Filter Selection ---
st.sidebar.header("⚙️ Case Selection Settings")
ane_type = st.sidebar.selectbox("Anesthesia Type", df_cases["ane_type"].unique())
variables = st.sidebar.multiselect("Required Variables", df_trks["tname"].unique(),
                                   default=["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP"])

bolus_cols = ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"]
excluded_boluses = st.sidebar.multiselect("Exclude cases with boluses:", bolus_cols)

if st.sidebar.button("Filter Valid Cases"):
    selector = CaseSelector(df_cases, df_trks, ane_type=ane_type,
                            required_variables=variables,
                            intraoperative_boluses=excluded_boluses)
    valid_ids = selector.select_valid_cases()
    df_cases_filtered = df_cases[df_cases["caseid"].isin(valid_ids)]
    st.session_state["filtered_ids"] = valid_ids
    st.success(f"{len(valid_ids)} valid cases found.")

# --- Main Section: Case Analysis ---
if "filtered_ids" in st.session_state:
    case_id = st.selectbox("Select a Case ID", st.session_state["filtered_ids"])

    if st.button("Run Analysis"):
        with st.spinner("Running pipeline..."):
            # 1. Load signal
            data = vitaldb.load_case(case_id, variables, interval=1)

            # 2. Global stats
            global_medians = {var: np.nanmedian(data[:, i]) for i, var in enumerate(variables)}
            global_mads = {var: np.nanmedian(np.abs(data[:, i] - global_medians[var])) or 1e-6 for i, var in enumerate(variables)}

            # 3. Analyze
            analyzer = SignalAnalyzer(caseid=case_id, data=data, variable_names=variables,
                                      global_medians=global_medians, global_mads=global_mads, plot=False)
            analyzer.analyze()
            fig = analyzer.plot()
            st.plotly_chart(fig, use_container_width=True)

            # 4. Interpolate and align
            processor = SignalProcessor(data=analyzer.data, issues=analyzer.issues,
                                        variable_names=variables)
            imputed = processor.process()

            # 5. Evaluate
            evaluator = Evaluator(raw_data=data, imputed_data=imputed, variable_names=variables)
            stats_df, length_df = evaluator.compute_stats(raw_length=data.shape[0])

            st.subheader("📊 Imputation Summary")
            st.dataframe(stats_df)
            st.write("⏱ Length Info:", length_df.to_dict("records")[0])
            evaluator.plot_comparison()
