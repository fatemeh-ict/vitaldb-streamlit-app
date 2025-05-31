
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import vitaldb
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import pearsonr
from scipy.signal import correlate

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
      # filter by anesthesia type
        df_cases_filtered = self.df_cases[self.df_cases['ane_type'] == self.ane_type].copy()
        valid_case_ids = set(df_cases_filtered['caseid'])

        # check if you have the required variables in df_trks
        for var in self.required_variables:
            trk_cases = set(self.df_trks[self.df_trks['tname'] == var]['caseid'])
            valid_case_ids &= trk_cases

        # remove cases that received specific medications
        if self.intraoperative_boluses:
            valid_boluses = [col for col in self.intraoperative_boluses if col in df_cases_filtered.columns]
            df_cases_filtered = df_cases_filtered[~df_cases_filtered[valid_boluses].gt(0).any(axis=1)]
            valid_case_ids &= set(df_cases_filtered['caseid'])

        return sorted(list(valid_case_ids))
#======================
#signalanalyze
#===========================
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

    def get_summary_table(self):
      rows = []
      for var in self.variable_names:
        n_nans = len(self.issues[var]['nan'])
        n_gaps = len(self.issues[var]['gap'])
        n_outliers = len(self.issues[var]['outlier'])
        n_jumps = len(self.issues[var]['jump'])
        gap_long_count = sum(1 for g in self.issues[var]['classified_gaps'] if g['type'] == 'long')
        gap_long_pct = (gap_long_count / n_gaps) * 100 if n_gaps else 0
        rows.append({
            "Signal": var,
            "NaNs": n_nans,
            "Gaps": f"{n_gaps} ({gap_long_pct:.1f}% long)",
            "Outliers": n_outliers,
            "Jumps": n_jumps
        })
      return pd.DataFrame(rows)


    def analyze(self):
        #for i, var in enumerate (self.variable_names):  # Clean invalid BIS = 0 → NaN
         # if "BIS" in var:
          #  self.data[:, i][self.data[:, i] == 0] = np.nan

        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i]
            original = signal.copy()

            #  NaN Detection
            nan_idx = np.where(np.isnan(signal))[0]
            self.issues[var]['nan'] = nan_idx.tolist()
            if len(nan_idx) / len(signal) > self.thresholds["missing"]:
                self.warnings.append(f"[{self.caseid}] {var}: >{self.thresholds['missing']*100:.0f}% missing")

            # Gap Detection & Classification
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

            # Classification using MAD
            if gap_list:
                lengths = np.array([g["length"] for g in gap_list])
                median_gap = np.median(lengths)
                mad_gap = np.median(np.abs(lengths - median_gap)) or 1e-6
                for g in gap_list:
                    g["type"] = "long" if g["length"] > median_gap + 3.5 * mad_gap else "short"
                self.issues[var]['classified_gaps'] = gap_list

            #  Outlier Detection
            outliers = []
            if "RATE" in var:
                outliers = np.where(signal < 0)[0].tolist()

            elif "BIS" in var:
                outliers = np.where((signal <= 0) | (signal > 100))[0].tolist()

            elif "NIBP" in var:
                invalid_idx = np.where(signal <= 0)[0].tolist()
                outliers.extend(invalid_idx)
                if var in self.global_medians and var in self.global_mads:
                    mad = self.global_mads[var] or 1e-6
                    median = self.global_medians[var]
                    mad_idx = np.where(np.abs(signal - median) > 3.5 * mad)[0].tolist()
                    outliers.extend(mad_idx)

            elif var in self.global_medians and var in self.global_mads:
                mad = self.global_mads[var] or 1e-6
                median = self.global_medians[var]
                mad_idx = np.where(np.abs(signal - median) > 3.5 * mad)[0].tolist()
                outliers.extend(mad_idx)

            outliers = sorted(set(outliers))
            self.issues[var]['outlier'] = outliers
            self.issues[var]['outlier_values'] = original[outliers].tolist()

            #  Jump Detection using MAD
            diffs = np.diff(signal)
            if len(diffs) > 0:
                median_diff = np.median(diffs)
                mad_diff = np.median(np.abs(diffs - median_diff)) or 1e-6
                jump_idx = np.where(np.abs(diffs - median_diff) > 3.5 * mad_diff)[0]
                self.issues[var]['jump'] = jump_idx.tolist()
                if len(jump_idx):
                    self.warnings.append(f"[{self.caseid}] {var}: {len(jump_idx)} robust jumps")
            nan_pct = len(nan_idx) / len(signal) * 100
            gap_list = self.issues[var]['classified_gaps']
            num_long_gaps = sum(1 for g in gap_list if g['type'] == 'long')
            long_gap_pct = (num_long_gaps / len(gap_list)) * 100 if gap_list else 0

            print(f"{var}: NaNs={len(nan_idx)} ({nan_pct:.1f}%), Gaps={len(gap_list)} ({long_gap_pct:.1f}% long), Outliers={len(outliers)}, Jumps={len(jump_idx)}")


    def plot(self, signal_units=None):
        if not self.plot_enabled:
            return

        df = pd.DataFrame(self.data, columns=self.variable_names)
        df["time"] = np.arange(len(df))
        fig = make_subplots(rows=len(self.variable_names), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=self.variable_names)
        shown_legend={"outlier":False,"nan":False,"jump":False}

        for i, var in enumerate(self.variable_names):
            row = i + 1
            signal = df[var]
            # if signal.isna().all() or signal.min() == signal.max():
                # continue  # The signal is completely blank or unchanged.
            time = df["time"]
            unit = signal_units.get(var, "") if signal_units else ""

            fig.add_trace(go.Scatter(x=time, y=signal, mode='lines+markers', name=var), row=row, col=1)
            fig.update_yaxes(title_text=f"{var} ({unit})", row=row, col=1)

            if self.issues[var]['outlier']:
                idx = self.issues[var]['outlier']
                val = self.issues[var]['outlier_values']
                fig.add_trace(go.Scatter(x=time[idx], y=val, mode='markers',
                                         marker=dict(color='purple', size=6, symbol='star'),
                                         name="Outliers", showlegend=not shown_legend["outlier"]), row=row, col=1)
                shown_legend["outlier"]=True

            if self.issues[var]['nan']:
                idx = self.issues[var]['nan']
                fig.add_trace(go.Scatter(x=time[idx], y=[signal.min() - 5] * len(idx), mode='markers',
                                         marker=dict(color='gray', size=5, symbol='line-ns-open'),
                                         name="NaNs",showlegend=not shown_legend["nan"]), row=row, col=1)
                shown_legend["nan"]=True

            if self.issues[var]['jump']:
                idx = self.issues[var]['jump']
                fig.add_trace(go.Scatter(x=time[idx], y=signal[idx], mode='markers',
                                         marker=dict(color='orange', size=7, symbol='x'),
                                         name="Jumps",showlegend=not shown_legend["jump"]), row=row, col=1)
                shown_legend["jump"]=True

            if self.issues[var]['classified_gaps']:
                for gap in self.issues[var]['classified_gaps']:
                    start = gap['start']
                    end = gap['start'] + gap['length']
                    color = 'red' if gap['type'] == 'short' else 'black'
                    fig.add_shape(type="rect",
                                  x0=time[start], x1=time[end - 1],
                                  y0=signal.min(), y1=signal.max(),
                                  line=dict(width=0),
                                  fillcolor=color,
                                  opacity=0.2,
                                  row=row, col=1)

        fig.update_xaxes(title_text="Time (s)")
        fig.update_layout(title=f"Signal Diagnostics - Case {self.caseid}", height=100 * len(self.variable_names))
        # st.plotly_chart(fig, use_container_width=True)
        return fig
#=====================
#interpolate
#==============================
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
        if "RATE" in varname:
            return'linear'

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
        # We cut the signals from their first valid value and then make them equal in length.
        aligned_signals = []
        starts = []

        for i in range(self.data.shape[1]):
            signal = self.data[:, i]
            start_idx = np.argmax(~np.isnan(signal))
            starts.append(start_idx)
            aligned_signals.append(signal[start_idx:])

        # Minimum common length
        min_len = min(len(sig) for sig in aligned_signals)
        for i in range(len(aligned_signals)):
            aligned_signals[i] = aligned_signals[i][:min_len]

        self.data = np.column_stack(aligned_signals)

    def interpolate_short_gaps(self):
        for i, var in enumerate(self.variable_names):
            signal = self.data[:, i]
            # Convert BIS == 0 to NaN only for interpolation
            if "BIS" in var:
              original_nan_count = np.isnan(signal).sum()
              signal[signal == 0] = np.nan
              nan_after_zero_replace = np.isnan(signal).sum()
              n_added_from_zero = nan_after_zero_replace - original_nan_count
              self.issues[var]["nan_before"] = original_nan_count
              self.issues[var]["zero_to_nan"] = n_added_from_zero

            # if "BIS" in var:
              # signal[signal == 0] = np.nan

            gaps = self.issues[var].get('classified_gaps', [])
            x = np.arange(len(signal))
            valid_mask = ~np.isnan(signal)

            if valid_mask.sum() < 2:
                continue

            method = self.select_interp_method(signal, var)
            f = interp1d(x[valid_mask], signal[valid_mask], kind=method, fill_value='extrapolate')
            self.issues[var]["nan_after_interp"] = np.isnan(signal).sum()


            for gap in gaps:
                start = gap['start']
                end = start + gap['length']

                if gap['type'] == 'short':
                  if start==0:
                    signal[start:end] = signal[valid_mask][0]
                  elif end==len(signal):
                    signal[start:end] = signal[valid_mask][-1]
                  else :
                    signal[start:end] = f(x[start:end])

                elif gap['type'] == 'long':
                    if self.long_gap_strategy == 'zero':
                        signal[start:end] = 0
                    elif self.long_gap_strategy == 'nan':
                        signal[start:end] = np.nan
                    elif self.long_gap_strategy == 'leave':
                        continue

            self.data[:, i] = signal

    def process(self):
        if self.gap_strategy == 'interpolate_short':
            self.interpolate_short_gaps()
        self.align_signals_soft()
        return self.data
#====================================
#evaluator
#===========================
class Evaluator:
    def __init__(self, raw_data, imputed_data, variable_names):

        self.raw_df = pd.DataFrame(raw_data, columns=variable_names)
        self.imputed_df = pd.DataFrame(imputed_data, columns=variable_names)
        self.variable_names = variable_names
        self.stats = None
        self.length_info = None

    def compute_stats(self,raw_length):
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

        self.stats = pd.DataFrame(stats_list)

        self.length_info = {
            'original_length': raw_length if raw_length else self.raw_df.shape[0],  # Use self.raw_df for original length
            'final_length': self.imputed_df.shape[0],
            'trimmed_samples': (raw_length - self.imputed_df.shape[0]) if raw_length else (self.raw_df.shape[0] - self.imputed_df.shape[0]),
            'percentage_trimmed': round(
                (raw_length - self.imputed_df.shape[0]) / raw_length * 100, 2
            ) if raw_length else round(
                (self.raw_df.shape[0] - self.imputed_df.shape[0]) / self.raw_df.shape[0] * 100, 2
            )
        }
        return self.stats, pd.DataFrame([self.length_info])


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
        # plt.show()

        return fig

#=========================
#plot
#===============================

class StatisticsPlotter:
    def __init__(self, output_folder="plots_statistics"):
        # self.output_folder = output_folder
        # os.makedirs(output_folder, exist_ok=True)
        pass
    # Plot mean, median, std before and after interpolation for each case
    def plot_case_summary(self, df_stats, max_cases=None,rel_change_threshold=0.2):
       if 'caseid' not in df_stats.columns:
           print("Warning: 'caseid' column not found in df_stats. Skipping plot_case_summary.")
           return
       caseids = df_stats['caseid'].unique()
       if max_cases:
          caseids = caseids[:max_cases]

       for caseid in caseids:
        try:
            df_case = df_stats[df_stats['caseid'] == caseid]
            variables = df_case['variable'].values

            means_before = df_case['mean_before'].values
            means_after = df_case['mean_after'].values
            medians_before = df_case['median_before'].values
            medians_after = df_case['median_after'].values

            # detection large relative changes
            rel_mean_change = np.abs(means_after - means_before) / (np.abs(means_before) + 1e-6)
            rel_median_change = np.abs(medians_after - medians_before) / (np.abs(medians_before) + 1e-6)

            # if at least one signal has a large change
            if np.any(rel_mean_change > rel_change_threshold) or np.any(rel_median_change > rel_change_threshold):
                fig, axes = plt.subplots(3, 1, figsize=(12, 10))

                # Mean
                axes[0].bar(np.arange(len(variables)) - 0.2, means_before, width=0.4, label='Before')
                axes[0].bar(np.arange(len(variables)) + 0.2, means_after, width=0.4, label='After')
                axes[0].set_title(f'Case {caseid} - Mean Comparison')
                axes[0].set_xticks(np.arange(len(variables)))
                axes[0].set_xticklabels(variables, rotation=45, ha='right')
                axes[0].legend()
                axes[0].grid(True)

                # Median
                axes[1].bar(np.arange(len(variables)) - 0.2, medians_before, width=0.4, label='Before')
                axes[1].bar(np.arange(len(variables)) + 0.2, medians_after, width=0.4, label='After')
                axes[1].set_title(f'Case {caseid} - Median Comparison')
                axes[1].set_xticks(np.arange(len(variables)))
                axes[1].set_xticklabels(variables, rotation=45, ha='right')
                axes[1].legend()
                axes[1].grid(True)

                # Std
                stds_before = df_case['std_before'].values
                stds_after = df_case['std_after'].values
                axes[2].bar(np.arange(len(variables)) - 0.2, stds_before, width=0.4, label='Before')
                axes[2].bar(np.arange(len(variables)) + 0.2, stds_after, width=0.4, label='After')
                axes[2].set_title(f'Case {caseid} - Std Comparison')
                axes[2].set_xticks(np.arange(len(variables)))
                axes[2].set_xticklabels(variables, rotation=45, ha='right')
                axes[2].legend()
                axes[2].grid(True)

                plt.tight_layout()
                st.pyplot(fig)
                # plt.savefig(f"{self.output_folder}/case_{caseid}_statistics_comparison.png")
                plt.close()
                # print(f" Saved plot for Case {caseid} due to significant change.")

            else:
                print(f" No significant change for Case {caseid}, plot skipped.")

        except Exception as e:
            print(f" Error plotting case {caseid}: {e}")


    #  Compare histograms of numerical features before and after filtering
    def compare_numerical(self, df_all, df_filtered, columns):

        for col in columns:
            try:
                plt.figure(figsize=(8, 5))

                sns.histplot(df_all[col], kde=True, bins=30, color='skyblue', label="Before Filtering",alpha=0.4)
                sns.histplot(df_filtered[col], kde=True, bins=30, color='salmon', label='After Filtering', alpha=0.4)

                plt.title(f'{col} - Before vs After Filtering')
                plt.xlabel(col)
                plt.ylabel('Frequency')

                plt.tight_layout()
                fig = plt.gcf()
                st.pyplot(fig)
                # plt.savefig(f"{self.output_folder}/{col}_numerical_comparison.png")
                plt.close()
                print(f"Numerical comparison saved for: {col}")

            except Exception as e:
                print(f" Error plotting numerical column {col}: {e}")

    # Compare count plots of categorical features before and after filtering
    def compare_categorical(self, df_all, df_filtered, columns):

        for col in columns:
            try:
                plt.figure(figsize=(12, 5))

                plt.subplot(1, 2, 1)
                sns.countplot(x=df_all[col], palette='Blues')
                plt.title(f'{col} - Before Filtering')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)

                plt.subplot(1, 2, 2)
                sns.countplot(x=df_filtered[col], palette='Reds')
                plt.title(f'{col} - After Filtering')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)

                plt.tight_layout()
                fig = plt.gcf()
                st.pyplot(fig)
                # plt.savefig(f"{self.output_folder}/{col}_categorical_comparison.png")
                plt.close()
                print(f"Categorical comparison saved for: {col}")

            except Exception as e:
                print(f" Error plotting categorical column {col}: {e}")
#===================
#pipline
#=====================
class PipelineRunner:

    def __init__(self, case_ids, variables, drug_vars=None, df_cases=None, df_cases_filtered=None):
        self.case_ids = case_ids
        self.variables = variables
        self.drug_vars = drug_vars or []
        self.df_cases = df_cases
        self.df_cases_filtered = df_cases_filtered
        self.results = []


    def compute_global_stats(self, case_ids_for_stats, variables, n_samples=100):
        all_samples = {var: [] for var in variables}

        for cid in case_ids_for_stats:
            try:
                data = vitaldb.load_case(cid, variables, interval=1)
                n_rows = data.shape[0]
                if n_rows < n_samples:
                    continue
                idx = np.sort(np.random.choice(n_rows, size=n_samples, replace=False))
                for i, var in enumerate(variables):
                    sampled = data[idx, i]
                    sampled = sampled[~np.isnan(sampled)]
                    all_samples[var].extend(sampled.tolist())
            except Exception as e:
                print(f"Skipping case {cid} due to error: {e}")
                continue

        self.global_medians = {}
        self.global_mads = {}
        for var in variables:
            vec = np.array(all_samples[var])
            vec = vec[~np.isnan(vec)]
            self.global_medians[var] = np.median(vec)
            self.global_mads[var] = np.median(np.abs(vec - self.global_medians[var])) or 1e-6

        print("Global medians and MADs computed using random sampling.")



    def run(self):
        print("Loading all cases to compute global medians and MADs...")

        #  Process each case individually
        for cid in self.case_ids:
            try:
                print(f" Processing case {cid}...")

                # 1. Load raw data
                data = vitaldb.load_case(cid, self.variables, interval=1)

                # 2. Analyze signal
                signal_units = {
                    "BIS/BIS": "%",
                    "Solar8000/NIBP_SBP": "mmHg",
                    "Solar8000/NIBP_DBP": "mmHg",
                    "Orchestra/PPF20_RATE": "bpm",
                    "Orchestra/RFTN20_RATE": "bpm",
                }
                analyzer = SignalAnalyzer(
                    caseid=cid,
                    data=data,
                    variable_names=self.variables,
                    global_medians=self.global_medians,
                    global_mads=self.global_mads,
                    plot=True
                )
                analyzer.analyze()
                if cid == self.case_ids[0]: # Corrected the condition here
                  analyzer.plot(signal_units=signal_units)

                # 3. Process signal
                processor = SignalProcessor(
                    data=analyzer.data,
                    issues=analyzer.issues,
                    variable_names=self.variables,
                    gap_strategy='interpolate_short',
                    long_gap_strategy='nan',
                    interp_method='auto',
                    global_std_dict={var: np.std(data[:, i][~np.isnan(data[:, i])]) for i, var in enumerate(self.variables)}
                )
                clean_data = processor.process()

                # 4. Evaluate
                evaluator = Evaluator(raw_data=data,
                    imputed_data=clean_data,
                    variable_names=self.variables
                )
                stats_df, length_df = evaluator.compute_stats(raw_length=data.shape[0])
                stats_df["caseid"] = cid
                print("\n Imputation Statistics:")
                print(stats_df)
                print(length_df)
                evaluator.plot_comparison(max_points=1000)
                self.results.append(stats_df)


                plotter = StatisticsPlotter(output_folder="my_output_plots")

                # Statistical chart for caseid
                plotter.plot_case_summary(stats_df, max_cases=3)

                # Numerical comparison
                plotter.compare_numerical(self.df_cases, self.df_cases_filtered, columns=[
                                                      'caseid', 'subjectid','age', 'bmi', 'weight', 'height'])

                # categorical comparison
                plotter.compare_categorical(self.df_cases, self.df_cases_filtered, columns=[
                                  'sex', 'ane_type', 'optype', 'department', 'approach', 'position', 'cormack'])


            except Exception as e:
                print(f"Error in case {cid}: {e}")

    def get_summary(self):

        return pd.concat(self.results, ignore_index=True)

#=================
#Sorrelation
#===========================



class StatisticalTester:
    def __init__(self, raw_signals, imputed_signals, variable_names):
        self.raw_signals = raw_signals
        self.imputed_signals = imputed_signals
        self.variables = variable_names
        self.results = []
        self.df_tests = None

    def run_tests(self):
        from scipy.stats import pearsonr, ttest_rel
        results = []

        for var in self.variables:
            corrs, pvals, tstats, tpvals = [], [], [], []

            for cid in self.raw_signals:
                raw = self.raw_signals[cid][:, self.variables.index(var)]
                imp = self.imputed_signals[cid][:, self.variables.index(var)]

                # Make sure raw and imp have the same length
                min_len = min(len(raw), len(imp))
                raw = raw[:min_len]
                imp = imp[:min_len]

                mask = ~np.isnan(raw) & ~np.isnan(imp)
                if mask.sum() < 10:
                    continue

                raw_valid, imp_valid = raw[mask], imp[mask]

                corr, pval = pearsonr(raw_valid, imp_valid)
                tstat, tpval = ttest_rel(raw_valid, imp_valid)

                corrs.append(corr)
                pvals.append(pval)
                tstats.append(tstat)
                tpvals.append(tpval)

            results.append({
                "variable": var,
                "avg_corr": np.mean(corrs),
                "avg_corr_pval": np.mean(pvals),
                "avg_tstat": np.mean(tstats),
                "avg_ttest_pval": np.mean(tpvals),
                "n_cases": len(corrs),
                "corr_list": corrs,
                "pval_list": pvals
            })

        self.df_tests = pd.DataFrame(results)
        return self.df_tests

    def plot_boxplots(self, output_folder="plots_statistics"):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os

        os.makedirs(output_folder, exist_ok=True)

        corr_data = [r["corr_list"] for r in self.df_tests.to_dict("records")]
        pval_data = [r["pval_list"] for r in self.df_tests.to_dict("records")]

        # Boxplot correlations
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=corr_data)
        plt.xticks(ticks=range(len(self.variables)), labels=self.variables, rotation=45)
        plt.ylabel("Correlation (Raw vs Imputed)")
        plt.title("Boxplot of Correlations")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()
        # plt.savefig(f"{output_folder}/boxplot_correlation.png")

        # Boxplot p-values
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=pval_data)
        plt.xticks(ticks=range(len(self.variables)), labels=self.variables, rotation=45)
        plt.ylabel("P-Value")
        plt.title("Boxplot of P-values")
        plt.tight_layout()
        plt.savefig(f"{output_folder}/boxplot_pvalues.png")
        plt.close()
        print(" Saved boxplots.")

    def plot_heatmap(self, output_folder="plots_statistics"):
        
        # os.makedirs(output_folder, exist_ok=True)

        n = len(self.variables)
        mean_corr_matrix = np.zeros((n, n))
        count_matrix = np.zeros((n, n))

        for cid in self.imputed_signals:
            df = pd.DataFrame(self.imputed_signals[cid], columns=self.variables)
            corr = df.corr(method='pearson', min_periods=10)
            mask = ~corr.isna()
            mean_corr_matrix += corr.fillna(0).values
            count_matrix += mask.values.astype(int)

        mean_corr_matrix /= np.maximum(count_matrix, 1)
        df_corr = pd.DataFrame(mean_corr_matrix, index=self.variables, columns=self.variables)

        plt.figure(figsize=(10, 8))
        sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
        plt.title("Heatmap of Signal Correlations (Imputed Data)")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        # plt.savefig(f"{output_folder}/heatmap_correlations.png")
        plt.close()
        # print(" Saved heatmap.")


    def compute_lagged_correlation(self, x, y, max_lag=60):
      correlations = []
      lags = np.arange(0, max_lag + 1)
      for lag in lags:
          if lag == 0:
              x_lag = x
              y_lag = y
          else:
              x_lag = x[:-lag]
              y_lag = y[lag:]
          mask = ~np.isnan(x_lag) & ~np.isnan(y_lag)
          if mask.sum() > 10:
             corr, _ = pearsonr(x_lag[mask], y_lag[mask])
          else:
              corr = np.nan
          correlations.append(corr)
      best_lag = lags[np.nanargmax(correlations)]
      best_corr = np.nanmax(correlations)
      return best_lag, best_corr, correlations

    def run_lagged_tests(self, signal_x="Orchestra/PPF20_RATE", signal_y="BIS/BIS", max_lag=60):
      lag_list, corr_list = [], []

      for cid in self.imputed_signals:
        x = self.imputed_signals[cid][:, self.variables.index(signal_x)]
        y = self.imputed_signals[cid][:, self.variables.index(signal_y)]

        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]

        best_lag, best_corr, _ = self.compute_lagged_correlation(x, y, max_lag=max_lag)
        lag_list.append(best_lag)
        corr_list.append(best_corr)

      df_lag = pd.DataFrame({
        "caseid": list(self.imputed_signals.keys()),
        "best_lag": lag_list,
        "max_corr": corr_list
           })

      print(df_lag.describe())
      return df_lag

    def plot_lagged_summary(self,df_lag):

      plt.figure(figsize=(14, 6))

      plt.subplot(1, 2, 1)
      sns.boxplot(y=df_lag["best_lag"])
      plt.title("Boxplot of Best Lags")
      plt.ylabel("Lag (samples)")

      plt.subplot(1, 2, 2)
      sns.histplot(df_lag["max_corr"], bins=10, kde=True, color="skyblue")
      plt.title("Distribution of Maximum Lagged Correlations")
      plt.xlabel("Max Correlation")
      plt.ylabel("Frequency")

      plt.tight_layout()
      st.pyplot(plt.gcf())
      plt.close()


    def compute_cross_correlation(self, x, y, max_lag=60):
      min_len = min(len(x), len(y))
      x = x[:min_len]
      y = y[:min_len]

      mask = ~np.isnan(x) & ~np.isnan(y)
      x = x[mask]
      y = y[mask]

      if len(x) < 10:
          return None, None, None

      # محاسبه full cross-correlation
      corr = correlate(y - np.mean(y), x - np.mean(x), mode='full')
      corr /= (np.std(x) * np.std(y) * len(x))  #Normalization

      lags = np.arange(-len(x) + 1, len(x))
      center = len(corr) // 2
      half = max_lag

      corr = corr[center - half:center + half + 1]
      lags = lags[center - half:center + half + 1]

      best_lag = lags[np.argmax(corr)]
      best_corr = np.max(corr)
      return best_lag, best_corr, (lags, corr)

    def run_cross_correlation_tests(self, signal_x="Orchestra/PPF20_RATE", signal_y="BIS/BIS", max_lag=60):
      lag_list, corr_list, case_list = [], [], []

      for cid in self.imputed_signals:
          x = self.imputed_signals[cid][:, self.variables.index(signal_x)]
          y = self.imputed_signals[cid][:, self.variables.index(signal_y)]

          result = self.compute_cross_correlation(x, y, max_lag=max_lag)
          if result[0] is not None:
              best_lag, best_corr, _ = result
              lag_list.append(best_lag)
              corr_list.append(best_corr)
              case_list.append(cid)

      df_cross = pd.DataFrame({
        "caseid": case_list,
        "best_lag": lag_list,
        "max_corr": corr_list
      })

      print(df_cross.describe())
      return df_cross

    def plot_cross_correlation(self, caseid, signal_x, signal_y, max_lag=60):
      x = self.imputed_signals[caseid][:, self.variables.index(signal_x)]
      y = self.imputed_signals[caseid][:, self.variables.index(signal_y)]

      result = self.compute_cross_correlation(x, y, max_lag=max_lag)
      if result[0] is None:
          print(f"Not enough valid data in case {caseid}")
          return

      best_lag, best_corr, (lags, corr) = result

      plt.figure(figsize=(10, 5))
      plt.plot(lags, corr)
      plt.axvline(0, color='gray', linestyle='--', label="Zero Lag")
      plt.axvline(best_lag, color='red', linestyle='--', label=f"Best Lag = {best_lag}")
      plt.title(f"Cross-Correlation: {signal_x} vs {signal_y} (Case {caseid})")
      plt.xlabel("Lag (samples)")
      plt.ylabel("Correlation")
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      st.pyplot(plt.gcf())
      plt.close()


#------------------------------------------------------------------
@st.cache_data
# def get_global_stats_cached(case_ids, variables):
def get_global_stats_cached(case_ids, variables, n_samples=100):
    runner = PipelineRunner(case_ids, variables)
    runner.compute_global_stats(case_ids, variables, n_samples=100)
    # runner.compute_global_stats(case_ids, variables)
    return runner.global_medians, runner.global_mads


# Rewriting Tab 1 with signal group selection, anesthesia type, and bolus exclusions + download buttons

tabs = st.tabs([" Select Cases", " Signal Quality", " Interpolation", " Evaluation", " Export"," Analysis","Correlation & T-Test"])

with tabs[0]:
    st.header("Step 1: Select Valid Cases")

    ane_type = st.selectbox("Anesthesia Type", ["General", "Spinal", "Regional"])
    intraoperative_boluses = st.multiselect("Exclude Boluses", [
        "intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"
    ])

    selected_group = st.radio("Signal Group", ["Group 1 (RFTN20)", "Group 2 (RFTN50)", "Both Groups"])

    group1_vars = [
        "BIS/BIS", "Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP",
        "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE",
        "Orchestra/PPF20_CE", "Orchestra/RFTN20_CE"
    ]
    group2_vars = [
        "BIS/BIS", "Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP",
        "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE",
        "Orchestra/PPF20_CE", "Orchestra/RFTN50_CE"
    ]

    if st.button(" Load and Filter Cases"):
        df_cases = pd.read_csv("https://api.vitaldb.net/cases")
        df_trks = pd.read_csv("https://api.vitaldb.net/trks")
        df_labs = pd.read_csv("https://api.vitaldb.net/labs")

        selector1 = CaseSelector(df_cases, df_trks, ane_type=ane_type,
                                 required_variables=group1_vars,
                                 intraoperative_boluses=intraoperative_boluses)

        selector2 = CaseSelector(df_cases, df_trks, ane_type=ane_type,
                                 required_variables=group2_vars,
                                 intraoperative_boluses=intraoperative_boluses)

        ids1 = set(selector1.select_valid_cases())
        ids2 = set(selector2.select_valid_cases())

        if selected_group == "Group 1 (RFTN20)":
            valid_ids = ids1
            variables = group1_vars
        elif selected_group == "Group 2 (RFTN50)":
            valid_ids = ids2
            variables = group2_vars
        else:
            valid_ids = ids1.union(ids2)
            variables = list(set(group1_vars + group2_vars))

        df_cases_filtered = df_cases[df_cases['caseid'].isin(valid_ids)]
        df_trks_filtered = df_trks[df_trks['caseid'].isin(valid_ids)]
        df_labs_filtered = df_labs[df_labs['caseid'].isin(valid_ids)]

        st.session_state["df_cases"] = df_cases
        st.session_state["df_trks"] = df_trks
        st.session_state["df_labs"] = df_labs
        st.session_state["valid_ids"] = sorted(list(valid_ids))
        st.session_state["df_cases_filtered"] = df_cases_filtered
        st.session_state["df_trks_filtered"] = df_trks_filtered
        st.session_state["df_labs_filtered"] = df_labs_filtered
        # st.session_state["variables"] = variables
        st.session_state["variables"] = ['BIS/BIS','Solar8000/NIBP_SBP','Solar8000/NIBP_DBP','Orchestra/PPF20_RATE',
                                         'Orchestra/RFTN20_RATE']
        


        # Calculating the global mean and MAD on the first 10 cases

        sample_ids = random.sample(st.session_state["valid_ids"], min(300, len(st.session_state["valid_ids"])))
        global_medians, global_mads = get_global_stats_cached(sample_ids, variables, n_samples=100)
        st.session_state["global_medians"] = global_medians
        st.session_state["global_mads"] = global_mads


        st.success(f"{len(valid_ids)} valid case(s) found.")
        st.dataframe(df_cases_filtered.head(10))

        st.download_button(" Download Filtered Cases",
                           df_cases_filtered.to_csv(index=False),
                           file_name="filtered_cases.csv")

        st.download_button(" Download Filtered Tracks",
                           df_trks_filtered.to_csv(index=False),
                           file_name="filtered_trks.csv")

        st.download_button(" Download Filtered Labs",
                           df_labs_filtered.to_csv(index=False),
                           file_name="filtered_labs.csv")        
        


#---------------------------------------
with tabs[1]:
    st.header("Step 2: Signal Quality Analysis")
    st.write(" The second tab is running.")


    if "valid_ids" not in st.session_state or "variables" not in st.session_state:
        st.warning(" Please select the cases first in step 1.")
    else:
        selected_case = st.selectbox(" Select Case ID", st.session_state["valid_ids"])

        if st.button(" Signal analysis of this case"):
            st.write(" The analyze button was clicked")

            try:
                data = vitaldb.load_case(selected_case, st.session_state["variables"], interval=1)
                st.write(" Sample data:")
                st.dataframe(pd.DataFrame(data, columns=st.session_state["variables"]).head())
                df = pd.DataFrame(data, columns=st.session_state["variables"])
                # Checking the percentage of NaN
                st.write(" Percentage of NaN data in each column:")
                st.write(df.isna().mean())
                # Check that all columns are not empty
                if df.dropna(how='all').empty:
                    st.warning(" All columns have only NaN values. Please choose another Case.")
                    st.stop()
                st.write(" Data loaded:", data.shape if hasattr(data, 'shape') else "without shape")


                runner = PipelineRunner(
                    case_ids=[selected_case],
                    variables=st.session_state["variables"],
                    df_cases=st.session_state["df_cases"],
                    df_cases_filtered=st.session_state["df_cases_filtered"]
                )

                # Calculate the global mean and MAD for this case only.
                # global_medians, global_mads = get_global_stats_cached([selected_case_interp], st.session_state["variables"])

                global_medians = st.session_state["global_medians"]
                global_mads = st.session_state["global_mads"]



                analyzer = SignalAnalyzer(
                    caseid=selected_case,
                    data=data,
                    variable_names=st.session_state["variables"],
                    global_medians=global_medians,
                    global_mads=global_mads,
                    plot=True
                )
                st.write(" Analyzer was created")

                analyzer.analyze()
                st.session_state["analyzer_issues"] = analyzer.issues

                summary_df = analyzer.get_summary_table()
                st.subheader(" Signals statistics table ")
                st.dataframe(summary_df)

                st.write(" Analysis was done")
                # fig= analyzer.plot()
                signal_units = {"BIS/BIS": " ", "Solar8000/NIBP_SBP": "mmHg", "Solar8000/NIBP_DBP": "mmHg",
                       "Orchestra/PPF20_RATE": "bpm", "Orchestra/RFTN20_RATE": "bpm",
                       "Orchestra/RFTN50_RATE": "bpm",
                         }

                fig = analyzer.plot(signal_units=signal_units)
                st.session_state["signal_summary_df"] = summary_df
                st.session_state["signal_plot"] = fig

                if fig is None:
                    st.error(" analyzer.plot() returned None.")
                elif len(fig.data) == 0:
                    st.warning(" Plot created but contains no traces.")
                else:
                    st.success(f"Plot created with {len(fig.data)} trace(s).")
                    st.plotly_chart(fig, use_container_width=True)
                    for i, trace in enumerate(fig.data):
                        st.write(f" Trace {i}: {trace.name} — {len(trace.x)} points")

                    st.write(" analyzer.plot() returned:", type(fig))
                    st.write("Number of traces in fig:", len(fig.data) if fig else "No figure")



                
                # if fig is None:
                #     st.error("plot() returned None.")
                # elif not hasattr(fig, 'data') or len(fig.data) == 0:
                #     st.error(" plot() is empty — no data in fig.")
                # else:
                #     st.success(f" Plot created with {len(fig.data)} trace(s).")
                #     for i, trace in enumerate(fig.data):
                #         st.write(f" Trace {i}: name = {trace.name}, points = {len(trace.x)}")

                # if fig and fig.data:
                #     st.plotly_chart(fig, use_container_width=True)
                #     st.success(" Signal quality analysis completed successfully.")
                # else:
                #     st.warning("No chart was generated. There may be no data to display.")

                
                # if fig:
                    # st.plotly_chart(fig, use_container_width=True)
                # else:
                    # st.warning("There are no charts to display..")


                # if fig is None or not fig.data:
                    # st.warning(" No chart was generated. There may be no data to display..")
                # else:
                    # st.plotly_chart(fig, use_container_width=True)
                    # st.success(" Signal quality analysis completed successfully..")
                    
                # st.write(" number trace in fig:", len(fig.data))
                # for i, trace in enumerate(fig.data):
                #     st.write(f" trace {i}: name={trace.name}, points={len(trace.x)}")
                # st.write(" Type of fig:", type(fig))
                # st.write(" number trace in fig:", len(fig.data))
                # for i, trace in enumerate(fig.data):
                #     st.write(f" trace {i}: name={trace.name}, points={len(trace.x)}")
                
                # st.plotly_chart(fig, use_container_width=True)
                # st.write(" The diagram was drawn")
                


                st.success(" Signal quality analysis completed successfully..")

            except Exception as e:
                st.error(f" Error loading or analyzing case {selected_case}: {e}")
#-------------------------------------
with tabs[2]:
    st.header("Step 3: Signal Interpolation & Alignment")

    if "valid_ids" not in st.session_state or "variables" not in st.session_state:
        st.warning("Please complete steps 1 and 2 first..")
    else:
        selected_case_interp = st.selectbox(" Selecting a Case for Interpolation", st.session_state["valid_ids"], key="interp_case")
        interp_method_option = st.selectbox(
        " Select the signal interpolation method.:",
        options=["auto", "linear", "cubic", "slinear"],
        index=0,  # default: auto
        help="In 'auto' mode, the appropriate method is automatically selected for each variable.."
        )


        if st.button(" Perform signal interpolation and alignment"):
            try:
                # Upload raw data
                raw_data = vitaldb.load_case(selected_case_interp, st.session_state["variables"], interval=1)

                # Re-run analysis to use global MAD/median
                runner = PipelineRunner(
                    case_ids=[selected_case_interp],
                    variables=st.session_state["variables"],
                    df_cases=st.session_state["df_cases"],
                    df_cases_filtered=st.session_state["df_cases_filtered"]
                )

                #global_medians, global_mads = get_global_stats_cached([selected_case], st.session_state["variables"])

                global_medians = st.session_state["global_medians"]
                global_mads = st.session_state["global_mads"]



                analyzer = SignalAnalyzer(
                    caseid=selected_case_interp,
                    data=raw_data,
                    variable_names=st.session_state["variables"],
                    global_medians=global_medians,
                    global_mads=global_mads,
                    plot=False
                )
                analyzer.analyze()

                processor = SignalProcessor(
                    data=analyzer.data,
                    issues=analyzer.issues,
                    variable_names=st.session_state["variables"],
                    gap_strategy='interpolate_short',
                    long_gap_strategy='nan',
                    interp_method=interp_method_option,
                    global_std_dict={var: np.std(raw_data[:, i][~np.isnan(raw_data[:, i])]) for i, var in enumerate(st.session_state["variables"])}
                )
                imputed_data = processor.process()

                # Save for next step
                st.session_state["imputed_data"] = imputed_data
                st.session_state["raw_data"] = raw_data
                st.session_state["selected_case_interp"] = selected_case_interp

                st.success(" The signal was successfully interpolated and aligned.")

                st.write(" An example of raw data and post-interpolation:")
                st.dataframe(pd.DataFrame(imputed_data, columns=st.session_state["variables"]).head())

            except Exception as e:
                st.error(f" Error in interpolation: {e}")
#-------------------------------------------------

# with tabs[3]:
#     st.header("Step 4: Evaluation of Imputed Signals")

#     if "raw_data" not in st.session_state or "imputed_data" not in st.session_state:
#         st.warning(" First, perform the interpolation step..")
#     else:
#         try:
#             evaluator = Evaluator(
#                 raw_data=st.session_state["raw_data"],
#                 imputed_data=st.session_state["imputed_data"],
#                 variable_names=st.session_state["variables"]
#             )

#             stats_df, length_df = evaluator.compute_stats(
#                 raw_length=st.session_state["raw_data"].shape[0]
#             )
#             stats_df["caseid"] = st.session_state["selected_case_interp"]

#             # اطمینان از وجود ستون‌ها
#             for field in ["nan_before", "zero_to_nan", "nan_after_interp", "zero_nan_ratio(%)"]:
#                 if field not in stats_df.columns:
#                     stats_df[field] = np.nan

#             # پیام و محاسبه ویژه فقط برای سیگنال BIS/BIS
#             if "analyzer_issues" in st.session_state:
#                 bis_info = st.session_state["analyzer_issues"].get("BIS/BIS", {})
#                 if "zero_to_nan" in bis_info and "nan_before" in bis_info:
#                     zero_nan = bis_info["zero_to_nan"]
#                     total_nan = bis_info["nan_before"]
#                     ratio = round(100 * zero_nan / total_nan, 2) if total_nan else 0

#                     # پیام به کاربر
#                     st.info(f"🔍 از بین {total_nan} مقدار NaN در سیگنال BIS قبل از درون‌یابی، تعداد {zero_nan} مقدار (معادل {ratio}٪) به دلیل مقدار صفر بوده و به NaN تبدیل شده‌اند.")

#                     # درج درصد فقط برای BIS در جدول آمار
#                     idx = stats_df[stats_df["variable"] == "BIS/BIS"].index
#                     if not idx.empty:
#                         stats_df.loc[idx, "zero_nan_ratio(%)"] = ratio

#             # نمایش آمار همه سیگنال‌ها
#             st.subheader("Statistics before and after interpolation")
#             st.dataframe(stats_df)

#             st.subheader(" Signal length information")
#             st.table(length_df)

#             st.subheader("Signals comparison chart")
#             fig = evaluator.plot_comparison(max_points=1000)
#             st.pyplot(fig)

#             # ذخیره در session_state
#             st.session_state["eval_stats"] = stats_df

#         except Exception as e:
#             st.error(f" Error in comparative analysis: {e}")



#------------------------------------
with tabs[3]:
    st.header("Step 4: Evaluation of Imputed Signals")

    if "raw_data" not in st.session_state or "imputed_data" not in st.session_state:
        st.warning(" First, perform the interpolation step..")
    else:
        try:
            evaluator = Evaluator(
                raw_data=st.session_state["raw_data"],
                imputed_data=st.session_state["imputed_data"],
                variable_names=st.session_state["variables"]
            )

            stats_df, length_df = evaluator.compute_stats(
                raw_length=st.session_state["raw_data"].shape[0]
            )
            stats_df["caseid"] = st.session_state["selected_case_interp"]


            st.subheader("Statistics before and after interpolation")
            st.dataframe(stats_df)

            st.subheader(" Signal length information")
            st.table(length_df)

            st.subheader("Signals comparison chart")
            fig = evaluator.plot_comparison(max_points=1000)
            st.pyplot(fig)

            # Save for final output
            st.session_state["eval_stats"] = stats_df

        except Exception as e:
            st.error(f" Error in comparative analysis: {e}")
            
        st.session_state["eval_stats"] = stats_df
        st.session_state["raw_data"] = st.session_state["raw_data"]  # Optional because it already exists but becomes clearer.
        st.session_state["imputed_data"] = st.session_state["imputed_data"]
        
        

#----------------------------------------------
with tabs[4]:
    st.header("Step 5: Export Final Results")

    if "eval_stats" not in st.session_state or "imputed_data" not in st.session_state:
        st.warning(" Complete the previous steps first..")
    else:
        # Display statistical information
        st.subheader(" Final Evaluation Statistics")
        st.dataframe(st.session_state["eval_stats"])

        #Creating a DataFrame from Interpolated Data
        df_imputed = pd.DataFrame(st.session_state["imputed_data"], columns=st.session_state["variables"])
        df_imputed["time"] = np.arange(len(df_imputed))
        st.subheader(" Interpolated Signal Data")
        st.dataframe(df_imputed.head(10))

        # Download files
        st.subheader(" Download Files")

        st.download_button(
            " Download Imputed Signal Data (CSV)",
            df_imputed.to_csv(index=False),
            file_name="imputed_signals.csv"
        )

        st.download_button(
            " Download Evaluation Statistics (CSV)",
            st.session_state["eval_stats"].to_csv(index=False),
            file_name="evaluation_statistics.csv"
        )

        st.success(" Output files are ready for download.")

#=====================

#==================================
with tabs[5]:
    st.header("Step 6: Analysis")

    # Check that the data exists.
    if "eval_stats" not in st.session_state:
        st.warning(" First complete the assessment step (tab 4).")
        st.stop()

    if "df_cases" not in st.session_state or "df_cases_filtered" not in st.session_state:
        st.warning(" There is no filtered information. Please run the first tab.")
        st.stop()

    #Loading dataا
    df_stats = st.session_state["eval_stats"]
    df_all = st.session_state["df_cases"]
    df_filtered = st.session_state["df_cases_filtered"]

    # create plotterر
    plotter = StatisticsPlotter()

    # Part One: Comparative statistical chart for mean, median, standard deviation
    st.subheader(" Statistical changes of signals before and after interpolation.")
    plotter.plot_case_summary(df_stats, max_cases=10, rel_change_threshold=0.0)

    # Part Two: Comparison of Numerical Features
    st.subheader(" Comparison of numerical characteristics of patients")
    numeric_cols = st.multiselect("Selecting numeric columns for comparison:", 
                                  df_all.select_dtypes(include=np.number).columns.tolist(),
                                  default=['age', 'bmi', 'weight', 'height'])
    if numeric_cols:
        plotter.compare_numerical(df_all, df_filtered, numeric_cols)

    # Part Three: Comparing Batch Features
    st.subheader(" Comparing patient group characteristics")
    categorical_cols = st.multiselect("Selecting categorical columns for comparison:",
                                      df_all.select_dtypes(include='object').columns.tolist(),
                                      default=['sex', 'ane_type', 'optype', 'department', 'position'])
    if categorical_cols:
        plotter.compare_categorical(df_all, df_filtered, categorical_cols)
#---------------------
with tabs[6]:
    st.header("Step 7: Correlation & Statistical Testing")

    if "raw_data" not in st.session_state or "imputed_data" not in st.session_state:
        st.warning("Please run interpolation first (Tab 3).")
        st.stop()

    st.subheader("Choose type of statistical test:")
    test_type = st.selectbox("Test type", [
        "Pearson correlation + T-Test (all cases)",
        "Pearson correlation + T-Test (one case)",
        "Lagged Correlation (all cases)",
        "Cross Correlation (one case)"
    ])

    signal_x = st.selectbox("Signal X", st.session_state["variables"])
    signal_y = st.selectbox("Signal Y", st.session_state["variables"])

    if test_type in ["Pearson correlation + T-Test (one case)", "Cross Correlation (one case)"]:
        selected_case = st.selectbox("Select Case ID", st.session_state["valid_ids"])

    if st.button("Run Test"):
        try:
            tester = StatisticalTester(
                raw_signals={cid: st.session_state["raw_data"] for cid in [selected_case]} if "one case" in test_type else {cid: st.session_state["raw_data"] for cid in st.session_state["valid_ids"]},
                imputed_signals={cid: st.session_state["imputed_data"] for cid in [selected_case]} if "one case" in test_type else {cid: st.session_state["imputed_data"] for cid in st.session_state["valid_ids"]},
                variable_names=st.session_state["variables"]
            )

            if "Pearson" in test_type:
                df_results = tester.run_tests()
                st.dataframe(df_results)
                st.success("Test completed.")
                st.session_state["corr_results"] = df_results

            elif "Lagged" in test_type:
                df_lag = tester.run_lagged_tests(signal_x=signal_x, signal_y=signal_y)
                st.dataframe(df_lag)
                tester.plot_lagged_summary(df_lag)

            elif "Cross" in test_type:
                df_cross = tester.run_cross_correlation_tests(signal_x=signal_x, signal_y=signal_y)
                st.dataframe(df_cross)
                tester.plot_cross_correlation(selected_case, signal_x, signal_y)

        except Exception as e:
            st.error(f"Error during test: {e}")

