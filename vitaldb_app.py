import streamlit as st
import pandas as pd
import numpy as np
from vitaldb import load_case
from analyzer import SignalAnalyzer, SignalProcessor, Evaluator
from selector import CaseSelector

st.set_page_config(layout="wide", page_title="VitalDB Signal Analyzer")
st.title("📊 VitalDB Signal Analyzer with Plotly")

# ---- 1. Load Metadata
@st.cache_data
def load_metadata():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    return df_cases, df_trks

df_cases, df_trks = load_metadata()
st.success("✅ Metadata loaded from VitalDB.")

# ---- 2. Sidebar Configuration
st.sidebar.header("⚙️ Filter Settings")

signal_options = sorted(df_trks['tname'].value_counts().index.tolist())
selected_signals = st.sidebar.multiselect(
    "Select Signal Variables", signal_options[:200],
    default=["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE"]
)

drug_columns = ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"]
excluded_drugs = st.sidebar.multiselect("Exclude Cases with These Drugs", drug_columns)

ane_type = st.sidebar.selectbox("Anesthesia Type", ["General", "Regional", "MAC"], index=0)

if st.sidebar.button("🚦 Filter Valid Cases"):
    selector = CaseSelector(df_cases, df_trks,
                            ane_type=ane_type,
                            required_variables=selected_signals,
                            intraoperative_boluses=excluded_drugs)
    valid_ids = selector.select_valid_cases()
    df_valid = df_cases[df_cases['caseid'].isin(valid_ids)]
    st.session_state.valid_ids = valid_ids
    st.success(f"✅ Found {len(valid_ids)} valid cases.")
    st.dataframe(df_valid[["caseid", "subjectid", "age", "sex", "bmi"]].head())

# ---- 3. Select Case IDs to Process
if "valid_ids" in st.session_state:
    selected_ids = st.multiselect("🔎 Select Case IDs to Analyze", st.session_state.valid_ids[:50])
    plot_ids = st.multiselect("📈 Select Case IDs to Plot", selected_ids)

    if st.button("🚀 Run Full Analysis"):
        for cid in selected_ids:
            try:
                st.subheader(f"Case {cid}")
                data = load_case(cid, selected_signals, interval=1)

                # Step 1: Compute global medians and MADs
                global_medians = {var: np.nanmedian(data[:, i]) for i, var in enumerate(selected_signals)}
                global_mads = {
                    var: np.nanmedian(np.abs(data[:, i] - global_medians[var])) or 1e-6
                    for i, var in enumerate(selected_signals)
                }

                # Step 2: Analyze
                analyzer = SignalAnalyzer(
                    caseid=cid,
                    data=data,
                    variable_names=selected_signals,
                    global_medians=global_medians,
                    global_mads=global_mads,
                    plot=False
                )
                analyzer.analyze()

                # Step 3: Process
                processor = SignalProcessor(
                    data=analyzer.data,
                    issues=analyzer.issues,
                    variable_names=selected_signals,
                    gap_strategy='interpolate_short',
                    long_gap_strategy='nan',
                    interp_method='auto'
                )
                clean_data = processor.process()

                # Step 4: Evaluate
                evaluator = Evaluator(raw_data=data, imputed_data=clean_data, variable_names=selected_signals)
                stats_df, length_df = evaluator.compute_stats(raw_length=data.shape[0])
                st.write("📊 Summary:")
                st.dataframe(stats_df)

                # Step 5: Plot if selected
                if cid in plot_ids:
                    fig = analyzer.plot(signal_units={
                        "BIS/BIS": "%",
                        "Solar8000/NIBP_SBP": "mmHg",
                        "Solar8000/NIBP_DBP": "mmHg",
                        "Orchestra/PPF20_RATE": "bpm"
                    })
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error in case {cid}: {e}")
