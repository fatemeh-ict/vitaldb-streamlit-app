import streamlit as st
import pandas as pd
import numpy as np
from vitaldb import load_case
import plotly.graph_objects as go
from src.analyzer import SignalAnalyzer, SignalProcessor, Evaluator
from src.selector import CaseSelector

st.set_page_config(layout="wide", page_title="VitalDB Analyzer")
st.title("🧠 VitalDB Signal Quality & Preprocessing Dashboard")

# ---- Section 1: Load Metadata from VitalDB
@st.cache_data
def load_metadata():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    return df_cases, df_trks

df_cases, df_trks = load_metadata()
st.success("✅ Metadata loaded successfully.")

# ---- Section 2: User selects signals and filtering
st.sidebar.header("🔧 Configuration")

# Signal choices
all_signals = sorted(df_trks['tname'].value_counts().index.to_list())
selected_signals = st.sidebar.multiselect("Select signal variables", all_signals[:200], default=[
    "BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP",
    "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"
])

# Anesthesia type
ane_type = st.sidebar.selectbox("Anesthesia type", ["General", "Regional", "MAC"], index=0)

# Drug filter
drug_columns = ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"]
selected_drugs = st.sidebar.multiselect("Exclude cases with these drugs", drug_columns)

# Run filter
if st.sidebar.button("🚦 Filter Valid Cases"):
    selector = CaseSelector(
        df_cases=df_cases,
        df_trks=df_trks,
        ane_type=ane_type,
        required_variables=selected_signals,
        intraoperative_boluses=selected_drugs
    )
    valid_ids = selector.select_valid_cases()
    df_valid = df_cases[df_cases['caseid'].isin(valid_ids)].copy()
    st.session_state.valid_cases = valid_ids
    st.success(f"✅ Found {len(valid_ids)} valid cases.")
    st.dataframe(df_valid.head())

# ---- Section 3: Case selection
if "valid_cases" in st.session_state:
    selected_case = st.selectbox("🩺 Select a case for analysis", st.session_state.valid_cases)

    if st.button("📈 Run Signal Analysis"):
        try:
            data = load_case(selected_case, selected_signals, interval=1)
            st.info(f"Case {selected_case} loaded.")

            # Compute global stats
            global_medians = {
                var: np.nanmedian(data[:, i]) for i, var in enumerate(selected_signals)
            }
            global_mads = {
                var: np.nanmedian(np.abs(data[:, i] - global_medians[var])) or 1e-6
                for i, var in enumerate(selected_signals)
            }

            # Analyzer
            analyzer = SignalAnalyzer(
                caseid=selected_case,
                data=data,
                variable_names=selected_signals,
                global_medians=global_medians,
                global_mads=global_mads,
                plot=False
            )
            analyzer.analyze()
            st.success("✅ Signal analysis completed.")

            # Visualization
            fig = analyzer.plot()
            st.plotly_chart(fig, use_container_width=True)

            # Processor
            processor = SignalProcessor(
                data=analyzer.data,
                issues=analyzer.issues,
                variable_names=selected_signals
            )
            cleaned_data = processor.process()

            # Evaluator
            evaluator = Evaluator(
                raw_data=data,
                imputed_data=cleaned_data,
                variable_names=selected_signals
            )
            stats_df, length_df = evaluator.compute_stats(raw_length=data.shape[0])

            st.subheader("📊 Signal Summary Statistics")
            st.dataframe(stats_df)

            st.subheader("📥 Download Summary")
            csv = stats_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", data=csv, file_name=f"case_{selected_case}_summary.csv")

        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
