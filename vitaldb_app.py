import streamlit as st
import pandas as pd
import numpy as np
import vitaldb
from modules import *  # تمام کلاس‌های شما باید در فایل‌های جداگانه داخل پوشه modules قرار بگیرند

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
