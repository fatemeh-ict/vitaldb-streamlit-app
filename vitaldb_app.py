import streamlit as st
import pandas as pd
import numpy as np
import vitaldb
from io import BytesIO

# ------------------ UTILS -------------------
def load_data():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    df_labs = pd.read_csv("https://api.vitaldb.net/labs")
    return df_cases, df_trks, df_labs

# ------------------ PAGE -------------------
st.set_page_config(page_title="VitalDB Streamlit", layout="wide")
st.title("VitalDB Signal Preprocessing Tool")

# ------------------ LOAD DATA -------------------
@st.cache_data
def cached_data():
    return load_data()

df_cases, df_trks, df_labs = cached_data()

# ------------------ SESSION INIT -------------------
if "valid_ids" not in st.session_state:
    st.session_state.valid_ids = []
    st.session_state.df_cases_filtered = pd.DataFrame()
    st.session_state.df_trks_filtered = pd.DataFrame()
    st.session_state.df_labs_filtered = pd.DataFrame()

# ------------------ TAB 1: FILTER -------------------
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

        df_cases_filtered = df_cases[df_cases['caseid'].isin(valid_ids)].copy()
        df_trks_filtered = df_trks[df_trks['caseid'].isin(valid_ids)].copy()
        df_labs_filtered = df_labs[df_labs['caseid'].isin(valid_ids)].copy()

        st.session_state.valid_ids = valid_ids
        st.session_state.df_cases_filtered = df_cases_filtered
        st.session_state.df_trks_filtered = df_trks_filtered
        st.session_state.df_labs_filtered = df_labs_filtered

        st.success(f"✅ Filtered {len(valid_ids)} valid case IDs.")
        st.write("Filtered `df_cases` size:", df_cases_filtered.shape)
        st.write("Filtered `df_trks` size:", df_trks_filtered.shape)
        st.write("Filtered `df_labs` size:", df_labs_filtered.shape)

        # DOWNLOADS
        st.download_button("Download Filtered Cases CSV", df_cases_filtered.to_csv(index=False), "filtered_cases.csv")
        st.download_button("Download Filtered Trks CSV", df_trks_filtered.to_csv(index=False), "filtered_trks.csv")
        st.download_button("Download Filtered Labs CSV", df_labs_filtered.to_csv(index=False), "filtered_labs.csv")

# ------------------ TAB 2: ANALYZE -------------------

# Tab 2 - Signal Analysis
with st.expander("2. Signal Analysis", expanded=False):
    if "valid_ids" in st.session_state:
        selected_ids = st.multiselect("Select Case IDs to Analyze", st.session_state["valid_ids"], default=st.session_state["valid_ids"][:3])

        if st.button("Analyze Selected Signals"):
            import numpy as np
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            summaries = []

            for caseid in selected_ids:
                try:
                    signal_list = [
                              "BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP",
                               "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "Orchestra/RFTN50_RATE"
                               ]
                    data = vitaldb.load_case(caseid, signal_list, interval=1)
                    df = pd.DataFrame(data, columns=[
                        "BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP",
                        "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "Orchestra/RFTN50_RATE"
                    ])
                    df["time"] = np.arange(len(df))
                    
                    st.subheader(f"📊 Case ID: {caseid}")
                    fig = make_subplots(rows=6, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.03, subplot_titles=df.columns[:-1])

                    summary_case = []

                    for i, col in enumerate(df.columns[:-1]):
                        sig = df[col]
                        time = df["time"]
                        row = i + 1

                        nans = sig.isna()
                        outliers = (sig < 20) | (sig > 120)
                        jumps = np.abs(np.diff(sig.fillna(method='pad'))) > 30
                        jumps = np.insert(jumps, 0, False)

                        fig.add_trace(go.Scatter(x=time, y=sig, mode='lines+markers', name=col), row=row, col=1)
                        fig.add_trace(go.Scatter(x=time[nans], y=[sig.min() - 5]*nans.sum(),
                                                 mode='markers', name="NaNs", marker=dict(color='gray')), row=row, col=1)
                        fig.add_trace(go.Scatter(x=time[outliers], y=sig[outliers],
                                                 mode='markers', name="Outliers", marker=dict(color='purple')), row=row, col=1)
                        fig.add_trace(go.Scatter(x=time[jumps], y=sig[jumps],
                                                 mode='markers', name="Jumps", marker=dict(color='orange')), row=row, col=1)

                        summary_case.append({
                            "Signal": col,
                            "NaNs": int(nans.sum()),
                            "NaN %": round(100 * nans.sum() / len(sig), 2),
                            "Outliers": int(outliers.sum()),
                            "Outlier %": round(100 * outliers.sum() / len(sig), 2),
                            "Jumps": int(jumps.sum()),
                            "Jump %": round(100 * jumps.sum() / len(sig), 2)
                        })

                    fig.update_layout(height=300 * 6, title=f"Signal Analysis - Case {caseid}")
                    st.plotly_chart(fig)

                    st.dataframe(pd.DataFrame(summary_case))
                    summaries.extend(summary_case)

                except Exception as e:
                    st.error(f"❌ Error processing case {caseid}: {e}")

        else:
            st.info("Select case IDs and press 'Analyze Selected Signals'")
    else:
        st.warning("⚠️ Please apply filters first in Tab 1.")

