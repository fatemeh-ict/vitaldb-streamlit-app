# vitaldb_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import vitaldb

st.set_page_config(page_title="VitalDB Dual Group Analyzer", layout="wide")
st.title("🧠 VitalDB Signal Pipeline Analyzer")

@st.cache_data(show_spinner=False)
def load_data():
    return (
        pd.read_csv("https://api.vitaldb.net/cases"),
        pd.read_csv("https://api.vitaldb.net/trks")
    )

df_cases, df_trks = load_data()

# --- Settings ---
group1 = ["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"]
group2 = ["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP", "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"]
all_signals = sorted(list(set(group1 + group2)))
drug_cols = ["intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe"]

# --- Utility ---
def filter_cases(required_signals, ane_types, exclude_drugs):
    ids = set(df_trks['caseid'].unique())
    for sig in required_signals:
        ids &= set(df_trks[df_trks['tname'] == sig]['caseid'])
    df = df_cases[df_cases['caseid'].isin(ids)]
    if ane_types:
        df = df[df['ane_type'].isin(ane_types)]
    if exclude_drugs:
        df = df[df[exclude_drugs].sum(axis=1) == 0]
    return df

# --- Tabs ---
tabs = st.tabs(["🧬 Case Selection", "📊 Signal Quality", "🛠 Interpolation", "📈 Dataset Stats"])

# --- Tab 1 ---
with tabs[0]:
    st.header("Step 1: Filter and Select Case IDs")
    selected_signals = st.multiselect("Select signals (from both groups):", all_signals, default=all_signals)
    selected_ane_types = st.multiselect("Filter by anesthesia type:", sorted(df_cases['ane_type'].dropna().unique()))
    selected_drugs = st.multiselect("Exclude if drugs used:", drug_cols)
    limit = st.radio("Number of cases:", [10, "All"], horizontal=True)

    ids1 = set(df_trks[df_trks['tname'].isin(group1)]['caseid'])
    ids2 = set(df_trks[df_trks['tname'].isin(group2)]['caseid'])
    valid_ids = ids1.union(ids2)

    df_valid = filter_cases(selected_signals, selected_ane_types, selected_drugs)
    df_valid = df_valid[df_valid['caseid'].isin(valid_ids)]
    final_ids = df_valid['caseid'].tolist()
    if limit == 10:
        final_ids = final_ids[:10]
    st.success(f"✅ Using {len(final_ids)} case IDs.")

# --- Tab 2 ---
with tabs[1]:
    st.header("Step 2: Signal Quality Analysis")
    if not final_ids:
        st.warning("No valid case IDs found.")
        st.stop()
    data_list = [vitaldb.load_case(cid, selected_signals, interval=1) for cid in final_ids]
    min_len = min(len(d) for d in data_list)
    merged_data = np.concatenate([d[:min_len] for d in data_list], axis=0)
    medians = {sig: np.nanmedian(merged_data[:, i]) for i, sig in enumerate(selected_signals)}
    mads = {sig: np.median(np.abs(merged_data[:, i] - medians[sig])) or 1e-6 for i, sig in enumerate(selected_signals)}

    st.subheader("⚠️ Quality Summary")
    table = []
    for i, sig in enumerate(selected_signals):
        x = merged_data[:, i]
        nan = int(np.isnan(x).sum())
        jump = int((np.abs(np.diff(x)) > 3.5 * mads[sig]).sum())
        outlier = int((np.abs(x - medians[sig]) > 3.5 * mads[sig]).sum())
        table.append([sig, nan, jump, outlier])
    df_qc = pd.DataFrame(table, columns=["Signal", "NaNs", "Jumps", "Outliers"])
    st.dataframe(df_qc, use_container_width=True)

    st.subheader("📈 Plot a Case")
    show_id = st.selectbox("Select Case ID:", final_ids)
    data = vitaldb.load_case(show_id, selected_signals, interval=1)
    for i, sig in enumerate(selected_signals):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(data[:, i], label=sig)
        ax.set_title(f"Raw Signal - {sig}")
        ax.legend()
        st.pyplot(fig)

# --- Tab 3 ---
with tabs[2]:
    st.header("Step 3: Interpolation & Comparison")
    interp_data = merged_data.copy()
    for i in range(interp_data.shape[1]):
        sig = interp_data[:, i]
        if np.isnan(sig).sum():
            valid = ~np.isnan(sig)
            if valid.sum() > 1:
                f = interp1d(np.where(valid)[0], sig[valid], kind='linear', fill_value='extrapolate')
                interp_data[:, i] = f(np.arange(len(sig)))

    st.subheader("📊 Before vs After Stats")
    stats = []
    for i, sig in enumerate(selected_signals):
        stats.append({
            "Signal": sig,
            "Mean Before": np.nanmean(merged_data[:, i]),
            "Mean After": np.mean(interp_data[:, i]),
            "NaNs Before": int(np.isnan(merged_data[:, i]).sum()),
            "NaNs After": int(np.isnan(interp_data[:, i]).sum())
        })
    st.dataframe(pd.DataFrame(stats), use_container_width=True)

    selected_signal = st.selectbox("Visualize interpolation for:", selected_signals)
    idx = selected_signals.index(selected_signal)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(merged_data[:, idx], label='Raw', alpha=0.6)
    ax.plot(interp_data[:, idx], label='Interpolated', linestyle='--')
    ax.set_title(f"{selected_signal} - Before vs After Interpolation")
    ax.legend()
    st.pyplot(fig)

# --- Tab 4 ---
with tabs[3]:
    st.header("Step 4: Dataset Summary")
    st.subheader("📌 Numerical Description")
    st.dataframe(df_valid.select_dtypes(include=np.number).describe().T)

    st.subheader("📌 Categorical Counts")
    for cat in ['sex', 'ane_type', 'optype', 'department', 'position']:
        if cat in df_valid.columns:
            st.markdown(f"**{cat}**")
            st.dataframe(df_valid[cat].value_counts())
