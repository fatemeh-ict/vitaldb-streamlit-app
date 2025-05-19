import streamlit as st
import pandas as pd

# ✅ تعریف کلاس CaseSelector به‌صورت مستقیم (به‌جای import کردن)
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

# 📥 بارگذاری داده‌ها از API
@st.cache_data
def load_data():
    df_cases = pd.read_csv("https://api.vitaldb.net/cases")
    df_trks = pd.read_csv("https://api.vitaldb.net/trks")
    df_labs = pd.read_csv("https://api.vitaldb.net/labs")
    return df_cases, df_trks, df_labs

df_cases, df_trks, df_labs = load_data()

# 🧪 عنوان اپلیکیشن
st.title("VitalDB Signal Preprocessing Tool")

# 📌 مرحله اول: فیلتر کردن داده‌ها
with st.expander("1. Data Filtering", expanded=True):
    ane_type = st.selectbox("Select Anesthesia Type:", df_cases["ane_type"].dropna().unique())

    intraoperative_boluses = [
        "intraop_mdz", "intraop_ftn", "intraop_epi", "intraop_phe", "intraop_eph"
    ]
    removed_boluses = st.multiselect("Remove cases with these drugs:", options=intraoperative_boluses, default=intraoperative_boluses)

    required_variables_1 = [
        "Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
        "Orchestra/PPF20_CE", "Orchestra/RFTN20_CE",
        "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE"
    ]

    required_variables_2 = [
        "Solar8000/NIBP_DBP", "Solar8000/NIBP_SBP", "BIS/BIS",
        "Orchestra/PPF20_CE", "Orchestra/RFTN50_CE",
        "Orchestra/PPF20_RATE", "Orchestra/RFTN50_RATE"
    ]

    if st.button("Apply Filters"):
        selector1 = CaseSelector(df_cases, df_trks, ane_type=ane_type, required_variables=required_variables_1, intraoperative_boluses=removed_boluses)
        valid_ids_1 = set(selector1.select_valid_cases())

        selector2 = CaseSelector(df_cases, df_trks, ane_type=ane_type, required_variables=required_variables_2, intraoperative_boluses=removed_boluses)
        valid_ids_2 = set(selector2.select_valid_cases())

        valid_ids = sorted(list(valid_ids_1.union(valid_ids_2)))

        df_cases_filtered = df_cases[df_cases['caseid'].isin(valid_ids)].copy()
        df_trks_filtered = df_trks[df_trks['caseid'].isin(valid_ids)].copy()
        df_labs_filtered = df_labs[df_labs['caseid'].isin(valid_ids)].copy()

        # ذخیره‌سازی در session_state برای مراحل بعد
        st.session_state["valid_ids"] = valid_ids
        st.session_state["df_cases_filtered"] = df_cases_filtered
        st.session_state["df_trks_filtered"] = df_trks_filtered
        st.session_state["df_labs_filtered"] = df_labs_filtered

        st.success(f"✅ Filtered {len(valid_ids)} valid case IDs.")

        st.write("Filtered `df_cases` size:", df_cases_filtered.shape)
        st.write("Filtered `df_trks` size:", df_trks_filtered.shape)
        st.write("Filtered `df_labs` size:", df_labs_filtered.shape)

        st.download_button("Download Filtered Cases CSV", df_cases_filtered.to_csv(index=False), "filtered_cases.csv")
        st.download_button("Download Filtered Trks CSV", df_trks_filtered.to_csv(index=False), "filtered_trks.csv")
        st.download_button("Download Filtered Labs CSV", df_labs_filtered.to_csv(index=False), "filtered_labs.csv")

##############################################################################################################
# ✅ مرحله دوم: آنالیز سیگنال‌ها در تب جداگانه
import numpy as np

with st.expander("2. Signal Analysis", expanded=False):
    if "df_cases_filtered" not in st.session_state or "valid_ids" not in st.session_state:
        st.warning("⚠️ Please apply filters in Tab 1 first.")
    else:
        # انتخاب تعداد کیس برای بررسی
        num_cases = st.slider("Select number of cases to analyze:", min_value=1, max_value=len(st.session_state["valid_ids"]), value=5)
        selected_ids = st.session_state["valid_ids"][:num_cases]

        # تعریف متغیرها
        variables = ["BIS/BIS", "Solar8000/NIBP_SBP", "Solar8000/NIBP_DBP",
                     "Orchestra/PPF20_RATE", "Orchestra/RFTN20_RATE", "Orchestra/RFTN50_RATE"]

        # محاسبه آمار سراسری (global medians and mads)
        st.info("🔍 Computing global medians and MADs...")
        all_data_list = []
        for cid in selected_ids:
            try:
                data = vitaldb.load_case(cid, variables, interval=1)
                all_data_list.append(data)
            except:
                continue

        min_len = min(d.shape[0] for d in all_data_list)
        trimmed_data = np.concatenate([d[:min_len, :] for d in all_data_list], axis=0)
        global_medians = {}
        global_mads = {}
        for i, var in enumerate(variables):
            sig = trimmed_data[:, i]
            sig = sig[~np.isnan(sig)]
            global_medians[var] = np.median(sig)
            global_mads[var] = np.median(np.abs(sig - global_medians[var])) or 1e-6

        st.success("✅ Global statistics computed.")

        # اجرای تحلیل برای هر کیس و ترسیم نمودار
        for cid in selected_ids:
            st.subheader(f"Case ID: {cid}")
            try:
                data = vitaldb.load_case(cid, variables, interval=1)
                analyzer = SignalAnalyzer(
                    caseid=cid,
                    data=data,
                    variable_names=variables,
                    global_medians=global_medians,
                    global_mads=global_mads,
                    plot=True
                )
                analyzer.analyze()
                analyzer.plot()
            except Exception as e:
                st.error(f"❌ Error analyzing case {cid}: {e}")

