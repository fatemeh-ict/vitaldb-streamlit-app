
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

    if st.button("📥 Load and Filter Cases"):
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
        st.session_state["variables"] = variables

        st.success(f"{len(valid_ids)} valid case(s) found.")
        st.dataframe(df_cases_filtered.head(10))

        st.download_button("⬇️ Download Filtered Cases",
                           df_cases_filtered.to_csv(index=False),
                           file_name="filtered_cases.csv")

        st.download_button("⬇️ Download Filtered Tracks",
                           df_trks_filtered.to_csv(index=False),
                           file_name="filtered_trks.csv")

        st.download_button("⬇️ Download Filtered Labs",
                           df_labs_filtered.to_csv(index=False),
                           file_name="filtered_labs.csv")
