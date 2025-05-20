

# ==========================
# Tab 2: Signal Quality Analysis (Accurate with Counts)
# ==========================
with tabs[1]:
    st.header("Step 2: Signal Quality Analysis (Accurate + Counts)")

    variables = [
        "BIS/BIS",
        "Solar8000/NIBP_DBP",
        "Solar8000/NIBP_SBP",
        "Orchestra/RFTN50_RATE",
        "Orchestra/RFTN20_RATE",
        "Orchestra/PPF20_RATE"
    ]

    if 'case_ids' in st.session_state:
        selected_case = st.selectbox("Select a case to analyze", st.session_state['case_ids'], key="select_case_tab2")
        data = vitaldb.load_case(selected_case, variables, interval=1)

        # Compute global medians and MADs
        global_medians = {}
        global_mads = {}
        for i, var in enumerate(variables):
            sig = data[:, i]
            sig = sig[~np.isnan(sig)]
            global_medians[var] = np.median(sig)
            global_mads[var] = np.median(np.abs(sig - global_medians[var])) or 1e-6

        analyzer = SignalAnalyzer(caseid=selected_case, data=data, variable_names=variables,
                                  global_medians=global_medians, global_mads=global_mads)
        results = analyzer.analyze()
        df = pd.DataFrame(data, columns=variables)
        df["time"] = np.arange(len(df))

        fig = make_subplots(rows=len(variables), cols=1, shared_xaxes=True, vertical_spacing=0.03,
                            subplot_titles=variables)

        for i, var in enumerate(variables):
            row = i + 1
            signal = df[var]
            time = df["time"]

            # Display numeric counts
            st.subheader(f"🔍 {var}")
            st.markdown(
                f"**NaNs:** {len(results[var]['nan'])} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Gaps:** {len(results[var]['gap'])} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Outliers:** {len(results[var]['outlier'])} &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"**Jumps:** {len(results[var]['jump'])}"
            )

            # Plot signal line
            fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name=var), row=row, col=1)

            # Plot NaNs
            if results[var]['nan']:
                fig.add_trace(go.Scatter(x=time[results[var]['nan']],
                                         y=[signal.min() - 5] * len(results[var]['nan']),
                                         mode='markers', marker=dict(color='gray', symbol='line-ns-open', size=6),
                                         name='NaNs', showlegend=(i == 0)), row=row, col=1)

            # Plot Outliers
            if results[var]['outlier']:
                fig.add_trace(go.Scatter(x=time[results[var]['outlier']],
                                         y=signal[results[var]['outlier']],
                                         mode='markers', marker=dict(color='purple', symbol='star', size=7),
                                         name='Outliers', showlegend=(i == 0)), row=row, col=1)

            # Plot Jumps
            if results[var]['jump']:
                fig.add_trace(go.Scatter(x=time[results[var]['jump']],
                                         y=signal[results[var]['jump']],
                                         mode='markers', marker=dict(color='orange', symbol='x', size=7),
                                         name='Jumps', showlegend=(i == 0)), row=row, col=1)

            # Plot Gaps
            for gap in results[var]['gap']:
                start = gap['start']
                end = gap['start'] + gap['length']
                fig.add_shape(type="rect",
                              x0=time[start], x1=time[end - 1],
                              y0=signal.min(), y1=signal.max(),
                              fillcolor="red", opacity=0.15,
                              line=dict(width=0), row=row, col=1)

        fig.update_layout(height=280 * len(variables), title=f"Signal Quality - Case {selected_case}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please select cases in Tab 1 first.")
