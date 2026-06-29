import pandas as pd
import streamlit as st
import altair as alt


def render_backtest_summary(backtest_report) -> None:
    st.subheader("VaR Backtesting")

    first_method = next(iter(backtest_report.breach_series))
    coverage_window = len(backtest_report.breach_series[first_method])
    traffic_window = backtest_report.summary[first_method].traffic_light.total_days

    col1, col2 = st.columns(2)
    col1.metric("Coverage Test Window", f"{coverage_window:,} days")
    col2.metric("Traffic Light Window", f"{traffic_window:,} days")

    rows = []

    for method, result in backtest_report.summary.items():
        rows.append({
            "Method": method.title(),
            "Observations": result.observations,
            "Violations": result.violations,
            "Expected Violations": result.expected_violations,
            "Binomial p-value": result.binomial.p_value,
            "Kupiec LR": result.kupiec.lr_statistic,
            "Kupiec p-value": result.kupiec.p_value,
            "Christoffersen UC p-value": result.christoffersen.unconditional.p_value,
            "Christoffersen IND p-value": result.christoffersen.independence.p_value,
            "Christoffersen CC p-value": result.christoffersen.conditional.p_value,
            "Traffic Light": result.traffic_light.zone,
            "Traffic Light Violations": result.traffic_light.observed_violations,
        })

    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def render_breach_chart(backtest_report) -> None:
    st.subheader("Backtest Breaches")

    method = st.selectbox(
        "Backtest method",
        options=list(backtest_report.breach_series.keys()),
        format_func=lambda x: x.title(),
    )

    breach_df = backtest_report.breach_series[method].copy().reset_index()

    if "Date" not in breach_df.columns:
        breach_df = breach_df.rename(columns={breach_df.columns[0]: "Date"})

    breach_df["breach"] = breach_df["breach"].astype(bool)

    returns_chart = alt.Chart(breach_df).mark_line().encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("actual_return:Q", title="Portfolio return"),
        tooltip=[
            alt.Tooltip("Date:T"),
            alt.Tooltip("actual_return:Q", format=".2%"),
            alt.Tooltip("var_return:Q", format=".2%"),
            alt.Tooltip("breach:N"),
        ],
    )

    var_chart = alt.Chart(breach_df).mark_line(strokeDash=[6, 4]).encode(
        x="Date:T",
        y="var_return:Q",
        tooltip=[
            alt.Tooltip("Date:T"),
            alt.Tooltip("var_return:Q", format=".2%"),
        ],
    )

    breaches = alt.Chart(breach_df[breach_df["breach"]]).mark_point(
        size=70,
        filled=True,
    ).encode(
        x="Date:T",
        y="actual_return:Q",
        tooltip=[
            alt.Tooltip("Date:T"),
            alt.Tooltip("actual_return:Q", format=".2%"),
            alt.Tooltip("var_return:Q", format=".2%"),
        ],
    )

    chart = (returns_chart + var_chart + breaches).properties(height=350)

    st.altair_chart(chart, use_container_width=True)

    st.caption("Solid line = realized return, dashed line = VaR threshold, points = VaR breaches.")
    