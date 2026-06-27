import pandas as pd
import streamlit as st
import altair as alt


def render_cumulative_returns(report) -> None:
    st.subheader("Portfolio Performance")

    portfolio_returns = report.portfolio_returns.dropna()

    if portfolio_returns.empty:
        st.warning("No portfolio returns available.")
        return

    cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    chart_df = cumulative_returns.reset_index()
    chart_df.columns = ["Date", "Cumulative Return"]

    st.line_chart(
        chart_df,
        x="Date",
        y="Cumulative Return",
        use_container_width=True,
    )


def render_return_distribution(report) -> None:
    st.subheader("Return Distribution")

    portfolio_returns = report.portfolio_returns.dropna()

    if portfolio_returns.empty:
        st.warning("No portfolio returns available.")
        return

    returns_df = pd.DataFrame({
        "Portfolio Return": portfolio_returns
    })

    base = alt.Chart(returns_df).mark_bar().encode(
        alt.X("Portfolio Return:Q", bin=alt.Bin(maxbins=50), title="Daily portfolio return"),
        alt.Y("count():Q", title="Frequency"),
    )

    var_lines = pd.DataFrame([
        {"Method": "Historical VaR", "Return": report.historical.var_return},
        {"Method": "Parametric VaR", "Return": report.parametric.var_return},
        {"Method": "Monte Carlo VaR", "Return": report.monte_carlo.var_return},
    ])

    lines = alt.Chart(var_lines).mark_rule(strokeWidth=2).encode(
        x=alt.X("Return:Q"),
        color=alt.Color("Method:N"),
        tooltip=["Method", alt.Tooltip("Return:Q", format=".2%")],
    )

    chart = (base + lines).properties(height=350)

    st.altair_chart(chart, use_container_width=True)


def render_var_comparison(report) -> None:
    st.subheader("VaR / ES Comparison")

    comparison_df = pd.DataFrame([
        {
            "Method": report.historical.method,
            "VaR": report.historical.var_dollars,
            "ES": report.historical.es_dollars,
        },
        {
            "Method": report.parametric.method,
            "VaR": report.parametric.var_dollars,
            "ES": report.parametric.es_dollars,
        },
        {
            "Method": report.monte_carlo.method,
            "VaR": report.monte_carlo.var_dollars,
            "ES": report.monte_carlo.es_dollars,
        },
    ])

    chart_df = comparison_df.melt(
        id_vars="Method",
        value_vars=["VaR", "ES"],
        var_name="Metric",
        value_name="Dollars",
    )

    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Method:N", title="Method"),
        y=alt.Y("Dollars:Q", title="Dollars"),
        color=alt.Color("Metric:N"),
        tooltip=[
            "Method",
            "Metric",
            alt.Tooltip("Dollars:Q", format="$,.0f"),
        ],
    ).properties(height=350)

    st.altair_chart(chart, use_container_width=True)