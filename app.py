import streamlit as st

from src.pipeline import RiskPipeline

from ui.sidebar import render_sidebar
from ui.metrics import render_portfolio_metrics, render_risk_metrics
from ui.charts import (
    render_cumulative_returns,
    render_return_distribution,
    render_var_comparison,
)
from ui.tables import render_holdings, render_worst_days
from ui.backtesting import render_backtest_summary, render_breach_chart


st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    layout="wide",
)

st.title("Portfolio Risk Dashboard")
st.caption("Historical, Parametric, and Monte Carlo VaR / Expected Shortfall")


@st.cache_data
def run_pipeline(config):
    pipeline = RiskPipeline(config)
    return pipeline.run()


@st.cache_data
def run_backtest(config):
    pipeline = RiskPipeline(config)
    pipeline.run()
    return pipeline.run_backtest()


config, run_analysis = render_sidebar()

if run_analysis:
    try:
        with st.spinner("Running risk analysis..."):
            report = run_pipeline(config)
            backtest_report = run_backtest(config)

        render_portfolio_metrics(report)
        render_risk_metrics(report)

        render_var_comparison(report)
        render_cumulative_returns(report)
        render_return_distribution(report)

        render_backtest_summary(backtest_report)
        render_breach_chart(backtest_report)

        render_holdings(report)
        render_worst_days(report)

    except Exception as e:
        st.error(f"Risk analysis failed: {e}")

else:
    st.info("Configure the sidebar and click **Run Risk Analysis**.")