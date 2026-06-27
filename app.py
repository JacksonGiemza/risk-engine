import streamlit as st
import pandas as pd

from src.pipeline import RiskPipeline


st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    layout="wide"
)

st.title("Portfolio Risk Dashboard")
st.caption("Historical, Parametric, and Monte Carlo VaR / Expected Shortfall")


st.sidebar.header("Configuration")

portfolio_path = st.sidebar.text_input(
    "Portfolio CSV path",
    value=r"data\raw\portfolio\portfolio.csv"
)

end_date = st.sidebar.text_input("End date", value="2026-06-17")

lookback_days = st.sidebar.number_input(
    "Lookback days",
    min_value=30,
    max_value=2000,
    value=365,
    step=30
)

confidence_level = st.sidebar.selectbox(
    "Confidence level",
    options=[0.90, 0.95, 0.99],
    index=2
)

num_simulations = st.sidebar.number_input(
    "Monte Carlo simulations",
    min_value=1000,
    max_value=100000,
    value=10000,
    step=1000
)

num_worst_days = st.sidebar.number_input(
    "Worst days",
    min_value=5,
    max_value=50,
    value=10,
    step=5
)

run_button = st.sidebar.button("Run Risk Analysis")


@st.cache_data
def run_pipeline(config):
    pipeline = RiskPipeline(config)
    return pipeline.run()


if run_button:
    config = {
        "portfolio_path": portfolio_path,
        "start_date": None,
        "end_date": end_date,
        "lookback_days": lookback_days,
        "confidence_level": confidence_level,
        "num_simulations": num_simulations,
        "random_seed": 42,
        "num_worst_days": num_worst_days,
    }

    with st.spinner("Running risk analysis..."):
        report = run_pipeline(config)

    st.subheader("Portfolio Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Gross Exposure", f"${report.portfolio_summary.gross_exposure:,.0f}")
    col2.metric("Net Exposure", f"${report.portfolio_summary.net_exposure:,.0f}")
    col3.metric("Long Exposure", f"${report.portfolio_summary.long_exposure:,.0f}")
    col4.metric("Short Exposure", f"${report.portfolio_summary.short_exposure:,.0f}")

    st.subheader("Risk Summary")

    risk_table = report.risk_table.copy()

    risk_table["VaR Return"] = risk_table["VaR Return"].apply(lambda x: f"{x:.2%}")
    risk_table["ES Return"] = risk_table["ES Return"].apply(lambda x: f"{x:.2%}")
    risk_table["VaR $"] = risk_table["VaR $"].apply(lambda x: f"${x:,.0f}")
    risk_table["ES $"] = risk_table["ES $"].apply(lambda x: f"${x:,.0f}")

    st.dataframe(risk_table, use_container_width=True, hide_index=True)

    st.subheader("Portfolio Returns")

    portfolio_returns = report.portfolio_returns.dropna()
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1

    st.line_chart(cumulative_returns)

    st.subheader("Holdings")

    holdings = report.holdings.copy()

    display_cols = [
        "symbol",
        "quantity",
        "latest_price",
        "market_value",
        "abs_exposure",
        "weight",
        "side",
    ]

    holdings = holdings[[col for col in display_cols if col in holdings.columns]]

    if "weight" in holdings.columns:
        holdings["weight"] = holdings["weight"].apply(lambda x: f"{x:.2%}")

    for col in ["latest_price", "market_value", "abs_exposure"]:
        if col in holdings.columns:
            holdings[col] = holdings[col].apply(lambda x: f"${x:,.2f}")

    st.dataframe(holdings, use_container_width=True, hide_index=True)

    st.subheader("Worst Trading Days")

    worst_days_display = report.worst_days.reset_index()

    if "Return" in worst_days_display.columns:
        worst_days_display["Return"] = worst_days_display["Return"].apply(lambda x: f"{x:.2%}")

    if "Dollar Loss" in worst_days_display.columns:
        worst_days_display["Dollar Loss"] = worst_days_display["Dollar Loss"].apply(lambda x: f"${x:,.0f}")

    st.dataframe(worst_days_display, use_container_width=True, hide_index=True)

else:
    st.info("Configure the sidebar and click **Run Risk Analysis**.")