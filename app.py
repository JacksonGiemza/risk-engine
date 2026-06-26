import streamlit as st
import pandas as pd

from src.pipeline import RiskPipeline

st.set_page_config(
    page_title="Portfolio Risk Dashboard",
    layout="wide"
)

st.title("Portfolio Risk Dashboard")
st.caption("Historical, Parametric, and Monte Carlo VaR / Expected Shortfall")


# --------------------
# Sidebar controls
# --------------------
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


# --------------------
# Pipeline runner
# --------------------
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
        results = run_pipeline(config)

    portfolio_summary = results["portfolio_summary"]
    historical = results["historical"]
    parametric = results["parametric"]
    monte_carlo = results["monte_carlo"]
    worst_days = results["worst_days"]

    # --------------------
    # Portfolio summary
    # --------------------
    st.subheader("Portfolio Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Gross Exposure", f"${portfolio_summary['gross_exposure']:,.0f}")
    col2.metric("Net Exposure", f"${portfolio_summary['net_exposure']:,.0f}")
    col3.metric("Long Exposure", f"${portfolio_summary['long_exposure']:,.0f}")
    col4.metric("Short Exposure", f"${portfolio_summary['short_exposure']:,.0f}")

    # --------------------
    # Risk summary
    # --------------------
    st.subheader("Risk Summary")

    risk_table = pd.DataFrame([
        {
            "Method": "Historical",
            "VaR Return": historical["var_return"],
            "VaR $": historical["var_dollars"],
            "ES Return": historical["es_return"],
            "ES $": historical["es_dollars"],
        },
        {
            "Method": "Parametric",
            "VaR Return": parametric["var_return"],
            "VaR $": parametric["var_dollars"],
            "ES Return": parametric["es_return"],
            "ES $": parametric["es_dollars"],
        },
        {
            "Method": "Monte Carlo",
            "VaR Return": monte_carlo["var_return"],
            "VaR $": monte_carlo["var_dollars"],
            "ES Return": monte_carlo["es_return"],
            "ES $": monte_carlo["es_dollars"],
        },
    ])

    st.dataframe(risk_table, use_container_width=True)

    # --------------------
    # Worst days
    # --------------------
    st.subheader("Worst Trading Days")

    worst_days_display = worst_days.reset_index()
    st.dataframe(worst_days_display, use_container_width=True)

else:
    st.info("Configure the sidebar and click **Run Risk Analysis**.")