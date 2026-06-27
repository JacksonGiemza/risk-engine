import streamlit as st
from src.models import RiskConfig

def render_sidebar() -> tuple[dict, bool]:
    st.sidebar.header("Configuration")

    portfolio_path = st.sidebar.text_input(
        "Portfolio CSV path",
        value=r"data\raw\portfolio\portfolio.csv",
    )

    end_date = st.sidebar.text_input(
        "End date",
        value="2026-06-17",
    )

    lookback_days = st.sidebar.number_input(
        "Lookback days",
        min_value=30,
        max_value=2000,
        value=365,
        step=30,
    )

    confidence_level = st.sidebar.selectbox(
        "Confidence level",
        options=[0.90, 0.95, 0.99],
        index=2,
    )

    num_simulations = st.sidebar.number_input(
        "Monte Carlo simulations",
        min_value=1_000,
        max_value=100_000,
        value=10_000,
        step=1_000,
    )

    num_worst_days = st.sidebar.number_input(
        "Worst days",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )

    random_seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=999_999,
        value=42,
        step=1,
    )

    run_analysis = st.sidebar.button("Run Risk Analysis")

    config = RiskConfig(
        portfolio_path=portfolio_path,
        start_date=None,
        end_date=end_date,
        lookback_days=int(lookback_days),
        confidence_level=float(confidence_level),
        num_simulations=int(num_simulations),
        random_seed=int(random_seed),
        num_worst_days=int(num_worst_days),
    )

    return config, run_analysis