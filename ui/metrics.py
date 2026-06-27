import pandas as pd
import streamlit as st


def render_portfolio_metrics(report) -> None:
    st.subheader("Portfolio Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Gross Exposure",
        f"${report.portfolio_summary.gross_exposure:,.0f}",
    )

    col2.metric(
        "Net Exposure",
        f"${report.portfolio_summary.net_exposure:,.0f}",
    )

    col3.metric(
        "Long Exposure",
        f"${report.portfolio_summary.long_exposure:,.0f}",
    )

    col4.metric(
        "Short Exposure",
        f"${report.portfolio_summary.short_exposure:,.0f}",
    )


def render_risk_metrics(report) -> None:
    st.subheader("Risk Summary")

    risk_table = report.risk_table.copy()

    risk_table["VaR Return"] = risk_table["VaR Return"].apply(lambda x: f"{x:.2%}")
    risk_table["ES Return"] = risk_table["ES Return"].apply(lambda x: f"{x:.2%}")
    risk_table["VaR $"] = risk_table["VaR $"].apply(lambda x: f"${x:,.0f}")
    risk_table["ES $"] = risk_table["ES $"].apply(lambda x: f"${x:,.0f}")

    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True,
    )