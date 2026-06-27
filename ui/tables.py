import streamlit as st


def render_holdings(report) -> None:
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

    st.dataframe(
        holdings,
        use_container_width=True,
        hide_index=True,
    )


def render_worst_days(report) -> None:
    st.subheader("Worst Trading Days")

    worst_days = report.worst_days.copy().reset_index()

    if "Return" in worst_days.columns:
        worst_days["Return"] = worst_days["Return"].apply(lambda x: f"{x:.2%}")

    if "Dollar Loss" in worst_days.columns:
        worst_days["Dollar Loss"] = worst_days["Dollar Loss"].apply(lambda x: f"${x:,.0f}")

    st.dataframe(
        worst_days,
        use_container_width=True,
        hide_index=True,
    )