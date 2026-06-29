from dataclasses import asdict

def to_frontend_payload(self, backtest_output: dict) -> dict:
    return {
        "summary": {
            method: asdict(result)
            for method, result in backtest_output["summary"].items()
        },
        "breach_series": {
            method: df.reset_index().to_dict(orient="records")
            for method, df in backtest_output["breach_series"].items()
        },
    }
