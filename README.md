# Risk Engine

This is my current pet project, a modular Python based portfolio risk engine for calculating, backtesting, and visualizing market risk using multiple Value at Risk (VaR) methodologies. The project emphasizes clean software architecture, quantitative finance concepts, and extensibility toward institutional grade risk systems.

References:<br>
- Options Futures and Other Derivatives by John C Hull
- [MathWorks Overview of VaR Backtesting](https://www.mathworks.com/help/risk/overview-of-var-backtesting.html)


### Installation

Clone the repository:
```
git clone https://github.com/<your-username>/risk-engine.git
cd risk-engine
```

Create and activate a virtual environment:
```
python -m venv .venv
```
Windows:
```
.venv\Scripts\activate
```
macOS / Linux:
```
source .venv/bin/activate
```
Install the required packages:
```
pip install -r requirements.txt
```
Start the Streamlit application:
```
streamlit run app.py
```


---

## Project Structure

```text
risk-engine/
│
├── app.py                      # Streamlit dashboard
│
├── data/
│   ├── raw/
│   └── cache/
│
├── src/
│   ├── market_data.py
│   ├── portfolio.py
│   ├── risk_engine.py
│   ├── backtesting.py
│   ├── pipeline.py
│   └── models.py
│
├── ui/
│   ├── sidebar.py
│   ├── metrics.py
│   ├── charts.py
│   ├── tables.py
│   └── backtesting.py
│
└── README.md
```

---

## Risk Models

### Historical VaR

Computes Value at Risk directly from the empirical distribution of historical portfolio returns.

### Parametric VaR

Uses the variance covariance approach assuming multivariate normally distributed asset returns.

### Monte Carlo VaR

Generates simulated portfolio returns using a multivariate normal distribution estimated from historical asset returns.

---

## Backtesting

The engine supports rolling out of sample validation of VaR forecasts.

Implemented statistical tests include:

* Exact Binomial Test
* Kupiec Unconditional Coverage Test
* Christoffersen Independence Test
* Christoffersen Conditional Coverage Test
* Basel Traffic Light Test

These tests evaluate both the frequency and clustering of VaR violations.

---

## Technologies

* Python
* Pandas
* NumPy
* SciPy
* yfinance
* Streamlit
* Altair
* Dataclasses

---

## Design Goals

The project is designed around several principles:

* Modular architecture
* Separation of concerns
* Strong typing with dataclasses
* Reusable risk components
* Scalable pipeline design
* Extensible instrument model

The long term goal is to evolve the engine from a simple linear equity portfolio into a flexible multi asset risk platform.

---

## Roadmap

### In Progress

* Refactoring instrument architecture
* Improved dashboard organization
* Expanded market data abstraction

### Planned

* Support for futures
* Support for FX spot positions
* European option pricing
* Black-Scholes implementation
* Greeks calculation
* Non linear portfolio VaR
* Stress testing
* Scenario analysis
* Historical simulation improvements
* EWMA volatility model
* Student's t-distribution VaR
* Filtered Historical Simulation (FHS)
* GARCH volatility forecasting
* Multi-portfolio support
* Parallel risk calculations
* PostgreSQL market data storage
* Market data provider abstraction layer

---
