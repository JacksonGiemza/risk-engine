# Design Document — Full-Revaluation Multi-Asset VaR Engine

**Status:** Proposed
**Owner:** Jackson Giemza
**Scope:** Evolve the current linear/return-based VaR engine into a risk-factor-based, full-revaluation engine that prices linear *and* non-linear (option) instruments under a common framework.

---

## 1. Motivation & Goals

### 1.1 The problem with the current design
Today the whole risk stack is **delta-one / return-based**. In `src/risk_engine.py`, every VaR method consumes a per-symbol `asset_returns` matrix and a `weights` vector, and models portfolio P&L as `weights · returns`. This is exact for ETFs and an acceptable linear approximation for futures and FX spot, but it fails for options in three ways:

1. **Non-linearity** — option P&L is not proportional to the underlying return (gamma/convexity).
2. **Multiple risk factors** — an option depends on underlying spot, volatility, the risk-free rate, and time, not one price series.
3. **Wrong driver** — the option premium is not the risk driver; the underlying is.

### 1.2 Why full revaluation
Full revaluation reprices every position under each scenario and takes portfolio P&L as the sum of value changes. It is the only design in which linear and non-linear instruments **cohere under one engine**, and most of the existing roadmap — stress testing, scenario analysis, filtered historical simulation, GARCH — is *literally* revaluation under different scenario generators. Building the revaluation core turns those later features from separate projects into plug-ins.

### 1.3 Design principles
- **One valuation path.** Base valuation and scenario valuation use the *same* pricers. Current market value becomes just "revaluation under today's factors."
- **Risk factors are first-class.** Instruments declare the factors they depend on; the engine shocks factors, not instrument prices.
- **Separate static from dynamic.** Contract terms (strike, expiry, multiplier) are metadata; market state (spot, vol, rate, date) is a snapshot passed in.
- **Vectorize the hot loop.** Scenario repricing is `S scenarios × N positions`; price each position across all scenarios in array form.
- **Keep the fast path.** For linear-only portfolios, keep the cheap return-based engine and auto-select.

### 1.4 Non-goals (for this phase)
- American/exotic option pricing (European BS only to start; the interface should not preclude adding a lattice/PDE pricer later).
- A full stochastic-vol or real implied-vol surface (start with realized vol; leave a seam for implied vol).
- Intraday / sub-daily horizons (1-day VaR is the default horizon).

---

## 2. Core Concept: Revaluation VaR

Let the risk factors be a vector **F** = (f₁, …, f_k): underlying spot prices, volatilities, rates, FX rates.

- **Base value:** V₀ = Σᵢ Vᵢ(**F**) — the portfolio value under today's factors.
- **Scenario s** applies a shock to the factors → **F_s**.
- **Scenario P&L:** PnL_s = Σᵢ [ Vᵢ(**F_s**) − Vᵢ(**F**) ].
- **VaR(α)** = −quantile of {PnL_s} at (1−α); **ES(α)** = mean of the tail beyond VaR.

Everything reduces to three questions the architecture must answer cleanly:

| Question | Component |
|---|---|
| What are the factors and their current values? | Risk-factor model + market snapshot |
| How do we generate shocked factor vectors **F_s**? | Scenario generator |
| How do we compute Vᵢ(**F**) for any factor vector? | Revaluation-capable pricers |

**Scenario generators (all feed the same repricer):**
- **Historical simulation** — for each historical day, take that day's *factor* change and apply it to today's snapshot. (Note: factor changes, not instrument-price returns — for equities they coincide; for options they don't.)
- **Monte Carlo** — draw factor changes from a fitted distribution (multivariate normal on factor returns to start; Student-t / EWMA / GARCH later).
- **Parametric (delta-gamma)** — closed-form approximation using Greeks; the only "parametric" option once the book is non-linear. Useful as a fast cross-check, not the primary path.

**Shock application rules matter.** Each factor needs a rule: **multiplicative** for prices/FX (apply a return), **additive** for rates and (typically) volatility. This must be a property of the risk factor, not hard-coded.

---

## 3. Target Architecture

```
                 ┌──────────────────────────────────────────────────┐
                 │                 RiskPipeline                      │
                 └──────────────────────────────────────────────────┘
                    │                │                    │
          ┌─────────▼──────┐  ┌──────▼─────────┐   ┌──────▼─────────┐
          │ Portfolio      │  │ FactorProvider │   │ RiskConfig     │
          │ (positions)    │  │ (market data)  │   │ (date, α, H)   │
          └─────────┬──────┘  └──────┬─────────┘   └────────────────┘
                    │                │
                    │        ┌───────▼──────────┐
                    │        │  MarketSnapshot   │  current factor values
                    │        │  + FactorHistory  │  aligned factor-change matrix
                    │        └───────┬──────────┘
                    │                │
          ┌─────────▼────────────────▼──────────┐
          │        ScenarioGenerator             │  → S shocked snapshots
          │  (historical / MC / delta-gamma)     │
          └─────────┬────────────────────────────┘
                    │
          ┌─────────▼──────────┐     ┌────────────────────────┐
          │ RevaluationEngine  │────▶│ PricerRegistry          │
          │ Σ ΔV per scenario  │     │  LinearPricer /         │
          │  → P&L vector       │     │  OptionPricer (BS+greeks)│
          └─────────┬──────────┘     └────────────────────────┘
                    │
          ┌─────────▼──────────┐
          │ RiskMetrics core   │  VaR / ES / worst-days from a P&L vector
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │ RiskReport / UI    │  + Greeks, factor attribution
          └────────────────────┘
```

---

## 4. Key Abstractions (the data model)

These are described as *interfaces/responsibilities*, not implementations.

- **RiskFactor** — identity (`id`, `kind` ∈ {EQUITY_SPOT, FX_RATE, RATE, VOL}), current value, and a **shock rule** (multiplicative vs additive). This is the linchpin that makes all instruments uniform.
- **MarketSnapshot** — a map `factor_id → value` at a single `valuation_date`. Both the base valuation and every scenario are a MarketSnapshot; a scenario is just a shocked copy.
- **PricingContext** — the slice of a snapshot a given instrument needs, plus `valuation_date` (so theta / time-to-expiry are consistent everywhere). This is the object your current `OptionPricer.black_scholes` is missing — it reaches for `metadata.sigma`, but spot/vol/rate are *market state*, not contract terms.
- **Instrument dependency declaration** — each instrument type answers "which factor ids do I depend on?" (ETF → its own spot; option → underlying spot + underlying vol + rate). The engine uses this to know which factors to source and shock.
- **Pricer interface** — `price(instrument, context) → Valuation` where `Valuation` carries the **absolute value** and **Greeks** (delta, gamma, vega, theta, rho). Absolute value is essential: the P&L is a *difference of valuations*, so `market_value` must come from the same function that reprices scenarios.
- **ScenarioSet** — produces S shocked MarketSnapshots from the base snapshot + a FactorHistory (for historical) or a fitted model (for MC).
- **RevaluationEngine** — the loop: for each scenario snapshot, reprice all positions, sum to a portfolio value, subtract base → a length-S P&L vector.

**The single most important refactor:** `PricingResult` (`src/pricing/models.py`) currently holds only `market_value` and `abs_exposure`. Grow it into a `Valuation` carrying absolute value + Greeks, and make *every* pricer return it. Once base valuation and scenario valuation share one code path, the linear and non-linear worlds unify.

---

## 5. Data Sources

Prioritize free sources; the engine stays fully functional on free data, with clearly identified fidelity gaps that a paid feed would close.

### 5.1 Free — the default stack

| Factor / need | Free source | Notes |
|---|---|---|
| Equity / ETF spot & history | **yfinance** (already used) | Underlyings for options must be added to the fetch set. |
| Index / futures proxies | **yfinance** (`ES=F`, `^GSPC`, …) | Continuous `=F` contracts are convenient but not roll-adjusted — a fidelity gap (see §5.3). |
| FX rates & history | **yfinance** (`EURUSD=X`) or **FRED** | Needed both for pricing foreign positions *and* as shock-able risk factors. |
| Risk-free rate / curve | **FRED** (`DGS1MO`, `DGS3MO`, `DGS1`, `DGS2`, `DGS10`, …) | Free API, clean, the right way to build a discount curve. Much better than scraping. |
| Risk-free (quick proxy) | **yfinance** yield tickers (`^IRX` 13-wk, `^TNX` 10-yr) | Fine for a single-rate BS input before you build a curve. |
| Volatility (realized) | Computed from underlying returns (free) | Annualized stdev of log returns over a window — the pragmatic starting σ. |
| Volatility (index proxy) | **yfinance** `^VIX` | Free proxy for equity-index implied vol and for *shocking* vol in scenarios (vega risk) without a per-name surface. |
| Dividends | **yfinance** dividend history | Improves option accuracy; optional at first (can fold into a dividend yield `q`). |

### 5.2 Where paid data becomes worth it

| Upgrade | Paid sources | What it buys you |
|---|---|---|
| **Per-name implied vol surface** | ORATS, IVolatility, OptionMetrics (IvyDB), CBOE DataShop, Polygon.io options | Accurate option **base prices** (market implied vol, not realized) and **real vega risk** with historical vol dynamics. This is the single biggest fidelity gap on free data. |
| **Roll-adjusted continuous futures** | Norgate (inexpensive), CSI | Correct futures P&L and clean history without roll gaps/survivorship bias. |
| **Dividend forecasts / borrow rates** | ORATS, IHS Markit | More accurate option pricing (forward), esp. for single names. |
| **Curve / OIS data** | Refinitiv, Bloomberg | Proper multi-curve discounting; overkill for this project. |

**Recommendation:** ship on **yfinance + FRED**. It produces a genuinely credible engine. Document precisely the two sacrifices — (a) option base prices use *realized* vol, so premiums won't match the market, and (b) `=F` futures aren't roll-adjusted — and note that an implied-vol feed (ORATS is the usual entry point) is the highest-ROI paid upgrade.

---

## 6. Step-by-Step Implementation Plan

Each phase below follows the same template: **Goal → Design decisions → Build order → Watch out for → Gate.** The phases are ordered so that every gate can be met using only the phases before it, and so the risky architectural bets (Phases 1, 3, 6) are validated early against numbers you already trust.

A note on dependencies: Phases 1–3 are the *pricing spine* (a position can be valued under any factor vector), Phases 4–5 are the *scenario supply* (where factor vectors come from), and Phases 6–7 are the *engine* that joins them. You can build the spine and the supply somewhat in parallel; they meet at Phase 6.

---

### Phase 0 — Prerequisites (small)
**Goal.** Make an option constructible and priceable in isolation so Phase 1 has something to wrap.
**Do.** Complete the instrument refactor (underlying as a *symbol*, populate option currency) and make sure a single risk-free rate is available from config. The small pricer bugs you already know about live here — not worth detailing.
**Gate.** One option prices to a Hull textbook value in a throwaway script.

---

### Phase 1 — Risk-factor model & pricing context
**Goal.** Give the system a vocabulary for "market state" so pricing stops depending on `position.market_price` and `datetime.today()` and starts depending on an explicit, swappable snapshot.

**Design decisions.**
- **Factor identity is a namespaced string.** Adopt a convention like `EQ:SPY`, `VOL:SPY`, `FX:EURUSD`, `RATE:USD:3M`. The payoff: an *option on SPY* and a *held SPY ETF* both resolve to the same `EQ:SPY` factor, so they automatically share a shock and their correlation is captured for free. Naming is the cheapest correlation model you'll ever write — get it right here.
- **Shock rule belongs to the factor kind, not the call site.** `EQ`/`FX` shock *multiplicatively* (apply a return), `RATE`/`VOL` shock *additively* (apply a level change). Encode this once on the `FactorKind` so every generator and pricer treats a factor identically.
- **`MarketSnapshot` is immutable.** It's a `{factor_id → value}` map plus a `valuation_date`. Give it a `with_shocks(shock_map) → MarketSnapshot` that returns a *copy*. You will create thousands of shocked snapshots; shared mutable state here is a guaranteed aliasing bug.
- **`PricingContext` is a per-instrument *view*, addressed by role not id.** The pricer should ask for `context.spot`, `context.vol`, `context.rate` — semantic roles — and the context translates role → factor_id using the instrument's declared dependencies (Phase 2). This keeps pricers ignorant of the global naming scheme and makes them trivially unit-testable.
- **Pricers become pure functions of `(metadata, context)`.** No clocks, no globals. `valuation_date` comes only from the context. This one rule is what makes historical repricing and reproducible tests possible at all.

**Build order.** FactorKind + shock rule → id-naming convention → `MarketSnapshot` (construct, `get`, `with_shocks`) → `PricingContext` (role accessors) → retrofit `OptionPricer` to read from context (plug in a placeholder constant vol for now; real vol arrives in Phase 4).

**Watch out for.** A missing factor must raise loudly, not return NaN — you saw how a silent NaN market_price already forces defensive checks in `Portfolio._validate_portfolio`; don't repeat that pattern in the factor layer.

**Gate.** A snapshot builds for the sample portfolio; `with_shocks` produces a correctly shocked copy *and leaves the original unchanged* (assert immutability); the option prices identically whether the date is supplied by context or by "today" when the two coincide.

---

### Phase 2 — Instrument → factor dependency declaration
**Goal.** Let each position answer "which factors do I depend on?" — this drives both the per-instrument context (Phase 1) and the portfolio-wide list of factors to source (Phase 4).

**Design decisions.**
- **The mapping is derived from the metadata instance, not a static per-type list.** An option's factor ids depend on its *underlying symbol*; a foreign ETF's depend on its *currency*. So the resolver takes a metadata object and returns a list of `(role, factor_id, kind)` requirements.
- **This mapping does double duty.** Per instrument it builds the `PricingContext`; across the portfolio the **union** of requirements is exactly the set the provider must fetch and the scenario generator must shock. Make "collect all required factors" a first-class portfolio operation — it's the contract between the pricing spine and the data layer.
- **Currency becomes an explicit factor dependency.** A position whose currency ≠ base implicitly depends on an `FX:*` factor. Formalizing this here replaces the ad-hoc `if metadata.currency != position.currency` branches now scattered through `LinearPricer`, and it makes FX a properly shock-able risk (today it's only a static conversion).

**Build order.** Define the requirement record → per-type resolvers (ETF: own spot [+FX]; Future: underlying spot [+FX]; FXSpot: the FX factor itself; Option: underlying spot + vol + rate [+FX]) → portfolio-level union/dedupe.

**Watch out for.** Two instruments on the same underlying *must* produce the identical factor id (the Phase 1 convention guarantees this) — otherwise you double-source data and lose their correlation in scenarios.

**Gate.** Option → `{EQ:SPY, VOL:SPY, RATE:USD}`; a EUR-denominated ETF → `{its spot, FX:EURUSD}`; the portfolio union dedupes shared underlyings.

---

### Phase 3 — Revaluation-capable pricers + registry
**Goal.** One valuation code path for every instrument, returning value *and* Greeks, so base valuation and scenario valuation are literally the same function.

**Design decisions.**
- **Split each pricer into a pure per-unit core and a thin position wrapper.** The core is a pure function — e.g. BS over `(S, K, T, r, σ, type)` — returning per-unit price and per-unit Greeks. The wrapper scales by `qty × multiplier` and tags the currency. Why: the per-unit core is what you unit-test against Hull and what you'll call *vectorized* in Phase 6; the wrapper is where contract scaling lives. Keep the core free of pandas/objects — plain numbers and numpy arrays only.
- **`Valuation` is per-position, in instrument currency, with Greeks.** Returning a per-position present value keeps the engine's aggregation a dumb sum. Carry the currency so the engine can convert in Phase 6. **Nail Greek conventions now and write them in the docstring** — vega per 1.00 vol (i.e. 100 vol points) vs per 0.01; theta per calendar day vs per year; delta in underlying units. Phase 9's vol scenarios and the delta-gamma cross-check both depend on these units being unambiguous.
- **A registry replaces the if-ladders.** Map `instrument_type → InstrumentSpec`, where a spec bundles the three per-type behaviors that currently sprawl across files: metadata loader, factor resolver (Phase 2), and pricer. Adding an asset class becomes one registration, not edits in `instrument_loader`, `pricing_engine`, and `linear_pricers`.
- **Linear pricers now read spot from the context, not `position.market_price`.** This is the subtle unlock: once an ETF's value is `qty × context.spot × 1`, the *same* function reprices it under a shocked spot. The base snapshot's spot must equal today's market_price, so the numbers reconcile exactly — which is the gate.

**Build order.** Define `Valuation` (documented Greek units) → pure per-unit BS core (price + closed-form Greeks) → option position wrapper reading `S/σ/r` from context and `T` from `valuation_date`→expiry → reimplement linear pricers against the context → build the registry and delete the if-ladders.

**Watch out for.** `T ≤ 0` (expired) must return intrinsic value, not NaN — historical windows in Phase 6 will cross expiries. Guard `d1/d2` for deep ITM/OTM and very small `T` (gamma/theta blow up; acceptable, but don't divide by zero).

**Gate.** Per-unit BS matches Hull for a call *and* a put; put-call parity holds across a range of spot; closed-form Greeks match central finite-difference bumps (this simultaneously validates that repricing is deterministic); and linear base valuations reconcile *exactly* to the current `market_value` on the sample portfolio.

---

### Phase 4 — Market data upgraded to a multi-factor provider
**Goal.** Supply two distinct things the current `MarketData` conflates: the **base snapshot** (latest value per factor) and the **factor history** (aligned level series used to build scenarios).

**Design decisions.**
- **Introduce a provider interface, keep concrete providers small.** `FactorProvider` orchestrates a `YFinanceProvider` (equity/ETF/underlying/FX/VIX) and a `FredProvider` (rate tenors), and returns `MarketSnapshot + FactorHistory`. This is the "provider abstraction layer" already on your roadmap; it's also the seam where a paid vol feed later slots in without touching the engine.
- **Store *levels* in the history; derive changes with a rule-aware transform.** Keep the raw aligned level matrix (dates × factor_id) auditable, and provide `to_changes()` that applies each factor's mult/add rule. Historical simulation needs the changes directly; Monte Carlo needs their covariance — both derive cleanly from levels, and you keep one place where the mult/add distinction is applied.
- **Volatility is a sourced factor with its own level series.** Compute rolling annualized realized vol from each underlying's returns → a `VOL:*` level series. Because it's a level, vol can be both a base-pricing input *and* shocked in scenarios (additively). Document that realized vol is laggy and won't match market premiums; `^VIX` is the free index-vol overlay.
- **Rates: source a few FRED tenors, interpolate to each option's `T`.** Start with a single tenor if you want to move fast; leave the interpolation seam so a real curve is a drop-in later.
- **Redesign the cache around per-symbol full history.** The current key `(ticker-set hash, start, end)` fragments into a new file per date range (you already have ~10 overlapping CSVs). Cache each symbol's full series once and slice in memory; historical-window requests become cache hits, which matters enormously once backtesting reprices every day.

**Build order.** Provider interface + implementations → source the Phase-2 union (underlyings, FX pairs, rate tenors) → compute realized vol → assemble aligned level history + base snapshot → new cache.

**Watch out for.** Rate calendars differ from equity calendars (align + forward-fill deliberately). Realized vol needs a warm-up window *before* the earliest scenario date. A newly listed underlying shorter than the window → shorten or error clearly, never silently pad. Mind FRED percent-vs-decimal and annualization.

**Gate.** Every factor in the union is sourced (no missing columns); history is aligned, monotonic, warm-up-trimmed, and the latest history date ≤ `valuation_date` (no look-ahead); cache-slice equals a fresh fetch byte-for-byte.

---

### Phase 5 — Scenario generation
**Goal.** Turn the base snapshot + factor history into `S` shocked factor vectors — the only thing that differs between Historical, Monte Carlo, and stress.

**Design decisions.**
- **Represent a scenario set as a shock *matrix* `(S × K)`, not `S` snapshot objects.** Pair it with the base snapshot and let the engine apply-and-reprice in array form. Materializing thousands of snapshot objects is the naive path; a matrix keeps Phase 6 vectorizable.
- **One `apply_shocks(base, shock_row, rules)` primitive, used by every generator.** This guarantees Historical and Monte Carlo treat each factor identically (mult vs add). It's the single source of truth for how a shock lands on a factor.
- **Historical generator = last `W` rows of factor *changes*.** One-day horizon → one-day changes. For an `H`-day horizon, use actual overlapping `H`-day changes rather than √H scaling — scaling is only valid in the linear-normal world and would defeat the point of revaluation.
- **Monte Carlo generator = fit then sample.** Estimate mean/cov of factor changes, draw `N` (seeded — you already carry `random_seed`). Start multivariate-normal; Student-t / EWMA-weighted are Phase 9 swaps of *this step only*.
- **Delta-gamma is a different animal — keep it separate.** It doesn't reprice; it combines Phase-3 Greeks with the factor covariance to get an analytic P&L distribution. Keep it as an alternate fast path and cross-check, not part of the reval scenario machinery.

**Build order.** `apply_shocks` primitive → historical set (changes → matrix) → Monte Carlo set (fit + sample → matrix) → optional delta-gamma analytic path.

**Watch out for.** Additive vol shocks can push vol negative → floor at a small positive. Additive rate shocks can legitimately go negative → allow but bound. Overlapping multi-day windows reduce the count of *independent* samples — note it for backtest interpretation.

**Gate.** Historical scenario count == window length; asserting a known shock shows price/FX moved multiplicatively and rate/vol additively; MC with a fixed seed reproduces exactly, and sample covariance → input covariance as `N` grows.

---

### Phase 6 — Revaluation engine
**Goal.** Join the spine and the supply: reprice every position across all `S` scenarios and produce the portfolio P&L vector. This is the phase the whole design exists for.

**Design decisions.**
- **Loop over instruments (N), vectorize over scenarios (S).** For each position: slice the shock matrix to its factor columns, build arrays of its shocked factor values, call the *array-safe per-unit core* from Phase 3 → an array of `S` values, subtract the base value → that position's `S` P&Ls. Never loop over scenarios in Python.
- **Convert to base currency per scenario, then sum.** FX differs by position and by scenario, so apply each position's `FX:*` scenario path *before* summing across positions. Convert-then-sum, not sum-then-convert.
- **Return the `S × N` P&L matrix, not just the length-`S` vector.** The row-sum is your P&L vector for VaR; the full matrix powers instrument- and factor-level tail attribution in reporting at almost no extra cost.
- **Base value (all shocks zero) must equal the Phase-3 base valuation.** Assert it inside the engine — it's a free, permanent correctness tripwire.

**Build order.** Confirm per-unit cores are array-safe → implement slice→shock→reprice→diff per instrument → per-scenario currency conversion → return matrix + vector.

**Watch out for.** Cash/factorless positions → a zero P&L row (don't special-case them into a crash). Options expiring inside the horizon → intrinsic, inherited from Phase 3. If performance bites, profile before optimizing — the intended shape (vectorized over S) is usually enough for a resume-scale book.

**Gate (the pivotal one).** On a **linear-only** portfolio, the reval P&L vector ≈ the return-based P&L (`weights·returns × portfolio_value`) within floating tolerance — proof the new engine is a strict superset of the old. Additionally: a one-option book's finite-difference delta (reprice at ±small spot shock) ≈ its analytic Phase-3 delta; and zero-shock value == Phase-3 base valuation.

---

### Phase 7 — Risk metrics from a P&L vector
**Goal.** Collapse the three VaR "methods" onto one metric core, so a method name just selects a scenario generator.

**Design decisions.**
- **Extract `metrics_from_pnl(pnl, portfolio_value, α) → RiskMetrics`.** VaR = −quantile(pnl, 1−α); ES = −mean of the sub-VaR tail; report both `$` and `%`. Today `historical_var` and `monte_carlo_var` each re-derive this — unify it.
- **Method = generator, not a separate math path.** Historical = metrics on the historical P&L vector; Monte Carlo = metrics on the MC vector; Parametric = the delta-gamma analytic distribution (or a normal fit to the P&L vector). This is a genuine conceptual simplification worth stating explicitly in the code's design.
- **Preserve the linear fast path.** For linear-only books, keep `weights·returns` (no repricing) but run it through the *same* `metrics_from_pnl`, so fast-path and reval numbers are directly comparable.
- **`worst_days` operates on scenarios now.** Worst `n` scenarios by P&L; for the historical generator, map each back to its driving calendar date so the "worst days" table still reads meaningfully.

**Build order.** `metrics_from_pnl` core → adapt historical/MC to call it on reval vectors → refit `worst_days` → wire method→generator selection.

**Watch out for.** Match the quantile interpolation the current `historical_var` uses, or the comparison in the gate will show spurious differences. Keep one sign convention (gains positive, VaR reported positive) end to end.

**Gate.** Reval-Historical VaR on a linear book matches the current `historical_var` within tolerance; ES ≥ VaR always; both scale linearly when you double position sizes.

---

### Phase 8 — Pipeline / reporting integration
**Goal.** Make the whole thing run from `RiskPipeline` on a mixed book, and expose the new risk surface.

**Design decisions.**
- **Auto-select the engine, allow an override.** If every instrument is linear → fast path; if any is non-linear → revaluation. A config flag to *force* revaluation is invaluable for the Phase-6/7 regression checks.
- **`RiskPipeline` assembles the new stack.** FactorProvider (4) → ScenarioSet (5) → RevaluationEngine (6) → metrics (7), replacing the current `MarketData`/`PricingEngine` calls. The pipeline's shape barely changes; what it wires up does.
- **Grow the report.** `RiskReport` gains portfolio-aggregated Greeks (Δ/Γ/vega/θ) and tail attribution from the `S × N` matrix (which positions/factors drove the worst scenarios). Surface these in `ui/`.

**Watch out for.** Backtesting now reprices historically each day and gets much heavier — it reuses the cached factor history from Phase 4, and it's the first place worth parallelizing (already on your roadmap).

**Gate.** End-to-end run on an equities + options book; backtest completes; and a linear-only book returns the *same* numbers as before the refactor (top-level regression).

---

### Phase 9 — Scenario realism (now cheap plug-ins)
**Goal.** Add fidelity by adding scenario generators — the engine, pricers, and metrics don't change.

- **Vega / vol risk** — include `VOL:*` factors in the shock set (historical vol changes, or VIX-scaled), validated against the Phase-3 vega via the delta-gamma path.
- **Filtered Historical Simulation** — standardize historical residuals by an EWMA/GARCH vol estimate and rescale to current vol; plugs into the historical generator only.
- **EWMA / GARCH** — a vol model feeding either the MC draws or FHS.
- **Stress & scenario analysis** — hand-specified shock rows through the same `apply_shocks` + reval; a stress test *is* a one-row scenario set.
- **Student-t VaR** — swap the MC sampling distribution; nothing else moves.

**Gate.** Each new generator collapses to an existing one in its degenerate case (e.g. FHS with constant vol == plain historical) — a cheap correctness anchor.

---

## 7. Improvements to Existing Systems

Concrete upgrades, keyed to your files:

- **`src/market_data.py` — cache fragmentation.** The cache key is `(ticker-set hash, start, end)`, which spawns a new file per date range (you already have ~10 raw-price CSVs for overlapping windows). Switch to caching **full per-ticker history** and slicing in memory; date-range requests become cache hits. This also makes historical revaluation windows cheap.
- **`src/market_data.py` — single-purpose.** It only knows equity closes. It needs to become the multi-factor provider (underlyings, FX, rates, vol, dividends) behind a provider interface. This is where your roadmap's "market data provider abstraction layer" lands.
- **`src/pricing/pricing_engine.py` & `instrument_loader.py` — if-ladders.** Both dispatch on `instrument_type` with growing `if` chains across multiple files. Move to a **registry** keyed by instrument type so each new asset class is one additive registration, not edits in three places.
- **`src/pricing/models.py` — `PricingResult` is too thin.** Grow it to a `Valuation` with absolute value + Greeks so base and scenario pricing share one path.
- **`src/risk_engine.py` — return-centric API.** Methods take returns/weights directly. Extract a **P&L-vector core** (VaR/ES/worst-days from a loss vector) that both the linear path and the revaluation path call. This removes duplication and is the seam where revaluation plugs in.
- **`src/portfolio.py` — row-by-row pricing.** `_price_positions` loops with `itertuples`; fine for base valuation, but the revaluation engine must **not** loop 10k scenarios × N positions in Python. Design the repricer to price a position across all scenarios in one vectorized call.
- **`src/pricing/currency_conversion.py` — live-only, latest-rate.** `CurrencyConverter` fetches only *current* rates from a live API. Historical revaluation needs **historical FX**, and FX should be a **shock-able risk factor**. Keep the live converter for reporting-currency conversion of the final snapshot; source historical FX from the factor provider.
- **`src/models.py` `RiskConfig` — missing risk-engine controls.** Add `valuation_date`, risk-free rate/curve config, **VaR horizon (holding period)**, scenario settings (window, method), and a vol-model choice. These are inputs the revaluation engine needs that don't exist yet.
- **`src/backtesting.py` — cost awareness.** Once VaR is revaluation-based, backtesting reprices historically each day and gets much heavier. Plan for it: reuse cached factor histories, vectorize, and consider parallelization (already on your roadmap) here first.

---

## 8. Testing & Validation Strategy

Extend the clean pricer tests you already have in `tests/test_pricing.py`.

- **Pricer correctness:** BS vs Hull reference values; **put-call parity**; Greeks vs **finite-difference bumps** (this doubles as a correctness check on the revaluation loop itself).
- **Superset regression (critical):** revaluation VaR on a linear-only portfolio must match the current return-based VaR. This is the proof you didn't regress the equity engine.
- **Delta-gamma vs full reval:** they should agree for small underlying moves and diverge for large ones / near expiry — a sanity check on both.
- **Golden portfolio snapshot:** freeze a mixed portfolio + fixed factor history + seed, snapshot the VaR/ES/Greeks, and regression-test against it.
- **Scenario integrity:** assert factor histories are date-aligned and carry no look-ahead; assert shock rules (mult vs add) are applied correctly.

---

## 9. Risks & Pitfalls

- **Look-ahead / survivorship bias** in historical scenarios — align windows and use point-in-time data.
- **Factor coverage gaps** — a newly listed underlying may lack history for the full window; define a fallback (shorter window, proxy factor).
- **Expired options in historical windows** — time-to-expiry can go ≤ 0 across a long lookback; decide handling (hold time fixed for 1-day VaR, or roll consistently).
- **Volatility data scarcity on free feeds** — realized vol won't reproduce market premiums; be explicit that base option values are model prices until an implied-vol feed is added.
- **Valuation-date consistency** — all factors and theta must reference one `valuation_date`; a mismatch silently corrupts P&L.
- **Currency correctness** — non-base positions need their FX factor shocked, and P&L aggregated in base currency.
- **Performance** — `S × N` repricing is the hot path; vectorize per instrument before optimizing anything else.

---

## 10. Milestone Summary

| Phase | Deliverable | Test gate |
|---|---|---|
| 0 | Instrument refactor fixed; BS stub priceable | Matches a Hull value |
| 1 | RiskFactor / MarketSnapshot / PricingContext | Snapshot builds; pricer reads context |
| 2 | Instrument factor-dependency declaration | Correct factor lists per type |
| 3 | Unified pricer interface + registry | Linear reconciles; Greeks vs finite-diff |
| 4 | Multi-factor provider + FRED rates + vol | Aligned history, no look-ahead |
| 5 | Scenario generators (hist / MC / DG) | Correct shock rules & counts |
| 6 | Revaluation engine → P&L vector | **Linear reval ≈ return-based** |
| 7 | RiskMetrics from P&L vector | Matches fast path on linear book |
| 8 | Pipeline + UI + reporting integration | End-to-end mixed-book run |
| 9 | Vega / FHS / stress / scenario / t-VaR | Each added as a scenario plug-in |

---

## 11. Resume Framing

The line "**refactored a linear equity VaR tool into a risk-factor-based, full-revaluation multi-asset engine**" is strong and true once Phase 6 lands. The **risk-factor abstraction** (Phase 1) is the intellectual centerpiece worth being able to whiteboard end to end: factors → scenarios → revaluation → P&L distribution → VaR.
