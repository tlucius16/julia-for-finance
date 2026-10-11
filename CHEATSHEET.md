# Quant finance in Julia: cheat sheet

The key formulas and code from each chapter on one page. Each section links to the file where the code lives and runs. Conventions used throughout:

- `X` is a T × N matrix of daily **log** returns (rows = days, columns = assets), built by `load_returns()` in [common.jl](common.jl).
- Losses (VaR, ES) are **positive** numbers: a 3% loss is `0.03`.
- Annualise with 252 trading days: mean × 252, volatility × √252.

## Data and returns ([01_data.jl](01_data.jl))

| What | Formula | Julia |
| --- | --- | --- |
| Log return | $r_t = \ln(P_t / P_{t-1})$ | `diff(log.(p))` |
| Growth of $1 | $P_t / P_0$ | `p ./ first(p)` |
| Annual return / vol | $252\,\bar r$, $\sqrt{252}\,s$ | `252 * mean(r)`, `sqrt(252) * std(r)` |
| Duration (from data) | $D \approx -\mathrm{cov}(r, \Delta y) / \mathrm{var}(\Delta y)$ | `-cov(r, dy) / var(dy)` |

Duration is the % price drop for a 1 percentage-point rise in yield. `Δy` is the daily change in the 10-year yield as a decimal (`^TNX` / 100).

## VaR and expected shortfall ([02_var_es.jl](02_var_es.jl))

At confidence α (e.g. 0.95), **VaR** is the loss you exceed on a fraction 1 − α of days, and **ES** is the average loss on those days.

| Method | VaR | ES |
| --- | --- | --- |
| Historical | $-q_{1-\alpha}(r)$ | $-\mathrm{mean}(r \mid r \le q_{1-\alpha})$ |
| Normal | $-(\mu + \sigma z)$, with $z = \Phi^{-1}(1-\alpha)$ | $-\left(\mu - \sigma\,\varphi(z)/(1-\alpha)\right)$ |
| Monte Carlo | Historical method on simulated returns | Same |

```julia
q = quantile(r, 1 - α);  VaR, ES = -q, -mean(r[r .<= q])            # historical
sims = permutedims(rand(MvTDist(ν, μ, Σ .* (ν - 2) / ν), n))         # Student-t sims (n × N)
```

- The Normal understates ES because real returns have fatter tails. Student-t (ν ≈ 4) gets closer.
- A portfolio's VaR is lower than the weighted sum of its assets' VaRs. That gap is diversification.
- `MvTDist` takes a **scale** matrix, not the covariance: covariance = scale × ν / (ν − 2).

## Backtesting VaR ([common.jl](common.jl))

A **breach** is a day whose return is below −VaR. The two tests ask different questions:

| Test | Question | Statistic | Reject the model if |
| --- | --- | --- | --- |
| Kupiec | Is the breach rate equal to 1 − α? | $LR = -2\ln\dfrac{(1-p)^{T-x}p^x}{(1-\hat\pi)^{T-x}\hat\pi^x} \sim \chi^2_1$ | p-value < 0.05 |
| Christoffersen | Are breaches independent of yesterday's? | Compares $P(\text{breach} \mid \text{breach yesterday})$ with $P(\text{breach} \mid \text{calm yesterday})$, $\sim \chi^2_1$ | p-value < 0.05 |

Here $p = 1-\alpha$, $x$ = number of breaches, $T$ = number of days and $\hat\pi = x/T$. A model can pass Kupiec (right count) and still fail Christoffersen (breaches cluster).

## Volatility models ([03_volatility_models.jl](03_volatility_models.jl))

| Model | Variance forecast | Notes |
| --- | --- | --- |
| Rolling | $\sigma_t^2 = \mathrm{var}(r_{t-252:t-1})$ | Slow to react; drops a shock abruptly 252 days later |
| EWMA | $\sigma_t^2 = \lambda\sigma_{t-1}^2 + (1-\lambda)\,r_{t-1}^2$ | λ = 0.94 (RiskMetrics daily) |
| GARCH(1,1) | $\sigma_t^2 = \omega + a\,\varepsilon_{t-1}^2 + b\,\sigma_{t-1}^2$ | Fit by max likelihood; long-run variance $\omega/(1-a-b)$ |
| FHS | VaR = $-(\mu + \sigma_t\, q_{1-\alpha}(\varepsilon/\sigma))$ | GARCH vol × empirical quantile of standardised residuals |

- **Persistence** is $a + b$. A shock's half-life is $\ln 0.5 / \ln(a+b)$ days.
- GARCH negative log-likelihood: $\tfrac12\sum_t \left[\ln(2\pi\sigma_t^2) + \varepsilon_t^2/\sigma_t^2\right]$.
- Optimise over unconstrained θ and map it to valid parameters (ω > 0, a + b < 1) with `exp` and a softmax, as `garch_params` does.

## Portfolio optimisation ([04_portfolio_optimization.jl](04_portfolio_optimization.jl))

| Portfolio | Problem | Solution |
| --- | --- | --- |
| Min variance (shorts allowed) | min $w'\Sigma w$ s.t. $\sum w = 1$ | $w = \Sigma^{-1}\mathbf 1 / (\mathbf 1'\Sigma^{-1}\mathbf 1)$ → `w = Σ \ ones(N); w ./= sum(w)` |
| Efficient frontier (long-only) | min $w'\Sigma w$ s.t. $\mu'w = $ target, $w \ge 0$ | Quadratic program (JuMP + HiGHS) |
| Max Sharpe | max $(\mu'w - r_f)/\sqrt{w'\Sigma w}$ | Not convex as written. Substitute $y = w/((\mu-r_f)'w)$: min $y'\Sigma y$ s.t. $(\mu-r_f)'y = 1$, $y \ge 0$, then $w = y/\sum y$ |
| Min ES | min $\zeta + \frac{1}{(1-\alpha)T}\sum_t u_t$, with $u_t \ge -X_t w - \zeta$, $u_t \ge 0$ | Linear program (Rockafellar–Uryasev); at the optimum ζ is the VaR |
| Risk parity | Every asset contributes 1/N of the risk | min $\tfrac12 y'\Sigma y - \tfrac1N\sum\ln y_i$, then $w = y/\sum y$ |

- **Risk contribution:** $RC_i = w_i(\Sigma w)_i / (w'\Sigma w)$, and the contributions sum to 1. In Julia: `w .* (Σ * w) ./ (w' * Σ * w)`.
- **Ledoit–Wolf shrinkage:** $\Sigma = \delta F + (1-\delta) S$, with $F = (\mathrm{tr}\,S/n)\, I$. It matters when the number of assets is large relative to the number of days.
- **Max drawdown:** `1 - exp(-maximum(accumulate(max, c) .- c))`, where `c = cumsum(r)`.
- Expected returns estimated from history are very noisy. Optimisers that use them (max Sharpe) disappoint out of sample, while equal weight and risk parity hold up.

## Options data ([05_option_data.jl](05_option_data.jl))

- **Clean the chain first:** drop contracts with no bid or with ask ≤ bid, then use the mid price `(bid + ask) / 2`.
- **Implied spot from put–call parity:** $S \approx K + C - P$ at the strike where call and put prices are closest.
- **Smile:** plot out-of-the-money options only, puts below spot and calls above.
- **Put/call open-interest ratio:** above 1 means more puts than calls outstanding.

## Option pricing ([07_option_pricing.jl](07_option_pricing.jl))

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt T}, \qquad d_2 = d_1 - \sigma\sqrt T$$

| Quantity | Call | Put |
| --- | --- | --- |
| Price | $S N(d_1) - K e^{-rT} N(d_2)$ | $K e^{-rT} N(-d_2) - S N(-d_1)$ |
| Delta ∂V/∂S | $N(d_1)$ | $N(d_1) - 1$ |
| Gamma ∂²V/∂S² | $n(d_1) / (S\sigma\sqrt T)$ | Same as the call |
| Vega ∂V/∂σ | $S\, n(d_1)\sqrt T$ | Same as the call |
| Theta (per year) | $-\dfrac{S\,n(d_1)\,\sigma}{2\sqrt T} - rKe^{-rT}N(d_2)$ | Not shown |
| Put–call parity | $C - P = S - Ke^{-rT}$ | |

In the table, N is the standard normal CDF (`cdf(Normal(), x)`) and n is its density (`pdf(Normal(), x)`).

```julia
# Implied vol by Newton's method: step along vega until the model price matches the market
σ = σ - (bs(S, K, T, r, σ) - price) / vega(S, K, T, r, σ)

# Monte Carlo price of a call (error shrinks like 1/√n)
ST = S .* exp.((r - σ^2 / 2) * T .+ σ * sqrt(T) .* randn(n))
price = exp(-r * T) * mean(max.(ST .- K, 0))

# Check any Greek with a finite difference
delta_fd = (bs_call(S + h, K, T, r, σ) - bs_call(S - h, K, T, r, σ)) / 2h
```

## Machine learning ([06_machine_learning.jl](06_machine_learning.jl))

| Feature | Formula | Measures |
| --- | --- | --- |
| Robust momentum | $\mathrm{Median}(r) / \mathrm{IQR}(r)$ | Consistent absolute returns |
| Market-relative | $\mathrm{Median}(r - r_m) / \mathrm{IQR}(r - r_m)$ | Consistent outperformance of the market |
| Residual | $\mathrm{Median}(\varepsilon) / \mathrm{IQR}(\varepsilon)$, where ε is the residual of r regressed on $r_m$ | Return left after removing beta |
| Rolling Sharpe | $\mathrm{Mean}(r) / \mathrm{Std}(r)$ | Ordinary return per unit of risk |

| Tool | Formula | Julia |
| --- | --- | --- |
| Ridge regression | $\beta = (Z'Z + \lambda n I)^{-1} Z'y$ | `(Z'Z + λ * n * I) \ (Z'y)` |
| Cross-sectional z-score | Each day, $(x - \bar x)/s$ across assets | `transform!(groupby(df, :t), cols .=> zscore .=> cols)` |
| Rank IC | Spearman correlation of predicted vs realised ranks, per date | `cor(ranks(p), ranks(y))`, with `ranks(x) = invperm(sortperm(x))` |
| IC t-stat | $\overline{IC} / s_{IC} \cdot \sqrt{n}$ on **non-overlapping** periods | Use one date every H days |
| PCA | Eigenvectors of $\mathrm{cov}(X)$, sorted by eigenvalue | `eigen(cov(X))` (ascending, so reverse it) |
| Out-of-sample R² | $1 - \sum(y - \hat y)^2 / \sum y^2$ | Measured against predicting 0 |

### Avoiding look-ahead and other traps

- **Use only past data in features.** For example, an asset's "average return" at date t uses returns up to t only.
- **Purge the gap between train and test.** If the target looks H days ahead, drop the last H training rows.
- **Overlapping targets inflate t-stats.** Daily rows with 21-day targets share 20 of their 21 days. Evaluate on non-overlapping dates.
- **Scale with training statistics only,** never with statistics from the test set.
- **An intercept can carry a regime into the test.** With excess-return targets, predict 0 as the baseline.
- **If ridge picks the largest λ in the grid,** the features have no usable signal.
- **Check that a result survives a longer sample.** One episode (such as 2022 for bonds) can produce a fake edge.
- **Ask what the cross-section is.** When assets share one big factor (PC1 for bonds is rates), rank them against each other instead of predicting the direction.

## Economics with FRED ([08_economics.jl](08_economics.jl))

| What | Formula / series | Julia |
| --- | --- | --- |
| Fetch a series | IDs like `DGS10`, `DFF`, `CPIAUCSL` | `CSV.read(download("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"), DataFrame; missingstring = ["", "."])` |
| Yield curve slope | 10-year − 2-year (`DGS10` − `DGS2`); negative = inverted | `y10 .- y2` after joining on date |
| Inflation (12-month) | $100\,(CPI_t / CPI_{t-12} - 1)$ | `100 .* (c[13:end] ./ c[1:end-12] .- 1)` |
| Real rate | fed funds − inflation | |
| Rates + credit exposure | $r = a + \beta_{rate}\Delta y_{10} + \beta_{credit}\Delta s$; duration $= -\beta$ | `F = [ones(n) dy ds]; β = F \ r` |

- The CSV download needs no API key. If you use an API, keep the key in an environment variable, never in code you share.
- FRED leaves holidays blank or marks them `.`, so drop them before taking differences.
- ICE spread series (such as `BAMLH0A0HYM2`, the high-yield spread) only go back 3 years on FRED.

## Julia and DataFrames idioms

| Task | Code |
| --- | --- |
| Long → wide | `unstack(df, :timestamp, :ticker, :ret)` |
| Wide → long | `stack(df, Not(:timestamp), variable_name = :ticker)` |
| Per-group column | `transform!(groupby(df, :ticker), :p => (x -> x ./ first(x)) => :norm)` |
| Per-group summary | `combine(groupby(df, :ticker), :r => mean => :avg)` |
| Function of the whole sub-frame | `combine(sdf -> (spot = f(sdf),), groupby(df, :ticker))` |
| Join | `innerjoin(a, b, on = [:date, :ticker])` |
| Drop missing values | `dropmissing(df, [:bid, :ask])` |
| Narrow `Vector{Any}` columns | `identity.(df)` |
| Destructure a NamedTuple | `(; R, X, tick) = load_returns()` |
| Reproducible random numbers | `rng = Xoshiro(1); randn(rng, n)` |

- **`UndefVarError: nrow` with an "ambiguity" hint:** two loaded packages export the same name (for example TSFrames and DataFrames). Restart the REPL, or write `using DataFrames: nrow`.
- **Caching:** each `load_*` function reads `data/*.csv` if it exists. Pass `refresh = true` to download again, for example after changing `START` or `TICKERS`.
