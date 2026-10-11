# 02 — Value at Risk and Expected Shortfall
#
#   1. Historical VaR / ES
#   2. Parametric (Normal) VaR / ES
#   3. Monte Carlo VaR (multivariate Normal and Student-t)
#   4. Compare methods across tickers
#   5. Portfolio VaR and diversification
#   6. Rolling VaR and the Kupiec backtest
#
# Takeaways:
# - VaR is similar across methods, but the Normal understates ES for every ticker: real
#   returns have fatter tails, so the days beyond VaR are worse than a Normal predicts.
#   Student-t Monte Carlo gets ES close to historical.
# - Portfolio VaR (≈3.4%) < weighted sum of individual VaRs (≈4.0%): diversification.
# - Rolling historical VaR passes Kupiec (70 breaches vs 70 expected); rolling Normal VaR
#   has too many (84) - the fat-tail problem again.

include("common.jl")

(; R, X, tick) = load_returns()
T, N = size(X)

## Historical VaR / ES
# VaR = loss at the (1-α) quantile of past returns; ES = average loss on days worse than VaR.
# (hist_var_es is defined in common.jl)

hist_var_es(X[:, 1]; α)   # first ticker

## Parametric (Normal) VaR / ES
# Assume returns ~ Normal(μ, σ). Then VaR and ES have closed forms:
#   VaR = -(μ + σ z),   ES = -(μ - σ φ(z) / (1-α)),   where z = Φ⁻¹(1-α)
# (normal_var_es is defined in common.jl)

normal_var_es(X[:, 1]; α)

## Monte Carlo VaR
# Fit a distribution to all assets jointly, simulate many days, then use the historical
# method on the simulated returns. Student-t (ν = 4) has fatter tails than the Normal.
# MvTDist's matrix is a scale, not the covariance: cov = scale * ν/(ν-2), so rescale it.
# Memory: each simulation matrix is nsim × N Float64s (10M × 5 ≈ 400 MB).

Random.seed!(42)
nsim = 10_000_000
μ̂ = vec(mean(X, dims = 1))
Σ̂ = cov(X)
ν = 4

sims_norm = permutedims(rand(MvNormal(μ̂, Σ̂), nsim))                 # nsim × N
sims_t    = permutedims(rand(MvTDist(ν, μ̂, Σ̂ .* (ν - 2) / ν), nsim))

## Compare methods across tickers

methods = [("historical", X,         hist_var_es),
           ("normal",     X,         normal_var_es),
           ("mc_normal",  sims_norm, hist_var_es),
           ("mc_t",       sims_t,    hist_var_es)]

results = DataFrame()
for (method, M, f) in methods, j in 1:N
    push!(results, (ticker = tick[j], method = method, f(M[:, j]; α)...))
end

unstack(results, :ticker, :method, :VaR)   # rows = tickers, columns = methods
unstack(results, :ticker, :method, :ES)

## Portfolio VaR
# Portfolio return each day = X * w. (Strictly, weights combine simple returns, not log
# returns; for daily data the difference is tiny.)

w_eq = fill(1 / N, N)   # equal weights
rp = X * w_eq

port = DataFrame(
    method = ["historical", "normal", "mc_normal", "mc_t"],
    VaR_ES = [hist_var_es(rp; α), normal_var_es(rp; α),
              hist_var_es(sims_norm * w_eq; α), hist_var_es(sims_t * w_eq; α)])
port = select(port, :method, :VaR_ES => AsTable)

# Diversification: portfolio VaR vs. the weighted sum of each asset's own VaR
indiv = [hist_var_es(X[:, j]; α).VaR for j in 1:N]
(portfolio = hist_var_es(rp; α).VaR, sum_of_parts = w_eq' * indiv)

## Rolling VaR and backtesting
# Each day, estimate VaR from the previous `win` days only, then check whether that day's
# return was worse. At 95%, roughly 5% of days should be breaches.

idx = win+1:length(rp)
dates = R.timestamp

roll_hist = rolling_var(rp, hist_var_es)
roll_norm = rolling_var(rp, normal_var_es)
actual = rp[idx]
breach = actual .< -roll_hist

plot(dates[idx], actual, label = "portfolio return", color = :gray, alpha = 0.5,
     title = "Rolling $(Int(α*100))% VaR ($win-day window)", ylabel = "Daily log return")
plot!(dates[idx], -roll_hist, label = "-VaR historical", lw = 2)
plot!(dates[idx], -roll_norm, label = "-VaR normal", lw = 2)
scatter!(dates[idx][breach], actual[breach], label = "breaches", color = :red, ms = 3)

## Kupiec test (defined in common.jl)

kupiec(actual .< -roll_hist, α)
kupiec(actual .< -roll_norm, α)
