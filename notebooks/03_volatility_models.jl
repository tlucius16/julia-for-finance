# 03 — Volatility models: EWMA, GARCH(1,1) and filtered historical simulation
#
#   1. EWMA (RiskMetrics) volatility
#   2. GARCH(1,1) fitted by maximum likelihood
#   3. Filtered historical simulation (FHS)
#   4. Plots: volatility over time, all VaR models
#   5. Backtest every model (Kupiec + Christoffersen)
#   6. Export risk features for machine learning -> data/risk_features.csv
#
# Takeaways:
# - Rolling-window volatility lags shocks and drops them abruptly 250 days later; EWMA and
#   GARCH react within days and decay smoothly.
# - GARCH fit: a ≈ 0.10, b ≈ 0.88, persistence ≈ 0.98 -> a shock's half-life is ~37 days.
# - EWMA with Normal tails has too many breaches (94 vs 70, Kupiec rejects).
# - GARCH gets the count right but *under*-clusters: after a breach it jumps VaR so much
#   that P(breach | breach yesterday) is ~1% instead of 5% (Christoffersen rejects).
# - FHS (GARCH volatility + empirical tails) is the only model passing both tests.
# - The 2022 rolling-VaR breaches looked clustered on the plot but aren't statistically:
#   that was a high-volatility period, not breaches on back-to-back days.

include("common.jl")
using Optim

(; R, X, tick) = load_returns()
T, N = size(X)

w_eq = fill(1 / N, N)
rp = X * w_eq            # equal-weight portfolio, as in chapter 02
idx = win+1:length(rp)
dates = R.timestamp
actual = rp[idx]

μp = mean(rp)
z = quantile(Normal(), 1 - α)

## EWMA (RiskMetrics) volatility
# A rolling window weights the last 252 days equally, so it reacts slowly to a shock and
# then drops it abruptly 252 days later. EWMA gives yesterday the most weight and decays:
#   σ²[t] = λ σ²[t-1] + (1-λ) r[t-1]²      (λ = 0.94 is the RiskMetrics daily standard)
# σ²[t] only uses returns up to t-1, so it is a genuine one-day-ahead forecast.

function ewma_var(r; λ = 0.94, init = var(r[1:win]))
    σ² = similar(r)
    σ²[1] = init
    for t in 2:length(r)
        σ²[t] = λ * σ²[t-1] + (1 - λ) * r[t-1]^2
    end
    σ²
end

σ_ewma = sqrt.(ewma_var(rp))
var_ewma = -(μp .+ z .* σ_ewma[idx])

## GARCH(1,1) volatility
#   σ²[t] = ω + a ε[t-1]² + b σ²[t-1]
# Like EWMA, but ω pulls the variance back toward a long-run level ω / (1 - a - b),
# and a, b are estimated from the data by maximum likelihood instead of fixed.
# Caveat: fitting on the full sample means the backtest below peeks at the future
# through the parameters (the σ² recursion itself still only uses the past).

ε = rp .- μp

function garch_var(ε, ω, a, b)
    σ² = similar(ε)
    σ²[1] = var(ε)
    for t in 2:length(ε)
        σ²[t] = ω + a * ε[t-1]^2 + b * σ²[t-1]
    end
    σ²
end

# Optimize over unconstrained θ and map it to valid parameters: ω > 0, a, b > 0, a + b < 1
function garch_params(θ)
    e = exp.(θ[2:3])
    (ω = exp(θ[1]), a = e[1] / (1 + sum(e)), b = e[2] / (1 + sum(e)))
end

function garch_nll(θ, ε)   # negative Normal log-likelihood
    p = garch_params(θ)
    σ² = garch_var(ε, p.ω, p.a, p.b)
    0.5 * sum(log.(2π .* σ²) .+ ε .^ 2 ./ σ²)
end

θ0 = [log(var(ε) * 0.05), log(0.1), log(8.0)]   # start near a ≈ 0.1, b ≈ 0.8
garch_fit = optimize(θ -> garch_nll(θ, ε), θ0, NelderMead())
gp = garch_params(Optim.minimizer(garch_fit))
(gp..., persistence = gp.a + gp.b, longrun_annual_vol = sqrt(gp.ω / (1 - gp.a - gp.b) * 252))

σ_garch = sqrt.(garch_var(ε, gp.ω, gp.a, gp.b))
var_garch = -(μp .+ z .* σ_garch[idx])

## Filtered historical simulation (GARCH volatility + empirical tails)
# Divide out GARCH volatility to get standardized residuals, which are close to i.i.d. but
# still fat-tailed. Use their empirical quantile (previous `win` days) instead of Normal z.

zres = ε ./ σ_garch
var_fhs = [-(μp + σ_garch[t] * quantile(zres[t-win:t-1], 1 - α)) for t in idx]

## Volatility over time

σ_roll = [std(rp[t-win:t-1]) for t in idx]

plot(dates, sqrt(252) .* σ_ewma, label = "EWMA", lw = 1.5,
     title = "Portfolio volatility (annualized)", ylabel = "Volatility")
plot!(dates, sqrt(252) .* σ_garch, label = "GARCH(1,1)", lw = 1.5)
plot!(dates[idx], sqrt(252) .* σ_roll, label = "252-day rolling", lw = 1.5)

## Compare all VaR models

roll_hist = rolling_var(rp, hist_var_es)
roll_norm = rolling_var(rp, normal_var_es)

plot(dates[idx], actual, label = "portfolio return", color = :gray, alpha = 0.4,
     title = "$(Int(α*100))% VaR models", ylabel = "Daily log return")
plot!(dates[idx], -roll_hist, label = "historical (rolling)", lw = 1.5)
plot!(dates[idx], -var_ewma, label = "EWMA", lw = 1.5)
plot!(dates[idx], -var_garch, label = "GARCH", lw = 1.5)

## Backtest every model (kupiec and christoffersen are in common.jl)

models = ["historical" => roll_hist, "normal" => roll_norm, "EWMA" => var_ewma,
          "GARCH" => var_garch, "FHS" => var_fhs]

backtest = DataFrame([(model = name, kupiec(actual .< -v, α)..., christoffersen(actual .< -v)...)
                      for (name, v) in models])
select(backtest, :model, :breaches, :expected, :pvalue, :P_breach_after_breach, :pvalue_ind)

## Export risk features for machine learning
# One row per day from day win+1 on (where every model has a value). Every column on row t
# is a forecast made with data up to t-1, except port_ret (day t's actual return, the
# natural prediction target) and breach_* (whether day t's return was worse than the VaR).
# Per-ticker EWMA volatilities are included as extra features.
# Leakage note: GARCH parameters (and so vol_garch, var_garch, var_fhs) were fitted on the
# full sample, so they carry a little information from the future.

risk_features = DataFrame(timestamp = dates[idx], port_ret = actual,
                          vol_roll = σ_roll, vol_ewma = σ_ewma[idx], vol_garch = σ_garch[idx])
for (name, v) in models
    risk_features[!, "var_" * lowercase(name)] = v
    risk_features[!, "breach_" * lowercase(name)] = actual .< -v
end
for j in 1:N
    risk_features[!, "vol_ewma_" * tick[j]] = sqrt.(ewma_var(X[:, j]))[idx]
end

save_output(risk_features, "risk_features")
first(risk_features, 5)
