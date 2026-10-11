using Pkg

Pkg.add(["Dates", "Plots", "StatsPlots", "CSV", "YFinance", "DataFrames", "TimeSeries", "TSFrames", "Distributions", "JuMP", "HiGHS", "Optim"])

##
using YFinance
using DataFrames, Dates, Plots, StatsPlots, TimeSeries, TSFrames

## Getting stock prices for multiple tickers and making a DataFrame

tickers = ["HYG", "TQQQ", "EEM", "AMZN", "TSLA"]
prices = get_prices.(tickers,startdt="2020-01-01",enddt="2026-08-01") |> (x-> DataFrame.(x)) |> (x->vcat(x...))

## Plotting the stock prices over time

@df prices plot(:timestamp, :adjclose, group=:ticker, title="Stock Prices Over Time", xlabel="Date", ylabel="Adj. Closing Price", legend=:topleft)

## normalize: divide each ticker's prices by its first price so every line starts at 1

sort!(prices, [:ticker, :timestamp])
transform!(groupby(prices, :ticker), :adjclose => (x -> x ./ first(x)) => :norm_adjclose)

@df prices plot(:timestamp, :norm_adjclose, group=:ticker, title="Normalized Prices (start = 1)", xlabel="Date", ylabel="Growth of \$1", legend=:topleft)

## Returns matrix
using Statistics, LinearAlgebra

rets = combine(groupby(prices, :ticker),
    :timestamp => (t -> t[2:end]) => :timestamp,
    :adjclose  => (p -> diff(log.(p))) => :ret)

R = dropmissing(unstack(rets, :timestamp, :ticker, :ret))
X = Matrix(R[:, Not(:timestamp)])   # T × N matrix of daily log returns

## Historical VaR / ES
# VaR = loss at the (1-α) quantile of past returns; ES = average loss on days worse than VaR.
# Both are reported as positive numbers (a loss of 3% -> 0.03).
using Distributions, Random

α = 0.95
tick = names(R, Not(:timestamp))
T, N = size(X)

function hist_var_es(r; α=0.95)
    q = quantile(r, 1 - α)
    (VaR = -q, ES = -mean(r[r .<= q]))
end

hist_var_es(X[:, 1]; α)   # try one ticker

## Parametric (Normal) VaR
# Assume returns ~ Normal(μ, σ). Then VaR and ES have closed forms:
#   VaR = -(μ + σ z),   ES = -(μ - σ φ(z) / (1-α)),   where z = Φ⁻¹(1-α)

function normal_var_es(r; α=0.95)
    μ, σ = mean(r), std(r)
    z = quantile(Normal(), 1 - α)
    (VaR = -(μ + σ * z), ES = -(μ - σ * pdf(Normal(), z) / (1 - α)))
end

normal_var_es(X[:, 1]; α)

## Monte Carlo VaR
# Fit a distribution to all assets jointly, simulate many days, then use the historical
# method on the simulated returns. Student-t (ν = 4) has fatter tails than the Normal.
# MvTDist's matrix is a scale, not the covariance: cov = scale * ν/(ν-2), so rescale it.

Random.seed!(42)
nsim = 10_000_000
μ̂ = vec(mean(X, dims=1))
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

win = 250
idx = win+1:length(rp)
dates = R.timestamp

roll_hist = [hist_var_es(rp[t-win:t-1]; α).VaR   for t in idx]
roll_norm = [normal_var_es(rp[t-win:t-1]; α).VaR for t in idx]
actual = rp[idx]
breach = actual .< -roll_hist

plot(dates[idx], actual, label="portfolio return", color=:gray, alpha=0.5,
     title="Rolling $(Int(α*100))% VaR ($win-day window)", ylabel="Daily log return")
plot!(dates[idx], -roll_hist, label="-VaR historical", lw=2)
plot!(dates[idx], -roll_norm, label="-VaR normal", lw=2)
scatter!(dates[idx][breach], actual[breach], label="breaches", color=:red, ms=3)

# Kupiec test: is the breach rate consistent with 1-α? Small p-value (< 0.05) -> model rejected.
function kupiec(breaches, α)
    T, x = length(breaches), sum(breaches)
    p, π̂ = 1 - α, x / T
    LR = -2 * ((T - x) * log(1 - p) + x * log(p) - (T - x) * log(1 - π̂) - x * log(π̂))
    (breaches = x, expected = p * T, rate = π̂, LR = LR, pvalue = ccdf(Chisq(1), LR))
end

kupiec(actual .< -roll_hist, α)
kupiec(actual .< -roll_norm, α)

## EWMA (RiskMetrics) volatility
# A rolling window weights the last 250 days equally, so it reacts slowly to a shock and
# then drops it abruptly 250 days later. EWMA gives yesterday the most weight and decays:
#   σ²[t] = λ σ²[t-1] + (1-λ) r[t-1]²      (λ = 0.94 is the RiskMetrics daily standard)
# σ²[t] only uses returns up to t-1, so it is a genuine one-day-ahead forecast.

function ewma_var(r; λ=0.94, init=var(r[1:win]))
    σ² = similar(r)
    σ²[1] = init
    for t in 2:length(r)
        σ²[t] = λ * σ²[t-1] + (1 - λ) * r[t-1]^2
    end
    σ²
end

μp = mean(rp)
z = quantile(Normal(), 1 - α)
σ_ewma = sqrt.(ewma_var(rp))
var_ewma = -(μp .+ z .* σ_ewma[idx])

## GARCH(1,1) volatility
#   σ²[t] = ω + a ε[t-1]² + b σ²[t-1]
# Like EWMA, but ω pulls the variance back toward a long-run level ω / (1 - a - b),
# and a, b are estimated from the data by maximum likelihood instead of fixed.
# Caveat: fitting on the full sample means the backtest below peeks at the future
# through the parameters (the σ² recursion itself still only uses the past).
using Optim

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

plot(dates, sqrt(252) .* σ_ewma, label="EWMA", lw=1.5,
     title="Portfolio volatility (annualized)", ylabel="Volatility")
plot!(dates, sqrt(252) .* σ_garch, label="GARCH(1,1)", lw=1.5)
plot!(dates[idx], [sqrt(252) * std(rp[t-win:t-1]) for t in idx], label="250-day rolling", lw=1.5)

## Compare all VaR models

plot(dates[idx], actual, label="portfolio return", color=:gray, alpha=0.4,
     title="$(Int(α*100))% VaR models", ylabel="Daily log return")
plot!(dates[idx], -roll_hist, label="historical (rolling)", lw=1.5)
plot!(dates[idx], -var_ewma, label="EWMA", lw=1.5)
plot!(dates[idx], -var_garch, label="GARCH", lw=1.5)

# Christoffersen independence test: are breaches clustered? Counts how often a breach follows
# a breach (n11) vs. a normal day (n01). Small p-value -> breaches cluster.
xlogp(n, p) = n == 0 ? 0.0 : n * log(p)   # treats 0 * log(0) as 0

function christoffersen(breaches)
    prev, curr = breaches[1:end-1], breaches[2:end]
    n00 = sum(.!prev .& .!curr); n01 = sum(.!prev .& curr)
    n10 = sum(prev .& .!curr);   n11 = sum(prev .& curr)
    π01, π11 = n01 / (n00 + n01), n11 / (n10 + n11)
    π̂ = (n01 + n11) / (n00 + n01 + n10 + n11)
    LR = -2 * (xlogp(n00 + n10, 1 - π̂) + xlogp(n01 + n11, π̂) -
               xlogp(n00, 1 - π01) - xlogp(n01, π01) - xlogp(n10, 1 - π11) - xlogp(n11, π11))
    (P_breach_after_calm = π01, P_breach_after_breach = π11, LR_ind = LR, pvalue_ind = ccdf(Chisq(1), LR))
end

models = ["historical" => roll_hist, "normal" => roll_norm, "EWMA" => var_ewma,
          "GARCH" => var_garch, "FHS" => var_fhs]

backtest = DataFrame([(model = name, kupiec(actual .< -v, α)..., christoffersen(actual .< -v)...)
                      for (name, v) in models])
select(backtest, :model, :breaches, :expected, :pvalue, :P_breach_after_breach, :pvalue_ind)


## Portfolio optimization: inputs
# Annualize daily mean and covariance (252 trading days). Expected returns estimated from
# history are very noisy - keep that in mind when reading the "optimal" weights below.
using JuMP, HiGHS

μa = 252 .* vec(mean(X, dims=1))
Σa = 252 .* cov(X)
rf = 0.04   # annual risk-free rate, used for Sharpe ratios

port_stats(w; μ=μa, Σ=Σa) = (ret = μ' * w, vol = sqrt(w' * Σ * w), sharpe = (μ' * w - rf) / sqrt(w' * Σ * w))

DataFrame(ticker = tick, ann_return = μa, ann_vol = sqrt.(diag(Σa)))

## Random portfolios
# Long-only random weights (Dirichlet = random vectors that are ≥ 0 and sum to 1).
# The upper-left edge of this cloud is the efficient frontier.

Random.seed!(1)
Wrand = rand(Dirichlet(ones(N)), 100_000)   # N × 10000, one portfolio per column
rand_stats = [port_stats(Wrand[:, k]) for k in 1:size(Wrand, 2)]

scatter([s.vol for s in rand_stats], [s.ret for s in rand_stats],
        zcolor = [s.sharpe for s in rand_stats], ms = 2, msw = 0, label = "random portfolios",
        xlabel = "Annual volatility", ylabel = "Annual return", colorbar_title = "Sharpe")

## Minimum variance: closed form (shorting allowed)
# Minimize w'Σw subject to sum(w) = 1  ->  w = Σ⁻¹1 / (1'Σ⁻¹1). Weights can be negative.

w_mv_short = Σa \ ones(N)
w_mv_short ./= sum(w_mv_short)
DataFrame(ticker = tick, weight = w_mv_short)

## Long-only efficient frontier (quadratic program)
# Minimize variance subject to: weights ≥ 0, sum to 1, and expected return ≥ target.

# `cap` = maximum weight per asset (1.0 = no cap)
function frontier_portfolio(μ, Σ; target = nothing, cap = 1.0)
    n = length(μ)
    m = Model(HiGHS.Optimizer); set_silent(m)
    @variable(m, 0 <= w[1:n] <= cap)
    @constraint(m, sum(w) == 1)
    target === nothing || @constraint(m, μ' * w >= target)
    @objective(m, Min, w' * Σ * w)
    optimize!(m)
    value.(w)
end

w_mv = frontier_portfolio(μa, Σa)   # no target -> long-only minimum variance
targets = range(port_stats(w_mv).ret, maximum(μa), length = 40)
frontier = [port_stats(frontier_portfolio(μa, Σa; target = tg)) for tg in targets]

## Maximum Sharpe ratio
# Maximizing (μ'w - rf) / √(w'Σw) is not convex, but substituting y = w / ((μ - rf)'w) turns it
# into a quadratic program: minimize y'Σy subject to (μ - rf)'y = 1, y ≥ 0. Then w = y / sum(y).

# A cap w ≤ cap becomes y ≤ cap * sum(y), since w = y / sum(y).
function max_sharpe(μ, Σ; rf = rf, cap = 1.0)
    n = length(μ)
    m = Model(HiGHS.Optimizer); set_silent(m)
    @variable(m, y[1:n] >= 0)
    @constraint(m, (μ .- rf)' * y == 1)
    @constraint(m, y .<= cap * sum(y))
    @objective(m, Min, y' * Σ * y)
    optimize!(m)
    value.(y) ./ sum(value.(y))
end

w_ms = max_sharpe(μa, Σa)

## Minimum expected shortfall (Rockafellar-Uryasev linear program)
# Works directly on the historical return scenarios - no Normal assumption. At the optimum,
# ζ equals the portfolio VaR and the objective equals the portfolio ES.

function min_es(X; α = 0.95, cap = 1.0)
    T, n = size(X)
    m = Model(HiGHS.Optimizer); set_silent(m)
    @variable(m, 0 <= w[1:n] <= cap)
    @variable(m, ζ)              # becomes the VaR at the optimum
    @variable(m, u[1:T] >= 0)    # losses beyond ζ
    @constraint(m, sum(w) == 1)
    @constraint(m, [t = 1:T], u[t] >= -X[t, :]' * w - ζ)
    @objective(m, Min, ζ + sum(u) / ((1 - α) * T))
    optimize!(m)
    value.(w)
end

w_es = min_es(X; α)

## Compare portfolios

ports = ["equal" => w_eq, "min var" => w_mv, "max Sharpe" => w_ms, "min ES" => w_es]

summary_tbl = DataFrame([(portfolio = name, port_stats(w)..., hist_var_es(X * w; α)...)
                         for (name, w) in ports])

weights_tbl = DataFrame(ticker = tick)
for (name, w) in ports
    weights_tbl[!, name] = round.(w, digits = 3)
end
weights_tbl

## Plot: frontier with the chosen portfolios

scatter([s.vol for s in rand_stats], [s.ret for s in rand_stats], color = :lightgray, ms = 2, msw = 0,
        label = "random", xlabel = "Annual volatility", ylabel = "Annual return",
        title = "Long-only efficient frontier", legend = :topleft)
plot!([s.vol for s in frontier], [s.ret for s in frontier], lw = 3, color = :black, label = "frontier")
for (name, w) in ports
    s = port_stats(w)
    scatter!([s.vol], [s.ret], ms = 7, label = name)
end
current()

## Plot: weights of each portfolio

groupedbar(first.(ports), reduce(hcat, last.(ports))', bar_position = :stack,
           label = permutedims(tick), ylabel = "Weight", title = "Portfolio weights", legend = :outerright)

## Out-of-sample test
# Everything above is in-sample: weights were chosen using the same data they are judged on.
# Fit on data before the cutoff, then see how each portfolio did afterwards.

cutoff = Date(2024, 1, 1)
is_train = R.timestamp .< cutoff
Xtr, Xte = X[is_train, :], X[.!is_train, :]
μtr, Σtr = 252 .* vec(mean(Xtr, dims=1)), 252 .* cov(Xtr)

oos_ports = ["equal" => w_eq, "min var" => frontier_portfolio(μtr, Σtr),
             "max Sharpe" => max_sharpe(μtr, Σtr), "min ES" => min_es(Xtr; α)]

oos_tbl = DataFrame([(portfolio = name,
                      predicted = port_stats(w; μ = μtr, Σ = Σtr),    # what the training data promised
                      realized  = port_stats(w; μ = 252 .* vec(mean(Xte, dims=1)), Σ = 252 .* cov(Xte)),
                      test_ES   = hist_var_es(Xte * w; α).ES)
                     for (name, w) in oos_ports])
select(oos_tbl, :portfolio, :predicted => ByRow(p -> p.sharpe) => :pred_sharpe,
       :realized => ByRow(r -> r.ret) => :real_ret, :realized => ByRow(r -> r.vol) => :real_vol,
       :realized => ByRow(r -> r.sharpe) => :real_sharpe, :test_ES)

plot(title = "Out-of-sample growth of \$1 (from $cutoff)", ylabel = "Growth of \$1", legend = :topleft)
for (name, w) in oos_ports
    plot!(R.timestamp[.!is_train], exp.(cumsum(Xte * w)), label = name, lw = 2)
end
current()

## Fix 1: weight caps
# The optimizers above went all-in on one asset. A cap (here 40%) forces at least 3 holdings.
# In-sample, the capped frontier must lie below the uncapped one - that's the price of the
# constraint. The hope is that it pays for itself out of sample.

cap = 0.4

# Highest return reachable under the cap: fill the best assets up to `cap` each
function max_return(μ, cap)
    r, left = 0.0, 1.0
    for μi in sort(μ, rev = true)
        x = min(cap, left)
        r += x * μi
        left -= x
    end
    r
end

w_mv_cap = frontier_portfolio(μa, Σa; cap)
targets_cap = range(port_stats(w_mv_cap).ret, max_return(μa, cap), length = 40)
frontier_cap = [port_stats(frontier_portfolio(μa, Σa; target = tg, cap)) for tg in targets_cap]

plot([s.vol for s in frontier], [s.ret for s in frontier], lw = 3, label = "no cap",
     xlabel = "Annual volatility", ylabel = "Annual return", title = "Frontier with a $(Int(cap*100))% cap", legend = :topleft)
plot!([s.vol for s in frontier_cap], [s.ret for s in frontier_cap], lw = 3, label = "cap $(Int(cap*100))%")

## Fix 2: Ledoit-Wolf covariance shrinkage
# The sample covariance S is noisy. Shrink it toward a simple target F = (average variance) * I:
#   Σ_shrunk = δ F + (1-δ) S
# Ledoit & Wolf (2004) give a formula for the δ that minimizes expected estimation error:
# δ grows when S is noisy (the b² term) and shrinks when S is far from F (the d² term).

function ledoit_wolf(X)
    T, n = size(X)
    Y = X .- mean(X, dims = 1)
    S = Y' * Y / T
    F = (tr(S) / n) * I(n)
    d² = sum(abs2, S - F)                                              # distance from S to F
    b̄² = sum(sum(abs2, Y[t, :] * Y[t, :]' - S) for t in 1:T) / T^2    # noise in S
    δ = min(b̄², d²) / d²
    (Σ = δ * F + (1 - δ) * S, δ = δ)
end

lw_fit = ledoit_wolf(X)
Σa_lw = 252 .* lw_fit.Σ
lw_fit.δ   # shrinkage intensity: 0 = keep S, 1 = use F

## Fix 3: risk parity (equal risk contribution)
# Asset i's contribution to portfolio variance is RC_i = w_i (Σw)_i / (w'Σw); they sum to 1.
# Risk parity sets every RC_i = 1/n. It needs no expected returns at all, only Σ.
# Solve: minimize ½ y'Σy - (1/n) Σ log(y_i) over y > 0, then w = y / sum(y).
# (Setting the gradient to zero gives y_i (Σy)_i = 1/n - exactly equal risk contributions.)
# Optimize over x = log(y) so y stays positive automatically.

risk_contrib(w, Σ) = w .* (Σ * w) ./ (w' * Σ * w)

function risk_parity(Σ)
    n = size(Σ, 1)
    f(x) = 0.5 * exp.(x)' * Σ * exp.(x) - sum(x) / n
    y = exp.(Optim.minimizer(optimize(f, zeros(n), LBFGS())))
    y ./ sum(y)
end

w_rp = risk_parity(Σa_lw)

DataFrame(ticker = tick,
          w_equal = w_eq,  risk_equal = risk_contrib(w_eq, Σa),
          w_rp    = w_rp,  risk_rp    = risk_contrib(w_rp, Σa))

groupedbar(tick, hcat(risk_contrib(w_eq, Σa), risk_contrib(w_rp, Σa)), label = ["equal weight" "risk parity"],
           ylabel = "Share of portfolio risk", title = "Risk contributions")

## Compare everything, in-sample and out-of-sample
# build_portfolios estimates everything from the X it is given, so it can be run on the
# full sample (in-sample) or on the training period only (out-of-sample test).

function build_portfolios(X; α = 0.95, cap = cap)
    n = size(X, 2)
    μ, Σ = 252 .* vec(mean(X, dims = 1)), 252 .* cov(X)
    Σlw = 252 .* ledoit_wolf(X).Σ
    ["equal"                => fill(1 / n, n),
     "min var"              => frontier_portfolio(μ, Σ),
     "max Sharpe"           => max_sharpe(μ, Σ),
     "min var (cap+LW)"     => frontier_portfolio(μ, Σlw; cap),
     "max Sharpe (cap+LW)"  => max_sharpe(μ, Σlw; cap),
     "min ES (cap)"         => min_es(X; α, cap),
     "risk parity (LW)"     => risk_parity(Σlw)]
end

# Max drawdown: worst peak-to-trough fall of cumulative wealth
max_drawdown(r) = (c = cumsum(r); 1 - exp(-maximum(accumulate(max, c) .- c)))

function evaluate(ports, X; α = 0.95)
    DataFrame([(portfolio = name,
                ret       = 252 * mean(X * w),
                vol       = sqrt(252) * std(X * w),
                sharpe    = (252 * mean(X * w) - rf) / (sqrt(252) * std(X * w)),
                ES        = hist_var_es(X * w; α).ES,
                max_dd    = max_drawdown(X * w),
                max_w     = maximum(w))
               for (name, w) in ports])
end

ports_in  = build_portfolios(X)
ports_oos = build_portfolios(Xtr)

evaluate(ports_in, X)      # in-sample: fit and judged on the same data
evaluate(ports_oos, Xte)   # out-of-sample: fit before cutoff, judged after

weights_oos = DataFrame(ticker = tick)
for (name, w) in ports_oos
    weights_oos[!, name] = round.(w, digits = 3)
end
weights_oos

plot(title = "Out-of-sample growth of \$1 (from $cutoff)", ylabel = "Growth of \$1", legend = :outerright, size = (900, 450), left_margin = 5Plots.mm)
for (name, w) in ports_oos
    plot!(R.timestamp[.!is_train], exp.(cumsum(Xte * w)), label = name, lw = 2)
end
current()