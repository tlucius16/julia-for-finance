# 04 — Portfolio optimization
#
#   1. Inputs: annualized mean returns and covariance
#   2. Random portfolios
#   3. Minimum variance, closed form (shorting allowed)
#   4. Long-only efficient frontier (quadratic program)
#   5. Maximum Sharpe ratio
#   6. Minimum expected shortfall (linear program)
#   7. Compare portfolios + plots
#   8. Out-of-sample test (fit before 2024, judge after)
#   9. Fixes: weight caps, Ledoit-Wolf shrinkage, risk parity
#  10. Compare everything in- and out-of-sample -> data/portfolio_weights.csv
#
# Takeaways:
# - Unconstrained optimizers go all-in: long-only min variance = 100% HYG; max Sharpe fit
#   before 2024 = 100% TSLA.
# - Max Sharpe depends on expected returns, which are very noisy: it promised a Sharpe of
#   0.74 and delivered 0.08 out of sample. Equal weight (no estimates at all) beat it.
# - A 40% cap costs a little in-sample but helped most out of sample: capped min variance had
#   the best out-of-sample Sharpe (≈0.84).
# - Ledoit-Wolf barely mattered here (δ ≈ 0.008): with 5 assets and ~1,650 days the covariance
#   is already well estimated. Shrinkage matters when assets are many relative to days.
# - Risk parity brings every asset to 20% of portfolio risk (TQQQ carries 41% under equal
#   weight) and came second out of sample without using any return estimates.

include("common.jl")
using JuMP, HiGHS, Optim

(; R, X, tick) = load_returns()
T, N = size(X)
w_eq = fill(1 / N, N)

## Inputs
# Annualize daily mean and covariance (252 trading days). Expected returns estimated from
# history are very noisy - keep that in mind when reading the "optimal" weights below.

μa = 252 .* vec(mean(X, dims = 1))
Σa = 252 .* cov(X)
rf = 0.04   # annual risk-free rate, used for Sharpe ratios

port_stats(w; μ = μa, Σ = Σa) = (ret = μ' * w, vol = sqrt(w' * Σ * w), sharpe = (μ' * w - rf) / sqrt(w' * Σ * w))

DataFrame(ticker = tick, ann_return = μa, ann_vol = sqrt.(diag(Σa)))

## Random portfolios
# Long-only random weights (Dirichlet = random vectors that are ≥ 0 and sum to 1).
# The upper-left edge of this cloud is the efficient frontier.

Random.seed!(1)
Wrand = rand(Dirichlet(ones(N)), 100_000)   # N × 100_000, one portfolio per column
rand_stats = [port_stats(w) for w in eachcol(Wrand)]

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
# `cap` = maximum weight per asset (1.0 = no cap; used in the fixes section).

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
# (min var and min ES coincide here, so one marker hides the other)

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
μtr, Σtr = 252 .* vec(mean(Xtr, dims = 1)), 252 .* cov(Xtr)

oos_ports = ["equal" => w_eq, "min var" => frontier_portfolio(μtr, Σtr),
             "max Sharpe" => max_sharpe(μtr, Σtr), "min ES" => min_es(Xtr; α)]

oos_tbl = DataFrame([(portfolio = name,
                      predicted = port_stats(w; μ = μtr, Σ = Σtr),    # what the training data promised
                      realized  = port_stats(w; μ = 252 .* vec(mean(Xte, dims = 1)), Σ = 252 .* cov(Xte)),
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

plot(title = "Out-of-sample growth of \$1 (from $cutoff)", ylabel = "Growth of \$1",
     legend = :outerright, size = (900, 450), left_margin = 5Plots.mm)
for (name, w) in ports_oos
    plot!(R.timestamp[.!is_train], exp.(cumsum(Xte * w)), label = name, lw = 2)
end
current()

## Export weights (fit on data before the cutoff) for later chapters

save_output(weights_oos, "portfolio_weights")
