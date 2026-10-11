# Shared setup for the quant-finance guide. Each chapter starts with include("common.jl").
#
#   setup.jl                       install packages (run once)
#   01_data.jl                     download prices, plot, build the returns matrix
#   02_var_es.jl                   VaR / ES: historical, parametric, Monte Carlo, backtesting
#   03_volatility_models.jl        EWMA, GARCH(1,1), filtered historical simulation  -> data/risk_features.csv
#   04_portfolio_optimization.jl   frontier, max Sharpe, min ES, caps, shrinkage, risk parity
#
# Prices are downloaded once and cached in data/prices.csv; delete it (or pass refresh = true)
# to download again.

using DataFrames, Dates, CSV, Statistics, LinearAlgebra, Random, Distributions
using Plots, StatsPlots
using YFinance

## Settings

const TICKERS  = ["HYG", "TLT", "LQD", "IEF", "SHY", "EMB", "TIP", "MBB"]
const START    = "2016-01-01"
const STOP     = "2026-10-01"
const DATA_DIR = joinpath(@__DIR__, "data")

α   = 0.95   # VaR / ES confidence level
win = 252    # rolling window length (≈ one trading year)

## Data

function load_prices(; refresh = false)
    path = joinpath(DATA_DIR, "prices.csv")
    if isfile(path) && !refresh
        return CSV.read(path, DataFrame)
    end
    prices = vcat(DataFrame.(get_prices.(TICKERS, startdt = START, enddt = STOP))...)
    sort!(prices, [:ticker, :timestamp])
    save_output(prices, "prices")
    prices
end

# Daily log returns in long form (one row per ticker-day) and wide form (one column per ticker)
function load_returns(; refresh = false)
    prices = load_prices(; refresh)
    rets = combine(groupby(prices, :ticker),
        :timestamp => (t -> t[2:end]) => :timestamp,
        :adjclose  => (p -> diff(log.(p))) => :ret)
    R = dropmissing(unstack(rets, :timestamp, :ticker, :ret))
    X = Matrix{Float64}(R[:, Not(:timestamp)])   # T × N matrix of daily log returns
    tick = names(R, Not(:timestamp))
    (; prices, R, X, tick)
end

save_output(df, name) = (mkpath(DATA_DIR); CSV.write(joinpath(DATA_DIR, "$name.csv"), df))

# Option chains (nearest expiration only) for each ticker. Unlike prices this is a
# snapshot of today's market, so :asof records when it was downloaded.
function load_options_data(; refresh = false)
    path = joinpath(DATA_DIR, "options.csv")
    if isfile(path) && !refresh
        return CSV.read(path, DataFrame)
    end
    asof = now()
    opts = mapreduce(vcat, TICKERS) do t
        chain = get_Options(t, throw_error = true)
        df = vcat(DataFrame(chain["calls"]), DataFrame(chain["puts"]))
        insertcols!(df, 1, :ticker => t, :asof => asof)
    end
    opts = identity.(opts)   # narrow the Vector{Any} columns YFinance builds
    sort!(opts, [:ticker, :type, :expiration, :strike])
    save_output(opts, "options")
    opts
end

function implied_spot(df)
    c = df[df.type .== "call", [:strike, :mid]]
    p = df[df.type .== "put",  [:strike, :mid]]
    both = innerjoin(c, p, on = :strike, renamecols = "_c" => "_p")
    isempty(both) && return missing
    i = argmin(abs.(both.mid_c .- both.mid_p))
    both.strike[i] + both.mid_c[i] - both.mid_p[i]
end


## VaR / ES (both reported as positive losses: a 3% loss -> 0.03)

function hist_var_es(r; α = 0.95)
    q = quantile(r, 1 - α)
    (VaR = -q, ES = -mean(r[r .<= q]))
end

function normal_var_es(r; α = 0.95)
    μ, σ = mean(r), std(r)
    z = quantile(Normal(), 1 - α)
    (VaR = -(μ + σ * z), ES = -(μ - σ * pdf(Normal(), z) / (1 - α)))
end

# VaR for each day t in win+1:end, estimated from the previous `win` days only
rolling_var(r, f; win = win, α = α) = [f(r[t-win:t-1]; α).VaR for t in win+1:length(r)]

## Backtests

# Kupiec: is the breach rate consistent with 1-α? Small p-value (< 0.05) -> model rejected.
function kupiec(breaches, α)
    T, x = length(breaches), sum(breaches)
    p, π̂ = 1 - α, x / T
    LR = -2 * ((T - x) * log(1 - p) + x * log(p) - (T - x) * log(1 - π̂) - x * log(π̂))
    (breaches = x, expected = p * T, rate = π̂, LR = LR, pvalue = ccdf(Chisq(1), LR))
end

xlogp(n, p) = n == 0 ? 0.0 : n * log(p)   # treats 0 * log(0) as 0

# Christoffersen: are breaches clustered? Compares P(breach | breach yesterday) with
# P(breach | calm yesterday). Small p-value -> breaches are not independent.
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
