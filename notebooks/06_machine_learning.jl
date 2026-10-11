# 06 — Machine learning: which bond ETFs will beat the others next month?
#
#   1. Returns, the market (AGG) and excess returns
#   2. Features: 4 momentum measures × 6 lookback windows = 24, plus EWMA volatility (ch. 03)
#   3. PCA (why compare ETFs with each other), then the cross-sectional setup
#   4. Ridge regression, fit before 2024
#   5. Test after 2024: rank IC and a top-3 rotation strategy
#
# Why cross-sectional? All eight ETFs mostly move with interest rates, so "will TLT go up?"
# is a rates forecast. Momentum is usually about ranking: "will TLT beat the other seven?"
# Subtracting each day's average across the ETFs removes the shared rates move.
#
# Takeaways (train 2017-2023, test 2024 to 2026-10, 32 months):
# - No signal. Test rank IC ≈ -0.01 (t ≈ -0.1); the top-3 rotation made +9.9% vs +10.0% for
#   equal weight and beat it in 16 of 32 months - a coin flip. Momentum alone or vol alone:
#   the same, IC ≈ 0.
# - Sample length matters. Trained on 2020-2023 only, the model leaned hard on low vol and
#   looked promising (IC ≈ 0.11). That was 2022: rate hikes crushed TLT (highest vol) while
#   SHY (lowest) held up. Adding 2017-2019 diluted that one episode and the edge vanished -
#   a result that depends on one year of training data isn't a signal.
# - The largest coefficients now are 1-year relative strength (rel_252, positive) and
#   medium-term Sharpe (shp_63, negative), but they don't help out of sample either.
# - Eight bond ETFs is a small cross-section: one month's ranking is 8 numbers, so even a real
#   signal would take years to show up. Momentum studies use hundreds of assets.

include("common.jl")

const MARKET  = "AGG"                    # aggregate US bond index
const WINDOWS = [7, 14, 21, 63, 126, 252]  # lookbacks in trading days (≈ 1.5 weeks … 1 year)
const H       = 21                       # forecast horizon (≈ 1 month)
const CUTOFF  = Date(2024, 1, 1)

## 1. Returns and the market

function load_market(; refresh = false)
    path = joinpath(DATA_DIR, "market.csv")
    if isfile(path) && !refresh
        return CSV.read(path, DataFrame)
    end
    m = DataFrame(get_prices(MARKET, startdt = START, enddt = STOP))
    save_output(m, "market")
    m
end

(; R, tick) = load_returns()
mkt = load_market()
mkt_ret = DataFrame(timestamp = mkt.timestamp[2:end], mret = diff(log.(mkt.adjclose)))
R = innerjoin(R, mkt_ret, on = :timestamp)   # keep days where both exist
X = Matrix{Float64}(R[:, tick])              # T × N daily log returns
xm = R.mret                                  # market daily log returns
dates = Date.(R.timestamp)
T, N = size(X)

## 2. Features
# Excess return = return minus the asset's own average return. The average at date t uses
# only returns up to t, so no feature sees the future. With r = excess returns in a window:
#   mom: Median(r) / IQR(r)              consistent absolute returns
#   rel: Median(r - r_m) / IQR(r - r_m)  consistent outperformance of the market
#   res: Median(ε) / IQR(ε)              ε = residual of r regressed on r_m (beta removed)
#   shp: Mean(r) / Std(r)                ordinary Sharpe ratio
# Median/IQR are "robust": one huge day barely moves them, unlike mean/std.

iqr(x) = quantile(x, 0.75) - quantile(x, 0.25)
robust(x) = median(x) / iqr(x)
sharpe(x) = mean(x) / std(x)
residuals(y, x) = (b = cov(x, y) / var(x); y .- mean(y) .- b .* (x .- mean(x)))

feature_names = [Symbol(f, "_", w) for w in WINDOWS for f in ("mom", "rel", "res", "shp")]

function features(r, m)   # r, m = excess returns up to today
    vals = Float64[]
    for w in WINDOWS
        rw, mw = r[end-w+1:end], m[end-w+1:end]
        push!(vals, robust(rw), robust(rw .- mw), robust(residuals(rw, mw)), sharpe(rw))
    end
    NamedTuple{Tuple(feature_names)}(vals)
end

# One row per (ETF, day), from the first day with a full 252-day window. fwd = the next H
# days' return; missing at the end of the data where the future isn't known yet.
rows = []
for (j, name) in enumerate(tick), t in maximum(WINDOWS):T
    r = X[1:t, j] .- mean(X[1:t, j])
    m = xm[1:t] .- mean(xm[1:t])
    fwd = t + H <= T ? sum(X[t+1:t+H, j]) : missing
    push!(rows, (; ticker = name, t, date = dates[t], fwd, features(r, m)...))
end
panel = DataFrame(rows)

## Risk features from chapter 03
# risk_features.csv is mostly portfolio-level (portfolio vol, VaR, breaches): the same number
# for every ETF on a given day, so it can't help rank one ETF against another. The per-ETF
# columns are vol_ewma_<ticker>, each ETF's EWMA volatility forecast. Add that as a feature:
# do riskier ETFs earn more next month than the others (or less - the "low-vol anomaly")?
# It's a forecast made with data up to the day before, so no look-ahead.

risk = CSV.read(joinpath(DATA_DIR, "risk_features.csv"), DataFrame)
vols = stack(select(risk, :timestamp, r"^vol_ewma_"), Not(:timestamp),
             variable_name = :ticker, value_name = :vol)
vols.ticker = replace.(vols.ticker, "vol_ewma_" => "")
vols.date = Date.(vols.timestamp)
panel = innerjoin(panel, select(vols, :date, :ticker, :vol), on = [:date, :ticker])
push!(feature_names, :vol)

## Why cross-sectional? PCA of the returns
# Principal components = the independent directions the eight ETFs move in, ordered by how
# much variance each explains. They're the eigenvectors of the return covariance matrix; the
# eigenvalues are the variance along each one.

pca = eigen(cov(X))                     # eigen sorts ascending, so reverse
explained = reverse(pca.values) ./ sum(pca.values)
loadings = reverse(pca.vectors, dims = 2)
loadings = loadings .* sign.(sum(loadings, dims = 1))   # flip so each mostly points "up"
round.(100 .* explained, digits = 1)    # % of variance per component

# PC1: every ETF moves the same way, TLT (longest duration) the most -> the interest-rate move.
# PC2: credit (EMB, HYG) against Treasuries (TLT) -> the credit-spread move.
# Together they're ~90% of the variance. Subtracting each day's cross-ETF average (below)
# strips out most of PC1, leaving the ETF-specific part that momentum might predict.
DataFrame(ticker = tick, PC1 = loadings[:, 1], PC2 = loadings[:, 2])

## 3. Cross-sectional setup
# Each day: z-score every feature across the ETFs, and subtract the cross-ETF average from
# fwd. Now a feature value of +1 means "1 std above the other ETFs today", and the target
# means "beat the average ETF by this much".

zscore(x) = (x .- mean(x)) ./ std(x)
transform!(groupby(panel, :t), feature_names .=> zscore .=> feature_names,
           :fwd => (f -> f .- mean(f)) => :target)
first(panel, 5)

## 4. Ridge regression
# Minimise Σ(y - Zβ)² + λ|β|². The 25 features overlap heavily (mom and shp at the same
# window are nearly the same signal); ridge shrinks correlated coefficients together instead
# of letting them blow up in opposite directions. λ = 1 is a moderate, untuned choice.
#
# Train stops H days before CUTOFF: those rows' targets would run into the test period.

tc = findfirst(>=(CUTOFF), dates)   # first test day
train = dropmissing(panel[panel.t .< tc - H, :], :target)
test  = dropmissing(panel[panel.t .>= tc, :], :target)

Z = Matrix(train[:, feature_names])
λ = 1.0
β = (Z'Z + λ * nrow(train) * I) \ (Z' * train.target)
test.pred = Matrix(test[:, feature_names]) * β

# Features are z-scores, so each β is the extra monthly return (vs the average ETF) for being
# 1 std above the others on that feature. Shown in basis points.
coefs = DataFrame(feature = string.(feature_names), β_bp = 1e4 .* β)
sort!(coefs, :β_bp)
bar(coefs.β_bp, color = "#2a78d6", lw = 0, legend = false, size = (800, 450),
    xticks = (1:nrow(coefs), coefs.feature), xrotation = 60, ylims = 1.15 .* extrema(coefs.β_bp),
    left_margin = 5Plots.mm, bottom_margin = 5Plots.mm,
    ylabel = "bp per month, per 1 std", title = "Ridge coefficients")

## 5. Test: rank IC
# Rank IC = Spearman correlation between predicted and actual ranks across the ETFs on one day.
# +1 = perfect ranking, 0 = no skill. Use one day per month (every H days) so the 21-day
# targets don't overlap; overlapping months would make the t-stat look far too good.

ranks(x) = invperm(sortperm(x))
months = test[in.(test.t, Ref(tc:H:T)), :]
ic = combine(groupby(months, :date), [:pred, :target] => ((p, y) -> cor(ranks(p), ranks(y))) => :ic)
(mean_IC = mean(ic.ic), t_stat = mean(ic.ic) / std(ic.ic) * sqrt(nrow(ic)), months = nrow(ic))

## Strategy: each month hold the 3 ETFs with the highest predictions
# Compared with holding all eight equally. Earns actual returns; ignores trading costs.

monthly = combine(groupby(months, :date),
    [:pred, :fwd] => ((p, f) -> mean(exp.(f[sortperm(p, rev = true)[1:3]]))) => :top3,
    :fwd => (f -> mean(exp.(f))) => :equal)

plot(monthly.date, cumprod(monthly.top3), label = "top 3 predicted", lw = 2, color = "#2a78d6",
     ylabel = "Growth of \$1", title = "Out-of-sample rotation (from $CUTOFF)", legend = :topleft)
plot!(monthly.date, cumprod(monthly.equal), label = "equal weight", lw = 2, color = "#eb6834")

##
(top3 = prod(monthly.top3) - 1, equal = prod(monthly.equal) - 1,
 months_beaten = mean(monthly.top3 .> monthly.equal))
