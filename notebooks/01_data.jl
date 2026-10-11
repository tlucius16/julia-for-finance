# 01 — Data: prices, plots and the returns matrix
#
#   1. Download (or load cached) prices
#   2. Plot raw and normalized prices
#   3. Build the T × N matrix of daily log returns used by every later chapter
#   4. Duration: how much each bond ETF moves when the 10-year Treasury yield moves
#
# Takeaway: raw prices can't be compared across tickers - normalize to a common start, and
# work with returns, not prices, for anything statistical.
# Duration from the 10-year yield: TLT ≈ 15, IEF ≈ 7, SHY ≈ 1.4 - but HYG only ≈ 1 (R² ≈ 0.01),
# because credit spreads tend to offset rate moves. These are not all "rates" ETFs equally.

include("common.jl")

## Prices

prices = load_prices(refresh = true)   # load_prices(refresh = true) to download again
first(prices, 5)

## Plotting the stock prices over time

@df prices plot(:timestamp, :adjclose, group = :ticker, title = "Stock Prices Over Time",
                xlabel = "Date", ylabel = "Adj. Closing Price", legend = :topleft)

## Normalize: divide each ticker's prices by its first price so every line starts at 1
# groupby splits by ticker; transform! runs the function on each group's adjclose column.

transform!(groupby(prices, :ticker), :adjclose => (x -> x ./ first(x)) => :norm_adjclose)

@df prices plot(:timestamp, :norm_adjclose, group = :ticker, title = "Normalized Prices (start = 1)",
                xlabel = "Date", ylabel = "Growth of \$1", legend = :topleft)

# Same plot on a log scale: equal percentage moves look the same size for every ticker
@df prices plot(:timestamp, :norm_adjclose, group = :ticker, yscale = :log10,
                title = "Normalized Prices (log scale)", ylabel = "Growth of \$1", legend = :topleft)

## Returns matrix
# Log return = diff(log(price)). load_returns() does this per ticker, then unstacks to wide
# form (one column per ticker) and drops days where any ticker is missing.

(; R, X, tick) = load_returns()
T, N = size(X)

first(R, 5)
DataFrame(ticker = tick, ann_return = 252 .* vec(mean(X, dims = 1)), ann_vol = sqrt(252) .* vec(std(X, dims = 1)))

save_output(R, "returns")   # data/returns.csv for later use (e.g. machine learning)

## Duration from data
# Duration ≈ % price change for a 1 percentage point rise in yields (with the sign flipped):
# duration 7 means a +1% yield move costs about 7%. Estimate it by regressing each ETF's daily
# return on the daily change in the 10-year Treasury yield (^TNX, quoted in percent):
#   r = a + β·Δy   ->   duration ≈ -β = -cov(r, Δy) / var(Δy)
# R² = share of the ETF's daily variance explained by the 10-year yield alone.

tnx = DataFrame(get_prices("^TNX", startdt = START, enddt = STOP))
tnx = tnx[isfinite.(tnx.adjclose), :]   # Yahoo has the odd NaN day
dy = DataFrame(date = Date.(tnx.timestamp[2:end]), dy = diff(tnx.adjclose) ./ 100)
D = innerjoin(transform(R, :timestamp => ByRow(Date) => :date), dy, on = :date)

duration = DataFrame(ticker = tick,
    duration = [-cov(D[:, t], D.dy) / var(D.dy) for t in tick],
    R2 = [cor(D[:, t], D.dy)^2 for t in tick])
sort!(duration, :duration)

# Treasuries line up with maturity: SHY (1-3 yr) < IEF (7-10 yr) < TLT (20+ yr), with high R².
# Credit is different: HYG barely reacts (R² ≈ 0.01). When yields rise because the economy
# is strong, credit spreads usually tighten, offsetting the rate loss - high yield trades
# more like stocks than bonds. SHY shows less than its ~2 stated duration because short
# rates don't move one-for-one with the 10-year.
bar(duration.ticker, duration.duration, color = "#2a78d6", lw = 0, legend = false,
    ylims = (0, 1.1 * maximum(duration.duration)),
    ylabel = "Duration (years)", title = "Duration estimated from the 10-year yield")
