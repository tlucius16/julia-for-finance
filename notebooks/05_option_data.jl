# 05 — Option data: cleaning, implied spot, volatility smile, open interest
#
#   1. Load the chain and drop contracts with no real market
#   2. Implied spot from put-call parity
#   3. Volatility smile (out-of-the-money options only)
#   4. Open interest by strike
#
# The chain is a snapshot (see :asof) of the nearest expiration only, so these plots describe
# one expiry on one day. Re-run with refresh = true for today's market.

##
include("common.jl")

const CALL_COLOR = "#2a78d6"   # blue
const PUT_COLOR  = "#eb6834"   # orange

## 1. Load and clean
# Yahoo includes stale contracts: no bid, or a last trade from days ago. Their implied
# volatilities are junk (deep in-the-money strikes show IVs above 100%), so keep only
# contracts someone is quoting right now and use the bid-ask midpoint as the price.

opts = load_options_data()
first(opts, 10)

clean = dropmissing(opts, [:bid, :ask, :impliedVolatility])
filter!(r -> r.bid > 0 && r.ask > r.bid, clean)
clean.mid = (clean.bid .+ clean.ask) ./ 2
println("kept $(nrow(clean)) of $(nrow(opts)) contracts")
combine(groupby(clean, [:ticker, :type]), nrow => :contracts)

## 2. Implied spot from put-call parity
# C - P = S - K·e^(-rT). With a few days to expiry the discount barely matters, so
# S ≈ K + C - P. Use the strike where call and put prices are closest (near the money),
# where both quotes are tightest. This gives spot at the moment the chain was downloaded,
# which the price file (ending at STOP) can't.

spots = combine(sdf -> (spot = implied_spot(sdf),), groupby(clean, :ticker))
leftjoin!(clean, spots, on = :ticker)
dropmissing!(clean, :spot)
clean.moneyness = clean.strike ./ clean.spot   # K/S: < 1 below spot, > 1 above
spots

## 3. Volatility smile
# Out-of-the-money options carry the information: puts below spot, calls above. (ITM
# options are mostly intrinsic value, so their IV is noisy.) Equity smiles usually skew
# left: downside puts are priced at higher IV because investors pay for crash protection.

otm = filter(r -> (r.type == "put" && r.moneyness < 1) || (r.type == "call" && r.moneyness >= 1), clean)
filter!(r -> 0.8 <= r.moneyness <= 1.2, otm)   # ±20% of spot; the wings are thin
sort!(otm, [:ticker, :strike])

function smile_panel(t)
    d = otm[otm.ticker .== t, :]
    p = plot(title = t, xlabel = "Strike / spot", ylabel = "Implied vol", legend = false,
             ylims = (0, 1.1 * maximum(d.impliedVolatility)),
             yformatter = y -> "$(round(100y, digits = 1))%")
    for (type, color) in (("put", PUT_COLOR), ("call", CALL_COLOR))
        s = d[d.type .== type, :]
        plot!(p, s.moneyness, s.impliedVolatility, color = color, lw = 2,
              marker = :circle, ms = 4, msw = 0, label = type)
    end
    vline!(p, [1.0], color = :gray, ls = :dash, lw = 1, label = "spot")
    p
end

tickers = sort(unique(otm.ticker))
panels = smile_panel.(tickers)
plot!(panels[1], legend = :topright)   # one legend is enough for the grid
plot(panels..., layout = length(panels), size = (1200, 750), left_margin = 5Plots.mm, bottom_margin = 3Plots.mm,
     plot_title = "Volatility smile, expiry $(Date(first(otm.expiration)))")

## 4. Open interest by strike
# Open interest = contracts outstanding. Calls are drawn up and puts down, so you can see
# where positioning piles up (often at round strikes) and whether it leans call or put.

function oi_panel(t)
    d = clean[(clean.ticker .== t) .& (0.8 .<= clean.moneyness .<= 1.2), :]
    c, p = d[d.type .== "call", :], d[d.type .== "put", :]
    w = 0.8 * minimum(diff(sort(unique(d.strike))))   # bar width from the strike spacing
    top, bottom = maximum(c.openInterest; init = 0), maximum(p.openInterest; init = 0)
    pl = plot(title = t, xlabel = "Strike", ylabel = "Open interest", legend = false,
              ylims = (-1.05 * bottom, 1.05 * top),
              yformatter = y -> string(round(Int, abs(y))))
    bar!(pl, c.strike, c.openInterest, bar_width = w, color = CALL_COLOR, lw = 0, label = "call")
    bar!(pl, p.strike, -p.openInterest, bar_width = w, color = PUT_COLOR, lw = 0, label = "put")
    vline!(pl, [first(d.spot)], color = :gray, ls = :dash, lw = 1, label = "spot")
    pl
end

panels = oi_panel.(tickers)
plot!(panels[1], legend = :topright)
plot(panels..., layout = length(panels), size = (1200, 750), left_margin = 5Plots.mm, bottom_margin = 3Plots.mm,
     plot_title = "Open interest (calls up, puts down), strikes within ±20% of spot")

## Put/call open-interest ratio
# Above 1 = more puts than calls outstanding (hedging or bearish); below 1 = call-heavy.

pc = combine(groupby(clean, :ticker),
    [:type, :openInterest] => ((t, oi) -> sum(oi[t .== "put"]) / sum(oi[t .== "call"])) => :put_call_oi)
