# 07 — Option pricing: Black-Scholes, Greeks, implied volatility, Monte Carlo
#
#   1. Black-Scholes call and put prices
#   2. Greeks (delta, gamma, vega, theta), checked by finite differences
#   3. Implied volatility by Newton's method, compared with Yahoo's numbers (chain from ch. 05)
#   4. Monte Carlo pricing: simulate the stock, average the payoff, compare with Black-Scholes
#
# Black-Scholes assumes the log price is a random walk with constant volatility σ, no
# dividends, a constant risk-free rate r, and continuous trading. None of these hold exactly,
# which is why real option prices need a different σ for each strike (the smile in ch. 05).
#
# Takeaways:
# - Put-call parity holds to machine precision and every Greek matches its finite difference,
#   so the formulas are right. Finite differences are a cheap check for any pricing code.
# - Gamma at the money is ~3× higher at 0.1 years than at 1 year: short-dated options flip
#   from worthless to in-the-money over a small move, so their hedges need constant adjusting.
# - Our implied vols are within ~1.3 vol points of Yahoo's (median). The gaps are input
#   differences: many of these options cost 1.5-10 cents with a 1-cent tick, and a half-cent
#   change in the price moves a days-to-expiry vol by several points.
# - Monte Carlo needs ~1,000,000 paths to get within a cent of Black-Scholes; the error shrinks
#   like 1/√n. Use it when there's no formula, not when there is.

include("common.jl")

const CALL_COLOR = "#2a78d6"   # blue
const PUT_COLOR  = "#eb6834"   # orange
r = 0.04                       # annual risk-free rate (as in ch. 04)

## 1. Black-Scholes prices
#   d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T),   d2 = d1 - σ√T
#   call = S·N(d1) - K·e^(-rT)·N(d2)
#   put  = K·e^(-rT)·N(-d2) - S·N(-d1)
# S = spot, K = strike, T = years to expiry, σ = annual volatility, N = standard normal CDF.

N(x) = cdf(Normal(), x)
n(x) = pdf(Normal(), x)
d1(S, K, T, r, σ) = (log(S / K) + (r + σ^2 / 2) * T) / (σ * sqrt(T))
d2(S, K, T, r, σ) = d1(S, K, T, r, σ) - σ * sqrt(T)

bs_call(S, K, T, r, σ) = S * N(d1(S, K, T, r, σ)) - K * exp(-r * T) * N(d2(S, K, T, r, σ))
bs_put(S, K, T, r, σ)  = K * exp(-r * T) * N(-d2(S, K, T, r, σ)) - S * N(-d1(S, K, T, r, σ))

# Example: at-the-money, 1 year, 20% vol
S, K, T, σ = 100.0, 100.0, 1.0, 0.20
c, p = bs_call(S, K, T, r, σ), bs_put(S, K, T, r, σ)
# Put-call parity (C - P = S - K·e^(-rT)) must hold exactly: a check on the formulas
(call = c, put = p, parity_error = (c - p) - (S - K * exp(-r * T)))

## Price vs spot
# Before expiry the price sits above the payoff (max(S-K, 0)): that gap is time value,
# biggest at the money where the option is most uncertain to finish in or out.

Ss = 60:0.5:140
plot(Ss, bs_call.(Ss, K, T, r, σ), label = "call, 1 year", lw = 2, color = CALL_COLOR,
     xlabel = "Spot", ylabel = "Option price", title = "Black-Scholes call, K = $K, σ = $σ")
plot!(Ss, bs_call.(Ss, K, 0.1, r, σ), label = "call, 0.1 year", lw = 2, color = CALL_COLOR, ls = :dash)
plot!(Ss, max.(Ss .- K, 0), label = "payoff at expiry", lw = 1, color = :gray)

## 2. Greeks: how the price moves when one input changes
#   delta = ∂C/∂S = N(d1)                     put: N(d1) - 1
#   gamma = ∂²C/∂S² = n(d1) / (S σ √T)         same for puts
#   vega  = ∂C/∂σ = S n(d1) √T                  same for puts
#   theta = ∂C/∂T (time decay, per year)       written here as the drop per year that passes

delta(S, K, T, r, σ) = N(d1(S, K, T, r, σ))
gamma(S, K, T, r, σ) = n(d1(S, K, T, r, σ)) / (S * σ * sqrt(T))
vega(S, K, T, r, σ)  = S * n(d1(S, K, T, r, σ)) * sqrt(T)
theta(S, K, T, r, σ) = -S * n(d1(S, K, T, r, σ)) * σ / (2 * sqrt(T)) -
                       r * K * exp(-r * T) * N(d2(S, K, T, r, σ))

# Check each formula against a finite difference: nudge one input, reprice, divide.
h = 1e-4
DataFrame(greek = ["delta", "gamma", "vega", "theta"],
    formula = [delta(S, K, T, r, σ), gamma(S, K, T, r, σ), vega(S, K, T, r, σ), theta(S, K, T, r, σ)],
    finite_difference = [
        (bs_call(S + h, K, T, r, σ) - bs_call(S - h, K, T, r, σ)) / 2h,
        (bs_call(S + h, K, T, r, σ) - 2c + bs_call(S - h, K, T, r, σ)) / h^2,
        (bs_call(S, K, T, r, σ + h) - bs_call(S, K, T, r, σ - h)) / 2h,
        -(bs_call(S, K, T + h, r, σ) - bs_call(S, K, T - h, r, σ)) / 2h])

## Delta and gamma vs spot
# Delta goes from 0 (far out of the money) to 1 (deep in): it's the number of shares that
# hedges one call. Gamma is how fast delta changes - it peaks at the money, and sharpens as
# expiry approaches, which is why short-dated at-the-money options are hard to hedge.

p1 = plot(Ss, delta.(Ss, K, T, r, σ), label = "1 year", lw = 2, color = CALL_COLOR, title = "Delta",
          xlabel = "Spot", legend = :topleft)
plot!(p1, Ss, delta.(Ss, K, 0.1, r, σ), label = "0.1 year", lw = 2, color = CALL_COLOR, ls = :dash)
p2 = plot(Ss, gamma.(Ss, K, T, r, σ), label = "1 year", lw = 2, color = CALL_COLOR, title = "Gamma",
          xlabel = "Spot")
plot!(p2, Ss, gamma.(Ss, K, 0.1, r, σ), label = "0.1 year", lw = 2, color = CALL_COLOR, ls = :dash)
plot(p1, p2, layout = (1, 2), size = (1000, 400), bottom_margin = 5Plots.mm)

## 3. Implied volatility
# The market quotes a price; implied vol is the σ that makes Black-Scholes match it. There's
# no formula for it, so solve numerically. Newton's method: σ_new = σ - (BS(σ) - price) / vega,
# i.e. follow the slope (vega) to where the pricing error is zero. Usually 3-5 steps.

function implied_vol(price, S, K, T, r, type; σ = 0.2)
    bs = type == "call" ? bs_call : bs_put
    for _ in 1:50
        err = bs(S, K, T, r, σ) - price
        abs(err) < 1e-8 && return σ
        σ = max(σ - err / vega(S, K, T, r, σ), 1e-4)   # vol can't go negative
    end
    NaN   # didn't converge
end

# Round trip: price an option at 20% vol, recover the vol from the price
implied_vol(bs_call(S, K, T, r, 0.2), S, K, T, r, "call", σ = 0.5)

## Implied vol on the real chain from ch. 05
# Same cleaning and implied spot as ch. 05. Out-of-the-money options only: in-the-money
# prices are mostly intrinsic value, so a tiny price error becomes a huge vol error. Strikes
# within ±10% of spot: far strikes have stale quotes that no σ can match.

opts = load_options_data()
clean = dropmissing(opts, [:bid, :ask, :impliedVolatility])
filter!(row -> row.bid > 0 && row.ask > row.bid, clean)
clean.mid = (clean.bid .+ clean.ask) ./ 2
spots = combine(sdf -> (spot = implied_spot(sdf),), groupby(clean, :ticker))
clean = dropmissing(leftjoin(clean, spots, on = :ticker), :spot)
otm = filter(row -> (row.type == "put") == (row.strike < row.spot) && abs(row.strike / row.spot - 1) <= 0.1, clean)

# Years to expiry, from the download time to the expiration date
otm.T = Dates.value.(otm.expiration .- otm.asof) ./ (1000 * 60 * 60 * 24 * 365)
otm.iv = implied_vol.(otm.mid, otm.spot, otm.strike, otm.T, r, otm.type)
select(otm, :ticker, :type, :strike, :spot, :mid, :T, :iv, :impliedVolatility)

## Our implied vol vs Yahoo's
# Points on the diagonal agree. Differences come from inputs, not the method: Yahoo uses its
# own price (not necessarily our bid-ask mid), its own rate, time to expiry and spot. With
# only days to expiry, small input differences move the vol a lot.

ok = otm[isfinite.(otm.iv), :]
lim = (0, 1.05 * maximum(vcat(ok.iv, ok.impliedVolatility)))
scatter(ok.impliedVolatility, ok.iv, group = ok.type, color = [CALL_COLOR PUT_COLOR], ms = 4, msw = 0,
        xlabel = "Yahoo implied vol", ylabel = "Our implied vol", xlims = lim, ylims = lim,
        title = "Implied vol: ours vs Yahoo", legend = :topleft, aspect_ratio = 1, size = (550, 500))
plot!(collect(lim), collect(lim), color = :gray, ls = :dash, label = "agree")

## 4. Monte Carlo pricing
# Under Black-Scholes the stock at expiry is S_T = S·exp((r - σ²/2)T + σ√T·Z), Z ~ Normal(0,1).
# Simulate many S_T, average the payoff, discount: price ≈ e^(-rT)·mean(max(S_T - K, 0)).
# It needs no formula, so it also prices options Black-Scholes can't (path-dependent payoffs).
# The error shrinks like 1/√n: 100× more paths for 10× more accuracy.

function mc_call(S, K, T, r, σ, n; rng = Xoshiro(1))
    Z = randn(rng, n)
    ST = S .* exp.((r - σ^2 / 2) * T .+ σ * sqrt(T) .* Z)
    payoff = exp(-r * T) .* max.(ST .- K, 0)
    (price = mean(payoff), se = std(payoff) / sqrt(n))   # se = standard error of the mean
end

ns = [10^k for k in 2:6]
mc = DataFrame([(n = n, mc_call(S, K, T, r, σ, n)...) for n in ns])
mc.error = mc.price .- c
mc   # c = the Black-Scholes call price from section 1

## Convergence
# The estimate wanders around the true price, inside the ±2 standard error band, and the
# band narrows as paths are added.

plot(mc.n, mc.price, ribbon = 2 .* mc.se, xscale = :log10, lw = 2, marker = :circle, ms = 4,
     msw = 0, color = CALL_COLOR, fillalpha = 0.2, label = "Monte Carlo ± 2 se",
     xlabel = "Number of paths", ylabel = "Call price", title = "Monte Carlo converges to Black-Scholes")
hline!([c], color = :gray, ls = :dash, label = "Black-Scholes")
