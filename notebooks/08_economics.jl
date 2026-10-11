# 08 — Economics with FRED: rates, the yield curve, inflation, and what moves bond ETFs
#
#   1. Load series from FRED (Federal Reserve Economic Data)
#   2. Policy and market rates: fed funds, 2-year, 10-year
#   3. The yield curve (10y - 2y) and recessions
#   4. Inflation vs the fed funds rate
#   5. Back to the ETFs: how much of each one is rates, and how much is credit?
#
# No API key or extra package needed: every FRED series can be downloaded as a plain CSV.
#
# Takeaways (data to 2026-10):
# - Fed funds went from ~0% to over 5% in 2022-23; the 2-year led it, the 10-year moved less.
# - The curve (10y - 2y) inverted briefly in Aug 2019 (the 2020 recession followed, though
#   COVID was the cause), then from mid-2022 to Sep 2024 - the longest inversion on record.
#   Positive since, and no NBER recession has followed so far.
# - Inflation peaked at 9.0% in Jun 2022 with fed funds at 1.2%: a real rate near -8%. Fed funds
#   stayed above inflation from mid-2023 through 2025; in 2026 the two are about level.
# - Two factors explain most bond ETF moves: rates and credit. Treasuries and MBB are nearly
#   pure rates (IEF R² ≈ 0.95). HYG is the one to watch: R² ≈ 0.01 on rates alone (ch. 01), but
#   ≈ 0.6 with the credit spread added - credit duration ≈ 2.7 and rate duration ≈ 3.6.

include("common.jl")

## 1. Load FRED series
# Each series has an ID (search fred.stlouisfed.org; it's in the page URL). The CSV link
# below is the same one as the "Download" button on a series page. Missing days (holidays)
# come as blanks or "."; drop them. Cached in data/fred_<ID>.csv like the other downloads.
# (The FredData.jl package also works, but it needs an API key and is unmaintained since
# 2022, which pins the HTTP package to an old version for the whole environment.)

function load_fred(id; refresh = false)
    path = joinpath(DATA_DIR, "fred_$id.csv")
    if isfile(path) && !refresh
        return CSV.read(path, DataFrame)
    end
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=$id&cosd=$START&coed=$STOP"
    d = CSV.read(download(url), DataFrame; missingstring = ["", "."])
    d = dropmissing(DataFrame(date = d[:, 1], value = d[:, 2]))
    d.value = Float64.(d.value)   # USREC comes as integers
    save_output(d, "fred_$id")
    d
end

ff   = load_fred("DFF")        # effective fed funds rate, daily, %
y2   = load_fred("DGS2")       # 2-year Treasury yield, daily, %
y10  = load_fred("DGS10")      # 10-year Treasury yield, daily, %
hy   = load_fred("BAMLH0A0HYM2") # high-yield credit spread over Treasuries, daily, % (last 3 years only)
cpi  = load_fred("CPIAUCSL")   # consumer price index, monthly
rec  = load_fred("USREC")      # 1 in NBER recession months, 0 otherwise

## 2. Policy and market rates
# The Fed sets the overnight rate (fed funds). The market sets the 2- and 10-year yields,
# which reflect where it expects fed funds to go over those horizons. The 2-year follows the
# Fed closely; the 10-year moves less. In 2022 the Fed raised rates from ~0% to over 5%
# in about a year - the fastest hiking cycle in decades, and the cause of the bond crash.

plot(ff.date, ff.value, label = "fed funds", lw = 2, color = "#2a78d6",
     ylabel = "%", title = "US interest rates", legend = :topleft)
plot!(y2.date, y2.value, label = "2-year", lw = 2, color = "#eb6834")
plot!(y10.date, y10.value, label = "10-year", lw = 2, color = "#1baf7a")

## 3. The yield curve
# Normally the 10-year yields more than the 2-year (investors want extra for locking money up
# longer), so 10y - 2y > 0. When it goes negative ("inverted"), the market expects the Fed to
# cut rates - usually because it expects a slowdown. Inversions preceded every US recession
# since the 1970s, with a lag of 6-24 months. The 2022-2024 inversion was the longest on
# record; whether a recession follows is the test of the signal.

curve = innerjoin(rename(y10, :value => :y10), rename(y2, :value => :y2), on = :date)
curve.spread = curve.y10 .- curve.y2

plot(curve.date, curve.spread, label = "10y - 2y", lw = 2, color = "#2a78d6",
     ylabel = "Percentage points", title = "Yield curve slope", legend = :bottomleft)
hline!([0], color = :gray, ls = :dash, label = "")
# Shade recession months (USREC = 1). Since 2016 that's one spell: the 2020 COVID recession.
rec_months = rec.date[rec.value .== 1]
vspan!([first(rec_months), last(rec_months) + Month(1)], color = :gray, alpha = 0.2,
       label = "recession")

## 4. Inflation vs the fed funds rate
# Inflation = % change in prices over 12 months. The Fed targets 2%. When inflation jumped
# in 2021, the Fed waited ("transitory"), then hiked hard in 2022. Real rate = fed funds -
# inflation: deeply negative in 2021-22 (easy money), positive since 2023 (tight).

cpi.inflation = [fill(missing, 12); 100 .* (cpi.value[13:end] ./ cpi.value[1:end-12] .- 1)]
infl = dropmissing(cpi, :inflation)
ff_monthly = combine(groupby(transform(ff, :date => ByRow(firstdayofmonth) => :month), :month),
                     :value => mean => :ff)
m = innerjoin(infl, ff_monthly, on = :date => :month)

plot(m.date, m.inflation, label = "inflation (CPI, 12-month)", lw = 2, color = "#eb6834",
     ylabel = "%", title = "Inflation and the Fed's response", legend = :topleft)
plot!(m.date, m.ff, label = "fed funds", lw = 2, color = "#2a78d6")
hline!([2], color = :gray, ls = :dash, label = "2% target")

## 5. Back to the ETFs: rates vs credit
# Regress each ETF's daily return on the day's change in the 10-year yield and in the
# high-yield credit spread (both in decimal, so 0.01 = 1 percentage point):
#   r = a + β_rate·Δy10 + β_credit·Δspread
# -β_rate is the duration (as in ch. 01), -β_credit the "spread duration". Using both factors
# shows what ch. 01 couldn't: HYG barely reacts to rates but does react to credit. These two
# factors are roughly the first two principal components from ch. 06.
#
# FRED only serves the last 3 years of ICE's spread indexes (licensing), so this section
# covers ~750 days, not the full sample. A full-history alternative is BAA10Y (Baa yield -
# 10-year), but it's a worse credit measure: it also moves when long yields move more than
# the 10-year, which wrongly gives TLT a large "credit" exposure.

(; R, tick) = load_returns()
f = innerjoin(rename(y10, :value => :y10), rename(hy, :value => :spread), on = :date)
f = DataFrame(date = f.date[2:end], dy = diff(f.y10) ./ 100, ds = diff(f.spread) ./ 100)
D = innerjoin(transform(R, :timestamp => ByRow(Date) => :date), f, on = :date)

F = [ones(nrow(D)) D.dy D.ds]   # design matrix: intercept, rate change, spread change
exposures = DataFrame(map(tick) do t
    β = F \ D[:, t]             # least squares
    fitted = F * β
    (ticker = t, rate_duration = -β[2], spread_duration = -β[3],
     R2 = 1 - sum((D[:, t] .- fitted) .^ 2) / sum((D[:, t] .- mean(D[:, t])) .^ 2))
end)
sort!(exposures, :rate_duration)

## Rates vs credit exposure
# Treasuries (SHY, IEF, TLT) and MBB are almost pure rates. HYG and EMB carry the most credit
# risk, and LQD (investment-grade corporates) has both. HYG's R² jumps from ≈ 0.01 with rates
# alone (ch. 01) to ≈ 0.6 once credit is included.

groupedbar(exposures.ticker, [exposures.rate_duration exposures.spread_duration],
           label = ["rates (10y yield)" "credit (high-yield spread)"], color = ["#2a78d6" "#eb6834"],
           lw = 0, ylabel = "Years (% loss per 1pp rise)", title = "What moves each bond ETF",
           ylims = (-0.5, 1.1 * maximum(exposures.rate_duration)),
           legend = :topleft)
