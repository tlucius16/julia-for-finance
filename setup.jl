# Run once to install everything the quant-finance chapters (01-08) need.
using Pkg

Pkg.add(["DataFrames", "Dates", "CSV", "Statistics", "LinearAlgebra", "Random", "Distributions",
         "Plots", "StatsPlots", "YFinance", "Optim", "JuMP", "HiGHS"])
