using Revise
using Minuit2
using Plots
using Distributions
using FHist
using FiniteDiff: finite_difference_gradient

const t_μ, t_σ = 1.0, 0.1
xdata = rand(Normal(t_μ, t_σ), 1000);

#h = Hist1D(xdata, binedges=range(.5, 1.5, 51))

xe = [0.5 , 0.52, 0.54, 0.56, 0.58, 0.6 , 0.62, 0.64, 0.66, 0.68, 0.7 ,
       0.72, 0.74, 0.76, 0.78, 0.8 , 0.82, 0.84, 0.86, 0.88, 0.9 , 0.92,
       0.94, 0.96, 0.98, 1.  , 1.02, 1.04, 1.06, 1.08, 1.1 , 1.12, 1.14,
       1.16, 1.18, 1.2 , 1.22, 1.24, 1.26, 1.28, 1.3 , 1.32, 1.34, 1.36,
       1.38, 1.4 , 1.42, 1.44, 1.46, 1.48, 1.5 ]
nx = [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0,  5,  3,  5, 11, 10, 20,
29, 35, 55, 50, 60, 72, 81, 74, 93, 73, 65, 62, 69, 34, 26, 15, 18,
14, 10,  3,  2,  1,  1,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0]

model(x, μ, σ) = cdf(Normal(μ, σ), x)
mgrad(x, μ, σ) = finite_difference_gradient(p -> model(x, p...), [μ, σ])

#ROOT!Minuit2!MnPrint!SetGlobalLevel(3)
cost = BinnedNLL(nx, xe, model, grad=mgrad, use_pdf=:none)
cgrad(cost, μ, σ) = finite_difference_gradient(p -> value(cost, p), [μ, σ])

m = Minuit(cost, μ=0., σ=1., limit_σ=(0, Inf))

migrad!(m)
#println(m)
#=
cost2 = BinnedNLL(h, model)
m2 = Minuit(cost2, μ=0., σ=1., limit_σ=(0, Inf))

migrad!(m2)
#println(m2)
=#
