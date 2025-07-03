using Minuit2
using Distributions         # Distributions
using Plots                 # Plotting
using FHist                 # Histogramming
using QuadGK                # Numerical integration

const a, b = (0., 10.) # range of the x-axis

sig1(x, μ, σ1, f_bkg, f_sig1) = (1-f_bkg) * f_sig1 * pdf(truncated(Normal(μ,σ1),a,b),x)
sig2(x, μ, σ2, f_bkg, f_sig1) = (1-f_bkg) * (1-f_sig1) * pdf(truncated(Normal(μ,σ2),a,b),x)
bkg(x, a0, a1, f_bkg) = f_bkg * pdf(ChebyshevDist([1., a0, a1], a, b), x)
model(x, μ, σ1, σ2, a0, a1, f_sig1, f_bkg) = bkg(x, a0, a1, f_bkg) + sig1(x, μ, σ1,f_bkg,f_sig1) + sig2(x, μ, σ2, f_bkg, f_sig1)

@assert quadgk(x -> sig1(x, 5., 1., 0.0, 1.), a, b)[1] ≈ 1.
@assert quadgk(x -> sig2(x, 5., 1., 0.0, 0.0), a, b)[1] ≈ 1.
@assert quadgk(x -> bkg(x, 0.2, 0.5, 1.), a, b)[1] ≈ 1.

f_sig1 = 0.8
f_bkg = 0.5
μ = 5.
σ1 = 0.5
σ2 = 1.
a0 = 0.5
a1 = 0.2
const N = 1000
xsig1 = rand(truncated(Normal(μ,σ1),a,b), Int(round(N*(1-f_bkg)*f_sig1)))
xsig2 = rand(truncated(Normal(μ,σ2),a,b), Int(round(N*(1-f_bkg)*(1-f_sig1))))
xbkg = rand(ChebyshevDist([1., a0, a1], a, b), Int(round(N*f_bkg)))
data = vcat(xsig1, xsig2, xbkg);

h = Hist1D(data, nbins=50)
plot(bincenters(h), bincounts(h), yerr=sqrt.(bincounts(h)), seriestype=:scatter, label="data")
plot!(x -> bkg(x, a0, a1, f_bkg)*N*(b-a)/50, a, b, label="bkg")
plot!(x -> (bkg(x, a0, a1, f_bkg)+sig2(x, μ, σ2, f_bkg, f_sig1))*N*(b-a)/50, a, b, label="sig2+bkg")
plot!(x -> (bkg(x, a0, a1, f_bkg)+sig2(x, μ, σ2, f_bkg, f_sig1)+sig1(x, μ, σ1, f_bkg, f_sig1))*N*(b-a)/50, a, b, label="sig1+sig2+bkg")

cost1 = UnbinnedNLL(data, model)    # Define the cost function

m = Minuit(cost1, μ=5., σ1=0.5, σ2=1., a0=0.5, a1=0.2, f_sig1=0.8, f_bkg=0.5) # Define the minimizer
m.limits["σ1", "σ2"] = (0., Inf) # Set limits for the sigma
m.limits["f_sig1", "f_bkg"] = (0., 1.) # Set limits for the fractions
m = migrad!(m) # Perform the fit

visualize(m)
plot!(x -> model(x, μ, σ1, σ2, a0, a1, f_sig1, f_bkg)* N * (b-a)/50, a, b, label="truth")

sig1_(x, μ, σ1, f_bkg, f_sig1) = N * (1-f_bkg) * f_sig1, N * (1-f_bkg) * f_sig1 * pdf(truncated(Normal(μ,σ1),a,b),x)
sig2_(x, μ, σ2, f_bkg, f_sig1) = N * (1-f_bkg) * (1-f_sig1), N *(1-f_bkg) * (1-f_sig1) * pdf(truncated(Normal(μ,σ2),a,b),x)
bkg_(x, a0, a1, f_bkg) = N * f_bkg, N * f_bkg * pdf(ChebyshevDist([1., a0, a1], a, b), x)

csig1 = ExtendedUnbinnedNLL(xsig1, sig1_)
csig2 = ExtendedUnbinnedNLL(xsig2, sig2_)
cbkg = ExtendedUnbinnedNLL(xbkg, bkg_)

cost2 = csig1 + csig2 + cbkg

m = Minuit(cost2, μ=5., σ1=0.5, σ2=1., a0=0.5, a1=0.2, f_sig1=0.8, f_bkg=0.5)
m.limits["σ1", "σ2"] = (0., Inf)
m.limits["f_sig1", "f_bkg"] = (0., 1.)
m = migrad!(m)

visualize(m)
