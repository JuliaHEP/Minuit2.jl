# # Combined Fits
#
# A quick guide on how to combined several fits into a single one using the `Minuit2` package.
#  
# 

#md # !!! note "Note that"
#md #     You can also download this example as a
#md #     [Jupyter notebook](combined.ipynb) and a plain
#md #     [Julia source file](combined.jl).
#
#md # #### Table of contents
#md # ```@contents
#md # Pages = ["combined.md"]
#md # Depth = 2:3
#md # ```

# Load the `Minuit2` module.
# We will also use the `Distributions`, `FHist` and `Plots` modules to define cost functions 
# and display results.

using Minuit2
using Distributions         # Distributions
using DistributionsHEP      # Distributions for HEP
using FHist                 # Histogramming
using QuadGK                # Numerical integration
using Plots                 # Plotting

# ## Define the model
# We define a model with two signal distributions and a background distribution.
# The model is defined as:
#
# $ \text{pdf} = f_{\text{bkg}} \times \text{bkg}(x,a_0,a_1) + (1-f_{\text{bkg}}) \times (f_{\text{sig1}} \times \text{sig1}(x,m,s_1) + (1-f_{\text{sig1}}) \times \text{sig2}(x,m,s_2)) $
#
# where:
# - `a0` and `a1` are the parameters of the background distribution,
# - `μ`, `σ1` and `σ2` are the parameters of the signal distributions,
# - `f_sig1` and `f_bkg` are the fractions of the signal and background distributions.
#
# We define the background and signal functions as follows:

const a, b = (0., 10.) # range of the x-axis

sig1(x, μ, σ1, f_bkg, f_sig1) = (1-f_bkg) * f_sig1 * pdf(truncated(Normal(μ,σ1),a,b),x)
sig2(x, μ, σ2, f_bkg, f_sig1) = (1-f_bkg) * (1-f_sig1) * pdf(truncated(Normal(μ,σ2),a,b),x)
bkg(x, a0, a1, f_bkg) = f_bkg * pdf(Chebyshev([1., a0, a1], a, b), x)
model(x, μ, σ1, σ2, a0, a1, f_sig1, f_bkg) = bkg(x, a0, a1, f_bkg) + sig1(x, μ, σ1,f_bkg,f_sig1) + sig2(x, μ, σ2, f_bkg, f_sig1) 

# Verify that the functions are normalized
@assert quadgk(x -> sig1(x, 5., 1., 0.0, 1.), a, b)[1] ≈ 1.
@assert quadgk(x -> sig2(x, 5., 1., 0.0, 0.0), a, b)[1] ≈ 1.
@assert quadgk(x -> bkg(x, 0.2, 0.5, 1.), a, b)[1] ≈ 1.

# Lets generate some data
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
xbkg = rand(Chebyshev([1., a0, a1], a, b), Int(round(N*f_bkg)))
data = vcat(xsig1, xsig2, xbkg);

# Plot the data and the model
h = Hist1D(data, nbins=50)
plot(bincenters(h), bincounts(h), yerr=sqrt.(bincounts(h)), seriestype=:scatter, label="data")
plot!(x -> bkg(x, a0, a1, f_bkg)*N*(b-a)/50, a, b, label="bkg")
plot!(x -> (bkg(x, a0, a1, f_bkg)+sig2(x, μ, σ2, f_bkg, f_sig1))*N*(b-a)/50, a, b, label="sig2+bkg")
plot!(x -> (bkg(x, a0, a1, f_bkg)+sig2(x, μ, σ2, f_bkg, f_sig1)+sig1(x, μ, σ1, f_bkg, f_sig1))*N*(b-a)/50, a, b, label="sig1+sig2+bkg")

# Define the unbinned NLL cost function for the model
cost1 = UnbinnedNLL(data, model)    # Define the cost function

# Define the minimizer and perform the fit
m = Minuit(cost1, μ=5., σ1=0.5, σ2=1., a0=0.5, a1=0.2, f_sig1=0.8, f_bkg=0.5) # Define the minimizer
m.limits["σ1", "σ2"] = (0., Inf) # Set limits for the sigma
m.limits["f_sig1", "f_bkg"] = (0., 1.) # Set limits for the fractions
m = migrad!(m) # Perform the fit

# Visualize the fit
visualize(m)
plot!(x -> model(x, μ, σ1, σ2, a0, a1, f_sig1, f_bkg)* N * (b-a)/50, a, b, label="truth")

# ## Combination of two signal distributions and a background distribution
# For each sample we define an `ExtendedUnbinnedNLL` cost function since we 
# would like to also fit the number of events for each of the sample, which is 
# equivalent to the fractions for ech sample.
#
# Please note that the `ExtendedUnbinnedNLL` cost function is used to fit the number of events for each sample.
# the model returns two values: the number of events and the value of the pdf at the given point.
# 

sig1_(x, μ, σ1, f_bkg, f_sig1) = N * (1-f_bkg) * f_sig1, N * (1-f_bkg) * f_sig1 * pdf(truncated(Normal(μ,σ1),a,b),x)
sig2_(x, μ, σ2, f_bkg, f_sig1) = N * (1-f_bkg) * (1-f_sig1), N *(1-f_bkg) * (1-f_sig1) * pdf(truncated(Normal(μ,σ2),a,b),x)
bkg_(x, a0, a1, f_bkg) = N * f_bkg, N * f_bkg * pdf(Chebyshev([1., a0, a1], a, b), x)

csig1 = ExtendedUnbinnedNLL(xsig1, sig1_)
csig2 = ExtendedUnbinnedNLL(xsig2, sig2_)
cbkg = ExtendedUnbinnedNLL(xbkg, bkg_)


# Combining the fit is simply done by summing the cost functions
cost2 = csig1 + csig2 + cbkg

# Define the minimizer and perform the fit
m = Minuit(cost2, μ=5., σ1=0.5, σ2=1., a0=0.5, a1=0.2, f_sig1=0.8, f_bkg=0.5)
m.limits["σ1", "σ2"] = (0., Inf)
m.limits["f_sig1", "f_bkg"] = (0., 1.)
m = migrad!(m)

# Visualize the fit. In this case we visualize the fit for each sample
# separately.
visualize(m)
