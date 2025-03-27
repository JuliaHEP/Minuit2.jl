# # Cost Functions
#
# A quick guide on how to use the built-in cost functions.
#  
# The Minuit2.jl package comes with few of common cost functions. Of course, you can write your own 
# cost functions to use with Minuit2, but most of the cost function is always the same.
# What really varies is the statistical model which predicts the probability density as a function 
# of the parameter values. This tutorial is based on an equivalent one in the `iminuit` package.
# 
# We demonstrate each cost function on a standard example from high-energy physics, the fit of a 
# peak over some smooth background.

#md # !!! note "Note that"
#md #     You can also download this example as a
#md #     [Jupyter notebook](costfunctions.ipynb) and a plain
#md #     [Julia source file](costfunctions.jl).
#
#md # #### Table of contents
#md # ```@contents
#md # Pages = ["costfunctions.md"]
#md # Depth = 2:3
#md # ```

# Load the `Minuit2` module.
# We will also use the `Distributions`, `FHist` and `Plots` modules to define cost functions 
# and display results.

using Minuit2
using Distributions
using FHist
using Plots
using Random

# We generate our data by sampling from a Gaussian peak and from exponential background 
# in the range 0 to 2. The original data is then binned. One can fit the original or the binned data.
Random.seed!(4321)
const N = 1000 
const a, b = 0, 2                           # range of the data
const μ, σ, τ, ζ = 1, 0.1, 1, 0.5           # true values of the parameters    
xdata = rand(Normal(μ, σ), N)               # Normal and Exponential are from Distributions.jl
ydata = rand(Exponential(τ), N)
xmix = vcat(xdata, ydata)                   # mix the data
xmix = xmix[(a .< xmix .< b)]
h = Hist1D(xmix, nbins=20)                  # Hist1D is from FHist.jl
x = bincenters(h)
y = bincounts(h)
dy = sqrt.(y)
## Plot the generated data
plot(x, y, yerr=dy, seriestype=:scatter, label="Data")

# We also generate some 2D data to demonstrate multivariate fits. 
# In this case, a Gaussian along axis 1 and independently an exponential along axis 2. 
# In this case, the distributions are not restricted to some range in x and y.

h2 = Hist2D((xdata, ydata), binedges=(range(a, b, 21), range(0., maximum(ydata), 6)))
plot(h2)
scatter!(xdata, ydata, markersize=2, color=:white)

# ## Maximum-likelihood fits
# Maximum-likelihood fits are the state-of-the-art when it comes to fitting models to data. 
# They can be applied to unbinned and binned data (histograms). 
# 
# Unbinned fits are the easiest to use, because no data binning is needed. 
# They become slow when the sample size is large.
# Binned fits require you to appropriately bin the data. The binning has to be fine enough 
# to retain all essential information. Binned fits are much faster when the sample size is large.

# ### Unbinned fits
# The cost function an unbinned maximum-likelihood fit is really simple, it is the sum of the 
# logarithm of the pdf evaluated at each sample point (times -1 to turn maximization into minimization).
# You can easily write this yourself, but a naive implementation will suffer from instabilities 
# when the pdf becomes locally zero. Our implementation mitigates the instabilities to some extent.
# To perform the unbinned fit you need to provide the pdf of the model.
#
# The `pdf` must be normalized, which means that the integral over the sample value range must be a
# constant for any combination of model parameters. The model pdf in this case is a linear combination 
# of the normal and the exponential pdfs. The parameters are  (the weight),  and  of the normal 
# distribution and  of the exponential. The cost function detects the parameter names.
# 
# It is important to put appropriate limits on the parameters, so that the problem does not become 
# mathematically undefined: $0 < z < 1$, $\sigma > 0$, $\tau > 0$
# 
# In addition, it can be beneficial to use $0 < \mu < 2$, but it is not required. We use the function 
# `truncated` provided by the `Distributions` package,  which normalizes inside the data range (0, 2).

my_pdf(x, ζ, μ, σ, τ) = ζ * pdf(truncated(Normal(μ, σ), a, b),x) + (1 - ζ) * pdf(truncated(Exponential(τ), a, b), x)

cost = UnbinnedNLL(xmix, my_pdf)


# The `Minuit` object is created using the cost function and the initial values limits for the
# parameters. The `migrad!`function is called to minimize the cost function. The results are then
# displayed.
m = Minuit(cost, ζ=0.5, μ=1., σ=0.5, τ=1.,
                 limit_ζ=(0, 1), limit_μ=(0, 2), limit_σ=(0, Inf), limit_τ=(0, Inf))
migrad!(m)

# We can also display the results of the fit graphically with the `visualize` function.
visualize(m)

# And finally, the `minos!` function can be used to calculate the errors on the parameters.
minos!(m)

# We can also see the contour choosing a pair of parameters
draw_mncontour(m, "σ", "τ", cl=1:4)
scatter!([m.values["σ"]], [m.values["τ"]], label="fit")
scatter!([σ], [τ], label="true")

# And the profile
draw_mnprofile(m, "σ")

# #### Extended UnbinnedNLL
# An important variant of the unbinned NLL fit is described by [Roger Barlow, Nucl.Instrum.Meth.A 297 (1990) 496-506](https://inspirehep.net/literature/297773). 
# Use this if both the shape and the integral of the density are of interest. 
# In practice, this is often the case, for example, if you want to estimate a cross-section or yield.

# The model in this case has to return the integral of the density and the density itself (which must be vectorized).
# The parameters in this case are those already discussed in the previous section and in addition `s` (integral of the signal density), 
# `b` (integral of the uniform density). The additional limits are:
# - `s > 0`,
# - `b > 0`

# Compared to the previous case, we have one more parameter to fit.
density(x, s, b, μ, σ, τ) = (s + b, s * pdf(truncated(Normal(μ, σ), a, b),x) + b * pdf(truncated(Exponential(τ), a, b), x))
cost = ExtendedUnbinnedNLL(xmix, density) 

m = Minuit(cost; s=300, b=1500, μ=0., σ=0.2, τ=2)
m.limits["s", "b", "σ", "τ"] = (0, Inf)
migrad!(m)

# visualize the results
visualize(m)

# #### Multivariate fits
# We can also fit a multivariate model to multivariate data. 
# We pass the model as a `logpdf` this time, which works well because the pdfs factorize. 
# The package `Distributions.jl` provides directly the function `logpdf`.

function my_logpdf(xy, μ, σ, τ)
    x, y = xy
    logpdf(Normal(μ, σ), x) + logpdf(Exponential(τ), y)
end

c = UnbinnedNLL(hcat(xdata, ydata), my_logpdf, log=true)
m = Minuit(c, μ=1, σ=2, τ=2, limit_σ=(0,Inf), limit_τ=(0,Inf))
migrad!(m)

# ### Binned fits
# Binned fits are computationally more efficient and numerically more stable when samples are large.
# The caveat is that one has to choose an appropriate binning. The binning should be fine enough so
# that the essential information in the original is retained.
# Using large bins does not introduce a bias, but the parameters have a larger-than-minimal variance.
# In this case, 50 bins are fine enough to retain all information. 
# Using many bins is safe, since the maximum-likelihood method correctly takes Poisson statistics 
# into account, which works even if bins have zero entries. Using more bins than necessary just 
# increases the computational cost.
# 
# Instead of a `pdf``, you need to provide a `cdf`` for a binned fit in order to better calculate
# the probability of the data. The difference of `cdf` at the bin edges is the integral of the 
# `pdf` over the bin range. In this example we use `Hist1D` from FHist.jl to create the histogram
# and the `BinnedNLL` cost function. Other histogram types are also possible.

my_cdf(x, ζ, μ, σ, τ) = ζ * cdf(truncated(Normal(μ, σ), a, b),x) + (1 - ζ) * cdf(truncated(Exponential(τ), a, b), x)

h = Hist1D(xmix, nbins=20)
c = BinnedNLL(bincounts(h), binedges(h), my_cdf)
m = Minuit(c, ζ=0.4, μ=1.0, σ=0.2, τ=2.0, limit_ζ=(0, 1), limit_σ=(0, Inf), limit_τ=(0, Inf))
migrad!(m)

# visualize the results
visualize(m)

# Sometimes the `cdf` is expensive to calculate. In this case, you can approximate taking 
# the `pdf` evaluated at the center of the bin. 
# This can be done with `use_pdf=:approximate` when defining the BinnedNNL cost.

my_pdf(x, ζ, μ, σ, τ) = ζ * pdf(truncated(Normal(μ, σ), a, b),x) + (1 - ζ) * pdf(truncated(Exponential(τ), a, b), x)

c = BinnedNLL(bincounts(h), binedges(h), my_pdf, use_pdf=:approximate)
m = Minuit(c, ζ=0.4, μ=0., σ=0.2, τ=2.0, limit_ζ=(0, 1), limit_σ=(0, Inf), limit_τ=(0, Inf))
migrad!(m)

# visualize the results
visualize(m)

# #### Extended BinnedNLL Fits
# As in the unbinned case, the binned extended maximum-likelihood fit should be used when also 
# the amplitudes of the pdfs are of interest.

# Instead of a density, you need to provide the integrated density in this case (which must be vectorized).
# There is no need to separately return the total integral of the density, 
# like in the unbinned case. The parameters are the same as in the unbinned extended fit.

integral(x, sig, bkg, μ, σ, τ) = sig * cdf(truncated(Normal(μ, σ), a, b),x) + bkg * cdf(truncated(Exponential(τ), a, b), x)

cost = ExtendedBinnedNLL(bincounts(h), binedges(h), integral)
m = Minuit(cost, sig=500, bkg=800, μ=0, σ=0.2, τ=2, strategy=2)
m.limits["sig", "bkg", "σ", "τ"] = (0, Inf)
migrad!(m)

#
visualize(m)

# #### Multi-dimensional Binned fit
# Fitting a multidimensional histogram is easy. Since the `pdfs` in this example factorize, 
# the `cdf` of the 2D model is the product of the `cdfs` along each axis.
my_pdf2(xy, μ, σ, τ) = pdf(Normal(μ, σ),xy[1]) * pdf(Exponential(τ), xy[2])

h2 = Hist2D((xdata, ydata), nbins=(20, 20))
c = BinnedNLL(bincounts(h2), binedges(h2), my_pdf2, use_pdf=:approximate)
m = Minuit(c, ζ=0.4, μ=1., σ=2., τ=2., limit_σ=(0, Inf), limit_τ=(0, Inf))
migrad!(m)

# ## Least-squares fits
# A cost function for a general weighted least-squares fit (aka chi-square fit) is also included. 
# In statistics this is called non-linear regression. In this case you need to provide a model 
# that predicts the y-values as a function of the x-values and the parameters. 
# The fit needs estimates of the y-errors. If those are wrong, the fit may be biased. 

## Define the model
model(x, a, b) = a + b * x^2

## Define the data and truth
truth = 1, 2
x = range(0, 1., 20)
yt = model.(x, truth...)
ye = 0.4 .* x.^5 .+ 0.1
y = yt + ye .* randn(length(x))

## Plot with error bars
plot(x, y, yerr=ye, seriestype=:scatter, label="Data")
plot!(x, yt, label="Truth", linestyle=:dash)

# Define the `LeastSquares` cost function and create the `Minuit` object.

c = LeastSquares(x, y, ye, model)
m1 = Minuit(c, a=0, b=0)
migrad!(m1)
visualize(m1)

# the property `parameters` can be used to get the parameter values and errors.
m1.parameters

# ### Multivariate model fits
# In this case we fit a plane in 2D. The model is a linear combination of the x and y values.
# It is easy to provide the gradient of the model.
function model2(xy, a, bx, by)
    x, y = xy
    return a + bx * x + by * y
end

function model2_grad(xy, a, bx, by)
    x, y = xy
    return [1, x, y]
end

# Lets generate some 2D data with some normal distributed error and plot it. 
## generate a regular grid in x and y
xy = [(x,y) for x in range(-1.,1.,10) for y in range(-1.,1.,10)]

## model truth 
zt = model2.(xy, 1, 2, 3)

zerror = 1.
z = zt .+ zerror .* randn(length(xy))

scatter(xy, zcolor=z)

# Define the `LeastSquares` cost function and create the `Minuit` object.
c2 = LeastSquares(xy, z, zerror, model2)
m2 = Minuit(c2, 0, 0, 0)
migrad!(m2)

# Multivariate fits are difficult to check by eye. Here we use color to indicate the function value.
# To guarantee that plot of the function and the plot of the data use the same color scale.
heatmap(range(-1.,1.,100), range(-1.,1.,100), (x,y)->model2((x,y), m2.values...))
scatter!(xy, zcolor=z)

# Let's use the gradient in a multi-variate
c2 = LeastSquares(xy, z, zerror, model2, grad=model2_grad)
m2 = Minuit(c2, 0, 0, 0)
migrad!(m2)

# ### Robust least-squared
# The built-in least-squares function also supports robust fitting with an alternative loss functions.
# Builtin loss functions are:
# - linear (default): gives ordinary weighted least-squares
# - soft_l1: quadratic ordinary loss for small deviations (<< 1σ), linear loss for large deviations (>> 1σ), 
#   and smooth interpolation in between
#
# Let’s create one outlier and see what happens with ordinary loss.

c.y[4] = 3.0   # Generate an outlier
migrad!(m1)
visualize(m1)
plot!(x, yt, label="Truth", linestyle=:dash)

# 
m1.parameters

# The result is distorted by the outlier. Note that the error did not increase! 
# The size of the error computed by Minuit does **not** include mismodelling.
# We can try first to mask the outlier temporary

mask = c.y .!= 3.0
c.mask = mask
migrad!(m1)
visualize(m1)
plot!(x, yt, label="Truth", linestyle=:dash)

# Alternatively, we can repair this with by switching to `:soft_l1` loss.
c.mask = nothing
c.loss = :soft_l1
migrad!(m1)
visualize(m1)
plot!(x, yt, label="Truth", linestyle=:dash)

# And the parameters are
m1.parameters 