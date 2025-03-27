# # RooFit Modelling
#
# A quick guide on how to use the `RooFit` module to perform fits.
#  
# 

#md # !!! note "Note that"
#md #     You can also download this example as a
#md #     [Jupyter notebook](roofit.ipynb) and a plain
#md #     [Julia source file](roofit.jl).
#
#md # #### Table of contents
#md # ```@contents
#md # Pages = ["roofit.md"]
#md # Depth = 2:3
#md # ```

# Load the `Minuit2` module.
# We will also use the `Distributions`, `FHist` and `Plots` modules to define cost functions 
# and display results.

using Minuit2
using Minuit2.RooFit        # Load the RooFit module
using Plots                 # Plotting    

# ## Define the model
# We define a model with two signal distributions and a background distribution.
# The model is defined as:
#
# $ \text{pdf} = f_{\text{bkg}} \times \text{bkg}(x,c) + (1-f_{\text{bkg}}) \times (f_{\text{sig1}} \times \text{sig1}(x,m_1,s_1) + (1-f_{\text{sig1}}) \times \text{sig2}(x,m_2,s_2)) $
#
# where:
# - `c` is the parameter of the background distribution,
# - `μ1`, `σ1`, 'μ2` and `σ2` are the parameters of the signal distributions,
# - `f_sig1` and `f_bkg` are the fractions of the signal and background distributions.

x =  RealVar(:x, limits=(0., 10.), nbins=20)
μ1 = RooFit.RealVar(:μ1, 3., limits=(0., 5.))
σ1 = RooFit.RealVar(:σ1, .8, limits=(0.5, 3.))
μ2 = RooFit.RealVar(:μ2, 6., limits=(5., 10.))
σ2 = RooFit.RealVar(:σ2, 1., limits=(0.5, 3.))
sig1 = RooFit.Gaussian(:sig1, x, μ1, σ1)
sig2 = RooFit.Gaussian(:sig2, x, μ2, σ2)

f_sig1 = RooFit.RealVar(:f_sig1, 0.5, limits=(0., 1.))
sig = RooFit.AddPdf(:sig, sig1, sig2, f_sig1)

c = RooFit.RealVar(:c, -0.5, limits=(-0.8, -0.2))
bkg = RooFit.Exponential(:bkg, x, c)

f_bkg = RooFit.RealVar(:f_bkg, 0.4, limits=(0., 1.))
model =  RooFit.AddPdf(:model, bkg, sig, f_bkg)

# Lets now generate some data
N = 2000
data = RooFit.generate(model, N);

# ### Fit the data with UnbinnedNLL
# For this example, we will use the `UnbinnedNLL` cost function to fit the data.
# We will use the `Minuit` optimizer to minimize the cost function. The function `minuitkwargs` 
# is used to randomize the initial parameters. We take a uniform distribution between -0.9 and 0.9
# times the parameter limits.

cost = UnbinnedNLL(data, model.pdf)
kwargs = RooFit.minuitkwargs(model, randomize=true)  # Randomize the initial parameters
m = Minuit(cost; kwargs...)
migrad!(m)

# Visualize the results the results and the different components of the model. The `pdf` needs to be
# scaled to the number of events in the data and the bin widths.
nbins = 50
scale = N * (x.limits[2] - x.limits[1]) / nbins

visualize(m, nbins=nbins)
plot!(x-> model.pdf(x, c.value, μ1.value, σ1.value,μ2.value, σ2.value, f_sig1.value, f_bkg.value) * scale, x.limits..., label="truth", linestyle=:dash)
plot!(x-> bkg.pdf(x) * f_bkg.value * scale, x.limits..., label="bkg", linestyle=:dash)
plot!(x-> sig1.pdf(x) * f_sig1.value * (1-f_bkg.value) * scale, x.limits..., label="sig1", linestyle=:dash)
plot!(x-> sig2.pdf(x) * (1-f_sig1.value) * (1-f_bkg.value) * scale, x.limits..., label="sig2", linestyle=:dash)

# ### Fit the data with BinnedNLL
# We do the same but this time using a binned cost function. We generate a histogram with the default 
# number of bins defined in the variable `x`.

N = 2000
data = RooFit.generateBinned(model, N);

# `data`in this case is a `Hist1D` object.
plot(data, label="data")

# and fit the data with the `BinnedNLL` cost function. 

cost = BinnedNLL(data, model.pdf, use_pdf=:approximate)
kwargs = RooFit.minuitkwargs(model, randomize=true)  # Randomize the initial parameters
m = Minuit(cost; kwargs...)
migrad!(m)

# Visualize the results together with the truth parameters used to generate the data.
scale = N * (x.limits[2] - x.limits[1]) / x.nbins
visualize(m)
plot!(x-> model.pdf(x, c.value, μ1.value, σ1.value,μ2.value, σ2.value, f_sig1.value, f_bkg.value) * scale, x.limits..., label="truth", linestyle=:dash)

