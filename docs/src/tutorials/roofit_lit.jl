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
theme(:boxed)
# ## RooFit Modelling
# The `RooFit` module is a powerful tool for defining and fitting models to data.
# It is based on the RooFit library from CERN and provides a Julia interface to it.
#
# Lets by something simple. We will define a model with a single Gaussian distribution and fit it to some data.
#
x = RealVar(:x, 0., limits=(-5., 5.)) # Create a RooRealVar for x
μ = RealVar(:μ, 0., limits=(-5., 5.)) # Create a RooRealVar for μ
σ = RealVar(:σ, 1., limits=(0.1, 5.)) # Create a RooRealVar for σ
gaus = Gaussian(:gaus, x, μ, σ) # Create a RooGaussian PDF

# We can just plot the PDF with the default parameters.
visualize(gaus)

# which is equivalent to the following code:
# ```julia
# plot(x->gaus(x), x.limits..., label="gaus")
# ```

# Lets now generate some data from the PDF.

data = generate(gaus, 1000); # Generate 1000 observations from the model PDF

# Lets fit the data with an `UnbinnedNLL` cost function. It returns a `Minuit` object.
m = fitTo(gaus, data)

# Lets now plot the data and the PDF with the fitted parameters.
visualize(m)

# ## Combining several PDFs to create a model
# Here is a first example of model defined in RooFit that is subsequently used for event generation, an
# unbinned maximum likelihood fit and plotting.
#
# ### Define the model
# We define a model with a signal distribution (gaussian) and a background distribution (argus BG) combining 
# them with the number of events in each category.

##---Observable
mes =  RealVar(:mes, limits=(5.20, 5.30), nbins=50)

##---Gaussian signal
sigmean = RealVar(:sigmean, 5.28, limits=(5.20, 5.30))
sigwidth = RealVar(:sigwidth, 0.0027, limits=(0.001, 0.1))
sig = Gaussian(:sig, mes, sigmean, sigwidth)

##---Build Argus background
argpar = RealVar(:argpar, -20.0, limits=(-100., -1.))
argus = ArgusPdf(:argus, mes, ConstVar(:m₀, 5.291), argpar)

##---Build the model
nsig = RealVar(:nsig, 200., limits=(0., 10000.))
nbkg = RealVar(:nbkg, 800., limits=(0., 10000.))
model = AddPdf(:model, [sig, argus], [nsig, nbkg])

##--- Generate a toyMC sample from composite PDF ---
data = generate(model, 2000)

##--- Perform extended NLL fit ---
m = fitTo(model, data)

# ### Visualize the results
# The `visualize` function is used to plot the results of the fit. It takes the `Minuit` object as input
# and plots the data, the model and the fit results.

visualize(m; legend=:topleft)

# We can also visualize the different components of the model overlaid on the data.

visualize(m, model, components=(:sig, :argus); nbins=50, linestyle=:dash, legend=:topleft)

# ## Combining several PDFs to create a model sharing some variables
# We define a model with two signal distributions and a background distribution.
# The model is defined as:
#
# ``\text{pdf} = f_{\text{bkg}} \times \text{bkg}(x,c) + (1-f_{\text{bkg}}) \times (f_{\text{sig1}} \times \text{sig1}(x,m_1,s_1) + (1-f_{\text{sig1}}) \times \text{sig2}(x,m_2,s_2)) ``
#
# where:
# - `c` is the parameter of the background distribution,
# - `μ`, `σ1`, and `σ2` are the parameters of the signal distributions,
# - `f_sig1` and `f_bkg` are the fractions of the signal and background distributions.

## Define the observable
x =  RealVar(:x, limits=(0., 10.), nbins=20)

## Define the two signals with different widths
μ = RooFit.RealVar(:μ, 3., limits=(0., 5.))
σ1 = RooFit.RealVar(:σ1, .3, limits=(0., 3.))
σ2 = RooFit.RealVar(:σ2, 1., limits=(0., 3.))
sig1 = RooFit.Gaussian(:sig1, x, μ, σ1)
sig2 = RooFit.Gaussian(:sig2, x, μ, σ2)

## Define the background as an exponential function
c = RooFit.RealVar(:c, -0.5, limits=(-0.8, -0.2))
bkg = RooFit.Exponential(:bkg, x, c)

## Define the model as the sum of the two signals and the background
## The background is multiplied by a factor f_bkg, and the first signal by f_sig1
f_bkg = RooFit.RealVar(:f_bkg, 0.4, limits=(0., 1.))
f_sig1 = RooFit.RealVar(:f_sig1, 0.5, limits=(0., 1.))
model =  RooFit.AddPdf(:model, [bkg, sig1, sig2], [f_bkg, f_sig1])

# Lets now generate some data
N = 2000
data = RooFit.generate(model, N);

# ### Fit the data with UnbinnedNLL
# For this example, we will use the `UnbinnedNLL` cost function to fit the data.
# We will use the `Minuit` optimizer to minimize the cost function.

m = fitTo(model, data)

# Visualize the results the results and the different components of the model. The `pdf` needs to be
# scaled to the number of events in the data and the bin widths and this is done automatically

visualize(m, model)

# We can also visualize the components of the overall model

visualize(m, model, components=[:bkg, :sig1, :sig2], fill=0, alpha=0.4)


# ### Fit the data with BinnedNLL
# We do the same but this time using a binned cost function. We generate a histogram with the default 
# number of bins defined in the variable `x`.

N = 2000
data = RooFit.generateBinned(model, N);

# The generated `data` in this case is directly an `Hist1D` object.
plot(data, label="data", c=:blue)

# and fit the data with the `BinnedNLL` cost function. 

m = fitTo(model, data)

# Visualize the results together with the components

visualize(m, model, components=[:bkg, :sig1, :sig2], fill=0, alpha=0.4)


