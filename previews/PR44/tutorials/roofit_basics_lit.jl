# # RooFit Basics
#
# Basic functionality: fitting, plotting, toy data generation on one-dimensional PDFs.
#  
# 

#md # !!! note "Note that"
#md #     You can also download this example as a
#md #     [Jupyter notebook](roofit_basics.ipynb) and a plain
#md #     [Julia source file](roofit_basics.jl).
#
#md # #### Table of contents
#md # ```@contents
#md # Pages = ["roofit_basics.md"]
#md # Depth = 2:3
#md # ```

# Load the `Minuit2.RooFit` module and other needed modules.

using Minuit2.RooFit
using Plots
theme(:boxed)

# ## Setup the model

## Declare variables, x, mean and sigma with initial value and allowed range 
x = RealVar(:x, 0.0, limits=(-10, 10), nbins=30)
mean = RealVar(:mean, 1.0, limits=(-10, 10))
sigma = RealVar(:sigma, 1.0, limits=(0.1, 10))

gauss = Gaussian(:gauss, x, mean, sigma)

# ## Plot the model and change the parameter values_v

plot(gauss, title="Gaussian pdf")
gauss.sigma.value = 3.0
plot!(gauss, label="new sigma")

# ## Generate some data
data = generate(gauss, 1000);

# ## Plot the data with the model
plot(data, title="gaussian with data", label="data")
plot!(gauss, label="pdf")

# ## Fit model with the data
result = fitTo(gauss, data)
## Print values of mean and sigma (that now reflect fitted values and errors)
println("Mean is $(gauss.mean.value) ± $(gauss.mean.error))")
println("Sigma is $(gauss.sigma.value) ± $(gauss.sigma.error))")

# ## Plot the fit result

plot(result, title="Gaussian fit")

