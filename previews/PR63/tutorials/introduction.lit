# # Introduction to Minuit2.jl
#
# This tutorial is based in the `iminuit` introduction [tutorial](https://indico.cern.ch/event/833895/contributions/3577808/attachments/1927550/3191336/iminuit_intro.html) 
# It is a simple example that shows how to interact with the Minuit2 classes.

#md # !!! note "Note that"
#md #     You can also download this example as a
#md #     [Jupyter notebook](introduction.ipynb) and a plain
#md #     [Julia source file](introduction.jl).
#
#md # #### Table of contents
#md # ```@contents
#md # Pages = ["introduction.md"]
#md # Depth = 2:3
#md # ```

# ## Loading the necessary Julia modules
# Load the `Minuit2` module.
# We will also use the `Polynomials` and `Plots` modules to define cost functions 
# display results.
using Minuit2
using Polynomials
using FHist
using Plots

# ## Define some data to fit
# We generate some data points that follow a linear model with some noise.
## let's make a line model
line(x, a, b) = a + x * b

a_true = 1.0
b_true = 2.0

## let's make some data
x = range(0, 1., 10)

## precomputed random numbers from standard normal distribution
z = [-0.49783783, -0.33041722, -1.71800806,  1.60229399,
      1.36682387, -1.15424221, -0.91425267, -0.03395604,
      -1.27611719, -0.7004073 ]

sigma_y = 0.1
y = line.(x, a_true, b_true) + sigma_y * z

## Plot with error bars
plot(x, y, yerr=sigma_y, seriestype=:scatter)

# - Want to estimate parameters (𝑎,𝑏) of line model from data
# - Need a `score`` which is minimal when model best agrees with data
#     - Sum of residuals squared (least-squares method)
#     - Negated sum of log-likelihood values (maximum-likelihood method)
# - MINUIT always minimizes; negate score function to maximize
# - Use `Minuit2` to numerically minimize score as function of model parameters
#
# ## Define a cost function
# We will use the least squares method to define the cost function.
# The cost function is the sum of the residuals squared.

LSQ(a, b) = sum((y - line.(x, a, b)) .^ 2 ./ sigma_y .^ 2)

# ## Create a Minuit object
# Create a `Minuit` object with the cost function and initial parameter values.
# We fix the parameter `a` to an arbitrary value on this first attempt. 

m = Minuit(LSQ; a=2, b=5, fix_a=true,
                error_a=0.1, error_b=0.1,
                errordef=1)

# ## Minimize the cost function
# Minimize the cost function using the `migrad` method

migrad!(m)

# Plot the data and the fitted line

## get parameter values
a_fit, b_fit = m.values

## Plot with error bars
plot(x, y, yerr=sigma_y, seriestype=:scatter)
plot!(x, line.(x, a_fit, b_fit))

# Lets change the initial value of `a` and `b` and run `migrad` again

m = Minuit(LSQ; a=5, b=5, error_a=0.1, error_b=0.1,
                limit_a=(0, Inf), limit_b=(0, 10.), errordef=1)
migrad!(m)

# Plot the data and the fitted line
## get parameter values
a_fit, b_fit = m.values

## plot with error bars
plot(x, y, yerr=sigma_y, seriestype=:scatter)
plot!(x, line.(x, a_fit, b_fit))

# ## Fit of model with flexible number of parameters
# - Sometimes the model has large or variable number of parameters
#   - Example: fit a polynomial of degree 2, 3, 4, ... ?
#   - Minuit2 has alternative interface which passes parameters as `AbstractVector` to the score function
#
# Define a polynomial model

function LSQ_v(par)       # par is a vector of parameters
    pol = Polynomial(par) # for len(par) == 2 this is a line
    sum((y - pol.(x)) .^ 2 ./ sigma_y .^ 2)
end
##  This is the order of coefficients in the polynomial (reverse order in np.polyval)
Polynomial([1,2,3,4])

# Create a Minuit object with the cost function and initial parameter values
m = Minuit(LSQ_v, [5, 5], error=[0.1, 0.1], errordef=1)
migrad!(m)

# names are automatically generated for the parameters, or can be explicitly set
m = Minuit(LSQ_v, [2, 1, 3, 5], error=0.1,
           names=("a", "b", "c", "d"), errordef=1)
migrad!(m)

# Lets plot the data and the fitted polynomial
## get parameter values
par_fit = m.values
pol = Polynomial(par_fit)

## Plot with error bars
plot(x, y, yerr=sigma_y, seriestype=:scatter)
plot!(x, line.(x, a_fit, b_fit), label="pol2")
plot!(x, pol.(x), label="pol4")

# Lets check reduced chi2, goodness-of-fit estimate, should be around 1
m.fval / (length(y) - length(par_fit))

# ## Parameter uncertainties
# - Minuit2 can compute symmetric uncertainty intervals ("Hesse errors")
#    - automatically done during standard minimisation
#    - to make sure you get accurate errors, call hesse!(m) explicitly after migrad!(m)
#    - slow, computation time scales with  𝑁par^2 
# - Minuit2 can also compute asymmetric uncertainty intervals ("Minos errors")
#    - need to explicitly call m.minos()
#    - very slow, computation time scales with  𝑁par^2 

# Call hesse to get parameter errors
hesse!(m)

# Get the covariance matrix
matrix(m)

# Get the correlation matrix
matrix(m, correlation=true)

# ## Call asymmetric uncertainty intervals ("Minos errors")
# Call `minos!` to get parameter errors
minos!(m)

# Get the asymmetric errors
m.minos |> show

# Minos can fail, check messages:
# - "Valid": everything is chipper
# - "At Limit": Minos hit parameter limit before finishing contour
# - "Max FCN": Minos reached call limit before finishing contour
# - "New Min": Minos found a new minimum while scanning
#
# get the minos errors
d = m.minos["d"]
d.lower, d.upper

# Plot the parameters with the asymmetric errors
# plot parameters with errors
v = m.values |> collect
ve = m.errors |> collect
vm = m.minos |> values
vmean = [(e.upper - e.lower) / 2 for e in vm]

npar = length(v)
indices = 1:npar

plot(indices .- 0.05, v, yerr=ve, seriestype=:scatter, label="Hesse errors")
plot!(indices .+ 0.05, v, yerr=vmean, seriestype=:scatter, label="Minos errors")

# ## Built-in plotting
# These are some functions that provide some plotting functionality. 
# There are implemented as an extension module to Minuit2, which is loaded 
# when the module `Plots` is available.

draw_contour(m, "d", "c", bound=4, subtract_min=false)

# Draw the contour plot for the parameters d and c

draw_mncontour(m, "d", "c", cl=1:4)

# Draw the profile plot for the parameter d

draw_profile(m, "d", bound=2)

# Draw the profile plot for the parameter d

draw_mnprofile(m, "d", size=20)

# ## Summary
# In this example we have shown how to use the `Minuit2` module to fit a line model to some data and 
# how to estimate the parameter uncertainties. We have also shown how to fit a polynomial model with
# a flexible number of parameters.