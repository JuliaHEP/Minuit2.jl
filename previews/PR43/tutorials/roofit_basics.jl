using Minuit2.RooFit
using Plots
theme(:boxed)

# Declare variables, x, mean and sigma with initial value and allowed range
x = RealVar(:x, 0.0, limits=(-10, 10), nbins=30)
mean = RealVar(:mean, 1.0, limits=(-10, 10))
sigma = RealVar(:sigma, 1.0, limits=(0.1, 10))

gauss = Gaussian(:gauss, x, mean, sigma)

plot(gauss, title="Gaussian pdf")
gauss.sigma.value = 3.0
plot!(gauss, label="new sigma")

data = generate(gauss, 1000);

plot(data, title="gaussian with data", label="data")
plot!(gauss, label="pdf")

result = fitTo(gauss, data)
# Print values of mean and sigma (that now reflect fitted values and errors)
println("Mean is $(gauss.mean.value) ± $(gauss.mean.error))")
println("Sigma is $(gauss.sigma.value) ± $(gauss.sigma.error))")

plot(result, title="Gaussian fit")
