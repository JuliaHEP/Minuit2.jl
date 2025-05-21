using Minuit2
using Random
using Distributions
using Plots
toydata(μ, σ, n) = randn(n) .* σ .+ μ

const μ, σ = 0., 1.
const N = 100
model(x, μ, σ) = pdf(Normal(μ, σ), x)

data = toydata(μ, σ, 100000)
cost = UnbinnedNLL(data, model)

fit_toy(c, μ, σ) = begin
    m = Minuit(c, μ + (rand()-0.5)*0.1, σ + (rand()-0.5)*0.1)
    migrad!(m)
    Tuple(m.values)
end
μs = zeros(N)
σs = zeros(N)

@time Threads.@threads for i in 1:N
    p1, p2 = fit_toy(cost, 0., 1.)
    μs[i] = p1
    σs[i] = p2
end

@time for i in 1:N
    p1, p2 = fit_toy(cost, 0., 1.)
    μs[i] = p1
    σs[i] = p2
end