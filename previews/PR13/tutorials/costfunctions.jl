using Minuit2
using Distributions
using FHist
using Plots
using Random

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
# Plot the generated data
plot(x, y, yerr=dy, seriestype=:scatter, label="Data")

h2 = Hist2D((xdata, ydata), binedges=(range(a, b, 21), range(0., maximum(ydata), 6)))
plot(h2)
scatter!(xdata, ydata, markersize=2, color=:white)

my_pdf(x, ζ, μ, σ, τ) = ζ * pdf(truncated(Normal(μ, σ), a, b),x) + (1 - ζ) * pdf(truncated(Exponential(τ), a, b), x)

cost = UnbinnedNLL(xmix, my_pdf)

m = Minuit(cost, ζ=0.5, μ=1., σ=0.5, τ=1.,
                 limit_ζ=(0, 1), limit_μ=(0, 2), limit_σ=(0, Inf), limit_τ=(0, Inf))
migrad!(m)

visualize(m)

minos!(m)

draw_mncontour(m, "σ", "τ", cl=1:4)
scatter!([m.values["σ"]], [m.values["τ"]], label="fit")
scatter!([σ], [τ], label="true")

draw_mnprofile(m, "σ")

density(x, s, b, μ, σ, τ) = (s + b, s * pdf(truncated(Normal(μ, σ), a, b),x) + b * pdf(truncated(Exponential(τ), a, b), x))
cost = ExtendedUnbinnedNLL(xmix, density)

m = Minuit(cost; s=300, b=1500, μ=0., σ=0.2, τ=2)
m.limits["s", "b", "σ", "τ"] = (0, Inf)
migrad!(m)

visualize(m)

function my_logpdf(xy, μ, σ, τ)
    x, y = xy
    logpdf(Normal(μ, σ), x) + logpdf(Exponential(τ), y)
end

c = UnbinnedNLL(hcat(xdata, ydata), my_logpdf, log=true)
m = Minuit(c, μ=1, σ=2, τ=2, limit_σ=(0,Inf), limit_τ=(0,Inf))
migrad!(m)

my_cdf(x, ζ, μ, σ, τ) = ζ * cdf(truncated(Normal(μ, σ), a, b),x) + (1 - ζ) * cdf(truncated(Exponential(τ), a, b), x)

h = Hist1D(xmix, nbins=20)
c = BinnedNLL(bincounts(h), binedges(h), my_cdf)
m = Minuit(c, ζ=0.4, μ=1.0, σ=0.2, τ=2.0, limit_ζ=(0, 1), limit_σ=(0, Inf), limit_τ=(0, Inf))
migrad!(m)

visualize(m)

my_pdf(x, ζ, μ, σ, τ) = ζ * pdf(truncated(Normal(μ, σ), a, b),x) + (1 - ζ) * pdf(truncated(Exponential(τ), a, b), x)

c = BinnedNLL(bincounts(h), binedges(h), my_pdf, use_pdf=:approximate)
m = Minuit(c, ζ=0.4, μ=0., σ=0.2, τ=2.0, limit_ζ=(0, 1), limit_σ=(0, Inf), limit_τ=(0, Inf))
migrad!(m)

visualize(m)

integral(x, sig, bkg, μ, σ, τ) = sig * cdf(truncated(Normal(μ, σ), a, b),x) + bkg * cdf(truncated(Exponential(τ), a, b), x)

cost = ExtendedBinnedNLL(bincounts(h), binedges(h), integral)
m = Minuit(cost, sig=500, bkg=800, μ=0, σ=0.2, τ=2, strategy=2)
m.limits["sig", "bkg", "σ", "τ"] = (0, Inf)
migrad!(m)

visualize(m)

my_pdf2(xy, μ, σ, τ) = pdf(Normal(μ, σ),xy[1]) * pdf(Exponential(τ), xy[2])

h2 = Hist2D((xdata, ydata), nbins=(20, 20))
c = BinnedNLL(bincounts(h2), binedges(h2), my_pdf2, use_pdf=:approximate)
m = Minuit(c, ζ=0.4, μ=1., σ=2., τ=2., limit_σ=(0, Inf), limit_τ=(0, Inf))
migrad!(m)

# Define the model
model(x, a, b) = a + b * x^2

# Define the data and truth
truth = 1, 2
x = range(0, 1., 20)
yt = model.(x, truth...)
ye = 0.4 .* x.^5 .+ 0.1
y = yt + ye .* randn(length(x))

# Plot with error bars
plot(x, y, yerr=ye, seriestype=:scatter, label="Data")
plot!(x, yt, label="Truth", linestyle=:dash)

c = LeastSquares(x, y, ye, model)
m1 = Minuit(c, a=0, b=0)
migrad!(m1)
visualize(m1)

m1.parameters

function model2(xy, a, bx, by)
    x, y = xy
    return a + bx * x + by * y
end

function model2_grad(xy, a, bx, by)
    x, y = xy
    return [1, x, y]
end

# generate a regular grid in x and y
xy = [(x,y) for x in range(-1.,1.,10) for y in range(-1.,1.,10)]

# model truth
zt = model2.(xy, 1, 2, 3)

zerror = 1.
z = zt .+ zerror .* randn(length(xy))

scatter(xy, zcolor=z)

c2 = LeastSquares(xy, z, zerror, model2)
m2 = Minuit(c2, 0, 0, 0)
migrad!(m2)

heatmap(range(-1.,1.,100), range(-1.,1.,100), (x,y)->model2((x,y), m2.values...))
scatter!(xy, zcolor=z)

c2 = LeastSquares(xy, z, zerror, model2, grad=model2_grad)
m2 = Minuit(c2, 0, 0, 0)
migrad!(m2)

c.y[4] = 3.0   # Generate an outlier
migrad!(m1)
visualize(m1)
plot!(x, yt, label="Truth", linestyle=:dash)

m1.parameters

mask = c.y .!= 3.0
c.mask = mask
migrad!(m1)
visualize(m1)
plot!(x, yt, label="Truth", linestyle=:dash)

c.mask = nothing
c.loss = :soft_l1
migrad!(m1)
visualize(m1)
plot!(x, yt, label="Truth", linestyle=:dash)

m1.parameters
