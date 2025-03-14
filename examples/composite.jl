using Revise
using Minuit2
using Polynomials
using Distributions
using Plots
using QuadGK
using BenchmarkTools
using Random

# pdf = f_bkg * bkg(x,a0,a1) + (1-f_bkg) * (f_sig1 * sig1(x,m,s1 + (1-f_sig1) * sig2(x,m,s2)))

const a, b = (0., 10.) # range of the x-axis

struct ChebyshevDist <: ContinuousUnivariateDistribution
    polynomial::ChebyshevT{Float64, :x}
    integral::Float64
    a::Float64
    b::Float64
    function ChebyshevDist(coeffs, a=-1., b=1.)
        pol = ChebyshevT(coeffs)
        int = integrate(pol)
        new(pol, (int(1.)-int(-1.)), a, b)
    end
end

function Distributions.pdf(d::ChebyshevDist, x::Real)
    x′ = (2x - d.a - d.b) / (d.b - d.a)  
    d.polynomial(x′)/(d.integral*(d.b - d.a)/2)
end
function Distributions.pdf(d::ChebyshevDist, x::AbstractArray{<:Real})
    x′ = (2x .- d.a .- d.b) ./ (d.b - d.a)  
    d.polynomial.(x′)/(d.integral*(d.b - d.a)/2)
end
function Distributions.cdf(d::ChebyshevDist, x::Real)
    x′ = (2x - d.a - d.b) / (d.b - d.a)
    integrate(d.polynomial, -1., x′)/d.integral
end
function Base.rand(rng::AbstractRNG, d::ChebyshevDist)
    x = rand(rng, Uniform(-1, 1))
    integrate(d.polynomial, -1., x)/d.integral*(d.b-d.a) + d.a
end

# 


# Define the background and signal functions
sig1(x,m,s1,bkgfrac,sig1frac) = (1-bkgfrac) * sig1frac * pdf(truncated(Normal(m,s1),a,b),x)
sig2(x,m,s2,bkgfrac,sig1frac) = (1-bkgfrac) * (1-sig1frac) * pdf(truncated(Normal(m,s2),a,b),x)
bkg(x,a0,a1,bkgfrac) = bkgfrac * pdf(ChebyshevDist([1., a0, a1], a, b), x)

# Verify that the functions are normalized
@assert quadgk(x -> sig1(x, 5., 1., 0.0, 1.), a, b)[1] ≈ 1.
@assert quadgk(x -> sig2(x, 5., 1., 0.0, 0.0), a, b)[1] ≈ 1.
@assert quadgk(x -> bkg(x, 0.2, 0.5, 1.), a, b)[1] ≈ 1.

plot(x -> bkg(x,.5, .2, 1.), a, b, label="bkg")

x = rand(ChebyshevDist([1., .5, .2], a, b), 10000)
histogram(x, nbins=100, label="bkg")

# Lets sample the different distributions

rand(ChebyshevDist([1., .5, .2], a, b))


