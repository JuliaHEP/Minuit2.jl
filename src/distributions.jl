using Distributions
using Polynomials
using Random

export ChebyshevDist

#---Extended Distributions.jl with a Chebyshev polynomial distribution------------------------------
struct ChebyshevDist <: ContinuousUnivariateDistribution
    polynomial::ChebyshevT{Float64, :x}
    integral::Float64
    a::Float64
    b::Float64
    function ChebyshevDist(coeffs, a=-1., b=1.)
        polynomial = ChebyshevT(coeffs)
        integral = integrate(polynomial)
        new(polynomial, (integral(1.)-integral(-1.)), a, b)
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
    max = sum(abs, d.polynomial)   # estimate the maximum of the polynomial
    x = rand(rng, Uniform(-1., 1.))
    while rand(rng) > d.polynomial(x)/max
        x = rand(rng, Uniform(-1., 1.))
    end
    return (x*(d.b - d.a) + d.a + d.b)/2
end
#-------------------------------------------------------------------------------------------------- 
