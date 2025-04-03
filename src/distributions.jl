using Distributions
using SpecialFunctions
using Polynomials
using Random

export ChebyshevDist, ArgusBGDist

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
struct ArgusBGDist <: ContinuousUnivariateDistribution
    m₀::Float64
    c::Float64
    p::Float64
    a::Float64
    b::Float64
    integral::Float64
    function ArgusBGDist(m₀, c, p = 0.5, a = 0., b = m₀)
        integral = _argus_int(b, m₀, c, p) - _argus_int(a, m₀, c, p)
        new(m₀, c, p, a, b, integral)
    end
end

function _argus(m, m₀, c, p)
    m >= m₀ && return 0.0
    m * (1 - (m/m₀)^2)^p * exp(c * (1 - (m/m₀)^2))
end

function _argus_int(m, m₀, c, p)
    m >= m₀ && (m = m₀-1e-10)
    f = (m/m₀)^2 - 1
    w = -(c * f + 0im)^(-p) * m₀^2 * (-f + 0im)^p * gamma(1 + p, c * f + 0im)/2c
    return isreal(w) ? real(w) : zero(m)
end

function Distributions.pdf(d::ArgusBGDist, m::Real)
    (; m₀, c, p) = d
    _argus(m, m₀, c, p) / d.integral
end

function Distributions.cdf(d::ArgusBGDist, m::Real)
    (; m₀, c, p) = d
    (_argus_int(m, m₀, c, p) - _argus_int(d.a, m₀, c, p)) / d.integral
end

function Base.rand(rng::AbstractRNG, d::ArgusBGDist, n::Int64=1)
    (; m₀, c, p, a, b) = d
    max = maximum(_argus.(range(a,b,100), m₀, c, p)) # estimate the maximum
    r = Float64[]
    for i in 1:n
        m = rand(rng, Uniform(a, b))
        while rand(rng) > _argus(m, m₀, c, p)/max
            m = rand(rng, Uniform(a, b))
        end
        push!(r, m)
    end
    return n == 1 ? r[1] : r
end

#-------------------------------------------------------------------------------------------------- 

