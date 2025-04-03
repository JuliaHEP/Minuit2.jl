module RooFit

import Distributions: UnivariateDistribution, Exponential as _Exponential, Normal as _Normal, Uniform as _Uniform, pdf, cdf, truncated
import FHist: Hist1D, AbstractHistogram
import StatsBase: wsample
import Minuit2: ArgusBGDist, ChebyshevDist, BinnedNLL, ExtendedBinnedNLL, UnbinnedNLL, ExtendedUnbinnedNLL, CostFunction, Minuit, migrad!
import Base: getproperty, setproperty!, show, isconst, getindex, setindex!


export AbstractPdf, RealVar, ConstVar
export Gaussian, Exponential, ArgusPdf, Chebyshev
export AddPdf, generate, generateBinned
export minuitkwargs, fitTo

#--------------------------------------------------------------------------------------------------
#---AbstractPdf type-------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
"""
    AbstractPdf
Abstract type for all RooFit distributions.
"""
abstract type AbstractPdf end
function getproperty(d::AbstractPdf, name::Symbol)
    if hasfield(typeof(d), name)
        return getfield(d, name)
    else
        # Check if the name is a parameter of the distribution
        idx = findfirst(p -> p.name == name, d.params)
        if !isnothing(idx)
            return d.params[idx]
        elseif hasfield(typeof(d), :pdfs)
            # Check if the name is component distribution 
            idx = findfirst(p -> p.name == name, d.pdfs)
            if !isnothing(idx)
                return d.pdfs[idx]
            end
        end
    end
end
(model::AbstractPdf)(x) = model.pdf(x, (p.value for p in model.params)...)
show(io::IO, d::AbstractPdf) = print(io, "$(nameof(typeof(d))){$(d.name)} PDF with parameters $([p.name for p in d.params])")
function minuitkwargs(d::AbstractPdf; randomize=false)
    function value(p)
        if randomize
            mul = rand(_Uniform(-.9,.9))
            p.value + mul * (mul > 0 ? (p.limits[2] - p.value) : (p.value - p.limits[1]))
        else
            p.value
        end
    end
    merge(
    Dict(p.name => value(p) for p in d.params), 
    Dict(Symbol(:limit_, p.name)  => p.limits for p in d.params))
end
"""
    generateBinned(d::AbstractPdf, n=1000)

Generate `n` random numbers from a binned distribution `d`.
"""
generateBinned(d::AbstractPdf, n=1000) = Hist1D(generate(d, n), binedges=range(d.x.limits..., d.x.nbins+1))

"""
    generate(d::AbstractPdf, n=1000)

Generate `n` random numbers from a distribution `d`.
"""
generate(d::AbstractPdf, n::Integer) = rand(d.distribution, n)
generate(d::AbstractPdf) = rand(d.distribution)
"""
    fitTo(d::AbstractPdf, data)

Fit a distribution `d` to a data set `data`.
"""
function fitTo(d::AbstractPdf, data)
    # Check if the data is binned or unbinned
    if data isa AbstractHistogram
        if d.extendable
            cost = ExtendedBinnedNLL(data, d.pdf, use_pdf=:approximate)
        else
            cost = BinnedNLL(data, d.pdf, use_pdf=:approximate)
        end
    elseif(data isa AbstractArray)
        if !isnothing(d.extendable) && d.extendable 
            # Need to change the model function to include the integral
            fname = gensym("$(d.name)_ext_pdf")
            pdf = eval( quote
                function $(fname)(x, $((param.name for param in d.params)...))
                    +($((f.name for f in d.fractions)...)), $(d.pdf)(x, $((param.name for param in d.params)...))
                end
            end )
            cost = ExtendedUnbinnedNLL(data, pdf)
        else
            cost = UnbinnedNLL(data, d.pdf)
        end
    else
        throw(ArgumentError("Data type not supported"))
    end
    # Create a Minuit object with the cost function and initial parameter values
    m = Minuit(cost; minuitkwargs(d)...)
    # Run the minimization
    migrad!(m)
    # Check if the fit was successful
    if m.is_valid
        # Update the parameters of the distribution with the fitted values
        for (param, value) in zip(d.params, m.values)
            param.value = value
        end
        # Return the fitted distribution
        return m
    else
        throw(ArgumentError("Fit failed"))
    end
end     

#--------------------------------------------------------------------------------------------------
#---RealVar type-----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
"""
    RealVar{T<:Real}
Entity to represent a real variable with a name, value, limits and number of bins.
"""
mutable struct RealVar{T<:Real}
    const name::Symbol
    value::T
    const limits::Tuple{T, T}
    nbins::Int
    const isconst::Bool
end

"""
    RealVar(name, value=0.;  limits=(-Inf,Inf), nbins=50)

Construct a RealVar with a name, value, limits and number of bins.
"""
function RealVar(name, value::T=0.; limits=(-Inf,Inf), nbins=50) where T<:Real
    RealVar{T}(name, value, limits, nbins, false)
end
function setproperty!(v::RealVar{T}, name::Symbol, value) where T<:Real
    if name == :value
        v.isconst && throw(ArgumentError("Cannot set value of a constant variable '$(v.name)'"))
        # Check if the value is within the limits
        a, b = v.limits
        if value < a
            setfield!(v, name, a)
            @warn("Value $(value) is below limits $(v.limits): limited to $(a)")
        elseif value > b
            setfield!(v, name, b)
            @warn("Value $(value) is above limits $(v.limits): limited to $(b)")
        else
            setfield!(v, name, T(value))
        end
    else
        throw(ArgumentError("Field $(name) not found in $(nameof(typeof(v)))"))
    end
end
setindex!(v::RealVar, value) = setproperty!(v, :value, value)
isconst(v::RealVar) = v.isconst

"""
    ConstVar(name, value=0.)

Construct a ConstVar with a name ans value.
"""
function ConstVar(name, value::T) where T<:Real
    RealVar{T}(name, value, (value,value), 0, true)
end
function ConstVar(value::T=0.) where T<:Real
    RealVar{T}(:none, value, (value,value), 0, true)
end


#--------------------------------------------------------------------------------------------------
#---Gaussian distribution--------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
struct Gaussian{T<:Real, PDF<:Function} <: AbstractPdf
    name::Symbol
    x::RealVar{T}
    μ::RealVar{T}
    σ::RealVar{T}
    params::Tuple{Vararg{RealVar{T}}}
    distribution::UnivariateDistribution
    pdf::PDF
end
"""
    Gaussian(name, x, μ, σ)

Construct a Gaussian distribution with a name, a RealVar x, a RealVar μ and a RealVar σ.
"""
function Gaussian(name, x, μ, σ)
    a, b = x.limits
    μ_ = μ.isconst ? μ.value : μ.name
    σ_ = σ.isconst ? σ.value : σ.name
    params = Tuple(p for p in (μ,σ) if !isconst(p))
    fname = gensym("$(name)_pdf")
    pdf = eval( quote
                    function $(fname)(x)
                        $(fname)(x, $((param.value for param in params)...))
                    end
                    function $(fname)(x, $((param.name for param in params)...))
                        d = _Normal($(μ_),$(σ_))
                        scale = cdf(d, $(b)) - cdf(d, $(a))
                        ifelse.($(a) .<= x .<= $(b), pdf.(d, x) ./ scale, zero(x)) 
                    end
                end )
    dist = truncated(_Normal(μ.value, σ.value), a, b)
    Gaussian(name, x, μ, σ, params, dist, pdf)
end

#
#---Exponential distribution---------------------------------------------------------------
#
struct Exponential{T<:Real,PDF<:Function} <: AbstractPdf
    name::Symbol
    x::RealVar{T}
    c::RealVar{T}
    params::Tuple{Vararg{RealVar{T}}}
    distribution::UnivariateDistribution
    pdf::PDF
end
"""
    Exponential(name, x, c)

Construct an Exponential distribution with a `name`, a RealVar `x` and a RealVar `c`.

``\\mathrm{Exponential}(x, c) = \\mathcal{N} \\cdot \\exp(c\\cdot x)``, 
where ``\\mathcal{N}`` is a normalization constant that depends on the range and values of the arguments
"""
function Exponential(name, x, c)
    a, b = x.limits
    c_ = c.isconst ? c.value : c.name
    params = Tuple(p for p in (c,) if !isconst(p))
    fname = gensym("$(name)_pdf")
    pdf = eval( quote
                    function $(fname)(x)
                        $(fname)(x, $((param.value for param in params)...))
                    end
                    function $(fname)(x, $((param.name for param in params)...))
                        d = _Exponential(-1/$(c_))
                        scale = cdf(d, $(b)) - cdf(d, $(a))
                        ifelse.($(a) .<= x .<= $(b), pdf.(d, x) ./ scale, zero(x)) 
                    end
                end )
    dist = truncated(_Exponential(-1/c.value), a, b)
    Exponential(name, x, c, params, dist, pdf)
end

#
#---ArgusPdf distribution---------------------------------------------------------------
#
struct ArgusPdf{T<:Real,PDF<:Function} <: AbstractPdf
    name::Symbol
    x::RealVar{T}
    m₀::RealVar{T}
    c::RealVar{T}
    p::RealVar{T}
    params::Tuple{Vararg{RealVar{T}}}
    distribution::UnivariateDistribution
    pdf::PDF
end
"""
    ArgusPdf(name, m, m₀, c, p)

ArgusPdf describes the ARGUS background shape.

``  \\mathrm{Argus}(m, m_0, c, p) = \\mathcal{N} \\cdot m \\cdot \\left[ 1 - \\left( \\frac{m}{m_0} \\right)^2 \\right]^p
  \\cdot \\exp\\left[ c \\cdot \\left(1 - \\left(\\frac{m}{m_0}\\right)^2 \\right) \\right]``
"""
function ArgusPdf(name, m, m₀, c, p=ConstVar(:p, 0.5))
    a, b = m.limits
    m₀_ = m₀.isconst ? m₀.value : m₀.name
    c_ = c.isconst ? c.value : c.name
    p_ = p.isconst ? p.value : p.name
    params = Tuple(p for p in (m₀, c, p) if !isconst(p))
    fname = gensym("$(name)_pdf")
    pdf = eval( quote
                    function $(fname)(x)
                        $(fname)(x, $((param.value for param in params)...))
                    end
                    function $(fname)(x, $((param.name for param in params)...))
                        d = ArgusBGDist($(m₀_), $(c_), $(p_), $(a), $(b))
                        ifelse.($(a) .<= x .<= $(b), pdf.(d, x), zero(x))
                    end
                end )
    dist = ArgusBGDist(m₀.value, c.value, p.value, a, b)
    ArgusPdf(name, m, m₀, c, p, params, dist, pdf)
end

#---Chebyshev distribution-------------------------------------------------------------------------
struct Chebyshev{T<:Real,PDF<:Function} <: AbstractPdf
    name::Symbol
    x::RealVar{T}
    params::Tuple{Vararg{RealVar{T}}}
    distribution::UnivariateDistribution
    pdf::PDF
end
"""
    Chebyshev(name, x, coeffs)

Construct a Chebyshev distribution with a name, a RealVar x and a vector of coefficients.
"""
function Chebyshev(name, x, coeffs)
    a, b = x.limits
    params = Tuple(p for p in coeffs if !isconst(p))
    body = "d = ChebyshevDist([" * join([isconst(c) ? "$(c.value)" : "$(c.name)" for c in coeffs], ", ") * "], $(a), $(b))" |> Meta.parse
    fname = gensym("$(name)_pdf")
    pdf = eval( quote
                    function $(fname)(x)
                        $(fname)(x, $((param.value for param in params)...))
                    end
                    function $(fname)(x, $((param.name for param in params)...))
                        $(body)
                        pdf.(d, x)
                    end
                end )
    dist = ChebyshevDist([c.value for c in coeffs], a, b)
    Chebyshev(name, x, params, dist, pdf)
end

#--------------------------------------------------------------------------------------------------
#---AddPdf distribution----------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
struct AddPdf{T<:Real,PDFS<:Tuple,F<:Function} <: AbstractPdf
    name::Symbol
    x::RealVar{T}
    params::Tuple{Vararg{RealVar{T}}}
    pdfs::PDFS
    fractions::Vector{RealVar{T}}
    pdf::F
    extendable::Bool  # Flag indicating if all PDF components are extendable
    recursive::Bool   # Flag indicating is fractions are treated recursively
end

"""
    AddPdf(name, pdfs, fractions)

Construct an AddPdf distribution with a name, a vector of AbstractPdf pdfs and a vector of RealVar fractions.
## Arguments
- `name::Symbol`: Name of the distribution.
- `pdfs::Vector{<:AbstractPdf}`: Vector of AbstractPdf pdfs.
- `fractions::Vector{<:RealVar}`: Vector of RealVar fractions or coefficients.
   - If the number of pdfs is equal to the number of fractions, the fractions represent absolute populations.
   - If the number of pdfs is equal to the number of fractions + 1, the fractions represent recursive fractions.
## Returns
-  An AddPdf distribution.
"""
function AddPdf(name, pdfs, coefs)
    extendable, recursive = false, false
    xs = unique([pdf.x for pdf in pdfs])
    length(xs) != 1 && throw(ArgumentError("All pdfs must have the same x variable $(xs)"))
    params = unique(vcat([param for pdf in pdfs for param in pdf.params], coefs))
    fname = gensym("$(name)_pdf")
    if length(pdfs) == length(coefs)
        # The coefs represents absolute populations
        extendable = true
        body = join(["$(coef.name)" * " * var\"$(pdf.pdf)\"(x, $(join([param.name for param in pdf.params], ", ")))" for (coef, pdf) in zip(coefs, pdfs)], " + ") |> Meta.parse
    elseif(length(pdfs) == length(coefs) + 1)
        # The coefs represents recursive fractions
        recursive = true
        coeffs = get_coefficients([fraction.name for fraction in coefs])
        body = join([coeff * " * var\"$(pdf.pdf)\"(x, $(join([param.name for param in pdf.params], ", ")))" for (coeff, pdf) in zip(coeffs, pdfs)], " + ") |> Meta.parse
    else
        throw(ArgumentError("Number of pdfs and coefficients are inconsistent"))
    end
    pdf = eval( quote
                    function $(fname)(x)
                        $(fname)(x, $((param.value for param in params)...))
                    end
                    function $(fname)(x, $((param.name for param in params)...))
                        $(body)
                    end
                end )
    AddPdf(name, xs[1], Tuple(params), Tuple(pdfs), coefs, pdf, extendable, recursive)
end
"""
    AddPdf(name, pdf1::AbstractPdf, pdf2::AbstractPdf, fraction::RealVar)

Construct an AddPdf distribution with a name, two AbstractPdf pdfs and a RealVar fraction.
"""
function AddPdf(name, pdf1::AbstractPdf, pdf2::AbstractPdf, fraction::RealVar)
    AddPdf(name, [pdf1, pdf2], [fraction])
end
show(io::IO, d::AddPdf) = print(io, "$(nameof(typeof(d))){$(join(map(x -> "$(x.name)", d.pdfs), ", "))} PDF with parameters $([p.name for p in d.params])")
function generate(d::AddPdf, n::Integer)
    len = length(d.pdfs)
    if d.recursive
        r = []
        for i in 1:len
            pdf, w = d[i]
            m = Int(round(n * w))
            push!(r, generate(pdf, m))
        end
        return vcat(r...)
    else   # d.extendable
        map(wsample(eachindex(d.pdfs), [f.value for f in d.fractions], n)) do i
            generate(d.pdfs[i])
        end
    end
end
function _generate(index, d::AddPdf, n=1000)
    if index == lastindex(d.pdfs)
        return generate(d.pdfs[index], n)
    else
        n1 = Int(n * d.fractions[index].value)
        n2 = n - n1
        return vcat(generate(d.pdfs[index], n1), _generate(index+1, d, n2))
    end
end
function get_coefficients(c)
    s = []
    for i in 1:length(c)+1
        f = "" 
        for j in 1:i-1
            f = f * "(1-$(c[j]))"
            j < length(c) && (f = f * " * ")
        end
        push!(s, i > length(c) ? f : f * "$(c[i])")
    end
    return s
end
function getindex(d::AddPdf, idx::Integer)
    len = length(d.pdfs)
    isnothing(idx) && throw(ArgumentError("Component $(c) not found in $(d.name)"))
    if d.recursive
        f = idx == len ? 1. : d.fractions[idx].value
        for i in 1:idx-1
            f = f * (1. - d.fractions[i].value)
        end
        return d.pdfs[idx], f
    else
        return d.pdfs[idx], d.fractions[idx].value
    end
end
function getindex(d::AddPdf, c::Symbol)
    idx = findfirst(pdf -> pdf.name == c, d.pdfs)
    return d[idx]
end

end  # module RooFit
export RooFit
