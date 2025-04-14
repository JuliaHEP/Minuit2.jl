module RooFit

import Distributions: UnivariateDistribution, Exponential as _Exponential, Normal as _Normal, Uniform as _Uniform, pdf, cdf, truncated
import FHist: Hist1D, AbstractHistogram
import StatsBase: wsample, mean, std
import Minuit2: ArgusBGDist, ChebyshevDist, BinnedNLL, ExtendedBinnedNLL, UnbinnedNLL, ExtendedUnbinnedNLL, CostFunction, Minuit, migrad!
import Base: getproperty, setproperty!, show, isconst, getindex, setindex!
import Random: AbstractRNG, default_rng


export AbstractPdf, AbstractData, RealVar, ConstVar, DataSet, AbstractHistogram, FitResult
export Gaussian, Exponential, ArgusPdf, Chebyshev
export AddPdf, generate, distribution
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
mean(d::AbstractPdf) = mean(distribution(d))
std(d::AbstractPdf) = std(distribution(d))
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
    generate(d::AbstractPdf, n::Integer=1000; nbins=0)

Generate `n` random numbers from a distribution `d`.
"""
function generate(d::AbstractPdf, n::Int64; nbins=0)
    nbins = nbins > 0 ? nbins : d.x.nbins > 0 ? d.x.nbins : 0
    if nbins > 0
        data = Hist1D(rand(default_rng(), d, n), binedges=range(d.x.limits..., nbins+1))
    else
        data = rand(default_rng(), d, n)
    end
    DataSet(data, (d.x,))
end
function Base.rand(rng::AbstractRNG, d::AbstractPdf, n::Int64=1)
    return rand(rng, distribution(d), n)
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
    error::T
    const limits::Tuple{T, T}
    nbins::Int
    const isconst::Bool
end

"""
    RealVar(name, value=0.;  limits=(-Inf,Inf), nbins=0)

Construct a RealVar with a name, value, limits and number of bins.
"""
function RealVar(name, value::T=T(0), error::T=T(0); limits=(-Inf,Inf), nbins=0) where T<:Real
    RealVar{T}(name, value, error, limits, nbins, false)
end
function RealVar{T}(name, value=0, error=0; limits=(-Inf,Inf), nbins=0) where T<:Real
    RealVar{T}(name, value, error, limits, nbins, true)
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
    elseif name == :error
        setfield!(v, name, T(value))
    elseif name == :measurement
        setfield!(v, :value, value(value))
        setfield!(v, :error, uncertainty(value))
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
    RealVar{T}(name, value, 0, (value,value), 0, true)
end
function ConstVar(value::T=0.) where T<:Real
    RealVar{T}(:none, value, 0, (value,value), 0, true)
end

#--------------------------------------------------------------------------------------------------
#---DataSet type-----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
abstract type AbstractData end
"""
    DataSet
DataSet is a container for a set of data points.
"""
struct DataSet{T<:Real,N} <: AbstractData
    data::Union{Array{T,N}, AbstractHistogram}
    observables::NTuple{N, RealVar{T}}
end
#--------------------------------------------------------------------------------------------------
#---FitResult type-----------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
"""
    FitResult
FitResult is a container for the result of a fit.
"""
struct FitResult
    data::DataSet
    model::AbstractPdf
    engine
end

"""
    fitTo(d::AbstractPdf, data)

Fit a distribution `d` to a data set `data`.
"""
function fitTo(model::AbstractPdf, ds)
    data = ds isa DataSet ? ds.data : ds
    # Check if the data is binned or unbinned
    if data isa AbstractHistogram
        if !isnothing(model.extendable) && model.extendable
            cost = ExtendedBinnedNLL(data, model.pdf, use_pdf=:approximate)
        else
            cost = BinnedNLL(data, model.pdf, use_pdf=:approximate)
        end
    elseif(data isa AbstractArray)
        if !isnothing(model.extendable) && model.extendable 
            # Need to change the model function to include the integral
            fname = gensym("$(model.name)_ext_pdf")
            pdf = eval( quote
                function $(fname)(x, $((param.name for param in model.params)...))
                    +($((f.name for f in model.fractions)...)), $(model.pdf)(x, $((param.name for param in model.params)...))
                end
            end )
            cost = ExtendedUnbinnedNLL(data, pdf)
        else
            cost = UnbinnedNLL(data, model.pdf)
        end
    else
        throw(ArgumentError("Data type not supported"))
    end
    # Create a Minuit object with the cost function and initial parameter values
    m = Minuit(cost; minuitkwargs(model)...)
    # Run the minimization
    migrad!(m)
    # Check if the fit was successful
    if m.is_valid
        # Update the parameters of the distribution with the fitted values
        for (param, value, error) in zip(model.params, m.values, m.errors)
            param.value = value
            param.error = error
        end
        # Return the fitted distribution
        return FitResult(ds, model, m)
    else
        throw(ArgumentError("Fit failed"))
    end
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
    Gaussian(name, x, μ, σ, params, pdf)
end
distribution(d::Gaussian) = truncated(_Normal(d.μ.value, d.σ.value), d.x.limits...)#
#---Exponential distribution---------------------------------------------------------------
#
struct Exponential{T<:Real,PDF<:Function} <: AbstractPdf
    name::Symbol
    x::RealVar{T}
    c::RealVar{T}
    params::Tuple{Vararg{RealVar{T}}}
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
    Exponential(name, x, c, params,pdf)
end
distribution(d::Exponential) = truncated(_Exponential(-1/d.c.value), d.x.limits...)

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
    ArgusPdf(name, m, m₀, c, p, params, pdf)
end
distribution(d::ArgusPdf) = truncated(ArgusBGDist(d.m₀.value, d.c.value, d.p.value), d.x.limits...)

#---Chebyshev distribution-------------------------------------------------------------------------
struct Chebyshev{T<:Real,PDF<:Function} <: AbstractPdf
    name::Symbol
    x::RealVar{T}
    params::Tuple{Vararg{RealVar{T}}}
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
    Chebyshev(name, x, params, dist, pdf)
end
distribution(d::Chebyshev) = ChebyshevDist([c.value for c in d.coeffs], d.x.limits...)

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
function Base.rand(rng::AbstractRNG, d::AddPdf, n::Int64=1)
    len = length(d.pdfs)
    r = []
    for i in 1:len
        pdf, w = d[i]
        m = Int(round(n * w))
        push!(r, rand(rng, pdf, m))
    end
    return vcat(r...)
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
        total = sum(f.value for f in d.fractions)
        return d.pdfs[idx], d.fractions[idx].value/total
    end
end
function getindex(d::AddPdf, c::Symbol)
    idx = findfirst(pdf -> pdf.name == c, d.pdfs)
    return d[idx]
end

end  # module RooFit
export RooFit
