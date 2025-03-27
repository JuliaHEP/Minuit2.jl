module RooFit

import Distributions: Exponential as _Exponential, Normal as _Normal, Uniform as _Uniform, pdf, cdf, truncated
import FHist: Hist1D

import Base: getproperty, show
export RealVar, Gaussian, Exponential, AddPdf, generate, generateBinned, minuitkwargs

abstract type AbstractPdf end
function getproperty(d::AbstractPdf, name::Symbol)
    if hasfield(typeof(d), name)
        return getfield(d, name)
    elseif name == :pdf
        return getpdf(d)
    elseif name == :params
        return params(d)
    else
        return getfield(d, name)
    end
end
show(io::IO, d::AbstractPdf) = print(io, "$(nameof(typeof(d))){$(d.name)} PDF with parameters $([p.name for p in d.params])")
generateBinned(d::AbstractPdf, n=1000) = Hist1D(generate(d, n), binedges=range(d.x.limits..., d.x.nbins+1))
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
    RealVar
Entity to represent a real variable with a name, value, limits and number of bins.
"""
struct RealVar{T<:Real}
    name::Symbol
    value::T
    limits::Tuple{T, T}
    nbins::Int
end
"""
    RealVar(name, value=0.;  limits=(-Inf,Inf), nbins=50)

Construct a RealVar with a name, value, limits and number of bins.
"""
function RealVar(name, value::T=0.0; limits=(-Inf,Inf), nbins=50) where T<:Real
    RealVar{T}(name, value, limits, nbins)
end

#
#---Gaussian distribution-------------------------------------------------------------------
#
struct Gaussian{PDF<:Function} <: AbstractPdf
    name::Symbol
    x::RealVar
    μ::RealVar
    σ::RealVar
    pdf::PDF
end
"""
    Gaussian(name, x, μ, σ)

Construct a Gaussian distribution with a name, a RealVar x, a RealVar μ and a RealVar σ.
"""
function Gaussian(name, x, μ, σ)
    a, b = x.limits
    fname = Symbol("pdf_", name)
    _pdf = eval( quote
                    function $(fname)(x, $(μ.name)=$(μ.value), $(σ.name)=$(σ.value))
                    d = _Normal($(μ.name),$(σ.name))
                    $(a) <= x <= $(b) ? pdf(d, x)/(cdf(d, $(b)) - cdf(d, $(a))) : zero(x) 
                    end
                end )
    Gaussian(name, x, μ, σ, _pdf)
end
"""
    generate(d::Gaussian, n=1000)

Generate n random numbers from a Gaussian distribution d.
"""
function generate(d::Gaussian, n=1000)
    rand(truncated(_Normal(d.μ.value, d.σ.value), d.x.limits...),n)
end
params(d::Gaussian) = (d.μ, d.σ)

#
#---Exponential distribution---------------------------------------------------------------
#
struct Exponential{PDF<:Function} <: AbstractPdf
    name::Symbol
    x::RealVar
    c::RealVar
    pdf::PDF
end
"""
    Exponential(name, x, c)

Construct an Exponential distribution with a name, a RealVar x and a RealVar c.

\\f[
  \\mathrm{Exponential}(x, c) = \\mathcal{N} \\cdot \\exp(c\\cdot x) 
\\f], where \\f[ \\mathcal{N} \\f] is a normalisation constant that depends on the range and values of the arguments
"""
function Exponential(name, x, c)
    a, b = x.limits
    fname = Symbol("pdf_", name)
    pdf = eval( quote
                    function $(fname)(x, $(c.name)= $(c.value))
                    d = _Exponential(-1/$(c.name))
                    $(a) <= x <= $(b) ? pdf(d, x)/(cdf(d, $(b)) - cdf(d, $(a))) : zero(x) 
                    end
                end )
    Exponential(name, x, c, pdf)
end
params(d::Exponential) = (d.c,)
generate(d::Exponential, n=1000) = rand(truncated(_Exponential(-1/d.c.value), d.x.limits...),n)

#
#---AddPdf distribution---------------------------------------------------------------------
#
struct AddPdf{T<:Real,PDFS<:Tuple,F<:Function} <: AbstractPdf
    name::Symbol
    x::RealVar{T}
    params::Tuple{Vararg{RealVar{T}}}
    pdfs::PDFS
    fractions::Vector{RealVar{T}}
    pdf::F
end
"""
    AddPdf(name, pdfs, fractions)

Construct an AddPdf distribution with a name, a vector of AbstractPdf pdfs and a vector of RealVar fractions.
"""
function AddPdf(name, pdfs, fractions)
    length(pdfs) != 1+length(fractions) && throw(ArgumentError("Number of pdfs and c inconsistent"))
    params = unique(vcat([param for pdf in pdfs for param in pdf.params], [fraction for fraction in fractions]...))
    xs = unique([pdf.x for pdf in pdfs])
    length(xs) != 1 && throw(ArgumentError("All pdfs must have the same x variable $(xs)"))
    fname = Symbol("pdf_", name)
    coeffs = get_coefficients([fraction.name for fraction in fractions])
    body = join([coeff * " * $(pdf.pdf)(x, $(join([param.name for param in pdf.params], ", ")))" for (coeff, pdf) in zip(coeffs, pdfs)], " + ") |> Meta.parse
    pdf = eval( quote
                    function $(fname)(x, $((param.name for param in params)...))
                    $(body)
                    end
                end )
    AddPdf(name, xs[1], Tuple(params), Tuple(pdfs), fractions, pdf)
end
"""
    AddPdf(name, pdf1::AbstractPdf, pdf2::AbstractPdf, fraction::RealVar)

Construct an AddPdf distribution with a name, two AbstractPdf pdfs and a RealVar fraction.
"""
function AddPdf(name, pdf1::AbstractPdf, pdf2::AbstractPdf, fraction::RealVar)
    AddPdf(name, [pdf1, pdf2], [fraction])
end
show(io::IO, d::AddPdf) = print(io, "$(nameof(typeof(d))){$(join(map(x -> "$(x.name)", d.pdfs), ", "))} PDF with parameters $([p.name for p in d.params])")
generate(d::AddPdf, n=1000) = _generate(1, d, n)
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

end  # module RooFit
export RooFit
