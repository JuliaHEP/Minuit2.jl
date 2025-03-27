#-------------------------------------------------------------------------------------------
#    Standard cost functions to minimize for statistical fits
#    We provide these for convenience, so that you do not have to write your own for standard
#    fits. You can also use them as a template to write your own cost functions.
#-------------------------------------------------------------------------------------------
import IterTools
abstract type CostFunction end
import Base: getproperty, setproperty!, propertynames, show, getindex, length
export LeastSquares, Constant, CostFunction, UnbinnedCostFunction, BinnedCostFunction,
       BaseCost, UnbinnedNLL, BinnedNLL, ExtendedUnbinnedNLL, ExtendedBinnedNLL
export value, grad, has_grad

mutable struct BaseCost
    verbose::Int
    parameters::Vector{String}
end

const CHISQUARE = 1.0
const NEGATIVE_LOG_LIKELIHOOD = 0.5
const F64_TINY = floatmin(Float64)

unbinned_nll(x) = -sum(log.(x .+ F64_TINY))
z_squared(y, ye, ym) = ((y .- ym) ./ ye).^2

"""
    chi2(y, ye, ym)

Compute (potentially) chi2-distributed cost.

The value returned by this function is chi2-distributed, if the observed values are
normally distributed around the expected values with the provided standard
deviations.

## Arguments
- `y` : Observed values.
- `ye` : Uncertainties of values.
- `ym` : Expected values.

# Returns
- Value of cost function.
"""
chi2(y, ye, ym) = sum(z_squared(y, ye, ym))

"""
    soft_l1_cost(y, ye, ym)
"""
soft_l1_cost(y, ye, ym) = 2 * sum(sqrt.(1 .+ z_squared(y, ye, ym)) .- 1)

"""
    chi2_grad(y, ye, ym, gym)

Compute gradient of function `chi2`.

## Arguments
- `y` : Observed values.
- `ye` : Uncertainties of values.
- `ym` : Expected values.
- `gym` : Gradient of ym with respect to K model parameters.

## Returns
- Gradient of cost function with respect to model parameters.  
"""
chi2_grad(y, ye, ym, gym) = -2 * sum((y .- ym) .* gym .* ye.^-2)

"""
    soft_l1_cost_grad(y, ye, ym, gym)

"""
function soft_l1_cost_grad(y, ye, ym, gym)
    z = (y .- ym) ./ ye
    f = (1 .+ z.^2).^-0.5
    return -2 * sum(z ./ ye .* f .* gym)
end

log_or_zero(x) = x <= 0.0 ? 0.0 : log(x)

"""
    multinomial_chi2(n, mu)

Compute asymptotically chi2-distributed cost for multinomially-distributed data.
See Baker & Cousins, NIM 221 (1984) 437-442.

## Arguments
- `n` : Observed counts.
- `mu` Expected counts. Must satisfy sum(mu) == sum(n).

## Returns
- Cost function value.

## Notes
The implementation makes the result asymptotically chi2-distributed,
which helps to maximise the numerical accuracy for Minuit.
"""
function multinomial_chi2(n, mu)
    return 2 * sum(n .* (log_or_zero.(n) .- log_or_zero.(mu)))
end

"""
    multinomial_chi2_grad(n, mu, gmu)

Compute gradient of function `multinomial_chi2`.
"""
function multinomial_chi2_grad(n, mu, gmu)
    return - 2 * sum( (n ./ mu)' * gmu, dims=1)
end

"""
    poisson_chi2(n, mu)

Compute asymptotically chi2-distributed cost for Poisson-distributed data.
See Baker & Cousins, NIM 221 (1984) 437-442.

## Arguments
- `n` : Observed counts.
- `mu` : Expected counts per bin.

## Returns
- Cost function value.

## Notes
The implementation makes the result asymptotically chi2-distributed,
which helps to maximise the numerical accuracy for Minuit.

If sum(mu) == sum(n), the result is equal to `multinomial_chi2`.
"""
function poisson_chi2(n, mu)
    return 2 * sum(n .* (log_or_zero.(n) .- log_or_zero.(mu)) .+ mu .- n)
end


"""
    poisson_chi2_grad(n, mu, gmu)

Compute gradient of function `poisson_chi2`.
"""
function poisson_chi2_grad(n, mu, gmu)
    return 2 * sum( (1 .- n ./ mu)' * gmu, dims=1)
end

function model_parameters(model, names)
    # strip first argument from model
    args = get_argument_names(model)[2:end]
    if length(names) > 0
        if length(args) == length(names) || length(args) == 1
            params = [n for n in names]
        else
            throw(ArgumentError("length of names does not match number of model parameters"))
        end
    else
        params = args
    end
    return params
end

#---Base cost functions----------------------------------------------------------------------------
errordef(::CostFunction) = CHISQUARE
npar(cost::CostFunction) = length(cost.base.parameters)
verbose(cost::CostFunction) = cost.base.verbose
verbose(cost::CostFunction, level::Int) = (cost.base.verbose = level; cost)
(cost::CostFunction)(args...) = value(cost, args)
function cost_value(cost::CostFunction, args)
    targs = Tuple(args[])
    r = value(cost, targs)
    if verbose(cost) > 0
        println("$targs -> $r")
    end
    return r
end

function Base.show(io::IO, cost::CostFunction)
    modelname = hasproperty(cost, :model) ? "$(cost.model)" : "unknown"
    print(io, "$(typeof(cost)) cost function of \"$modelname\" with parameters $(cost.parameters)")
end

function getproperty(cost::CostFunction, sym::Symbol)
    if hasproperty(getfield(cost, :base), sym)
        return getproperty(getfield(cost, :base), sym)
    elseif sym == :npar
        return length(cost.parameters)
    elseif sym == :errordef
        return errordef(cost)
    elseif sym == :ndim
        return ndim(cost)
    elseif sym == :items
        return [cost[i] for i in 1:length(cost)]
    elseif sym == :ndata
        return ndata(cost)
    else
        return getfield(cost, sym)
    end
end

#---Abstract Binned cost function------------------------------------------------------------------
abstract type BinnedCostFunction{T<:Real,N} <: CostFunction end
has_grad(cost::BinnedCostFunction) = cost.model_grad !== nothing
ndata(cost::BinnedCostFunction) = length(cost.bincounts)

function getproperty(cost::BinnedCostFunction, sym::Symbol)
    if hasproperty(getfield(cost, :base), sym)
        return getproperty(getfield(cost, :base), sym)
    elseif sym == :errordef
        return errordef(cost)
    elseif sym == :npar
        return npar(cost)
    elseif sym == :ndata
        return ndata(cost)
    elseif sym == :ndim
        return ndims(cost.bincounts)
    else
        return getfield(cost, sym)
    end
end
bincenters(edges::AbstractVector) = [(edges[i] + edges[i+1])/2 for i in 1:length(edges)-1]
bincenters(edges::Tuple) = IterTools.product([bincenters(collect(e)) for e in edges]...) |> collect
binwidths(edges::AbstractVector) = diff(edges)
binwidths(edges::Tuple) = IterTools.product([binwidths(e) for e in edges]...) |> collect
loweredges(edges::AbstractVector) = [edges[i] for i in 1:length(edges)-1]
loweredges(edges::Tuple) = IterTools.product([loweredges(e) for e in edges]...) |> collect
upperedges(edges::AbstractVector) = [edges[i+1] for i in 1:length(edges)-1]
upperedges(edges::Tuple) = IterTools.product([upperedges(e) for e in edges]...) |> collect

#---Abstract Unbinned cost function----------------------------------------------------------------
abstract type UnbinnedCostFunction <: CostFunction end

errordef(::UnbinnedCostFunction) = NEGATIVE_LOG_LIKELIHOOD
has_grad(cost::UnbinnedCostFunction) = cost.model_grad !== nothing
ndata(cost::UnbinnedCostFunction) = Inf

function getproperty(cost::UnbinnedCostFunction, sym::Symbol)
    if hasproperty(getfield(cost, :base), sym)
        return getproperty(getfield(cost, :base), sym)
    elseif sym == :errordef
        return errordef(cost)
    elseif sym == :npar
        return npar(cost)
    elseif sym == :ndata
        return ndata(cost)
    elseif sym == :ndim
        return Base.size(cost.data, 2)
    elseif sym == :len
        return Base.size(cost.data, 1)
    else
        return getfield(cost, sym)
    end
end

#---Constant cost function-------------------------------------------------------------------------
struct Constant <: CostFunction
    base::BaseCost
    value::Float64
end

"""
    Constant(value::Float64; verbose::Int=0)

Constant cost function with fixed value.
"""
Constant(value::Float64; verbose::Int=0) = Constant(BaseCost(verbose, []), value)
value(cost::Constant, args=[]) = cost.value
grad(cost::Constant, args=[]) = 0.0
has_grad(cost::Constant) = true
ndata(cost::Constant) = 0
Base.show(io::IO, cost::Constant) = print(io, "Constant cost function with value $(cost.value)")


#---CostSum cost function-------------------------------------------------------------------------
mutable struct CostSum <: CostFunction
    base::BaseCost
    costs::Tuple
    argsmapping::Vector{Vector{Int}}
end

"""
    CostSum(costs::CostFunction...; verbose::Int=0)
"""
function CostSum(costs::CostFunction...; verbose::Int=0)
    items = CostFunction[]
    for item in costs
        if item isa CostSum
            append!(items, item.items)
        elseif item isa Real
            push!(items, Constant(item))
        else
            push!(items, item)
        end
    end
    parameters = unique(vcat([cost.parameters for cost in items]...))
    argsmapping = [[findfirst(p -> p == param, parameters) for param in c.parameters] for c in items]
    CostSum(BaseCost(verbose, parameters), tuple(items...), argsmapping)
end

has_grad(cost::CostSum) = all(has_grad(c) for c in cost.costs)
ndata(cost::CostSum) = sum(c.ndata for c in cost.costs)
ndim(cost::CostSum) = maximum(c.ndim for c in cost.costs)
length(cost::CostSum) = length(cost.costs)
getindex(cost::CostSum, key) = cost.costs[key]
Base.:+(costs::CostFunction...) = CostSum(costs...)

function value(cost::CostSum, args)
    sum(value.(cost.costs, [args[m] for m in cost.argsmapping]))
end

function grad(cost::CostSum, args)
    r = zeros(cost.npar)
    for (c, m) in zip(cost.costs, cost.argsmapping)
        r[m] .+= grad(c, args[m])
    end
    return r
end

#---UnbinnedNLL cost function---------------------------------------------------------------------
mutable struct UnbinnedNLL{T<:Real,N,F<:Function} <: UnbinnedCostFunction
    base::BaseCost
    log::Bool
    data::Array{T,N}
    mask::Union{Vector{Bool}, BitVector, Nothing}
    model::F
    model_grad::Union{Function, Nothing}
end

"""
    UnbinnedNLL(data::AbstractArray, pdf::Function; log=false, verbose=0, mask=nothing, grad=nothing, names=())

Unbinned negative log-likelihood cost function.

## Arguments
- `data::AbstractArray` : Sample of observations. If the observations are multidimensional, data must
  have the shape (D, N), where D is the number of dimensions and N the number of data points.
- `pdf::Function` : Probability density function of the form f(data, par0, [par1, ...]), where
  data is the data sample and par0, ... are model parameters. If the data are
  multivariate, data passed to f has shape (D,), where D is the number of
  dimensions and N the number of data points.
- `verbose::Int` : Verbosity level. 0: is no output (default). 1: print current args and
  negative log-likelihood value.
- `log::Bool=false` : Distributions of the exponential family (normal, exponential, poisson, ...)
  allow one to compute the logarithm of the pdf directly, which is more
  accurate and efficient than numerically computing ``log(pdf)``. Set this
  to `true`, if the model returns the logpdf instead of the pdf.
- `mask::Union{Vector{Bool}, BitVector, Nothing}` : Optional mask to select a subset of the data. Must have the same length as data.
- `grad::Union{Function, Nothing}` : Optionally pass the gradient of the pdf. Has the same calling signature like
  the pdf, but must return an array with the shape (K, N), where N is the
  number of data points and K is the number of parameters. If `log` is True,
  the function must return the gradient of the logpdf instead of the pdf. The
  gradient can be used by Minuit to improve or speed up convergence and to
  compute the sandwich estimator for the variance of the parameter estimates.
- `names` : Optional names for each parameter of the model (in order). Must have the
  same length as there are model parameters.

## Returns
- Cost function object.
"""
function UnbinnedNLL(data::AbstractArray, pdf::F; log=false, verbose=0, mask=nothing, grad=nothing, names=()) where F<:Function
    if ndims(data) == 1 && eltype(data) <: Tuple
        data = reduce(vcat, [[t...]' for t in data])
    end
    T = eltype(data)
    N = ndims(data)
    params = model_parameters(pdf, names)
    UnbinnedNLL{T,N,F}(BaseCost(verbose, params), log, data, mask, pdf, grad)
end


function value(cost::UnbinnedNLL, args)
    data = isnothing(cost.mask) ? cost.data : cost.data[cost.mask]
    if cost.ndim == 1
        f = cost.model.(data, args...)
    else
        f = [cost.model(data[i,:], args...) for i in 1:cost.len] 
    end
    cost.log ? -2.0 * sum(f) : 2.0 * unbinned_nll(f)
end

function grad(cost::UnbinnedNLL, args)
    isnothing(cost.model_grad) && throw(ArgumentError("no gradient available"))
    data = isnothing(cost.mask) ? cost.data : cost.data[cost.mask]
    if cost.ndim == 1
        g = cost.model_grad.(data, args...)
        if cost.log == false
            f = cost.model.(data, args...)
            g = g ./ f       
        end
    else
        g = [cost.model_grad(data[i,:], args...) for i in 1:cost.len]
        if cost.log == false
            f = [cost.model(data[i,:], args...) for i in 1:cost.len]
            g = g ./ f
        end
    end
    return -2.0 * sum(g)
end

#---ExtendedUnbinnedNLL cost function-------------------------------------------------------------
mutable struct ExtendedUnbinnedNLL{T<:Real,N,F<:Function} <: UnbinnedCostFunction
    base::BaseCost
    log::Bool
    data::Array{Float64,N}
    mask::Union{Vector{Bool}, BitVector, Nothing}
    model::F
    model_grad::Union{Function, Nothing}
end

"""
    ExtendedUnbinnedNLL(data::AbstractArray, scaled_pdf::Function; log=false, verbose=0, mask=nothing, grad=nothing, names=())

Unbinned extended negative log-likelihood.

Use this if shape and normalization of the fitted PDF are of interest and the
original unbinned data is available. The data can be one- or multi-dimensional.

## Arguments
- `data::AbstractArray` : Sample of observations. If the observations are multidimensional, data must
   have the shape (D, N), where D is the number of dimensions and N the number of data points.
- `scaled_pdf::Function` : Probability density function of the form f(data, par0, [par1, ...]), where
   data is the sample and par0, ... are model parameters. Must return a tuple (<integral
   over f in data window>, <f evaluated at data points>). The first value is the density integrated over 
   the data window, the interval that we consider for the fit. For example, if the data are exponentially distributed, but we
   fit only the interval (0, 5), then the first value is the density integrated from 0 to 5. 
   If the data are multivariate, data passed to f has shape (D,N), where D is the number of dimensions and N the number of data points.
- `verbose::Int` : Verbosity level. 0: is no output.
- `log::Bool=false` : Distributions of the exponential family (normal, exponential, poisson, ...)
   allow one to compute the logarithm of the pdf directly, which is more
   accurate and efficient than numerically computing ``log(pdf)``. Set this
   to `true`, if the model returns the logpdf instead of the pdf.
- `mask::Union{Vector{Bool}, BitVector, Nothing}` : Optional mask to select a subset of the data. Must have the same length as data.
- `grad::Union{Function, Nothing}` : Optionally pass the gradient of the pdf. Has the same calling signature like
   the pdf, but must return an array with the shape (K, N), where N is the number of data points and K is the number of parameters.
- `names` : Optional names for each parameter of the model (in order). Must have the same length as there are model parameters.
"""
function ExtendedUnbinnedNLL(data::AbstractArray, scaled_pdf::F; log=false, verbose=0, mask=nothing, grad=nothing, names=()) where F<:Function
    if ndims(data) == 1 && eltype(data) <: Tuple
        data = reduce(vcat, [[t...]' for t in data])
    end
    N = ndims(data)
    T = eltype(data)
    params = model_parameters(scaled_pdf, names)
    ExtendedUnbinnedNLL{T,N,F}(BaseCost(verbose, params), log, data, mask, scaled_pdf, grad)
end

function value(cost::ExtendedUnbinnedNLL, args)
    data = isnothing(cost.mask) ? cost.data : cost.data[cost.mask]
    fint, f = cost.model(data, args...)
    cost.log ? 2 * (fint - sum(f)) : 2 * (fint + unbinned_nll(f))
end

function grad(cost::ExtendedUnbinnedNLL, args)
    isnothing(cost.model_grad) && throw(ArgumentError("no gradient available"))
    data = isnothing(cost.mask) ? cost.data : cost.data[cost.mask]
    gint, g = cost.model_grad(data, args...)
    if !cost.log
        _, f = cost.model(data, args...)
        g = g ./ f       
    end 
    return -2 * sum(g .- (gint ./ cost.len))
end

#---BinnedNLL cost function-----------------------------------------------------------------------
mutable struct BinnedNLL{T<:Real,N,F<:Function} <: BinnedCostFunction{T,N}
    base::BaseCost
    bincounts::Array{T,N}
    model::F
    model_grad::Union{Function, Nothing}
    use_pdf::Symbol
    entries::Int
    bincenters::Union{Nothing, Vector{T}, Array{NTuple{N,T}, N}}
    binwidths::Union{Nothing, Vector{T}, Array{NTuple{N,T}, N}}
    loweredges::Union{Nothing, Vector{T}, Array{NTuple{N,T}, N}}
    upperedges::Union{Nothing, Vector{T}, Array{NTuple{N,T}, N}}
end

"""
    BinnedNLL(bincounts::AbstractArray, binedges::Union{AbstractArray, Tuple}, cdf::Function; use_pdf=:none, verbose=0,  grad=nothing, names=())

Binned negative log-likelihood.

Use this if only the shape of the fitted PDF is of interest and the data is binned. This cost function works with normal and weighted histograms. 
The histogram can be one- or multi-dimensional.

## Arguments
- `bincounts::AbstractArray` : Histogram counts. If this is an array with dimension D, where D is the number of histogram axes.
- `xe::Union{AbstractArray, Tuple}` : Bin edge locations, must be len(n) + 1, where n is the number of bins. 
  If the histogram has more than one axis, xe must be a collection of the bin edge locations along each axis.
- `cdf::Function` : Cumulative density function of the form f(xe, par0, par1, ..., parN),
   where xe is a bin edge and par0, ... are model parameters. The corresponding density must be normalized to unity 
   over the space covered by the histogram. If the model is multivariate, xe must be an array-like with shape (D, N),
   where D is the dimension and N is the number of points where the model is evaluated.
- `verbose::Int` : Verbosity level. 0: is no output.
- `grad::Union{Function, Nothing} : Optionally pass the gradient of the `cdf``. Has the same calling signature like the cdf, 
   but must return an array with the shape (K,), where K is the number of parameters. The gradient can be used by Minuit to 
   improve or speed up convergence.
- `use_pdf::Symbol`: Either `:none`, `:numerical`, or `:approximate`. If the model cdf is not available, but the model pdf is, 
   this option can be set to "numerical" or "approximate" to compute the integral of the pdf over the bin. The option "numerical" 
   uses numerical integration, which is accurate but computationally expensive and only supported for 1D histograms. The
   option "approximate" uses the zero-order approximation of evaluating the pdf at the bin center, multiplied with the bin area.
   This is fast and works in higher dimensions, but can lead to biased results if the curvature of the pdf inside the bin is significant.
- `names` : Optional names for each parameter of the model (in order). Must have the same length as there are model parameters.
"""
function BinnedNLL(bincounts::AbstractArray, binedges::Union{AbstractArray, Tuple}, cdf::F; use_pdf=:none, verbose=0,  grad=nothing, names=()) where F<:Function
    ndim = ndims(bincounts)
    T = eltype(bincounts)
    ndim < 2 || binedges isa Tuple || throw(ArgumentError("binedges must be a Tuple for ndim > 1"))
    params = model_parameters(cdf, names)
    if use_pdf == :numerical
        throw(ArgumentError("numerical calculation not implemented yet"))
    elseif use_pdf == :approximate || use_pdf == :none
        centers = bincenters(binedges)
        widths = binwidths(binedges)
        ledges = loweredges(binedges)
        uedges = upperedges(binedges)
    else
        throw(ArgumentError("unknown use_pdf: $use_pdf"))
    end
    BinnedNLL{T,ndim,F}(BaseCost(verbose, params), bincounts, cdf, grad, use_pdf, sum(bincounts), centers, widths, ledges, uedges)
end

function _pred(cost::BinnedCostFunction, args)
    if cost.use_pdf == :approximate
        f = cost.model.(cost.bincenters, args...) .* prod(cost.binwidths[1])
    else
        f = cost.model(cost.upperedges, args...) .- cost.model(cost.loweredges, args...)
        f[f .<= 0] .= F64_TINY
    end
    return f
end

function _pred_grad(cost::BinnedCostFunction, args)
    if cost.use_pdf == :approximate
        gf = cost.model_grad.(cost.bincenters, args...) .* prod(cost.binwidths[1])
    else
        gf = (cost.model_grad.(cost.upperedges, args...) - cost.model_grad.(cost.loweredges, args...))
    end
    return  [gf[i][j] for i in 1:cost.ndata, j in 1:cost.npar]
end

function value(cost::BinnedNLL, args)
    p = _pred(cost, args) * cost.entries
    multinomial_chi2(cost.bincounts, p)
end

function grad(cost::BinnedNLL, args)
    isnothing(cost.model_grad) && throw(ArgumentError("no gradient available"))
    p = _pred(cost, args) * cost.entries
    gf = _pred_grad(cost, args) * cost.entries # scale gradients with number entries
    multinomial_chi2_grad(cost.bincounts, p, gf) |> vec
end

#---ExtendedBinnedNLL cost function-------------------------------------------------------------
mutable struct ExtendedBinnedNLL{T<:Real,N,F<:Function} <: BinnedCostFunction{T,N}
    base::BaseCost
    bincounts::Array{T,N}
    model::F
    model_grad::Union{Function, Nothing}
    use_pdf::Symbol
    entries::Int
    bincenters::Union{Nothing, Vector{T}, Array{NTuple{N,T}, N}}
    binwidths::Union{Nothing, Vector{T}, Array{NTuple{N,T}, N}}
    loweredges::Union{Nothing, Vector{T}, Array{NTuple{N,T}, N}}
    upperedges::Union{Nothing, Vector{T}, Array{NTuple{N,T}, N}}
end

"""
ExtendedBinnedNLL(bincounts::AbstractArray, binedges::Union{AbstractArray, Tuple}, cdf::Function; use_pdf=:none, verbose=0,  grad=nothing, names=())

Binned extended negative log-likelihood.

Use this if shape and normalization of the fitted PDF are of interest and the data
is binned. This cost function works with normal and weighted histograms. The
histogram can be one- or multi-dimensional.

The cost function works for both weighted data. The cost function assumes that
the weights are independent of the data.

The cost function has a minimum value that is asymptotically chi2-distributed. It is
constructed from the log-likelihood assuming a poisson distribution and using the
saturated model as a reference.

## Arguments
- `bincounts::AbstractArray` : Histogram counts. If this is an array with dimension D, where D is the number of histogram axes.
- `xe::Union{AbstractArray, Tuple}` : Bin edge locations, must be len(n) + 1, where n is the number of bins. 
  If the histogram has more than one axis, xe must be a collection of the bin edge locations along each axis.
- `cdf::Function` : Cumulative density function of the form f(xe, par0, par1, ..., parN),
   where xe is a bin edge and par0, ... are model parameters. The corresponding density must be normalized to unity 
   over the space covered by the histogram. If the model is multivariate, xe must be an array-like with shape (D, N),
   where D is the dimension and N is the number of points where the model is evaluated.
- `verbose::Int` : Verbosity level. 0: is no output.
- `grad::Union{Function, Nothing} : Optionally pass the gradient of the `cdf``. Has the same calling signature like the cdf, 
   but must return an array with the shape (K,), where K is the number of parameters. The gradient can be used by Minuit to 
   improve or speed up convergence.
- `use_pdf::Symbol`: Either `:none`, `:numerical`, or `:approximate`. If the model cdf is not available, but the model pdf is, 
   this option can be set to "numerical" or "approximate" to compute the integral of the pdf over the bin. The option "numerical" 
   uses numerical integration, which is accurate but computationally expensive and only supported for 1D histograms. The
   option "approximate" uses the zero-order approximation of evaluating the pdf at the bin center, multiplied with the bin area.
   This is fast and works in higher dimensions, but can lead to biased results if the curvature of the pdf inside the bin is significant.
- `names` : Optional names for each parameter of the model (in order). Must have the same length as there are model parameters.

"""
function ExtendedBinnedNLL(bincounts::AbstractArray, binedges::Union{AbstractArray, Tuple}, cdf::F; use_pdf=:none, verbose=0,  grad=nothing, names=()) where F<:Function
    ndim = ndims(bincounts)
    T = eltype(bincounts)
    ndim < 2 || binedges isa Tuple || throw(ArgumentError("binedges must be a Tuple"))
    params = model_parameters(cdf, names)
    if use_pdf == :numerical
        throw(ArgumentError("numerical calculation not implemented yet"))
    elseif use_pdf == :approximate || use_pdf == :none
        centers = bincenters(binedges)
        widths = binwidths(binedges)
        ledges = loweredges(binedges)
        uedges = upperedges(binedges)
    end
    ExtendedBinnedNLL{T,ndim,F}(BaseCost(verbose, params), bincounts, cdf, grad, use_pdf, sum(bincounts), centers, widths, ledges, uedges)
end

function value(cost::ExtendedBinnedNLL, args)
    mu = _pred(cost, args)
    return poisson_chi2(cost.bincounts, mu)
end

function grad(cost::ExtendedBinnedNLL, args)
    isnothing(cost.model_grad) && throw(ArgumentError("no gradient available"))
    mu = _pred(cost, args)
    gmu = _pred_grad(cost, args)
    poisson_chi2_grad(cost.bincounts, mu, gmu)
end

#---LeastSquares cost function--------------------------------------------------------------------
mutable struct LeastSquares <: CostFunction
    base::BaseCost
    data::Array{Float64, 2}
    mask::Union{Vector{Bool}, BitVector, Nothing}
    loss::Union{Function, Symbol}
    cost::Function
    cost_grad::Union{Function, Nothing}
    model::Function
    model_grad::Union{Function, Nothing}
    ndim::Int
end

"""
    LeastSquares(x::AbstractArray, y::AbstractVector, yerror, model::Function; 
                     loss=:linear, verbose=0, model_grad=nothing, names=(), mask=nothing)

Least-squares cost function (aka chisquare function).

Use this if you have data of the form (x, y +/- yerror), where x can be one-dimensional or multi-dimensional, 
but y is always one-dimensional.

## Arguments
- `x::AbstractArray` : Locations where the model is evaluated. If the model is multivariate, x must
        have shape (D, N), where D is the number of dimensions and N the number of data points.
- `y::AbstractVector` : Observed values. Must have the same length as x.
- `yerror` : Estimated uncertainty of observed values. Must have same shape as y or be a
   scalar, which is then broadcasted to same shape as y.
- `model::Function` : Function of the form f(x, par0, [par1, ...]) whose output is compared to
   observed values, where x is the location and par0, ... are model parameters. If the model is multivariate, 
   x has shape (D,), where D is the N the number of data points.
- `loss::Union{Symbol, Function}` : The loss function can be modified to make the fit robust against outliers. Only ``:linear` and
  `:soft_l1` are currently implemented, but users can pass any loss function as this argument. It should be a monotonic, twice differentiable function,
   which accepts the squared residual and returns a modified squared residual.
- `verbose::Int` :  Verbosity level. 0: is no output.
- `model_grad::Union{Function, Nothing}` : Optionally pass the gradient of the model. Has the same calling signature like
  the model, but must return an array with the shape (K,), where K is the number of parameters. 
  The gradient can be used by Minuit to improve or speed up convergence.
- `names` : Optional names for each parameter of the model (in order). Must have the same length as there are model parameters.
- `mask::Union{Vector{Bool}, BitVector, Nothing}` : Optional mask to select a subset of the data. Must have the same length as x.

## Notes
Alternative loss functions make the fit more robust against outliers by
weakening the pull of outliers. The mechanical analog of a least-squares fit is
a system with attractive forces. The loss function can be modified to make the
fit robust against outliers.
"""
function LeastSquares(x::AbstractArray, y::AbstractVector, yerror, model::Function; 
                     loss=:linear, verbose=0, grad=nothing, names=(), mask=nothing)
    #---Check in x is a vector of tuples-----------------------------------------------------------
    if ndims(x) == 1 && eltype(x) <: Tuple
        x = reduce(vcat, [[t...]' for t in x])
    end
    #---Check input arguments dimensions-----------------------------------------------------------
    len, ndim = Base.size(x, 1), Base.size(x, 2)
    length(y) == len || throw(DimensionMismatch("length of x and y do not match"))
    if length(yerror) == 1
        yerror = fill(yerror, len)
    else
        length(yerror) == len || throw(DimensionMismatch("length of x and yerror do not match"))
    end
    isnothing(mask) || length(mask) == len || throw(DimensionMismatch("length of x and mask do not match"))
    data = hcat(x, y, yerror)
    params = model_parameters(model, names)
    c =  LeastSquares(BaseCost(verbose, params), data, mask, loss, chi2, chi2_grad, model, grad, ndim)
    c.loss = loss
    return c
end

function getproperty(cost::LeastSquares, sym::Symbol)
    if hasproperty(getfield(cost, :base), sym)
        return getproperty(getfield(cost, :base), sym)
    elseif sym == :errordef
        return errordef(cost)
    elseif sym == :npar
        return npar(cost)
    elseif sym == :ndata
        return ndata(cost)
    elseif sym == :x
        return @view cost.data[:,1:cost.ndim]
    elseif sym == :y
        return @view cost.data[:,cost.ndim+1]
    elseif sym == :yerror
        return @view cost.data[:,cost.ndim+2]
    else
        return getfield(cost, sym)
    end
end
function setproperty!(cost::LeastSquares, sym::Symbol, value)
    if hasproperty(getfield(cost, :base), sym)
        return setproperty!(getfield(cost, :base), sym, value)
    elseif sym == :x
        cost.data[:,1:cost.ndim] = value
    elseif sym == :y
        cost.data[:,cost.ndim+1] = value
    elseif sym == :yerror
        cost.data[:,cost.ndim+2] = value
    elseif sym == :loss
        setfield!(cost, sym, value)
        if value isa Symbol
            if value == :linear
                cost.cost = chi2
                cost.cost_grad = chi2_grad
            elseif value == :soft_l1
                cost.cost = soft_l1_cost
                cost.cost_grad = soft_l1_cost_grad
            else
                throw(ArgumentError("Unknown loss $value"))
            end
        elseif value isa Function
            cost.cost = (y, ye, ym) -> sum(value(z_squared(y, ye, ym)))
            cost.cost_grad = nothing
        else
            throw(ArgumentError("loss must be Symbol or Function"))
        end
    else
        return setfield!(cost, sym, value)
    end
end

function value(cost::LeastSquares, args)
    x = isnothing(cost.mask) ? cost.x : cost.x[cost.mask]
    y = isnothing(cost.mask) ? cost.y : cost.y[cost.mask]
    ye = isnothing(cost.mask) ? cost.yerror : cost.yerror[cost.mask]
    if cost.ndim == 1
        ym = cost.model.(x, args...)
    else
        ym = [cost.model(x[i,:], args...) for i in 1:length(y)] 
    end
   return cost.cost(y, ye, ym)
end
function grad(cost::LeastSquares, args)
    x = isnothing(cost.mask) ? cost.x : cost.x[cost.mask]
    y = isnothing(cost.mask) ? cost.y : cost.y[cost.mask]
    ye = isnothing(cost.mask) ? cost.yerror : cost.yerror[cost.mask]
    if cost.ndim == 1
        ym = cost.model.(x, args...)
        ymg = cost.model_grad.(x, args...)
    else
        ym = [cost.model(x[i,:], args...) for i in 1:length(y)] 
        ymg = [cost.model_grad(x[i,:], args...) for i in 1:length(y)] 
    end
    return cost.cost_grad(y, ye, ym, ymg)
end
has_grad(cost::LeastSquares) = cost.cost_grad !== nothing && cost.model_grad !== nothing
ndata(cost::LeastSquares) = Base.size(cost.data, 1)
