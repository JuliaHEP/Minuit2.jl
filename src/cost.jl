"""
Standard cost functions to minimize for statistical fits.

    We provide these for convenience, so that you do not have to write your own for standard
    fits. The cost functions optionally use Numba to accelerate some calculations, if Numba
    is installed.
"""

abstract type CostFunction end
import Base: getproperty, setproperty!, propertynames, show
export LeastSquares, Constant, CostFunction, BaseCost, UnbinnedNLL, BinnedNLL
export value, grad, has_grad

"""
    BaseCost
"""
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

## Parameters
- `y` : Observed values.
- `ye` : Uncertainties of values.
- `ym` : Expected values.

# Returns
- Value of cost function.
"""
chi2(y, ye, ym) = sum(z_squared(y, ye, ym))
soft_l1_cost(y, ye, ym) = 2 * sum(sqrt.(1 .+ z_squared(y, ye, ym)) .- 1)

"""
Compute gradient of :func:`chi2`.

    Parameters
    ----------
    y : array-like  with shape (N,)
        Observed values.
    ye : array-like  with shape (N,)
        Uncertainties of values.
    ym : array-like with shape (N,)
        Expected values.
    gym : array-like with shape (K, N)
        Gradient of ym with respect to K model parameters.

    Returns
    -------
    array with shape (K,)
        Gradient of cost function with respect to model parameters.
    """
chi2_grad(y, ye, ym, gym) = -2 * sum((y .- ym) .* gym .* ye.^-2)
function soft_l1_cost_grad(y, ye, ym, gym)
    z = (y .- ym) ./ ye
    f = (1 .+ z.^2).^-0.5
    return -2 * sum(z ./ ye .* f .* gym)
end

log_or_zero(x) = x <= 0.0 ? 0.0 : log(x)
"""
    Compute asymptotically chi2-distributed cost for multinomially-distributed data.

    See Baker & Cousins, NIM 221 (1984) 437-442.

    Parameters
    ----------
    n : array-like
        Observed counts.
    mu : array-like
        Expected counts. Must satisfy sum(mu) == sum(n).

    Returns
    -------
    float
        Cost function value.

    Notes
    -----
    The implementation makes the result asymptotically chi2-distributed,
    which helps to maximise the numerical accuracy for Minuit.

"""
multinomial_chi2(n, mu) = 2 * sum(n .* (log_or_zero.(n) .- log_or_zero.(mu)))
multinomial_chi2_grad(n, mu, gmu) = - 2 * sum( n ./( mu .* gmu))

function model_parameters(model, names)
    # strip first argument from model
    args = get_argument_names(model)[2:end]
    if length(names) > 0
        if length(args) == length(names)
            params = names
        else
            throw(ValueError("length of names does not match number of model parameters"))
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
function show(io::IO, cost::CostFunction)
    modelname = hasproperty(cost, :model) ? "$(cost.model)" : "unknown"
    print(io, "$(typeof(cost)) cost function of \"$modelname\" with parameters $(cost.parameters)")
end
#---Constant cost function-------------------------------------------------------------------------
struct Constant <: CostFunction
    base::BaseCost
    value::Float64
end
Constant(value::Float64; verbose::Int=0) = Constant(BaseCost(verbose, []), value)
value(cost::Constant) = cost.value
grad(cost::Constant) = 0.0
has_grad(cost::Constant) = true
ndata(cost::Constant) = 0

#---UnbinnedNLL cost function---------------------------------------------------------------------
mutable struct UnbinnedNLL <: CostFunction
    base::BaseCost
    log::Bool
    data::Array{Float64}
    mask::Union{Vector{Bool}, BitVector, Nothing}
    model::Function
    model_grad::Union{Function, Nothing}
end

function UnbinnedNLL(data::AbstractArray, pdf::Function; log=false, verbose=0, mask=nothing, pdf_grad=nothing, names=())
    if ndims(data) == 1 && eltype(data) <: Tuple
        data = reduce(vcat, [[t...]' for t in data])
    end
    params = model_parameters(pdf, names)
    UnbinnedNLL(BaseCost(verbose, params), log, data, mask, pdf, pdf_grad)
end
errordef(::UnbinnedNLL) = NEGATIVE_LOG_LIKELIHOOD
has_grad(cost::UnbinnedNLL) = cost.model_grad !== nothing
ndata(cost::UnbinnedNLL) = size(cost.data, 1)

function getproperty(cost::UnbinnedNLL, sym::Symbol)
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
    g = cost.model_grad.(data, args)
    -2.0 * sum(g)
end

#---BinnedNLL cost function-----------------------------------------------------------------------
mutable struct BinnedNLL <: CostFunction
    base::BaseCost
    bincounts::Array{Float64}
    binedges::Union{AbstractArray, Tuple}
    model::Function
    model_grad::Union{Function, Nothing}
    use_pdf::Symbol
end

function BinnedNLL(bincounts::AbstractArray, binedges::Union{AbstractArray, Tuple}, cdf::Function; use_pdf=:none, verbose=0,  grad=nothing, names=())
    ndim = ndims(bincounts)
    ndim < 2 || binedges isa Tuple || throw(ArgumentError("binedges must be a Tuple"))
    params = model_parameters(cdf, names)
    BinnedNLL(BaseCost(verbose, params), bincounts, binedges, cdf, grad, use_pdf)
end

has_grad(cost::BinnedNLL) = cost.model_grad !== nothing
ndata(cost::BinnedNLL) = length(cost.bincounts)

function getproperty(cost::BinnedNLL, sym::Symbol)
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

function value(cost::BinnedNLL, args)
    s = Base.size(cost.bincounts)
    @show args[1], args[2], args[3], args[4]
    if cost.use_pdf == :approximate
        if cost.ndim == 1
            centers = [(cost.binedges[i] + cost.binedges[i+1])/2 for i in 1:s[1]]
            f = cost.model.(centers, args...)
        elseif cost.ndim == 2
            edgx, edgy = cost.binedges
            f = [cost.model(((edgx[i]+edgx[i+1])/2, (edgy[i]+edgy[i+1])/2), args...) for i in 1:s[1], j in 1:s[2]]
        else
            throw(ArgumentError("only 1D and 2D histograms are supported"))
        end
    elseif cost.use_pdf == :numerical
        throw(ArgumentError("numerical calculation not implemented yet"))
    else
        if cost.ndim == 1
            f = diff(cost.model.(cost.binedges, args...))
            f[f .<= 0] .= F64_TINY
        elseif cost.ndim == 2
            edgx, edgy = cost.binedges
            f = [cost.model((edgx[i+1], edgy[j+1]),args...) - cost.model((edgx[i], edgy[j]),args...) for i in 1:s[1], j in 1:s[2]]
        else
            throw(ArgumentError("only 1D and 2D histograms are supported"))
        end
    end
    r = multinomial_chi2(cost.bincounts, f)
    @show r
    return r
end

function grad(cost::BinnedNLL, args)
    if cost.use_pdf == :approximate
        centers = [0.5*(cost.binedges[i] + cost.binedges[i+1]) for i in 1:length(cost.binedges)-1]
        f = cost.model.(centers, args...)
    elseif cost.use_pdf == :numerical
        throw(ArgumentError("numerical calculation not implemented yet"))
    else
        f = diff(cost.model.(cost.binedges, args...))
        f[f .<= 0] = F64_TINY
    end
    gf = cost.model_grad.(centers, args)
    multinomial_chi2_grad(n, f, gf)
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

function LeastSquares(x::AbstractArray, y::AbstractVector, yerror, model::Function; 
                     loss=:linear, verbose=0, model_grad=nothing, names=(), mask=nothing)
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
    c =  LeastSquares(BaseCost(verbose, params), data, mask, loss, chi2, chi2_grad, model, model_grad, ndim)
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
ndata(cost::LeastSquares) = size(cost.data, 1)
