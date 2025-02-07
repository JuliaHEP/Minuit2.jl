"""
Standard cost functions to minimize for statistical fits.

    We provide these for convenience, so that you do not have to write your own for standard
    fits. The cost functions optionally use Numba to accelerate some calculations, if Numba
    is installed.
"""

abstract type CostFunction end
import Base: getproperty, setproperty!, propertynames, show
export LeastSquares, Constant, CostFunction, BaseCost, value, grad, has_grad

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

unbinned_nll(x) = return -sum(log.(x .+ F64_TINY))
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
show(io::IO, cost::CostFunction) = print(io, "$(typeof(cost)) cost function with $(cost.parameters) parameters")

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

#---LeastSquares cost function--------------------------------------------------------------------
mutable struct LeastSquares <: CostFunction
    base::BaseCost
    data::Array{Float64, 2}
    loss::Function
    cost::Function
    cost_grad::Union{Function, Nothing}
    model::Function
    model_grad::Union{Function, Nothing}
    ndim::Int
end

function LeastSquares(x::AbstractVector{Float64}, y::AbstractVector{Float64}, yerror::AbstractVector{Float64}, model::Function; 
                     loss=nothing, verbose=0, model_grad=nothing, names=())
    ndim = Base.size(x, 2)
    data = hcat(x, y, yerror)
    loss = loss === nothing ? chi2 : loss
    params = model_parameters(model, names)
    return LeastSquares(BaseCost(verbose, params), data, loss, chi2, chi2_grad, model, model_grad, ndim)
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
    else
        return setfield!(cost, sym, value)
    end
end

function value(cost::LeastSquares, args)
    x  = @view cost.data[:,1:cost.ndim]
    y  = @view cost.data[:,cost.ndim+1]
    ye = @view cost.data[:, cost.ndim+2]
    ym = cost.model.(x, args...)
   return cost.cost(y, ye, ym)
end
function grad(cost::LeastSquares, args)
    x  = @view cost.data[:,1:cost.ndim]
    y  = @view cost.data[:,cost.ndim+1]
    ye = @view cost.data[:, cost.ndim+2]
    ym  = cost.model.(x, args...) |> vec
    ymg = cost.model_grad.(x, args...)
    return cost.cost_grad(y, ye, ym, ymg)
end
has_grad(cost::LeastSquares) = cost.cost_grad !== nothing && cost.model_grad !== nothing
ndata(cost::LeastSquares) = size(cost.data, 1)









