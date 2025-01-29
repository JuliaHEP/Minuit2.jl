import Base: getindex, setindex!, getproperty, length, iterate, show, isapprox

function keypair(m::Minuit, key::Union{Int, String})
    if key isa Int
        1 <= key <= m.npar || throw(ArgumentError("Parameter index out of range"))
        return key, m.names[key]
    else
        ikey = findfirst(isequal(key), m.names)
        ikey === nothing && throw(ArgumentError("Parameter $key not found"))
        return ikey, key
    end
end

#---AbstractView-----------------------------------------------------------------------------------
abstract type AbstractView end
function getindex(view::AbstractView, key)
    ipar , _ = keypair(view.minuit, key)
    _get(view, ipar)
end
function setindex!(view::AbstractView, value, key)
    ipar, _ = keypair(view.minuit, key)
    _set(view, ipar, value)
end
length(view::AbstractView) = view.minuit.npar
iterate(view::AbstractView, state=1) = state > length(view) ? nothing : (view[state], state + 1)
show(io::IO, view::AbstractView) = show(io, collect(view))
isapprox(view::AbstractView, v::Vector; kwargs...) = isapprox(collect(view), v; kwargs...)

#---Concrete views---------------------------------------------------------------------------------

struct ValueView <: AbstractView;  minuit::Minuit; end
_get(view::ValueView, ipar::Int) = Value(view.minuit.last_state, ipar-1)
_set(view::ValueView, ipar::Int, value) = SetValue(view.minuit.last_state, ipar, value)

struct ErrorView <: AbstractView;  minuit::Minuit; end
_get(view::ErrorView, ipar::Int) = Error(view.minuit.last_state, ipar-1)
_set(view::ErrorView, ipar::Int, value) = SetError(view.minuit.last_state, ipar, value)

struct FixedView <: AbstractView;  minuit::Minuit; end
_get(view::FixedView, ipar::Int) = IsFixed(view.minuit.last_state[ipar])
function _set(view::FixedView, ipar::Int, value::Bool)
    if value
        Fix(view.minuit.last_state, ipar-1)
    else
        Release(view.minuit.last_state, ipar-1)
    end
end

struct LimitView <: AbstractView;  minuit::Minuit; end
function _get(view::LimitView, ipar::Int)
    p = view.minuit.last_state[ipar]
    upper = HasUpperLimit(p) ? UpperLimit(p) : Inf
    lower = HasLowerLimit(p) ? LowerLimit(p) : -Inf
    return (upper, lower)
end
function _set(view::LimitView, ipar::Int, value)
    lower, upper = value
    state = view.minuit.last_state
    val = state[ipar].value
    err = state[ipar].error
    RemoveLimits(state, ipar-1)
    if isfinite(upper) && isfinite(lower)
        if upper < lower
            throw(ArgumentError("Upper limit must be greater than lower limit"))
        elseif upper == lower
            Fix(state, ipar-1)
        else
            SetLimits(state, ipar-1, lower, upper)
        end
    elseif isfinite(upper)
        SetUpperLimit(state, ipar-1, upper)
    elseif isfinite(lower)
        SetLowerLimit(state, ipar-1, lower)
    end
    # bug in Minuit2: must set parameter value and error again after changing limits
    val < lower && (val = lower)
    val > upper && (val = upper)
    SetValue(state, ipar, val)
    SetError(state, ipar, err)
    return (upper, lower)
end

#---Minuit property accessors-----------------------------------------------------------------------
function getproperty(m::Minuit, name::Symbol)
    if name == :values
        return ValueView(m)
    elseif name == :errors
        return ErrorView(m)
    elseif name == :fixed
        return FixedView(m)
    elseif name == :limits
        return LimitView(m)
    elseif name == :fval
        return Fval(m.fmin)
    elseif name == :edm
        return Edm(m.fmin)
    elseif name == :nfcn
        return NFcn(m.fmin)
    elseif name == :niter
        return NIter(m.fmin)
    elseif name == :up
        return Up(m.fmin)
    elseif name == :is_valid
        return IsValid(m.fmin)
    else
        return getfield(m, name)
    end
end

function matrix(m::Minuit; correlation=false)
    state = State(m.app)
    if HasCovariance(state)
        cov = Covariance(state)
        a = [cov(i, j) for i in 1:m.npar, j in 1:m.npar]
        if correlation
            d = diag(a) .^ 0.5
            a ./= d .* d'
        end
        return a
    end
    return
end
