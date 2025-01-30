import Base: getindex, setindex!, getproperty, length, iterate, show, isapprox, eltype, show

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
eltype(::ValueView) = Float64

struct ErrorView <: AbstractView;  minuit::Minuit; end
_get(view::ErrorView, ipar::Int) = Error(view.minuit.last_state, ipar-1)
_set(view::ErrorView, ipar::Int, value) = SetError(view.minuit.last_state, ipar, value)
eltype(::ErrorView) = Float64

struct FixedView <: AbstractView;  minuit::Minuit; end
_get(view::FixedView, ipar::Int) = IsFixed(view.minuit.last_state[ipar])
function _set(view::FixedView, ipar::Int, value::Bool)
    if value
        Fix(view.minuit.last_state, ipar-1)
    else
        Release(view.minuit.last_state, ipar-1)
    end
end
eltype(::FixedView) = Bool

struct LimitView <: AbstractView;  minuit::Minuit; end
function _get(view::LimitView, ipar::Int)
    p = view.minuit.last_state[ipar]
    upper = HasUpperLimit(p) ? UpperLimit(p) : Inf
    lower = HasLowerLimit(p) ? LowerLimit(p) : -Inf
    return (lower, upper)
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
    return (lower, upper)
end
eltype(::LimitView) = Tuple{Float64, Float64}

struct MinosView <: AbstractView;  minuit::Minuit; end
function _get(view::MinosView, ipar::Int)
    _, key = keypair(view.minuit, ipar)
    get(view.minuit.mino, key, nothing)
end
function _set(::MinosView, ::Int, value)
    throw(ArgumentError("Cannot set Minos results"))
end
eltype(::MinosView) = Union{MinosError, Nothing}

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
    elseif name == :is_above_max_edm
        return IsAboveMaxEdm(m.fmin)
    elseif name == :has_parameters_at_limit
        return HasParametersAtLimit(m.fmin)
    elseif name == :has_accurate_covar
        return HasAccurateCovar(m.fmin)
    elseif name == :has_posdef_covar
        return HasPosDefCovar(m.fmin)
    elseif name == :has_made_posdef_covar
        return HasMadePosDefCovar(m.fmin)
    elseif name == :hesse_failed
        return HesseFailed(m.fmin)
    elseif name == :has_covariance
        return HasCovariance(m.fmin)
    elseif name == :has_accurate_covar
        return HasAccurateCovar(m.fmin)
    elseif name == :has_valid_parameters
        return HasValidParameters(m.fmin)
    elseif name == :has_reached_call_limit
        return HasReachedCallLimit(m.fmin)
    elseif name == :minos
        return MinosView(m)
    else
        return getfield(m, name)
    end
end


function Base.show(io::IO, m::Minuit)
    if m.app === nothing
        print(io, "Minuit(FCN = $(m.funcname), X0 = $(m.x0), Method = $(m.method))")
        return
    end
    if !isnothing(m.fmin)
        #---Print the minimization results-------------------------------------------------------------
        data1 = ["FCN"        "Method"     "Ncalls"   "Iterations" "Up";
                 m.fval       m.method      m.nfcn     m.niter     m.up;
                 "Valid Min."     "Valid Param."	      "Above EDM"           "Call limit"              "Edm";
                 m.is_valid	      m.has_valid_parameters  m.is_above_max_edm	m.has_reached_call_limit  m.edm;
                 "Hesse failed"	  "Has cov."	          "Accurate"	        "Pos. def."               "Forced";
                 m.hesse_failed   m.has_covariance        m.has_accurate_covar	m.has_posdef_covar        m.has_made_posdef_covar]
        pretty_table(io, data1; alignment=:l, show_header=false, body_hlines = [2,4])
    end
    if !isnothing(m.app)
        #---Print the parameters
        npar     = m.npar
        header = ["Name", "Value", "Hesse Error",  "Minos-", "Minos+", "Limit-", "Limit+", "Fixed"]
        names = m.names
        values = m.values |> collect
        errors = m.errors |> collect
        if m.mino === nothing
            minos_err_low = [ " " for i in 1:npar]
            minos_err_high = [ " " for i in 1:npar]
        else
            minos_err_low =  [ isnothing(m.minos[i]) ? " " : m.minos[i].lower for i in 1:npar]
            minos_err_high = [ isnothing(m.minos[i]) ? " " : m.minos[i].upper for i in 1:npar]
        end
        limit_low = [m.limits[i][1] == -Inf ?  " " : m.limits[i][1] for i in 1:npar]
        limit_up  = [m.limits[i][2] ==  Inf ?  " " : m.limits[i][2] for i in 1:npar]
        fixed = [ m.fixed[i] ? true : " " for i in 1:npar]
        pretty_table(io, [names values errors minos_err_low minos_err_high limit_low limit_up fixed]; header=header, alignment=:l)
    end
end