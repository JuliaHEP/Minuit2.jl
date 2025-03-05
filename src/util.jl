import Base: getindex, setindex!, getproperty, length, lastindex, iterate, show, isapprox, eltype, show, ==

function keypair(m::Minuit, key::Union{Int, String})1
    if key isa Int
        1 <= key <= m.npar || throw(BoundsError("Parameter index out of range"))
        return key, m.names[key]
    else
        ikey = findfirst(isequal(key), m.names)
        ikey === nothing && throw(BoundsError("Parameter $key not found"))
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
    if key isa UnitRange{Int64}
        for (i, k) in enumerate(key)
            _set(view, k, value[i])
        end
        return
    elseif key isa Colon
        for k in 1:length(view)
            _set(view, k, value[k])
        end
        return
    end
    ipar, _ = keypair(view.minuit, key)
    _set(view, ipar, value)
end
length(view::AbstractView) = view.minuit.npar
lastindex(view::AbstractView) = view.minuit.npar
iterate(view::AbstractView, state=1) = state > length(view) ? nothing : (view[state], state + 1)
show(io::IO, view::AbstractView) = show(io, collect(view))
isapprox(view::AbstractView, v::Vector; kwargs...) = isapprox(collect(view), v; kwargs...)
==(view::AbstractView, v::Vector) = collect(view) == v

#---Concrete views---------------------------------------------------------------------------------

struct ValueView <: AbstractView;  minuit::Minuit; end
_get(view::ValueView, ipar::Int) = Value(view.minuit.last_state, ipar-1)
_set(view::ValueView, ipar::Int, value) = SetValue(view.minuit.last_state, ipar-1, value)
eltype(::ValueView) = Float64

struct ErrorView <: AbstractView;  minuit::Minuit; end
_get(view::ErrorView, ipar::Int) = Error(view.minuit.last_state, ipar-1)
_set(view::ErrorView, ipar::Int, value) = SetError(view.minuit.last_state, ipar-1, value)
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
    SetValue(state, ipar-1, val)
    SetError(state, ipar-1, err)
    return (lower, upper)
end
eltype(::LimitView) = Tuple{Float64, Float64}

struct MinosView <: AbstractView;  minuit::Minuit; end
function _get(view::MinosView, ipar::Int)
    _, key = keypair(view.minuit, ipar)
    view.minuit.mino === nothing && return nothing
    get(view.minuit.mino, key, nothing)
end
function _set(::MinosView, ::Int, value)
    throw(ArgumentError("Cannot set Minos results"))
end
eltype(::MinosView) = Union{MinosError, Nothing}

#---Parameters--------------------------------------------------------------------------------------
struct Param
    parameter::MinuitParameter
    minos::Union{MinosError, Nothing}
end

function getproperty(p::Param, name::Symbol)
    if name == :value
        return p.parameter.value
    elseif name == :error
        return p.parameter.error
    elseif name == :lower_limit
        return p.parameter.has_lower_limit ? p.parameter.lower_limit : -Inf
    elseif name == :upper_limit
        return p.parameter.has_upper_limit ? p.parameter.upper_limit : Inf
    elseif name == :is_fixed
        return p.parameter.is_fixed
    elseif name == :is_const
        return p.parameter.is_const
    elseif name == :has_limits
        return p.parameter.has_lower_limit || p.parameter.has_upper_limit
    elseif name == :name
        return p.parameter.name
    elseif name == :number
        return p.parameter.number
    elseif name == :merror
        return p.minos === nothing ? nothing : (p.minos.lower, p.minos.upper)
    else
        return getfield(p, name)
    end
end

struct ParamView <: AbstractView;  minuit::Minuit; end
function _get(view::ParamView, ipar::Int)
    _, key = keypair(view.minuit, ipar)
    mino = view.minuit.mino
    merror = !isnothing(mino) && haskey(mino, key) ?  mino[key] : nothing 
    Param(view.minuit.last_state[ipar][], merror)
end

function _set(::ParamView, ::Int, value)
    throw(ArgumentError("Cannot set Parameter values"))
end
eltype(::ParamView) = Param

function Base.show(io::IO, p::Param)
    print(io, "Parameter $(p.parameter.number): $(p.parameter.name) = $(p.parameter.value) ± $(p.parameter.error)")
    if p.parameter.has_lower_limit || p.parameter.has_upper_limit
        print(io, " [$(p.parameter.lower_limit), $(p.parameter.upper_limit)]")
    end
    if p.parameter.is_fixed
        print(io, " (fixed)")
    end
    if p.parameter.is_const
        print(io, " (constant)")
    end
    if p.minos !== nothing
        print(io, " Minos: $(p.minos.lower) $(p.minos.upper)")
    end
end
function Base.show(io::IO, view::ParamView)
    header = [" ", "Name", "Value", "Hesse Error",  "Minos-", "Minos+", "Limit-", "Limit+", "Fixed", "Const"]
    number = [p.number for p in view]
    names = [p.name for p in view]
    values = [p.value for p in view]
    errors = [p.error for p in view]
    minos_err_low = [ isnothing(p.minos) ? " " : p.minos.lower for p in view]
    minos_err_high = [ isnothing(p.minos) ? " " : p.minos.upper for p in view]
    limit_low = [p.lower_limit == -Inf ?  " " : p.lower_limit for p in view]
    limit_up  = [p.upper_limit ==  Inf ?  " " : p.upper_limit for p in view]
    fixed = [ p.is_fixed ? true : " " for p in view]
    consta = [ p.is_const ? true : " " for p in view]
    pretty_table(io, [number names values errors minos_err_low minos_err_high limit_low limit_up fixed consta]; header=header, alignment=:l)
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
    elseif name == :fmin
        fmin = getfield(m, :fmin)
        fmin === nothing && throw(ArgumentError("Minimization results not available"))
        return fmin
    elseif name == :fval
        return Fval(m.fmin)
    elseif name == :edm
        return Edm(m.fmin)
    elseif name == :nfcn
        return m.fcn.nfcn
    elseif name == :ngrad
        return m.fcn.ngrad
    elseif name == :ndof
        return ndof(m)
    elseif name == :niter
        return NIter(m.fmin)
    elseif name == :up
        return Up(m.fmin)
    elseif name == :errordef
        return ErrorDef(m.fcn)
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
    elseif name == :covariance
        return UserCovariance(m.fmin)[]
    elseif name == :has_accurate_covar
        return HasAccurateCovar(m.fmin)
    elseif name == :has_valid_parameters
        return HasValidParameters(m.fmin)
    elseif name == :has_reached_call_limit
        return HasReachedCallLimit(m.fmin)
    elseif name == :minos
        return MinosView(m)
    elseif name == :parameters
        return ParamView(m)
    else
        return getfield(m, name)
    end
end

function setproperty!(m::Minuit, name::Symbol, value)
    if name == :values
        ValueView(m)[:] = value
    elseif name == :errors
        ErrorView(m)[:] = value
    elseif name == :fixed
        FixedView(m)[:] = value
    elseif name == :limits
        LimitView(m)[:] = value
    else
        setfield!(m, name, value)
    end
end

function Base.show(io::IO, f::FunctionMinimum, m::Minuit=nothing)
    # additional info not in FunctionMinimum
    fval = string(round(f.fval, digits=3))
    nfcn = string(f.nfcn)
    if !isnothing(m)
        rc = reduced_chi2(m)
        !isnan(rc) && (fval = "$fval χ²/ndof=$(round(rc,digits=3))")
        m.ngrad > 0 && (nfcn = "nfcn=$nfcn ngrad=$(m.ngrad)")
        m.method == :scan && (nfcn = "nfcn=$(m.nfcn)")
    end

    data1 = ["FCN"        "Method"     "Ncalls"   "Iterations" "Up";
    fval       ifelse(isnothing(m), " ", m.method)     nfcn     f.niter     f.up;
    "Valid Min."     "Valid Param."	      "Above EDM"           "Call limit"              "Edm";
    f.is_valid	     f.has_valid_parameters  f.is_above_max_edm	f.has_reached_call_limit  f.edm;
    "Hesse failed"	 "Has cov."	          "Accurate"	        "Pos. def."               "Forced";
    f.hesse_failed   f.has_covariance     f.has_accurate_covar	f.has_posdef_covar        f.has_made_posdef_covar]
    pretty_table(io, data1; alignment=:l, show_header=false, body_hlines = [2,4])
end

function Base.show(io::IO, m::Minuit)
    if m.app === nothing
        print(io, "Minuit(FCN = $(m.funcname), X0 = $(m.x0), Method = $(m.method))")
    else
        if !isnothing(m.fmin)
            show(io, m.fmin, m)
        end
        show(io, m.parameters)
        if m.has_covariance
            cov = [m.covariance(i, j) for i in 1:m.npar, j in 1:m.npar]
            header = [" ", m.names...]
            data = hcat(m.names, cov)
            pretty_table(io, data; header=header, alignment=:l, show_header=true)
        end
    end
end

function get_argument_names(f)
    m = first(methods(f))
    tv, decls, file, line = Base.arg_decl_parts(m)
    [d[1] for d in decls[2:end]]
end

function get_nargs(f)
    m = first(methods(f))
    length(m.sig.parameters)-1
end