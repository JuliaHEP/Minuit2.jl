import Base: getproperty, setproperty!, setindex!, getindex, ==, length, copy

#---MnStrategy-------------------------------------------------------------------------------------
export MnStrategy
const MnStrategy = ROOT!Minuit2!MnStrategy

function getproperty(s::MnStrategy, sym::Symbol)
    if sym == :strategy
        return Strategy(s)
    elseif sym == :gradient_ncycles
        return GradientNCycles(s)
    elseif sym == :gradient_step_tolerance
        return GradientStepTolerance(s)
    elseif sym == :gradient_tolerance
        return GradientTolerance(s)
    elseif sym == :hessian_ncycles
        return HessianNCycles(s)
    elseif sym == :hessian_step_tolerance
        return HessianStepTolerance(s)
    elseif sym == :hessian_g2_tolerance
        return HessianG2Tolerance(s)
    elseif sym == :hessian_gradient_nccles
        return HessianGradientNCycles(s)
    elseif sym == :storage_level
        return StorageLevel(s)
    else
        return getfield(s, sym)
    end
end
function setproperty!(s::MnStrategy, sym::Symbol, value)
    if sym == :strategy
        if value == 0
            SetLowStrategy(s)
        elseif value == 1
            SetMediumStrategy(s)
        elseif value == 2
            SetHighStrategy(s)
        else
            throw(ArgumentError("invalid strategy"))
        end
    elseif sym == :gradient_ncycles
        SetGradientNCycles(s, value)
    elseif sym == :gradient_step_tolerance
        SetGradientStepTolerance(s, value)
    elseif sym == :gradient_tolerance
        SetGradientTolerance(s, value)
    elseif sym == :hessian_ncycles
        SetHessianNCycles(s, value)
    elseif sym == :hessian_step_tolerance
        SetHessianStepTolerance(s, value)
    elseif sym == :hessian_g2_tolerance
        SetHessianG2Tolerance(s, value)
    elseif sym == :hessian_gradient_ncycles
        SetHessianGradientNCycles(s, value)
    elseif sym == :storage_level
        SetStorageLevel(s, value)
    else
        setfield!(s, sym, value)
    end
end

#---MnUserCovariance--------------------------------------------------------------------------------
export MnUserCovariance
const MnUserCovariance = ROOT!Minuit2!MnUserCovariance

function getproperty(c::MnUserCovariance, sym::Symbol)
    if sym == :nrow
        return Nrow(c)
    else
        return getfield(c, sym)
    end
end
function getindex(c::MnUserCovariance, i::Int, j::Int)
    paren(c, i-1, j-1)[]
end
function ==(c1::MnUserCovariance, c2::MnUserCovariance)
    c1.nrow == c2.nrow && Data(c1)[] == Data(c2)[]
end
function (c::Union{MnUserCovariance, ConstCxxRef{MnUserCovariance}})(i::Int, j::Int)
    paren(c, i-1, j-1)[]
end

#---MnGlobalCorrelationCoeff------------------------------------------------------------------------
export MnGlobalCorrelationCoeff
const MnGlobalCorrelationCoeff = ROOT!Minuit2!MnGlobalCorrelationCoeff
function getproperty(c::Union{MnGlobalCorrelationCoeff, ConstCxxRef{MnGlobalCorrelationCoeff}}, sym::Symbol)
    if sym == :globalcc
        return GlobalCC(c)
    elseif sym == :is_valid
        return IsValid(c)
    else
        return getfield(c, sym)
    end
end
==(c1::MnGlobalCorrelationCoeff, c2::MnGlobalCorrelationCoeff) = 
    c1.globalcc == c2.globalcc && c1.is_valid == c2.is_valid    

#---MnUserParameterState----------------------------------------------------------------------------
export MnUserParameterState,set_value, set_error, set_limits, set_upper_limit, set_lower_limit, remove_limits, fix, release
const MnUserParameterState = ROOT!Minuit2!MnUserParameterState

length(p::MnUserParameterState) = length(MinuitParameters(p))
function getindex(p::MnUserParameterState, i::Int)
    1 <= i <= length(p) || throw(BoundsError(p, i))
    Parameter(p, i-1)
end
function getproperty(p::MnUserParameterState, sym::Symbol)
    if sym == :fval
        return Fval(p)
    elseif sym == :edm
        return Edm(p)
    elseif sym == :covariance
        return Covariance(p)[]
    elseif sym == :has_covariance
        return HasCovariance(p)
    elseif sym == :is_valid
        return IsValid(p)
    elseif sym == :globalcc
        return GlobalCC(p)
    else
        return getfield(p, sym)
    end
end
copy(p::MnUserParameterState) = ROOT!Minuit2!createMnUserParameterState(p)
==(p1::MnUserParameterState, p2::MnUserParameterState) = 
    length(p1) == length(p2) && all(p1[i] == p2[i] for i in 1:length(p1)) &&
    p1.fval == p2.fval && p1.edm == p2.edm && p1.covariance == p2.covariance &&
    p1.has_covariance == p2.has_covariance && p1.is_valid == p2.is_valid && 
    p1.globalcc.globalcc[] == p2.globalcc.globalcc[] && p1.globalcc.is_valid == p2.globalcc.is_valid

set_value(p::MnUserParameterState, i::Int, value) = SetValue(p, i-1, value)
set_error(p::MnUserParameterState, i::Int, error) = SetError(p, i-1, error)
set_limits(p::MnUserParameterState, i::Int, lower, upper) = SetLimits(p, i-1, lower, upper)
set_upper_limit(p::MnUserParameterState, i::Int, upper) = SetUpperLimit(p, i-1, upper)
set_lower_limit(p::MnUserParameterState, i::Int, lower) = SetLowerLimit(p, i-1, lower)
remove_limits(p::MnUserParameterState, i::Int) = RemoveLimits(p, i-1)
fix(p::MnUserParameterState, i::Int) = Fix(p, i-1)
release(p::MnUserParameterState, i::Int) = Release(p, i-1)

#---MinuitParameter---------------------------------------------------------------------------------
export MinuitParameter
const MinuitParameter = ROOT!Minuit2!MinuitParameter

function getproperty(s::ConstCxxRef{MinuitParameter}, sym::Symbol)
    if sym == :number
        return Number(s)
    elseif sym == :name
        return Name(s)
    elseif sym == :value
        return Value(s)
    elseif sym == :error
        return Error(s)
    elseif sym == :lower_limit
        return LowerLimit(s)
    elseif sym == :upper_limit
        return UpperLimit(s)
    elseif sym == :is_const
        return IsConst(s)
    elseif sym == :is_fixed
        return IsFixed(s)
    elseif sym == :has_limits
        return HasLimits(s)
    elseif sym == :has_lower_limit
        return HasLowerLimit(s)
    elseif sym == :has_upper_limit
        return HasUpperLimit(s)
    else
        return getfield(s, sym)
    end
end

==(p1::Union{MinuitParameter, ConstCxxRef{MinuitParameter}}, p2::Union{MinuitParameter, ConstCxxRef{MinuitParameter}}) = 
    p1.number == p2.number && p1.name == p2.name && p1.value == p2.value && p1.error == p2.error &&
    p1.lower_limit == p2.lower_limit && p1.upper_limit == p2.upper_limit && p1.is_const == p2.is_const &&
    p1.is_fixed == p2.is_fixed && p1.has_limits == p2.has_limits && p1.has_lower_limit == p2.has_lower_limit &&
    p1.has_upper_limit == p2.has_upper_limit

#---JuliaFcn-----------------------------------------------------------------------------------------
function getproperty(f::JuliaFcn, sym::Symbol)
    if sym == :nfcn
        return Nfcn(f)
    elseif sym == :ngrad
        return Ngrad(f)
    else
        return getfield(f, sym)
    end
end

function (f::JuliaFcn)(x)
    paren(f, x)
end

#---MnMigrad----------------------------------------------------------------------------------------
export MnMigrad
const MnMigrad = ROOT!Minuit2!MnMigrad

function (m::MnMigrad)(maxfcn = 0, tolerance = 0.1)
    paren(m, maxfcn, tolerance)
end

#---MnScan------------------------------------------------------------------------------------------
export MnScan
const MnScan = ROOT!Minuit2!MnScan
function (m::MnScan)(maxfcn = 0, tolerance = 0.1)
    paren(m, maxfcn, tolerance)
end

#---MnSimplex---------------------------------------------------------------------------------------
export MnSimplex
const MnSimplex = ROOT!Minuit2!MnSimplex
function (m::MnSimplex)(maxfcn = 0, tolerance = 0.1)
    paren(m, maxfcn, tolerance)
end

#---MnContours--------------------------------------------------------------------------------------
export MnContours
const MnContours = ROOT!Minuit2!MnContours
function (m::MnContours)(x, y, size )
    paren(m, x-1, y-1, size)
end

#---FunctionMinimum---------------------------------------------------------------------------------
export FunctionMinimum
const FunctionMinimum = ROOT!Minuit2!FunctionMinimum

function getproperty(f::FunctionMinimum, sym::Symbol)
    if sym == :state
        return UserState(f)
    elseif sym == :fval
        return Fval(f)
    elseif sym == :edm
        return Edm(f)
    elseif sym == :nfcn
        return Nfcn(f)
    elseif sym == :is_valid
        return IsValid(f)
    else
        return getfield(f, sym)
    end
end

FunctionMinimum(fcn::JuliaFcn, state, strategy, edm_goal) = 
   ROOT!Minuit2!createFunctionMinimum(fcn, state, strategy, edm_goal)

#---MnHesse-----------------------------------------------------------------------------------------
export MnHesse
const MnHesse = ROOT!Minuit2!MnHesse
function (m::MnHesse)(fcn::JuliaFcn, fmin::FunctionMinimum, maxcalls = 0)
    paren(m, fcn, fmin, maxcalls)
end

#---MnMinos-----------------------------------------------------------------------------------------
export MnMinos
const MnMinos = ROOT!Minuit2!MnMinos

#---MinosError------------------------------------------------------------------------------------
export MinosError
const MinosError = ROOT!Minuit2!MinosError

function getproperty(e::MinosError, sym::Symbol)
    if sym == :number
        return Parameter(e) + 1
    elseif sym == :lower
        return Lower(e)
    elseif sym == :upper
        return Upper(e)
    elseif sym == :is_valid
        return IsValid(e)
    elseif sym == :lower_valid
        return LowerValid(e)
    elseif sym == :upper_valid
        return UpperValid(e)
    elseif sym == :at_lower_limit
        return AtLowerLimit(e)
    elseif sym == :at_upper_limit
        return AtUpperLimit(e)
    elseif sym == :at_lower_max_fcn
        return AtLowerMaxFcn(e)
    elseif sym == :at_upper_max_fcn
        return AtUpperMaxFcn(e)
    elseif sym == :lower_new_min
        return LowerNewMin(e)
    elseif sym == :upper_new_min
        return UpperNewMin(e)
    elseif sym == :nfcn
        return NFcn(e)
    elseif sym == :min
        return Min(e)
    else
        return getfield(e, sym)
    end
end
Base.show(io::IO, m::Dict{String, MinosError}) = show.(io, collect(values(m)))

#---MnApplication-----------------------------------------------------------------------------------
export MnApplication
const MnApplication = ROOT!Minuit2!MnApplication


