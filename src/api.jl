using CxxWrap
using PrettyTables
using LinearAlgebra
using Distributions

export Minuit, MinosError, migrad!, hesse!, minos!
export values, fval, method, nfcn, niter, up, isvalid, matrix, minos, errors, merrors
export name, valid, lower, upper, symmetric

abstract type OptimizationResults end

struct MinosError
    ipar::Int
    name::String
    valid::Bool
    lower::Float64
    upper::Float64
    merror::ROOT!Minuit2!MinosError
end
name(me::MinosError) = me.name
valid(me::MinosError) = me.valid
lower(me::MinosError) = me.lower
upper(me::MinosError) = me.upper

mutable struct Minuit
    funcname::String                            # Name of the function
    fcn::JuliaFcn                               # The function to minimize    
    scfunc::CxxWrap.SafeCFunction               # The safe C function (reference to keep alive)
    x0::AbstractVector                          # Initial parameters values 
    npar::Int                                   # Number of parameters
    method::Symbol                              # The minimization method
    tolerance::Real                             # The tolerance for the minimization    
    kwargs::Dict{Symbol, Any}                   # Keyword arguments
    userpars::ROOT!Minuit2!MnUserParameterState        # The user parameters
    app::Union{ROOT!Minuit2!MnApplication, Nothing}    # The Minuit application
    fmin::Union{ROOT!Minuit2!FunctionMinimum, Nothing} # The result of the minimization
    minos::Union{Dict{String, MinosError}, Nothing}    # The Minos errors
end

function Base.values(m::Minuit)
    state = State(m.app)
    [Value(state, i) for i in 0:m.npar-1]
end

fval(m::Minuit) = Fval(State(m.app))
method(m::Minuit) = m.method
nfcn(m::Minuit) = NFcn(m.fmin)
niter(m::Minuit) = NIter(m.fmin)
up(m::Minuit) = Up(m.fmin)
isvalid(m::Minuit) = IsValid(m.fmin)
function matrix(m::Minuit; correlation=false)
    state = State(m.app)
    if HasCovariance(state)
        cov = Covariance(state)
        a = [paren(cov, i, j) for i in 0:m.npar-1, j in 0:m.npar-1]
        if correlation
            d = diag(a) .^ 0.5
            a ./= d .* d'
        end
        return a
    end
    return
end
minos(m::Minuit) = m.minos === nothing ? Dict{String, MinosError}() : minos!(m).minos
minos(m::Minuit, key) = m.minos[key]
errors(m::Minuit) = [Error(State(m.app), i) for i in 0:m.npar-1]
merrors(m::Minuit) = [m.minos[Name(m.userpars, i)] for i in 0:m.npar-1]
Base.show(io::IO, m::Dict{String, MinosError}) = show.(io, collect(values(m)))

function normalize_par(m::Minuit, key::Union{Int, String})
    if isa(key, Int)
        return key - 1, Name(State(m.app), key-1)
    end
    if isa(key, String)
        names = [Name(State(m.app), i) for i in 0:m.npar-1]
        if key in names
            return findfirst(isequal(key), names), key
        end
    end
    throw(ArgumentError("Parameter $key not found"))
end

"""
    get_argument_names(f)

Return the names of the arguments of the function `f`.
"""
function get_argument_names(f)
    m = first(methods(f))
    tv, decls, file, line = Base.arg_decl_parts(m)
    [d[1] for d in decls[2:end]]
end

"""
    get_nargs(f)

Return the number of arguments of the function `f`.
"""
function get_nargs(f)
    m = first(methods(f))
    length(m.sig.parameters)-1
end

"""
    Minuit(fcn, x0=(); grad=nothing, error=(), errordef=1.0, names=(), method=:migrad, maxfcn=0, tolerance=0, kwargs...)

Initialize a Minuit object.

This does not start the minimization or perform any other work yet. Algorithms 
are started by calling the corresponding methods.

## Arguments
- `fcn::Function` : Function to minimize. See notes for details on what kind of functions are accepted.
- `x0::AbstractArray` : Starting values for the minimization. See notes for details on how to set starting values.
- `grad::Union{Function,Nothing}` : If `grad` is a function, it must be a function that calculates the gradient
  and returns an iterable object with one entry for each parameter, which is
  the derivative of `fcn` for that parameter. If `nothing` (default), Minuit will compute the gradient numerically.
- `error::AbstractArray` : Starting values for the errors of the parameters. If not provided, Minuit will use 0.1 for all parameters.
- `errordef::Real` : Error definition of the function. Minuit defines parameter errors as the change in parameter 
  value required to change the function value by `errordef`. Normally, for chisquared fits it is 1, and for negative log likelihood,
  its value is 0.5. If the user wants instead the 2-sigma errors for chisquared fits, it becomes 4, as `Chi2(x+n*sigma) = Chi2(x) + n*n`.
- `names::Sequence{String}` : Names of the parameters. If not provided, Minuit will try to extract the names from the function signature.
- `method::Symbol` : The minimization algorithm to use. Possible values are `:migrad`, `:simplex`
- `maxfcn::Int` : Maximum number of function calls. If set to 0, Minuit will use a default value.
- `tolerance::Real` : Tolerance for the minimization. If set to 0, Minuit will use a default value.
- `kwargs` : Additional keyword arguments. Starting values for the minimization as keyword arguments. See notes for details on how
  to set starting values.

## Notes
### Function to minimize
By default, Minuit assumes that the callable `fcn` behaves like chi-square function, meaning that the function minimum in repeated identical random
experiments is chi-square distributed up to an arbitrary additive constant. This
is important for the correct error calculation. If `fcn` returns a
log-likelihood, one should multiply the result with -2 to adapt it. If the
function returns the negated log-likelihood, one can alternatively set the
attribute `errordef` to make Minuit calculate errors properly.

Minuit reads the function signature of `fcn` to detect the number and names of
the function parameters. Two kinds of function signatures are understood.

a.  Function with positional arguments.

The function has positional arguments, one for each fit parameter. Example:

    fcn(a, b, c) =  ...

The parameters a, b, c must accept a real number
Minuit automatically detects the parameters names in this case.

b.  Function with arguments passed as a single AbstractArray.

The function has a single argument which is an AbstractArray. Example:

    function fcn_v(x) =  ...

To use this form, starting values (`x0`) needs to be passed to Minuit in form of
a `Tuple` or `Vector`. In some cases, the detection may fail, and will be necessary to use the `names` keyword to set the parameter names.

### Parameter initialization
Initial values for the minimization can be set with positional arguments or
via keywords. This is best explained through an example:

    fcn(x, y) =  (x - 2)^2 + (y - 3)^2

The following ways of passing starting values are equivalent:

    Minuit(fcn, x=1, y=2)
    Minuit(fcn, y=2, x=1) # order is irrelevant when keywords are used ...
    Minuit(fcn, [1,2])    # ... but here the order matters

Positional arguments can also be used if the function has no signature:

    fcn_no_sig(args...) =  ...
    Minuit(fcn_no_sig, [1,2])

If the arguments are explicitly named with the `names` keyword described
further below, keywords can be used for initialization:

    Minuit(fcn_no_sig, x=1, y=2, names=("x", "y"))  # this also works

If the function accepts a single AbstractVector, then the initial values
must be passed as a single array-like object:

    fcn_v(x) = return (x[1] - 2) ** 2 + (x[2] - 3) ** 2
    Minuit(fcn_v, (1, 2))

Setting the values with keywords is not possible in this case. Minuit
deduces the number of parameters from the length of the initialization
sequence.
"""
function Minuit(fcn, x0=(); grad=nothing, error=(), errordef=1.0, names=(), method=:migrad, maxfcn=0, tolerance=0.1, kwargs...)
    #---Check if the function has a list of parameters or a single array---------------------------
    if get_nargs(fcn) == 1 && length(x0) > 1   # vector form
        vf = fcn
        if names === ()
            n1 = get_argument_names(fcn)[1]
            names = ["$n1[$i]" for i in 1:length(x0)]
        end
    else
        vf(x) = fcn(x...)
        if names === ()
            names = get_argument_names(fcn)
        end
    end
    #---Create the safe C function and the Julia function------------------------------------------
    sf = eval( quote  
            @safe_cfunction($vf, Float64, (ConstCxxRef{StdVector{Float64}},)) 
         end )
    jf = JuliaFcn(sf)
    #---Set the error definition-------------------------------------------------------------------
    errordef != 1.0 && SetErrorDef(jf, errordef)
    funcname = string(first(methods(fcn)))
    funcname = funcname[1:findfirst('@',funcname)-2]
    # If x0 is not provided, use the keyword arguments of the form <par>=<value>-------------------
    if x0 === ()
        x0 = [kwargs[Symbol(n)] for n in names]
    end
    #---Create the user parameters-----------------------------------------------------------------
    userpars = ROOT!Minuit2!MnUserParameterState()
    npar = length(x0)
    for i in 1:npar
        name = i > length(names) ? "p$i" : names[i]
        ei = i > length(error) ? haskey(kwargs, Symbol("error_", name)) ? kwargs[Symbol("error_", name)] : 0.1 : error[i]
        xi = x0[i]
        Add(userpars, name, xi, ei)
        if haskey(kwargs, Symbol("limit_", name))
            limit = kwargs[Symbol("limit_", name)]
            if limit[1] == -Inf
                SetUpperLimit(userpars, i-1, limit[2])
            elseif limit[2] == Inf
                SetLowerLimit(userpars, i-1, limit[1])
            else
                SetLimits(userpars, i-1, limit[1], limit[2])
            end
        end
        haskey(kwargs, Symbol("fix_", name)) && Fix(userpars, i-1)
    end
    kwargs = Dict(:method=>method, :maxfcn=>maxfcn, :tolerance=>tolerance)
    #migrad = ROOT!Minuit2!MnMigrad(jf, userpars)
    Minuit(funcname, jf, sf, x0, npar, method, tolerance, kwargs, userpars, nothing, nothing, nothing)
end

function Base.show(io::IO, m::Minuit)
    if m.app === nothing
        print(io, "Minuit(FCN = $(m.funcname), X0 = $(m.x0), Method = $(m.kwargs[:method]))")
        return
    end
    if !isnothing(m.fmin)
        #---Print the minimization results-------------------------------------------------------------
        min = m.fmin
        data1 = ["FCN"        "Method"  "Ncalls" "Iterations" "up";
                 fval(m)       method(m)  nfcn(m) niter(m)  up(m);
                 "Valid Min." "Valid Param."	      "Above EDM"  " "	                        "Reached call limit";
                 isvalid(m)	   HasValidParameters(min) IsAboveMaxEdm(min)	" "                 HasReachedCallLimit(min);
                 "Hesse failed"	  "Has cov."	          "Accurate"	        "Pos. def."         "Forced";
                 HesseFailed(min) HasValidCovariance(min) HasAccurateCovar(min)	HasPosDefCovar(min) HasMadePosDefCovar(min)]
        pretty_table(io, data1; alignment=:l, show_header=false, body_hlines = [2,4])
    end
    if !isnothing(m.app)
        #---Print the parameters
        userpars = State(m.app)
        npar     = m.npar
        mnpars = [Parameter(m.app, i) for i in 0:npar-1]
        header = ["Name", "Value", "Hesse Error",  "Minos-", "Minos+", "Limit-", "Limit+", "Fixed"]
        names = [Name(userpars, i) for i in 0:npar-1]
        values = [Value(userpars, i) for i in 0:npar-1]
        errors = [Error(userpars, i) for i in 0 : npar-1]
        if m.minos === nothing
            minos_err_low = [ " " for i in 0:npar-1]
            minos_err_high = [ " " for i in 0:npar-1]
        else
            minos_err_low = [ haskey(m.minos,Name(userpars, i-1)) ? m.minos[Name(userpars, i-1)].lower : " " for i in 1:npar]
            minos_err_high = [ haskey(m.minos,Name(userpars, i-1)) ? m.minos[Name(userpars, i-1)].upper : " " for i in 1:npar]
        end
        limit_low = [ HasLowerLimit(mnpars[i]) ? LowerLimit(mnpars[i]) : " " for i in 1:npar]
        limit_up = [ HasUpperLimit(mnpars[i]) ? UpperLimit(mnpars[i]) : " " for i in 1:npar]
        fixed = [ IsFixed(mnpars[i]) ? true : " " for i in 1:npar]
        pretty_table(io, [names values errors minos_err_low minos_err_high limit_low limit_up fixed]; header=header, alignment=:l)
    end
end

"""
    migrad!(m::Minuit, strategy=1)

Run Migrad minimization.

Migrad from the Minuit2 library is a robust minimisation algorithm which earned
its reputation in 40+ years of almost exclusive usage in high-energy physics.
How Migrad works is described in the [Minuit]() paper. It uses first and
approximate second derivatives to achieve quadratic convergence near the
minimum.

## Parameters
- `m::Minuit` : The Minuit object to minimize.
- `strategy::Int` : The minimization strategy. The default value is 1, which is
    the recommended value for most cases. The value 0 is faster, but less
    reliable. The value 2 is slower, but more reliable. The value 3 or higher is slower,
    but even more reliable.
"""
function migrad!(m::Minuit, strategy=1)
    migrad = ROOT!Minuit2!MnMigrad(m.fcn, m.userpars, ROOT!Minuit2!MnStrategy(strategy))
    min = paren(migrad)   # calls the operator () to do the minimization
    #---Update the Minuit object with the results---------------------------------------------------
    m.app = migrad
    m.fmin = min
    m.minos = nothing
    return m
end

function edm_goal(m::Minuit; migrad_factor=false)
    edm_goal = max( m.tolerance * Up(m.fcn), 4 * sqrt(eps()))
    migrad_factor && (edm_goal *= 2e-3)
    edm_goal
end

"""
    hesse!(m::Minuit, ncall::Int=0)

Run Hesse algorithm to compute asymptotic errors.

The Hesse method estimates the covariance matrix by inverting the matrix of
[second derivatives (Hesse matrix) at the minimum](https://en.wikipedia.org/wiki/Hessian_matrix). 
To get parameters correlations, you need to use this. The Minos algorithm is another way to 
estimate parameter uncertainties, see function `minos`.

## Arguments
- `ncall::Int=0` : Approximate upper limit for the number of calls made by the Hesse algorithm.
  If set to 0, use the adaptive heuristic from the Minuit2 library.

## Notes
The covariance matrix is asymptotically (in large samples) valid. By valid we
mean that confidence intervals constructed from the errors contain the true
value with a well-known coverage probability (68 % for each interval). In finite
samples, this is likely to be true if your cost function looks like a
hyperparabola around the minimum.

In practice, the errors very likely have correct coverage if the results from
Minos and Hesse methods agree. It is possible to construct artificial functions
where this rule is violated, but in practice it should always work.
"""
function hesse!(m::Minuit; strategy=1, maxcalls=0)
    hesse = ROOT!Minuit2!MnHesse(strategy)
    if m.fmin === nothing || !IsValid(m.fmin)
        migrad!(m)
    end
    paren(hesse, m.fcn, m.fmin, maxcalls)
    #---Update the Minuit object with the results---------------------------------------------------
    #fmin = ROOT!Minuit2!createFunctionMinimum(m.fcn, State(m.app), ROOT!Minuit2!MnStrategy(strategy), edm_goal(m, migrad_factor=true))
    #m.fmin = fmin
    return m
end

"""
    minos!(m::Minuit, name::String)
    
Run Minos algorithm to compute asymmetric errors for a single parameter.

The Minos algorithm uses the profile likelihood method to compute (generally
asymmetric) confidence intervals. It scans the negative log-likelihood or
(equivalently) the least-squares cost function around the minimum to construct a
confidence interval.

## Arguments
- `m::Minuit` : The Minuit object to minimize.
- `parameters::AbstractVector{String}` : Names of the parameters to compute the Minos errors for.
- `cl::Number` : Confidence level of the interval. If not set, a standard 68 %
   interval is computed (default). If 0 < cl < 1, the value is interpreted as
   the confidence level (a probability). For convenience, values cl >= 1 are
   interpreted as the probability content of a central symmetric interval
   covering that many standard deviations of a normal distribution. For
   example, cl=1 is interpreted as 68.3 %, and cl=2 is 84.3 %, and so on. Using
   values other than 0.68, 0.9, 0.95, 0.99, 1, 2, 3, 4, 5 require the scipy module.
- `ncall::Int` : Limit the number of calls made by Minos. If 0, an adaptive internal
   heuristic of the Minuit2 library is used (Default: 0).

## Notes
Asymptotically (large samples), the Minos interval has a coverage probability
equal to the given confidence level. The coverage probability is the probability
for the interval to contain the true value in repeated identical experiments.

The interval is invariant to transformations and thus not distorted by parameter
limits, unless the limits intersect with the confidence interval. As a
rule-of-thumb: when the confidence intervals computed with the Hesse and Minos
algorithms differ strongly, the Minos intervals are preferred. Otherwise, Hesse
intervals are preferred.

Running Minos is computationally expensive when there are many fit parameters.
Effectively, it scans over one parameter in small steps and runs a full
minimisation for all other parameters of the cost function for each scan point.
This requires many more function evaluations than running the Hesse algorithm.
"""
function minos!(m::Minuit; cl=0.68, ncall=0, parameters=(), strategy=1)
    cl >= 1.0 && (cl = cdf(Chisq(1), cl^2))    # convert sigmas into confidence level
    factor = quantile(Chisq(1), cl)            # convert confidence level to errordef

    # If the function minimum does not exist or the last state was modified, run Hesse
    if m.fmin === nothing || !IsValid(m.fmin)
        hesse!(m)
    end    
    if !isvalid(m)
        throw(ErrorException("Function minimum is not valid"))
    end
    #---Get the parameters to run Minos-------------------------------------------------------------
    if length(parameters) == 0
        ipars = [ipar for ipar in 0:m.npar-1 if !IsFixed(Parameter(State(m.app), ipar))]
    else
        ipars = []
        for par in parameters
            ip, pname = normalize_par(m, par)
            if IsFixed(Parameter(State(m.app), ip))
                warn("Cannot scan over fixed parameter $pname")
            else
                push!(ipars, ip)
            end
        end
    end
    #---Run Minos for each parameter----------------------------------------------------------------
    minos = ROOT!Minuit2!MnMinos(m.fcn, m.fmin, strategy)
    merrors = Dict{String, MinosError}()
    for ipar in ipars
        mn = Minos(minos, ipar, ncall, m.tolerance)
        me = MinosError(ipar, Name(m.userpars, ipar), IsValid(mn), Lower(mn), Upper(mn), mn)
        merrors[Name(m.userpars, ipar)] = me
    end
    m.minos = merrors
    return m
end

function Base.show(io::IO, me::MinosError)
    e = me.merror
    header = [me.name, me.valid ? "valid" : "invalid", " " ]
    data = [ "Error"    me.lower      me.upper;
             "Valid"    LowerValid(e) UpperValid(e);
             "At Limit" AtLowerLimit(e) AtUpperLimit(e);
             "Max Fcn"  AtLowerMaxFcn(e) AtUpperMaxFcn(e);
             "New Min"  LowerNewMin(e) UpperNewMin(e)]
    pretty_table(io, data; header=header, alignment=:l)
end

"""
    contour(m::Minuit, x, y; size=50, bound=2, grid=(), subtract_min=false)

Get a 2D contour of the function around the minimum.

It computes the contour via a function scan over two parameters, while keeping
all other parameters fixed. The related :meth:`mncontour` works differently: for
each pair of parameter values in the scan, it minimises the function with the
respect to all other parameters.

This method is useful to inspect the function near the minimum to detect issues
(the contours should look smooth). It is not a confidence region unless the
function only has two parameters. Use :meth:`mncontour` to compute confidence
regions.

## Arguments
    x : int or str
        First parameter for scan.
    y : int or str
        Second parameter for scan.
    size : int or tuple of int, optional
        Number of scanning points per parameter (Default: 50). A tuple is
        interpreted as the number of scanning points per parameter.
        Ignored if grid is set.
    bound : float or tuple of floats, optional
        If bound is 2x2 array, [[v1min,v1max],[v2min,v2max]].
        If bound is a number, it specifies how many :math:`\sigma`
        symmetrically from minimum (minimum+- bound*:math:`\sigma`).
        (Default: 2). Ignored if grid is set.
    grid : tuple of array-like, optional
        Grid points to scan over. If grid is set, size and bound are ignored.
    subtract_min :
        Subtract minimum from return values (Default: False).

##  Returns
    array of float
        Parameter values of first parameter.
    array of float
        Parameter values of second parameter.
    2D array of float
        Function values.

        x: Union[int, str],
        y: Union[int, str],
        *,
        size: int = 50,
        bound: Union[float, Iterable[Tuple[float, float]]] = 2,
        grid: Tuple[ArrayLike, ArrayLike] = None,
        subtract_min: bool = False,

"""
function contour(m::Minuit, x, y; numpoints=100, sigma=1, cl=0.68, ncall=0, strategy=1)

end
