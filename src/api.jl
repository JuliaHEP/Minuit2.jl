using CxxWrap
using PrettyTables
using LinearAlgebra
using Distributions

import Base: values, show

export FCN, Minuit, migrad!, hesse!, matrix, minos!, simplex!, scan!, contour, mncontour, profile, mnprofile

"""
    Minuit structure

## Direct or calculated fields   
- `funcname::String` : Name of the function
- `x0::AbstractVector` : Initial parameters values
- `method::Symbol` : The minimization algorithm to use. Possible values are `:migrad`, `:simplex`
- `tolerance::Real` : Tolerance for the minimization. If set to 0, Minuit will use a default value.
- `precision::Union{Real,Nothing}` : The precision for the minimization
- `strategy::Int` : The strategy for the minimization (0,1(default),2). See the manual for details.
- `values` : The values of the parameters at the minimum
- `errors` : The errors of the parameters at the minimum
- `fixed` : The fixed status of the parameters
- `limits` : The limits of the parameters
- `is_valid` : If the minimization was successful
- `fval` : The function value at the minimum
- `edm` : The estimated distance to minimum
- `nfcn` : The number of function calls
- `ngrad` : The number of gradient calls
- `niter` : The number of iterations
- `npars` : The number of parameters
- `ndof` : Number of degrees of freedom
- `covariance` : The covariance matrix of the parameters
- `is_above_max_edm` : If the estimated distance to minimum is above the maximum
- `has_parameters_at_limit` : If any of the parameters are at the limits
- `has_accurate_covar` : If the covariance matrix is accurate
- `has_posdef_covar` : If the covariance matrix is positive definite
- `has_made_posdef_covar` : If the covariance matrix was made positive definite
- `hesse_failed` : If the Hesse algorithm failed
- `has_covariance` : If the covariance matrix is available
- `covariance` : The covariance matrix of the parameters
- `has_accurate_covar` : If the covariance matrix is accurate
- `has_valid_parameters` : If the parameters are valid
- `has_reached_call_limit` : If the maximum number of function calls was reached
- `minos` : The Minos errors
- `parameters` : The parameters values and errors
"""
mutable struct Minuit
    funcname::String                                # Name of the function
    cost::Union{CostFunction, Nothing}              # The cost function to minimize
    fcn::JuliaFcn                                   # The function to minimize    
    x0::AbstractVector                              # Initial parameters values 
    npar::Int                                       # Number of parameters
    names::Vector{String}                           # Names of the parameters
    method::Symbol                                  # The minimization method
    tolerance::Real                                 # The tolerance for the minimization
    precision::Union{Real,Nothing}                  # The precision for the minimization
    strategy::Int                                   # The strategy for the minimization    
    init_state::MnUserParameterState                # Initial user parameters
    last_state::MnUserParameterState                # The last user parameters
    app::Union{MnApplication, Nothing}              # The Minuit application
    fmin::Union{FunctionMinimum, Nothing}           # The result of the minimization
    mino::Union{Dict{String, MinosError}, Nothing} # The Minos errors
end

#---Minuit struct functions------------------------------------------------------------------------
include("util.jl")

const callbacks = CxxWrap.SafeCFunction[]
"""
    FCN(fnc, grad=nothing, arraycall=false, errordef=1.0)

Create a JuliaFcn object from a Julia function `fnc` and its gradient `grad`.

## Arguments
- `fnc::Function` : The Julia function to minimize. It can either accept a set of discrete arguments or a single argument of type `AbstractVector`. 
   This is decided in conjunction with the argument `arraycall`.
- `grad::Function=nothing`; Gradient Julia function. The input arguments follow the same as for `fcn` and it returns a `Vector`, of length the number of parameters, with the gradients.
- `arraycall::Bool=false` : If `true`, the function `fcn` accepts a single argument of type `AbstractVector`. If `false`, the function accepts a set of discrete arguments.
- `errordef::Real=1.0` : Error definition of the function. Minuit defines parameter errors as the change in parameter 
   value required to change the function value by `errordef`. Normally, for chisquared fits it is 1, and for negative log likelihood, its value is 0.5. 

## Returns
- `JuliaFcn` : A JuliaFcn object inheriting from the abstract C++ class `Minuit::FCNBase`that can be used in Minuit.

## Usage
```julia
fcn(x, y) = (x - 2)^2 + (y - 3)^2
grad(x, y) = [2*(x - 2), 2*(y - 3)]
jf = FCN(fcn, grad)

jf(1.0, 1.0)  # returns 5.0
jf.grad(1.0, 1.0)  # returns [-2.0, -4.0]

jf.nfcn # returns the number of function calls
jf.ngrad # returns the number of gradient calls

jf.has_gradient # returns true
```
"""
function FCN(fnc::Function, grad=nothing, arraycall=false, errordef=1.0)
    if arraycall
        vf = fnc
    else
        vf(x) = fnc(x...)
    end
    sf = eval( quote  
            @safe_cfunction($vf, Float64, (ConstCxxRef{StdVector{Float64}},)) 
         end )
    push!(callbacks, sf)
    if grad === nothing
        jf = JuliaFcn(sf, errordef)
    else
        if arraycall
            vg = grad
        else
            vg(x) = grad(x...)
        end
        function gd(g, x)
            _gv = vg(x)
            for i in 1:length(_gv)
                g[i] = _gv[i]
            end
            return
        end
        sg = eval( quote
                @safe_cfunction($gd, Cvoid, (CxxRef{StdVector{Float64}}, ConstCxxRef{StdVector{Float64}})) 
             end )
        push!(callbacks, sg)
        jf = JuliaFcn(sf, sg, errordef)
    end
    return jf
end

"""
    FCN(cost::CostFunction, grad=true)

Create a JuliaFcn object from a CostFunction.

## Arguments
- `fnc::CostFunction` : The CostFunction to minimize.
- `grad::Bool=true` : If `true`, the gradient of the cost function is used. If `false`, the gradient is not used. 

## Returns
- `JuliaFcn` : A JuliaFcn object inheriting from the abstract C++ class `Minuit::FCNBase`that can be used in Minuit.
"""
function FCN(cost::CostFunction, grad=true)
    errordef = cost.errordef
    COST = typeof(cost)
    #---Check if the cost function has a gradient---------------------------------------------------
    if grad && has_grad(cost)
        fcn = Symbol(:_internal_cost_, length(callbacks))
        gra = Symbol(:_internal_grad_, length(callbacks))
        eval( 
            quote
                function $fcn(obj::Ptr{Cvoid}, args)::Float64
                    costfunc = unsafe_pointer_to_objref(obj)::$COST
                    Minuit2.value(costfunc, args)
                end
                function $gra(obj::Ptr{Cvoid}, grad, args)::Nothing
                    costfunc = unsafe_pointer_to_objref(obj)::$COST
                    _grad = Minuit2.grad(costfunc, args)
                    for i in 1:length(grad)
                        grad[i] = _grad[i]
                    end
                end
            end )
        sf = eval( quote
                    @safe_cfunction($fcn, Float64, (Ptr{Cvoid}, ConstCxxRef{StdVector{Float64}},))
                   end )
        sg = eval( quote
                    @safe_cfunction($gra, Cvoid, (Ptr{Cvoid}, CxxRef{StdVector{Float64}}, ConstCxxRef{StdVector{Float64}})) 
                   end )
        push!(callbacks, sf)
        push!(callbacks, sg)
        return JuliaFcn(sf, sg, pointer_from_objref(cost), errordef)
    else
        fcn = Symbol(:_internal_cost_, length(callbacks))
        eval( 
            quote
                function $fcn(obj::Ptr{Cvoid}, args)::Float64
                    costfunc = unsafe_pointer_to_objref(obj)::$COST
                    Minuit2.value(costfunc, args)
                end
            end )
        sf = eval( quote
                    @safe_cfunction($fcn, Float64, (Ptr{Cvoid}, ConstCxxRef{StdVector{Float64}},)) 
                end )
        push!(callbacks, sf)
        return JuliaFcn(sf,  pointer_from_objref(cost), errordef)
    end
end


"""
    Minuit(fcn, x0...; grad=nothing, error=(), errordef=1.0, names=(), method=:migrad, maxfcn=0, tolerance=0, strategy=1,  kwargs...)

Initialize a Minuit object.

This does not start the minimization or perform any other work yet. Algorithms 
are started by calling the corresponding methods.

## Arguments
- `fcn::Union{Function,CostFunction}` : Function to minimize. See notes for details on what kind of functions are accepted.
- `x0::AbstractArray` : Starting values for the minimization. See notes for details on how to set starting values.
- `grad::Union{Function,Nothing, Bool}` : If `grad` is a function, it must be a function that calculates the gradient
  and returns an iterable object with one entry for each parameter, which is
  the derivative of `fcn` for that parameter. If `nothing` (default), Minuit will use the gradient of the provided 
  cost function. If it does not exists Minuit will compute the gradient numerically. If `grad` is `false`, Minuit will not use the gradient.
- `error::AbstractArray` : Starting values for the errors of the parameters. If not provided, Minuit will use 0.1 for all parameters.
- `errordef::Real` : Error definition of the function. Minuit defines parameter errors as the change in parameter 
  value required to change the function value by `errordef`. Normally, for chisquared fits it is 1, and for negative log likelihood,
  its value is 0.5. If the user wants instead the 2-sigma errors for chisquared fits, it becomes 4, as `Chi2(x+n*sigma) = Chi2(x) + n*n`.
- `names::Vector{String}` : Names of the parameters. If not provided, Minuit will try to extract the names from the function signature.
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
function Minuit(fcn, x0...; grad=nothing, error=(), errordef=1.0, names=(), method=:migrad, maxfcn=0, 
                tolerance=0.1, precision=nothing, strategy=1, kwargs...)
    if fcn isa CostFunction
        cost = fcn
        jf = FCN(fcn, grad isa Bool ? grad : true) # If grad is a boolean, use it to control it
        funcname = "$(typeof(fcn))"
        names = fcn.parameters
    else
        cost = nothing        
        #---Check if the function has a list of parameters or a single array---------------------------
        arraycall = get_nargs(fcn) == 1 && (length(x0) > 1 || length(x0[1]) > 1) ? true : false
        #---Get the arguments names-------------------------------------------------------------------
        if names === ()
            if arraycall
                n1 = get_argument_names(fcn)[1]
                names = ["$n1[$i]" for i in 1:length(x0)]
            else
                names = get_argument_names(fcn)
            end
        end
        #---Construct the FCN object-------------------------------------------------------------------
        jf = FCN(fcn, grad, arraycall, errordef)
        #---Get the function name---------------------------------------------------------------------
        funcname = string(first(methods(fcn)))
        funcname = funcname[1:findfirst('@',funcname)-2]
    end  
    # If x0 is not provided, use the keyword arguments of the form <par>=<value>-------------------
    if x0 === ()
        x0 = [kwargs[Symbol(n)] for n in names]
    elseif length(x0) == 1
        x0 = collect(x0[1])
    else
        x0 = collect(x0)
    end
    #---Create the user parameters-----------------------------------------------------------------
    userpars = MnUserParameterState()
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
    names = [Name(userpars, i-1) for i in 1:npar]
    Minuit(funcname, cost, jf, x0, npar, names, method, tolerance, precision, strategy, userpars, userpars, nothing, nothing, nothing)
end

function ndof(m::Minuit)
    if m.cost === nothing
        return Inf
    else
        return m.cost.ndata - (m.npar - sum(m.fixed))
    end
end
function reduced_chi2(m::Minuit)
    if isfinite(ndof(m)) && ndof(m) > 0 && m.up == 1
        return m.fmin.fval / ndof(m)
    else
        return NaN
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
    migrad = MnMigrad(m.fcn, m.last_state, MnStrategy(strategy))
    min = migrad(0, m.tolerance)   # calls the operator () to do the minimization
    #---Update the Minuit object with the results---------------------------------------------------
    m.app = migrad
    m.method = :migrad
    m.fmin = min
    m.mino = nothing
    m.strategy = strategy
    m.last_state = UserState(min)[]
    return m
end

function edm_goal(m::Minuit; migrad_factor=false)
    edm_goal = max( m.tolerance * Up(m.fcn), 4 * sqrt(eps()))
    migrad_factor && (edm_goal *= 2e-3)
    edm_goal
end

"""
    simplex!(m::Minuit, strategy=1)

Run Simplex minimization.

## Parameters
- `m::Minuit` : The Minuit object to minimize.
- `strategy::Int` : The minimization strategy. The default value is 1, which is
    the recommended value for most cases. The value 0 is faster, but less
    reliable. The value 2 is slower, but more reliable. The value 3 or higher is slower,
    but even more reliable.
"""
function simplex!(m::Minuit, strategy=1)
    simplex = MnSimplex(m.fcn, m.last_state,  MnStrategy(strategy))
    min = simplex(0, m.tolerance)
    m.app = simplex
    m.method = :simplex
    m.fmin = min
    m.mino = nothing
    m.last_state = UserState(min)[]
    return m
end


function scan!(m::Minuit, maxfcn = 0, strategy=1)
    if (maxfcn == 0)
        maxfcn = 200 + 100 * m.npar + 5 * m.npar^2
    end
    npar = m.npar - sum(m.fixed)
    npoints = maxfcn ^(1/npar) |> round |> Int
    ranges = [ m.fixed[p] ? range(m.values[p],1) : range(m.limits[p]..., npoints) for p in 1:m.npar ]
    hypercube = Iterators.product(ranges...)
    fmin, xmin = Inf, nothing
    for x in hypercube
        f = m.fcn(collect(x))
        if f < fmin
            fmin, xmin = f, x
        end
    end
    m.values = xmin
    scan = MnScan(m.fcn,  m.last_state,  MnStrategy(strategy))
    min = scan(0, m.tolerance)
    m.app = scan
    m.method = :scan
    m.fmin = min
    m.mino = nothing
    m.last_state = UserState(min)[]
    return m
end

"""
    matrix(m::Minuit; correlation=false)

Get the covariance matrix of the parameters.
"""
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
    hesse = MnHesse(strategy)
    if m.fmin === nothing || !IsValid(m.fmin)
        migrad!(m)
    end
    hesse(m.fcn, m.fmin, maxcalls)
    m.strategy = strategy
    m.last_state = UserState(m.fmin)[]
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
    if !m.is_valid
        throw(ErrorException("Function minimum is not valid"))
    end
    #---Get the parameters to run Minos-------------------------------------------------------------
    if length(parameters) == 0
        ipars = [ipar for ipar in 1:m.npar if !m.fixed[ipar]]
    else
        ipars = []
        for par in parameters
            ip, pname = keypair(m, par)
            if m.fixed[ip]
                warn("Cannot scan over fixed parameter $pname")
            else
                push!(ipars, ip)
            end
        end
    end
    #---Run Minos for each parameter----------------------------------------------------------------
    sav_errordef = m.fcn.errordef
    merrors = Dict{String, MinosError}()
    try
        m.fcn.errordef = sav_errordef * factor
        minos = MnMinos(m.fcn, m.fmin, strategy)
        for ipar in ipars
            mn = Minos(minos, ipar-1, ncall, m.tolerance)
            merrors[m.names[ipar]] = mn
        end
    finally
        m.fcn.errordef = sav_errordef
    end
    m.mino = merrors
    m.strategy = strategy
    return m
end

function Base.show(io::IO, me::MinosError)
    header = [me.number, me.is_valid ? "valid" : "invalid", " " ]
    data = [ "Error"    me.lower      me.upper;
             "Valid"    me.lower_valid me.upper_valid;
             "At Limit" me.at_lower_limit me.at_upper_limit;
             "Max Fcn"  me.at_lower_max_fcn me.at_upper_max_fcn;
             "New Min"  me.lower_new_min me.upper_new_min;]
    pretty_table(io, data; header=header, alignment=:l)
end

"""
    contour(m::Minuit, x, y; size=50, bound=2, grid=nothing, subtract_min=false)

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
- `x` : First parameter for scan (name or index).
- `y``: Second parameter for scan (name or index).
- `size=50` : Number of scanning points per parameter (Default: 50). It can be tuple `(nx,ny)` as 
           the number of scanning points per parameter. Ignored if `grid` is set.
- `bound=2` : Either ((v1min,v1max),(v2min,v2max)) or the number of `sigma`s to scan symmetrically from minimum.
- `grid::Tuple{AbstractVector,AbstractVector} : Grid points to scan over. If `grid`` is set, `size`` and `bound` are ignored.
- `subtract_min::Bool=false` : Subtract minimum from return values

##  Returns
- Tuple(`xv`, `yv`, `zv`) : Tuple of 1D arrays with the x and y values and a 2D array with the function values. 
"""
function contour(m::Minuit, x, y; size=50, bound=2, grid=nothing, subtract_min=false)
    ix, xname = keypair(m, x)
    iy, yname = keypair(m, y)

    if !isnothing(grid)
        xv, yv = grid
        ndims(xv) == 1 || ndims(yv) || throw(ArgumentError("grid per parameter must be 1D array-like"))
    else
        if bound isa Tuple || bound isa Tuple
            xrange, yrange = bound
        else
            start = m.values
            sigma = m.errors
            xrange = ( start[ix] - bound * sigma[ix], start[ix] + bound * sigma[ix])
            yrange = ( start[iy] - bound * sigma[iy], start[iy] + bound * sigma[iy])
        end
        if size isa Tuple || size isa Tuple
            xsize, ysize = size
        else
            xsize = ysize = size
        end    
        xv = range(xrange[1], xrange[2], length=xsize)
        yv = range(yrange[1], yrange[2], length=ysize)
    end
    zv = zeros(Float64, length(xv), length(yv))
    values_v = StdVector(collect(m.values))
    for i in eachindex(xv)
        values_v[ix] = xv[i]
        for j in eachindex(yv)
            values_v[iy] = yv[j]
            zv[i, j] = m.fcn(values_v)
        end
    end
    if subtract_min
        zv .-= minimum(zv)
    end
    return xv, yv, zv
end

"""
    nmcontour(x, y; cl=0.68, size=50, interpolated=0, ncall=0, iterate=5, use_simplex=true)

Get 2D Minos confidence region.

This scans over two parameters and minimises all other free parameters for each
scan point. This scan produces a statistical confidence region according to the
[profile likelihood method](https://en.wikipedia.org/wiki/Likelihood_function)
with a confidence level `cl`, which is asymptotically equal to the coverage
probability of the confidence region according to [Wilks' theorem](https://en.wikipedia.org/wiki/Wilks%27_theorem).
Note that 1D projections of the 2D confidence region are larger than 1D Minos intervals computed for the
same confidence level. This is not an error, but a consequence of Wilks'theorem.

The calculation is expensive since a numerical minimisation has to be performed at various points.

## Arguments
- `x` : Variable name of the first parameter.
- `y` : Variable name of the second parameter.
- `cl::Real=0.68` : Confidence level of the contour. If not set a standard 68 % contour
  is computed (default). If 0 < cl < 1, the value is interpreted as the
  confidence level (a probability). For convenience, values cl >= 1 are
  interpreted as the probability content of a central symmetric interval
  covering that many standard deviations of a normal distribution.
- `size::Int=50` : Number of points on the contour to find. Increasing this
  makes the contour smoother, but requires more computation time.
- `interpolated::Int=0` : Number of interpolated points on the contour. If you set this
  to a value larger than size, cubic spline interpolation is used to generate
  a smoother curve and the interpolated coordinates are returned. Values
  smaller than size are ignored. Good results can be obtained with size=20,
  interpolated=200.

## Returns
- `contour::Vector(Tuple{Float64,Float64})` : Contour points of the form [(x1, y1)...(xn, yn)].
"""
function mncontour(m::Minuit, x, y; cl=0.68, size=50, interpolated=0)
    ix, xname = keypair(m, x)
    iy, yname = keypair(m, y)

    cl >= 1.0 && (cl = cdf(Chisq(1), cl^2))    # convert sigmas into confidence level
    factor = quantile(Chisq(2), cl)            # convert confidence level to errordef

    m.is_valid || throw(ErrorException("Function minimum is not valid: $(m.fmin)"))
    m.fixed[ix] && throw(ErrorException("Cannot scan over fixed parameter $xname"))
    m.fixed[iy] && throw(ErrorException("Cannot scan over fixed parameter $yname"))

    # Set temporary errordef before calling MnContours
    sav_errordef = m.fcn.errordef
    points = nothing
    try
        m.fcn.errordef = sav_errordef * factor
        mnc = MnContours(m.fcn, m.fmin, m.strategy)
        points = mnc(ix, iy, size)
    finally
        m.fcn.errordef = sav_errordef
    end

    contour = [ (X(p), Y(p)) for p in points]
    push!(contour, contour[1])  # close the contour

    if interpolated > size
        @warn "Interpolation not yet implemented"
    end
    return contour
end

"""
    profile(m::Minuit, var; size=100, bound=2, grid=nothing, subtract_min=false)
            
Calculate 1D cost function profile over a range.

A 1D scan of the cost function around the minimum, useful to inspect the
minimum. For a fit with several free parameters this is not the same as the
Minos profile computed by `mncontour`.

## Arguments
- `m::Minuit` : The Minuit object to minimize.
- `var` : The parameter to scan over (name or index).
- `size=100` : Number of scanning points. Ignored if `grid` is set.
- `bound=2` : Number of `sigma`s to scan symmetrically around the minimum. Ignored if `grid` is set.
- `grid::AbstractVector` : Grid points to scan over. If `grid` is set, `size` and `bound` are ignored.
- `subtract_min::Bool=false` : Subtract minimum from return values.

## Returns
- Tuple(`x`, `y`) : Tuple of 1D arrays with the parameter values and the function values.
"""
function profile(m::Minuit, var; size=100, bound=2, grid=nothing, subtract_min=false)

    ipar, pname = keypair(m, var)
    m.fixed[ipar] && throw(ErrorException("Cannot profile over fixed parameter $pname"))
    if !isnothing(grid)
        x = grid
        ndims(x) != 1 && throw(ArgumentError("grid must be 1D array-like"))
    else
        start = m.values[ipar]
        sigma= m.errors[ipar]
        x = range(start - bound * sigma, start + bound * sigma, length=size)
    end

    y = zeros(Float64, length(x))
    values_v = StdVector(m.values |> collect)
    for i in eachindex(x)
        values_v[ipar] = x[i]
        y[i] = m.fcn(values_v)
    end
    if subtract_min
        y .-= minimum(y)
    end
    return x, y
end


"""
    mnprofile(m::Minuit, var; size=30, bound=2, grid=nothing, subtract_min=false, 
                   ncall=0, iterate=5, use_simplex=true)

Get Minos profile over a specified interval.

Scans over one parameter and minimises the function with respect to all other
parameters for each scan point.

## Arguments
- `var` : Parameter to scan over.
- `size=30` : Number of scanning points. Ignored if grid is set.
- `bound=2` : If bound is a tuple, (left, right) scanning bound, or the number of sigmas to scan
  symmetrically around the minimum. Ignored if grid is set.
- `grid` : Parameter values on which to compute the profile. If `grid` is set, `size` and
   `bound` are ignored.
- `subtract_min=false` : If true, subtract offset so that smallest value is zero.
- `ncall=0` : Approximate maximum number of calls before minimization will be aborted.
   If set to 0, use the adaptive heuristic from the Minuit2 library. 
   Note: The limit may be slightly violated, because the condition is checked only after 
   a full iteration of the algorithm, which usually performs several function calls.
        iterate : int, optional
            Automatically call Migrad up to N times if convergence was not reached
            (Default: 5). This simple heuristic makes Migrad converge more often even if
            the numerical precision of the cost function is low. Setting this to 1
            disables the feature.
        use_simplex: bool, optional
            If we have to iterate, set this to True to call the Simplex algorithm before
            each call to Migrad (Default: True). This may improve convergence in
            pathological cases (which we are in when we have to iterate).
## Returns
- Tuple(`x`, `y`, `ok`) : Tuple of 1D arrays with the parameter values, function values and
  booleans whether the fit succeeded or not.
"""
function mnprofile(m::Minuit, var; size=30, bound=2, grid=nothing, subtract_min=false, 
                   ncall=0, iterate=5, use_simplex=true)

    ipar, pname = keypair(m, var)
    if !isnothing(grid)
        x = grid
        ndims(x) != 1 && throw(ArgumentError("grid must be 1D array-like"))
    else
        if bound isa Tuple
            xrange= bound
        else
            start = m.values[ipar]
            sigma = m.errors[ipar]
            xrange = start - bound * sigma, start + bound * sigma
        end
        x = range(xrange..., length=size)
    end
    y = zeros(Float64, length(x))
    status = zeros(Bool, length(x))

    state = copy(m.fmin.state)  # copy
    Fix(state, ipar-1)  # fix the parameter we are scanning over
    # strategy 0 to avoid expensive computation of Hesse matrix
    strategy = MnStrategy(0)
    for (i,v) in enumerate(x)
        SetValue(state, ipar-1, v)
        fmin = robust_low_level_fit(m.fcn, state, ncall, strategy, m.tolerance, m.precision, iterate, use_simplex)
        IsValid(fmin) || @warn("MIGRAD fails to converge for $pname=$v")
        status[i] = IsValid(fmin)
        y[i] = Fval(fmin)
    end
    subtract_min && (y .-= minimum(y))
    return x, y, status
end

function robust_low_level_fit(fcn, state, ncall, strategy, tolerance, precision, iterate, use_simplex)
    migrad = MnMigrad(fcn, state, strategy)
    isnothing(precision) || SetPrecision(migrad, precision)
    fmin = migrad(ncall, tolerance)
    strategy = MnStrategy(2)
    migrad = MnMigrad(fcn, UserState(fmin), strategy)
    while !IsValid(fmin) && !HasReachedCallLimit(fmin) && iterate > 1
        if use_simplex
            simplex = MnSimplex(fcn, State(fmin), strategy)
            if !isnothing(precision)
                simplex.precision = precision
            end
            fmin = simplex(ncall, tolerance)
            migrad = MnMigrad(fcn, State(fmin), strategy)
        end
        if !isnothing(precision)
            migrad.precision = precision
        end
        fmin = migrad(ncall, tolerance)
        iterate -= 1
    end
    return fmin
end
