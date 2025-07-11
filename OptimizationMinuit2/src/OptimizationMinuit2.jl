module OptimizationMinuit2

using Minuit2
using ComponentArrays: ComponentArray
using Optimization
using Optimization.SciMLBase

export MinuitOptimizer

@kwdef struct MinuitOptimizer
    strategy::Int = 1
end

SciMLBase.allowsconstraints(::MinuitOptimizer) = false
SciMLBase.allowsbounds(opt::MinuitOptimizer) = true
SciMLBase.requiresbounds(opt::MinuitOptimizer) = false
SciMLBase.supports_opt_cache_interface(opt::MinuitOptimizer) = false
SciMLBase.requiresgradient(opt::MinuitOptimizer) = false
SciMLBase.requireshessian(::MinuitOptimizer) = false
SciMLBase.requiresconsjac(::MinuitOptimizer) = false
# SciMLBase.requiresconshess(::MinuitOptimizer) = false

function __map_optimizer_args(prob::OptimizationProblem, opt::MinuitOptimizer;
        callback = nothing,
        maxiters::Union{Number, Nothing} = 0,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing)
    if !isnothing(maxtime)
        @warn "common maxtime is currently not used by $(opt)"
    end
    if !isnothing(abstol)
        @warn "common abstol is currently not used by $(opt)"
    end
    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(opt)"
    end

    mapped_args = (;)

    if !isnothing(maxiters)
        mapped_args = (; mapped_args..., ncall = maxiters)
    else
        mapped_args = (; mapped_args..., ncall = 0) # MINUIT2 will use the default value
    end

    return mapped_args
end

function SciMLBase.__solve(prob::OptimizationProblem, opt::MinuitOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        kwargs...)
    local x, _loss

    maxiters = Optimization._check_and_convert_maxiters(maxiters)

    _loss = function (θ)
        x = prob.f(θ, prob.p)
        return first(x)
    end

    opt_arg = __map_optimizer_args(prob, opt; maxiters = maxiters, maxtime = maxtime,
        abstol = abstol, reltol = reltol, kwargs...)

    m = Minuit(_loss, prob.u0; opt.strategy)
    migrad!(m, opt_arg.ncall)

    stats = Optimization.OptimizationStats(; time = m.elapsed)
    sol = m.values
    SciMLBase.build_solution(SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        sol isa ComponentArray ? sol : collect(sol), m.fval;
        errors = m.errors, stats = stats)
end

end
