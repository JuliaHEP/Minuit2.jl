module Minuit2OptimizationExt

using Minuit2
using ComponentArrays: ComponentArray
using Optimization
using Optimization.SciMLBase

SciMLBase.allowsconstraints(::MigradOptimizer) = false
SciMLBase.allowsbounds(opt::MigradOptimizer) = true
SciMLBase.requiresbounds(opt::MigradOptimizer) = false
SciMLBase.supports_opt_cache_interface(opt::MigradOptimizer) = false
SciMLBase.requiresgradient(opt::MigradOptimizer) = false
SciMLBase.requireshessian(::MigradOptimizer) = false
SciMLBase.requiresconsjac(::MigradOptimizer) = false
SciMLBase.requiresconshess(::MigradOptimizer) = false

function __map_optimizer_args(
        ::OptimizationProblem, opt::MigradOptimizer;
        maxiters::Union{Number, Nothing} = 0,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing
    )
    if !isnothing(maxtime)
        @warn "common maxtime is currently not used by $(opt)"
    end
    if !isnothing(abstol)
        @warn "common abstol is currently not used by $(opt), use `tolerance` instead"
    end
    if !isnothing(reltol)
        @warn "common reltol is currently not used by $(opt), use `tolerance` instead"
    end

    mapped_args = (;)

    if !isnothing(maxiters)
        mapped_args = (; mapped_args..., ncall = maxiters)
    else
        mapped_args = (; mapped_args..., ncall = 0) # MINUIT2 will use the default value
    end

    return mapped_args
end

function SciMLBase.__solve(
        prob::OptimizationProblem, opt::MigradOptimizer;
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        tolerance::Number = 0.1,
        kwargs...
    )
    local x, _loss

    maxiters = Optimization._check_and_convert_maxiters(maxiters)

    _loss = function (θ)
        x = prob.f(θ, prob.p)
        return first(x)
    end

    opt_arg = __map_optimizer_args(
        prob, opt; maxiters = maxiters, maxtime = maxtime,
        abstol = abstol, reltol = reltol, kwargs...
    )

    if !isnothing(prob.lb) && length(prob.lb) != length(prob.u0)
        throw(ArgumentError("Length of lb must match number of parameters"))
    end
    if !isnothing(prob.ub) && length(prob.ub) != length(prob.u0)
        throw(ArgumentError("Length of ub must match number of parameters"))
    end
    lb = isnothing(prob.lb) ? fill(-Inf, length(prob.u0)) : prob.lb
    ub = isnothing(prob.ub) ? fill(Inf, length(prob.u0)) : prob.ub

    m = Minuit(_loss, prob.u0; opt.strategy, tolerance, opt.errordef, opt.maxfcn, limits = collect(zip(lb, ub)))
    migrad!(m, opt_arg.ncall)

    stats = Optimization.OptimizationStats(; time = m.elapsed)
    u = m.values
    return SciMLBase.build_solution(
        SciMLBase.DefaultOptimizationCache(prob.f, prob.p), opt,
        u isa ComponentArray ? u : collect(u), m.fval;
        stats = stats, original = m
    )
end

end
