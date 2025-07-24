using Optimization, ComponentArrays

@testset "Optimization.jl Integration" begin

    opf = OptimizationFunction((x,p)->x^2);
    opp = OptimizationProblem(opf, [1.0])
    sol = solve(opp, MigradOptimizer(strategy=2))

    @test sol.u ≈ [0.0] atol=1e-6
    @test sol.original.errors ≈ [1.0] atol=1e-3

    opf = OptimizationFunction((x,p)->x^2);
    opp = OptimizationProblem(opf, ComponentArray(; x = [1.0]))
    sol = solve(opp, MigradOptimizer(strategy=2))

    @test sol.u.x ≈ [0.0] atol=1e-6
    @test sol.original.errors.x ≈ [1.0] atol=1e-3
end
