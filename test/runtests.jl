using Minuit2
using Test
using CxxWrap
using StatsBase
using LinearAlgebra

testf(par) = par[1] + par[2]

gaussian(pars, x) = pars[3] * exp(-0.5*(x-pars[1])^2/(pars[2]^2))/(√(2.0*π)*pars[2])

@testset "Minuit2.jl" begin
    c_func = @safe_cfunction(testf, Float64, (CxxWrap.StdLib.StdVectorAllocated{Float64}, ))
    println("Result is: ", Minuit2.testCallback(c_func, StdVector(Float64[1, 2])))


    data = normalize(fit(Histogram, randn(100000), range(-3, stop=3, length=100)), mode=:density)
    x = collect(data.edges[1])[1:end-1]
    y = data.weights 
    var = ones(length(y))
    init = [0.0, 1.0, 100.0]
    gaussFunc = @safe_cfunction(gaussian, Float64, (CxxWrap.StdLib.StdVectorAllocated{Float64}, Float64))
    fitter = Minuit2.JuliaFcn(c_func, y, x, var)
end
