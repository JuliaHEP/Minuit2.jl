using Minuit2
using Test
using CxxWrap
using StatsBase
using LinearAlgebra

testf(par) = par[1] + par[2]

gaussian(pars, x) = pars[3] * exp(-0.5*(x-pars[1])^2/(pars[2]^2))/(√(2.0*π)*pars[2])

@testset "basic callback" begin
    c_func = @safe_cfunction(testf, Float64, (CxxWrap.StdLib.StdVectorAllocated{Float64}, ))
    println("Result is: ", Minuit2.testCallback(c_func, StdVector(Float64[1, 2])))
end

@testset "VariableMetric" begin
    data = normalize(fit(Histogram, randn(100000), range(-3, stop=3, length=100)), mode=:density)
    x = collect(data.edges[1])[1:end-1]
    y = data.weights 
    var = ones(length(y))
    gaussFunc = @safe_cfunction(gaussian, Float64, (CxxWrap.StdLib.StdVectorAllocated{Float64}, Float64))

    println("VariableMetric:")
    parameters = [0.0, 1.0, 100.0]
    fitter = Minuit2.JuliaFcn(gaussFunc, y, x, var)
    Minuit2.fit_VariableMetric(fitter, parameters)
    println("best parameters are ", parameters)
end

@testset "Migrad" begin
    data = normalize(fit(Histogram, randn(100000), range(-3, stop=3, length=100)), mode=:density)
    x = collect(data.edges[1])[1:end-1]
    y = data.weights 
    var = ones(length(y))
    gaussFunc = @safe_cfunction(gaussian, Float64, (CxxWrap.StdLib.StdVectorAllocated{Float64}, Float64))

    println("Migrad:")
    parameters = [0.0, 1.0, 100.0]
    errors = [0.1, 0.1, 0.1]
    fitter = Minuit2.JuliaFcn(gaussFunc, y, x, var)
    Minuit2.fit_Migrad(fitter, parameters, errors)
    println("best parameters are ", parameters)
end

@testset "Minos" begin
    data = normalize(fit(Histogram, randn(100000), range(-3, stop=3, length=100)), mode=:density)
    x = collect(data.edges[1])[1:end-1]
    y = data.weights 
    var = ones(length(y))
    gaussFunc = @safe_cfunction(gaussian, Float64, (CxxWrap.StdLib.StdVectorAllocated{Float64}, Float64))

    println("Minos:")
    parameters = [0.0, 1.0, 100.0]
    lower = [0.1, 0.1, 0.1]
    upper = [0.1, 0.1, 0.1]
    fitter = Minuit2.JuliaFcn(gaussFunc, y, x, var)
    Minuit2.fit_Migrad_Minos(fitter, parameters, lower, upper)
    println("best parameters are ", parameters)
    println("1-sigma errors:")
    println("lower bounds: ", lower)
    println("upper bounds: ", upper)
end
