# Common test functions for optimizers.
# Also see: https://en.wikipedia.org/wiki/Test_functions_for_optimization

"""
    rosenbrock(x, y)

Rosenbrock function. Minimum: f(1, 1) = 0.

https://en.wikipedia.org/wiki/Rosenbrock_function
"""
function rosenbrock(x, y)
    return  (1 - x)^2 + 100 * (y - x^2)^2
end
function rosenbrock_grad(x, y)
    return [-400 * x * (-(x^2) + y) + 2 * x - 2, -200 * x^2 + 200 * y]
end

"""
    ackley(x, y)

Ackley function. Minimum: f(0, 0) = 0.

https://en.wikipedia.org/wiki/Ackley_function
"""
function ackley(x, y)
    term1 = -20 * exp(-0.2 * sqrt(0.5 * (x^2 + y^2)))
    term2 = -exp(0.5 * (cos(2 * pi * x) + cos(2 * pi * y)))
    return term1 + term2 + 20 + ℯ
end

"""
    beale(x, y)

Beale function. Minimum: f(3, 0.5) = 0.

https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
function beale(x, y)
    term1 = 1.5 - x + x * y
    term2 = 2.25 - x + x * y^2
    term3 = 2.625 - x + x * y^3
    return term1^2 + term2^2 + term3^2
end


"""
    matyas(x, y)

Matyas function. Minimum: f(0, 0) = 0.

https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
function matyas(x, y)
    return 0.26 * (x^2 + y^2) - 0.48 * x * y
end

"""
    sphere(x...)

Sphere function for variable number of arguments. Minimum: f(0, ..., 0) = 0.

https://en.wikipedia.org/wiki/Test_functions_for_optimization
"""
function sphere(x)
    return sum(x .^ 2)
end
