# Optimization.jl interface

The MIGRAD optimizer can also be used from Optimization.jl:
```@example optimizationjl
using Minuit2, Optimization
opf = OptimizationFunction((x,p)->x^2);
opp = OptimizationProblem(opf, [1.0])
solve(opp, MigradOptimizer(strategy=2, tolerance=0.01), maxiters = 100)
```

[ComponentArrays.jl](https://github.com/SciML/ComponentArrays.jl) is also supported:
```@example optimizationjl
using ComponentArrays
opf = OptimizationFunction((x,p)->x^2);
opp = OptimizationProblem(opf, ComponentArray(x=1.0))
sol = solve(opp, MigradOptimizer(strategy=2, tolerance=0.01), maxiters = 100)
```

## Access original `Minuit` object
Minuit provides much information that does not fit into `Optimization.jl`'s model, thus it's useful to access the
original `Minuit` object:
```@example optimizationjl
sol.original
```
