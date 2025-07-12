using Minuit2
using Minuit2.RooFit
using ComponentArrays: ComponentArray
using Test

@testset "Minuit2 tests" verbose = true begin 
    include("test_wrapper.jl")
    include("test_functions.jl")
    include("test_util.jl")
    include("test_api.jl")
    include("test_cost.jl")
    include("test_distributions.jl")
    include("test_roofit.jl")
    include("test_threads.jl")
    include("test_optimization_integration.jl")
end
