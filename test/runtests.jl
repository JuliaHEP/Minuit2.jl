using Minuit2
using Test

@testset "Minuit2 tests" verbose = true begin 
    include("test_wrapper.jl")
    include("test_functions.jl")
    include("test_util.jl")
    include("test_api.jl")
end
