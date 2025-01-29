using Minuit2
using Test

@testset "Minuit2 tests" verbose = true begin 
    include("test_wrapper.jl")
    include("test_functions.jl")
end
