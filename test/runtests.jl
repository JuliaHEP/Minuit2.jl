using ComponentArrays: ComponentArray
import Distributions
import DistributionsHEP
import FHist
import Plots
import RecipesBase
import StatsBase
using Minuit2.RooFit
using Random: seed!
using Minuit2
using CxxWrap
using Test
include("functions.jl")

@testset "Minuit2 tests" verbose = true begin
    include("test_wrapper.jl")
    include("test_functions.jl")
    include("test_util.jl")
    include("test_api.jl")
    include("test_cost.jl")
    include("test_roofit.jl")
    include("test_threads.jl")
    include("test_optimization_integration.jl")
end
