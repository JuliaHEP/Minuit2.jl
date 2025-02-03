using Revise
using Minuit2
using CxxWrap 

include("../test/functions.jl")

m = Minuit(sphere, [2,2,2], names=["x","y","z"], tolerance=1e-4)
migrad!(m)
