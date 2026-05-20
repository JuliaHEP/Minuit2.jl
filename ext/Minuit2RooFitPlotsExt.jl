module Minuit2RooFitPlotsExt

using Minuit2
using Distributions
using DistributionsHEP
using FHist
using Plots
using RecipesBase
using StatsBase

_generating_output() = ccall(:jl_generating_output, Cint, ()) == 1

function __init__()
    _generating_output() && return
    isdefined(Minuit2.RooFit, :Distributions) ||
        Core.eval(Minuit2.RooFit, :(const Distributions = $Distributions))
    isdefined(Minuit2.RooFit, :DistributionsHEP) ||
        Core.eval(Minuit2.RooFit, :(const DistributionsHEP = $DistributionsHEP))
    isdefined(Minuit2.RooFit, :FHist) ||
        Core.eval(Minuit2.RooFit, :(const FHist = $FHist))
    isdefined(Minuit2.RooFit, :StatsBase) ||
        Core.eval(Minuit2.RooFit, :(const StatsBase = $StatsBase))
    isdefined(Minuit2.RooFit, :AbstractPdf) ||
        Core.eval(Minuit2.RooFit, :(include(joinpath(@__DIR__, "..", "src", "roofit.jl"))))
    isdefined(@__MODULE__, :plot_attributes) ||
        Core.eval(@__MODULE__, :(include(joinpath(@__DIR__, "..", "src", "roofit_plots.jl"))))
end

end
