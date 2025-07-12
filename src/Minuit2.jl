module Minuit2
    using CxxWrap
    using Libdl
    using ComponentArrays: ComponentArrays, ComponentArray, getaxes

    # Check whether the wrappers have been build locally otherwise use the binary package Minuit2_Julia_Wrapper_jll
    gendir = normpath(joinpath(@__DIR__, "../gen"))
    if isdir(joinpath(gendir, "build/lib"))
        include(joinpath(gendir, "jl/src/Minuit2-export.jl"))
        @wrapmodule(()->joinpath(gendir, "build/lib", "libMinuit2Wrap.$(Libdl.dlext)"))
    else
        using Minuit2_Julia_Wrapper_jll
        include(Minuit2_Julia_Wrapper_jll.Minuit2_exports)
        @wrapmodule(()->Minuit2_Julia_Wrapper_jll.libMinuit2Wrap)
    end

    function __init__()
        @initcxx
    end

    include("wrap.jl")
    include("cost.jl")
    include("api.jl")
    include("distributions.jl")
    include("roofit.jl")

    export draw_contour, draw_mncontour, draw_profile, draw_mnprofile, visualize, MigradOptimizer

    function draw_contour end
    function draw_mncontour end
    function draw_profile end
    function draw_mnprofile end
    function visualize end

    """
        @kwdef struct MigradOptimizer
            strategy::Int = 1
            tolerance::Float64 = 0.1
            errordef::Float64 = 1.0
        end

    Optimizer structure for the Migrad algorithm to be used by `Optimization.jl` ecosystem.
    Not all options supported by [`Minuit`](@ref) are avaiable yet, as they may not map neatly to the `Optimization.jl` interface.
    """
    @kwdef struct MigradOptimizer
        strategy::Int = 1
        tolerance::Float64 = 0.1
        errordef::Float64 = 1.0
    end

end

