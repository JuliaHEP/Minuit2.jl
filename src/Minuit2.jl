module Minuit2
    using CxxWrap
    using Libdl

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

    export draw_contour, draw_mncontour, draw_profile, draw_mnprofile, visualize
    function draw_contour end
    function draw_mncontour end
    function draw_profile end
    function draw_mnprofile end
    function visualize end
end

