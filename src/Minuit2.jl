module Minuit2
    using Minuit2_jll
    using CxxWrap
    using Libdl

    # Check whether the wrappers have been build locally otherwise use the binary package Minuit2_julia_jll
    gendir = normpath(joinpath(@__DIR__, "../gen"))
    if isdir(joinpath(gendir, "build/lib"))
        include(joinpath(gendir, "jl/src/Minuit2-export.jl"))
        @wrapmodule(()->joinpath(gendir, "build/lib", "libMinuit2Wrap.$(Libdl.dlext)"))
    else
        using Minuit2_julia_jll
        include(Minuit2_julia_jll.Minuit2_exports)
        @wrapmodule(()->Minuit2_julia_jll.libMinuit2Wrap)
    end

    function __init__()
        @initcxx
    end

    include("api.jl")
end

