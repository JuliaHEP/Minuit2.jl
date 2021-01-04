# Minuit2

This is a Julia package to use C++ Minuit2 package for function minimization.
The binaries for Minuit2 are built here: <https://github.com/JuliaPackaging/Yggdrasil/tree/master/M/Minuit2>

The interface code between C++ and Julia can be found here <https://github.com/jstrube/Minuit2_Julia_Wrapper> and needs to be installed locally for the time being.
If you cannot build the wrapper binaries using BinaryBuilder, you can download the artifacts from here: <https://github.com/jstrube/Minuit2_Julia_Wrapper_jll.jl>

The Pkg documentation may have some helpful tips to achieve this:
<https://julialang.github.io/Pkg.jl/v1/artifacts/#Overriding-artifact-locations>
