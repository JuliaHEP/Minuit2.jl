# Minuit2.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https:///JuliaHEP.github.io/Minuit2.jl/dev/)
[![Build Status](https://github.com/JuliaHEP/Minuit2.jl/workflows/CI/badge.svg)](https://github.com/JuliaHEP/Minuit2.jl/actions)
[![codecov](https://codecov.io/gh/JuliaHEP/Minuit2.jl/graph/badge.svg?token=AS74WXOYT6)](https://codecov.io/gh/JuliaHEP/Minuit2.jl)


This package provides a Julia-friendly interface to the **Minuit2** C++ library, maintained by CERN's [ROOT team](https://root.cern.ch). Minuit2 is designed for function minimization, particularly in statistical applications such as maximum-likelihood estimation and least-squares fitting. It computes best-fit parameters and error estimates using likelihood profile analysis.  

The API and additional features of this package are heavily inspired by [iminuit](https://github.dev/scikit-hep/iminuit), which offers a Python-friendly interface to the same library.

In addition, we have provided an additional module `Minuit2.RooFit` with an initial implementation in Julia of the [RooFit](https://root.cern/manual/roofit/) modeling package often used in HEP data analysis.

## Implementation Details  

- The wrapper for Minuit2's C++ classes is implemented using the [CxxWrap.jl](https://github.com/JuliaInterop/CxxWrap.jl) package.  
- The wrapper code is generated with [WrapIt](https://github.com/grasph/wrapit).  
- The Minuit2 binaries and the wrapper library are built using [BinaryBuilder.jl](https://github.com/JuliaPackaging/Yggdrasil/tree/master/M/Minuit2).  
