
# Release Notes

## 0.4.1 (03-09-2025)
### Additions
- `m = Minuit(f, x)` can be set up with `x::ComponentArray` and the resultant `m.values, m.errors` will be `ComponentArray`s as well.
- Added `MigradOptimizer()` which can be used as a solver in the `Optimization.jl` ecosystem.
### Other changes
- Upgraded to CxxWrap 0.17

## 0.4.0 (21-07-2025)
### Breaking changes
- Replaced local distributions (Chebyshev, ArgusBG) with the ones in `DistributionsHEP` package
- To instantiate a `RealVar` requires a initial value to deduce the `T` or to specify the parameter `T` with `RealVar{Float64}` for example.
- Removed function `RooFit.generateBinned`. The data is generated binned if nbins > 0 in the observable variable.
- The function `RooFit.generate` returns a `DataSet`

## 0.3.3 (03-07-2025)
### Fixes
- Issue #15. Minhit API: Added keyword arguments for `limits` and `fixed`. 
- Issue #17. use_pdf=:approximate math regarding binwidth
- Issue #23. Extra CxxWrap.CxxWrapCore.ConstCxxRef{} when calling array-input FCN?
- Fix the parameter names for vector FCN without `names` argument
  
### Additions
- Added `error` in the definition of `RealVar`
- Added roofit_basics tutorial
- Plotting is done by using macro `@recipe` of module `RecipesBase`
- Support for non-uniform binned cost functions
- Added protection in global state (callback array) for multi-threading

## 0.3.0 (04-04-2025)
### Additions
- Extended cost functions `ExtendedBinnedNLL` and `ExtendedUnbinnedNLL`.
  - Added examples in `costfunctions` example
  - Added unit tests
- Added `CostSum` to combine cost functions
  - Added unit test
  - Added `combined` tutorial
- Extended Distributions with a Chebyshev polynomial distribution
- Added `elapsed` time of the minuit printout
- Added module `RooFit` with some very simple elements to build models
  - Added tutorial `roofit` and unit tests

## 0.2.1 (5-03-2025) 
### Fixes
- Keyword argument for gradient in cost functions is `grad`
- Fixed convergence for BinnedNLL with gradient
- Fix for making assignments of properties with `:` such as m.values[:] = [...] 
### Additions
- Two notebooks for execution performance (julia, python)
- Added docstring for all properties of `Minuit` type
- Added `ndof` function and property to get the number of degrees of freedom
- Added extension module `Minuit2HistExt` to accept FHist histograms as input for BinnedNLL
- Added simplex! and scan! functions to perform minimization with other methods
 
## 0.2.0 (18-02-2025)
- Initial release that includes the wrapper to all major C++ classes available in Minuit2
- The currently offered API is minimalistic. It allows to run some introductory  tutorials:
    - Introduction to Minuit2.jl
    - Cost Functions
- The currently available cost functions are:
    - UnbinnedNLL - Unbinned negative log-likelihood cost function.
    - BinnedNLL - Binned negative log-likelihood.
    - LeastSquares - Least-squares cost function (aka chisquare function).
- The package provides a extension module (if Plots and FHist are loaded)
  for plotting the data and results of the fit.
- Documentation is generated using Documenter


