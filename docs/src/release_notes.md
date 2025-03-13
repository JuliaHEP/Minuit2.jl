
# Release Notes

## 0.2.*
### Fixes
- 
### Additions
- Extended cost functions `ExtendedBinnedNLL` and `ExtendedUnbinnedNLL`.
  - Added examples in costfunctions example
  - Added unit tests
- Added `CostSum` to combine cost functions
  - Added unit test

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


