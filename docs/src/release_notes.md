
# Release Notes

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


