#include "Minuit2Wrap.h"
#include "Minuit2/FCNBase.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnStrategy.h"
#include "Minuit2/MnUserFcn.h"
#include "Minuit2/MnSeedGenerator.h"
#include <Minuit2/Numerical2PGradientCalculator.h>

using namespace ROOT::Minuit2;
FunctionMinimum ROOT::Minuit2::createFunctionMinimum(const JuliaFcn& fcn, const MnUserParameterState& st,
                                                     const MnStrategy& str, double edm_goal) {
  MnUserFcn mfcn(fcn, st.Trafo());
  MnSeedGenerator gen;
  Numerical2PGradientCalculator gc(mfcn, st.Trafo(), str);
  MinimumSeed seed = gen(mfcn, gc, st, str);

  const auto& val = seed.Parameters().Vec();
  const auto n = seed.Trafo().VariableParameters();

  MnAlgebraicVector err(n);
  for (unsigned int i = 0; i < n; i++) {
    err(i) = std::sqrt(2. * mfcn.Up() * seed.Error().InvHessian()(i, i));
  }

  MinimumParameters minp(val, err, seed.Fval());
  std::vector<MinimumState> minstv(1, MinimumState(minp, seed.Edm(), fcn.nfcn));
  if (minstv.back().Edm() < edm_goal) return FunctionMinimum(seed, minstv, fcn.Up());
  return FunctionMinimum(seed, minstv, fcn.Up(), FunctionMinimum::MnAboveMaxEdm);
}
