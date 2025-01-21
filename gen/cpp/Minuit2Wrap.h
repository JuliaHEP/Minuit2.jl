#ifndef Minuit2Wrap_H
#define Minuit2Wrap_H

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"
#include "Minuit2/FCNBase.h"
#include "Minuit2/FunctionMinimum.h"
#include <utility>

// BinaryBuilder uses old SDK which doesn't have ranges
#ifdef __APPLE__
  #undef JLCXX_HAS_RANGES
#endif
typedef double (*fcn_f)(const std::vector<double>&);
class JuliaFcn : public ROOT::Minuit2::FCNBase{
  public:
    JuliaFcn(fcn_f func) : m_func(func), m_errorDef(1.0) { }
    virtual double Up() const { return m_errorDef; }
    virtual double operator()(const std::vector<double>& par) const { return m_func(par);}
    void setErrorDef(double def) { m_errorDef = def; }
  private:
    fcn_f  m_func;       // the function, from the julia side
    double m_errorDef;   // error definition (chi2 or logL)
};

inline int NIter(const ROOT::Minuit2::FunctionMinimum& min) { return min.States().size(); }

//---Template instantiations-----------------------------------------------------------------------
//template class std::vector<ROOT::Minuit2::MinimumState>;
template class std::pair<double, double>;

#endif
