#ifndef Minuit2Wrap_H
#define Minuit2Wrap_H

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"
#include "Minuit2/FCNBase.h"
#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserFcn.h"
#include <utility>

// BinaryBuilder uses old SDK which doesn't have ranges
#ifdef __APPLE__
  #undef JLCXX_HAS_RANGES
#endif
typedef double (*fcn_f)(const std::vector<double>&);
typedef void (*fcn_g)(std::vector<double>&, const std::vector<double>&);
typedef double (*cost_f)(void*, const std::vector<double>&);
typedef void (*cost_g)(void*, std::vector<double>&, const std::vector<double>&);
class JuliaFcn : public ROOT::Minuit2::FCNBase{
  public:
    JuliaFcn(fcn_f func, double errordef) : m_func(func), m_grad(nullptr), m_costf(nullptr), m_costg(nullptr), m_costobj(nullptr), m_errorDef(errordef) { }
    JuliaFcn(fcn_f func, fcn_g grad, double errordef) : m_func(func), m_grad(grad), m_costf(nullptr), m_costg(nullptr), m_costobj(nullptr), m_errorDef(errordef) { }
    JuliaFcn(cost_f func, void* obj, double errordef) : m_func(nullptr), m_grad(nullptr), m_costf(func), m_costg(nullptr), m_costobj(obj), m_errorDef(errordef) { }
    JuliaFcn(cost_f func, cost_g grad, void* obj, double errordef) : m_func(nullptr), m_grad(nullptr), m_costf(func), m_costg(grad), m_costobj(obj), m_errorDef(errordef) { }
    virtual double Up() const override { return m_errorDef; }
    virtual double operator()(const std::vector<double>& par) const override { nfcn++; return m_costobj == nullptr ? m_func(par) : m_costf(m_costobj, par);}
    virtual std::vector<double> Gradient(std::vector<double> const& par) const override { 
      ngrad++;
      std::vector<double> grad(par.size());
      m_costobj == nullptr ? m_grad(grad, par) : m_costg(m_costobj, grad, par);
      return grad;
    }
    virtual bool HasGradient() const override { return m_grad != nullptr || m_costg != nullptr; }
    void SetErrorDef(double def) override { m_errorDef = def; }
    unsigned int Nfcn() const { return nfcn; }
    unsigned int Ngrad() const {return ngrad;}
  private:
    fcn_f  m_func;       // the function, from the julia side
    fcn_g  m_grad;       // the gradient, from the julia side
    cost_f  m_costf;     // the cost method, from the julia side
    cost_g  m_costg;     // the cost gradient, from the julia side
    void*  m_costobj;    // Cost function object
    double m_errorDef;   // error definition (chi2 or logL)
    mutable unsigned int nfcn = 0;
    mutable unsigned int ngrad = 0;
};

inline int NIter(const ROOT::Minuit2::FunctionMinimum& min) { return min.States().size(); }

namespace ROOT {
  namespace Minuit2 {
    class MnUserParameterState;
    class MnStrategy;
    class MnContours;
    FunctionMinimum createFunctionMinimum(const JuliaFcn& fcn, const MnUserParameterState& st,
                                          const MnStrategy& str, double edm_goal);
    MnUserParameterState createMnUserParameterState(const MnUserParameterState& state);  
  }
}

//---MnContours---------------------------------------------------------------------------------------
// Use a strucure instead of a pair to avoid CxxWrap problems
class XYPoint {
  public:
    XYPoint(double x, double y) : x(x), y(y) {}
    XYPoint() : x(0), y(0) {}
    XYPoint(const XYPoint&) = default;
    double X() const { return x; }
    double Y() const { return y; }
  private:
    double x;
    double y;
};
std::vector<XYPoint> paren(const ROOT::Minuit2::MnContours& contour, unsigned int i, unsigned int j, unsigned int npoints);

//---Template instantiations-----------------------------------------------------------------------
//template class std::vector<ROOT::Minuit2::MinimumState>;
template class std::pair<double, double>;

#endif
