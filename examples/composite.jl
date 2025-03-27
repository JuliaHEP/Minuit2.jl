using Revise
using Minuit2
using Plots
using FHist
using QuadGK
using BenchmarkTools
using Random


Random.seed!(4321)   # seed for reproducibility
using FiniteDiff: finite_difference_gradient
using ForwardDiff: gradient
using ForwardDiff, Preferences
set_preferences!(ForwardDiff, "nansafe_mode" => true)

function  numerical_cost_gradient(fcn)
    x -> finite_difference_gradient(p -> fcn(p...), x[])
end
function ad_cost_gradient(cost::CostFunction)
    x -> gradient(p -> value(cost, p), x[])
end
function numerical_model_gradient(fcn)
    (x, args...) -> finite_difference_gradient(p -> fcn(x, p...), collect(args))
end
function ad_model_gradient(fcn)
    (x, args...) -> gradient(Forward, p -> fcn(x, p...), collect(args))
end

#=
const x =  RealVar(:x, limits=(0., 10.), nbins=20)
const μ1 = RealVar(:μ1, 3., limits=(0., 5.))
const σ1 = RealVar(:σ1, .8, limits=(0.01, 3.))
const sig1 = Gaussian(:sig1, x, μ1, σ1)

const c = RealVar(:c, -0.5, limits=(-0.8, 0.2))
const bkg = Exponential(:bkg, x, c)

const model = AddPdf(:model, sig1, bkg, RealVar(:f1, 0.5, limits=(0., 1.)))

@assert model.pdf(5., 0., 1., -2., .5) ≈ .5 * sig1.pdf(5., 0., 1.) + .5 * bkg.pdf(5., -2.)
@assert quadgk(x -> model.pdf(x, 0., 1., -2., .5), x.limits...)[1] ≈ 1.
=#

function create_model(x::RooFit.RealVar, ch::Symbol) 
    μ1 = RooFit.RealVar(Symbol(:μ1_, ch), 3., limits=(0., 5.))
    σ1 = RooFit.RealVar(Symbol(:σ1_, ch), .8, limits=(0.5, 3.))
    μ2 = RooFit.RealVar(Symbol(:μ2_, ch), 6., limits=(5., 10.))
    σ2 = RooFit.RealVar(Symbol(:σ2_, ch), 1., limits=(0.5, 3.))
    sig1 = RooFit.Gaussian(Symbol(:sig1_, ch), x, μ1, σ1)
    sig2 = RooFit.Gaussian(Symbol(:sig2_, ch), x, μ2, σ2)

    f_sig1 = RooFit.RealVar(Symbol(:f_sig1_, ch), 0.5, limits=(0., 1.))
    sig = RooFit.AddPdf(Symbol(:sig_, ch), sig1, sig2, f_sig1)

    c = RooFit.RealVar(Symbol(:c_, ch), -0.5, limits=(-0.8, -0.2))
    bkg = RooFit.Exponential(Symbol(:bkg_, ch), x, c)
    f_sig = RooFit.RealVar(Symbol(:f_sig_, ch), 0.4, limits=(0., 1.))
    return RooFit.AddPdf(Symbol(:model_, ch), sig, bkg, f_sig)
end

const models = []
const data = []
const costs = []
nexp = length(ARGS) > 0 ? parse(Int, ARGS[1]) : 2
println("Number of experiments: $nexp")
for ch in 1:nexp
    _x =  RooFit.RealVar(Symbol(:x_, ch), limits=(0., 10.), nbins=20)
    model = create_model(_x, Symbol(ch))
    #model_grad = ad_model_gradient(model.pdf)
    push!(models, model)
    push!(data, RooFit.generateBinned(model, 1000))
    push!(costs, BinnedNLL(data[ch], model.pdf, use_pdf=:approximate, names=[string(p.name) for p in model.params]))
end

cost = sum(costs)
kwargs = merge([RooFit.minuitkwargs(model, randomize=true) for model in models]...)

#=
println("No gradient (built-in Minuit gradient)")
r = @benchmark begin
    m = Minuit(cost; strategy=0, kwargs...)
    migrad!(m);
end
display(r)

=#

numgrad = numerical_cost_gradient(cost)
adgrad = ad_cost_gradient(cost)

x0 = [kwargs[Symbol(p)] for p in cost.parameters] 

#=
println("\n\nNumerical gradient")    
r = @benchmark begin
    m = Minuit(x->Minuit2.cost_value(cost,x), x0; names=cost.parameters, grad=numgrad, strategy=0, kwargs...) 
    migrad!(m)
end
display(r)

println("\n\nAD gradient")
r = @benchmark begin
#    m = Minuit(cost, strategy=0, kwargs...) 
    m = Minuit(x->Minuit2.cost_value(cost,x), x0; names=cost.parameters, grad=adgrad, strategy=0, kwargs...) 
    migrad!(m)
end
display(r)
=#

m = Minuit(x->Minuit2.cost_value(cost,x), x0; names=cost.parameters, grad=adgrad, strategy=0, kwargs...) 
migrad!(m)

using Profile, PProf
m = Minuit(x->Minuit2.cost_value(cost,x), x0; names=cost.parameters, grad=adgrad, strategy=0, kwargs...) 
Profile.clear()
Profile.@profile migrad!(m)
PProf.pprof()


#m = Minuit(cost; strategy=0, kwargs...)
#Profile.clear()
#@profile begin
#    #m = Minuit(cost; strategy=0, kwargs...)
#    migrad!(m);
#end
#ProfileView.view()
