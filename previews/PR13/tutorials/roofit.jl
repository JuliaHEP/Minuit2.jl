using Minuit2
using Minuit2.RooFit        # Load the RooFit module
using Plots                 # Plotting

x =  RealVar(:x, limits=(0., 10.), nbins=20)
μ1 = RooFit.RealVar(:μ1, 3., limits=(0., 5.))
σ1 = RooFit.RealVar(:σ1, .8, limits=(0.5, 3.))
μ2 = RooFit.RealVar(:μ2, 6., limits=(5., 10.))
σ2 = RooFit.RealVar(:σ2, 1., limits=(0.5, 3.))
sig1 = RooFit.Gaussian(:sig1, x, μ1, σ1)
sig2 = RooFit.Gaussian(:sig2, x, μ2, σ2)

f_sig1 = RooFit.RealVar(:f_sig1, 0.5, limits=(0., 1.))
sig = RooFit.AddPdf(:sig, sig1, sig2, f_sig1)

c = RooFit.RealVar(:c, -0.5, limits=(-0.8, -0.2))
bkg = RooFit.Exponential(:bkg, x, c)

f_bkg = RooFit.RealVar(:f_bkg, 0.4, limits=(0., 1.))
model =  RooFit.AddPdf(:model, bkg, sig, f_bkg)

N = 2000
data = RooFit.generate(model, N);

cost = UnbinnedNLL(data, model.pdf)
kwargs = RooFit.minuitkwargs(model, randomize=true)  # Randomize the initial parameters
m = Minuit(cost; kwargs...)
migrad!(m)

nbins = 50
scale = N * (x.limits[2] - x.limits[1]) / nbins

visualize(m, nbins=nbins)
plot!(x-> model.pdf(x, c.value, μ1.value, σ1.value,μ2.value, σ2.value, f_sig1.value, f_bkg.value) * scale, x.limits..., label="truth", linestyle=:dash)
plot!(x-> bkg.pdf(x) * f_bkg.value * scale, x.limits..., label="bkg", linestyle=:dash)
plot!(x-> sig1.pdf(x) * f_sig1.value * (1-f_bkg.value) * scale, x.limits..., label="sig1", linestyle=:dash)
plot!(x-> sig2.pdf(x) * (1-f_sig1.value) * (1-f_bkg.value) * scale, x.limits..., label="sig2", linestyle=:dash)

N = 2000
data = RooFit.generateBinned(model, N);

plot(data, label="data")

cost = BinnedNLL(data, model.pdf, use_pdf=:approximate)
kwargs = RooFit.minuitkwargs(model, randomize=true)  # Randomize the initial parameters
m = Minuit(cost; kwargs...)
migrad!(m)

scale = N * (x.limits[2] - x.limits[1]) / x.nbins
visualize(m)
plot!(x-> model.pdf(x, c.value, μ1.value, σ1.value,μ2.value, σ2.value, f_sig1.value, f_bkg.value) * scale, x.limits..., label="truth", linestyle=:dash)
