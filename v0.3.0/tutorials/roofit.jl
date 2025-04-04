using Minuit2
using Minuit2.RooFit        # Load the RooFit module
using Plots                 # Plotting
theme(:boxed)

x = RealVar(:x, 0., limits=(-5., 5.)) # Create a RooRealVar for x
μ = RealVar(:μ, 0., limits=(-5., 5.)) # Create a RooRealVar for μ
σ = RealVar(:σ, 1., limits=(0.1, 5.)) # Create a RooRealVar for σ
gaus = Gaussian(:gaus, x, μ, σ) # Create a RooGaussian PDF

visualize(gaus)

data = generate(gaus, 1000); # Generate 1000 observations from the model PDF

m = fitTo(gaus, data)

visualize(m)

##---Observable
mes =  RealVar(:mes, limits=(5.20, 5.30), nbins=50)

##---Gaussian signal
sigmean = RealVar(:sigmean, 5.28, limits=(5.20, 5.30))
sigwidth = RealVar(:sigwidth, 0.0027, limits=(0.001, 0.1))
sig = Gaussian(:sig, mes, sigmean, sigwidth)

##---Build Argus background
argpar = RealVar(:argpar, -20.0, limits=(-100., -1.))
argus = ArgusPdf(:argus, mes, ConstVar(:m₀, 5.291), argpar)

##---Build the model
nsig = RealVar(:nsig, 200., limits=(0., 10000.))
nbkg = RealVar(:nbkg, 800., limits=(0., 10000.))
model = AddPdf(:model, [sig, argus], [nsig, nbkg])

##--- Generate a toyMC sample from composite PDF ---
data = generate(model, 2000)

##--- Perform extended NLL fit ---
m = fitTo(model, data)

visualize(m; legend=:topleft)

visualize(m, model, components=(:sig, :argus); nbins=50, linestyle=:dash, legend=:topleft)

# Define the observable
x =  RealVar(:x, limits=(0., 10.), nbins=20)

# Define the two signals with different widths
μ = RooFit.RealVar(:μ, 3., limits=(0., 5.))
σ1 = RooFit.RealVar(:σ1, .3, limits=(0., 3.))
σ2 = RooFit.RealVar(:σ2, 1., limits=(0., 3.))
sig1 = RooFit.Gaussian(:sig1, x, μ, σ1)
sig2 = RooFit.Gaussian(:sig2, x, μ, σ2)

# Define the background as an exponential function
c = RooFit.RealVar(:c, -0.5, limits=(-0.8, -0.2))
bkg = RooFit.Exponential(:bkg, x, c)

# Define the model as the sum of the two signals and the background
# The background is multiplied by a factor f_bkg, and the first signal by f_sig1
f_bkg = RooFit.RealVar(:f_bkg, 0.4, limits=(0., 1.))
f_sig1 = RooFit.RealVar(:f_sig1, 0.5, limits=(0., 1.))
model =  RooFit.AddPdf(:model, [bkg, sig1, sig2], [f_bkg, f_sig1])

N = 2000
data = RooFit.generate(model, N);

m = fitTo(model, data)

visualize(m, model)

visualize(m, model, components=[:bkg, :sig1, :sig2], fill=0, alpha=0.4)

N = 2000
data = RooFit.generateBinned(model, N);

plot(data, label="data", c=:blue)

m = fitTo(model, data)

visualize(m, model, components=[:bkg, :sig1, :sig2], fill=0, alpha=0.4)

draw_mncontour(m, :c, :μ)
