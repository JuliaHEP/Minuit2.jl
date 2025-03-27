using Minuit2.RooFit
import Distributions: Exponential as _Exponential, Normal, pdf, cdf, truncated
using FHist
using QuadGK

@testset "RooFit types" verbose=true begin

    @testset "RealVar" begin
        x = RealVar(:x, limits=(0., 10.), nbins=20)
        @test x.name == :x
        @test x.value == 0.
        @test x.limits == (0., 10.)
        @test x.nbins == 20

        x = RealVar(:x, 1.0)
        @test x.value == 1.0
        @test x.limits == (-Inf, Inf)
        @test x.nbins == 50

        # Other Real types
        x = RealVar(:x, 0.0f0, limits=(0.0f0, 10.0f0))
        @test x.value == 0.0f0
        @test x.limits == (0.0f0, 10.0f0)
    end

    @testset "Gaussian" begin
        x = RealVar(:x, limits=(0., 10.), nbins=20)
        μ = RealVar(:μ, 3., limits=(0., 5.))
        σ = RealVar(:σ, .8, limits=(0.5, 3.))
        g = Gaussian(:gauss, x, μ, σ)
        @test g.name == :gauss
        @test g.x == x
        @test g.μ == μ
        @test g.σ == σ
        @test g.pdf(5.) ≈ pdf(truncated(Normal(3., .8),0.,10.), 5.)
        @test g.pdf(5., 0., 1.) ≈ pdf(truncated(Normal(0., 1.),0.,10.), 5.)
        @test quadgk(x -> g.pdf(x), x.limits...)[1] ≈ 1.

        x1 = RealVar(:x)
        g1 = Gaussian(:gauss_nolimit, x1, μ, σ)
        @test g1.pdf(5.) ≈ pdf(Normal(3., .8), 5.)
        @test quadgk(x -> g1.pdf(x), x1.limits...)[1] ≈ 1.

        # Generate
        data = generate(g, 1000)
        @test length(data) == 1000
        @test minimum(data) >= 0.
        @test maximum(data) <= 10.

        # Other Real types
        x = RealVar(:x, limits=(0.0f0, 10.0f0), nbins=20)
        μ = RealVar(:μ, 3.0f0, limits=(0.0f0, 5.0f0))
        σ = RealVar(:σ, 0.8f0, limits=(0.5f0, 3.0f0))
        g = Gaussian(:gauss32, x, μ, σ)
        @test g.pdf(5.0f0) ≈ pdf(truncated(Normal(3.0f0, 0.8f0),0.0f0,10.0f0), 5.0f0)
        @test quadgk(x -> g.pdf(x), x.limits...)[1] ≈ 1.0f0
    end

    @testset "Exponential" begin
        x = RealVar(:x, limits=(0., 10.), nbins=20)
        c = RealVar(:c, -0.5, limits=(-0.8, -0.2))
        e = Exponential(:exp, x, c)
        @test e.name == :exp
        @test e.x == x
        @test e.c == c
        @test e.pdf(5.) ≈ pdf(truncated(_Exponential(1/0.5), 0.,10), 5.)
        @test e.pdf(5., -2.) ≈ pdf(truncated(_Exponential(1/2), 0.,10), 5.)
        @test quadgk(x -> e.pdf(x), x.limits...)[1] ≈ 1.

        x = RealVar(:x)
        e = Exponential(:exp_nolimit, x, c)
        @test e.pdf(5.) ≈ pdf(_Exponential(1/0.5), 5.)
        @test quadgk(x -> e.pdf(x), x.limits...)[1] ≈ 1.

        # Other Real types
        x = RealVar(:x, limits=(0.0f0, 10.0f0), nbins=20)
        c = RealVar(:c, -0.5f0, limits=(-0.8f0, -0.2f0))
        e = Exponential(:exp32, x, c)
        @test e.pdf(5.0f0) ≈ pdf(truncated(_Exponential(1/0.5f0), 0.0f0, 10.0f0), 5.0f0)
        @test quadgk(x -> e.pdf(x), x.limits...)[1] ≈ 1.0f0
    end

    @testset "AddPdf" begin
        x = RealVar(:x, limits=(0., 10.))
        μ1 = RealVar(:μ1, 3., limits=(0., 5.))
        σ1 = RealVar(:σ1, .8, limits=(0.5, 3.))
        sig1 = Gaussian(:sig1, x, μ1, σ1)

        μ2 = RealVar(:μ2, 6., limits=(5., 10.))
        σ2 = RealVar(:σ2, 1., limits=(0.5, 3.))
        sig2 = Gaussian(:sig2, x, μ2, σ2)

        f_sig1 = RealVar(:f_sig1, 0.5, limits=(0., 1.))
        sig = AddPdf(:sig, sig1, sig2, f_sig1)
        @test sig.pdf(5., μ1.value, σ1.value, μ2.value, σ2.value, 0.5) ≈ .5 * sig1.pdf(5.) + .5 * sig2.pdf(5.)

        c = RealVar(:c, -0.5, limits=(-0.8, -0.2))
        bkg = Exponential(:bkg, x, c)
        f_sig = RealVar(:f_sig, 0.4, limits=(0., 1.))
        model = AddPdf(:model, sig, bkg, f_sig)

        @test model.pdf(5., μ1.value, σ1.value, μ2.value, σ2.value, 0.5, c.value, 0.8) ≈ 0.8 * (0.5 * sig1.pdf(5.) + 0.5 * sig2.pdf(5.)) + 0.2 * bkg.pdf(5.)
        @test quadgk(x -> model.pdf(x, 2., 1., 6., 1., 0.5, -2., .5), x.limits...)[1] ≈ 1.

        # Fit the model with generated data
        h = generateBinned(model, 1000)
        m = Minuit(BinnedNLL(h, model.pdf, use_pdf=:approximate), [p.value for p in model.params])
        migrad!(m)
        @test m.is_valid
        @test m.values ≈ [μ1.value, σ1.value, μ2.value, σ2.value, f_sig1.value, c.value, f_sig.value] rtol=1e-1
    end

end
