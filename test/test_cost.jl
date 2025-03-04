using Distributions
using FiniteDiff: finite_difference_gradient
using FHist

@testset "Cost functions" verbose=true begin

    _pdf(x, μ, σ) = pdf(Normal(μ, σ), x)
    _cdf(x, μ, σ) = cdf(Normal(μ, σ), x)
    _logpdf(x, μ, σ) = logpdf(Normal(μ, σ), x)
    _pdf_nosig(x, par...) =  _pdf(x, par...)

    function unbinned()
        x = rand(Normal(0, 1), 1000)
        return x, (mean(x), std(x), length(x)) 
    end
    function binned()
        x, (μ, σ, n) = unbinned()
        h = Hist1D(x, binedges = range(-3, 3, 51))
        return bincounts(h), binedges(h), (μ, σ)
    end

    function  numerical_cost_gradient(fcn)
        (args...) -> finite_difference_gradient(p -> fcn(p...), collect(args))
    end
    
    function numerical_model_gradient(fcn)
        (x, args...) -> finite_difference_gradient(p -> fcn(x, p...), collect(args))
    end

    function mvnorm(μx, μy, σx, σy, ρ)
        C = [σx^2 ρ*σx*σy; ρ*σx*σy σy^2]
        return MvNormal([μx, μy], C)
    end

    glabel(x) = x ? "_grad" : ""

    @testset "UnbinnedNLL$model$(glabel(use_grad))" for model in (_pdf, _logpdf,), use_grad in (false, true)
        x, (μ, σ, n) = unbinned()

        cost = UnbinnedNLL(x, model, log=(model ==_logpdf), grad=numerical_model_gradient(model))
        @test cost.errordef == 0.5
        @test cost.npar == 2

        if model ==_logpdf
            @test cost(μ, σ) ≈ -2.0*sum(model.(x, μ, σ))
        end
        if use_grad
            ref = numerical_cost_gradient(cost)
            @test grad(cost, [1., 2.]) ≈ ref(1., 2.)
            @test grad(cost, [-1., 3.]) ≈ ref(-1., 3.)
        end

        m = Minuit(cost, μ=0, σ=1, grad=use_grad)
        m.limits["σ"] = (0, Inf)
        migrad!(m)
        @test m.is_valid
        @test m.values ≈ [μ,σ] atol=0.005
        @test m.errors["μ"] ≈ √n/n atol=0.05

        if use_grad
            @test m.fcn.ngrad > 0
        else
            @test m.fcn.ngrad == 0
        end

        # model without signature
        cost2 = UnbinnedNLL(x, _pdf_nosig, names=("μ", "σ"))
        m2 = Minuit(cost2, μ=0, σ=1)
        @test m2.names == ["μ", "σ"]
        m2.limits["σ"] = (0, Inf)
        migrad!(m2)
        @test  m2.is_valid

        # invalid 
        @test_throws ArgumentError UnbinnedNLL([1, 2, 3], _pdf, names=("μ", "σ", "foo"))
            
    end

    @testset "UnbinnedNLL_2D$(glabel(use_grad))" for use_grad in (false, true)
        model(xy, μx, μy, σx, σy, ρ) = pdf(mvnorm(μx, μy, σx, σy, ρ), xy)
        truth = 0.1, 0.2, 0.3, 0.4, 0.5
        xy = rand(mvnorm(truth...), 100)'

        cost = UnbinnedNLL(xy, model, grad=numerical_model_gradient(model))
        m = Minuit(cost, truth, grad=use_grad)
        m.limits["σx"] = (0, Inf)
        m.limits["σy"] = (0, Inf)
        m.limits["ρ"] = (-1, 1)
        migrad!(m)
        @test m.is_valid

    end

    @testset "BinnedNLL$(glabel(use_grad))" for use_grad in (false,)

        nx, xe , (μ, σ)= binned()
        cost = BinnedNLL(nx, xe, _cdf, grad=numerical_model_gradient(_cdf))
        @test cost.ndata == length(nx)
    
        if use_grad
            ref = numerical_cost_gradient(cost)
            @test grad(cost, [μ, σ]) ≈ ref(μ, σ)
            @test grad(cost, [-1., 3.]) ≈ ref(-1., 3.)
        end

        m = Minuit(cost, μ=0, σ=1, grad=use_grad)
        m.limits["σ"] = (0, Inf)
        migrad!(m)
        @test m.is_valid
        # binning loses information compared to unbinned case
        @test m.values ≈ [μ,σ] atol=0.15
        @test m.errors["μ"] ≈ √sum(nx)/sum(nx) atol=0.05
        #@test m.ndof == length(nx) - 2

        if use_grad
            @test m.fcn.ngrad > 0
        else
            @test m.fcn.ngrad == 0
        end
    end
end


