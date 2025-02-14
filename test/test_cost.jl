using Distributions
using FiniteDiff: finite_difference_gradient
using FHist

function  numerical_cost_gradient(fcn)
    (args...) -> finite_difference_gradient(p -> fcn(p...), collect(args))
end

function numerical_model_gradient(fcn)
    (x, args...) -> finite_difference_gradient(p -> fcn(x, p...), collect(args))
end


@testset "Cost functions" verbose=true begin

    _pdf(x, μ, σ) = pdf(Normal(μ, σ), x)
    _logpdf(x, μ, σ) = logpdf(Normal(μ, σ), x)

    function unbinned()
        x = rand(Normal(0, 1), 1000)
        return x, (mean(x), std(x), length(x)) 
    end

    glabel(x) = x ? "_grad" : ""

    @testset "UnbinnedNLL$model$(glabel(use_grad))" for model in (_pdf, _logpdf), use_grad in (false, true)
        x, (μ, σ, n) = unbinned()

        cost = UnbinnedNLL(x, model, log=(model ==_logpdf), model_grad=numerical_model_gradient(model))
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

    end
end


