using Distributions
using FHist

@testset "Cost functions" verbose=true begin

    _pdf(x, μ, σ) = pdf(Normal(μ, σ), x)
    _logpdf(x, μ, σ) = logpdf(Normal(μ, σ), x)

    function unbinned()
        x = rand(Normal(0, 1), 1000)
        return x, (mean(x), std(x), length(x)) 
    end

    gradlabel(x) = x ? "_grad" : ""

    @testset "UnbinnedNLL$model" for model in (_pdf, _logpdf)
        x, (μ, σ, n) = unbinned()

        fcn = UnbinnedNLL(x, model, log=model==_logpdf)
        
        @test fcn.errordef == 0.5

    end
end