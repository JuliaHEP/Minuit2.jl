using Distributions
using FiniteDiff: finite_difference_gradient
using FHist

@testset "Cost functions" verbose=true begin

    _pdf(x, μ, σ) = map(Base.Fix1(pdf, Normal(μ, σ)), x)
    _cdf(x, μ, σ) = map(Base.Fix1(cdf, Normal(μ, σ)), x)
    _logpdf(x, μ, σ) = map(Base.Fix1(logpdf, Normal(μ, σ)), x)
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

    function numerical_extended_model_gradient(fcn)
        fint = (x, args...) -> finite_difference_gradient(p -> fcn(x, p...)[1], collect(args))
        f = (x, args...) -> finite_difference_gradient(p -> fcn(x, p...)[2], collect(args))
        (x, args...) -> (fint.(x, args...), f.(x, args...))
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

    @testset "BinnedNLL$(glabel(use_grad))_$use_pdf" for use_grad in (false, true), use_pdf in (:none, :approximate)
        nx, xe , (μ, σ)= binned()
        model = use_pdf == :none ? _cdf : _pdf
        cost = BinnedNLL(nx, xe, model, grad=numerical_model_gradient(model), use_pdf=use_pdf)
        @test cost.ndata == length(nx)
    
        if use_grad
            ref = numerical_cost_gradient(cost)
            @test grad(cost, [μ, σ])' ≈ ref(μ, σ)
            @test grad(cost, [-1., 3.])' ≈ ref(-1., 3.)
        end

        m = Minuit(cost, μ=0, σ=1, grad=use_grad)
        m.limits["σ"] = (0, Inf)
        migrad!(m)
        @test m.is_valid
        # binning loses information compared to unbinned case
        @test m.values ≈ [μ,σ] atol=0.15
        @test m.errors["μ"] ≈ √sum(nx)/sum(nx) atol=0.05
        @test m.ndof == length(nx) - 2

        if use_grad
            @test m.fcn.ngrad > 0
        else
            @test m.fcn.ngrad == 0
        end
    end

    @testset "ExtendedUnbinnedNLL_names" begin
        
        x, (μ, σ, n) = unbinned()
        cost = ExtendedUnbinnedNLL(x, (x, par...) -> (par[1], par[1] * _pdf(x, par[2:end]...)), names=("n", "μ", "σ"))
        @test cost.ndata == Inf

        m = Minuit(cost, n=n, μ=0, σ=1)
        m.limits["σ"] = (0, Inf)
        m.limits["n"] = (0, Inf)

        migrad!(m)
        @test m.is_valid
        @test m.values ≈ [n, μ, σ] atol=0.05

    end

    @testset "ExtendedUnbinnedNLL$model$(glabel(use_grad))" for model in (_pdf, _logpdf,), use_grad in (false, true)

        x, (μ, σ, n) = unbinned()

        is_log = model == _logpdf
        function density(x, n, μ, σ)
            if is_log
                return n, log.(n) .+ model(x, μ, σ)
            else
                return n, n * model(x, μ, σ)
            end
        end

        cost = ExtendedUnbinnedNLL(x, density, log=is_log, grad=numerical_extended_model_gradient(density))
        @test cost.errordef == 0.5
        @test cost.npar == 3
        @test cost.ndata == Inf
        
        if model ==_logpdf
            fint, f = density(x, n, μ, σ)
            @test cost(n, μ, σ) ≈ 2.0 * (fint - sum(f))
        end
        if use_grad
            ref = numerical_cost_gradient(cost)
            @test grad(cost, [1000., 1., 2.]) ≈ ref(1000., 1., 2.)
            @test grad(cost, [1000, -1., 3.]) ≈ ref(1000., -1., 3.)
        end

        m = Minuit(cost, n=length(x), μ=0, σ=1, grad=use_grad)
        m.limits["σ"] = (0, Inf)
        m.limits["n"] = (0, Inf)
        migrad!(m)

        @test m.is_valid
        @test m.values ≈ [length(x), μ, σ] atol=0.005
        @test m.errors["μ"] ≈ √n/n atol=0.05

        if use_grad
            @test m.fcn.ngrad > 0
        else
            @test m.fcn.ngrad == 0
        end
            
    end


    @testset "ExtendedBinnedNLL$(glabel(use_grad))" for use_grad in (false, true)
        nx, xe , (μ, σ)= binned()
        model = (nx, n, μ, σ) -> n .* _cdf(nx, μ, σ)

        cost = ExtendedBinnedNLL(nx, xe, model, grad=numerical_model_gradient(model))
        @test cost.ndata == length(nx)

        if use_grad
            ref = numerical_cost_gradient(cost)
            @test grad(cost, [1000., μ, σ])' ≈ ref(1000., μ, σ) rtol=1e-6
            @test grad(cost, [500., -1., 3.])' ≈ ref(500., -1., 3.)
        end

        m = Minuit(cost, nx=nx, n=length(nx), μ=0, σ=1, grad=use_grad)
        m.limits["σ"] = (0, Inf)
        m.limits["n"] = (0, Inf)
        migrad!(m)
        @test m.is_valid
        @test m.values ≈ [1000., μ, σ] rtol=0.2
        @test m.errors["μ"] ≈ √sum(nx)/sum(nx) atol=0.05
        @test m.ndof == length(nx) - 3

        if use_grad
            @test m.fcn.ngrad > 0
        else
            @test m.fcn.ngrad == 0
        end
    end

    @testset "CostSum$(glabel(use_grad))" for use_grad in (false, true)
        use_grad = false

        model1(x, a) = a + x
        grad1(x, a) = 1.
        model2(x, b, a) = a + b * x
        grad2(x, b, a) = [x, 1.]
        model3(x, c) = c
        grad3(x, c) = 1.

        lsq1 = LeastSquares([1.], [2.], 3., model1, grad=grad1)
        @test lsq1.parameters == ["a"]
        if use_grad
            grad(lsq1, 2.) ≈ numerical_cost_gradient(lsq1)(2.)[1]
        end

        lsq2 = LeastSquares([1.], [3.], 4, model2, grad=grad2)
        @test lsq2.parameters == ["b", "a"]
        if use_grad
            grad(lsq2, (2., 3.)) ≈ numerical_cost_gradient(lsq2)(2., 3.)
        end

        lsq3 = LeastSquares([1.], [1.], 1, model3, grad=grad3)
        @test lsq3.parameters == ["c"]
        if use_grad
            grad(lsq3, 4.) ≈ numerical_cost_gradient(lsq3)(4.)[1]
        end

        lsq12 = lsq1 + lsq2
        @test lsq12.items == [lsq1, lsq2]
        @test lsq12 isa Minuit2.CostSum
        @test lsq1 isa LeastSquares
        @test lsq2 isa LeastSquares
        @test lsq12.parameters == ["a", "b"]
        @test lsq12.ndata == 2
        @test lsq12(1, 2) == lsq1(1) + lsq2(2, 1)

        if use_grad
            a = 2
            b = 3
            ref = zeros(2)
            ref[1] += grad(lsq1, a)[1]
            ref[[2,1]] += grad(lsq2, (b, a))
            @test grad(lsq12, (a, b)) ≈ ref
        end

        lsq121 = lsq12 + lsq1
        @test lsq121.items == [lsq1, lsq2, lsq1]
        @test lsq121.parameters == ["a", "b"]
        @test lsq121.ndata == 3
        if use_grad
            a = 2
            b = 3
            ref = zeros(2)
            ref[1] += grad(lsq1, a)[1]
            ref[[2,1]] += grad(lsq2, (b, a))
            ref[1] += grad(lsq1, a)[1]
            @test grad(lsq121, (a,b)) ≈ ref
        end

        lsq312 = lsq3 + lsq12
        @test lsq312.items == [lsq3, lsq1, lsq2]
        @test lsq312.parameters == ["c", "a", "b"]
        @test lsq312.ndata == 3
        if use_grad
            a = 2
            b = 3
            c = 4
            ref = zeros(3)
            ref[1] += grad(lsq3, c)[1]
            ref[2] += grad(lsq1, a)[1]
            ref[[3, 2]] += grad(lsq2, (b, a))
            @test grad(lsq312, (c, a, b)) ≈ ref
        end
        lsq31212 = lsq312 + lsq12
        @test lsq31212.items == [lsq3, lsq1, lsq2, lsq1, lsq2]
        if use_grad
            a = 2
            b = 3
            c = 4
            ref = zeros(3)
            ref[:] += grad(lsq312, (c, a, b))
            ref[2:end] += grad(lsq12, (a, b))
            @test grad(lsq31212, (c, a, b)) ≈ ref
        end

        lsq31212 += lsq1
        @test lsq31212.items == [lsq3, lsq1, lsq2, lsq1, lsq2, lsq1]
        @test lsq31212.parameters == ["c", "a", "b"]
        @test lsq31212.ndata == 6

        m = Minuit(lsq12, a=0, b=0, grad=use_grad)
        migrad!(m)
        @test m.names == ["a", "b"]
        @test m.values ≈ [1, 2] atol=0.005
        @test m.errors ≈ [3, 5] atol=0.005
        @test m.covariance(1,1) ≈ 9
        @test m.covariance(1,2) ≈ -9
        @test m.covariance(2,2) ≈ 25

        if use_grad
            @test m.ngrad > 0
        else
            @test m.ngrad == 0
        end
    end
end
