import Plots

@testset "Plots extension" begin
    backend = Plots.backend()
    ext = Base.get_extension(Minuit2, :Minuit2PlotsExt)
    @test ext !== nothing

    @testset "LeastSquares recipe" begin
        model(x, a, b) = a + b * x
        x = collect(1.0:5.0)
        cost = LeastSquares(x, 2.0 .+ 3.0 .* x, 0.1, model)
        m = Minuit(cost, a=1.0, b=1.0)
        migrad!(m)

        @test Plots.plot(m) isa Plots.Plot
        @test Plots.plot(cost, true, collect(m.values)) isa Plots.Plot
        @test visualize(m) isa Plots.Plot
    end

    @testset "Unbinned recipe" begin
        data = [-1.0, -0.5, 0.0, 0.5, 1.0]
        model(x, mu, sigma) = exp(-0.5 * ((x - mu) / sigma)^2) / (sqrt(2pi) * sigma)
        cost = UnbinnedNLL(data, model, names=("mu", "sigma"))

        @test Plots.plot(cost, true, [0.0, 1.0], bins=5) isa Plots.Plot
        @test ext.visualize(cost, true, [0.0, 1.0], nbins=5) isa Plots.Plot
    end

    @testset "Binned recipe" begin
        counts = [1.0, 2.0, 1.0]
        edges = 0.0:1.0:3.0
        model(x, scale) = scale * x / 3
        cost = BinnedNLL(counts, edges, model, use_pdf=:approximate)

        @test Plots.plot(cost, true, [1.0]) isa Plots.Plot
        @test ext.visualize(cost, true, [1.0]) isa Plots.Plot
    end

    @testset "CostSum recipe" begin
        model1(x, a) = a + x
        model2(x, b) = b * x
        cost1 = LeastSquares([1.0, 2.0], [2.0, 3.0], 0.1, model1)
        cost2 = LeastSquares([1.0, 2.0], [3.0, 6.0], 0.1, model2)
        cost = cost1 + cost2

        @test Plots.plot(cost, true, [1.0, 3.0]) isa Plots.Plot
        @test ext.visualize(cost, true, [1.0, 3.0]) isa Plots.Plot
    end

    @test backend === Plots.backend()
end
