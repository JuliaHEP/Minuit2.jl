@testset "Functions" verbose=true begin
    @testset "Rosenbrock" begin
        m = Minuit(rosenbrock, x=0, y=0, 
                   grad=rosenbrock_grad, tolerance=1e-4)
        migrad!(m)
        @test m.fval ≈ 0 atol=1e-6

        @test m.values[1] ≈ 1 atol=1e-3
        @test m.values[2] ≈ 1 atol=1e-3

        @test m.values["x"] ≈ 1 atol=1e-3
        @test m.values["y"] ≈ 1 atol=1e-3

        @test m.fcn.nfcn > 0
        @test m.fcn.ngrad > 0
        @test matrix(m) ≈ [1.0382382795695269 2.076484826253819; 2.076484826253819 4.162986186802997]
    end

    @testset "Ackley" begin
        m = Minuit(ackley, x=0.3, y=-0.2, tolerance=1e-4)
        migrad!(m)
        @test m.fval ≈ 0 atol=1e-6
        @test m.values ≈ [0, 0] atol=1e-6

        @test m.fcn.nfcn > 0
        @test m.fcn.ngrad == 0
    end

    @testset "Beale" begin
        m = Minuit(beale, x=0.5, y=0.25, tolerance=1e-4)
        migrad!(m)
        @test m.fval ≈ 0 atol=1e-6
        @test m.values ≈ [3, 0.5] atol=1e-3

        @test m.fcn.nfcn > 0
        @test m.fcn.ngrad == 0
    end

    @testset "Matyas" begin
        m = Minuit(matyas, x=0.5, y=0.5, tolerance=1e-4)
        migrad!(m)
        @test m.fval ≈ 0 atol=1e-14
        @test m.values ≈ [0, 0] atol=1e-14

        @test m.fcn.nfcn > 0
        @test m.fcn.ngrad == 0
    end

    @testset "Matyas one-sided" begin
        m = Minuit(matyas, x=2.5, y=0.5, tolerance=1e-4)
        m.limits["x"] = (1, Inf)
        migrad!(m)
        @test m.values ≈ [1, 0.923] atol=1e-3
    end

    @testset "Sphere" begin
        m = Minuit(sphere, [2,2,2], names=["x","y","z"], tolerance=1e-4)
        migrad!(m)
        @test m.fval ≈ 0 atol=1e-6
        @test m.values ≈ [0, 0, 0] atol=1e-3

        @test m.fcn.nfcn > 0
        @test m.fcn.ngrad == 0
    end
end
