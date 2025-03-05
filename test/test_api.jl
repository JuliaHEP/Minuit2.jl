@testset "API" verbose=true begin
    @testset "FCN" begin
        fcn1 = FCN(rosenbrock, rosenbrock_grad)

        @test fcn1.up == 1.0 # default
        fcn1.errordef = 0.5
        @test fcn1.up == 0.5

        @test fcn1.nfcn == 0
        @test fcn1.ngrad == 0
        @test fcn1(1.0, 1.0) ≈ 0.0
        @test fcn1.nfcn == 1

        @test fcn1.has_gradient
        @test fcn1.grad(1.0, 1.0) ≈ [0.0, 0.0]
        @test fcn1.ngrad == 1

        v = StdVector([1.0,1.0])
        @test fcn1(v) ≈ 0.0
        @test fcn1.vgrad(v) ≈ [0.0, 0.0]

        fcn2 = FCN(sphere, nothing, true, 0.5)

        @test fcn2.up == 0.5
        @test fcn2(1.,1.,1.) ≈ 3.0
        @test fcn2.nfcn == 1
        @test fcn2.ngrad == 0
        @test !fcn2.has_gradient

    end

    @testset "Minuit" begin
        m = Minuit(rosenbrock, [0.0, 0.0])   # names are automatically generated

        @test_throws ArgumentError m.is_valid
        @test_throws ArgumentError m.fval

        @test m.npar == 2
        @test m.names == ["x", "y"]
        @test m.values == [0.0, 0.0]
        @test m.errors == [0.1, 0.1]
        @test m.fixed == [false, false]
        @test m.limits == [(-Inf, Inf), (-Inf, Inf)]
        @test m.errordef == 1.0
        @test m.funcname == "rosenbrock(x, y)"
        @test m.method == :migrad
        @test m.strategy == 1
        @test m.tolerance == 0.1

        m1 = Minuit(rosenbrock, x=10., y=10, 
                    error_x=0.2, error_y=0.3,
                    fix_x=true, limit_y=(0, 10),
                    errordef=0.5, strategy=2, tolerance=1e-3)

        @test m1.npar == 2
        @test m1.names == ["x", "y"]
        @test m1.values == [10.0, 10.0]
        @test m1.errors == [0.2, 0.3]
        @test m1.fixed == [true, false]
        @test m1.limits == [(-Inf, Inf), (0.0, 10.0)]
        @test m1.fcn.up == 0.5
        @test m1.funcname == "rosenbrock(x, y)"
        @test m1.method == :migrad
        @test m1.strategy == 2
        @test m1.tolerance == 1e-3

        m3 = Minuit(sphere, [1.0, 1.0, 1.0, 1.0], 
                    names=["x", "y", "z", "t"])
        @test m3.npar == 4
        @test m3.names == ["x", "y", "z", "t"]
        @test m3.values["x"] == 1.0
        @test m3.values["y"] == 1.0
        @test m3.values["z"] == 1.0
        @test m3.values["t"] == 1.0
        @test m3.fcn([1.0, 1.0, 1.0, 1.0]) ≈ 4.0
        @test m3.fcn(1.0, 1.0, 1.0, 1.0) ≈ 4.0

        # Do a minimization now
        migrad!(m)
        @test m.is_valid
        @test m.has_valid_parameters
        @test m.has_accurate_covar
        @test m.has_posdef_covar
        @test !m.has_made_posdef_covar
        @test m.has_covariance
        @test !m.hesse_failed
        @test !m.has_reached_call_limit
        @test m.fval ≈ 0.0 atol=1e-4
        @test m.edm ≈ 0.0  atol=1e-4
        @test m.nfcn > 0
        @test m.niter > 0
        @test m.values ≈ [1.0, 1.0] atol=2e-2
        @test m.errors ≈ [1.0, 2.0] atol=2e-2

        # Check simplex interface
        simplex!(m)
        @test m.is_valid
        @test m.values ≈ [1.0, 1.0] atol=2e-2

        # Check scan interface
        m = Minuit(rosenbrock, [0.0, 0.0])
        m.limits = [(-1, 1), (-1, 1)]
        scan!(m)
        @test m.is_valid
        @test m.values ≈ [1.0, 1.0] atol=2e-2


    end

    @testset "Minuit with gradient" begin
        m = Minuit(rosenbrock, [0.0, 0.0], grad=rosenbrock_grad, tolerance=1e-5)

        migrad!(m)
        @test m.is_valid
        @test m.has_valid_parameters
        @test m.has_accurate_covar
        @test m.has_posdef_covar
        #@test m.has_made_posdef_covar
        @test m.has_covariance
        @test !m.hesse_failed
        @test !m.has_reached_call_limit
        @test m.fval ≈ 0.0 atol=1e-3
        @test m.edm ≈ 0.0 atol=1e-3
        @test m.nfcn > 0
        @test m.niter > 0
        @test m.values ≈ [1.0, 1.0] atol=1e-3
    end

    @testset "Hesse Errros" begin
        m = Minuit(rosenbrock, [0.0, 0.0])
        migrad!(m)
        nfcn = m.nfcn
        @test m.is_valid

        hesse!(m)
        @test m.is_valid
        @test m.nfcn > nfcn
    end

    @testset "Minos" begin
        m = Minuit(rosenbrock, [0.0, 0.0])
        migrad!(m)
        @test m.is_valid
        errors = m.errors

        minos!(m, cl=1.)
        @test (m.minos[1].upper - m.minos[1].lower)/2 ≈ errors[1] atol=5e-2
        @test (m.minos[2].upper - m.minos[2].lower)/2 ≈ errors[2] atol=5e-2

        minos!(m, cl=0.68)
        @test (m.minos[1].upper - m.minos[1].lower)/2 ≈ errors[1] atol=5e-2
        @test (m.minos[2].upper - m.minos[2].lower)/2 ≈ errors[2] atol=5e-2

        minos!(m, cl=2.)
        @test (m.minos[1].upper - m.minos[1].lower)/2 > 2*errors[1]
        @test (m.minos[2].upper - m.minos[2].lower)/2 > 2*errors[2]
    end

    @testset "Contour" begin
        m = Minuit(sphere, [1.0, 1.0, 1.0], names=["x", "y", "z"])
        migrad!(m)
        @test m.is_valid

        x, y, f = contour(m, "x", "y", size=20)
        @test length(x) == 20
        @test length(y) == 20
        @test Base.size(f) == (20, 20)


        @test m.fcn(x[1], y[1], m.values["z"]) == f[1,1]
        @test m.fcn(x[10], y[10], m.values["z"]) == f[10,10]

        points = mncontour(m, "x", "y", size=20)
        @test length(points) == 20 + 1
        @test length(points[1]) == 2
    end

    @testset "Profile"  begin
        m = Minuit(sphere, [1.0, 1.0, 1.0], names=["x", "y", "z"])
        migrad!(m)
        @test m.is_valid


        x, f = Minuit2.profile(m, "x", size=20)
        @test length(x) == 20
        @test length(f) == 20
        @test m.fcn(x[1], m.values["y"], m.values["z"]) == f[1]
        @test m.fcn(x[10], m.values["y"], m.values["z"]) == f[10]

        x, y, ok = mnprofile(m, "x", size=20)
        @test length(x) == 20
        @test length(y) == 20
        @test length(ok) == 20
        @test all(ok)
    end

    @testset "ShowFunctions" begin
        m = Minuit(sphere, [1.0, 1.0, 1.0], names=["x", "y", "z"])
        migrad!(m)
        minos!(m)
        @test show(devnull, m) === nothing
        @test show(devnull, m.fmin) === nothing
        @test show(devnull, m.parameters) === nothing
        @test show(devnull, m.minos) === nothing
    end

end
