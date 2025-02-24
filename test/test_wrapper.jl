using CxxWrap

fn(x, y) = 10 + x^2 + ((y - 1) / 2)^2
fn_grad(x, y) = (2 * x, y - 1)

@testset "Wrapper" verbose=true begin

    @testset "MnStrategy" begin
        @test MnStrategy().strategy  == 1
        @test MnStrategy(0).strategy == 0 
        @test MnStrategy(2).strategy == 2
        s = MnStrategy()
        s.strategy = 2
        @test s.strategy == 2
        s.gradient_ncycles = 3
        @test s.gradient_ncycles == 3
        s.hessian_ncycles = 4
        @test s.hessian_ncycles == 4
    end

    @testset "MnUserCovariance" begin
        c1 = MnUserCovariance([1., 2., 3.], 2)
        @test c1.nrow == 2
        @test c1[1,1] == 1
        @test c1[2,1] == 2
        @test c1[1,2] == 2
        @test c1[2,2] == 3
        c2 = MnUserCovariance([1., 2., 4.], 2)
        @test c1 != c2
        @test c1 == c1
    end

    @testset "MnUserParameterState" begin
        st = MnUserParameterState()
        Add(st, "x", 1, 0.2)
        Add(st, "😁", 3, 0.3, 1, 4)
        @test length(st) == 2
        @test st[1].number == 1
        @test st[1].name == "x"
        @test st[1].value == 1
        @test st[1].error == 0.2
        @test st[2].number == 2
        @test st[2].name == "😁"
        @test st[2].value == 3
        @test st[2].error == 0.3
        @test st[2].lower_limit == 1
        @test st[2].upper_limit == 4

        st2 = copy(st)
        @test st2 == st

        set_value(st2, 1, 1.1)
        @test st2[1].value == 1.1
        set_error(st2, 1, 0.3)
        @test st2[1].error == 0.3
        set_limits(st2, 1, 0.5, 1.5)
        @test st2[1].has_limits
        @test st2[1].lower_limit == 0.5
        @test st2[1].upper_limit == 1.5
        set_upper_limit(st2, 1, 1.2)
        @test st2[1].upper_limit == 1.2
        set_lower_limit(st2, 1, 0.8)
        @test st2[1].lower_limit == 0.8
        remove_limits(st2, 1)
        @test !st2[1].has_limits
        fix(st2, 1)
        @test st2[1].is_fixed
        release(st2, 1)
        @test !st2[1].is_fixed
  
        @test st2 != st
    end

    @testset "MnMigrad" begin
        fcn = FCN(fn, nothing, false, 1.0)
        state = MnUserParameterState()
        Add(state, "x", 5, 0.1)
        Add(state, "y", 3, 0.2, -5, 5)
        migrad = MnMigrad(fcn, state, MnStrategy(1))
        fmin = migrad(0, 0.1)
        @test fmin.is_valid
        state = fmin.state
        @test isapprox(state[1].value, 0, atol=5e-3)
        @test isapprox(state[1].error, 1, atol=5e-3)
        @test isapprox(state[2].value, 1, atol=5e-3)
        @test isapprox(state[2].error, 2, atol=6e-2)
        @test fcn.nfcn > 0
        @test fcn.ngrad == 0
    end

    @testset "MnMigrad_grad" begin
        fcn = FCN(fn, fn_grad, false, 1.0)
        state = MnUserParameterState()
        Add(state, "x", 5, 0.1)
        Add(state, "y", 3, 0.2, -5, 5)
        migrad = MnMigrad(fcn, state, MnStrategy(1))
        fmin = migrad(0, 0.1)
        @test fmin.is_valid
        state = fmin.state
        @test length(state) == 2
        @test isapprox(state[1].value, 0, atol=1e-2)
        @test isapprox(state[1].error, 1, atol=1e-2)
        @test isapprox(state[2].value, 1, atol=1e-2)
        @test isapprox(state[2].error, 2, atol=6e-2)
        @test fcn.nfcn > 0
        @test fcn.ngrad > 0
    end

    @testset "MnScan" begin
        fcn = FCN(x  -> 10 + x^2, nothing, false, 1.0)
        state = MnUserParameterState()
        Add(state, "x", 2, 5)
        scan = MnScan(fcn, state, MnStrategy(1))
        fmin = scan(0, 0.1)
        @test fmin.is_valid
        state = fmin.state
        @test isapprox(state[1].value, 0, atol=1e-2)
    end

    @testset "MnSimplex" begin
        fcn = FCN(x -> 10 + x^2, nothing, false, 1.0)
        state = MnUserParameterState()
        Add(state, "x", 2, 5)
        simplex = MnSimplex(fcn, state, MnStrategy(1))
        fmin = simplex(0, 0.1)
        @test fmin.is_valid
        state = fmin.state
        @test length(state) == 1
        @test isapprox(state[1].value, 0, atol=5e-2)
    end

    @testset "FunctionMinimum" begin
        fcn = FCN(x -> 10 + x^2, nothing, false, 1.0)
        state = MnUserParameterState()
        Add(state, "x", 0.01, 5)
        str = MnStrategy(1)
        fm1 = FunctionMinimum(fcn, state, str, 0.2)
        @test fm1.is_valid
        @test length(fm1.state) == 1
        @test isapprox(fm1.fval, 10.0001, atol=1e-4)
        fm2 = FunctionMinimum(fcn, state, str, 0)
        @test !fm2.is_valid
    end
end
