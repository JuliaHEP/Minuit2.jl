@testset "Utilities" verbose=true begin
    @testset "ValueView" begin
        m = Minuit((x,y,z) -> 0. ,[1.0, 2.2, 3.3])   # Fake Minuit object
        v = Minuit2.ValueView(m)                     # ValueView object

        @test v == v
        @test v == [1.0, 2.2, 3.3]
        @test v != [1.0, 2.1, 3.3]
        @test v != 0

        v[1:end] = (1, 2, 3)

        @test v[1] == 1.0
        @test v[2] == 2.0
        @test v[3] == 3.0

        @test v["x"] == 1.0
        @test v["y"] == 2.0
        @test v["z"] == 3.0

        v[:] = (3, 2, 1)
        @test v[1] == 3.0
        @test v[2] == 2.0
        @test v[3] == 1.0

        @test ndims(v) == 1
           
    end
    @testset "ErrorView" begin
        m = Minuit((x,y,z) -> 0. ,[1.0, 2.2, 3.3])   # Fake Minuit object
        v = Minuit2.ErrorView(m)                     # ErrorView object

        @test v == v
        @test v == [0.1, 0.1, 0.1]  # The default error is 0.1
        @test v != [0.1, 0.2, 0.1]
        @test v != 0

        v[1:end] = (1., 2., 3.)

        @test v[1] == 1.0
        @test v[2] == 2.0
        @test v[3] == 3.0

        @test v["x"] == 1.0
        @test v["y"] == 2.0
        @test v["z"] == 3.0

        @test ndims(v) == 1
    end

    @testset "FixedView" begin
        m = Minuit((x,y,z) -> 0. ,[1.0, 2.2, 3.3])   # Fake Minuit object
        v = Minuit2.FixedView(m)                     # FixedView object

        @test v == v
        @test v == [false, false, false]  # The default is not fixed
        @test v != [false, true, false]
        @test v != 0

        v[1:end] = (true, false, true)

        @test v[1] == true
        @test v[2] == false
        @test v[3] == true

        @test v["x"] == true
        @test v["y"] == false
        @test v["z"] == true

    end

    @testset "LimitView" begin
        m = Minuit((x,y,z) -> 0. ,[1.0, 2.2, 3.3])   # Fake Minuit object
        v = Minuit2.LimitView(m)                     # LimitView object

        @test v == v
        @test v == [(-Inf, Inf), (-Inf, Inf), (-Inf, Inf)]  # The default is no limits
        @test v != [(0.0, Inf), (-Inf, Inf), (-Inf, Inf)]
        @test v != 0

        v[1:end] = [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]
        @test v[1] == (0.0, 1.0)
        @test v[2] == (0.0, 2.0)
        @test v[3] == (0.0, 3.0)

        @test v["x"] == (0.0, 1.0)
        @test v["y"] == (0.0, 2.0)
        @test v["z"] == (0.0, 3.0)

        v[2] = (-1.0, 1.0)
        @test v[2] == (-1.0, 1.0)
        
        v[1] = (-1.0, Inf)
        @test v[1] == (-1.0, Inf)

        v["z"] = (-Inf, 3.0)
        @test v[3] == (-Inf, 3.0)

        @test_throws BoundsError v[4] = (0.0, 1.0)
        @test_throws BoundsError v["w"] = (0.0, 1.0)
        @test_throws ArgumentError v[1] = (2.0, 1.0)

        @test m.fixed[1] == false
        v[1] = (1.0, 1.0)         # This is a valid limit
        @test v[1] == (-Inf, Inf) # The limit is removed
        @test m.fixed[1] == true
        @test m.values[1] == 1.0

        m.limits = [(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]
        @test v[1] == (0.0, 1.0)
        @test v[2] == (0.0, 2.0)
        @test v[3] == (0.0, 3.0)    

    end

    @testset "MinosView" begin
        m = Minuit(sphere, [1,1,1], names=("x","y","z")) # Fake Minuit object
        v = Minuit2.MinosView(m)                # MinosView object
        @test v == [nothing, nothing, nothing]  # The default is no Minos results
        migrad!(m)
        minos!(m, parameters=["x", "z"])
        @test v["x"].lower ≈ -1.0 atol=1e-2
        @test v["x"].upper ≈ 1.0 atol=1e-2
        @test v["z"].lower ≈ -1.0 atol=1e-2
        @test v["z"].upper ≈ 1.0 atol=1e-2
        @test v["y"] === nothing
        @test_throws ArgumentError v["x"] = (0.0, 1.0)  # Cannot set the Minos result
        @test v[1].is_valid == true
        @test v[3].is_valid == true
        @test v[2] === nothing

    end
end

