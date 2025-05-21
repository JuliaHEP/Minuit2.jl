@testset "Multi-threading" verbose=true begin
    @testset "@threads" begin
        @test Threads.nthreads() > 1
        results = Vector{Tuple{Float64,Float64}}(undef, 100)
        valid = Vector{Bool}(undef, 100)
        Threads.@threads for i in 1:100
            m = Minuit(rosenbrock, x=rand(), y=rand(), tolerance=1e-6)
            migrad!(m)
            results[i] = Tuple(m.values)
            valid[i] = m.is_valid
        end
        # Check if all threads have valid results
        @test all(valid)
        for i in 1:100
            @test results[i][1] ≈ 1.0 atol=1e-4
            @test results[i][2] ≈ 1.0 atol=1e-4
        end
    end
end  
