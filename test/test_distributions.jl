import FiniteDiff: finite_difference_derivative as derivative
import Distributions: pdf, cdf, truncated
using QuadGK

@testset "Distributions types" verbose=true begin
    @testset "ChebyshevDist" begin
        coeffs = [1., 0., 0., 0., 0., 0.]
        a = -1.
        b = 1.
        d = ChebyshevDist(coeffs, a, b)
        @test d.a == a
        @test d.b == b
        @test d.integral ≈ quadgk(d.polynomial, -1., 1.)[1]
        @test quadgk(x->pdf(d, x), a,b)[1] ≈ 1.
        @test cdf(d, a) ≈ 0.
        @test cdf(d, b) ≈ 1.
        @test derivative(x->cdf(d, x), 0.) ≈ pdf(d, 0.)
        @test derivative(x->cdf(d, x), -0.5) ≈ pdf(d, -0.5)
        @test derivative(x->cdf(d, x), 0.5) ≈ pdf(d, 0.5)

        coeffs = [1., 2., -2., 3., -3.]
        a = 0.
        b = 10.
        d = ChebyshevDist(coeffs, a, b)
        @test d.a == a
        @test d.b == b
        @test d.integral ≈ quadgk(d.polynomial, -1., 1.)[1]
        @test quadgk(x->pdf(d, x), a,b)[1] ≈ 1.
        @test cdf(d, a) ≈ 0.
        @test cdf(d, b) ≈ 1.
        @test derivative(x->cdf(d, x), 0.1) ≈ pdf(d, 0.1) atol=1e-8
        @test derivative(x->cdf(d, x), 5.) ≈ pdf(d, 5.) atol=1e-8
        @test derivative(x->cdf(d, x), 9.9) ≈ pdf(d, 9.9) atol=1e-8
    end


    @testset "ArgusBGDist" begin
        m₀ = 1.0
        c = -2.
        p = 0.5
        a = 0.0
        b = 2.0
        d = ArgusBGDist(m₀, c, p, a, b)
        @test d.m₀ == m₀
        @test d.c == c
        @test d.p == p
        @test d.a == a
        @test d.b == b
        @test quadgk(x->pdf(d, x), a,b)[1] ≈ 1. atol=1e-3
        @test cdf(d, a) ≈ 0.
        @test cdf(d, b) ≈ 1.
        @test derivative(x->cdf(d, x), 0.1) ≈ pdf(d, 0.1)
        @test derivative(x->cdf(d, x), 0.5) ≈ pdf(d, 0.5)
        @test derivative(x->cdf(d, x), 0.8) ≈ pdf(d, 0.8)
        @test derivative(x->cdf(d, x), 1.5) ≈ pdf(d, 1.5)
    end

end
