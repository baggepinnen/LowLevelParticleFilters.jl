using LowLevelParticleFilters
using LowLevelParticleFilters: triangular, invtriangular, factor_from_logchol, cov_from_logchol, logchol_from_cov
using Test, Random, LinearAlgebra, StaticArrays
import ForwardDiff
Random.seed!(3)

@testset "log-Cholesky parameterization" begin
    for n in 1:4
        m = n*(n+1) ÷ 2
        θ = randn(m)
        R = cov_from_logchol(θ)
        @test size(R) == (n, n)
        @test R == R'
        @test isposdef(R)
        @test logchol_from_cov(R) ≈ θ
        U = factor_from_logchol(θ)
        @test U isa UpperTriangular
        @test all(>(0), diag(U))
        @test U'U ≈ R
        # Agreement with triangular after exponentiating the diagonal
        Ut = triangular(θ)
        for i in 1:n
            Ut[i, i] = exp(Ut[i, i])
        end
        @test Ut ≈ U
        @test invtriangular(Ut) ≈ [i == j ? exp(θ[k]) : θ[k] for (k, (i, j)) in enumerate(((i, j) for i in 1:n for j in i:n))]

        Rs = cov_from_logchol(θ, Val(n))
        @test Rs isa SMatrix{n, n}
        @test Rs ≈ R
        @test parent(factor_from_logchol(θ, Val(n))) isa SMatrix{n, n}
        @test logchol_from_cov(cholesky(R)) ≈ θ
        @test logchol_from_cov(U) ≈ θ

        # Rows of a factor with negative diagonal entries are normalized
        D = Diagonal([isodd(i) ? -1.0 : 1.0 for i in 1:n])
        @test cov_from_logchol(logchol_from_cov(UpperTriangular(D*parent(U)))) ≈ R

        # Differentiation
        J = ForwardDiff.jacobian(θ -> vec(cov_from_logchol(θ, Val(n))), θ)
        δ = 1e-6
        Jfd = reduce(hcat, [(vec(cov_from_logchol(θ .+ δ .* ((1:m) .== i))) - vec(cov_from_logchol(θ .- δ .* ((1:m) .== i))))/(2δ) for i in 1:m])
        @test J ≈ Jfd rtol=1e-6
        @test ForwardDiff.jacobian(θ -> vec(cov_from_logchol(θ)), θ) ≈ J
    end
    # Extreme values remain positive definite
    @test isposdef(cov_from_logchol([-20.0, 10.0, 15.0]))
    @test_throws ArgumentError cov_from_logchol(randn(4))
    @test_throws ArgumentError triangular(randn(4))
    @test_throws ArgumentError cov_from_logchol(randn(4), Val(2))
    @test_throws ArgumentError logchol_from_cov([1.0 2.0; 2.0 1.0])
    if VERSION >= v"1.11"
        for name in (:cov_from_logchol, :factor_from_logchol, :logchol_from_cov, :triangular, :invtriangular, :sse, :prediction_errors!, :multistep_sse, :multistep_prediction_errors!)
            @test Base.ispublic(LowLevelParticleFilters, name)
        end
    end
end
