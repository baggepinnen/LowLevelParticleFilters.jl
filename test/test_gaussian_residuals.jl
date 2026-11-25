using Test, LinearAlgebra, ForwardDiff, LowLevelParticleFilters

@testset "Gaussian likelihood residuals of one correction" begin
    function residuals(θ)
        S = [exp(θ) 0.2; 0.2 1.4]
        e = [1.2, -0.7]
        ret = (; ll = -(dot(e, S \ e) + logdet(S) + 2log(2π))/2, e, S, Sᵪ = cholesky(S))
        r = zeros(typeof(θ), 3)
        @test gaussian_loglik_residuals!(r, ret, 2) === r
        sum(abs2, r), 2 - ret.ll
    end
    @test ≈(residuals(0.3)...)
    @test ForwardDiff.derivative(x -> first(residuals(x)), 0.3) ≈ ForwardDiff.derivative(x -> last(residuals(x)), 0.3)
    S = fill(0.01, 1, 1)
    ret = (; ll=0., e=[0.], S, Sᵪ=cholesky(S))
    @test_throws DimensionMismatch gaussian_loglik_residuals!(zeros(1), ret)
    @test_throws ErrorException gaussian_loglik_residuals!(zeros(2), ret)
    kf = KalmanFilter([0.8;;], [0.0;;], [1.0;;], 0, [0.1;;], S)
    u = fill([0.], 3)
    y = [[0.2], missing, [-0.3]]
    r = zeros(6)
    LowLevelParticleFilters.prediction_errors!(r, kf, u, y; loglik=true, offset=3)
    @test r[3:4] == [0, sqrt(3)]
    @test sum(abs2, r) ≈ -loglik(kf, u, y) + 9
end
