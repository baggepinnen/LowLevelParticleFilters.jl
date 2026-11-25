using Test, LinearAlgebra, ForwardDiff, StaticArrays, LowLevelParticleFilters

@testset "Gaussian correction" begin
    function residuals(θ)
        S = [exp(θ) 0.2; 0.2 1.4]
        e = [1.2, -0.7]
        ret = (; ll = -(dot(e, S \ e) + logdet(S) + 2log(2π))/2, e, S, Sᵪ = cholesky(S))
        r = zeros(typeof(θ), 3)
        @test loglik_residuals!(r, ret, 2) === r
        sum(abs2, r), 2 - ret.ll
    end
    @test ≈(residuals(0.3)...)
    @test ForwardDiff.derivative(x -> first(residuals(x)), 0.3) ≈ ForwardDiff.derivative(x -> last(residuals(x)), 0.3)
    S = fill(0.01, 1, 1)
    ret = (; ll=0., e=[0.], S, Sᵪ=cholesky(S))
    @test_throws DimensionMismatch loglik_residuals!(zeros(1), ret)
    @test_throws DimensionMismatch loglik_residuals!(zeros(3), ret)
    @test_throws ErrorException loglik_residuals!(zeros(2), ret) # negative radicand for offset = 0
    @test loglik_residuals!(zeros(2), ret, 3)[2] ≈ sqrt((log(0.01) + log(2π))/2 + 3)
end

@testset "Missing measurement" begin
    ret = LowLevelParticleFilters.missing_correction(KalmanFilter([0.8;;], [0.0;;], [1.0;;], 0, [0.1;;], [0.01;;]))
    @test loglik_residuals!(ones(3), ret, 4) == [0, 0, 2]
    @test_throws DimensionMismatch loglik_residuals!(zeros(0), ret, 4)
    @test_throws ErrorException loglik_residuals!(zeros(3), ret, -1)

    kf = KalmanFilter([0.8;;], [0.0;;], [1.0;;], 0, [0.1;;], [0.01;;])
    u = fill([0.], 3)
    y = [[0.2], missing, [-0.3]]
    r = zeros(6)
    LowLevelParticleFilters.prediction_errors!(r, kf, u, y; loglik=true, offset=3)
    @test r[3:4] == [0, sqrt(3)]
    @test sum(abs2, r) ≈ -loglik(kf, u, y) + 9
    @test_throws DimensionMismatch LowLevelParticleFilters.prediction_errors!(zeros(8), kf, fill([0.], 4), y; loglik=true)
end

@testset "BigFloat precision" begin
    S = big.([2.0;;])
    e = big.([2.0])
    ret = (; ll = -(dot(e, S \ e) + logdet(S) + log(2*big(π)))/2, e, S, Sᵪ = cholesky(S))
    r = loglik_residuals!(zeros(BigFloat, 2), ret, 1)
    @test sum(abs2, r) ≈ -ret.ll + 1
end

@testset "Quantized correction" begin
    A = SA[0.9;;]
    C = SA[1.0;;]
    dynamics(x, u, p, t) = A*x
    qmm = QuantizedMeasurementModel(LinearMeasurementModel(C, 0, SA[0.05^2;;]), 1.0)
    ekf() = ExtendedKalmanFilter(dynamics, qmm, SA[0.1;;], LowLevelParticleFilters.SimpleMvNormal(SA[0.0], SA[1.0;;]); nu = 1)
    f = ekf()
    ret = correct!(f, SA[0.0], SA[1.0])
    r = zeros(2)
    @test loglik_residuals!(r, ret, 1) === r
    @test sum(abs2, r) ≈ -ret.ll + 1
    @test_throws DimensionMismatch loglik_residuals!(zeros(3), ret, 1)

    # Missing measurements in a filter with a quantized measurement model
    u = fill(SA[0.0], 4)
    y = [SA[1.0], missing, SA[0.0], SA[1.0]]
    res = zeros(8)
    LowLevelParticleFilters.prediction_errors!(res, ekf(), u, y; loglik = true, offset = 2)
    @test res[3:4] == [0, sqrt(2)]
    @test sum(abs2, res) ≈ -loglik(ekf(), u, y) + 4*2
end
