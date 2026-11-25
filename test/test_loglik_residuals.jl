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
    ret2 = correct!(ekf(), SA[0.0], [1.0, 2.0]) # More measurements than outputs of the model
    @test_throws DimensionMismatch loglik_residuals!(ones(3), ret2, 1)

    # Missing measurements in a filter with a quantized measurement model
    u = fill(SA[0.0], 4)
    y = [SA[1.0], missing, SA[0.0], SA[1.0]]
    res = zeros(8)
    LowLevelParticleFilters.prediction_errors!(res, ekf(), u, y; loglik = true, offset = 2)
    @test res[3:4] == [0, sqrt(2)]
    @test sum(abs2, res) ≈ -loglik(ekf(), u, y) + 4*2
end

@testset "UIKalmanFilter" begin
    A = [0.9 0.2; -0.1 0.8]
    B = [0.0; 1.0;;]
    C = [1.0 0.0; 0.3 1.0]
    G = [1.0; 0.5;;]
    uikf() = UIKalmanFilter(A, B, C, zeros(2, 1), G, 0.01I(2), 0.1I(2), LowLevelParticleFilters.SimpleMvNormal(zeros(2), I(2)))
    f = uikf()
    x̂ = copy(f.kf.x)
    ret = correct!(f, [0.0], [2.0, -1.0])
    @test ret.e ≈ [2.0, -1.0] - C*x̂
    @test ret.e_star ≈ ret.e - C*G*ret.d
    r = loglik_residuals!(zeros(3), ret, 3)
    @test sum(abs2, r) ≈ -ret.ll + 3

    T = 6
    u = [[0.1k] for k in 1:T]
    y = [[sin(k), cos(k)] for k in 1:T]
    res = zeros(3T)
    LowLevelParticleFilters.prediction_errors!(res, uikf(), u, y; loglik = true, offset = 3)
    @test sum(abs2, res) ≈ -loglik(uikf(), u, y) + 3T
end

@testset "CompositeMeasurementModel" begin
    # A component model that writes its innovation into a shared buffer
    buf = zeros(1)
    mb = UKFMeasurementModel{Float64, false, false}((x, u, p, t) -> x, [1.0;;]; nx = 1, ny = 1, innovation = (y, ŷ) -> (buf .= y .- ŷ))
    kfb = UnscentedKalmanFilter((x, u, p, t) -> x, CompositeMeasurementModel(mb, mb), [1.0;;], LowLevelParticleFilters.SimpleMvNormal([0.0], [1.0;;]); nu = 1, ny = 2)
    ret = correct!(kfb, [0.0], [2.0])
    @test ret.e ≈ [2.0, 1.0]
    @test sum(abs2, loglik_residuals!(zeros(3), ret)) ≈ -ret.ll

    dyn(x, u, p, t) = [0.9x[1] + 0.1x[2], 0.8x[2]]
    # Every component model is corrected with the full measurement vector
    m1 = LinearMeasurementModel([1.0 0.0; 0.2 1.0], 0, [0.2 0.0; 0.0 0.3])
    m2 = EKFMeasurementModel{Float64, false}((x, u, p, t) -> [x[1]*x[2], x[2]], [0.3 0.05; 0.05 0.4]; nx = 2, ny = 2)
    q = QuantizedMeasurementModel(LinearMeasurementModel([1.0 0.0; 0.0 1.0], 0, [0.05^2 0.0; 0.0 0.2]), [0.5, 0.0])
    T = 5
    u = fill([0.0], T)
    d0(μ = [1.0, -0.5]) = LowLevelParticleFilters.SimpleMvNormal(μ, eltype(μ).([0.5 0.1; 0.1 0.4]))
    for (mm, y) in [
        (CompositeMeasurementModel(m1, m2), [[0.1k, -0.3] for k in 1:T]),
        (CompositeMeasurementModel(m1, q), [[0.5round(0.2k/0.5), -0.3] for k in 1:T]),
    ]
        ekf() = ExtendedKalmanFilter(dyn, mm, 0.01I(2), d0(); nu = 1)
        ret = correct!(ekf(), u[1], y[1])
        @test ret.S isa Tuple && ret.Sᵪ isa Tuple && ret.K isa Tuple
        @test length(ret.parts) == length(mm.models)
        @test ret.e ≈ reduce(vcat, [part.e for part in ret.parts])
        @test ret.ll ≈ sum(part.ll for part in ret.parts)
        r = zeros(mm.ny + 1)
        loglik_residuals!(r, ret, 4)
        @test sum(abs2, r) ≈ -ret.ll + 4
        res = zeros(T*(mm.ny + 1))
        LowLevelParticleFilters.prediction_errors!(res, ekf(), u, y; loglik = true, offset = 4)
        @test sum(abs2, res) ≈ -loglik(ekf(), u, y) + 4T

        # Differentiation with respect to the prior mean
        function cost(μ)
            f = ExtendedKalmanFilter(dyn, mm, 0.01I(2), d0(μ); nu = 1)
            r = zeros(eltype(μ), T*(mm.ny + 1))
            LowLevelParticleFilters.prediction_errors!(r, f, u, y; loglik = true, offset = 4)
            sum(abs2, r)
        end
        llfun(μ) = -loglik(ExtendedKalmanFilter(dyn, mm, 0.01I(2), d0(μ); nu = 1), u, y)
        @test ForwardDiff.gradient(cost, [1.0, -0.5]) ≈ ForwardDiff.gradient(llfun, [1.0, -0.5])
    end
end
