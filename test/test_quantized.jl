using LowLevelParticleFilters, Test, Random, LinearAlgebra, Statistics, StaticArrays, Distributions, ForwardDiff
import LeastSquaresOptim
using LowLevelParticleFilters: SimpleMvNormal, truncated_normal_moments
Random.seed!(0)

@testset "truncated_normal_moments" begin
    for (α, β) in [(-1.0, 1.0), (-0.3, 2.5), (0.5, 1.5), (-3.0, -2.0), (2.0, 3.0), (-Inf, 0.3), (0.2, Inf), (-8.0, -7.5), (6.0, 6.2), (-Inf, Inf)]
        lz, m, v = truncated_normal_moments(α, β)
        d = truncated(Normal(), α, β)
        @test lz ≈ log(cdf(Normal(), β) - cdf(Normal(), α)) rtol = 1e-8
        @test m ≈ mean(d) atol = 1e-8
        @test v ≈ var(d) rtol = 1e-6
    end
    # Far in the tail, where the probability of the band underflows in linear scale
    lz, m, v = truncated_normal_moments(40.0, 41.0)
    @test isfinite(lz) && lz < -800
    @test 40 < m < 40.1
    @test 0 < v < 1/40^2
    @test_throws ArgumentError truncated_normal_moments(1.0, 0.0)
end

# A single correction is exact up to the approximation of the posterior by a Gaussian: compare the
# mean and covariance with importance sampling of the exact posterior
@testset "single correction" begin
    m0 = SA[0.3, -0.2]
    P0 = SA[0.4 0.1; 0.1 0.2]
    c = SA[1.0 0.5]
    σv = 0.05
    Δ = 1.0
    y = SA[1.0]
    kf = KalmanFilter(SA[1.0 0; 0 1], SA[0.0; 0.0;;], c, 0, SA[0.01 0; 0 0.01], SA[σv^2;;], SimpleMvNormal(m0, P0))
    qmm = QuantizedMeasurementModel(LinearMeasurementModel(c, 0, SA[σv^2;;]), Δ)
    res = correct!(kf, qmm, SA[0.0], y)
    N = 2_000_000
    L = cholesky(P0).L
    xs = [m0 + L*randn(SVector{2}) for _ in 1:N]
    w = [cdf(Normal(), (y[1] + Δ/2 - (c*x)[1])/σv) - cdf(Normal(), (y[1] - Δ/2 - (c*x)[1])/σv) for x in xs]
    @test res.ll ≈ log(mean(w)) atol = 5e-3
    w ./= sum(w)
    μ = sum(w .* xs)
    Σ = sum(w[i]*(xs[i] - μ)*(xs[i] - μ)' for i in 1:N)
    @test kf.x ≈ μ atol = 5e-3
    @test kf.R ≈ Σ atol = 5e-3
    @test res.e ≈ y - c*m0
    @test res.S ≈ c*P0*c' + SA[σv^2;;]
end

@testset "inner measurement models" begin
    nx, ny = 3, 2
    m0 = SA[0.2, 1.1, -0.4]
    P0 = SMatrix{3,3}(Matrix(0.3I(3)) + 0.05ones(3, 3))
    C = SA[1.0 0 0.5; 0 1 -0.3]
    R2 = SA[0.02^2 0; 0 0.1^2]
    d0 = SimpleMvNormal(m0, P0)
    y = SA[0.0, 1.0]
    u = SA[0.0]
    h(x, u, p, t) = C*x
    haug(x, u, p, t, e) = C*x + e
    inners = [
        LinearMeasurementModel(C, 0, R2),
        EKFMeasurementModel{Float64, false}(h, R2; nx, ny),
        UKFMeasurementModel{Float64, false, false}(h, R2; nx, ny),
        UKFMeasurementModel{Float64, false, true}(haug, R2; nx, ny),
    ]
    results = map(inners) do inner
        kf = KalmanFilter(SMatrix{3,3}(1.0I(3)), @SMatrix(zeros(3, 1)), C, 0, SMatrix{3,3}(0.01I(3)), R2, d0)
        res = correct!(kf, QuantizedMeasurementModel(inner, 1.0), u, y)
        (kf.x, kf.R, res.ll)
    end
    for r in results[2:end]
        @test r[1] ≈ results[1][1] atol = 1e-10
        @test r[2] ≈ results[1][2] atol = 1e-10
        @test r[3] ≈ results[1][3] atol = 1e-10
    end
    # A step per output and a band for a truncating quantizer
    kf = KalmanFilter(SMatrix{3,3}(1.0I(3)), @SMatrix(zeros(3, 1)), C, 0, SMatrix{3,3}(0.01I(3)), R2, d0)
    qmm = QuantizedMeasurementModel(LinearMeasurementModel(C, 0, R2), [1.0, 0.5]; band = (y, Δ) -> (y, y + Δ))
    res = correct!(kf, qmm, u, y)
    @test isfinite(res.ll)
    # The band (0, 1) of the first output lies above its predicted mean, the estimate moves upwards
    @test (C*kf.x)[1] > (C*m0)[1]

    @test_throws ArgumentError QuantizedMeasurementModel(LinearMeasurementModel(C, 0, SA[1.0 0.1; 0.1 1.0]), 1.0)
    @test_throws ArgumentError QuantizedMeasurementModel(LinearMeasurementModel(C, 0, R2), [1.0, 1.0, 1.0])
    @test_throws ArgumentError QuantizedMeasurementModel(LinearMeasurementModel(C, 0, R2), -1.0)
    @test_throws ArgumentError QuantizedMeasurementModel(LinearMeasurementModel(C, 0, R2), [1.0, -1.0])
    @test_throws ArgumentError QuantizedMeasurementModel(LinearMeasurementModel(C, 0, R2; R12 = zeros(3, 2)), 1.0)
end

@testset "filtering with a quantized measurement model" begin
    a, Δ, σw, σv, T = 0.98, 1.0, 0.03, 0.01, 500
    A, B, C = SA[a;;], SA[1-a;;], SA[1.0;;]
    R1 = SA[σw^2;;]
    d0 = SimpleMvNormal(SA[3.2], SA[σw^2/(1-a^2);;])
    u = [SA[3.2 + 1.5sin(2π*k/400)] for k in 1:T]
    x = [d0.μ + sqrt(d0.Σ[1])*randn(SVector{1})]
    for k in 1:T-1
        push!(x, A*x[k] + B*u[k] + σw*randn(SVector{1}))
    end
    y = [Δ .* round.((C*xk + σv*randn(SVector{1})) ./ Δ) for xk in x]

    dynamics(x, u, p, t) = A*x + B*u
    qmm = QuantizedMeasurementModel(LinearMeasurementModel(C, 0, SA[σv^2;;]), Δ)
    ekf = ExtendedKalmanFilter(dynamics, qmm, R1, d0; nu = 1)
    sol = forward_trajectory(ekf, u, y)
    xh = getindex.(sol.xt, 1)
    σ = sqrt.(getindex.(sol.Rt, 1))
    e = getindex.(x, 1) .- xh
    @test sqrt(mean(abs2, e)) < 0.15
    @test 0.75 < mean(abs.(e) .<= 1.96σ) <= 1
    @test sol.ll ≈ loglik(ekf, u, y)
    ssol = smooth(sol, ekf)
    @test sqrt(mean(abs2, getindex.(x, 1) .- getindex.(ssol.xT, 1))) < sqrt(mean(abs2, e))

    # The same filter as a UKF, and the log-likelihood of a particle filter with the exact likelihood
    ukf = UnscentedKalmanFilter(dynamics, qmm, R1, d0; nu = 1, ny = 1)
    @test loglik(ukf, u, y) ≈ sol.ll rtol = 1e-8
    band_ll(x, u, y, p, t) = log(cdf(Normal(), (y[1] + Δ/2 - x[1])/σv) - cdf(Normal(), (y[1] - Δ/2 - x[1])/σv))
    pfdyn(x, u, p, t, noise = false) = A*x + B*u + noise*σw*randn(SVector{1})
    pfmeas(x, u, p, t, noise = false) = C*x
    pf = AdvancedParticleFilter(5000, pfdyn, pfmeas, band_ll, SimpleMvNormal(R1), d0)
    @test loglik(pf, u, y) ≈ sol.ll atol = 3

    # A quantized and an unquantized output in one model; with a very large noise variance of the
    # unquantized output, the result equals that of the quantized output alone
    C2 = SA[1.0; 1.0;;]
    y2 = [SA[y[k][1], x[k][1] + 0.5randn()] for k in 1:T]
    qmm2 = QuantizedMeasurementModel(LinearMeasurementModel(C2, 0, SA[σv^2 0; 0 0.5^2]), [Δ, 0])
    sol2 = forward_trajectory(ExtendedKalmanFilter(dynamics, qmm2, R1, d0; nu = 1), u, y2)
    @test isfinite(sol2.ll)
    @test sqrt(mean(abs2, getindex.(x, 1) .- getindex.(sol2.xt, 1))) < 0.15
    qmm3 = QuantizedMeasurementModel(LinearMeasurementModel(C2, 0, SA[σv^2 0; 0 1e12]), [Δ, 0])
    sol3 = forward_trajectory(ExtendedKalmanFilter(dynamics, qmm3, R1, d0; nu = 1), u, y2)
    @test getindex.(sol3.xt, 1) ≈ xh rtol = 1e-6
    # An unquantized output alone gives the Kalman filter
    lin = LinearMeasurementModel(C, 0, SA[0.5^2;;])
    y3 = [SA[x[k][1] + 0.5randn()] for k in 1:T]
    solq = forward_trajectory(ExtendedKalmanFilter(dynamics, QuantizedMeasurementModel(lin, 0.0), R1, d0; nu = 1), u, y3)
    solk = forward_trajectory(KalmanFilter(A, B, C, 0, R1, SA[0.5^2;;], d0), u, y3)
    @test solq.xt ≈ solk.xt
    @test solq.Rt ≈ solk.Rt
    @test solq.ll ≈ solk.ll

    # Cost functions based on prediction errors: sums of squares of Gaussian prediction errors are not
    # supported, the log-likelihood residuals satisfy res'res == -ll + N*offset
    @test_throws ArgumentError LowLevelParticleFilters.sse(ekf, u, y)
    @test_throws ArgumentError LowLevelParticleFilters.multistep_sse(ekf, u, y; h = 3)
    @test_throws ArgumentError LowLevelParticleFilters.multistep_prediction_errors!(zeros(3T), ekf, u, y; h = 3)
    @test_throws ArgumentError LowLevelParticleFilters.prediction_errors!(zeros(T), ekf, u, y)
    for (filt, yy, offset) in [(ekf, y, 0.0), (ekf, y, 2.0), (ExtendedKalmanFilter(dynamics, qmm2, R1, d0; nu = 1), y2, 10.0)]
        ny = length(yy[1])
        res = zeros(T*(ny + 1))
        LowLevelParticleFilters.prediction_errors!(res, filt, u, yy; loglik = true, offset)
        @test sum(abs2, res) ≈ -loglik(filt, u, yy) + T*offset
        @test all(isfinite, res)
    end
    @test_throws ErrorException LowLevelParticleFilters.prediction_errors!(zeros(2T), ExtendedKalmanFilter(dynamics, qmm3, R1, d0; nu = 1), u, y2; loglik = true)

    # Gauss-Newton on the log-likelihood residuals finds a stationary point of the log-likelihood
    function filt(θ)
        Tθ = eltype(θ)
        d0θ = SimpleMvNormal(SVector{1}(convert(Tθ, 3.2)), SMatrix{1,1}(convert(Tθ, σw^2/(1-a^2))))
        ExtendedKalmanFilter(dynamics, qmm, SA[exp(2θ[1]);;], d0θ; nu = 1, Ajac = (x, u, p, t) -> A)
    end
    res!(r, θ) = LowLevelParticleFilters.prediction_errors!(r, filt(θ), u, y; loglik = true)
    lsq = LeastSquaresOptim.optimize!(LeastSquaresOptim.LeastSquaresProblem(x = [log(0.1)], f! = res!, output_length = 2T, autodiff = :forward), LeastSquaresOptim.LevenbergMarquardt())
    @test lsq.converged
    @test abs(ForwardDiff.derivative(θ -> loglik(filt([θ]), u, y), lsq.minimizer[1])) < 1e-2

    # Gradient of the log-likelihood with respect to the process noise
    function ll(θ)
        R1θ = SA[θ[1]^2;;]
        d0θ = SimpleMvNormal(SVector{1}(convert(typeof(θ[1]), 3.2)), SA[θ[1]^2/(1-a^2);;])
        f = ExtendedKalmanFilter(dynamics, qmm, R1θ, d0θ; nu = 1)
        loglik(f, u, y)
    end
    g = ForwardDiff.gradient(ll, [σw])[1]
    h = 1e-6
    @test g ≈ (ll([σw + h]) - ll([σw - h]))/2h rtol = 1e-4
end
