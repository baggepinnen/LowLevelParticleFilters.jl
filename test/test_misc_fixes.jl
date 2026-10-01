using LowLevelParticleFilters, ForwardDiff, LinearAlgebra, Statistics, Test
const LLPF = LowLevelParticleFilters

@testset "UKFMeasurementModel explicit ne" begin
    meas(x, u, p, t, e) = x[1:1] .+ e
    mm = LLPF.UKFMeasurementModel{Float64,false,true}(meas, [0.1;;]; nx = 2, ny = 1, ne = 1)
    @test mm.ne == 1
    @test_throws ErrorException LLPF.UKFMeasurementModel{Float64,false,true}(meas, [0.1;;]; nx = 2, ny = 1, ne = 2)
end

@testset "MeasurementOop augmented UKF" begin
    nx, nu, ny = 2, 1, 1
    dyn(x, u, p, t) = [0.9 0.1; 0 0.9] * x .+ [0; 1] .* u
    meas(x, u, p, t, e) = [x[1] + e[1]]
    function meas!(y, x, u, p, t, e)
        y[1] = x[1] + e[1]
        y
    end
    R1 = 0.01I(nx)
    R2 = [0.1;;]
    d0 = LLPF.SimpleMvNormal([1.0, 2.0], Matrix(1.0I(nx)))
    x = [1.0, 2.0]
    for ukf in (
        UnscentedKalmanFilter{false,false,false,true}(dyn, meas, R1, R2, d0; ny, nu),
        UnscentedKalmanFilter{false,true,false,true}(dyn, meas!, R1, R2, d0; ny, nu),
    )
        @test LLPF.measurement_oop(ukf)(x, [0.0], nothing, 0.0) == [1.0]
    end
end

@testset "In-place EKF/IEKF measurement with dual numbers" begin
    nx, nu, ny = 2, 1, 1
    dyn(x, u, p, t) = [0.9 0.1; 0 0.9] * x .+ [0; 1] .* u
    function meas!(y, x, u, p, t)
        y[1] = p[1] * x[1]
        y
    end
    meas(x, u, p, t) = [p[1] * x[1]]
    u = [randn(nu) for _ in 1:20]
    y = [randn(ny) for _ in 1:20]
    for F in (ExtendedKalmanFilter, IteratedExtendedKalmanFilter)
        function cost(p, m)
            T = eltype(p)
            d0 = LLPF.SimpleMvNormal(zeros(T, nx), Matrix{T}(I, nx, nx))
            f = F(dyn, m, Matrix{T}(0.01I, nx, nx), T[0.1;;], d0; nu, ny, p)
            loglik(f, u, y, p)
        end
        p = [1.5]
        g_ip = ForwardDiff.gradient(p -> cost(p, meas!), p)
        g_oop = ForwardDiff.gradient(p -> cost(p, meas), p)
        @test g_ip ≈ g_oop
    end
end

@testset "SqKalmanFilter sample_state covariance" begin
    A = [0.9 0.1; 0 0.9]
    B = [0.0; 1;;]
    C = [1.0 0]
    R1 = [1.0 0.5; 0.5 2.0]
    R2 = [0.3;;]
    kf = SqKalmanFilter(A, B, C, 0, R1, R2)
    x = zeros(2)
    u = zeros(1)
    N = 200_000
    X = reduce(hcat, [LLPF.sample_state(kf, x, u) for _ in 1:N])
    @test mean(X, dims = 2) ≈ zeros(2) atol = 0.02
    @test cov(X, dims = 2) ≈ R1 rtol = 0.02
    Y = reduce(hcat, [LLPF.sample_measurement(kf, x, u) for _ in 1:N])
    @test var(Y) ≈ R2[] rtol = 0.02
end

@testset "rollout time convention" begin
    Ts = 0.1
    f(x, u, p, t) = [t]
    u = [[0.0] for _ in 1:5]
    x = LLPF.rollout(f, [-1.0], u; Ts)
    @test length(x) == length(u) + 1
    @test reduce(vcat, x[2:end]) ≈ (0:length(u)-1) .* Ts
end
