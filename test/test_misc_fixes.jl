using LowLevelParticleFilters, ForwardDiff, LinearAlgebra, Statistics, Test, Random, StaticArrays
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

@testset "smooth dispatch requires vector data" begin
    kf = KalmanFilter([0.9;;], [1.0;;], [1.0;;], 0, [0.1;;], [0.1;;])
    u = [randn(1) for _ in 1:10]
    y = [randn(1) for _ in 1:10]
    @test smooth(kf, u, y) isa LLPF.KalmanSmoothingSolution
    @test smooth(kf, u, y, LLPF.parameters(kf)) isa LLPF.KalmanSmoothingSolution
    @test !hasmethod(smooth, Tuple{typeof(kf), Nothing})
end

@testset "Particle smoother with SimpleMvNormal dynamics density" begin
    # Forward filtering, backward simulation evaluates the dynamics density at the difference between
    # a state and a noise-free prediction; compared with the RTS smoother of the same linear Gaussian model
    Random.seed!(1)
    A, B, C = SA[0.9;;], SA[1.0;;], SA[1.0;;]
    R1, R2 = SA[0.1^2;;], SA[0.2^2;;]
    d0 = LLPF.SimpleMvNormal(SA[0.0], SA[1.0;;])
    kf = KalmanFilter(A, B, C, 0, R1, R2, d0)
    T = 100
    u = [SA[sin(k/10)] for k in 1:T]
    x, u, y = LLPF.simulate(kf, u)
    xT = getindex.(smooth(kf, u, y).xT, 1)
    dyn(x, u, p, t, noise = false) = A*x + B*u + noise*0.1*randn(SVector{1})
    meas(x, u, p, t, noise = false) = C*x + noise*0.2*randn(SVector{1})
    meas_ll(x, u, y, p, t) = LLPF.extended_logpdf(LLPF.SimpleMvNormal(R2), y - C*x)
    apf = AdvancedParticleFilter(1000, dyn, meas, meas_ll, LLPF.SimpleMvNormal(R1), d0)
    pf = ParticleFilter(1000, (x, u, p, t) -> A*x + B*u, (x, u, p, t) -> C*x, LLPF.SimpleMvNormal(R1), LLPF.SimpleMvNormal(R2), d0)
    for f in (apf, pf)
        xb, ll = smooth(f, 200, u, y)
        @test size(xb) == (200, T)
        @test sqrt(mean(abs2, vec(smoothed_mean(xb)) .- xT)) < 0.03
    end
end
