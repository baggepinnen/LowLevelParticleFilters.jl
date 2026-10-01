using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal, multistep_sse, multistep_prediction_errors!, prediction_errors!
using Test, Random, LinearAlgebra, StaticArrays, Distributions
import ForwardDiff
Random.seed!(1)

nx, nu, ny = 2, 1, 1
Ts = 0.5
# Time-varying and input-dependent system with direct feedthrough, the zero-cost test below thereby verifies the alignment of u and t
Afun(x, u, p, t) = SA[0.95 0.1; -0.05 0.9+0.05sin(t)]
B = SA[0.0; 1.0;;]
Cfun(x, u, p, t) = SA[1.0 0.2cos(t)]
D = SA[0.5;;]
R1 = SA[0.01 0; 0 0.02]
R2 = SA[0.1;;]
d0 = SimpleMvNormal(SA[1.0, -0.5], SMatrix{2,2}(0.1I(2)))
kf = KalmanFilter(Afun, B, Cfun, D, R1, R2, d0; Ts, nx, nu, ny, check=false)
T = 80
u = [SA[randn()] for _ in 1:T]

@testset "Zero cost for noise-free data" begin
    x, _, y = simulate(kf, u; dynamics_noise=false, measurement_noise=false)
    for h in (1, 3, 10)
        @test multistep_sse(kf, u, y; h) < 1e-20
    end
    x, _, yn = simulate(kf, u)
    @test multistep_sse(kf, u, yn; h=5) > 1e-2

    # EKF with nonlinear pendulum dynamics, discretized with RK4
    pendulum(x, u, p, t) = SA[x[2], -p[1]*sin(x[1]) - p[2]*x[2] + u[1]]
    discrete_pendulum = LowLevelParticleFilters.rk4(pendulum, 0.05)
    meas(x, u, p, t) = SA[x[1]]
    ptrue = [9.81, 0.3]
    ekf = ExtendedKalmanFilter(discrete_pendulum, meas, 1e-4*SMatrix{2,2}(I(2)), SA[1e-2;;], SimpleMvNormal(SA[0.5, 0.0], SMatrix{2,2}(0.01I(2))); nu=1, p=ptrue, Ts=0.05)
    up = [SA[0.5sin(0.1k)] for k in 1:200]
    _, _, yp = simulate(ekf, up; dynamics_noise=false, measurement_noise=false)
    @test multistep_sse(ekf, up, yp; h=20) < 1e-20
    @test multistep_sse(ekf, up, yp, [9.0, 0.3]; h=20) > 1e-3

    ukf = UnscentedKalmanFilter(discrete_pendulum, meas, 1e-4*SMatrix{2,2}(I(2)), SA[1e-2;;], SimpleMvNormal(SA[0.5, 0.0], SMatrix{2,2}(0.01I(2))); nu=1, ny=1, p=ptrue, Ts=0.05)
    # The sigma-point mean of the UKF differs slightly from the propagated mean, the cost is therefore small but not zero
    c_true = multistep_sse(ukf, up, yp; h=20)
    @test c_true < 1e-3*multistep_sse(ukf, up, yp, [9.0, 0.3]; h=20)

    # In-place dynamics give the same result as out-of-place dynamics
    function discrete_pendulum!(xp, x, u, p, t)
        xp .= discrete_pendulum(x, u, p, t)
    end
    function meas!(y, x, u, p, t)
        y .= meas(x, u, p, t)
    end
    ekf_ip = ExtendedKalmanFilter(discrete_pendulum!, meas!, 1e-4*Matrix(I(2)), [1e-2;;], SimpleMvNormal([0.5, 0.0], Matrix(0.01I(2))); nu=1, ny=1, p=ptrue, Ts=0.05)
    yn = [y .+ 0.1randn() for y in yp]
    @test multistep_sse(ekf_ip, up, yn; h=7) ≈ multistep_sse(ekf, up, yn; h=7)
end

@testset "Relation to one-step prediction errors" begin
    x, _, y = simulate(kf, u)
    res1 = zeros(T*ny)
    prediction_errors!(res1, kf, u, y)
    resm = zeros(T*ny)
    multistep_prediction_errors!(resm, kf, u, y; h=1)
    @test resm[1:end-ny] ≈ res1[ny+1:end]
    @test all(iszero, resm[end-ny+1:end])
    e1 = res1[1:ny]
    @test multistep_sse(kf, u, y; h=1) ≈ LowLevelParticleFilters.sse(kf, u, y) - e1'e1

    # Layout for h = 3, compared with predictions computed from the filtered estimates
    h = 3
    sol = forward_trajectory(kf, u, y)
    resh = zeros(T*h*ny)
    multistep_prediction_errors!(resh, kf, u, y; h)
    for k in (1, 10, T-2, T-1, T)
        xj = sol.xt[k]
        for j in 1:h
            i = k + j
            inds = ((k-1)*h + j-1)*ny .+ (1:ny)
            if i > T
                @test all(iszero, resh[inds])
                continue
            end
            tprev = (i-2)*Ts
            xj = Afun(xj, u[i-1], nothing, tprev)*xj + B*u[i-1]
            ti = (i-1)*Ts
            ŷ = Cfun(xj, u[i], nothing, ti)*xj + D*u[i]
            @test resh[inds] ≈ y[i] - ŷ
        end
    end
    @test sum(abs2, resh) ≈ multistep_sse(kf, u, y; h)

    # Weighting
    λ = Diagonal(SA[4.0])
    @test multistep_sse(kf, u, y, LowLevelParticleFilters.parameters(kf), λ; h) ≈ 4multistep_sse(kf, u, y; h)
    @test multistep_sse(kf, u, y; h, horizon_weights=[1, 0, 0]) ≈ multistep_sse(kf, u, y; h=1)
    resw = zeros(T*h*ny)
    multistep_prediction_errors!(resw, kf, u, y; h, horizon_weights=[1, 2, 0.5])
    @test sum(abs2, resw) ≈ multistep_sse(kf, u, y; h, horizon_weights=[1, 2, 0.5])
end

@testset "Missing measurements" begin
    x, _, y = simulate(kf, u; dynamics_noise=false, measurement_noise=false)
    ym = Vector{Union{Missing, eltype(y)}}(y)
    ym[[5, 6, 30]] .= missing
    @test multistep_sse(kf, u, ym; h=4) < 1e-20
    x, _, yn = simulate(kf, u)
    ymn = Vector{Union{Missing, eltype(yn)}}(yn)
    ymn[[5, 6, 30]] .= missing
    h = 4
    res = zeros(T*h*ny)
    multistep_prediction_errors!(res, kf, u, ymn; h)
    for k in 1:T, j in 1:h
        i = k + j
        inds = ((k-1)*h + j-1)*ny .+ (1:ny)
        if i > T || ymn[i] === missing
            @test all(iszero, res[inds])
        end
    end
    @test sum(abs2, res) ≈ multistep_sse(kf, u, ymn; h)
end

@testset "Augmented noise and other filters" begin
    A = SA[0.95 0.1; -0.05 0.9]
    C = SA[1.0 0.0]
    dyn(x, u, p, t) = A*x + B*u
    meas(x, u, p, t) = C*x
    Bw = SA[0.0; 1.0;;]
    dyn_w(x, u, p, t, w) = A*x + B*u + Bw*w
    meas_e(x, u, p, t, e) = C*x + e
    Rw = SA[0.02;;]
    kf_lin = KalmanFilter(A, B, C, 0, Bw*Rw*Bw' + 1e-8I, R2, d0)
    _, _, y = simulate(kf_lin, u)
    h = 5
    c_kf = multistep_sse(kf_lin, u, y; h)
    ukf_w = UnscentedKalmanFilter{false,false,true,true}(dyn_w, meas_e, Rw, R2, d0; ny, nu)
    @test multistep_sse(ukf_w, u, y; h) ≈ c_kf rtol=1e-4
    sqkf = SqKalmanFilter(A, B, C, 0, Bw*Rw*Bw' + 1e-8I, R2, d0)
    @test multistep_sse(sqkf, u, y; h) ≈ c_kf rtol=1e-6
    iekf = IteratedExtendedKalmanFilter(dyn, meas, Bw*Rw*Bw' + 1e-8I, R2, d0; nu)
    @test multistep_sse(iekf, u, y; h) ≈ c_kf rtol=1e-6
    enkf = EnsembleKalmanFilter(dyn, meas, Bw*Rw*Bw' + 1e-8I, R2, d0, 2000; nu, ny)
    @test multistep_sse(enkf, u, y; h) ≈ c_kf rtol=0.2
    enkf_aug = EnsembleKalmanFilter{true,true}(dyn_w, meas_e, Rw, R2, d0, 2000; nu, ny)
    @test multistep_sse(enkf_aug, u, y; h) ≈ c_kf rtol=0.2
end

@testset "Gradient" begin
    A = SA[0.95 0.1; -0.05 0.9]
    C = SA[1.0 0.0]
    _, _, y = simulate(KalmanFilter(A, B, C, 0, R1, R2, d0), u)
    function cost(θ)
        T_ = eltype(θ)
        Aθ = SA[θ[1] 0.1; -0.05 θ[2]]
        kfθ = KalmanFilter(Aθ, B, C, 0, R1, R2, SimpleMvNormal(T_.(d0.μ), T_.(d0.Σ)); check=false)
        multistep_sse(kfθ, u, y; h=6)
    end
    θ0 = [0.9, 0.85]
    g = ForwardDiff.gradient(cost, θ0)
    δ = 1e-6
    gfd = [(cost(θ0 .+ δ .* (1:2 .== i)) - cost(θ0 .- δ .* (1:2 .== i)))/(2δ) for i in 1:2]
    @test g ≈ gfd rtol=1e-5
end

@testset "Argument errors" begin
    x, _, y = simulate(kf, u)
    @test_throws ArgumentError multistep_sse(kf, u, y; h=0)
    @test_throws ArgumentError multistep_prediction_errors!(zeros(T*ny), kf, u, y; h=2)
    @test_throws ArgumentError multistep_sse(kf, u, y; h=2, horizon_weights=[1.0])
    @test_throws ArgumentError multistep_sse(kf, u, y; h=2, horizon_weights=[1.0, -1.0])
    pf = ParticleFilter(100, (x,u,p,t)->x, (x,u,p,t)->x[1:1], MvNormal(zeros(2), I(2)), MvNormal(zeros(1), I(1)), MvNormal(zeros(2), I(2)); nu, ny)
    @test_throws ArgumentError multistep_sse(pf, u, y; h=2)
    imm = IMM([kf, kf], [0.5 0.5; 0.5 0.5], [0.5, 0.5])
    @test_throws ArgumentError multistep_sse(imm, u, y; h=2)
    uikf = UIKalmanFilter(SA[0.95 0.1; -0.05 0.9], B, SA[1.0 0.0], zeros(ny, nu), B, R1, R2; nu, ny)
    @test_throws ArgumentError multistep_sse(uikf, u, y; h=2)
end
