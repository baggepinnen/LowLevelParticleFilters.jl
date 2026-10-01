# Regression test for the RTS / MBF smoothers when A is time-varying.
# The forward pass evaluates A at time (t-1)*Ts so that the 3-D-array form
# returns A[:,:,t] (transition from step t to t+1). The smoother must use
# the SAME A_t to invert; a previous bug used time t*Ts (i.e. A[:,:,t+1]),
# which only happened to work when A was constant.

using LowLevelParticleFilters, LinearAlgebra, Random, Test
import LowLevelParticleFilters: SimpleMvNormal

@testset "smoother with 3-D time-varying A" begin
    Random.seed!(123)
    nx, nu, ny = 2, 1, 1
    T = 30

    # A_seq[:,:,t] is the transition from step t to t+1
    A_seq = zeros(nx, nx, T)
    for k = 1:T
        θ = 0.05 * k
        A_seq[:,:,k] = [cos(θ) -sin(θ); sin(θ) cos(θ)] .* 0.97
    end
    B = reshape([0.0, 1.0], nx, nu)
    C = [1.0 0.0]
    R1 = 0.01 * I(nx)
    R2 = 0.1 * I(ny)
    d0 = SimpleMvNormal(zeros(nx), Matrix(1.0I, nx, nx))

    kf = KalmanFilter(A_seq, B, C, 0, R1, R2, d0)

    # Simulate a trajectory using the same time-varying A so the smoother is
    # solving the right problem.
    Random.seed!(1)
    x_true = [zeros(nx) for _ in 1:T+1]
    x_true[1] = rand(d0)
    u_seq = [randn(nu) for _ in 1:T]
    y_seq = Vector{Vector{Float64}}(undef, T)
    for t = 1:T
        w = sqrt(0.01) * randn(nx)
        x_true[t+1] = A_seq[:,:,t]*x_true[t] + B*u_seq[t] + w
        v = sqrt(0.1) * randn(ny)
        y_seq[t] = C*x_true[t] + v
    end

    sol = forward_trajectory(kf, u_seq, y_seq)
    xT, RT, ll = smooth(sol, kf, u_seq, y_seq)

    # Reference smoother: a plain hand-rolled RTS recursion over A_seq.
    xt = sol.xt
    Rt = sol.Rt
    x_pred = sol.x
    R_pred = sol.R
    xT_ref = similar(xt)
    RT_ref = similar(Rt)
    xT_ref[end] = copy(xt[end])
    RT_ref[end] = copy(Rt[end])
    for t = T-1:-1:1
        Ck = Rt[t] * A_seq[:,:,t]' / cholesky(Symmetric(R_pred[t+1]))
        xT_ref[t] = xt[t] + Ck*(xT_ref[t+1] - x_pred[t+1])
        RT_ref[t] = Rt[t] + Ck*(RT_ref[t+1] - R_pred[t+1])*Ck'
        RT_ref[t] = 0.5*(RT_ref[t] + RT_ref[t]')
    end

    for t = 1:T
        @test xT[t] ≈ xT_ref[t] rtol=1e-8
        @test RT[t] ≈ RT_ref[t] rtol=1e-8
    end

    # MBF smoother should produce the same result up to numerical noise.
    ssol_mbf, _, _, _, _ = LowLevelParticleFilters.smooth_mbf(sol, kf, u_seq, y_seq)
    for t = 1:T
        @test ssol_mbf.xT[t] ≈ xT_ref[t] rtol=1e-6
        @test ssol_mbf.RT[t] ≈ RT_ref[t] rtol=1e-6
    end
end

# The EKF smoothers must linearize the dynamics at the same point as the
# forward pass (xt[t], u[t], (t-1)*Ts). With dynamics that are linear but
# time-varying, the EKF smoothers must coincide with the KF smoother.
@testset "EKF smoother with time-varying dynamics" begin
    Random.seed!(123)
    nx, nu, ny = 2, 1, 1
    T = 30
    Ts = 1.0
    A_seq = zeros(nx, nx, T)
    for k = 1:T
        θ = 0.05 * k
        A_seq[:,:,k] = [cos(θ) -sin(θ); sin(θ) cos(θ)] .* 0.97
    end
    B = reshape([0.0, 1.0], nx, nu)
    C = [1.0 0.0]
    R1 = 0.01 * Matrix(I(nx))
    R2 = 0.1 * Matrix(I(ny))
    d0 = SimpleMvNormal(zeros(nx), Matrix(1.0I, nx, nx))

    dynamics(x, u, p, t) = A_seq[:,:,round(Int, t/Ts)+1]*x + B*u
    measurement(x, u, p, t) = C*x

    kf = KalmanFilter(A_seq, B, C, 0, R1, R2, d0; Ts)
    ekf = ExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; nu, Ts)
    sqekf = SqExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; nu, Ts)

    u_seq = [randn(nu) for _ in 1:T]
    y_seq = [randn(ny) for _ in 1:T]

    solkf = forward_trajectory(kf, u_seq, y_seq)
    ssolkf = smooth(solkf, kf, u_seq, y_seq)
    for f in (ekf, sqekf)
        sol = forward_trajectory(f, u_seq, y_seq)
        @test sol.xt ≈ solkf.xt
        ssol = smooth(sol, f, u_seq, y_seq)
        for t = 1:T
            @test ssol.xT[t] ≈ ssolkf.xT[t] rtol=1e-6
            @test ssol.RT[t] ≈ ssolkf.RT[t] rtol=1e-6
        end
    end
end
