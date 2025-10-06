
using Test
using LinearAlgebra
using LowLevelParticleFilters
using Plots
using StaticArrays

# --- Recreate the RBPF tutorial system exactly ---
nxn, nxl, ny, nu = 1, 3, 2, 0
fn(xn, u, p, t) = atan.(xn)
g(xn, u, p, t)  = [0.1 * xn[]^2 * sign(xn[]), 0.0]
An_mat = [1.0 0.0 0.0]
Al = [ 1.0  0.3   0.0;
       0.0  0.92 -0.3;
       0.0  0.3   0.92 ]
Cl = [0.0 0.0 0.0;
      1.0 -1.0 1.0]
R1n_mat = [0.01;;]
R1l_mat = 0.01I(nxl)
R2_mat  = 0.1I(ny)
x0n = zeros(nxn); R0n = [1.0;;]
x0l = zeros(nxl); R0l = 0.01I(nxl)

d0n = LowLevelParticleFilters.SimpleMvNormal(x0n, R0n)
d0l = LowLevelParticleFilters.SimpleMvNormal(x0l, R0l)

# Use package RBPF to generate consistent data
kf_lin = KalmanFilter(Al, zeros(nxl,nu), Cl, 0, R1l_mat, R2_mat, d0l; ny, nu)
mm     = RBMeasurementModel(g, R2_mat, ny)

names = SignalNames(; x=["xnl", "xl1", "xl2", "xl3"], u=[], y=["y1", "y2"], name="RBPF_tutorial")
rbpf   = RBPF(300, kf_lin, fn, mm, R1n_mat, d0n; nu, An=An_mat, Ts=1.0, names)

T = 150
u_data = [zeros(nu) for _ in 1:T]
x_true, _, y_meas = simulate(rbpf, u_data)

# --- Run MUKF ---
mukf = MUKF(dynamics=fn, nl_measurement_model=mm, An=An_mat, kf=kf_lin, R1n=R1n_mat, d0n=d0n)
sol = forward_trajectory(mukf, u_data, y_meas)

plot(sol)
# Helpers
xn_true = [x_true[t][1] for t in 1:T]
xn_est  = [x[1] for x in sol.xt]  # First element is xn (nonlinear state)
rmse(v1, v2) = sqrt(sum((v1 .- v2).^2) / length(v1))

@testset "MUKF sanity" begin
    @test length(sol.xt) == T
    cov_mat = LowLevelParticleFilters.covariance(mukf)
    @test isposdef(cov_mat + cov_mat') # symmetric positive semi-def check (approx)
    @test rmse(xn_true, xn_est) < 1.0    # loose upper bound; adjust as needed

    # Test that simulate works with MUKF
    x_sim, u_sim, y_sim = simulate(mukf, u_data[1:10])
    @test length(x_sim) == 10
    @test length(y_sim) == 10
end

@testset "MUKF with StaticArrays" begin
    # Create a smaller system with static arrays
    nxn_s, nxl_s, ny_s, nu_s = 1, 2, 2, 0
    fn_s(xn, u, p, t) = atan.(xn)
    g_s(xn, u, p, t) = SVector{2}(0.1 * xn[]^2 * sign(xn[]), 0.0)

    An_s = @SMatrix [1.0 0.0]
    Al_s = @SMatrix [1.0 0.3; 0.0 0.92]
    Cl_s = @SMatrix [0.0 0.0; 1.0 -1.0]
    R1n_s = @SMatrix [0.01]
    R1l_s = @SMatrix [0.01 0.0; 0.0 0.01]
    R2_s = @SMatrix [0.1 0.0; 0.0 0.1]

    x0n_s = @SVector zeros(nxn_s)
    R0n_s = @SMatrix [1.0]
    x0l_s = @SVector zeros(nxl_s)
    R0l_s = @SMatrix [0.01 0.0; 0.0 0.01]

    d0n_s = LowLevelParticleFilters.SimpleMvNormal(x0n_s, R0n_s)
    d0l_s = LowLevelParticleFilters.SimpleMvNormal(x0l_s, R0l_s)

    kf_s = KalmanFilter(Al_s, zeros(SMatrix{nxl_s,nu_s}), Cl_s, 0, R1l_s, R2_s, d0l_s; ny=ny_s, nu=nu_s)
    mm_s = RBMeasurementModel(g_s, R2_s, ny_s)
    mukf_s = MUKF(dynamics=fn_s, nl_measurement_model=mm_s, An=An_s, kf=kf_s, R1n=R1n_s, d0n=d0n_s)

    # Verify types
    @test mukf_s.xn isa SVector
    @test mukf_s.Rn isa SMatrix
    @test mukf_s.xl[1] isa SVector
    @test mukf_s.Rl[1] isa SMatrix
    @test mukf_s.sigma_point_cache.x0[1] isa SVector

    # Test filtering
    T_s = 50
    u_s = [zeros(nu_s) for _ in 1:T_s]
    x_s, _, y_s = simulate(mukf_s, u_s)
    sol_s = forward_trajectory(mukf_s, u_s, y_s)

    @test length(sol_s.xt) == T_s
    @test all(length(x) == nxn_s + nxl_s for x in sol_s.xt)

    # Verify covariance is positive definite
    cov_s = LowLevelParticleFilters.covariance(mukf_s)
    @test isposdef(cov_s + cov_s')

    # Verify filtering produces reasonable estimates
    xn_true_s = [x_s[t][1] for t in 1:T_s]
    xn_est_s = [x[1] for x in sol_s.xt]
    @test rmse(xn_true_s, xn_est_s) < 1.0
end

@testset "MUKF with in-place dynamics" begin
    # Create system with regular arrays for in-place operations
    nxn_ip, nxl_ip, ny_ip, nu_ip = 1, 2, 2, 0

    # Out-of-place dynamics for comparison
    fn_oop(xn, u, p, t) = atan.(xn)

    # In-place dynamics
    function fn_ip(xp, xn, u, p, t)
        xp .= atan.(xn)
        xp
    end

    g_ip(xn, u, p, t) = [0.1 * xn[]^2 * sign(xn[]), 0.0]

    An_ip = [1.0 0.0]
    Al_ip = [1.0 0.3; 0.0 0.92]
    Cl_ip = [0.0 0.0; 1.0 -1.0]
    R1n_ip = [0.01;;]
    R1l_ip = [0.01 0.0; 0.0 0.01]
    R2_ip = [0.1 0.0; 0.0 0.1]

    x0n_ip = zeros(nxn_ip)
    R0n_ip = [1.0;;]
    x0l_ip = zeros(nxl_ip)
    R0l_ip = [0.01 0.0; 0.0 0.01]

    d0n_ip = LowLevelParticleFilters.SimpleMvNormal(x0n_ip, R0n_ip)
    d0l_ip = LowLevelParticleFilters.SimpleMvNormal(x0l_ip, R0l_ip)

    # Create KF and measurement model
    kf_ip = KalmanFilter(Al_ip, zeros(nxl_ip,nu_ip), Cl_ip, 0, R1l_ip, R2_ip, d0l_ip; ny=ny_ip, nu=nu_ip)
    mm_ip = RBMeasurementModel(g_ip, R2_ip, ny_ip)

    # Create out-of-place MUKF for reference
    mukf_oop = MUKF{false,false}(dynamics=fn_oop, nl_measurement_model=mm_ip, An=An_ip, kf=kf_ip, R1n=R1n_ip, d0n=d0n_ip)

    # Create in-place MUKF
    mukf_ip = MUKF{true,false}(dynamics=fn_ip, nl_measurement_model=mm_ip, An=An_ip, kf=kf_ip, R1n=R1n_ip, d0n=d0n_ip)

    # Generate data using out-of-place version
    T_ip = 50
    u_ip = [zeros(nu_ip) for _ in 1:T_ip]
    x_ip, _, y_ip = simulate(mukf_oop, u_ip)

    # Filter with both versions
    sol_oop = forward_trajectory(mukf_oop, u_ip, y_ip)
    sol_ip = forward_trajectory(mukf_ip, u_ip, y_ip)

    # Verify they produce the same results
    @test length(sol_ip.xt) == T_ip
    @test all(length(x) == nxn_ip + nxl_ip for x in sol_ip.xt)

    # Compare state estimates (should be very close)
    @test sol_ip.xt ≈ sol_oop.xt rtol=1e-6

    # Compare log-likelihoods
    @test sol_ip.ll ≈ sol_oop.ll rtol=1e-6

    @test_nowarn simulate(mukf_ip, u_ip[1:10])
end

@testset "MUKF with in-place measurement (unsupported)" begin
    # Verify that IPM=true throws an error
    nxn_ipm, nxl_ipm, ny_ipm, nu_ipm = 1, 2, 2, 0
    fn_ipm(xn, u, p, t) = atan.(xn)
    g_ipm(xn, u, p, t) = [0.1 * xn[]^2 * sign(xn[]), 0.0]

    An_ipm = [1.0 0.0]
    Al_ipm = [1.0 0.3; 0.0 0.92]
    Cl_ipm = [0.0 0.0; 1.0 -1.0]
    R1n_ipm = [0.01;;]
    R1l_ipm = [0.01 0.0; 0.0 0.01]
    R2_ipm = [0.1 0.0; 0.0 0.1]

    x0n_ipm = zeros(nxn_ipm)
    R0n_ipm = [1.0;;]
    x0l_ipm = zeros(nxl_ipm)
    R0l_ipm = [0.01 0.0; 0.0 0.01]

    d0n_ipm = LowLevelParticleFilters.SimpleMvNormal(x0n_ipm, R0n_ipm)
    d0l_ipm = LowLevelParticleFilters.SimpleMvNormal(x0l_ipm, R0l_ipm)

    kf_ipm = KalmanFilter(Al_ipm, zeros(nxl_ipm,nu_ipm), Cl_ipm, 0, R1l_ipm, R2_ipm, d0l_ipm; ny=ny_ipm, nu=nu_ipm)
    mm_ipm = RBMeasurementModel(g_ipm, R2_ipm, ny_ipm)

    # Create MUKF with IPM=true
    mukf_ipm = MUKF{false,true}(dynamics=fn_ipm, nl_measurement_model=mm_ipm, An=An_ipm, kf=kf_ipm, R1n=R1n_ipm, d0n=d0n_ipm)

    u_ipm = [zeros(nu_ipm) for _ in 1:10]
    y_ipm = [zeros(ny_ipm) for _ in 1:10]

    # Verify that correct! throws an error
    @test_throws ErrorException("Inplace measurement model not yet supported for MUKF") forward_trajectory(mukf_ipm, u_ipm, y_ipm)
end

## `examples/mukf_tutorial.jl`


using Statistics

# Collect arrays for plotting
xn_true = [x_true[t][1] for t in 1:T]
xn_est  = [x[1] for x in sol.xt]  # First element is xn
RMSE = sqrt(mean((xn_true .- xn_est).^2))

t = 1:T
plt = plot(t, xn_true, label="true xⁿ", lw=2)
plot!(plt, t, xn_est, label="MUKF xⁿ est", lw=2, ls=:dash)
xlabel!(plt, "time step")
ylabel!(plt, "xⁿ")
title!(plt, "MUKF on RBPF tutorial system (RMSE=$(round(RMSE, digits=3)))")
