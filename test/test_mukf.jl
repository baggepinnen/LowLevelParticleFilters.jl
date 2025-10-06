
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
