
using Test
using LinearAlgebra
using LowLevelParticleFilters
using Plots
using StaticArrays

# --- Recreate the RBPF tutorial system exactly ---
nxn, nxl, ny, nu = 1, 3, 2, 0
# Dynamics now returns [dn; dl] where dl=0 for parameter-only systems
fn(xn, u, p, t) = atan.(xn)
fn_mukf(xn, u, p, t) = [atan(xn[1]); SA[0.0, 0, 0]]
g_mukf(xn, u, p, t)  = [0.1 * xn[]^2 * sign(xn[]), 0.0]
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

# Unified initial distribution for MUKF
x0 = [x0n; x0l]
R0 = [R0n zeros(nxn, nxl); zeros(nxl, nxn) R0l]
d0 = LowLevelParticleFilters.SimpleMvNormal(x0, R0)

# Use package RBPF to generate consistent data
kf_lin = KalmanFilter(Al, zeros(nxl,nu), Cl, 0, R1l_mat, R2_mat, d0l; ny, nu)
mm     = RBMeasurementModel(g_mukf, R2_mat, ny)

names = SignalNames(; x=["xnl", "xl1", "xl2", "xl3"], u=[], y=["y1", "y2"], name="RBPF_tutorial")
rbpf   = RBPF(300, kf_lin, fn, mm, R1n_mat, d0n; nu, An=An_mat, Ts=1.0, names)

T = 150
u_data = [zeros(nu) for _ in 1:T]
x_true, _, y_meas = simulate(rbpf, u_data)

# --- Run MUKF ---
# Create full R1 matrix from blocks
R1_full = [R1n_mat zeros(nxn, nxl); zeros(nxl, nxn) R1l_mat]
mukf = MUKF(dynamics=fn_mukf, nl_measurement_model=mm, An=An_mat, Al=Al, Cl=Cl, R1=R1_full, d0=d0, nxn=nxn, nu=nu, ny=ny)
display(mukf)
sol = forward_trajectory(mukf, u_data, y_meas)

# plot(sol)  # TEMP: commented out while debugging
# Helpers
xn_true = [x_true[t][1] for t in 1:T]
xn_est  = [x[1] for x in sol.xt]  # First element is xn (nonlinear state)
rmse(v1, v2) = sqrt(sum((v1 .- v2).^2) / length(v1))

@testset "MUKF sanity" begin
    @test length(sol.xt) == T
    cov_mat = LowLevelParticleFilters.covariance(mukf)
    @test isposdef(cov_mat) # symmetric positive semi-def check (approx)
    @test rmse(xn_true, xn_est) < 3.0

    # Test that simulate works with MUKF
    x_sim, u_sim, y_sim = simulate(mukf, u_data[1:10])
    @test length(x_sim) == 10
    @test length(y_sim) == 10
end

@testset "MUKF with StaticArrays" begin
    # Create a smaller system with static arrays
    nxn_s, nxl_s, ny_s, nu_s = 1, 2, 2, 0
    # Dynamics now returns [dn; dl] where dl=0 for parameter-only systems
    fn_s(xn, u, p, t) = SVector{nxn_s + nxl_s}([atan.(xn)..., zeros(nxl_s)...])
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

    # Unified initial distribution for MUKF
    x0_s = SVector{nxn_s + nxl_s}([x0n_s..., x0l_s...])
    R0_s = [[R0n_s zeros(SMatrix{nxn_s,nxl_s})]; [zeros(SMatrix{nxl_s,nxn_s}) R0l_s]]
    d0_s = LowLevelParticleFilters.SimpleMvNormal(x0_s, R0_s)

    R1_s = [[R1n_s zeros(SMatrix{nxn_s,nxl_s})]; [zeros(SMatrix{nxl_s,nxn_s}) R1l_s]]
    mm_s = RBMeasurementModel(g_s, R2_s, ny_s)
    mukf_s = MUKF(dynamics=fn_s, nl_measurement_model=mm_s, An=An_s, Al=Al_s, Cl=Cl_s, R1=R1_s, d0=d0_s, nxn=nxn_s, nu=nu_s, ny=ny_s)

    # Verify types
    @test mukf_s.x isa SVector
    @test mukf_s.R isa SMatrix
    @test mukf_s.xn isa SVector  # Property accessor
    @test mukf_s.xl isa SVector  # Property accessor
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
    @test rmse(xn_true_s, xn_est_s) < 3.0
end

@testset "MUKF with in-place dynamics" begin
    # Create system with regular arrays for in-place operations
    nxn_ip, nxl_ip, ny_ip, nu_ip = 1, 2, 2, 0

    # Out-of-place dynamics returns [dn; dl]
    fn_oop(xn, u, p, t) = [atan.(xn); zeros(nxl_ip)]

    # In-place dynamics returns [dn; dl]
    function fn_ip(xp, xn, u, p, t)
        xp[1:nxn_ip] .= atan.(xn)
        xp[nxn_ip+1:end] .= 0
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

    # Unified initial distribution for MUKF
    x0_ip = [x0n_ip; x0l_ip]
    R0_ip = [R0n_ip zeros(nxn_ip, nxl_ip); zeros(nxl_ip, nxn_ip) R0l_ip]
    d0_ip = LowLevelParticleFilters.SimpleMvNormal(x0_ip, R0_ip)

    # Create KF and measurement model
    kf_ip = KalmanFilter(Al_ip, zeros(nxl_ip,nu_ip), Cl_ip, 0, R1l_ip, R2_ip, d0l_ip; ny=ny_ip, nu=nu_ip)
    mm_ip = RBMeasurementModel(g_ip, R2_ip, ny_ip)

    # Create full R1 matrix from blocks
    R1_ip_full = [R1n_ip zeros(nxn_ip, nxl_ip); zeros(nxl_ip, nxn_ip) R1l_ip]

    # Create out-of-place MUKF for reference
    mukf_oop = MUKF{false,false}(dynamics=fn_oop, nl_measurement_model=mm_ip, An=An_ip, Al=Al_ip, Cl=Cl_ip, R1=R1_ip_full, d0=d0_ip, nxn=nxn_ip, nu=nu_ip, ny=ny_ip)

    # Create in-place MUKF
    mukf_ip = MUKF{true,false}(dynamics=fn_ip, nl_measurement_model=mm_ip, An=An_ip, Al=Al_ip, Cl=Cl_ip, R1=R1_ip_full, d0=d0_ip, nxn=nxn_ip, nu=nu_ip, ny=ny_ip)

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
    @test sol_ip.xt ≈ sol_oop.xt rtol=1e-5

    # Compare log-likelihoods
    @test sol_ip.ll ≈ sol_oop.ll rtol=1e-6

    @test_nowarn simulate(mukf_ip, u_ip[1:10])
end

@testset "MUKF with in-place measurement" begin
    # Test that IPM=true works correctly
    nxn_ipm, nxl_ipm, ny_ipm, nu_ipm = 1, 2, 2, 0

    # Out-of-place versions for comparison - returns [dn; dl]
    fn_oop(xn, u, p, t) = [atan.(xn); zeros(nxl_ipm)]
    g_oop(xn, u, p, t) = [0.1 * xn[]^2 * sign(xn[]), 0.0]

    # In-place measurement function
    function g_ip(y, xn, u, p, t)
        y[1] = 0.1 * xn[]^2 * sign(xn[])
        y[2] = 0.0
        y
    end

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

    # Unified initial distribution for MUKF
    x0_ipm = [x0n_ipm; x0l_ipm]
    R0_ipm = [R0n_ipm zeros(nxn_ipm, nxl_ipm); zeros(nxl_ipm, nxn_ipm) R0l_ipm]
    d0_ipm = LowLevelParticleFilters.SimpleMvNormal(x0_ipm, R0_ipm)

    # Create full R1 matrix from blocks
    R1_ipm_full = [R1n_ipm zeros(nxn_ipm, nxl_ipm); zeros(nxl_ipm, nxn_ipm) R1l_ipm]

    # Create out-of-place MUKF for reference
    kf_oop = KalmanFilter(Al_ipm, zeros(nxl_ipm,nu_ipm), Cl_ipm, 0, R1l_ipm, R2_ipm, d0l_ipm; ny=ny_ipm, nu=nu_ipm)
    mm_oop = RBMeasurementModel(g_oop, R2_ipm, ny_ipm)
    mukf_oop = MUKF{false,false}(dynamics=fn_oop, nl_measurement_model=mm_oop, An=An_ipm, Al=Al_ipm, Cl=Cl_ipm, R1=R1_ipm_full, d0=d0_ipm, nxn=nxn_ipm, nu=nu_ipm, ny=ny_ipm)

    # Create in-place measurement MUKF
    kf_ipm = KalmanFilter(Al_ipm, zeros(nxl_ipm,nu_ipm), Cl_ipm, 0, R1l_ipm, R2_ipm, d0l_ipm; ny=ny_ipm, nu=nu_ipm)
    mm_ipm = RBMeasurementModel{true}(g_ip, R2_ipm, ny_ipm)
    mukf_ipm = MUKF{false,true}(dynamics=fn_oop, nl_measurement_model=mm_ipm, An=An_ipm, Al=Al_ipm, Cl=Cl_ipm, R1=R1_ipm_full, d0=d0_ipm, nxn=nxn_ipm, nu=nu_ipm, ny=ny_ipm)

    # Generate data using out-of-place version
    T_ipm = 50
    u_ipm = [zeros(nu_ipm) for _ in 1:T_ipm]
    x_ipm, _, y_ipm = simulate(mukf_oop, u_ipm)

    # Filter with both versions
    sol_oop = forward_trajectory(mukf_oop, u_ipm, y_ipm)
    sol_ipm = forward_trajectory(mukf_ipm, u_ipm, y_ipm)

    # Verify they produce the same results
    @test length(sol_ipm.xt) == T_ipm
    @test all(length(x) == nxn_ipm + nxl_ipm for x in sol_ipm.xt)

    # Compare state estimates (should be very close)
    @test sol_ipm.xt ≈ sol_oop.xt rtol=1e-6

    # Compare log-likelihoods
    @test sol_ipm.ll ≈ sol_oop.ll rtol=1e-6

    # Verify covariance is positive definite
    cov_ipm = LowLevelParticleFilters.covariance(mukf_ipm)
    @test isposdef(cov_ipm + cov_ipm')
end

@testset "MUKF vs KF on linear system" begin
    # Test MUKF on a purely linear system against the optimal Kalman filter
    # Even though the system is linear, we place part in the "nonlinear" part
    # The MUKF should match the KF solution exactly (up to numerical tolerance)

    # System dimensions
    nxn = 1  # Nonlinear state (but actually linear)
    nxl = 1  # Linear state
    ny = 2
    nu = 0

    # Linear system:
    # x1' = 0.9*x1 + 0.2*x2 + w1
    # x2' = 0.95*x2 + w2
    # y1 = 1.0*x1 + v1
    # y2 = 0.5*x2 + v2

    # MUKF parameters - dynamics returns [dn; dl]
    fn(xn, u, p, t) = SA[0.9 * xn[]; 0.0]  # [dn; dl] where dn is linear, dl=0
    An = SA[0.2;;]                  # Coupling from linear to nonlinear (1x1 matrix)
    Al = SA[0.95;;]                 # Linear dynamics

    g(xn, u, p, t) = SA[xn[]; 0]      # Measurement of nonlinear state
    Cl = SA[0.0; 0.5;;]               # Measurement of linear state (2x1)

    R1n = SA[0.01;;]
    R1l = SA[0.01;;]
    R2 = SA[0.1 0.0; 0.0 0.1]

    x0n = SA[0.5]
    R0n = SA[0.1;;]
    x0l = SA[0.3]
    R0l = SA[0.1;;]

    d0n = LowLevelParticleFilters.SimpleMvNormal(x0n, R0n)
    d0l = LowLevelParticleFilters.SimpleMvNormal(x0l, R0l)

    # Unified initial distribution for MUKF
    x0 = SA[x0n[]; x0l[]]
    R0 = SA[R0n[1] 0.0;
           0.0     R0l[1]]
    d0 = LowLevelParticleFilters.SimpleMvNormal(x0, R0)

    # Create full R1 matrix from blocks
    R1_full = SA[R1n[1]  0.0;
               0.0    R1l[1]]

    # Create MUKF
    kf_mukf = KalmanFilter(Al, zeros(nxl, nu), Cl, 0, R1l, R2, d0l; ny=ny, nu=nu)
    mm = RBMeasurementModel(g, R2, ny)
    mukf = MUKF(; dynamics=fn, nl_measurement_model=mm, An, Al, Cl, R1=R1_full, d0, nxn, nu, ny)

    # Create equivalent standard Kalman filter for full system
    A_full = SA[0.9  0.2;
              0.0  0.95]
    C_full = SA[1.0  0.0;
              0.0  0.5]
    x0_full = SA[x0n[]; x0l[]]
    R0_full = SA[R0n[1] 0.0;
               0.0        R0l[1]]

    d0_full = LowLevelParticleFilters.SimpleMvNormal(x0_full, R0_full)
    kf_full = KalmanFilter(A_full, zeros(2, nu), C_full, 0, R1_full, R2, d0_full; ny=ny, nu=nu)

    @test kf_full.x ≈ mukf.x atol=1e-8
    @test kf_full.R ≈ LowLevelParticleFilters.covariance(mukf) atol=1e-8

    # Generate data from the full KF
    T = 100
    u = [zeros(nu) for _ in 1:T]
    x_true, _, y = simulate(kf_full, u)
    y = SVector.(y)
    u = SVector{0}.(u)

    predict!(kf_full, u[1], y[1])
    predict!(mukf, u[1], y[1])

    @test kf_full.x ≈ mukf.x atol=1e-8
    @test kf_full.R ≈ LowLevelParticleFilters.covariance(mukf) atol=1e-8
    
    llkf, ekf = correct!(kf_full, u[1], y[1])
    llmukf, emukf = correct!(mukf, u[1], y[1])
    
    @test kf_full.x ≈ mukf.x atol=1e-8
    @test kf_full.R ≈ LowLevelParticleFilters.covariance(mukf) atol=1e-8
    @test ekf ≈ emukf atol=1e-8

    # Run both filters
    sol_mukf = forward_trajectory(mukf, u, y)
    a = @allocations forward_trajectory(mukf, u, y)
    @test a < 22 * 1.1
    sol_kf = forward_trajectory(kf_full, u, y)

    # Extract states - MUKF stores [xn; xl], KF stores [x1; x2]
    # They should be identical
    @test length(sol_mukf.xt) == T
    @test length(sol_kf.xt) == T

    # Compare state estimates at each time step
    @test sol_mukf.xt ≈ sol_kf.xt atol=1e-6

    # Compare log-likelihoods
    @test sol_mukf.ll ≈ sol_kf.ll atol=1e-6

    # Compare covariances
    cov_mukf = LowLevelParticleFilters.covariance(mukf)
    cov_kf = kf_full.R
    @test cov_mukf ≈ cov_kf atol=1e-6

    # Verify filtering error against true state
    rmse_mukf = sqrt(sum(sum((sol_mukf.xt[t] .- x_true[t]).^2) for t in 1:T) / T)
    rmse_kf = sqrt(sum(sum((sol_kf.xt[t] .- x_true[t]).^2) for t in 1:T) / T)
    @test rmse_mukf ≈ rmse_kf atol=1e-6

    println("MUKF vs KF comparison on linear system:")
    println("  State estimate difference: ", maximum(maximum(abs.(sol_mukf.xt[t] .- sol_kf.xt[t])) for t in 1:T))
    println("  Log-likelihood difference: ", abs(sol_mukf.ll - sol_kf.ll))
    println("  Covariance difference: ", maximum(abs.(cov_mukf .- cov_kf)))
    println("  RMSE MUKF: ", rmse_mukf)
    println("  RMSE KF: ", rmse_kf)
end

