using Test
using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using ControlSystemsBase
using LinearAlgebra
using StaticArrays
using Random
using Statistics



## Setup - reuse pattern from test_ukf.jl
eye(n) = SMatrix{n,n}(1.0I(n))
nx = 2 # Dimension of state
nu = 2 # Dimension of input
ny = 2 # Dimension of measurements

d0 = SimpleMvNormal(randn(nx), 2.0*eye(nx))   # Initial state Distribution
du = SimpleMvNormal(zeros(nu), eye(nu)) # Control input distribution

# Define linear state-space system
_A = SA[0.99 0.1; 0 0.2]
_B = @SMatrix [-0.7400216956683083 1.6097265310456392; -1.4384539113366408 1.7467811974822818]
_C = SMatrix{ny,ny}(eye(ny))

dynamics(x,u,p,t) = _A*x .+ _B*u
measurement(x,u,p,t) = _C*x

R1 = eye(nx)
R2 = eye(ny)

T = 2000 # Number of time steps for convergence

# Create filters
kf = KalmanFilter(_A, _B, _C, 0, R1, R2, d0)
ukf = UnscentedKalmanFilter(dynamics, measurement, R1, R2, d0; ny, nu)

# Augmented dynamics for UKF with noise input
dynamics_w(x,u,p,t,w) = _A*x .+ _B*u .+ w
ukfw = UnscentedKalmanFilter{false,false,true,false}(dynamics_w, measurement, R1, R2, d0; ny, nu)

# Simulate data
x, u, y = LowLevelParticleFilters.simulate(kf, T, du)
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
x, u, y = tosvec.((x, u, y))

@testset "kalman gain convergence" begin
    # Test for KalmanFilter - use last 10 values to check convergence
    sol_kf = forward_trajectory(kf, u, y)
    K_analytical = kalman(kf, x[1], u[1], nothing, 0.0)
    K_converged = sol_kf.K[end]

    # The converged K should be close to the steady-state analytical K
    # Use a reasonable tolerance since convergence may not be perfect
    @test K_analytical ≈ K_converged rtol=1e-3

    # Test for UnscentedKalmanFilter
    sol_ukf = forward_trajectory(ukf, u, y)
    K_analytical_ukf = kalman(ukf, x[1], u[1], nothing, 0.0)
    K_converged_ukf = sol_ukf.K[end]

    @test K_analytical_ukf ≈ K_converged_ukf rtol=1e-3

    # Test for UKF with noise input
    sol_ukfw = forward_trajectory(ukfw, u, y)
    K_analytical_ukfw = kalman(ukfw, x[1], u[1], nothing, 0.0)
    K_converged_ukfw = sol_ukfw.K[end]
    @test K_analytical_ukfw ≈ K_converged_ukfw rtol=1e-3

end

@testset "covar steady-state" begin
    # Test for KalmanFilter
    # The covar function computes stationary covariance without correction steps
    # So we need to call predict! in a loop until convergence
    kf_test = deepcopy(kf)
    for i in 1:T, j=1:3
        predict!(kf_test, u[i], nothing, 0.0)
    end
    P_analytical = covar(kf, x[1], u[1], nothing, 0.0)
    R_converged = kf_test.R

    @test P_analytical ≈ R_converged rtol=1e-3

    # Test for UnscentedKalmanFilter
    ukf_test = deepcopy(ukf)
    for i in 1:T, j=1:3
        predict!(ukf_test, u[i], nothing, 0.0)
    end
    P_analytical_ukf = covar(ukf, x[1], u[1], nothing, 0.0)
    R_converged_ukf = ukf_test.R

    @test P_analytical_ukf ≈ R_converged_ukf rtol=1e-3

    # Test for UKF with noise input
    ukfw_test = deepcopy(ukfw)
    for i in 1:T, j=1:3
        predict!(ukfw_test, u[i], nothing, 0.0)
    end
    P_analytical_ukfw = covar(ukfw, x[1], u[1], nothing, 0.0)
    R_converged_ukfw = ukfw_test.R

    @test P_analytical_ukfw ≈ R_converged_ukfw rtol=1e-3

    # KF and UKF should give similar covariances for linear systems
    @test P_analytical ≈ P_analytical_ukf rtol=0.01
end


@testset "controllability" begin
    # Test for UKF with noise input - fully controllable system
    ctrl = controllability(ukfw, x[1], u[1], nothing, 0.0)
    @test ctrl.iscontrollable

    # Create a system with uncontrollable mode from noise input
    # Use a system where noise only affects one state
    Bw_partial = @SMatrix [1.0; 0.0]
    dynamics_w_partial(x,u,p,t,w) = _A*x .+ _B*u .+ Bw_partial*w
    ukfw_partial = UnscentedKalmanFilter{false,false,true,false}(
        dynamics_w_partial, measurement, [1.0;;], R2, d0; ny, nu
    )

    # This system may have reduced controllability from noise input
    ctrl_partial = controllability(ukfw_partial, x[1], u[1], nothing, 0.0)

    # The controllability analysis should execute without error
    @test !ctrl_partial.iscontrollable
end

@testset "consistency with existing linearize tests" begin
    # The kalman and covar functions use linearize internally
    # Verify consistency with existing linearization tests from test_ukf.jl

    Al, Bl, Cl, Dl = ControlSystemsBase.linearize(ukf, x[1], u[1], nothing)
    @test Al ≈ _A
    @test Bl ≈ _B
    @test Cl ≈ _C
    @test iszero(Dl)

    Al_kf, Bl_kf, Cl_kf, Dl_kf = ControlSystemsBase.linearize(kf, x[1], u[1], nothing)
    @test Al_kf ≈ _A
    @test Bl_kf ≈ _B
    @test Cl_kf ≈ _C
    @test iszero(Dl_kf)

    # The linearization should be used consistently in kalman/covar functions
    G_sys = ss(Al, Bl, Cl, Dl, kf.Ts)
    K_from_sys = ControlSystemsBase.kalman(G_sys, R1, R2, direct=true)
    K_from_filter = kalman(kf, x[1], u[1], nothing, 0.0)
    @test K_from_sys ≈ K_from_filter
end

