using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using LinearAlgebra
using StaticArrays
using Test
using LeastSquaresOptim

@testset "autotune_covariances" begin

    # Common setup for all tests
    nx = 2   # Dimension of state
    nu = 2   # Dimension of input
    ny = 2   # Dimension of measurements
    T = 200  # Number of time steps

    # True noise distributions
    R1_true = I(nx)
    R2_true = I(ny)
    d0 = SimpleMvNormal(randn(nx), I(nx))

    # System matrices
    A = SA[1.0 0.1; 0.0 1.0]
    B = SA[0.0 0.1; 1.0 0.1]
    C = SA[1.0 0.0; 0.0 1.0]

    # Dynamics and measurement functions for nonlinear filters
    dynamics(x, u, p, t) = A * x .+ B * u
    measurement(x, u, p, t) = C * x

    # Simulate the true system
    u = [SVector{nu}(randn(nu)) for _ in 1:T]
    xs, _, y = let
        pf_sim = ExtendedKalmanFilter(dynamics, measurement, R1_true, R2_true, d0; nu, ny)
        LowLevelParticleFilters.simulate(pf_sim, T, SimpleMvNormal(zeros(nu), I(nu)))
    end

    @testset "KalmanFilter - diagonal parametrization" begin
        # Create filter with suboptimal covariances
        R1_initial = 0.5^2 * I(nx)
        R2_initial = 2.0^2 * I(ny)

        kf = KalmanFilter(
            A, B, C, 0,
            SMatrix{nx,nx}(R1_initial),
            SMatrix{ny,ny}(R2_initial),
            d0
        )

        sol_initial = forward_trajectory(kf, u, y)

        # Optimize with diagonal parametrization
        result = autotune_covariances(
            sol_initial;
            diagonal = true,
            optimize_x0 = false,
            show_trace = false,
            iterations = 30
        )

        @test result.sol_opt.ll > sol_initial.ll  # Log-likelihood should improve
        @test size(result.R1) == (nx, nx)
        @test size(result.R2) == (ny, ny)
        @test result.filter isa KalmanFilter
    end

    @testset "KalmanFilter - full parametrization" begin
        # Create filter with suboptimal covariances
        R1_initial = 0.5^2 * I(nx)
        R2_initial = 2.0^2 * I(ny)

        kf = KalmanFilter(
            A, B, C, 0,
            SMatrix{nx,nx}(R1_initial),
            SMatrix{ny,ny}(R2_initial),
            d0
        )

        sol_initial = forward_trajectory(kf, u, y)

        # Optimize with full parametrization
        result = autotune_covariances(
            sol_initial;
            diagonal = false,
            optimize_x0 = false,
            show_trace = false,
            iterations = 30
        )

        @test result.sol_opt.ll > sol_initial.ll  # Log-likelihood should improve
        @test size(result.R1) == (nx, nx)
        @test size(result.R2) == (ny, ny)
        # Check positive definiteness
        @test isposdef(result.R1)
        @test isposdef(result.R2)
    end

    @testset "KalmanFilter - optimize_x0=true" begin
        # Create filter with suboptimal covariances and wrong initial state
        R1_initial = 0.5^2 * I(nx)
        R2_initial = 2.0^2 * I(ny)
        d0_wrong = SimpleMvNormal(randn(nx) .+ 5.0, I(nx))  # Far from true initial state

        kf = KalmanFilter(
            A, B, C, 0,
            SMatrix{nx,nx}(R1_initial),
            SMatrix{ny,ny}(R2_initial),
            d0_wrong
        )

        sol_initial = forward_trajectory(kf, u, y)

        # Optimize with x0
        result = autotune_covariances(
            sol_initial;
            diagonal = true,
            optimize_x0 = true,
            show_trace = false,
            iterations = 30
        )

        @test result.sol_opt.ll > sol_initial.ll  # Log-likelihood should improve
        @test length(result.x0) == nx
        # Optimized x0 should be closer to true initial state than the wrong initial guess
        @test norm(result.x0 - xs[1]) < norm(d0_wrong.μ - xs[1])
    end

    @testset "ExtendedKalmanFilter - diagonal parametrization" begin
        # Create filter with suboptimal covariances
        R1_initial = 0.5^2 * I(nx)
        R2_initial = 2.0^2 * I(ny)

        ekf = ExtendedKalmanFilter(
            dynamics, measurement,
            SMatrix{nx,nx}(R1_initial),
            SMatrix{ny,ny}(R2_initial),
            d0;
            nu = nu
        )

        sol_initial = forward_trajectory(ekf, u, y)

        # Optimize
        result = autotune_covariances(
            sol_initial;
            diagonal = true,
            optimize_x0 = false,
            show_trace = false,
            iterations = 30
        )

        @test result.sol_opt.ll > sol_initial.ll
        @test result.filter isa ExtendedKalmanFilter
    end

    @testset "UnscentedKalmanFilter - diagonal parametrization" begin
        # Create filter with suboptimal covariances
        R1_initial = 0.5^2 * I(nx)
        R2_initial = 2.0^2 * I(ny)

        ukf = UnscentedKalmanFilter(
            dynamics, measurement,
            SMatrix{nx,nx}(R1_initial),
            SMatrix{ny,ny}(R2_initial),
            d0;
            nu = nu,
            ny = ny
        )

        sol_initial = forward_trajectory(ukf, u, y)

        # Optimize
        result = autotune_covariances(
            sol_initial;
            diagonal = true,
            optimize_x0 = false,
            show_trace = false,
            iterations = 30
        )

        @test result.sol_opt.ll > sol_initial.ll
        @test result.filter isa UnscentedKalmanFilter
    end

    @testset "UnscentedKalmanFilter - augmented dynamics (AUGD=true)" begin
        # Test UKF with augmented dynamics noise
        R1_initial = 0.5^2 * I(nx)
        R2_initial = 2.0^2 * I(ny)

        # Dynamics with explicit noise input
        dynamics_w(x, u, p, t, w) = A * x .+ B * u .+ w

        ukf = UnscentedKalmanFilter{false,false,true,false}(
            dynamics_w, measurement,
            SMatrix{nx,nx}(R1_initial),
            SMatrix{ny,ny}(R2_initial),
            d0;
            nu = nu,
            ny = ny
        )

        sol_initial = forward_trajectory(ukf, u, y)

        # Optimize
        result = autotune_covariances(
            sol_initial;
            diagonal = true,
            optimize_x0 = false,
            show_trace = false,
            iterations = 30
        )

        @test result.sol_opt.ll > sol_initial.ll
        @test result.filter isa UnscentedKalmanFilter
    end

    @testset "UnscentedKalmanFilter - augmented measurement (AUGM=true)" begin
        # Test UKF with augmented measurement noise
        R1_initial = 0.5^2 * I(nx)
        R2_initial = 2.0^2 * I(ny)

        # Measurement with explicit noise input
        measurement_v(x, u, p, t, v) = C * x .+ v

        ukf = UnscentedKalmanFilter{false,false,false,true}(
            dynamics, measurement_v,
            SMatrix{nx,nx}(R1_initial),
            SMatrix{ny,ny}(R2_initial),
            d0;
            nu = nu,
            ny = ny
        )

        sol_initial = forward_trajectory(ukf, u, y)

        # Optimize
        result = autotune_covariances(
            sol_initial;
            diagonal = true,
            optimize_x0 = false,
            show_trace = false,
            iterations = 30
        )

        @test result.sol_opt.ll > sol_initial.ll
        @test result.filter isa UnscentedKalmanFilter
    end

    @testset "Comparison: diagonal vs optimize_x0 vs full" begin
        # Create filter with suboptimal covariances
        R1_initial = 0.5^2 * I(nx)
        R2_initial = 2.0^2 * I(ny)

        kf = KalmanFilter(
            A, B, C, 0,
            SMatrix{nx,nx}(R1_initial),
            SMatrix{ny,ny}(R2_initial),
            d0
        )

        sol_initial = forward_trajectory(kf, u, y)

        # Optimize with different methods
        result_diag = autotune_covariances(
            sol_initial;
            diagonal = true,
            optimize_x0 = false,
            show_trace = false,
            iterations = 30
        )

        result_diag_x0 = autotune_covariances(
            sol_initial;
            diagonal = true,
            optimize_x0 = true,
            show_trace = false,
            iterations = 30
        )

        result_full = autotune_covariances(
            sol_initial;
            diagonal = false,
            optimize_x0 = false,
            show_trace = false,
            iterations = 30
        )

        # All should improve over initial
        @test result_diag.sol_opt.ll > sol_initial.ll
        @test result_diag_x0.sol_opt.ll > sol_initial.ll
        @test result_full.sol_opt.ll > sol_initial.ll

        # Optimizing x0 should give at least as good or better results
        @test result_diag_x0.sol_opt.ll >= result_diag.sol_opt.ll - 1e-6  # Allow small numerical difference
    end
end
