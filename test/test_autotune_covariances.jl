using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using LinearAlgebra
using StaticArrays
using Test
using LeastSquaresOptim
using Random

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

    @testset "MAP estimation with Inverse-Wishart prior" begin
        # Setup
        R1_initial = 0.5^2 * I(nx)
        R2_initial = 2.0^2 * I(ny)

        kf = KalmanFilter(
            A, B, C, 0,
            SMatrix{nx,nx}(R1_initial),
            SMatrix{ny,ny}(R2_initial),
            d0
        )

        sol_initial = forward_trajectory(kf, u, y)

        # MLE for comparison
        result_mle = autotune_covariances(
            sol_initial;
            diagonal = true,
            optimize_x0 = false,
            show_trace = false,
            iterations = 30
        )

        @testset "Weak prior on R1 only" begin
            v1 = nx + 2  # Weak prior

            result = autotune_covariances(
                sol_initial;
                diagonal = true,
                optimize_x0 = false,
                show_trace = false,
                iterations = 30,
                v_R1 = v1
            )

            @test result.sol_opt.ll > sol_initial.ll  # Should improve over initial
            # With weak prior, result should be close to MLE
            @test norm(diag(result.R1) - diag(result_mle.R1)) < 0.25*norm(diag(result_mle.R1))
        end

        @testset "Strong prior on R1" begin
            v1_strong = nx + 20  # Strong prior

            result = autotune_covariances(
                sol_initial;
                diagonal = true,
                optimize_x0 = false,
                show_trace = false,
                iterations = 30,
                v_R1 = v1_strong
            )

            @test result.sol_opt.ll > sol_initial.ll  # Should improve over initial
            # With strong prior, R1 should be closer to prior mean (R1_initial) than MLE is
            @test norm(diag(result.R1) - diag(R1_initial)) < norm(diag(result_mle.R1) - diag(R1_initial))
        end

        @testset "Prior on both R1 and R2" begin
            v1 = nx + 3
            v2 = ny + 3

            result = autotune_covariances(
                sol_initial;
                diagonal = true,
                optimize_x0 = false,
                show_trace = false,
                iterations = 30,
                v_R1 = v1,
                v_R2 = v2
            )

            @test result.sol_opt.ll > sol_initial.ll
            @test result.filter isa KalmanFilter
        end

        @testset "Prior validation" begin
            # Test v too small
            @test_throws ArgumentError autotune_covariances(
                sol_initial;
                v_R1 = nx - 1  # Too small
            )

            @test_throws ArgumentError autotune_covariances(
                sol_initial;
                v_R2 = ny - 1  # Too small
            )
        end

    end
end

using Distributions: InverseWishart, logpdf
import ForwardDiff
const LSOptExt = Base.get_extension(LowLevelParticleFilters, :LowLevelParticleFiltersLSOptExt)

@testset "autotune_covariances internals" begin
    Random.seed!(2)
    nx, nu, ny, T = 2, 1, 2, 150
    A = SA[0.95 0.1; 0.0 0.9]
    B = SA[0.0; 1.0;;]
    C = SA[1.0 0.0; 0.0 1.0]
    R1_true = SA[0.02 0.005; 0.005 0.05]
    R2_true = SA[0.1 0.02; 0.02 0.3]
    d0 = SimpleMvNormal(SA[0.5, -0.5], SMatrix{2,2}(1.0I(2)))
    kf_true = KalmanFilter(A, B, C, 0, R1_true, R2_true, d0)
    u = [SA[randn()] for _ in 1:T]
    x, _, y = simulate(kf_true, u)
    R1_initial = SMatrix{2,2}(0.1I(2))
    R2_initial = SMatrix{2,2}(1.0I(2))
    kf = KalmanFilter(A, B, C, 0, R1_initial, R2_initial, d0)
    sol_initial = forward_trajectory(kf, u, y)

    @testset "Inverse-Wishart residuals" begin
        n, v = 2, 6.0
        Ψ = [2.0 0.3; 0.3 1.0]
        Lmode = cholesky(Symmetric(Ψ/(v + n + 1))).L
        r(Σ) = LSOptExt.inverse_wishart_residuals!(zeros(eltype(Σ), 3), Σ, v, Lmode)
        Ra = [0.05 0.01; 0.01 0.02]
        Rb = [1.5 -0.2; -0.2 0.8]
        @test logdet(Ra) < 0 # The previous implementation had the wrong sign of the logdet term in this case
        dIW = InverseWishart(v, Ψ)
        @test sum(abs2, r(Ra)) - sum(abs2, r(Rb)) ≈ -(logpdf(dIW, Ra) - logpdf(dIW, Rb))
        Rc = [0.4 0.1; 0.1 0.2]
        @test sum(abs2, r(Ra)) - sum(abs2, r(Rc)) ≈ -(logpdf(dIW, Ra) - logpdf(dIW, Rc))
        # The residuals vanish at the mode, where their derivative remains finite
        θmode = LowLevelParticleFilters.logchol_from_cov(Ψ/(v + n + 1))
        @test sum(abs2, r(LowLevelParticleFilters.cov_from_logchol(θmode))) < 1e-10
        J = ForwardDiff.jacobian(θ -> r(LowLevelParticleFilters.cov_from_logchol(θ)), θmode)
        @test all(isfinite, J)
    end

    @testset "MAP objective" begin
        v1, v2 = nx + 4, ny + 4
        for diagonal in (true, false)
            s = LSOptExt.autotune_setup(sol_initial; diagonal, v_R1 = v1, v_R2 = v2)
            J(θ) = sum(abs2, LSOptExt.autotune_residuals!(zeros(s.output_length), θ, s))
            function ref(θ)
                R1, R2, x0 = LSOptExt.autotune_unpack(θ, s)
                f = LowLevelParticleFilters.reconstruct_filter(kf, R1, R2, x0)
                -(loglik(f, u, y) + logpdf(InverseWishart(v1, (v1 - nx - 1)*Matrix(R1_initial)), Matrix(R1)) + logpdf(InverseWishart(v2, (v2 - ny - 1)*Matrix(R2_initial)), Matrix(R2)))
            end
            θa = s.θ0
            θb = s.θ0 .+ 0.3 .* randn(length(s.θ0))
            @test J(θa) - J(θb) ≈ ref(θa) - ref(θb)
        end
        @test_throws ArgumentError autotune_covariances(sol_initial; v_R1 = nx + 1)
        @test_throws ArgumentError autotune_covariances(sol_initial; v_R2 = ny + 1)
        res = autotune_covariances(sol_initial; diagonal=false, show_trace=false, v_R1 = nx + 10, v_R2 = ny + 10)
        @test res.sol_opt.ll > sol_initial.ll
        @test isposdef(Matrix(res.R1))
    end

    @testset "Full parameterization" begin
        res = autotune_covariances(sol_initial; diagonal=false, show_trace=false)
        @test res.sol_opt.ll > sol_initial.ll
        @test res.R1 isa SMatrix
        @test isposdef(Matrix(res.R1)) && isposdef(Matrix(res.R2))
        # A singular initial covariance cannot be represented by the log-Cholesky parameterization
        kf_sing = KalmanFilter(A, B, C, 0, SA[0.0 0; 0 0.1], R2_initial, d0)
        @test_throws ArgumentError autotune_covariances(forward_trajectory(kf_sing, u, y); diagonal=false, show_trace=false)
    end

    @testset "SqKalmanFilter" begin
        skf = SqKalmanFilter(A, B, C, 0, R1_initial, R2_initial, d0)
        sols = forward_trajectory(skf, u, y)
        @test sols.ll ≈ sol_initial.ll
        for diagonal in (true, false)
            res = autotune_covariances(sols; diagonal, show_trace=false)
            @test res.filter isa SqKalmanFilter
            @test res.sol_opt.ll > sols.ll
            @test res.R1 ≈ res.filter.R1'res.filter.R1
            res_kf = autotune_covariances(sol_initial; diagonal, show_trace=false)
            @test res.sol_opt.ll ≈ res_kf.sol_opt.ll rtol=1e-4
        end
    end

    @testset "Missing measurements" begin
        ym = Vector{Union{Missing, eltype(y)}}(y)
        ym[10:20] .= missing
        sol_m = forward_trajectory(kf, u, ym)
        res = autotune_covariances(sol_m; show_trace=false)
        @test res.sol_opt.ll > sol_m.ll
    end

    @testset "Automatic offset for small innovation covariances" begin
        # With an innovation variance below 1/(2π), the log-determinant residual requires a positive offset
        kf_small_true = KalmanFilter(A, B, C, 0, 1e-3*R1_true, 1e-3*R2_true, d0)
        _, _, ys = simulate(kf_small_true, u)
        kf_small = KalmanFilter(A, B, C, 0, 1e-2*R1_initial, 1e-2*R2_initial, d0)
        sol_small = forward_trajectory(kf_small, u, ys)
        @test LSOptExt.default_offset(sol_small) > 0
        res = autotune_covariances(sol_small; show_trace=false)
        @test res.sol_opt.ll >= forward_trajectory(kf_small_true, u, ys).ll - 2
        @test_logs (:warn, r"offset") match_mode=:any autotune_covariances(sol_small; show_trace=false, offset=0.0)
    end

    @testset "Function-valued covariance" begin
        kf_fun = KalmanFilter(A, B, C, 0, (x,u,p,t)->R1_initial, R2_initial, d0; nx, nu, ny)
        @test_throws ArgumentError autotune_covariances(forward_trajectory(kf_fun, u, y); show_trace=false)
    end

    @testset "reconstruct_filter retains the filter configuration" begin
        rf = LowLevelParticleFilters.reconstruct_filter
        dyn(x, u, p, t) = A*x + B*u
        meas(x, u, p, t) = C*x
        na = Ref(0)
        nc = Ref(0)
        Ajac = (x, u, p, t) -> (na[] += 1; A)
        Cjac = (x, u, p, t) -> (nc[] += 1; C)
        R12 = SA[0.001 0; 0 0.001]
        ekf = ExtendedKalmanFilter(dyn, meas, R1_initial, R2_initial, d0; nu, Ajac, Cjac, R12)
        ekf2 = rf(ekf, R1_true, R2_true, d0.μ)
        @test getfield(ekf2, :Ajac) === Ajac
        @test ekf2.measurement_model.Cjac === Cjac
        @test ekf2.measurement_model.R12 == R12
        @test ekf2.R1 == R1_true
        @test ekf2.R2 == R2_true
        na[] = nc[] = 0
        forward_trajectory(ekf2, u, y)
        @test na[] > 0 && nc[] > 0

        # Default Jacobians are regenerated, which permits differentiation with respect to the covariance
        ekf_default = ExtendedKalmanFilter(dyn, meas, R1_initial, R2_initial, d0; nu)
        g = ForwardDiff.gradient(θ -> loglik(rf(ekf_default, exp(θ[1])*R1_initial, R2_initial, d0.μ), u, y), [0.0])
        @test isfinite(g[1])

        iekf = IteratedExtendedKalmanFilter(dyn, meas, R1_initial, R2_initial, d0; nu, maxiters=3)
        iekf2 = rf(iekf, R1_true, R2_true, d0.μ)
        @test iekf2.measurement_model isa IEKFMeasurementModel
        @test iekf2.measurement_model.maxiters == 3

        kf_fA = KalmanFilter((x,u,p,t)->A, B, C, 0, R1_initial, R2_initial, d0; nx, nu, ny)
        kf_fA2 = rf(kf_fA, R1_true, R2_true, d0.μ)
        @test forward_trajectory(kf_fA2, u, y).ll ≈ forward_trajectory(rf(kf, R1_true, R2_true, d0.μ), u, y).ll

        reject = x -> false
        innovation = (y, yh) -> y .- yh
        ukf = UnscentedKalmanFilter(dyn, meas, R1_initial, R2_initial, d0; ny, nu, reject, innovation)
        ukf2 = rf(ukf, R1_true, R2_true, d0.μ)
        @test ukf2.reject === reject
        @test ukf2.measurement_model.innovation === innovation
        @test typeof(ukf2.measurement_model.cache) == typeof(ukf.measurement_model.cache)

        @test_throws ArgumentError rf(IMM([kf, kf], [0.5 0.5; 0.5 0.5], [0.5, 0.5]), R1_true, R2_true, d0.μ)
    end
end
