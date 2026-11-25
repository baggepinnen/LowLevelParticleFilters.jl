# A failed Cholesky factorization inside a filter throws FilterPosDefException, whose
# message describes the matrix that failed, and which the documented catch pattern handles.
# Wrapped in `let` so that the fixtures do not collide with globals of other test files.
using LowLevelParticleFilters, LinearAlgebra, Test
let LLPF = LowLevelParticleFilters
    nx, nu, ny = 2, 1, 1
    A = [1.0 0.1; 0 1]; B = [0.0; 0.1;;]; C = [1.0 0]
    R1 = Matrix(0.01I(nx))
    R2bad = fill(-10.0, 1, 1)                 # innovation covariance not positive definite
    d0 = LLPF.SimpleMvNormal(zeros(nx), Matrix(0.01I(nx)))
    u = [[0.1] for _ in 1:10]
    y = [[0.0] for _ in 1:10]
    dyn = (x, u, p, t) -> A * x + B * u
    meas = (x, u, p, t) -> C * x

    kf = KalmanFilter(A, B, C, 0, R1, R2bad, d0)
    ekf = ExtendedKalmanFilter(dyn, meas, R1, R2bad, d0; nu, ny)
    ukf = UnscentedKalmanFilter(dyn, meas, R1, R2bad, d0; nu, ny)
    iekf = IteratedExtendedKalmanFilter(dyn, meas, R1, R2bad, d0; nu, ny)
    filters = (kf, ekf, ukf, iekf)

    @testset "innovation covariance: $(nameof(typeof(f)))" for f in filters
        @test_throws FilterPosDefException loglik(deepcopy(f), u, y)
        @test_throws FilterPosDefException forward_trajectory(deepcopy(f), u, y)
    end
    @testset "EnsembleKalmanFilter" begin
        # the constructor factorizes R2, so the indefinite R2 is passed to correct!
        enkf = EnsembleKalmanFilter(dyn, meas, R1, fill(0.1, 1, 1), d0, 50; nu, ny)
        @test_throws FilterPosDefException correct!(enkf, u[1], y[1]; R2 = R2bad)
    end
    @testset "prediction_errors!: $(nameof(typeof(f)))" for f in filters
        @test_throws FilterPosDefException LLPF.prediction_errors!(zeros(length(y) * ny), deepcopy(f), u, y)
    end

    @testset "UIKalmanFilter" begin
        uikf = UIKalmanFilter(A, B, C, zeros(ny, nu), reshape([0.0, 1.0], 2, 1), R1, R2bad; nu, ny)
        @test_throws FilterPosDefException forward_trajectory(uikf, u, y)
    end

    @testset "message" begin
        e = try
            loglik(deepcopy(kf), u, y)
        catch err
            err
        end
        @test e isa FilterPosDefException
        @test e.t == 0
        @test size(e.M) == (1, 1) && e.M[1] < 0
        msg = sprint(showerror, e)
        @test occursin(r"^FilterPosDefException: Cholesky factorization of innovation covariance failed at time step 0, got S = \[", msg)
        @test occursin("Troubleshooting-Kalman-filters", msg)
    end

    @testset "UKF sigma points of a state covariance that is not positive definite" begin
        ukf2 = UnscentedKalmanFilter(dyn, meas, R1, fill(0.1, 1, 1), d0; nu, ny)
        ukf2.R = -ukf2.R
        @test_throws FilterPosDefException predict!(ukf2, u[1])
    end

    @testset "the documented catch pattern returns Inf" begin
        cost(f) = try
            -loglik(deepcopy(f), u, y)
        catch e
            e isa Union{PosDefException, FilterPosDefException} || rethrow()
            Inf
        end
        @test all(isinf, cost.(filters))
    end
end
