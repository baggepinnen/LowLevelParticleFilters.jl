# A failed Cholesky factorization inside a filter throws LinearAlgebra.PosDefException,
# the type that the documentation tells users to catch during parameter estimation, and
# its description is available as a debug log message.
using LowLevelParticleFilters, LinearAlgebra, Test, Logging
const LLPF = LowLevelParticleFilters

nx, nu, ny = 2, 1, 1
A = [1.0 0.1; 0 1]; B = [0.0; 0.1;;]; C = [1.0 0]
R1 = 0.01I(nx) |> Matrix
R2bad = fill(-10.0, 1, 1)                 # innovation covariance not positive definite
d0 = LLPF.SimpleMvNormal(zeros(nx), 0.01I(nx) |> Matrix)
u = [[0.1] for _ in 1:10]
y = [[0.0] for _ in 1:10]
dyn(x, u, p, t) = A * x + B * u
meas(x, u, p, t) = C * x

kf = KalmanFilter(A, B, C, 0, R1, R2bad, d0)
ekf = ExtendedKalmanFilter(dyn, meas, R1, R2bad, d0; nu, ny)
ukf = UnscentedKalmanFilter(dyn, meas, R1, R2bad, d0; nu, ny)
iekf = IteratedExtendedKalmanFilter(dyn, meas, R1, R2bad, d0; nu, ny)

@testset "innovation covariance: $(nameof(typeof(f)))" for f in (kf, ekf, ukf, iekf)
    @test_throws PosDefException loglik(deepcopy(f), u, y)
    @test_throws PosDefException forward_trajectory(deepcopy(f), u, y)
    @test_throws PosDefException LLPF.prediction_errors!(zeros(length(y) * ny), deepcopy(f), u, y)
end

@testset "debug message" begin
    @test_logs (:debug, r"Cholesky factorization of innovation covariance failed at time step") min_level = Logging.Debug match_mode = :any begin
        try
            loglik(deepcopy(kf), u, y)
        catch e
            e isa PosDefException || rethrow()
        end
    end
end

@testset "UKF sigma points of a state covariance that is not positive definite" begin
    ukf2 = UnscentedKalmanFilter(dyn, meas, R1, fill(0.1, 1, 1), d0; nu, ny)
    ukf2.R = -ukf2.R
    @test_throws PosDefException predict!(ukf2, u[1])
end

@testset "the documented catch pattern returns Inf" begin
    cost(f) = try
        -loglik(deepcopy(f), u, y)
    catch e
        e isa PosDefException || rethrow()
        Inf
    end
    @test all(isinf, cost.((kf, ekf, ukf, iekf)))
end
