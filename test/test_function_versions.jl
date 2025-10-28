using LowLevelParticleFilters
const LLPF = LowLevelParticleFilters
using StaticArrays, LinearAlgebra, Test
using LowLevelParticleFilters: double_integrator_covariance, SimpleMvNormal

nx = 2   # Dimension of state
nu = 1   # Dimension of input
ny = 1   # Dimension of measurements


# Define random linenar state-space system
A = SA[1 0.1
             0 1]
B = SA[0.0; 1;;]
C = SA[1 0.0]

fw(m) = (x,u,p,t)->m

R1 = double_integrator_covariance(0.1) + 1e-6I
R2 = 1e-3I(1)

dynamics_fun(x,u,p,t) = A*x .+ B*u
measurement_fun(x,u,p,t) = C*x

T = 200 # Number of time steps

d0 = SimpleMvNormal(randn(nx),2.0I(nx))   # Initial state Distribution
du = SimpleMvNormal(I(nu)) # Control input distribution
kf = KalmanFilter(fw(A), fw(B), fw(C), 0, fw(R1), fw(R2), d0; nx, nu, ny)
ekf = ExtendedKalmanFilter(dynamics_fun, measurement_fun, fw(R1), fw(R2), d0; nu, ny, nx)
ukf = UnscentedKalmanFilter(dynamics_fun, measurement_fun, fw(R1), fw(R2), d0; nu, ny)

x,u,y = LLPF.simulate(kf,T,du)
x,u,y = LLPF.simulate(ekf,T,du)
x,u,y = LLPF.simulate(ukf,T,du)
solkf = forward_trajectory(kf, u, y)
solekf = forward_trajectory(ekf, u, y)
solukf = forward_trajectory(ukf, u, y)

@test solkf.x ≈ solekf.x
@test solkf.x ≈ solukf.x

@test solkf.Rt ≈ solekf.Rt
@test solkf.Rt ≈ solukf.Rt

plot(solkf)
plot(solekf)
plot(solukf)

##

@test_throws "A SimpleMvNormal distribution must be initialized with a covariance matrix, not a function." KalmanFilter(fw(A), fw(B), fw(C), 0, fw(R1), fw(R2); nx, nu, ny)
@test_throws "A SimpleMvNormal distribution must be initialized with a covariance matrix, not a function." ExtendedKalmanFilter(dynamics_fun, measurement_fun, fw(R1), fw(R2); nu, ny, nx)
@test_throws "A SimpleMvNormal distribution must be initialized with a covariance matrix, not a function." UnscentedKalmanFilter(dynamics_fun, measurement_fun, fw(R1), fw(R2); nu, ny)

## Test nothing matrix support
@testset "Nothing matrix support" begin
    # Test KalmanFilter with B=nothing (no input)
    kf_no_input = KalmanFilter(A, nothing, C, 0, R1, R2, d0; nx, nu=0, ny)
    x_no_u, _, y_no_u = LLPF.simulate(kf_no_input, T)
    @test length(x_no_u) == T
    @test length(y_no_u) == T

    # Test LinearMeasurementModel with D=nothing (no feedthrough)
    mm_no_D = LinearMeasurementModel(C, nothing, R2; nx, ny)
    @test_nowarn measurement(mm_no_D, zeros(nx), zeros(nu), nothing, 0)
end