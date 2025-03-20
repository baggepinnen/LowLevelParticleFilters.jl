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
ekf = LLPF.ExtendedKalmanFilter(dynamics_fun, measurement_fun, fw(R1), fw(R2), d0; nu, ny, nx)
ukf = LLPF.UnscentedKalmanFilter(dynamics_fun, measurement_fun, fw(R1), fw(R2), d0; nu, ny)

x,u,y = LLPF.simulate(kf,T,du)
solkf = forward_trajectory(kf, u, y)
solekf = forward_trajectory(ekf, u, y)
solukf = forward_trajectory(ukf, u, y)

@test solkf.x ≈ solekf.x
@test solkf.x ≈ solukf.x

@test solkf.Rt ≈ solekf.Rt
@test solkf.Rt ≈ solukf.Rt