using LowLevelParticleFilters
using Test, Random, LinearAlgebra, Statistics, StaticArrays, Distributions
using JET
Random.seed!(0)


## KF

eye(n) = SMatrix{n,n}(Matrix{Float64}(I,n,n))
nx = 2 # Dinemsion of state
nu = 2 # Dinemsion of input
ny = 2 # Dinemsion of measurements

d0 = MvNormal(@SVector(randn(nx)),2.0)   # Initial state Distribution
du = MvNormal(2,1) # Control input distribution

# Define linenar state-space system
const _A = SA[0.99 0.1; 0 0.2]
const _B = @SMatrix [-0.7400216956683083 1.6097265310456392; -1.4384539113366408 1.7467811974822818]
const _C = SMatrix{ny,ny}(eye(ny))
# C = SMatrix{p,n}([1 1])

dynamics(x,u,p,t) = _A*x .+ _B*u
measurement(x,u,p,t) = _C*x

T    = 200 # Number of time steps
kf   = KalmanFilter(_A, _B, _C, 0, eye(nx), eye(ny), d0)
@test kf.R isa SMatrix{2, 2, Float64, 4}
x,u,y = LowLevelParticleFilters.simulate(kf,T,du) # Simuate trajectory using the model in the filter
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
x,u,y = tosvec.((x,u,y))
@test_opt predict!(kf, u[1])
@report_call predict!(kf, u[1])

@test_opt correct!(kf, u[1], y[1])
@report_call correct!(kf, u[1], y[1])

ukf  = UnscentedKalmanFilter(dynamics, measurement, eye(nx), eye(ny), d0; ny, nu)
@test ukf.R1 isa SMatrix{2, 2, Float64, 4}
@test_opt predict!(ukf, u[1])
@report_call predict!(ukf, u[1])

@test_opt correct!(ukf, u[1], y[1])
@report_call correct!(ukf, u[1], y[1])


skf  = SqKalmanFilter(_A, _B, _C, 0, eye(nx), eye(ny), d0)
@test skf.R1.data isa SMatrix{2, 2, Float64, 4}
@test_opt predict!(skf, u[1])
@report_call predict!(skf, u[1])

@test_opt correct!(skf, u[1], y[1])
@report_call correct!(skf, u[1], y[1])



## Test allocations ============================================================
forward_trajectory(kf, u, y) 
a = @allocations forward_trajectory(kf, u, y) 
@test a <= 15


forward_trajectory(ukf, u, y) 
a = @allocations forward_trajectory(ukf, u, y) 
@test a <= 15

forward_trajectory(skf, u, y)
a = @allocations forward_trajectory(skf, u, y)

@test a <= 50 # was 7 on julia v1.10.6


## Test differentiability ======================================================