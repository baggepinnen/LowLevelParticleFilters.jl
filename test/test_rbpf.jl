using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using Random, StaticArrays, Test

f_n(xn, args...) = xn   # Identity function for demonstration
A_n(xn, args...) = SA[0.5;;]  # Example matrix (1x1)
A = SA[0.95;;]  # Example matrix (1x1)
C2 = SA[1.0;;]    # Example matrix (1x1)
h(xn, args...) = xn       # Nonlinear measurement function (the example is actually linear for simplicity)
nu = 0
ny = 1
nx = 2
B = @SMatrix zeros(1,0)
D = 0

# Noise covariances (1x1 matrices for 1D case)
R1n = SA[0.01;;]  # Nonlinear state noise covariance
R1l = SA[0.01;;]  # Linear state noise covariance (may also be a function of (x,u,p,t)
R2 = SA[0.1;;]    # Measurement noise (shared between linear and nonlinear parts)

# Initial states (1D)
x0n = SA[1.0]  # Initial state for the nonlinear part
x0l = SA[1.0]  # Initial state for the linear part
R0 = SA[1.0;;] # Initial covariance for the linear part
d0l = SimpleMvNormal(x0l, R0)  # Initial distribution for the linear part
d0n = SimpleMvNormal(x0n, R1n) # Initial distribution for the nonlinear part

kf = KalmanFilter(A, B, C2, D, R1l, R2, d0l) # Inner Kalman filter for the linear part

mm = RBMeasurementModel{false}(h, R2, ny) # Measurement model
pf = RBPF{false, false}(500, kf, f_n, mm, R1n, d0n; nu, An=A_n, Ts=1.0, names=SignalNames(x=["x1", "x2"], u=[], y=["y1"], name="RBPF"))

u = [SA_F64[] for _ in 1:10000]
x,u,y = simulate(pf, u)

sol = forward_trajectory(pf, u, y)
a = @allocated forward_trajectory(pf, u, y)
@test a < 200092592*1.1

a = @allocations forward_trajectory(pf, u, y)
@test a < (isinteractive() ? 10*1.1 : 59936*1.1) # For some reason, CI gives worse performance


using Plots
plot(sol, size=(1000,800), xreal=x)

# @time forward_trajectory(pf, u, y); with N = 500 and T=10000 
# 0.257749 seconds (5.86 M allocations: 204.700 MiB, 6.52% gc time)
# 0.006624 seconds (305.01 k allocations: 11.242 MiB) static arrays in dynamics and covs
# 0.009420 seconds (103.01 k allocations: 5.078 MiB) new method for rand on SimpleMvNormal with static length
# 0.005039 seconds (2.01 k allocations: 1.996 MiB)B also static
# 0.003756 seconds (10 allocations: 1.927 MiB) SimpleMvNormal everywhere

@test sol.x[1] isa RBParticle
@test length(sol.x[1]) == pf.nx == 2

dd = LowLevelParticleFilters.dynamics_density(pf)
@test dd isa SimpleMvNormal
@test length(dd) == 2

@test !LowLevelParticleFilters.isinplace(pf.nl_measurement_model)
@test LowLevelParticleFilters.has_oop(pf.nl_measurement_model)
@test LowLevelParticleFilters.to_mv_normal(dd) === dd

dm = LowLevelParticleFilters.measurement_density(pf)
@test dm == SimpleMvNormal(R2)

## Test simple linear setting where correct answer is known
import LowLevelParticleFilters as LLPF
using LinearAlgebra

nx = 2   # Dimension of state
nu = 1   # Dimension of input
ny = 1   # Dimension of measurements

# Define random linenar state-space system
A = SA[1 0.1
             0 1]
B = SA[0.0; 1;;]
C = SA[1 0.0]

R1 = LLPF.double_integrator_covariance(0.1) + 1e-6I
R2 = SMatrix{1,1}(10.0I(1))

# dynamics(x,u,p,t) = A*x .+ B*u
# measurement(x,u,p,t) = C*x

T = 500 # Number of time steps

d0 = SimpleMvNormal(SVector{nx}(randn(nx)),SMatrix{2,2}(2.0I(nx)))   # Initial state Distribution
du = SimpleMvNormal(I(nu)) # Control input distribution
kf = KalmanFilter(A, B, C, 0, R1, R2, d0)

x,u,y = simulate(kf,T,du)

solkf = forward_trajectory(kf, u, y)


## RBPF (everything is linear)
g = (xn,u,p,t) -> SA[0.0] # No nonlinear contribution
An = nothing
mm = RBMeasurementModel{false}(g, R2, ny)
R1n = SA_F64[0.0;;] # One cannot rand of an empty distribution so we fake one nonlinear state variable
d0n = SimpleMvNormal(R1n)

pf = RBPF{false, false}(500, kf, f_n, mm, R1n, d0n; nu, An, Ts=1.0, names=SignalNames(x=["x1", "x2"], u=["u"], y=["y1"], name="RBPF"))

solrb = forward_trajectory(pf, u, y)
@test solkf.ll ≈ solrb.ll rtol=1e-2

plot(solrb)
plot!(solkf, plotu=false, ploty=false, plotyh=false, sp=[2 3], size=(1000,1000))

## RBPF (everything is nonlinear)


A0 = SA[0.0;;]
B0 = SA[0.0;;]
C0 = SA[0.0;;]
R10 = SA[1.0;;]
d00 = SimpleMvNormal(SA_F64[0.0;;]) # Fake linear state

kf2 = KalmanFilter(A0, B0, C0, 0, R10, R2, d00)
g = (xn,u,p,t) -> C*xn
An = nothing
dyn = (x,u,p,t) -> A*x + B*u
mm = RBMeasurementModel{false}(g, R2, ny)
R1n = R1

d0n = d0
pf2 = RBPF{false, false}(500, kf2, dyn, mm, R1n, d0n; nu, An, Ts=1.0, names=SignalNames(x=["x1", "x2"], u=["u"], y=["y1"], name="RBPF"))

solrb2 = forward_trajectory(pf2, u, y)
@test solkf.ll ≈ solrb2.ll rtol=1e-2


plot(solrb2)
plot!(solkf, plotu=false, ploty=false, plotyh=false, sp=[1 2], size=(1000,1000))