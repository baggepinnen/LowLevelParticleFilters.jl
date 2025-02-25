using LowLevelParticleFilters, ForwardDiff, Distributions
const LLPF = LowLevelParticleFilters
using ControlSystemsBase
using StaticArrays, LinearAlgebra, Test

nx = 2   # Dimension of state
nu = 1   # Dimension of input
ny = 1   # Dimension of measurements


# Define random linenar state-space system
A = SA[1 0.1
             0 1]
B = SA[0.0; 1;;]
C = SA[1 0.0]

R1 = LLPF.double_integrator_covariance(0.1) + 1e-6I
R2 = 1e-3I(1)

dynamics(x,u,p,t) = A*x .+ B*u
measurement(x,u,p,t) = C*x

T = 5000 # Number of time steps

d0 = MvNormal(randn(nx),2.0)   # Initial state Distribution
du = MvNormal(nu,1) # Control input distribution
kf = KalmanFilter(A, B, C, 0, R1, R2, d0, α=1.01)

kf2 = KalmanFilter(ss(A, B, C, 0, 1), R1, R2, d0, α=1.01)
@test kf2.A == kf.A
@test kf2.B == kf.B
@test kf2.C == kf.C
@test all(iszero, kf2.D)
@test kf2.R1 == kf.R1
@test kf2.R2 == kf.R2
@test kf2.d0 == kf.d0

ekf = LLPF.IteratedExtendedKalmanFilter(kf, dynamics, measurement)
ekf2 = LLPF.IteratedExtendedKalmanFilter(dynamics, measurement, R1, R2, d0, α=1.01, nu=nu, maxiters=20, step=0.5, epsilon=1e-6)
ukf = LLPF.UnscentedKalmanFilter(dynamics, measurement, R1, R2, d0, nu=nu, ny=ny)
@test ekf2.kf.R1 == ekf.kf.R1
@test ekf2.kf.R2 == ekf.kf.R2
@test ekf2.kf.d0 == ekf.kf.d0
@test ukf.R1 == ekf.kf.R1
@test ukf.R2 == ekf.kf.R2
@test ukf.d0 == ekf.kf.d0

@test ekf.maxiters == 10
@test ekf.step == 1.0
@test ekf.epsilon == 1e-8
@test ekf2.maxiters == 20
@test ekf2.step == 0.5
@test ekf2.epsilon == 1e-6


@test ekf.measurement_model.Cjac([0,0],[0],[0],0) == C


x,u,y = LLPF.simulate(kf,T,du)

@test kf.nx == nx
@test kf.nu == nu
@test kf.ny == ny

@test LowLevelParticleFilters.measurement(kf)(x[1],u[1],0,0) ≈ measurement(x[1],u[1],0,0)
@test LowLevelParticleFilters.dynamics(kf)(x[1],u[1],0,0) ≈ dynamics(x[1],u[1],0,0)


sol = forward_trajectory(kf, u, y)
ekf = LLPF.IteratedExtendedKalmanFilter(kf, dynamics, measurement; maxiters=20, step=1.0, epsilon=1e-7)
@test ekf.maxiters == 20
@test ekf.step == 1.0
@test ekf.epsilon == 1e-7
sol2 = forward_trajectory(ekf, u, y)


# When the dynamics is linear, these should be the same
@test reduce(hcat, sol.x) ≈ reduce(hcat, sol2.x)
@test reduce(hcat, sol.xt) ≈ reduce(hcat, sol2.xt)
@test reduce(hcat, sol.R)  ≈ reduce(hcat, sol2.R)
@test reduce(hcat, sol.Rt) ≈ reduce(hcat, sol2.Rt)
@test sol.ll ≈ sol2.ll


## add nonlinear dynamics
dynamics2(x,u,p,t) = A*x - 0.01*sin.(x) .+ B*u
ekf = LLPF.IteratedExtendedKalmanFilter(kf, dynamics2, measurement)
x,u,y = LLPF.simulate(ekf,T,du)

sol = forward_trajectory(kf, u, y)
sol2 = forward_trajectory(ekf, u, y)

@test norm(reduce(hcat, x .- sol.x)) > norm(reduce(hcat, x .- sol2.x))
@test norm(reduce(hcat, x .- sol.xt)) > norm(reduce(hcat, x .- sol2.xt))
@test sol.ll < 0.999*sol2.ll # using the ekf should improve ll (we add a small margin since it otherwise fails in about 1/10)

xT,RT,ll = smooth(ekf, u, y)
@test norm(reduce(hcat, x .- xT)) < norm(reduce(hcat, x .- sol2.x)) # Smoothing solution better than filtering sol (normally around 8-20% better)


## Compare all
kf = KalmanFilter(A, B, C, 0, R1, R2, d0)
ekf = LLPF.IteratedExtendedKalmanFilter(kf, dynamics, measurement, maxiters=1)
ukf = LLPF.UnscentedKalmanFilter(dynamics, measurement, R1, R2, d0, nu=nu, ny=ny)

x,u,y = LLPF.simulate(kf,20,du)



sol = forward_trajectory(kf, u, y)
sol2 = forward_trajectory(ekf, u, y)
sol3 = forward_trajectory(ukf, u, y)

@test reduce(hcat, sol.x) ≈ reduce(hcat, sol2.x)
@test reduce(hcat, sol.xt) ≈ reduce(hcat, sol2.xt)
@test reduce(hcat, sol.R)  ≈ reduce(hcat, sol2.R)
@test reduce(hcat, sol.Rt) ≈ reduce(hcat, sol2.Rt)
@test sol.ll ≈ sol2.ll

@test reduce(hcat, sol.x) ≈ reduce(hcat, sol3.x)
@test reduce(hcat, sol.xt) ≈ reduce(hcat, sol3.xt)
@test reduce(hcat, sol.R)  ≈ reduce(hcat, sol3.R)
@test reduce(hcat, sol.Rt) ≈ reduce(hcat, sol3.Rt)
@test sol.ll ≈ sol3.ll




# More tests:
h(x,u,p,t) = [1.0/x[1]]
hjac(x,u,p,t) = hcat(-1.0/x[1]^2)
h2(x,u,p,t) = [u./x[1]]
h2jac(x,u,p,t) = hcat(-u./x[1]^2)

function f(x,u,p,t)
    for i in 1:length(x)
        x[i] = x[i]^i
    end
    x
end


Q = hcat(1.0)
RR = hcat(1.0) * 2
d0 = MvNormal([5.0],1.0*I)
iekf2 = LLPF.IteratedExtendedKalmanFilter(f, h,Q,RR,d0; nu = 0)
iekf = LLPF.IteratedExtendedKalmanFilter(f, h,Q,RR,d0; Cjac=hjac, nu = 0)

sol = correct!(iekf, 100,[1/4],0,0)
sol2 = correct!(iekf2, 100,[1/4],0,0)

# Check that the solution makes sense
@test iekf.x[1] < 5
@test iekf.x[1] > 4
@test iekf.R[1] < 1 # Test that P has decreased
# Test that the two methods give the same results
@test iekf.x ≈ iekf2.x
@test iekf.R ≈ iekf2.R

iekf = LLPF.IteratedExtendedKalmanFilter(f, h2,Q,RR,d0; nu = 1)
iekf2 = LLPF.IteratedExtendedKalmanFilter(f, h2,Q,RR,d0; Cjac=h2jac, nu = 1)

sol = correct!(iekf, 100,[100/4],0,0)
sol2 = correct!(iekf2, 100,[100/4],0,0)

# Check that the solution makes sense
@test iekf.x[1] < 5
@test iekf.x[1] > 4
@test iekf.R[1] < 1 # Test that P has decreased
# Test that the two methods give the same results
@test iekf.x ≈ iekf2.x
@test iekf.R ≈ iekf2.R





