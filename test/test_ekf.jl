using LowLevelParticleFilters, ForwardDiff, Distributions
const LLPF = LowLevelParticleFilters
using ControlSystemsBase
using StaticArrays, LinearAlgebra, Test

nx = 2   # Dimension of state
nu = 1   # Dimension of input
ny = 1   # Dimension of measurements


# Define random linenar state-space system
A = SA[0.97043   -0.097368
             0.09736    0.970437]
B = SA[0.1; 0;;]
C = SA[0 1.0]

dynamics(x,u,p,t) = A*x .+ B*u
measurement(x,u,p,t) = C*x

T = 5000 # Number of time steps

d0 = MvNormal(randn(nx),2.0)   # Initial state Distribution
du = MvNormal(nu,1) # Control input distribution
kf = KalmanFilter(A, B, C, 0, 0.001I(nx), I(ny), d0, α=1.01)

kf2 = KalmanFilter(ss(A, B, C, 0, 1), 0.001I(nx), I(ny), d0, α=1.01)
@test kf2.A == kf.A
@test kf2.B == kf.B
@test kf2.C == kf.C
@test all(iszero, kf2.D)
@test kf2.R1 == kf.R1
@test kf2.R2 == kf.R2
@test kf2.d0 == kf.d0

ekf = LLPF.ExtendedKalmanFilter(kf, dynamics, measurement)
ekf2 = LLPF.ExtendedKalmanFilter(dynamics, measurement, 0.001I(nx), I(ny), d0, α=1.01, nu=nu)
ukf = LLPF.UnscentedKalmanFilter(dynamics, measurement, 0.001I(nx), I(ny), d0, nu=nu, ny=ny)
@test ekf2.kf.R1 == ekf.kf.R1
@test ekf2.kf.R2 == ekf.kf.R2
@test ekf2.kf.d0 == ekf.kf.d0
@test ukf.R1 == ekf.kf.R1
@test ukf.R2 == ekf.kf.R2
@test ukf.d0 == ekf.kf.d0


x,u,y = LLPF.simulate(kf,T,du)

@test kf.nx == nx
@test kf.nu == nu
@test kf.ny == ny

@test LowLevelParticleFilters.measurement(kf)(x[1],u[1],0,0) ≈ measurement(x[1],u[1],0,0)
@test LowLevelParticleFilters.dynamics(kf)(x[1],u[1],0,0) ≈ dynamics(x[1],u[1],0,0)


sol = forward_trajectory(kf, u, y)
sol2 = forward_trajectory(ekf, u, y)


# When the dynamics is linear, these should be the same
@test reduce(hcat, sol.x) ≈ reduce(hcat, sol2.x)
@test reduce(hcat, sol.xt) ≈ reduce(hcat, sol2.xt)
@test reduce(hcat, sol.R)  ≈ reduce(hcat, sol2.R)
@test reduce(hcat, sol.Rt) ≈ reduce(hcat, sol2.Rt)
@test sol.ll ≈ sol2.ll


## add nonlinear dynamics
dynamics2(x,u,p,t) = A*x - 0.01*sin.(x) .+ B*u
ekf = LLPF.ExtendedKalmanFilter(kf, dynamics2, measurement)
x,u,y = LLPF.simulate(ekf,T,du)

sol = forward_trajectory(kf, u, y)
sol2 = forward_trajectory(ekf, u, y)

@test norm(reduce(hcat, x .- sol.x)) > norm(reduce(hcat, x .- sol2.x))
@test norm(reduce(hcat, x .- sol.xt)) > norm(reduce(hcat, x .- sol2.xt))
@test sol.ll < 0.999*sol2.ll # using the ekf should improve ll (we add a small margin since it otherwise fails in about 1/10)

# plot(reduce(hcat, x)', layout=2, lab="True")
# plot!(reduce(hcat, sol.x)', lab="Kf")
# plot!(reduce(hcat, sol2.x)', lab="EKF")

xT,RT,ll = smooth(ekf, u, y)
@test norm(reduce(hcat, x .- xT)) < norm(reduce(hcat, x .- sol2.x)) # Smoothing solution better than filtering sol (normally around 8-20% better)


## Compare all
kf = KalmanFilter(A, B, C, 0, 0.001I(nx), I(ny), d0)
ekf = LLPF.ExtendedKalmanFilter(kf, dynamics, measurement)
ukf = LLPF.UnscentedKalmanFilter(dynamics, measurement, 0.001I(nx), I(ny), d0, nu=nu, ny=ny)

# x,u,y = LLPF.simulate(kf,10,du)

u = [[-0.22972228297593472], [-0.7873642798704794], [0.7746976713162768], [-1.0677876347753752], [0.08952267711395145], [0.8071581234808373], [1.4628138971331086], [-0.5790514589219693], [0.5824671460456585], [0.9108480524305639]]
y = [[0.6858353350069983], [0.6561596665457351], [1.2423246627002342], [1.5839820173900971], [1.106666174890956], [1.562505325618183], [0.8821829329012782], [0.5652381762379538], [0.835880750271578], [0.8804822291334177]]

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


xT,RT,ll = smooth(sol, kf, u, y)
xT2,RT2,ll2 = smooth(sol2, ekf, u, y)
xT3,RT3,ll3 = smooth(sol3, ukf, u, y)


plot(reduce(hcat, x)', lab="true", layout=2)
plot!(reduce(hcat, sol.xt)', lab="Filter")
plot!(reduce(hcat, xT)', lab="Smoothed")

@test reduce(hcat, xT) ≈ reduce(hcat, xT2)
@test reduce(hcat, RT) ≈ reduce(hcat, RT2)

@test reduce(hcat, xT) ≈ reduce(hcat, xT3)
@test reduce(hcat, RT) ≈ reduce(hcat, RT3)

# Covariance should decrease by smoothing
@test all(reduce(hcat, vec.(RT)) .<= reduce(hcat, vec.(sol.R)))
@test all(reduce(hcat, vec.(RT)) .<= reduce(hcat, vec.(sol.Rt)))

@test all(reduce(hcat, vec.(RT2)) .<= reduce(hcat, vec.(sol2.R)))
@test all(reduce(hcat, vec.(RT2)) .<= reduce(hcat, vec.(sol2.Rt)))

@test all(reduce(hcat, vec.(RT3)) .<= reduce(hcat, vec.(sol3.R)))
@test all(reduce(hcat, vec.(RT3)) .<= reduce(hcat, vec.(sol3.Rt)))


@test norm(reduce(hcat, x .- xT)) < norm(reduce(hcat, x .- sol.x))
@test norm(reduce(hcat, x .- xT)) < norm(reduce(hcat, x .- sol.xt))

@test norm(reduce(hcat, x .- xT)) < norm(reduce(hcat, x .- sol2.x))
@test norm(reduce(hcat, x .- xT)) < norm(reduce(hcat, x .- sol2.xt))

@test norm(reduce(hcat, x .- xT)) < norm(reduce(hcat, x .- sol3.x))
@test norm(reduce(hcat, x .- xT)) < norm(reduce(hcat, x .- sol3.xt))



plot(reduce(hcat, vec.(sol.Rt))', lab="Filter", layout=4)
plot!(reduce(hcat, vec.(RT))', lab="Smoothed")


##
# using KalmanFilters
# # Process model
# F = kf.A
# # Process noise covariance
# Q = kf.R1
# # Measurement model
# H = kf.C
# # Measurement noise covariance
# R = kf.R2
# # Initial state and covariances
# x_init = kf.d0.μ
# P_init = kf.d0.Σ
# # Take first measurement
# mu = KalmanFilters.measurement_update(x_init, P_init, y[1], H, R)
# Rs = [mu.covariance]
# Xs = [mu.state]
# for i = 2:length(y)
#     tu = time_update(get_state(mu), get_covariance(mu), F, Q)
#     mu = measurement_update(get_state(tu), get_covariance(tu), y[i], H, R)
#     push!(Rs, mu.covariance)
#     push!(Xs, mu.state)
# end

# @test Rs ≈ sol.Rt
# @test Xs ≈ sol.xt # tested with sol = forward_trajectory(kf, 0 .* u, y)