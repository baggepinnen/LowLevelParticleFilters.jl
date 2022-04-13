using LowLevelParticleFilters, ForwardDiff, Distributions
const LLPF = LowLevelParticleFilters

n = 2   # Dimension of state
m = 1   # Dimension of input
p = 1   # Dimension of measurements


# Define random linenar state-space system
A = SA[0.97043   -0.097368
             0.09736    0.970437]
B = SA[0.1; 0;;]
C = SA[0 1.0]

dynamics(x,u,p,t) = A*x .+ B*u
measurement(x,u,p,t) = C*x

T = 800 # Number of time steps

d0 = MvNormal(randn(n),2.0)   # Initial state Distribution
du = MvNormal(m,1) # Control input distribution
kf = KalmanFilter(A, B, C, 0, 0.001I(n), I(p), d0)
ekf = LLPF.ExtendedKalmanFilter(kf, dynamics, measurement)
x,u,y = LLPF.simulate(kf,T,du)

@test kf.nx == n
@test kf.nu == m
@test kf.ny == p

@test LowLevelParticleFilters.measurement(kf)(x[1],u[1],0,0) == measurement(x[1],u[1],0,0)
@test LowLevelParticleFilters.dynamics(kf)(x[1],u[1],0,0) == dynamics(x[1],u[1],0,0)


sol = forward_trajectory(kf, u, y)
sol2 = forward_trajectory(ekf, u, y)


# When the dynamics is linear, these should be the same
@test reduce(hcat, sol.x) ≈ reduce(hcat, sol2.x)
@test reduce(hcat, sol.xt) ≈ reduce(hcat, sol2.xt)
@test reduce(hcat, sol.R)  ≈ reduce(hcat, sol2.R)
@test reduce(hcat, sol.Rt) ≈ reduce(hcat, sol2.Rt)
@test sol.ll ≈ sol2.ll


## add nonlinear dynamics
dynamics2(x,u,p,t) = A*x - 0.01*abs.(x) .+ B*u
ekf = LLPF.ExtendedKalmanFilter(kf, dynamics2, measurement)
x,u,y = LLPF.simulate(ekf,T,du)

sol = forward_trajectory(kf, u, y)
sol2 = forward_trajectory(ekf, u, y)

@test norm(reduce(hcat, x .- sol.x)) > norm(reduce(hcat, x .- sol2.x))
@test norm(reduce(hcat, x .- sol.xt)) > norm(reduce(hcat, x .- sol2.xt))
@test sol.ll < sol2.ll

# plot(reduce(hcat, x)', layout=2, lab="True")
# plot!(reduce(hcat, xf)', lab="Kf")
# plot!(reduce(hcat, xf2)', lab="EKF")

xT,RT,ll = smooth(ekf, u, y)
@test norm(reduce(hcat, x .- xT)) < norm(reduce(hcat, x .- sol2.x)) # Smoothing solution better than filtering sol