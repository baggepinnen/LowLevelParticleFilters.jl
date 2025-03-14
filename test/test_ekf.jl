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

dynamics_ekf(x,u,p,t) = A*x .+ B*u
measurement_ekf(x,u,p,t) = C*x

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

ekf = LLPF.ExtendedKalmanFilter(kf, dynamics_ekf, measurement_ekf)
ekf2 = LLPF.ExtendedKalmanFilter(dynamics_ekf, measurement_ekf, R1, R2, d0, α=1.01, nu=nu)
ukf = LLPF.UnscentedKalmanFilter(dynamics_ekf, measurement_ekf, R1, R2, d0, nu=nu, ny=ny)
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

@test LowLevelParticleFilters.measurement(kf)(x[1],u[1],0,0) ≈ measurement_ekf(x[1],u[1],0,0)
@test LowLevelParticleFilters.dynamics(kf)(x[1],u[1],0,0) ≈ dynamics_ekf(x[1],u[1],0,0)


sol = forward_trajectory(kf, u, y)
sol2 = forward_trajectory(ekf, u, y)


# When the dynamics_ekf is linear, these should be the same
@test reduce(hcat, sol.x) ≈ reduce(hcat, sol2.x)
@test reduce(hcat, sol.xt) ≈ reduce(hcat, sol2.xt)
@test reduce(hcat, sol.R)  ≈ reduce(hcat, sol2.R)
@test reduce(hcat, sol.Rt) ≈ reduce(hcat, sol2.Rt)
@test sol.ll ≈ sol2.ll


## add nonlinear dynamics_ekf
dynamics_ekf2(x,u,p,t) = A*x - 0.01*sin.(x) .+ B*u
ekf = LLPF.ExtendedKalmanFilter(kf, dynamics_ekf2, measurement_ekf)
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
kf = KalmanFilter(A, B, C, 0, R1, R2, d0)
ekf = LLPF.ExtendedKalmanFilter(kf, dynamics_ekf, measurement_ekf)
ukf = LLPF.UnscentedKalmanFilter(dynamics_ekf, measurement_ekf, R1, R2, d0, nu=nu, ny=ny)

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


xT,RT,ll = smooth(sol, kf, u, y)
xT2,RT2,ll2 = smooth(sol2, ekf, u, y)
xT3,RT3,ll3 = smooth(sol3, ukf, u, y)


# plot(reduce(hcat, x)', lab="true", layout=2)
# plot!(reduce(hcat, sol.xt)', lab="Filter")
# plot!(reduce(hcat, xT)', lab="Smoothed")

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



# plot(reduce(hcat, vec.(sol.Rt))', lab="Filter", layout=4)
# plot!(reduce(hcat, vec.(RT))', lab="Smoothed")


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
measurement_ekf(x,u,p,t) = C*x

function error_dynamics(x,u,p,t)
    error()
end

ekf = LLPF.ExtendedKalmanFilter(kf, error_dynamics, measurement_ekf)
@test_throws ErrorException forward_trajectory(ekf, u, y)

@test_logs (:error,r"State estimation failed") forward_trajectory(ekf, u, y, debug=true)