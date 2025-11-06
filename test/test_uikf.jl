using LowLevelParticleFilters, Plots
using LinearAlgebra, Random, Test

# Example from Section 5 of Darouach et al. (1995)
# "Kalman filtering with unknown inputs via optimal state estimation of singular systems"


# System dimensions
nx, nu, ny, nd = 3, 1, 2, 1

# System matrices from the paper
A = [0.0    0.6   0.075;
     0.75   0.0   0.0;
     0.0    0.75  0.0375]

B = reshape([1.0, 1.0, 0.0], 3, 1)

F = reshape([0.0, 1.0, 1.0], 3, 1)  # Unknown input matrix

C = [1.0  1.0  0.0;
     0.0  1.0  1.0]

# Noise covariances from the paper
R1 = diagm([3.0, 6.0, 9.0])
R2 = diagm([12.0, 12.0])

# Initial state covariance
P0 = 10.0 * I(nx)

# Create filter
uikf = UIKalmanFilter(A, B, C, zeros(ny, nu), F, R1, R2; nu, ny)

# Simulate system for 100 time steps
T = 100
x_true = zeros(nx, T+1)
d_true = zeros(nd, T)
x_true[:, 1] = randn(nx)  # Random initial state

# Generate true system trajectory
u_data = [0*randn(nu) for _ in 1:T]
y_data = Vector{Vector{Float64}}(undef, T)

for t in 1:T
    # Generate unknown input (random signal)
    d_true[:, t] .= 10 * sign(sin(2pi*t/50))

    # True state evolution
    x_true[:, t+1] = A * x_true[:, t] + B * u_data[t] + F * d_true[:, t] + sqrt.(diag(R1)) .* randn(nx)

    # Measurement
    y_data[t] = C * x_true[:, t] + sqrt.(diag(R2)) .* randn(ny)
end


sol = forward_trajectory(uikf, u_data, y_data)
x_errors = [norm(sol.xt[t] - x_true[:, t+1]) for t in 1:T]
mean_x_error = sum(x_errors) / T

plot(sol, ploty=false, plotyh=false)
plot!(x_true', label="True State", lw=2, ls=:dash)
plot!(reduce(vcat, sol.extra.d), label="Estimated Unknown Input", lw=2, ls=:dot, sp=4)


##
kf = KalmanFilter(A, B, C, zeros(ny, nu), R1, R2; nu, ny)
u_data = [randn(nu) for _ in 1:T]
x, u, y = simulate(kf, u_data)
solkf = forward_trajectory(kf, u, y)
soluikf = forward_trajectory(uikf, u, y)
@test solkf.ll ≈ soluikf.ll rtol=1e-1 # uikf is typically slightly worse

@test norm(soluikf.xt .- x) < 1.3*norm(solkf.xt .- x)
@test norm(soluikf.x .- x) < 1.3*norm(solkf.x .- x)