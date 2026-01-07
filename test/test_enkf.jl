using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using Test, Random, LinearAlgebra, Statistics, StaticArrays

Random.seed!(42)

mvnormal(d::Int, σ::Real) = SimpleMvNormal(zeros(d), float(σ)^2 * I(d))
mvnormal(μ::AbstractVector{<:Real}, σ::Real) = SimpleMvNormal(μ, float(σ)^2 * I(length(μ)))

eye(n) = SMatrix{n,n}(1.0I(n))

## Test basic EnKF construction and state access

nx = 2  # Dimension of state
nu = 2  # Dimension of input
ny = 2  # Dimension of measurements
N = 100  # Number of ensemble members

d0 = mvnormal(@SVector(randn(nx)), 2.0)

# Define linear state-space system
const _A = SA[0.99 0.1; 0.0 0.2]
const _B = SA[-0.74 1.61; -1.44 1.75]
const _C = SMatrix{ny,ny}(eye(ny))

dynamics(x, u, p, t) = _A * x .+ _B * u
measurement(x, u, p, t) = _C * x

R1 = eye(nx)
R2 = eye(ny)

# Create EnKF
enkf = EnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; nu, ny)
show(enkf)
println()
show(stdout, MIME"text/plain"(), enkf)

@test num_particles(enkf) == N
@test length(particles(enkf)) == N
@test length(state(enkf)) == nx
@test size(covariance(enkf)) == (nx, nx)

# Test that initial ensemble statistics approximately match d0
x̄_init = state(enkf)
P_init = covariance(enkf)
@test norm(x̄_init - d0.μ) < 1.0  # Mean should be close (statistical tolerance)
# Covariance should be approximately d0.Σ (with sampling variance)

## Test reset!
reset!(enkf)
@test enkf.t == 0
@test num_particles(enkf) == N

## Test predict! and correct!
u1 = @SVector randn(nu)
y1 = @SVector randn(ny)

predict!(enkf, u1)
@test enkf.t == 1

# State should have changed after prediction
x_after_pred = state(enkf)

reset!(enkf)
correct!(enkf, u1, y1)
x_after_corr = state(enkf)

# Correction should change the state
@test x_after_corr != d0.μ

## Test update! (correct + predict combined)
reset!(enkf)
ret = update!(enkf, u1, y1)
@test haskey(ret, :ll)
@test haskey(ret, :e)
@test haskey(ret, :S)
@test haskey(ret, :K)
@test enkf.t == 1

## Test callable interface
reset!(enkf)
ret = enkf(u1, y1)
@test enkf.t == 1

## Test simulate
T = 50
du = mvnormal(nu, 1.0)
@test_nowarn x, u, y = LowLevelParticleFilters.simulate(enkf, T, du)

## Comparison with KalmanFilter on linear system
# For linear Gaussian systems, EnKF should give similar results to KF (within sampling variance)

T = 200
kf = KalmanFilter(_A, _B, _C, 0, R1, R2, d0)

# Simulate trajectory
x_true, u, y = LowLevelParticleFilters.simulate(kf, T, du)
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat, y))[:] |> copy
x_true, u, y = tosvec.((x_true, u, y))

# Run both filters
reskf = forward_trajectory(kf, u, y)

# Use larger ensemble for better comparison
enkf_large = EnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, 500; nu, ny)
resenkf = forward_trajectory(enkf_large, u, y)

# Helper function
sse(x) = sum(sum.(abs2, x))

# EnKF should perform reasonably well (within a factor of KF performance)
sse_kf = sse(x_true .- reskf.xt)
sse_enkf = sse(x_true .- resenkf.xt)

@test sse_enkf < 1.2 * sse_kf  # EnKF should not be drastically worse
@test sse_enkf < 500  # Absolute bound on error

# Log-likelihood should be in reasonable range
@test resenkf.ll ≈ reskf.ll atol=5.0

## Test with different ensemble sizes
for N_test in [20, 50, 200]
    enkf_test = EnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N_test; nu, ny)
    @test num_particles(enkf_test) == N_test

    res = forward_trajectory(enkf_test, u[1:10], y[1:10])
    @test isfinite(res.ll)
end

## Test covariance inflation
enkf_inflated = EnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; nu, ny, inflation=1.05)
@test enkf_inflated.inflation == 1.05

res_inflated = forward_trajectory(enkf_inflated, u[1:20], y[1:20])
# @test res_inflated.ll

## Test with time-varying R1
R1_func(x, u, p, t) = t < 100 ? eye(nx) : 2 * eye(nx)
enkf_tvR1 = EnsembleKalmanFilter(dynamics, measurement, R1_func, R2, d0, N; nu, ny)
@test_nowarn forward_trajectory(enkf_tvR1, u[1:20], y[1:20])

## Test particletype and covtype
@test particletype(enkf) == eltype(enkf.ensemble)
@test LowLevelParticleFilters.covtype(enkf) == Matrix{eltype(eltype(enkf.ensemble))}

## Test sample_state and sample_measurement
x0 = state(enkf)
@test_nowarn LowLevelParticleFilters.sample_state(enkf)
@test_nowarn LowLevelParticleFilters.sample_state(enkf, x0, u1)
@test_nowarn LowLevelParticleFilters.sample_measurement(enkf, x0, u1)

## Test with Vector (non-static) arrays
d0_vec = SimpleMvNormal(randn(nx), Matrix(2.0 * I(nx)))
dynamics_vec(x, u, p, t) = Matrix(_A) * x .+ Matrix(_B) * u
measurement_vec(x, u, p, t) = Matrix(_C) * x

enkf_vec = EnsembleKalmanFilter(dynamics_vec, measurement_vec, Matrix(R1), Matrix(R2), d0_vec, N; nu, ny)
u_vec = Vector.(u[1:20])
y_vec = Vector.(y[1:20])
@test_nowarn forward_trajectory(enkf_vec, u_vec, y_vec)

## Test reset with custom x0
reset!(enkf; x0=zeros(nx))
x̄_after_reset = state(enkf)
@test norm(x̄_after_reset) < 2.0  # Should be centered around zeros

## Test that EnKF handles missing inputs gracefully
dynamics_no_input(x, u, p, t) = _A * x
enkf_no_input = EnsembleKalmanFilter(dynamics_no_input, measurement, R1, R2, d0, N; nu=0, ny)
u_empty = [SVector{0,Float64}() for _ in 1:20]
@test_nowarn forward_trajectory(enkf_no_input, u_empty, y[1:20])

## Test that output solution has correct format (KalmanFilteringSolution)
sol = forward_trajectory(enkf, u[1:20], y[1:20])
@test sol isa LowLevelParticleFilters.KalmanFilteringSolution
@test length(sol.x) == 20   # T time steps (predictions)
@test length(sol.xt) == 20  # T time steps (filtered)
@test length(sol.R) == 20   # Prediction covariances
@test length(sol.Rt) == 20  # Filtered covariances
@test length(sol.e) == 20   # Innovations

