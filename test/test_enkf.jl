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

## Test threads option
enkf_threaded = EnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; nu, ny, threads=true)
@test enkf_threaded.threads == true

# Test that threaded version produces valid results
res_threaded = forward_trajectory(enkf_threaded, u[1:20], y[1:20])
@test isfinite(res_threaded.ll)
@test length(res_threaded.xt) == 20

# Test that threaded and non-threaded produce similar results (same RNG seed)
rng_seed = 12345
enkf_serial = EnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; nu, ny, threads=false, rng=Random.Xoshiro(rng_seed))
enkf_parallel = EnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; nu, ny, threads=true, rng=Random.Xoshiro(rng_seed))
res_serial = forward_trajectory(enkf_serial, u[1:20], y[1:20])
res_parallel = forward_trajectory(enkf_parallel, u[1:20], y[1:20])
@test res_serial.ll ≈ res_parallel.ll
@test all(res_serial.xt .≈ res_parallel.xt)
@test all(res_serial.x .≈ res_parallel.x)

# Test with inflation and threads combined
enkf_threaded_inflated = EnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; nu, ny, threads=true, inflation=1.05)
@test enkf_threaded_inflated.threads == true
@test enkf_threaded_inflated.inflation == 1.05
res_threaded_inflated = forward_trajectory(enkf_threaded_inflated, u[1:20], y[1:20])
@test isfinite(res_threaded_inflated.ll)

# Test threaded version with Vector arrays
enkf_vec_threaded = EnsembleKalmanFilter(dynamics_vec, measurement_vec, Matrix(R1), Matrix(R2), d0_vec, N; nu, ny, threads=true)
@test_nowarn forward_trajectory(enkf_vec_threaded, u_vec, y_vec)

## Augmented form
dynamics_aug(x, u, p, t, w) = _A * x .+ _B * u .+ w
measurement_aug(x, u, p, t, e) = _C * x .+ e

enkf_aug = EnsembleKalmanFilter{true,true}(dynamics_aug, measurement_aug, R1, R2, d0, 500; nu, ny)
show(enkf_aug)
println()
show(stdout, MIME"text/plain"(), enkf_aug)
@test enkf_aug isa EnsembleKalmanFilter{true,true}
@test EnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; nu, ny) isa EnsembleKalmanFilter{false,false}

res_aug = forward_trajectory(enkf_aug, u, y)
@test sse(x_true .- res_aug.xt) < 1.2 * sse_kf
@test res_aug.ll ≈ reskf.ll atol=10.0

# Augmentation of only one of the functions
for (AUGD, AUGM) in [(true, false), (false, true)]
    dyn = AUGD ? dynamics_aug : dynamics
    meas = AUGM ? measurement_aug : measurement
    enkf_mixed = EnsembleKalmanFilter{AUGD,AUGM}(dyn, meas, R1, R2, d0, 500; nu, ny)
    res_mixed = forward_trajectory(enkf_mixed, u, y)
    @test sse(x_true .- res_mixed.xt) < 1.2 * sse_kf
    @test res_mixed.ll ≈ reskf.ll atol=10.0
end

# Sampling functions and simulation
x0 = state(enkf_aug)
@test LowLevelParticleFilters.sample_state(enkf_aug, x0, u1; noise=false) ≈ _A * x0 + _B * u1
@test LowLevelParticleFilters.sample_measurement(enkf_aug, x0, u1; noise=false) ≈ _C * x0
@test LowLevelParticleFilters.sample_state(enkf_aug, x0, u1) != _A * x0 + _B * u1
@test LowLevelParticleFilters.sample_measurement(enkf_aug, x0, u1) != _C * x0
@test_nowarn LowLevelParticleFilters.simulate(enkf_aug, 50, du)

## Augmented dynamics with fewer noise variables than state variables
const _Bw = SA[0.0; 1.0;;]
dynamics_aug_w(x, u, p, t, w) = _A * x .+ _B * u .+ _Bw * w
R1w = SA[0.5;;]
enkf_w = EnsembleKalmanFilter{true,false}(dynamics_aug_w, measurement, R1w, R2, d0, 500; nu, ny)
x_w, u_w, y_w = tosvec.(LowLevelParticleFilters.simulate(enkf_w, T, du))
kf_w = KalmanFilter(_A, _B, _C, 0, _Bw * R1w * _Bw', R2, d0)
res_kf_w = forward_trajectory(kf_w, u_w, y_w)
res_enkf_w = forward_trajectory(enkf_w, u_w, y_w)
@test sse(x_w .- res_enkf_w.xt) < 1.2 * sse(x_w .- res_kf_w.xt)
@test res_enkf_w.ll ≈ res_kf_w.ll atol=10.0

## Interchangeability with the augmented UKF, nonlinear system with non-additive noise
dynamics_nl(x, u, p, t, w) = SA[0.99x[1] + 0.1x[2], 0.2x[2] + 0.5sin(x[1])] .+ _B * u .+ _Bw * w
measurement_nl(x, u, p, t, e) = _C * x .+ (1 .+ 0.1 .* abs.(x)) .* e
ukf_nl = UnscentedKalmanFilter{false,false,true,true}(dynamics_nl, measurement_nl, R1w, R2, d0; nu, ny)
enkf_nl = EnsembleKalmanFilter{true,true}(dynamics_nl, measurement_nl, R1w, R2, d0, 500; nu, ny)
x_nl, u_nl, y_nl = tosvec.(LowLevelParticleFilters.simulate(ukf_nl, T, du))
res_ukf_nl = forward_trajectory(ukf_nl, u_nl, y_nl)
res_enkf_nl = forward_trajectory(enkf_nl, u_nl, y_nl)
@test sse(x_nl .- res_enkf_nl.xt) < 1.2 * sse(x_nl .- res_ukf_nl.xt)
@test res_enkf_nl.ll ≈ res_ukf_nl.ll atol=10.0

## Threaded augmented filter produces the same result as the serial one for equal seeds
enkf_aug_serial = EnsembleKalmanFilter{true,true}(dynamics_nl, measurement_nl, R1w, R2, d0, N; nu, ny, rng=Random.Xoshiro(rng_seed))
enkf_aug_parallel = EnsembleKalmanFilter{true,true}(dynamics_nl, measurement_nl, R1w, R2, d0, N; nu, ny, threads=true, rng=Random.Xoshiro(rng_seed))
res_aug_serial = forward_trajectory(enkf_aug_serial, u_nl[1:20], y_nl[1:20])
res_aug_parallel = forward_trajectory(enkf_aug_parallel, u_nl[1:20], y_nl[1:20])
@test res_aug_serial.ll ≈ res_aug_parallel.ll
@test all(res_aug_serial.xt .≈ res_aug_parallel.xt)

## Augmented form with Vector (non-static) arrays
dynamics_aug_vec(x, u, p, t, w) = Matrix(_A) * x .+ Matrix(_B) * u .+ w
measurement_aug_vec(x, u, p, t, e) = Matrix(_C) * x .+ e
enkf_aug_vec = EnsembleKalmanFilter{true,true}(dynamics_aug_vec, measurement_aug_vec, Matrix(R1), Matrix(R2), d0_vec, N; nu, ny)
res_aug_vec = forward_trajectory(enkf_aug_vec, u_vec, y_vec)
@test isfinite(res_aug_vec.ll)
