"""
    EnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; kwargs...)
    EnsembleKalmanFilter{AUGD,AUGM}(dynamics, measurement, R1, R2, d0, N; kwargs...)

An Ensemble Kalman Filter (EnKF) that uses an ensemble of states instead of explicitly
tracking the covariance matrix. This makes it suitable for high-dimensional systems
where covariance matrices become intractable.

This implementation uses the **Stochastic EnKF** formulation with perturbed observations.

The dynamics and measurement functions are on _either_ of the following forms
```
x' = dynamics(x, u, p, t) + w
y  = measurement(x, u, p, t) + e
```
```
x' = dynamics(x, u, p, t, w)
y  = measurement(x, u, p, t, e)
```
where `w ~ N(0, R1)` and `e ~ N(0, R2)`. The former (default) assumes that the noise is additive, while the latter, the _augmented_ form, allows the noise to enter the dynamics and measurement functions in an arbitrary way. See "Augmented EnKF" below.

# Arguments
- `dynamics`: Dynamics function `f(x, u, p, t) -> x⁺`, or `f(x, u, p, t, w) -> x⁺` if the dynamics is augmented
- `measurement`: Measurement function `h(x, u, p, t) -> y`, or `h(x, u, p, t, e) -> y` if the measurement is augmented
- `R1`: Process noise covariance matrix
- `R2`: Measurement noise covariance matrix
- `d0`: Initial state distribution (must support `rand` and have fields `μ` and `Σ`)
- `N::Int`: Number of ensemble members

# Keyword Arguments
- `nu::Int`: Number of inputs (required)
- `ny::Int = size(R2, 1)`: Number of outputs
- `p = NullParameters()`: Parameters passed to dynamics and measurement functions
- `Ts = 1.0`: Sample time
- `inflation = 1.0`: Covariance inflation factor (≥1.0). Values > 1.0 inflate the ensemble
  spread after each prediction step to prevent filter divergence.
- `rng = Random.Xoshiro()`: Random number generator
- `threads = false`: Use threads to propagate ensemble members in parallel. Only activate this
  if your dynamics and measurement functions are thread-safe.
- `names = default_names(...)`: Signal names for plotting

# Algorithm

## Predict Step
For each ensemble member `i = 1:N`:
```math
x_i^- = f(x_i, u, p, t) + w_i \\quad \\text{where } w_i \\sim \\mathcal{N}(0, R_1)
```

## Correct Step (Stochastic EnKF)
1. Ensemble mean: ``\\bar{x} = \\frac{1}{N} \\sum_i x_i``
2. Anomaly matrix: ``X' = [x_1 - \\bar{x}, \\ldots, x_N - \\bar{x}]``
3. Predicted measurements: ``y_i = h(x_i, u, p, t)``, ``\\bar{y} = \\frac{1}{N} \\sum_i y_i``
4. Measurement anomalies: ``Y' = [y_1 - \\bar{y}, \\ldots, y_N - \\bar{y}]``
5. Kalman gain: ``K = X'(Y')^T (Y'(Y')^T / (N-1) + R_2)^{-1}``
6. Perturbed observations: ``y_i^{pert} = y + \\varepsilon_i`` where ``\\varepsilon_i \\sim \\mathcal{N}(0, R_2)``
7. Update: ``x_i^+ = x_i^- + K(y_i^{pert} - y_i)``

# Augmented EnKF
If the noise is not additive, the augmented form of the EnKF may be used. This form is enabled by the typed constructor
```
EnsembleKalmanFilter{augmented_dynamics, augmented_measurement}(...)
```
where the Boolean type parameters have the following meaning
- `augmented_dynamics`: If `true`, the dynamics function takes the process noise as an additional argument, i.e., `dynamics(x, u, p, t, w)`. Default is `false`.
- `augmented_measurement`: If `true`, the measurement function takes the measurement noise as an additional argument, i.e., `measurement(x, u, p, t, e)`. Default is `false`.

The function signatures are the same as those used by the augmented [`UnscentedKalmanFilter`](@ref), i.e., the same functions may be used with both `EnsembleKalmanFilter{AUGD,AUGM}` and `UnscentedKalmanFilter{false,false,AUGD,AUGM}`. The dimensions of the noise vectors are given by the sizes of `R1` and `R2`, and may differ from the state and output dimensions. This allows, e.g., process noise that affects only a subset of the state variables without requiring a singular `R1`.

With augmented dynamics, the predict step is
```math
x_i^- = f(x_i, u, p, t, w_i) \\quad \\text{where } w_i \\sim \\mathcal{N}(0, R_1)
```
With augmented measurement, the correct step uses perturbed predicted observations rather than perturbed observations:
1. Predicted measurements: ``y_i = h(x_i, u, p, t, e_i)`` where ``e_i \\sim \\mathcal{N}(0, R_2)``
2. Innovation covariance: ``S = Y'(Y')^T / (N-1)``, where ``R_2`` is not added since the measurement noise is already present in the predicted measurements
3. Kalman gain: ``K = X'(Y')^T / (N-1) \\, S^{-1}``
4. Update: ``x_i^+ = x_i^- + K(y - y_i)``

For additive measurement noise, this update coincides in expectation with the perturbed-observation update. Since ``S`` is a sample covariance, it is singular if the ensemble size satisfies ``N - 1 < n_y``, and it may be ill-conditioned if the measurement noise does not affect all outputs while the spread of the ensemble in the output space is small. The log-likelihood is in this case a stochastic estimate that depends on the sampled measurement noise.

# Example
```julia
using LowLevelParticleFilters, LinearAlgebra, Distributions

nx, nu, ny = 2, 1, 1
N = 100  # Number of ensemble members

# Linear system
A = [0.9 0.1; 0.0 0.95]
B = [0.0; 1.0;;]
C = [1.0 0.0]

dynamics(x, u, p, t) = A*x + B*u
measurement(x, u, p, t) = C*x

R1 = 0.01*I(nx)
R2 = 0.1*I(ny)
d0 = MvNormal(zeros(nx), I(nx))

enkf = EnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; nu, ny)

# Use like other filters
u, y = [randn(nu)], [randn(ny)]
enkf(u[1], y[1])  # One filtering step
x̂ = state(enkf)  # Ensemble mean
P = covariance(enkf)  # Sample covariance

# Augmented form: scalar process noise entering through the input matrix and
# multiplicative measurement noise
dynamics_aug(x, u, p, t, w) = A*x + B*u + B*w
measurement_aug(x, u, p, t, e) = (C*x) .* (1 .+ e)
R1_aug = 0.01*I(1)
enkf_aug = EnsembleKalmanFilter{true,true}(dynamics_aug, measurement_aug, R1_aug, R2, d0, N; nu, ny)
enkf_aug(u[1], y[1])
```

See also [`UnscentedKalmanFilter`](@ref), [`ParticleFilter`](@ref)
"""
mutable struct EnsembleKalmanFilter{AUGD,AUGM,DT,MT,R1T,R2T,D0T,ET,XT,RT,P,RNGT} <: AbstractKalmanFilter
    dynamics::DT
    measurement::MT
    R1::R1T
    R2::R2T
    d0::D0T
    ensemble::ET
    x::XT      # Cached ensemble mean
    R::RT      # Cached sample covariance
    t::Int
    Ts::Float64
    ny::Int
    nu::Int
    nx::Int
    p::P
    rng::RNGT
    inflation::Float64
    threads::Bool
    names::SignalNames
end

function EnsembleKalmanFilter{AUGD,AUGM}(
    dynamics,
    measurement,
    R1,
    R2,
    d0,
    N::Integer;
    nu::Int,
    ny::Int = size(R2, 1),
    p = NullParameters(),
    Ts = 1.0,
    inflation = 1.0,
    rng = Random.Xoshiro(),
    threads = false,
    names = default_names(length(d0), nu, ny, "EnKF")
) where {AUGD,AUGM}
    nx = length(d0)
    inflation >= 1.0 || @warn "Inflation factor should be ≥ 1.0 to prevent filter divergence. Values < 1.0 will shrink the ensemble spread."

    # Initialize ensemble by sampling from initial distribution
    ensemble = [rand(rng, d0) for _ in 1:N]

    # Compute initial cached mean and covariance
    x0 = _ensemble_mean(ensemble)
    R0 = _ensemble_cov(ensemble, x0)

    R1 = R1 isa AbstractMatrix ? PDMats.PDMat(R1) : R1
    R2 = R2 isa AbstractMatrix ? PDMats.PDMat(R2) : R2

    EnsembleKalmanFilter{AUGD,AUGM,typeof(dynamics),typeof(measurement),typeof(R1),typeof(R2),
        typeof(d0),typeof(ensemble),typeof(x0),typeof(R0),typeof(p),typeof(rng)}(
        dynamics,
        measurement,
        R1,
        R2,
        d0,
        ensemble,
        x0,
        R0,
        0,
        Ts,
        ny,
        nu,
        nx,
        p,
        rng,
        inflation,
        threads,
        names
    )
end

function EnsembleKalmanFilter(dynamics, measurement, args...; kwargs...)
    AUGD = false
    AUGM = false
    EnsembleKalmanFilter{AUGD,AUGM}(dynamics, measurement, args...; kwargs...)
end

# Internal helper functions to compute ensemble statistics
function _ensemble_mean(ensemble)
    N = length(ensemble)
    x̄ = copy(ensemble[1])
    for i in 2:N
        @bangbang x̄ .+= ensemble[i]
    end
    @bangbang x̄ ./= N
    x̄
end

function _ensemble_cov(ensemble, x̄)
    N = length(ensemble)
    nx = length(x̄)
    R = zeros(eltype(x̄), nx, nx)
    for i in 1:N
        δx = ensemble[i] .- x̄
        mul!(R, δx, δx', 1, 1)
    end
    @bangbang R ./= (N - 1)
    R
end

# Update cached x and R from ensemble
function _update_ensemble_stats!(enkf::EnsembleKalmanFilter)
    enkf.x = _ensemble_mean(enkf.ensemble)
    enkf.R = _ensemble_cov(enkf.ensemble, enkf.x)
    nothing
end

# Accessor functions
num_particles(enkf::EnsembleKalmanFilter) = length(enkf.ensemble)
particles(enkf::EnsembleKalmanFilter) = enkf.ensemble
parameters(enkf::EnsembleKalmanFilter) = enkf.p
index(enkf::EnsembleKalmanFilter) = enkf.t
dynamics(enkf::EnsembleKalmanFilter) = enkf.dynamics
measurement(enkf::EnsembleKalmanFilter) = enkf.measurement

"""
    state(enkf::EnsembleKalmanFilter)

Return the cached ensemble mean (state estimate).
"""
state(enkf::EnsembleKalmanFilter) = enkf.x

"""
    covariance(enkf::EnsembleKalmanFilter)

Return the cached sample covariance computed from the ensemble.
"""
covariance(enkf::EnsembleKalmanFilter) = enkf.R

"""
    reset!(enkf::EnsembleKalmanFilter; x0 = nothing)

Reset the ensemble to the initial distribution. If `x0` is provided, the ensemble
is resampled around that mean.
"""
function reset!(enkf::EnsembleKalmanFilter; x0 = nothing, t = 0)
    N = num_particles(enkf)
    if x0 === nothing
        for i in 1:N
            enkf.ensemble[i] = rand(enkf.rng, enkf.d0)
        end
    else
        # Sample around provided x0 with initial covariance
        Σ = enkf.d0.Σ
        d_new = SimpleMvNormal(x0, Σ)
        for i in 1:N
            enkf.ensemble[i] = rand(enkf.rng, d_new)
        end
    end
    enkf.t = t
    _update_ensemble_stats!(enkf)
    nothing
end

"""
    predict!(enkf::EnsembleKalmanFilter, u, p = parameters(enkf), t = index(enkf) * enkf.Ts; R1 = enkf.R1, inflation = enkf.inflation)

Propagate each ensemble member through the dynamics with process noise. If the dynamics is augmented, the noise sample is passed as the last argument to the dynamics function, otherwise it is added to the output of the dynamics function.
"""
function predict!(
    enkf::EnsembleKalmanFilter{AUGD},
    u,
    p = parameters(enkf),
    t::Real = index(enkf) * enkf.Ts;
    R1 = get_mat(enkf.R1, enkf.x, u, p, t),
    inflation = enkf.inflation
) where AUGD
    f = dynamics(enkf)
    N = num_particles(enkf)

    # Create distribution for process noise
    d_w = SimpleMvNormal(PDMats.PDMat(R1))

    # Pre-generate noise samples
    noise_samples = [rand(enkf.rng, d_w) for _ in 1:N]

    # Propagate each ensemble member
    if enkf.threads
        Threads.@threads for i in 1:N
            xi = enkf.ensemble[i]
            @inbounds enkf.ensemble[i] = AUGD ? f(xi, u, p, t, noise_samples[i]) : f(xi, u, p, t) .+ noise_samples[i]
        end
    else
        for i in 1:N
            xi = enkf.ensemble[i]
            @inbounds enkf.ensemble[i] = AUGD ? f(xi, u, p, t, noise_samples[i]) : f(xi, u, p, t) .+ noise_samples[i]
        end
    end

    # Apply covariance inflation if > 1.0
    if inflation > 1.0
        x̄ = _ensemble_mean(enkf.ensemble)  # Compute fresh mean after propagation
        for i in 1:N
            enkf.ensemble[i] = x̄ .+ inflation .* (enkf.ensemble[i] .- x̄)
        end
    end

    enkf.t += 1
    _update_ensemble_stats!(enkf)  # Update cached x and R
    nothing
end

"""
    (; ll, e, S, Sᵪ, K) = correct!(enkf::EnsembleKalmanFilter, u, y, p = parameters(enkf), t = index(enkf) * enkf.Ts; R2 = enkf.R2)

Perform the Stochastic EnKF measurement update with perturbed observations. If the measurement is augmented, the measurement noise is instead included in the predicted measurements `h(xᵢ, u, p, t, eᵢ)`, and the members are updated using the unperturbed measurement `y`.

Returns log-likelihood `ll`, innovation `e`, innovation covariance `S`,
its Cholesky factor `Sᵪ`, and Kalman gain `K`.
"""
function correct!(
    enkf::EnsembleKalmanFilter{<:Any,AUGM},
    u,
    y,
    p = parameters(enkf),
    t::Real = index(enkf) * enkf.Ts;
    R2 = get_mat(enkf.R2, enkf.x, u, p, t)
) where AUGM
    h = measurement(enkf)
    N = num_particles(enkf)
    nx = enkf.nx
    ny = enkf.ny

    # Measurement noise samples for the augmented measurement function
    noise_samples = if AUGM
        d_e = SimpleMvNormal(PDMats.PDMat(R2))
        [rand(enkf.rng, d_e) for _ in 1:N]
    else
        nothing
    end

    # Compute predicted measurements for each ensemble member
    X = enkf.ensemble
    Y = Matrix{promote_type(eltype(y), eltype(X[1]))}(undef, ny, N)
    Xa = Matrix{eltype(X[1])}(undef, nx, N) # nx × N (anomaly matrix)
    if enkf.threads
        Threads.@threads for i in 1:N
            @inbounds Y[:, i] = AUGM ? h(X[i], u, p, t, noise_samples[i]) : h(X[i], u, p, t)
        end
    else
        for i in 1:N
            Y[:, i] = AUGM ? h(X[i], u, p, t, noise_samples[i]) : h(X[i], u, p, t)
        end
    end

    # Compute means
    x̄ = mean(X)  # nx
    ȳ = vec(mean(Y, dims=2))  # ny

    # Compute anomalies
    for i = 1:N
        Xa[:, i] = X[i] .- x̄
    end
    Ya = Y .- ȳ  # ny × N (measurement anomaly matrix)

    # Compute innovation covariance: S = Ya * Ya' / (N-1) + R2, where R2 is omitted
    # for augmented measurements since Y already contains the measurement noise
    S = (Ya * Ya') ./ (N - 1)
    if !AUGM
        S = S .+ R2
    end
    S = symmetrize(S)

    # Cholesky factorization
    Sᵪ = cholesky(Symmetric(S); check=false)
    if !issuccess(Sᵪ)
        hint = AUGM ? " With augmented measurement, S is the sample covariance of the predicted measurements, which is singular if the ensemble size is smaller than ny + 1 or if the measurement noise does not affect all outputs." : ""
        error("Cholesky factorization of innovation covariance failed at time step $(enkf.t), got S = $(printarray(S)).$hint")
    end

    # Compute Kalman gain: K = Xa * Ya' / (N-1) * inv(S)
    # Cross-covariance: Pxy = Xa * Ya' / (N-1)
    Rxy = (Xa * Ya')
    Rxy ./= (N - 1)
    K = Rxy / Sᵪ  # nx × ny

    # Innovation (for mean)
    e = y .- ȳ

    # Distribution for perturbed observations, not used for augmented measurements
    d_ε = AUGM ? nothing : SimpleMvNormal(PDMats.PDMat(R2))

    # Update each ensemble member with perturbed observations
    for i in 1:N
        yi_pert = AUGM ? y : y .+ rand(enkf.rng, d_ε)
        yi_pred = Y[:, i]
        if eltype(X) <: SVector
            X[i] = X[i] + K*(yi_pert .- yi_pred)
        else
            mul!(X[i], K, (yi_pert .- yi_pred), 1, 1)
        end
    end

    # Compute log-likelihood
    ll = extended_logpdf(SimpleMvNormal(PDMat(S, Sᵪ)), e)

    _update_ensemble_stats!(enkf)  # Update cached x and R

    (; ll, e, S, Sᵪ, K)
end

"""
    update!(enkf::EnsembleKalmanFilter, u, y, p = parameters(enkf), t = index(enkf) * enkf.Ts)

Perform one filtering step: correct followed by predict.
"""
function update!(enkf::EnsembleKalmanFilter, u, y, p = parameters(enkf), t::Real = index(enkf) * enkf.Ts)
    if y === missing || ismissing_measurement(y)
        ll_e = missing_correction(enkf)
        predict!(enkf, u, p, t)
        return ll_e
    end
    ll_e = correct!(enkf, u, y, p, t)
    predict!(enkf, u, p, t)
    ll_e
end

# Make the filter callable
(enkf::EnsembleKalmanFilter)(u, y, p = parameters(enkf), t = index(enkf) * enkf.Ts) = update!(enkf, u, y, p, t)

# Sampling functions for simulation
sample_state(enkf::EnsembleKalmanFilter, p = parameters(enkf); noise = true) = noise ? rand(enkf.rng, enkf.d0) : mean(enkf.d0)

function sample_state(enkf::EnsembleKalmanFilter, x, u, p = parameters(enkf), t = 0; noise = true)
    enkf.dynamics(x, u, p, t) .+ noise .* rand(enkf.rng, SimpleMvNormal(get_mat(enkf.R1, x, u, p, t)))
end

function sample_state(enkf::EnsembleKalmanFilter{true}, x, u, p = parameters(enkf), t = 0; noise = true)
    enkf.dynamics(x, u, p, t, noise .* rand(enkf.rng, SimpleMvNormal(get_mat(enkf.R1, x, u, p, t))))
end

function sample_measurement(enkf::EnsembleKalmanFilter, x, u, p = parameters(enkf), t = 0; noise = true)
    enkf.measurement(x, u, p, t) .+ noise .* rand(enkf.rng, SimpleMvNormal(get_mat(enkf.R2, x, u, p, t)))
end

function sample_measurement(enkf::EnsembleKalmanFilter{<:Any,true}, x, u, p = parameters(enkf), t = 0; noise = true)
    enkf.measurement(x, u, p, t, noise .* rand(enkf.rng, SimpleMvNormal(get_mat(enkf.R2, x, u, p, t))))
end

# For compatibility with particle filter interface
particletype(enkf::EnsembleKalmanFilter) = eltype(enkf.ensemble)
covtype(enkf::EnsembleKalmanFilter) = Matrix{eltype(eltype(enkf.ensemble))}

# Display
function Base.show(io::IO, enkf::EnsembleKalmanFilter)
    print(io, "EnsembleKalmanFilter(nx=$(enkf.nx), nu=$(enkf.nu), ny=$(enkf.ny), N=$(num_particles(enkf)))")
end

function Base.show(io::IO, ::MIME"text/plain", enkf::EnsembleKalmanFilter{AUGD,AUGM}) where {AUGD,AUGM}
    println(io, "EnsembleKalmanFilter{$AUGD,$AUGM}")
    println(io, "  State dimension: $(enkf.nx)")
    println(io, "  Input dimension: $(enkf.nu)")
    println(io, "  Output dimension: $(enkf.ny)")
    println(io, "  Augmented dynamics: $AUGD")
    println(io, "  Augmented measurement: $AUGM")
    println(io, "  Ensemble size: $(num_particles(enkf))")
    println(io, "  Inflation factor: $(enkf.inflation)")
    println(io, "  Threaded: $(enkf.threads)")
    println(io, "  Current time index: $(enkf.t)")
end
