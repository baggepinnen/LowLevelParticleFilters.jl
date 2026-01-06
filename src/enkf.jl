"""
    EnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, N; kwargs...)

An Ensemble Kalman Filter (EnKF) that uses an ensemble of states instead of explicitly
tracking the covariance matrix. This makes it suitable for high-dimensional systems
where covariance matrices become intractable.

This implementation uses the **Stochastic EnKF** formulation with perturbed observations.

# Arguments
- `dynamics`: Dynamics function `f(x, u, p, t) -> x⁺`
- `measurement`: Measurement function `h(x, u, p, t) -> y`
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
```

See also [`UnscentedKalmanFilter`](@ref), [`ParticleFilter`](@ref)
"""
mutable struct EnsembleKalmanFilter{DT,MT,R1T,R2T,D0T,ET,P,RNGT} <: AbstractKalmanFilter
    dynamics::DT
    measurement::MT
    R1::R1T
    R2::R2T
    d0::D0T
    ensemble::ET
    t::Int
    Ts::Float64
    ny::Int
    nu::Int
    nx::Int
    p::P
    rng::RNGT
    inflation::Float64
    names::SignalNames
end

function EnsembleKalmanFilter(
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
    names = default_names(length(d0), nu, ny, "EnKF")
)
    nx = length(d0)
    inflation >= 1.0 || @warn "Inflation factor should be ≥ 1.0 to prevent filter divergence. Values < 1.0 will shrink the ensemble spread."

    # Initialize ensemble by sampling from initial distribution
    ensemble = [rand(rng, d0) for _ in 1:N]

    EnsembleKalmanFilter(
        dynamics,
        measurement,
        R1 isa AbstractMatrix ? PDMats.PDMat(R1) : R1,
        R2 isa AbstractMatrix ? PDMats.PDMat(R2) : R2,
        d0,
        ensemble,
        0,
        Ts,
        ny,
        nu,
        nx,
        p,
        rng,
        inflation,
        names
    )
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

Return the ensemble mean (state estimate).
"""
function state(enkf::EnsembleKalmanFilter)
    N = num_particles(enkf)
    x̄ = copy(enkf.ensemble[1])
    for i in 2:N
        @bangbang x̄ .+= enkf.ensemble[i]
    end
    @bangbang x̄ ./= N
    x̄
end

"""
    covariance(enkf::EnsembleKalmanFilter)

Return the sample covariance computed from the ensemble.
"""
function covariance(enkf::EnsembleKalmanFilter)
    N = num_particles(enkf)
    x̄ = state(enkf)
    nx = enkf.nx

    # Compute sample covariance: (1/(N-1)) * Σ (x_i - x̄)(x_i - x̄)ᵀ
    R = zeros(nx, nx)
    for i in 1:N
        δx = enkf.ensemble[i] .- x̄
        @bangbang R .+= δx * δx'
    end
    @bangbang  R ./= (N - 1)
    R
end

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
    nothing
end

"""
    predict!(enkf::EnsembleKalmanFilter, u, p = parameters(enkf), t = index(enkf) * enkf.Ts; R1 = enkf.R1, inflation = enkf.inflation)

Propagate each ensemble member through the dynamics with process noise.
"""
function predict!(
    enkf::EnsembleKalmanFilter,
    u,
    p = parameters(enkf),
    t::Real = index(enkf) * enkf.Ts;
    R1 = get_mat(enkf.R1, state(enkf), u, p, t),
    inflation = enkf.inflation
)
    f = dynamics(enkf)
    N = num_particles(enkf)
    nx = enkf.nx

    # Create distribution for process noise
    R1pd = PDMats.PDMat(R1)  # Ensure positive-definite matrix type
    d_w = SimpleMvNormal(R1pd)

    # Propagate each ensemble member
    for i in 1:N
        xi = enkf.ensemble[i]
        wi = rand(enkf.rng, d_w)
        enkf.ensemble[i] = f(xi, u, p, t) .+ wi
    end

    if inflation > 1.0
        x̄ = state(enkf)
        for i in 1:N
            enkf.ensemble[i] = x̄ .+ inflation .* (enkf.ensemble[i] .- x̄)
        end
    end

    enkf.t += 1
    nothing
end

"""
    (; ll, e, S, Sᵪ, K) = correct!(enkf::EnsembleKalmanFilter, u, y, p = parameters(enkf), t = index(enkf) * enkf.Ts; R2 = enkf.R2)

Perform the Stochastic EnKF measurement update with perturbed observations.

Returns log-likelihood `ll`, innovation `e`, innovation covariance `S`,
its Cholesky factor `Sᵪ`, and Kalman gain `K`.
"""
function correct!(
    enkf::EnsembleKalmanFilter,
    u,
    y,
    p = parameters(enkf),
    t::Real = index(enkf) * enkf.Ts;
    R2 = get_mat(enkf.R2, state(enkf), u, p, t)
)
    h = measurement(enkf)
    N = num_particles(enkf)
    nx = enkf.nx
    ny = enkf.ny

    # Compute predicted measurements for each ensemble member
    X = enkf.ensemble
    Y = Matrix{eltype(y)}(undef, ny, N)
    Xa = Matrix{eltype(X[1])}(undef, nx, N) # nx × N (anomaly matrix)
    for i in 1:N
        Y[:, i] = h(X[i], u, p, t)
    end

    # Compute means
    x̄ = mean(X)  # nx
    ȳ = vec(mean(Y, dims=2))  # ny

    # Compute anomalies
    for i = 1:N
        Xa[:, i] = X[i] .- x̄
    end
    Ya = Y .- ȳ  # ny × N (measurement anomaly matrix)

    # Compute innovation covariance: S = Ya * Ya' / (N-1) + R2
    S = (Ya * Ya') ./ (N - 1) .+ R2
    S = symmetrize(S)

    # Cholesky factorization
    Sᵪ = cholesky(Symmetric(S); check=false)
    if !issuccess(Sᵪ)
        error("Cholesky factorization of innovation covariance failed at time step $(enkf.t), got S = $(printarray(S))")
    end

    # Compute Kalman gain: K = Xa * Ya' / (N-1) * inv(S)
    # Cross-covariance: Pxy = Xa * Ya' / (N-1)
    Rxy = (Xa * Ya')
    Rxy ./= (N - 1)
    K = Rxy / Sᵪ  # nx × ny

    # Innovation (for mean)
    e = y .- ȳ

    # Distribution for perturbed observations
    d_ε = SimpleMvNormal(PDMats.PDMat(R2))

    # Update each ensemble member with perturbed observations
    for i in 1:N
        εi = rand(enkf.rng, d_ε)
        yi_pert = y .+ εi
        yi_pred = Y[:, i]
        if eltype(X) <: SVector
            X[i] = X[i] + K*(yi_pert .- yi_pred)
        else
            mul!(X[i], K, (yi_pert .- yi_pred), 1, 1)
        end
    end

    # Compute log-likelihood
    ll = extended_logpdf(SimpleMvNormal(PDMat(S, Sᵪ)), e)

    (; ll, e, S, Sᵪ, K)
end

"""
    update!(enkf::EnsembleKalmanFilter, u, y, p = parameters(enkf), t = index(enkf) * enkf.Ts)

Perform one filtering step: correct followed by predict.
"""
function update!(enkf::EnsembleKalmanFilter, u, y, p = parameters(enkf), t::Real = index(enkf) * enkf.Ts)
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

function sample_measurement(enkf::EnsembleKalmanFilter, x, u, p = parameters(enkf), t = 0; noise = true)
    enkf.measurement(x, u, p, t) .+ noise .* rand(enkf.rng, SimpleMvNormal(get_mat(enkf.R2, x, u, p, t)))
end

# For compatibility with particle filter interface
particletype(enkf::EnsembleKalmanFilter) = eltype(enkf.ensemble)
covtype(enkf::EnsembleKalmanFilter) = Matrix{eltype(eltype(enkf.ensemble))}

"""
    forward_trajectory(enkf::EnsembleKalmanFilter, u::AbstractVector, y::AbstractVector, p = parameters(enkf))

Run the EnKF for a sequence of inputs and measurements. Returns a `ParticleFilteringSolution`
with ensemble members and uniform weights.
"""
function forward_trajectory(
    enkf::EnsembleKalmanFilter,
    u::AbstractVector,
    y::AbstractVector,
    p = parameters(enkf);
    t = range(0, step=enkf.Ts, length=length(y))
)
    reset!(enkf)
    T = length(y)
    N = num_particles(enkf)

    # Storage (same format as ParticleFilteringSolution)
    x = Array{particletype(enkf)}(undef, N, T)
    w = zeros(N, T)  # Log weights (uniform = 0)
    we = fill(1.0/N, N, T)  # Exp weights (uniform = 1/N)
    ll = 0.0

    @inbounds for k in 1:T
        ti = t[k]
        ret = correct!(enkf, u[k], y[k], p, ti)
        ll += ret.ll

        # Store current ensemble
        for i in 1:N
            x[i, k] = copy(enkf.ensemble[i])
        end

        predict!(enkf, u[k], p, ti)
    end

    ParticleFilteringSolution(enkf, u, y, x, w, we, ll, t)
end

# Display
function Base.show(io::IO, enkf::EnsembleKalmanFilter)
    print(io, "EnsembleKalmanFilter(nx=$(enkf.nx), nu=$(enkf.nu), ny=$(enkf.ny), N=$(num_particles(enkf)))")
end

function Base.show(io::IO, ::MIME"text/plain", enkf::EnsembleKalmanFilter)
    println(io, "EnsembleKalmanFilter")
    println(io, "  State dimension: $(enkf.nx)")
    println(io, "  Input dimension: $(enkf.nu)")
    println(io, "  Output dimension: $(enkf.ny)")
    println(io, "  Ensemble size: $(num_particles(enkf))")
    println(io, "  Inflation factor: $(enkf.inflation)")
    println(io, "  Current time index: $(enkf.t)")
end
