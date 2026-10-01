# Multi-step prediction errors
# The predictions are computed by repeated application of the noise-free dynamics of the filter, starting from the filtered estimate x(k|k).

"""
    NoiseFreeDynamics{IP}(f, w0 = nothing)

Out-of-place callable `(x, u, p, t) -> x⁺` that evaluates the dynamics `f` of a filter without noise. `IP` indicates whether `f` is in-place. If `w0` is not `nothing`, the dynamics are augmented with a noise argument and `w0` (zero) is passed as the noise.
"""
struct NoiseFreeDynamics{IP, F, W}
    f::F
    w0::W
end
NoiseFreeDynamics{IP}(f, w0 = nothing) where IP = NoiseFreeDynamics{IP, typeof(f), typeof(w0)}(f, w0)

function (d::NoiseFreeDynamics{IP})(x, u, p, t) where IP
    if IP
        xp = similar(x)
        fill!(xp, 0)
        d.w0 === nothing ? d.f(xp, x, u, p, t) : d.f(xp, x, u, p, t, d.w0)
        return xp
    else
        return d.w0 === nothing ? d.f(x, u, p, t) : d.f(x, u, p, t, d.w0)
    end
end

"""
    NoiseFreeMeasurement{IP}(g, e0, ny)

Out-of-place callable `(x, u, p, t) -> y` that evaluates the measurement function `g` of a filter without noise, see [`NoiseFreeDynamics`](@ref).
"""
struct NoiseFreeMeasurement{IP, G, E}
    g::G
    e0::E
    ny::Int
end
NoiseFreeMeasurement{IP}(g, e0, ny) where IP = NoiseFreeMeasurement{IP, typeof(g), typeof(e0)}(g, e0, ny)

function (m::NoiseFreeMeasurement{IP})(x, u, p, t) where IP
    if IP
        y = similar(x, m.ny)
        fill!(y, 0)
        m.e0 === nothing ? m.g(y, x, u, p, t) : m.g(y, x, u, p, t, m.e0)
        return y
    else
        return m.e0 === nothing ? m.g(x, u, p, t) : m.g(x, u, p, t, m.e0)
    end
end

struct LinearDynamics{K}
    kf::K
end

# Mirrors predict!(kf::AbstractKalmanFilter, ...)
function (d::LinearDynamics)(x, u, p, t)
    kf = d.kf
    At = get_mat(kf.A, x, u, p, t)
    if u === nothing || length(u) == 0
        At*x
    else
        At*x .+ get_mat(kf.B, x, u, p, t)*u |> vec
    end
end

_zero_noise(R::AbstractMatrix) = 0*R[:, 1]
_zero_noise(R) = throw(ArgumentError("Multi-step prediction with augmented noise inputs requires the noise covariance to be a matrix, got $(typeof(R))."))

multistep_unsupported(f) = throw(ArgumentError("Multi-step prediction errors are not supported for filters of type $(nameof(typeof(f))). Supported filters are KalmanFilter, SqKalmanFilter, ExtendedKalmanFilter, SqExtendedKalmanFilter, IteratedExtendedKalmanFilter, UnscentedKalmanFilter, EnsembleKalmanFilter and MUKF with out-of-place dynamics."))

noisefree_dynamics(f) = multistep_unsupported(f)
noisefree_measurement(f) = multistep_unsupported(f)
innovation_function(f) = -

noisefree_dynamics(f::Union{KalmanFilter, SqKalmanFilter}) = LinearDynamics(f)
noisefree_measurement(f::Union{KalmanFilter, SqKalmanFilter}) = measurement(f)

noisefree_dynamics(f::AbstractExtendedKalmanFilter{IPD}) where IPD = NoiseFreeDynamics{IPD}(f.dynamics)
function noisefree_measurement(f::AbstractExtendedKalmanFilter)
    mm = f.measurement_model
    NoiseFreeMeasurement{isinplace(mm)}(measurement(mm), nothing, f.ny)
end

function noisefree_dynamics(f::UnscentedKalmanFilter{IPD,<:Any,AUGD}) where {IPD, AUGD}
    w0 = AUGD ? zero(f.predict_sigma_point_cache.x0[1][length(f.x)+1:end]) : nothing
    NoiseFreeDynamics{IPD}(f.dynamics, w0)
end
function noisefree_measurement(f::UnscentedKalmanFilter)
    mm = f.measurement_model
    if mm isa UKFMeasurementModel{<:Any, true}
        e0 = zero(mm.cache.x0[1][length(f.x)+1:end])
    else
        e0 = nothing
    end
    NoiseFreeMeasurement{isinplace(mm)}(measurement(mm), e0, f.ny)
end
innovation_function(f::UnscentedKalmanFilter) = f.measurement_model isa UKFMeasurementModel ? f.measurement_model.innovation : -

function noisefree_dynamics(f::EnsembleKalmanFilter{AUGD}) where AUGD
    NoiseFreeDynamics{false}(f.dynamics, AUGD ? _zero_noise(f.R1) : nothing)
end
function noisefree_measurement(f::EnsembleKalmanFilter{<:Any, AUGM}) where AUGM
    NoiseFreeMeasurement{false}(f.measurement, AUGM ? _zero_noise(f.R2) : nothing, f.ny)
end

noisefree_dynamics(f::MUKF{false}) = NoiseFreeDynamics{false}(dynamics(f))
noisefree_measurement(f::MUKF{false}) = NoiseFreeMeasurement{false}(measurement(f), nothing, f.ny)

"""
    acc = multistep_fold(op, acc, f, u, y, p, h)

Run the filter `f` along the trajectory `u, y`. After the correction step at time step `k`, the outputs at time steps `k+1, ..., k+h` are predicted using the noise-free dynamics, and `acc = op(acc, k, j, e)` is called for each prediction error `e = y[k+j] - ŷ(k+j|k)`, where the measurement `y[k+j]` is not missing.
"""
function multistep_fold(op, acc, f, u, y, p, h)
    length(u) == length(y) || throw(ArgumentError("u and y must have the same length"))
    fx = noisefree_dynamics(f)
    gx = noisefree_measurement(f)
    innovation = innovation_function(f)
    reset!(f)
    N = length(y)
    Ts = f.Ts
    for k in 1:N
        tk = index(f)*Ts
        yk = y[k]
        yk === missing || ismissing_measurement(yk) || correct!(f, u[k], yk, p, tk)
        xj = state(f)
        for j in 1:min(h, N-k)
            i = k + j
            # x(k+j|k) = f(x(k+j-1|k), u(k+j-1), p, t(k+j-1))
            xj = fx(xj, u[i-1], p, tk + (j-1)*Ts)
            yi = y[i]
            (yi === missing || ismissing_measurement(yi)) && continue
            acc = op(acc, k, j, innovation(yi, gx(xj, u[i], p, tk + j*Ts)))
        end
        predict!(f, u[k], p, tk)
    end
    acc
end

function check_multistep_arguments(h, horizon_weights)
    h >= 1 || throw(ArgumentError("The prediction horizon h must be at least 1, got h = $h"))
    if horizon_weights !== nothing
        length(horizon_weights) == h || throw(ArgumentError("horizon_weights must have length h = $h, got length $(length(horizon_weights))"))
        all(>=(0), horizon_weights) || throw(ArgumentError("horizon_weights must be non-negative"))
    end
    nothing
end

"""
    multistep_prediction_errors!(res, f, u, y, p = parameters(f), λ = 1; h, horizon_weights = nothing)

Calculate the multi-step prediction errors of the filter `f` along the trajectory `u, y` and store them in `res`. This function is useful for estimation of model parameters with Gauss-Newton type optimizers, such as `LevenbergMarquardt` from LeastSquaresOptim.jl, when the model is to be used for prediction over a horizon of `h` time steps, for example, as the prediction model of an MPC controller, or when the dynamics are slow compared to the sample interval.

For each time step `k`, the filter is first corrected with the measurement `y[k]`, yielding the filtered estimate ``x(k|k)``. The noise-free dynamics are then applied repeatedly with the measured inputs to obtain the predictions
```math
\\hat x(k+j|k) = f(\\hat x(k+j-1|k), u(k+j-1), p, t_{k+j-1}), \\quad j = 1, \\dots, h
```
and the prediction errors ``e(k+j|k) = y(k+j) - g(\\hat x(k+j|k), u(k+j), p, t_{k+j})`` are formed. Finally, the filter performs a regular prediction step to ``x(k+1|k)``. The time of time step `k` is ``t_k = (k-1) T_s``.

With `h = 1`, the prediction errors coincide with those computed by [`LowLevelParticleFilters.prediction_errors!`](@ref) shifted by one time step, exactly for `KalmanFilter`, `SqKalmanFilter` and the extended Kalman filters, and approximately for sigma-point and ensemble filters, for which the one-step prediction of the filter is a sigma-point or ensemble mean rather than the propagated mean. As `h` increases, the criterion approaches a simulation-error criterion. The noise covariance `R1` of the filter determines how strongly the measurements correct the state estimate from which each prediction starts.

# Arguments
- `res`: A vector of length `length(y)*h*ny`. The prediction error ``e(k+j|k)`` is stored at the indices `((k-1)*h + j-1)*ny .+ (1:ny)`, multiplied by `sqrt(horizon_weights[j])*sqrt(λ)`. Entries for which `k + j > length(y)`, or for which `y[k+j]` is missing, are zero.
- `f`: A Kalman-type filter, see below.
- `λ`: A weighting factor or matrix, see [`LowLevelParticleFilters.prediction_errors!`](@ref).
- `h`: The prediction horizon (number of time steps).
- `horizon_weights`: An optional vector of length `h` with non-negative weights for each prediction step.

The sum of squares of `res` is not a log-likelihood, since the prediction errors of overlapping horizons are correlated. Noise covariances should thus not be estimated using this criterion, use, e.g., [`loglik`](@ref) or [`autotune_covariances`](@ref) with fixed model parameters for that purpose. The computational cost is approximately `h` evaluations of the dynamics and measurement functions per time step in addition to the cost of filtering.

Supported filters are `KalmanFilter`, `SqKalmanFilter`, `ExtendedKalmanFilter`, `SqExtendedKalmanFilter`, `IteratedExtendedKalmanFilter`, `UnscentedKalmanFilter` (including augmented dynamics and measurement models), `EnsembleKalmanFilter` (the result is random since the filter is stochastic) and `MUKF` with out-of-place dynamics. Missing measurements are handled as described in [`LowLevelParticleFilters.ismissing_measurement`](@ref). As for the other cost functions, the element type of the initial state distribution `d0` must be able to represent the element type of the parameters when, e.g., ForwardDiff is used to compute gradients.

See also [`LowLevelParticleFilters.multistep_sse`](@ref).
"""
function multistep_prediction_errors!(res, f::AbstractFilter, u, y, p = parameters(f), λ = 1; h::Integer, horizon_weights = nothing)
    check_prediction_error_support(f, "multistep_prediction_errors!")
    check_multistep_arguments(h, horizon_weights)
    N = length(y)
    ny = f.ny
    length(res) == N*h*ny || throw(ArgumentError("The residual vector must have length length(y)*h*ny = $(N*h*ny), got $(length(res))"))
    W = sqrt(λ)
    fill!(res, 0)
    multistep_fold(nothing, f, u, y, p, h) do acc, k, j, e
        inds = ((k-1)*h + j-1)*ny .+ (1:ny)
        if horizon_weights === nothing
            @views res[inds] .= W*e
        else
            @views res[inds] .= sqrt(horizon_weights[j]) .* (W*e)
        end
        acc
    end
    res
end

"""
    multistep_sse(f, u, y, p = parameters(f), λ = 1; h, horizon_weights = nothing)

Calculate the sum of squared multi-step prediction errors ``\\sum_k \\sum_{j=1}^{h} w_j \\, e(k+j|k)^T λ \\, e(k+j|k)``, where `w = horizon_weights` (all ones by default). See [`LowLevelParticleFilters.multistep_prediction_errors!`](@ref) for the definition of the prediction errors and the supported filters. The sum is not normalized and grows with the horizon `h`.
"""
function multistep_sse(f::AbstractFilter, u, y, p = parameters(f), λ = 1; h::Integer, horizon_weights = nothing)
    check_prediction_error_support(f, "multistep_sse")
    check_multistep_arguments(h, horizon_weights)
    acc0 = zero(float(eltype(state(f))))
    multistep_fold(acc0, f, u, y, p, h) do acc, k, j, e
        c = dot(e, λ, e)
        acc + (horizon_weights === nothing ? c : horizon_weights[j]*c)
    end
end
