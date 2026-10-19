## Quantized measurement model ==================================================

"""
    QuantizedMeasurementModel(inner, Δ; band = (y, Δ) -> (y - Δ/2, y + Δ/2))

A measurement model for measurements that are quantized with step `Δ` after the addition of the
measurement noise,
```math
y_i = Δ_i \\operatorname{round}\\big(z_i / Δ_i\\big), \\quad z = h(x, u, p, t) + e,
```
where the output `z` before quantization, including its noise, is described by the measurement
model `inner`, a [`LinearMeasurementModel`](@ref), [`EKFMeasurementModel`](@ref) or
[`UKFMeasurementModel`](@ref). A measurement `y_i` states that `z_i` lies in the band
`band(y_i, Δ_i) = (lo, hi)`, by default `(y_i - Δ_i/2, y_i + Δ_i/2)`.

The correction step processes the outputs one at a time. For each output, the predicted mean and
variance of `z_i` and its cross-covariance with the state are computed by `inner` at the current
estimate (exactly for a linear model, by linearization or by the unscented transform otherwise).
The distribution of `z_i` given the measurement is a normal distribution truncated to the band;
its mean and variance determine the updated mean and covariance of the state (assumed-density
filtering, or moment matching). The log-likelihood returned by `correct!` is the sum of the
logarithms of the predicted probabilities of the bands. The quantizer thus never enters a
Jacobian or a set of sigma points, the derivatives of which vanish almost everywhere.

# Arguments
- `inner`: The measurement model of the output before quantization. Its measurement-noise
  covariance must be diagonal, since the outputs are quantized independently; a cross-covariance
  `R12` with the dynamics noise is not supported.
- `Δ`: The quantization step, a number or a vector with one step per output. An output with
  `Δ_i = 0` is not quantized and receives the standard Kalman update, which allows quantized and
  unquantized outputs in one model.
- `band`: A function `(y_i, Δ_i) -> (lo, hi)` with the interval of `z_i` that corresponds to the
  measurement `y_i`. The bounds may be infinite, e.g., for a sensor that saturates. For a
  quantizer that truncates, `y = Δ floor(z/Δ)`, use `band = (y, Δ) -> (y, y + Δ)`.

The values returned by `correct!` are
- `ll`: the log-likelihood of the measurement, the sum of the logarithms of the band probabilities,
- `e`: the difference between the measurement and the predicted output before quantization,
- `S`: the predicted covariance of the output before quantization, and `Sᵪ` its Cholesky factorization,
- `K`: the gain of an update by the output before quantization, `Pxz / S`, which differs from the
  update applied,
- `lls`: the contribution of each output to `ll`,
- `r`: for each output, the standardized correction `(E[z_i | y_i] - ẑ_i)/σ_i` of the sequential update,
  which is the standardized innovation for an unquantized output,
- `quantized`: for each output, whether it is quantized.

`measurement(model)` returns the measurement function of `inner`, i.e., the output before
quantization. The model may be passed to the constructors of [`ExtendedKalmanFilter`](@ref) and
[`UnscentedKalmanFilter`](@ref), and to `correct!` of any Kalman-type filter.

!!! warning "Cost functions based on prediction errors"
    Sums of squared prediction errors compare the quantized measurement with the predicted output
    before quantization as if their difference were Gaussian, and their minimum is in general not
    located at the true parameters. For a filter with a quantized measurement model,
    [`LowLevelParticleFilters.sse`](@ref), [`LowLevelParticleFilters.prediction_errors!`](@ref) with
    `loglik = false`, [`LowLevelParticleFilters.multistep_sse`](@ref) and
    [`LowLevelParticleFilters.multistep_prediction_errors!`](@ref) therefore throw an error. Use
    [`loglik`](@ref), or `prediction_errors!` with `loglik = true`, which returns the residual
    ``\\sqrt{-\\log P_i}`` for each quantized output, with ``P_i`` the predicted probability of its
    band, such that the sum of squared residuals is the negative log-likelihood.

See the tutorial [Quantized measurements](@ref).
"""
struct QuantizedMeasurementModel{M <: AbstractMeasurementModel, DT, BT, RT} <: AbstractMeasurementModel
    inner::M
    Δ::DT
    band::BT
    ny::Int
    R2::RT
end

default_quantization_band(y, Δ) = (y - Δ/2, y + Δ/2)

function QuantizedMeasurementModel(inner::AbstractMeasurementModel, Δ; band = default_quantization_band)
    inner isa Union{LinearMeasurementModel, EKFMeasurementModel, UKFMeasurementModel} ||
        throw(ArgumentError("The inner measurement model of a QuantizedMeasurementModel must be a LinearMeasurementModel, EKFMeasurementModel or UKFMeasurementModel, got $(nameof(typeof(inner)))."))
    hasproperty(inner, :R12) && inner.R12 !== nothing &&
        throw(ArgumentError("A cross-covariance R12 between the dynamics and measurement noise is not supported by QuantizedMeasurementModel."))
    R2 = inner.R2
    R2 isa AbstractMatrix && !isdiag(R2) &&
        throw(ArgumentError("The measurement-noise covariance of the inner model of a QuantizedMeasurementModel must be diagonal, since each output is quantized independently."))
    Δ isa AbstractVector && length(Δ) != inner.ny &&
        throw(ArgumentError("The number of quantization steps ($(length(Δ))) must equal the number of outputs ($(inner.ny))."))
    all(>=(0), Δ) || throw(ArgumentError("The quantization step must be non-negative."))
    QuantizedMeasurementModel(inner, Δ, band, inner.ny, R2)
end

isinplace(model::QuantizedMeasurementModel) = isinplace(model.inner)
has_oop(model::QuantizedMeasurementModel) = !isinplace(model.inner)
measurement(model::QuantizedMeasurementModel) = measurement(model.inner)

quantization_step(Δ::Number, i) = Δ
quantization_step(Δ, i) = Δ[i]

# Logarithm of the standard normal distribution function, accurate in the lower tail
function normlogcdf_q(x)
    if x < 0
        log(SpecialFunctions.erfcx(-x/sqrt(2))/2) - x^2/2
    else
        log1p(-SpecialFunctions.erfc(x/sqrt(2))/2)
    end
end
normlogpdf_q(x) = -x^2/2 - log(2π)/2

"""
    lz, m, v = truncated_normal_moments(α, β)

Logarithm of the probability `P(α < s < β)`, and mean and variance of `s ~ N(0, 1)` truncated to
the interval `(α, β)`. The bounds may be infinite.
"""
function truncated_normal_moments(α, β)
    α < β || throw(ArgumentError("The lower bound of the band must be smaller than the upper bound, got ($α, $β)."))
    if α > 0 # Both bounds in the upper tail: use the symmetry of the normal distribution
        lz, m, v = truncated_normal_moments(-β, -α)
        return lz, -m, v
    end
    if β <= 0 # Both bounds in the lower tail
        lb = normlogcdf_q(β)
        lz = lb + log(-expm1(normlogcdf_q(α) - lb))
    else
        lz = log1p(-(normcdf(-β) + normcdf(α)))
    end
    pa = isfinite(α) ? exp(normlogpdf_q(α) - lz) : zero(lz)
    pb = isfinite(β) ? exp(normlogpdf_q(β) - lz) : zero(lz)
    m = pa - pb
    αpa = isfinite(α) ? α*pa : zero(lz)
    βpb = isfinite(β) ? β*pb : zero(lz)
    v = 1 + αpa - βpb - m^2
    lz, m, min(max(v, 1e-12), one(v))
end

"""
    ŷ, S, Pxz = output_moments(kf, model, u, p, t, R2)

Predicted mean `ŷ` and covariance `S` of the output of the measurement model `model`, and its
cross-covariance `Pxz` with the state, at the current estimate of the filter `kf`.
"""
function output_moments(kf, model::LinearMeasurementModel, u, p, t, R2)
    (; x, R) = kf
    C = get_mat(model.C, x, u, p, t)
    D = get_mat(model.D, x, u, p, t)
    ŷ = C*x
    iszero(D) || (ŷ = ŷ + D*u)
    Pxz = R*C'
    ŷ, symmetrize(C*Pxz) + R2, Pxz
end

function output_moments(kf, model::EKFMeasurementModel{IPM}, u, p, t, R2) where IPM
    (; x, R) = kf
    C = model.Cjac(x, u, p, t)
    if IPM
        ŷ = zeros(promote_type(eltype(x), eltype(R)), model.ny)
        model.measurement(ŷ, x, u, p, t)
    else
        ŷ = model.measurement(x, u, p, t)
    end
    Pxz = R*C'
    ŷ, symmetrize(C*Pxz) + R2, Pxz
end

function output_moments(kf, model::UKFMeasurementModel, u, p, t, R2)
    cache = model.cache
    sigmapoints_c!(kf, model, R2)
    propagate_sigmapoints_c!(kf, u, p, t, R2, model)
    ŷ = mean_with_weights(model.mean, cache.x1, model.weight_params)
    Pxz = cross_cov_with_weights(model.cross_cov, cache.x0, kf.x, cache.x1, ŷ, model.weight_params)
    ŷ, compute_S(model, R2, ŷ), Pxz
end

function correct!(
    kf::AbstractKalmanFilter,
    model::QuantizedMeasurementModel,
    u,
    y,
    p = parameters(kf),
    t::Real = index(kf) * kf.Ts;
    R2 = get_mat(model.R2, kf.x, u, p, t),
)
    ŷ, S, Pxz = output_moments(kf, model.inner, u, p, t, R2)
    e = y .- ŷ
    Sᵪ = cholesky(Symmetric(S); check = false)
    issuccess(Sᵪ) || error("Cholesky factorization of the covariance of the output before quantization failed at time step $(kf.t), got S = $(printarray(S))")
    K = Pxz / Sᵪ
    T = promote_type(eltype(kf.x), eltype(kf.R), eltype(y))
    ll = zero(T)
    lls = zeros(T, model.ny)
    r = zeros(T, model.ny)
    quantized = [!iszero(quantization_step(model.Δ, i)) for i in 1:model.ny]
    for i in 1:model.ny
        # The moments of the remaining outputs change with the estimate, recompute them after each update
        ŷi, Si, Pxzi = i == 1 ? (ŷ, S, Pxz) : output_moments(kf, model.inner, u, p, t, R2)
        Δi = quantization_step(model.Δ, i)
        σ = sqrt(Si[i, i])
        if iszero(Δi) # Not quantized: the standard update
            ri = (y[i] - ŷi[i])/σ
            lz, m, v = normlogpdf_q(ri) - log(σ), ri, zero(ri)
        else
            lo, hi = model.band(y[i], Δi)
            lz, m, v = truncated_normal_moments((lo - ŷi[i])/σ, (hi - ŷi[i])/σ)
        end
        g = Pxzi[:, i]
        Ki = g / Si[i, i]
        kf.x = kf.x + Ki*(σ*m)
        kf.R = symmetrize(kf.R - (1 - v)*(Ki*g'))
        ll += lz
        lls[i] = lz
        r[i] = m
    end
    (; ll, e, S, Sᵪ, K, lls, r, quantized)
end

has_quantized_measurement(model) = false
has_quantized_measurement(model::QuantizedMeasurementModel) = true
has_quantized_measurement(model::CompositeMeasurementModel) = any(has_quantized_measurement, model.models)

"""
    check_quantization_support(f, fname)

Throw an error if the filter `f` holds a quantized measurement model, for cost functions that treat
the prediction errors as Gaussian.
"""
function check_quantization_support(f, fname)
    if hasfield(typeof(f), :measurement_model) && has_quantized_measurement(getfield(f, :measurement_model))
        throw(ArgumentError("$fname compares the measurements with the predicted outputs as if their difference were Gaussian, which does not hold for a filter with a QuantizedMeasurementModel. Use loglik, or prediction_errors! with loglik = true."))
    end
    nothing
end

"""
    quantized_loglik_residuals!(res, ret, offset)

Residuals of one time step for [`LowLevelParticleFilters.prediction_errors!`](@ref) with
`loglik = true` from the result `ret` of `correct!` with a [`QuantizedMeasurementModel`](@ref):
``\\sqrt{-\\log P_i}`` for a quantized output, the whitened innovation ``r_i/\\sqrt{2}`` for an
unquantized output, and in the last entry the square root of the constant terms of the unquantized
outputs plus `offset`, such that `sum(abs2, res) == -ret.ll + offset`.
"""
function quantized_loglik_residuals!(res, ret, offset)
    ny = length(ret.lls)
    c = offset
    for i in 1:ny
        if ret.quantized[i]
            res[i] = sqrt(max(-ret.lls[i], zero(ret.lls[i])))
        else
            res[i] = ret.r[i]/sqrt(2)
            c += -ret.lls[i] - ret.r[i]^2/2 # log σ + log(2π)/2
        end
    end
    c < 0 && error("Negative value ($c) inside square root when calculating log-likelihood residuals. Increase the offset argument to prediction_errors! (currently set to offset=$offset)")
    res[ny+1] = sqrt(c)
    res
end
