"""
    correct_observed!(kf, u, y, p=parameters(kf), t=index(kf)*kf.Ts; observed, model)

Condition a `KalmanFilter` or `ExtendedKalmanFilter` on the observed channels of
a full measurement vector. By default, use all entries other than `missing`.
Alternatively, pass distinct integer indices in the desired order or a Boolean
mask. Unselected entries may contain arbitrary values, including `NaN`.
An empty selection skips the correction and contributes zero log likelihood.

`model` defaults to the filter's measurement model and may be a
[`LinearMeasurementModel`](@ref) or [`EKFMeasurementModel`](@ref).
The marginal correction retains correlations among selected measurement errors
and their cross-covariance with process noise. The returned innovation,
innovation covariance and columns of the gain follow the selected channel order.
"""
function correct_observed!(kf::Union{KalmanFilter,ExtendedKalmanFilter}, u, y::AbstractVector,
        p=parameters(kf), t=index(kf)*kf.Ts; observed=findall(!ismissing, y),
        model=_observation_model(kf))
    model isa Union{LinearMeasurementModel,EKFMeasurementModel} ||
        throw(ArgumentError("observed-channel correction requires a linear or EKF measurement model"))
    length(y) == model.ny || throw(DimensionMismatch("expected a full measurement vector of length $(model.ny)"))
    if observed isa AbstractVector{Bool}
        length(observed) == model.ny || throw(DimensionMismatch("mask length must match the measurement"))
        rows = findall(observed)
    else
        observed isa AbstractVector{<:Integer} || throw(ArgumentError("observed must contain channel indices or be a Boolean mask"))
        rows = observed
    end
    isempty(rows) && return missing_correction(kf)
    allunique(rows) || throw(ArgumentError("observed channel indices must be distinct"))
    values = y[rows]
    any(ismissing, values) && throw(ArgumentError("a selected channel is missing"))
    reduced_y = Missing <: eltype(values) ? collect(skipmissing(values)) : values
    x = state(kf)
    R2 = get_mat(model.R2, x, u, p, t)[rows, rows]
    R12 = isnothing(model.R12) ? nothing : get_mat(model.R12, x, u, p, t)[:, rows]
    reduced = _observed_model(model, rows, R2, R12, x, u, p, t)
    correct!(kf, reduced, u, reduced_y, p, t)
end

_observation_model(kf::KalmanFilter) = LinearMeasurementModel(kf.C, kf.D, kf.R2; ny=kf.ny)
_observation_model(kf::ExtendedKalmanFilter) = kf.measurement_model

function _observed_model(model::LinearMeasurementModel, rows, R2, R12, x, u, p, t)
    C = get_mat(model.C, x, u, p, t)[rows, :]
    D = get_mat(model.D, x, u, p, t)
    LinearMeasurementModel(C, iszero(D) ? 0 : D[rows, :], R2; R12)
end

function _observed_model(model::EKFMeasurementModel{IPM}, rows, R2, R12, x, u, p, t) where IPM
    function measurement(x, u, p, t)
        if IPM
            full = similar(x, promote_type(eltype(x), eltype(R2)), model.ny)
            model.measurement(full, x, u, p, t)
            return full[rows]
        end
        model.measurement(x, u, p, t)[rows]
    end
    Cjac = (x,u,p,t) -> model.Cjac(x,u,p,t)[rows, :]
    EKFMeasurementModel{false}(measurement, R2, length(rows), Cjac, R12)
end
