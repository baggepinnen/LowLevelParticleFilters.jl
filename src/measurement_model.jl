abstract type AbstractMeasurementModel end

measurement(model::AbstractMeasurementModel) = model.measurement

struct CompositeMeasurementModel{M} <: AbstractMeasurementModel
    models::M
    ny::Int
    R2
end

"""
    CompositeMeasurementModel(model1, model2, ...)

A composite measurement model that combines multiple measurement models. This model acts as all component models concatenated. The tuple returned from [`correct!`](@ref) will be
- `ll`: The sum of the log-likelihood of all component models
- `e`: The concatenated innovation vector
- `S`: A vector of the innovation covariance matrices of the component models
- `Sᵪ`: A vector of the Cholesky factorizations of the innovation covariance matrices of the component models
- `K`: A vector of the Kalman gains of the component models

If all sensors operate on at the same rate, and all measurement models are of the same type, it's more efficient to use a single measurement model with a vector-valued measurement function.

# Fields:
- `models`: A tuple of measurement models
"""
function CompositeMeasurementModel(m1, rest...)
    models = (m1, rest...)
    ny = sum(m.ny for m in models)
    R2 = cat([m.R2 for m in models]..., dims=(1,2))
    CompositeMeasurementModel(models, ny, R2)
end

isinplace(model::CompositeMeasurementModel) = isinplace(model.models[1])
has_oop(model::CompositeMeasurementModel) = has_oop(model.models[1])

function measurement(model::CompositeMeasurementModel)
    function (x,u,p,t)
        y = zeros(model.ny)
        i = 1
        for m in model.models
            y[i:i+m.ny-1] .= measurement(m)(x,u,p,t)
            i += m.ny
        end
        y
    end
end

function correct!(
    kf::AbstractKalmanFilter,
    measurement_model::CompositeMeasurementModel,
    u,
    y,
    p = parameters(kf),
    t::Real = index(kf) * kf.Ts;
)
    ll = 0.0
    e = zeros(measurement_model.ny)
    S = []
    Sᵪ = []
    K = []
    last_ind = 0
    for i = 1:length(measurement_model.models)
        lli, ei, Si, Sᵪi, Ki = correct!(kf, measurement_model.models[i], u, y, p, t)
        ll += lli
        inds = (1:measurement_model.models[i].ny) .+ last_ind
        e[inds] .= ei
        last_ind = inds[end]
        push!(S, Si)
        push!(Sᵪ, Sᵪi)
        push!(K, Ki)
    end
    ll, e, S, Sᵪ, K
end

struct UKFMeasurementModel{IPM,AUGM,MT,RT,IT,MET,CT,CCT,CAT} <: AbstractMeasurementModel
    measurement::MT
    R2::RT
    ny::Int
    ne::Int
    innovation::IT
    mean::MET
    cov::CT
    cross_cov::CCT
    cache::CAT
end

isinplace(::UKFMeasurementModel{IPM}) where IPM = IPM
has_oop(::UKFMeasurementModel{IPM}) where IPM = !IPM

"""
    UKFMeasurementModel{inplace_measurement,augmented_measurement}(measurement, R2, ny, ne, innovation, mean, cov, cross_cov, cache = nothing)

A measurement model for the Unscented Kalman Filter.

# Arguments:
- `measurement`: The measurement function `y = h(x, u, p, t)`
- `R2`: The measurement noise covariance matrix
- `ny`: The number of measurement variables
- `ne`: If `augmented_measurement` is `true`, the number of measurement noise variables
- `innovation(y::AbstractVector, yh::AbstractVector)` where the arguments represent (measured output, predicted output)
- `mean(ys::AbstractVector{<:AbstractVector})`: computes the mean of the vector of vectors of output sigma points.
- `cov(ys::AbstractVector{<:AbstractVector}, y::AbstractVector)`: computes the covariance matrix of the output sigma points.
- `cross_cov(xs::AbstractVector{<:AbstractVector}, x::AbstractVector, ys::AbstractVector{<:AbstractVector}, y::AbstractVector)` where the arguments represents (state sigma points, mean state, output sigma points, mean output). The function should return the **cross-covariance** matrix between the state and output sigma points.
"""
UKFMeasurementModel{IPM,AUGM}(
    measurement,
    R2,
    ny,
    ne,
    innovation,
    mean,
    cov,
    cross_cov,
    cache = nothing,
) where {IPM,AUGM} = UKFMeasurementModel{
    IPM,
    AUGM,
    typeof(measurement),
    typeof(R2),
    typeof(innovation),
    typeof(mean),
    typeof(cov),
    typeof(cross_cov),
    typeof(cache),
}(
    measurement,
    R2,
    ny,
    ne,
    innovation,
    mean,
    cov,
    cross_cov,
    cache,
)

"""
    UKFMeasurementModel{T,IPM,AUGM}(measurement, R2; nx, ny, ne = nothing, innovation = -, mean = safe_mean, cov = safe_cov, cross_cov = cross_cov, static = nothing)

- `T` is the element type used for arrays
- `IPM` is a boolean indicating if the measurement function is inplace
- `AUGM` is a boolean indicating if the measurement model is augmented
"""
function UKFMeasurementModel{T,IPM,AUGM}(
    measurement,
    R2;
    nx,
    ny,
    ne = nothing,
    innovation = -,
    mean = safe_mean,
    cov = safe_cov,
    cross_cov = cross_cov,
    static = nothing,
) where {T,IPM,AUGM}

    ne = if ne === nothing
        if !AUGM
            0
        elseif R2 isa AbstractArray
            size(R2, 1)
        else
            error(
                "The number of measurement noise variables, ne, can not be inferred from R2 when R2 is not an array, please provide the keyword argument `ne`.",
            )
        end
    else
        if AUGM && R2 isa AbstractArray && size(R2, 1) != ne
            error(
                "R2 must be square with size equal to the measurement vector length for non-augmented measurement",
            )
        end
    end
    if AUGM
        L = nx + ne
    else
        L = nx
    end
    static2 = something(static, L < 50 && !IPM)
    correct_sigma_point_cahce = SigmaPointCache{T}(nx, ne, ny, L, static2)
    UKFMeasurementModel{
        IPM,
        AUGM,
        typeof(measurement),
        typeof(R2),
        typeof(innovation),
        typeof(mean),
        typeof(cov),
        typeof(cross_cov),
        typeof(correct_sigma_point_cahce),
    }(
        measurement,
        R2,
        ny,
        ne,
        innovation,
        mean,
        cov,
        cross_cov,
        correct_sigma_point_cahce,
    )
end


struct SigmaPointCache{X0, X1}
    x0::X0
    x1::X1
end

"""
    SigmaPointCache(nx, nw, ny, L, static)


# Arguments:
- `nx`: Number of state variables
- `nw`: Number of process noise variables for augmented dynamics. If not using augmented dynamics, set to 0.
- `ny`: Number of transformed sigma points
- `L`: Number of sigma points
- `static`: If `true`, the cache will use static arrays for the sigma points. This can be faster for small systems.
"""
function SigmaPointCache{T}(nx, nw, ny, L, static) where T
    if static
        x0 = [@SVector zeros(T, nx + nw) for _ = 1:2L+1]
        x1 = [@SVector zeros(T, ny) for _ = 1:2L+1]
    else
        x0 = [zeros(T, nx + nw) for _ = 1:2L+1]
        x1 = [zeros(T, ny) for _ = 1:2L+1]
    end
    SigmaPointCache(x0, x1)
end

Base.eltype(spc::SigmaPointCache) = eltype(spc.x0)


## EKF measurement model =======================================================

struct EKFMeasurementModel{IPM,MT,RT,CJ,CAT} <: AbstractMeasurementModel
    measurement::MT
    R2::RT
    ny::Int
    Cjac::CJ
    cache::CAT
end

isinplace(::EKFMeasurementModel{IPM}) where IPM = IPM
has_oop(::EKFMeasurementModel{IPM}) where IPM = !IPM

"""
    EKFMeasurementModel{IPM}(measurement, R2, ny, Cjac, cache = nothing)

A measurement model for the Extended Kalman Filter.

# Arguments:
- `IPM`: A boolean indicating if the measurement function is inplace
- `measurement`: The measurement function `y = h(x, u, p, t)`
- `R2`: The measurement noise covariance matrix
- `ny`: The number of measurement variables
- `Cjac`: The Jacobian of the measurement function `Cjac(x, u, p, t)`. If none is provided, ForwardDiff will be used.
"""
EKFMeasurementModel{IPM}(
    measurement,
    R2,
    ny,
    Cjac,
    cache = nothing,
) where {IPM} = EKFMeasurementModel{
    IPM,
    typeof(measurement),
    typeof(R2),
    typeof(Cjac),
    typeof(cache),
}(
    measurement,
    R2,
    ny,
    Cjac,
    cache,
)

"""
    EKFMeasurementModel{T,IPM}(measurement::M, R2; nx, ny, Cjac = nothing)

- `T` is the element type used for arrays
- `IPM` is a boolean indicating if the measurement function is inplace
"""
function EKFMeasurementModel{T,IPM}(
    measurement::M,
    R2;
    nx,
    ny,
    Cjac = nothing,
) where {T,IPM,M}

    
    if Cjac === nothing
        if IPM
            outy = zeros(T, ny)
            jacy = zeros(T, ny, nx)
            Cjac = (x,u,p,t) -> ForwardDiff.jacobian!(jacy, (y,x)->measurement(y,x,u,p,t), outy, x)
        else
            Cjac = (x,u,p,t) -> ForwardDiff.jacobian(x->measurement(x,u,p,t), x)
        end
    end


    EKFMeasurementModel{
        IPM,
        typeof(measurement),
        typeof(R2),
        typeof(Cjac),
        typeof(nothing),
    }(
        measurement,
        R2,
        ny,
        Cjac,
        nothing,
    )
end


## Linear measurement model ====================================================

"""
    LinearMeasurementModel{CT, DT, RT, CAT}

A linear measurement model ``y = C*x + D*u + e``.

# Fields:
- `C` 
- `D`
- `R2`: The measurement noise covariance matrix
- `ny`: The number of measurement variables
"""
struct LinearMeasurementModel{CT,DT,RT,CAT} <: AbstractMeasurementModel
    C::CT
    D::DT
    R2::RT
    ny::Int
    cache::CAT
end

LinearMeasurementModel(C, D, R2; ny = size(R2, 1), cache = nothing, nx=nothing) = LinearMeasurementModel(C, D, R2, ny, cache)
isinplace(::LinearMeasurementModel) = false

function (model::LinearMeasurementModel)(x,u,p,t)
    y = model.C*x
    if !iszero(model.D)
        if y isa SVector
            y += model.D*u
        else
            mul!(y, model.D, u, 1, 1)
        end
    end
    y
end

function (model::LinearMeasurementModel)(y,x,u,p,t)
    mul!(y, model.C, x)
    if !iszero(model.D)
        mul!(y, model.D, u, 1, 1)
    end
    y
end

measurement(model::LinearMeasurementModel) = model