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
    R2 = nothing,
)
    R2 === nothing || @warn("correct! with a composite measurement model ignores the custom R2 argument, open an issue if you need this feature.", maxlog=3)
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

struct UKFMeasurementModel{IPM,AUGM,MT,RT,IT,MET,CT,CCT,CAT,WP} <: AbstractMeasurementModel
    measurement::MT
    R2::RT
    ny::Int
    ne::Int
    innovation::IT
    mean::MET
    cov::CT
    cross_cov::CCT
    cache::CAT
    weight_params::WP
end

isinplace(::UKFMeasurementModel{IPM}) where IPM = IPM
has_oop(::UKFMeasurementModel{IPM}) where IPM = !IPM

"""
    UKFMeasurementModel{inplace_measurement,augmented_measurement}(measurement, R2, ny, ne, innovation, mean, cov, cross_cov, weight_params, cache = nothing)

A measurement model for the Unscented Kalman Filter.

# Arguments:
- `measurement`: The measurement function `y = h(x, u, p, t)`
- `R2`: The measurement noise covariance matrix
- `ny`: The number of measurement variables
- `ne`: If `augmented_measurement` is `true`, the number of measurement noise variables
- `innovation(y::AbstractVector, yh::AbstractVector)` where the arguments represent (measured output, predicted output)
- `mean(ys::AbstractVector{<:AbstractVector})`: computes the mean of the vector of vectors of output sigma points.
- `cov(ys::AbstractVector{<:AbstractVector}, y::AbstractVector)`: computes the covariance matrix of the output sigma points.
- `cross_cov(xs::AbstractVector{<:AbstractVector}, x::AbstractVector, ys::AbstractVector{<:AbstractVector}, y::AbstractVector, W::UKFWeights)` where the arguments represents (state sigma points, mean state, output sigma points, mean output, weights). The function should return the weighted **cross-covariance** matrix between the state and output sigma points.
- `weight_params`: A type that holds the parameters for the unscented-transform weights. See [`UnscentedKalmanFilter`](@ref) and [Docs: Unscented transform](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/ut/) for more information.
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
    weight_params = TrivialParams(),
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
    typeof(weight_params),
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
    weight_params,
)

"""
    UKFMeasurementModel{T,IPM,AUGM}(measurement, R2; nx, ny, ne = nothing, innovation = -, mean = weighted_mean, cov = weighted_cov, cross_cov = cross_cov, static = nothing)

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
    mean = weighted_mean,
    cov = weighted_cov,
    cross_cov = cross_cov,
    static = nothing,
    names = nothing, # throwaway
    weight_params = TrivialParams(),
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
        typeof(weight_params),
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
        weight_params,
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

struct EKFMeasurementModel{IPM,MT,RT,CJ,R12T,CAT} <: AbstractMeasurementModel
    measurement::MT
    R2::RT
    ny::Int
    Cjac::CJ
    R12::R12T
    cache::CAT
end

isinplace(::EKFMeasurementModel{IPM}) where IPM = IPM
has_oop(::EKFMeasurementModel{IPM}) where IPM = !IPM

"""
    EKFMeasurementModel{IPM}(measurement, R2, ny, Cjac, R12 = nothing, cache = nothing)

A measurement model for the Extended Kalman Filter.

# Arguments:
- `IPM`: A boolean indicating if the measurement function is inplace
- `measurement`: The measurement function `y = h(x, u, p, t)`
- `R2`: The measurement noise covariance matrix
- `ny`: The number of measurement variables
- `Cjac`: The Jacobian of the measurement function `Cjac(x, u, p, t)`. If none is provided, ForwardDiff will be used.
- `R12`: Cross-covariance between dynamics noise at step `k` and measurement noise at step `k+1`. See Simon, D.: "Optimal state estimation: Kalman, H Infinity, and nonlinear approaches" sec. 7.1
"""
EKFMeasurementModel{IPM}(
    measurement,
    R2,
    ny,
    Cjac,
    R12 = nothing,
    cache = nothing,
) where {IPM} = EKFMeasurementModel{
    IPM,
    typeof(measurement),
    typeof(R2),
    typeof(Cjac),
    typeof(R12),
    typeof(cache),
}(
    measurement,
    R2,
    ny,
    Cjac,
    R12,
    cache,
)

"""
    EKFMeasurementModel{T,IPM}(measurement::M, R2; nx, ny, Cjac = nothing, R12 = nothing)

- `T` is the element type used for arrays
- `IPM` is a boolean indicating if the measurement function is inplace
- `R12` is the cross-covariance between dynamics noise and measurement noise
"""
function EKFMeasurementModel{T,IPM}(
    measurement::M,
    R2;
    nx,
    ny,
    Cjac = nothing,
    R12 = nothing,
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
        typeof(R12),
        typeof(nothing),
    }(
        measurement,
        R2,
        ny,
        Cjac,
        R12,
        nothing,
    )
end


## Linear measurement model ====================================================

"""
    LinearMeasurementModel{CT, DT, RT, R12T, CAT}

A linear measurement model ``y = C*x + D*u + e``.

# Fields:
- `C`
- `D`
- `R2`: The measurement noise covariance matrix
- `ny`: The number of measurement variables
- `R12`: Cross-covariance between dynamics noise at step `k` and measurement noise at step `k+1`. See Simon, D.: "Optimal state estimation: Kalman, H Infinity, and nonlinear approaches" sec. 7.1
"""
struct LinearMeasurementModel{CT,DT,RT,R12T,CAT} <: AbstractMeasurementModel
    C::CT
    D::DT
    R2::RT
    ny::Int
    R12::R12T
    cache::CAT
end

LinearMeasurementModel(C, D, R2; ny = size(R2, 1), R12 = nothing, cache = nothing, nx=nothing) = LinearMeasurementModel(C, D, R2, ny, R12, cache)
isinplace(::LinearMeasurementModel) = false

function (model::LinearMeasurementModel)(x,u,p,t)
    y = get_mat(model.C,x,u,p,t)*x
    D = get_mat(model.D,x,u,p,t)
    if !iszero(D)
        if y isa SVector
            y += D*u
        else
            mul!(y, D, u, 1, 1)
        end
    end
    y
end

function (model::LinearMeasurementModel)(y,x,u,p,t)
    C = get_mat(model.C,x,u,p,t)
    D = get_mat(model.D,x,u,p,t)
    mul!(y, C, x)
    if !iszero(D)
        mul!(y, D, u, 1, 1)
    end
    y
end

measurement(model::LinearMeasurementModel) = model


## IEKF measurement model ======================================================

struct IEKFMeasurementModel{IPM,MT,RT,CJ,R12T,CAT} <: AbstractMeasurementModel
    measurement::MT
    R2::RT
    ny::Int
    Cjac::CJ
    R12::R12T
    step::Real # (0.,1.) step size in the gauss-newton method
    maxiters::Int # maximum number of iterations
    epsilon::Real # convergence criterion
    cache::CAT
end

isinplace(::IEKFMeasurementModel{IPM}) where IPM = IPM
has_oop(::IEKFMeasurementModel{IPM}) where IPM = !IPM

"""
    IEKFMeasurementModel{IPM}(measurement, R2, ny, Cjac, R12 = nothing, step = 1.0, maxiters = 10, epsilon = 1e-8, cache = nothing)

A measurement model for the Iterated Extended Kalman Filter.

# Arguments:
- `IPM`: A boolean indicating if the measurement function is inplace
- `measurement`: The measurement function `y = h(x, u, p, t)`
- `R2`: The measurement noise covariance matrix
- `ny`: The number of measurement variables
- `Cjac`: The Jacobian of the measurement function `Cjac(x, u, p, t)`. If none is provided, ForwardDiff will be used.
- `R12`: Cross-covariance between dynamics noise at step `k` and measurement noise at step `k+1`. See Simon, D.: "Optimal state estimation: Kalman, H Infinity, and nonlinear approaches" sec. 7.1
- `step`: The step size in the Gauss-Newton method. Should be Float64 between 0 and 1.
- `maxiters`: The maximum number of iterations of the Gauss-Newton method inside the IEKF
- `epsilon`: The convergence criterion for the Gauss-Newton method inside the IEKF
- `cache`: A cache for the Jacobian
"""
IEKFMeasurementModel{IPM}(
    measurement,
    R2,
    ny,
    Cjac,
    R12 = nothing,
    step = 1.0,
    maxiters = 10,
    epsilon = 1e-8,
    cache = nothing,
) where {IPM} = IEKFMeasurementModel{
    IPM,
    typeof(measurement),
    typeof(R2),
    typeof(Cjac),
    typeof(R12),
    typeof(cache),
}(
    measurement,
    R2,
    ny,
    Cjac,
    R12,
    step,
    maxiters,
    epsilon,
    cache,
)

"""
    IEKFMeasurementModel{T,IPM}(measurement::M, R2; nx, ny, Cjac = nothing, R12 = nothing, step = 1.0, maxiters = 10, epsilon = 1e-8)

- `T` is the element type used for arrays
- `IPM` is a boolean indicating if the measurement function is inplace
- `R12` is the cross-covariance between dynamics noise and measurement noise
"""
function IEKFMeasurementModel{T,IPM}(
    measurement::M,
    R2;
    nx,
    ny,
    Cjac = nothing,
    R12 = nothing,
    step = 1.0,
    maxiters = 10,
    epsilon = 1e-8,
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

    if step < 0 || step > 1
        error("IEKF step size should be between 0 and 1")
    end

    IEKFMeasurementModel{
        IPM,
        typeof(measurement),
        typeof(R2),
        typeof(Cjac),
        typeof(R12),
        typeof(nothing),
    }(
        measurement,
        R2,
        ny,
        Cjac,
        R12,
        step,
        maxiters,
        epsilon,
        nothing,
    )
end