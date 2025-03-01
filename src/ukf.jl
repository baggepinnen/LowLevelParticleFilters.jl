"""
    sigmapoints(m, Σ)

Return a vector of (2n+1) static vectors, where `n` is the length of `m`, representing sigma points with mean `m` and covariance `Σ`.
"""
@inline function sigmapoints(m, Σ; static = true, cholesky! = cholesky!)
    T = promote_type(eltype(m), eltype(Σ))
    n = max(length(m), size(Σ,1))
    if static
        xs = [@SVector zeros(T, n) for _ in 1:(2n+1)]
    else
        xs = [zeros(T, n) for _ in 1:(2n+1)]
    end
    sigmapoints!(xs,m,Σ,cholesky!)
end

function sigmapoints!(xs, m, Σ::AbstractMatrix, cholesky! = cholesky!)
    n = length(xs[1])
    @assert n == length(m)
    # X = sqrt(Symmetric(n*Σ)) # 2.184 μs (16 allocations: 2.27 KiB)
    XX = cholesky!(Symmetric(n*Σ), check=false).L # 170.869 ns (3 allocations: 176 bytes)
    if !issuccess(XX)
        xs[1] = NaN*xs[1]
        return xs
    end
    X = XX.L
    @inbounds @views for i in 1:n
        @bangbang xs[i] .= X[:,i]
        @bangbang xs[i+n] .= .-xs[i] .+ m
        @bangbang xs[i] .= xs[i] .+ m
    end
    xs[end] = m
    xs
end

# UKF ==========================================================================

abstract type AbstractUnscentedKalmanFilter <: AbstractKalmanFilter end

mutable struct UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM,DT,MT,R1T,D0T,SPC,XT,RT,P,RJ,SMT,SCT,CH} <: AbstractUnscentedKalmanFilter
    dynamics::DT
    measurement_model::MT
    R1::R1T
    d0::D0T
    predict_sigma_point_cache::SPC
    x::XT
    R::RT
    t::Int
    Ts::Float64
    ny::Int
    nu::Int
    p::P
    reject::RJ
    state_mean::SMT
    state_cov::SCT
    cholesky!::CH
    names::SignalNames
end

function Base.getproperty(ukf::UnscentedKalmanFilter, s::Symbol)
    s ∈ fieldnames(typeof(ukf)) && return getfield(ukf, s)
    mm = getfield(ukf, :measurement_model)
    if s ∈ fieldnames(typeof(mm))
        return getfield(mm, s)
    elseif s === :nx
        return length(getfield(ukf, :x))
    elseif s === :measurement
        return measurement(mm)
    else
        throw(ArgumentError("$(typeof(ukf)) has no property named $s"))
    end
end

function Base.show(io::IO, ukf::UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}) where {IPD,IPM,AUGD,AUGM}
    println(io, "UnscentedKalmanFilter{$IPD,$IPM,$AUGD,$AUGM}")
    println(io, "  Inplace dynamics: $IPD")
    println(io, "  Inplace measurement: $IPM")
    println(io, "  Augmented dynamics: $AUGD")
    println(io, "  Augmented measurement: $AUGM")
    println(io, "  nx: $(length(ukf.x))")
    println(io, "  nu: $(ukf.nu)")
    println(io, "  ny: $(ukf.ny)")
    println(io, "  Ts: $(ukf.Ts)")
    println(io, "  t: $(ukf.t)")
    for field in fieldnames(typeof(ukf))
        field in (:ny, :nu, :Ts, :t) && continue
        if field in (:measurement_model, :predict_sigma_point_cache)
            println(io, "  $field: $(fieldtype(typeof(ukf), field)))")
        else
            println(io, "  $field: $(repr(getfield(ukf, field), context=:compact => true))")
        end
    end
end


"""
    UnscentedKalmanFilter(dynamics, measurement, R1, R2, d0=MvNormal(Matrix(R1)); p = NullParameters(), ny, nu)
    UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}(dynamics, measurement_model::AbstractMeasurementModel, R1, d0=SimpleMvNormal(R1); p=NullParameters(), nu)

A nonlinear state estimator propagating uncertainty using the unscented transform.

The dynamics and measurement function are on _either_ of the following forms
```
x' = dynamics(x, u, p, t) + w
y  = measurement(x, u, p, t) + e
```
```
x' = dynamics(x, u, p, t, w)
y  = measurement(x, u, p, t, e)
```
where `w ~ N(0, R1)`, `e ~ N(0, R2)` and `x(0) ~ d0`. The former (default) assums that the noise is additive and added _after_ the dynamics and measurement updates, while the latter assumes that the dynamics functions take an additional argument corresponding to the noise term. The latter form (sometimes refered to as the "augmented" form) is useful when the noise is multiplicative or when the noise is added _before_ the dynamics and measurement updates. See "Augmented UKF" below for more details on how to use this form.

The matrices `R1, R2` can be time varying such that, e.g., `R1[:, :, t]` contains the ``R_1`` matrix at time index `t`.
They can also be given as functions on the form
```
Rfun(x, u, p, t) -> R
```
For maximum performance, provide statically sized matrices from StaticArrays.jl

`ny, nu` indicate the number of outputs and inputs.

# Custom type of `u`
The input `u` may be of any type, e.g., a named tuple or a custom struct.
The `u` provided in the input data is passed directly to the dynamics and measurement functions,
so as long as the type is compatible with the dynamics it will work out.
The one exception where this will not work is when calling `simulate`, which assumes that `u` is an array.

# Augmented UKF
If the noise is not additive, one may use the augmented form of the UKF. In this form, the dynamics functions take additional input arguments that correspond to the noise terms. To enable this form, the typed constructor
```
UnscentedKalmanFilter{inplace_dynamics,inplace_measurement,augmented_dynamics,augmented_measurement}(...)
```
is used, where the Boolean type parameters have the following meaning
- `inplace_dynamics`: If `true`, the dynamics function operates in-place, i.e., it modifies the first argument in `dynamics(dx, x, u, p, t)`. Default is `false`.
- `inplace_measurement`: If `true`, the measurement function operates in-place, i.e., it modifies the first argument in `measurement(y, x, u, p, t)`. Default is `false`.
- `augmented_dynamics`: If `true` the dynamics function is augmented with an additional noise input `w`, i.e., `dynamics(x, u, p, t, w)`. Default is `false`.
- `augmented_measurement`: If `true` the measurement function is agumented with an additional noise input `e`, i.e., `measurement(x, u, p, t, e)`. Default is `false`. (If the measurement noise has fewer degrees of freedom than the number of measurements, you may failure in Cholesky factorizations, see "Custom Cholesky factorization" below).

Use of augmented dynamics incurs extra computational cost. The number of sigma points used is `2L+1` where `L` is the length of the augmented state vector. Without augmentation, `L = nx`, with augmentation `L = nx + nw` and `L = nx + ne` for dynamics and measurement, respectively.

# Sigma-point rejection
For problems with challenging dynamics, a mechanism for rejection of sigma points after the dynamics update is provided. A function `reject(x) -> Bool` can be provided through the keyword argument `reject` that returns `true` if a sigma point for ``x(t+1)`` should be rejected, e.g., if an instability or non-finite number is detected. A rejected point is replaced by the propagated mean point (the mean point cannot be rejected). This function may be provided either to the constructor of the UKF or passed to the [`predict!`](@ref) function.

# Custom measurement models

By default, standard arithmetic mean and `e(y, yh) = y - yh` are used as mean and innovation functions.

By passing and explicitly created [`UKFMeasurementModel`](@ref), one may provide custom functions that compute the mean, the covariance and the innovation. This is useful in situations where the state or a measurement lives on a manifold. One may further override the mean and covariance functions for the state sigma points by passing the keyword arguments `state_mean` and `state_cov` to the constructor.


- `state_mean(xs::AbstractVector{<:AbstractVector})` computes the mean of the vector of vectors of state sigma points.
- `state_cov(xs::AbstractVector{<:AbstractVector}, m = mean(xs))` where the first argument represent state sigma points and the second argument, which must be optional, represents the mean of those points. The function should return the covariance matrix of the state sigma points.

See [`UKFMeasurementModel`](@ref) for more details on how to set up a custom measurement model. Pass the custom measurement model as the second argument to the UKF constructor.

# Custom Cholesky factorization
The UnscentedKalmanFilter supports providing a custom function to compute the Cholesky factorization of the covariance matrices for use in sigma-point generation.

If either of the following conditions are met, you may experience failure in internal Cholesky factorizations:
- The dynamics noise or measurement noise covariance matrices (``R_1, R_2``) are singular
- The measurement is augmented and the measurement noise has fewer degrees of freedom than the number of measurements
- (Under specific technical conditions) The dynamics is augmented and the dynamics noise has fewer degrees of freedom than the number of state variables. The technical conditions are easiest to understand in the linear-systems case, where it corresponds to the Riccati equation associated with the Kalman gain not having a solution. This may happen when the pair ``(A, R1)`` has uncontrollable modes on the unit circle, for example, when there are integrating modes that are not affected through the noise.

The error message may look like
```
ERROR: PosDefException: matrix is not positive definite; Factorization failed.
```
In such situations, it is advicable to reconsider the noise model and covariance matrices, alternatively, you may provide a custom Cholesky factorization function to the UKF constructor through the keyword argument `cholesky!`. The function should have the signature `cholesky!(A::AbstractMatrix)::Cholesky`. A useful alternative factorizaiton when covariance matrices are expected to be singular is `cholesky! = R->cholesky!(Positive, Matrix(R))` where the "positive" Cholesky factorization is provided by the package PositiveFactorizations.jl, which must be manually installed and loaded by the user.
"""
function UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}(dynamics, measurement_model::AbstractMeasurementModel, R1, d0=SimpleMvNormal(R1); Ts=1.0, p=NullParameters(), nu::Int, ny=measurement_model.ny, nw = nothing, reject=nothing, state_mean=safe_mean, state_cov=safe_cov, cholesky! = cholesky!, names=default_names(length(d0), nu, ny, "UKF"), kwargs...) where {IPD,IPM,AUGD,AUGM}
    nx = length(d0)
    
    T = promote_type(eltype(d0), eltype(R1))

    if AUGD
        if nw === nothing && R1 isa AbstractArray
            nw = size(R1, 1) # nw may be smaller than nx for augmented dynamics
        elseif nw === nothing
            error("The number of dynamics noise variables, nw, can not be inferred from R1 when R1 is not an array, please provide the keyword argument `nw`.")
        end
        L = nx + nw
    else
        nw = 0
        L = nx
    end
    static = !(IPD || L > 50)
    predict_sigma_point_cache = SigmaPointCache{T}(nx, nw, nx, L, static)

    R = convert_cov_type(R1, d0.Σ)
    x0 = convert_x0_type(d0.μ)
    UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM,typeof(dynamics),typeof(measurement_model),typeof(R1),typeof(d0),
        typeof(predict_sigma_point_cache),typeof(x0),typeof(R),typeof(p),typeof(reject),typeof(state_mean),typeof(state_cov), typeof(cholesky!)}(
        dynamics, measurement_model, R1, d0, predict_sigma_point_cache, x0, R, 0, Ts, ny, nu, p, reject, state_mean, state_cov, cholesky!, names)
end

function UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}(dynamics, measurement, R1, R2, d0=SimpleMvNormal(R1), args...; Ts = 1.0, p = NullParameters(), ny, nu, reject=nothing, state_mean=safe_mean, state_cov=safe_cov, cholesky! = cholesky!, kwargs...) where {IPD,IPM,AUGD,AUGM}
    nx = length(d0)
    T = promote_type(eltype(d0), eltype(R1), eltype(R2))
    measurement_model = UKFMeasurementModel{T,IPM,AUGM}(measurement, R2; nx, ny, kwargs...)
    UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}(dynamics, measurement_model, R1, d0, args...; Ts, p, nu, reject, state_mean, state_cov, cholesky!, kwargs...)
end


function UnscentedKalmanFilter(dynamics,measurement,args...; kwargs...)
    IPD = !has_oop(dynamics)
    IPM = !has_oop(measurement)
    AUGD = false
    AUGM = false
    UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}(dynamics,measurement,args...;kwargs...)
end

sample_state(kf::AbstractUnscentedKalmanFilter, p=parameters(kf); noise=true) = noise ? rand(kf.d0) : mean(kf.d0)
sample_state(kf::AbstractUnscentedKalmanFilter, x, u, p=parameters(kf), t=index(kf)*kf.Ts; noise=true) = kf.dynamics(x,u,p,t) .+ noise.*rand(SimpleMvNormal(get_mat(kf.R1, x, u, p, t)))
sample_measurement(kf::AbstractUnscentedKalmanFilter, x, u, p=parameters(kf), t=index(kf)*kf.Ts; noise=true) = kf.measurement(x, u, p, t) .+ noise.*rand(SimpleMvNormal(get_mat(kf.R2, x, u, p, t)))
measurement(kf::AbstractUnscentedKalmanFilter) = kf.measurement
dynamics(kf::AbstractUnscentedKalmanFilter) = kf.dynamics

#                                        x(k+1)          x            u             p           t
@inline has_ip(fun) = hasmethod(fun, Tuple{AbstractArray,AbstractArray,AbstractArray,AbstractArray,Real})
@inline has_oop(fun) = hasmethod(fun, Tuple{       AbstractArray,AbstractArray,AbstractArray,Real})

"""
    predict!(ukf::UnscentedKalmanFilter, u, p = parameters(ukf), t::Real = index(ukf) * ukf.Ts; R1 = get_mat(ukf.R1, ukf.x, u, p, t), reject, mean, cov, dynamics)

The prediction step for an [`UnscentedKalmanFilter`](@ref) allows the user to override, `R1` and any of the functions, reject, mean, cov, dynamics`.

# Arguments:
- `u`: The input
- `p`: The parameters
- `t`: The current time
- `R1`: The dynamics noise covariance matrix, or a function that returns the covariance matrix.
- `reject`: A function that takes a sigma point and returns `true` if it should be rejected.
- `mean`: The function that computes the mean of the state sigma points.
- `cov`: The function that computes the covariance of the state sigma points.
"""
function predict!(ukf::UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}, u, p = parameters(ukf), t::Real = index(ukf)*ukf.Ts;
        R1 = get_mat(ukf.R1, ukf.x, u, p, t), reject = ukf.reject, mean = ukf.state_mean, cov = ukf.state_cov, dynamics = ukf.dynamics) where {IPD,IPM,AUGD,AUGM}
    (; dynamics,x,R) = ukf
    sigma_point_cache = ukf.predict_sigma_point_cache
    xsd = sigma_point_cache.x1
    # xtyped = eltype(xsd)(x)
    nx = length(x)
    xinds = 1:nx
    sigmapoints_p!(ukf, R1)
    propagate_sigmapoints_p!(ukf, u, p, t, R1)
    if reject !== nothing
        for i = 2:length(xsd)
            if reject(xsd[i])
                # @info "rejecting $(xsd[i]) at time $t"
                @bangbang xsd[i] .= xsd[1]
            end
        end
    end
    if AUGD
        ukf.x = mean(xsd)[xinds]
        @bangbang ukf.R .= symmetrize(cov([x[xinds] for x in xsd])) #+ 1e-16I # TODO: optimize
    else
        ukf.x = mean(xsd)
        @bangbang ukf.R .= symmetrize(cov(xsd, ukf.x)) .+ R1
    end
    ukf.t += 1
end

function propagate_sigmapoints_p!(ukf::UnscentedKalmanFilter{true,<:Any,true}, u, p, t, R1)
    (; dynamics, x) = ukf
    sigma_point_cache = ukf.predict_sigma_point_cache
    xsd,xsd0 = sigma_point_cache.x1, sigma_point_cache.x0
    nx = length(x)
    nw = size(R1, 1) # nw may be smaller than nx for augmented dynamics
    xinds = 1:nx
    winds = nx+1:nx+nw
    for i in eachindex(xsd)
        dynamics(xsd[i], xsd0[i][xinds], u, p, t, xsd0[i][winds])
    end
end

function propagate_sigmapoints_p!(ukf::UnscentedKalmanFilter{false,<:Any,true}, u, p, t, R1)
    (; dynamics, x) = ukf
    sigma_point_cache = ukf.predict_sigma_point_cache
    xsd,xsd0 = sigma_point_cache.x1, sigma_point_cache.x0
    nx = length(x)
    nw = size(R1, 1) # nw may be smaller than nx for augmented dynamics
    xinds = 1:nx
    winds = nx+1:nx+nw
    for i in eachindex(xsd)
        xsd[i] = dynamics(xsd0[i][xinds], u, p, t, xsd0[i][winds])
    end
end

function propagate_sigmapoints_p!(ukf::UnscentedKalmanFilter{true,<:Any,false}, u, p, t, R1)
    (; dynamics, x) = ukf
    sigma_point_cache = ukf.predict_sigma_point_cache
    xsd,xsd0 = sigma_point_cache.x1, sigma_point_cache.x0
    xp = similar(xsd[1])
    for i in eachindex(xsd)
        xp .= 0
        dynamics(xp, xsd0[i], u, p, t)
        xsd[i] .= xp
    end
end

function propagate_sigmapoints_p!(ukf::UnscentedKalmanFilter{false,<:Any,false}, u, p, t, R1)
    (; dynamics) = ukf
    sigma_point_cache = ukf.predict_sigma_point_cache
    xsd,xsd0 = sigma_point_cache.x1, sigma_point_cache.x0
    for i in eachindex(xsd)
        xsd[i] = dynamics(xsd0[i], u, p, t)
    end
end

function sigmapoints_p!(ukf::UnscentedKalmanFilter{<:Any,<:Any,true}, R1)
    sigma_point_cache = ukf.predict_sigma_point_cache
    xsd0 = sigma_point_cache.x0
    m = [ukf.x; 0*R1[:, 1]]
    Raug = cat(ukf.R, R1, dims=(1,2))
    sigmapoints!(xsd0, m, Raug, ukf.cholesky!)
    isnan(xsd0[1][1]) && error("Cholesky factorization of augmented state covariance failed at time step $(ukf.t), see https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/parameter_estimation/#Troubleshooting-Kalman-filters for more help. Got Raug = ", Raug)
    nothing
end

function sigmapoints_p!(ukf::UnscentedKalmanFilter{<:Any,<:Any,false}, R1)
    sigma_point_cache = ukf.predict_sigma_point_cache
    xsd0 = sigma_point_cache.x0
    sigmapoints!(xsd0, ukf.x, ukf.R, ukf.cholesky!)
    isnan(xsd0[1][1]) && error("Cholesky factorization of state covariance failed at time step $(ukf.t), see https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/parameter_estimation/#Troubleshooting-Kalman-filters for more help. Got R = ", ukf.R)
    nothing
end

# The functions below are JET-safe from dynamic dispatch if called with static arrays
safe_mean(xs) = mean(xs)
function safe_mean(xs::Vector{<:SVector})
    m = xs[1]
    for i = 2:length(xs)
        m += xs[i]
    end
    m ./ length(xs)
end

function safe_cov(xs, m=mean(xs))
    # if length(m) > 100
        Statistics.covm(reduce(hcat, xs), m, 2) # This is always faster :/
    # else
    #     Statistics.covm(xs, m)
    # end
end

safe_mean(xs::ColumnSlices) = vec(mean(xs.parent, dims=2))
safe_cov(xs::ColumnSlices, m=mean(xs)) = Statistics.covm(xs.parent, m, 2)

function safe_cov(xs::Vector{<:SVector{N}}, m = safe_mean(xs)) where N
    if N > 8
        Statistics.covm(reduce(hcat, xs), m, 2)
    else
        P = 0 .* m*m'
        for i in eachindex(xs)
            e = xs[i] .- m
            P += e*e'
        end
        c = P ./ (length(xs) - 1)
        c
    end
end

"""
    correct!(ukf::UnscentedKalmanFilter{IPD, IPM, AUGD, AUGM}, u, y, p = parameters(ukf), t::Real = index(ukf) * ukf.Ts; R2 = get_mat(ukf.R2, ukf.x, u, p, t), mean, cross_cov, innovation)

The correction step for an [`UnscentedKalmanFilter`](@ref) allows the user to override, `R2`, `mean`, `cross_cov`, `innovation`.

# Arguments:
- `u`: The input
- `y`: The measurement
- `p`: The parameters
- `t`: The current time
- `R2`: The measurement noise covariance matrix, or a function that returns the covariance matrix `(x,u,p,t)->R2`.
- `mean`: The function that computes the mean of the output sigma points.
- `cross_cov`: The function that computes the cross-covariance of the state and output sigma points.
- `innovation`: The function that computes the innovation between the measured output and the predicted output.

# Extended help
To perform separate measurement updates for different sensors, see the ["Measurement models" in the documentation](@ref measurement_models)
"""
function correct!(ukf::UnscentedKalmanFilter, u, y, p, t::Real; kwargs...)
    measurement_model = ukf.measurement_model
    correct!(ukf, measurement_model, u, y, p, t::Real; kwargs...)    
end


function correct!(
    kf::AbstractKalmanFilter,
    measurement_model::UKFMeasurementModel,
    u,
    y,
    p = parameters(kf),
    t::Real = index(kf) * kf.Ts;
    R2 = get_mat(measurement_model.R2, kf.x, u, p, t),
    mean = measurement_model.mean,
    cross_cov = measurement_model.cross_cov,
    innovation = measurement_model.innovation,
    measurement = measurement_model.measurement,
)

    sigma_point_cache = measurement_model.cache
    xsm = sigma_point_cache.x0
    ys = sigma_point_cache.x1
    (; x, R) = kf

    ns = length(xsm)
    sigmapoints_c!(kf, measurement_model, R2) # TODO: should this take other arguments?
    propagate_sigmapoints_c!(kf, u, p, t, R2, measurement_model)
    ym = mean(ys)
    C  = cross_cov(xsm, x, ys, ym)
    e  = innovation(y, ym)
    S  = compute_S(measurement_model, R2, ym)
    Sᵪ = cholesky(Symmetric(S); check = false)
    issuccess(Sᵪ) ||
        error("Cholesky factorization of innovation covariance failed at time step $(ukf.t), see https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/parameter_estimation/#Troubleshooting-Kalman-filters for more help. Got S = ", S)
    K = (C ./ (ns - 1)) / Sᵪ # ns normalization to make it a covariance matrix
    kf.x += K * e
    # mul!(x, K, e, 1, 1) # K and e will be SVectors if ukf correctly initialized
    RmKSKT!(kf, K, S)
    ll = extended_logpdf(SimpleMvNormal(PDMat(S, Sᵪ)), e) #- 1/2*logdet(S) # logdet is included in logpdf
    (; ll, e, S, Sᵪ, K)
end

# AUGM = false
function sigmapoints_c!(
    kf,
    measurement_model::UKFMeasurementModel{<:Any,false},
    R2,
)
    sigma_point_cache = measurement_model.cache
    xsm = sigma_point_cache.x0
    sigmapoints!(xsm, eltype(xsm)(kf.x), kf.R)
end

function sigmapoints_c!(
    kf,
    measurement_model::UKFMeasurementModel{<:Any,true},
    R2,
)
    (; x, R) = kf
    sigma_point_cache = measurement_model.cache
    xsm = sigma_point_cache.x0
    nx = length(x)
    nv = size(R2, 1)
    xm = [x; 0 * R2[:, 1]]
    Raug = cat(R, R2, dims = (1, 2))
    sigmapoints!(xsm, xm, Raug)
end

# IPM = true, AUGM = false
function propagate_sigmapoints_c!(
    kf,
    u,
    p,
    t,
    R2,
    measurement_model::UKFMeasurementModel{true, false},
)
    sigma_point_cache = measurement_model.cache
    xsm = sigma_point_cache.x0
    ys  = sigma_point_cache.x1
    for i in eachindex(xsm, ys)
        measurement_model.measurement(ys[i], xsm[i], u, p, t)
    end
end

# IPM = true, AUGM = true
function propagate_sigmapoints_c!(
    kf,
    u,
    p,
    t,
    R2,
    measurement_model::UKFMeasurementModel{true,true},
)
    sigma_point_cache = measurement_model.cache
    xsm = sigma_point_cache.x0
    ys = sigma_point_cache.x1
    x = kf.x
    nx = length(x)
    nv = size(R2, 1)
    xinds = 1:nx
    vinds = nx+1:nx+nv
    for i in eachindex(xsm, ys)
        measurement_model.measurement(ys[i], xsm[i][xinds], u, p, t, xsm[i][vinds])
    end
end

# AUGM = true
function propagate_sigmapoints_c!(
    kf,
    u,
    p,
    t,
    R2,
    measurement_model::UKFMeasurementModel{false,true},
)
    sigma_point_cache = measurement_model.cache
    xsm = sigma_point_cache.x0
    ys = sigma_point_cache.x1
    (; x) = kf
    nx = length(x)
    nv = size(R2, 1)
    xinds = 1:nx
    vinds = nx+1:nx+nv
    for i in eachindex(xsm, ys)
        ys[i] = measurement_model.measurement(xsm[i][xinds], u, p, t, xsm[i][vinds])
    end
end

# AUGM = false
function propagate_sigmapoints_c!(
    kf,
    u,
    p,
    t,
    R2,
    measurement_model::UKFMeasurementModel{false,false},
)
    sigma_point_cache = measurement_model.cache
    xsm = sigma_point_cache.x0
    ys  = sigma_point_cache.x1
    for i in eachindex(xsm, ys)
        ys[i] = measurement_model.measurement(xsm[i], u, p, t)
    end
end

function compute_S(measurement_model::UKFMeasurementModel{<:Any, AUGM}, R2, ym) where AUGM
    sigma_point_cache = measurement_model.cache
    ys = sigma_point_cache.x1
    cov = measurement_model.cov
    S = symmetrize(cov(ys, ym))
    if !AUGM
        if S isa SMatrix || S isa Symmetric{<:Any,<:SMatrix}
            S += R2
        else
            S .+= R2
        end
    end
    S
end

"""
    cross_cov(xsm, x, ys, y)

Default `measurement_cov` function for `UnscentedKalmanFilter`. Computes the cross-covariance between the state and output sigma points.
"""
function cross_cov(xsm, x, ys, y)
    T = eltype(x)
    nx = length(x)
    ny = length(y)
    xinds = 1:nx
    if x isa SVector
        C = @SMatrix zeros(T,nx,ny)
    else
        C = zeros(T,nx,ny)
    end
    @inbounds for i in eachindex(ys) # Cross cov between x and y
        d   = ys[i]-y
        C = add_to_C!(C, xsm[i], x, d, xinds)
    end
    C
end


@inline function RmKSKT!(ukf, K, S)
    R = ukf.R
    if R isa SMatrix
        ukf.R = symmetrize(R - K*S*K')
    else
        KS = K*S
        mul!(ukf.R, KS, K', -1, 1)
        symmetrize(ukf.R)
    end
    nothing
end

function add_to_C!(C::SMatrix, xsm, x, d, xinds)
    if length(xinds) == length(xsm)
        C += (xsm-x)*d'
    else
        C += (xsm[xinds]-x)*d'
    end
    C
end

function add_to_C!(C, xsm, x, d, xinds)
    @views if length(xinds) == length(x)
        @bangbang xsm .-= x
        mul!(C, xsm, d', one(eltype(d)), one(eltype(d)))
    else
        xsm[xinds] .-= x
        mul!(C, xsm[xinds], d', one(eltype(d)), one(eltype(d)))
    end
end

function smooth(sol::KalmanFilteringSolution, kf::UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}, u::AbstractVector=sol.u, y::AbstractVector=sol.y, p=parameters(kf)) where {IPD,IPM,AUGD,AUGM}
    # ref: "Unscented Rauch–Tung–Striebel Smoother" Simo Särkkä
    (; x,xt,R,Rt,ll) = sol
    T            = length(y)
    xT           = similar(xt)
    RT           = similar(Rt)
    xT[end]      = xt[end]      |> copy
    RT[end]      = Rt[end]      |> copy
    nx = length(x[1])
    xi = 1:nx
    xqxi = similar(xt[1])
    C = zeros(nx,nx)
    P⁻ = zeros(nx,nx)

    for t = T-1:-1:1
        tt = (t-1)*kf.Ts
        m = xt[t]
        m̃ = [m; 0*m]
        P̃ = cat(Rt[t], get_mat(kf.R1, xt[t], u[t], p, tt), dims=(1,2))
        X̃ = sigmapoints(m̃, P̃)
        X̃⁻ = map(X̃) do xq
            @views xqxi .= xq[xi]
            if AUGD
                if IPD
                    xd = similar(xq[xi]) .= 0
                    kf.dynamics(xd, xqxi, u[t], p, tt, xq[nx+1:end])
                    xd
                else
                    kf.dynamics(xqxi, u[t], p, tt, xq[nx+1:end])
                end
            else
                if IPD
                    xd = similar(xq, length(xi)) .= 0
                    kf.dynamics(xd, (xqxi), u[t], p, tt)
                    xd .+= @view(xq[nx+1:end])
                else
                    kf.dynamics(xqxi, u[t], p, tt) + xq[nx+1:end]
                end
            end
        end
        m⁻ = mean(X̃⁻)
        P⁻ .= 0
        for i in eachindex(X̃⁻)
            e = (X̃⁻[i] - m⁻)
            mul!(P⁻, e, e', 1, 1) # TODO: consider formulating as matrix-matrix multiply
        end
        ns = length(X̃⁻)-1
        @bangbang P⁻ .= P⁻ ./ ns
        C .= 0
        for i in eachindex(X̃⁻) # TODO: consider formulating as matrix-matrix multiply
            mul!(C, (X̃[i][xi] - m), (X̃⁻[i][xi] - m⁻)', 1, 1)
        end
        @bangbang C .= C ./ ns
        D = C / cholesky(P⁻)
        xT[t] = m + D*(xT[t+1]-m⁻[xi])
        RT[t] = Rt[t] + symmetrize(D*(RT[t+1] .- P⁻)*D')
    end
    xT,RT,ll
end


## DAE UKF
#= 
Nonlinear State Estimation of Differential Algebraic Systems

Ravi K. Mandela, Raghunathan Rengaswamy, Shankar Narasimhan

First, unscented samples are chosen for the differential
state variables. The unscented samples for the algebraic variables
are generated from the algebraic equations. This makes
all the sigma points consistent.

ẋ = f(x, z)
g(x, z) = 0
y = h(x, z)

1. sample sigma points for x
2. calculate z for each x using g(x, z) = 0
3. propagate each [x; z] through integrator
4. calc new covariance matrix
=#

# abstract type AbstractUnscentedKalmanFilter <: AbstractKalmanFilter end

# @with_kw struct DAEUnscentedKalmanFilter{DT,MT,R1T,R2T,D0T,VT,XT,RT,P,G,GXZ,BXZ,XZT,VZT} <: AbstractUnscentedKalmanFilter
#     ukf::UnscentedKalmanFilter{DT,MT,R1T,R2T,D0T,VT,XT,RT,P}
#     g::G
#     get_x_z::GXZ
#     build_xz::BXZ
#     xz::XZT
#     xzs::Vector{VZT}
#     nu::Int
#     threads::Bool
#     # TODO: root solver options
# end



# """
#     DAEUnscentedKalmanFilter(ukf; g, get_x_z, build_xz, xz0, threads=false)

# An Unscented Kalman filter for differential-algebraic systems (DAE).

# Ref: "Nonlinear State Estimation of Differential Algebraic Systems", 
# Mandela, Rengaswamy, Narasimhan

# !!! warning
#     This filter is still considered experimental and subject to change without respecting semantic versioning. Use at your own risk.

# # Arguments
# - `ukf` is a regular [`UnscentedKalmanFilter`](@ref) that contains `dynamics(xz, u, p, t)` that propagates the combined state `xz(k)` to `xz(k+1)` and a measurement function with signature `(xz, u, p, t)`
# - `g(x, z, u, p, t)` is a function that should fulfill `g(x, z, u, p, t) = 0`
# - `get_x_z(xz) -> x, z` is a function that decomposes `xz` into `x` and `z`
# - `build_xz(x, z)` is the inverse of `get_x_z`
# - `xz0` the initial full state.
# - `threads`: If true, evaluates dynamics on sigma points in parallel. This typically requires the dynamics to be non-allocating (use StaticArrays) to improve performance. 

# # Assumptions
# - The DAE dynamics is index 1 and can be written on the form 
# ```math
# \\begin{aligned}
# ẋ &= f(x, z, u, p, t) \\quad &\\text{Differential equations}\\
# 0 &= g(x, z, u, p, t) \\quad &\\text{Algebraic equations}\\
# y &= h(x, z, u, p, t) \\quad &\\text{Measurements}
# \\begin{aligned}
# ```
# the measurements may be functions of both differential state variables `x` and algebraic variables `z`.
# Please note, the actual dynamics and measurement functions stored in the internal `ukf` should have signatures `(xz, u, p, t)`, i.e.,
# they take the combined state (descriptor) containing both `x` and `z` in a single vector as dictated by the function `build_xz`.
# It is only the function `g` that is assumed to actually have the signature `g(x,z,u,p,t)`.
# """
# function DAEUnscentedKalmanFilter(ukf; g, get_x_z, build_xz, xz0, nu::Int, threads::Bool=false)
#     T = eltype(ukf.xs[1])
#     n = length(ukf.x)
#     xzs = [@SVector zeros(T, length(xz0)) for _ in 1:(2n+1)] # These vectors have the length of xz0 but the number of them is determined by the dimension of x only
#     DAEUnscentedKalmanFilter(ukf, g, get_x_z, build_xz, copy(xz0), xzs, nu, threads)
# end


# function Base.getproperty(ukf::DAEUnscentedKalmanFilter, s::Symbol)
#     s ∈ fieldnames(typeof(ukf)) && return getfield(ukf, s)
#     getproperty(getfield(ukf, :ukf), s) # Forward to inner filter
# end

# function Base.setproperty!(ukf::DAEUnscentedKalmanFilter, s::Symbol, val)
#     s ∈ fieldnames(typeof(ukf)) && return setproperty!(ukf, s, val)
#     setproperty!(getfield(ukf, :ukf), s, val) # Forward to inner filter
# end

# Base.propertynames(ukf::DAEUnscentedKalmanFilter) = (fieldnames(typeof(ukf))..., propertynames(getfield(ukf, :ukf))...)

# state(ukf::DAEUnscentedKalmanFilter) = ukf.xz

# function sample_state(kf::DAEUnscentedKalmanFilter, p=parameters(kf); noise=true)
#     @unpack get_x_z, build_xz, xz, g, dynamics, nu = kf
#     xh = noise ? rand(kf.d0) : mean(kf.d0)
#     calc_xz(get_x_z, build_xz, g, xz, zeros(nu), p, 0, xh)
# end
# function sample_state(kf::DAEUnscentedKalmanFilter, x, u, p, t; noise=true)
#     @unpack get_x_z, build_xz, xz, g, dynamics, R1 = kf
#     xzp = dynamics(x,u,p,t)
#     noise || return xzp
#     xh = get_x_z(xzp)[1]
#     xh += rand(SimpleMvNormal(Matrix(get_mat(R1, x, u, p, t))))
#     calc_xz(get_x_z, build_xz, g, xz, u, p, t, xh)
# end

# """
#     calc_xz(dae_ukf, xz, u, p, t, x=get_x_z(xz)[1])
#     calc_xz(get_x_z, build_xz, g, xz, u, p, t, x=get_x_z(xz)[1])

# Find `z` such that g(x, z) = 0 (zeros of length(x) + length(z))
# The z part of xz is used as initial guess

# # Arguments:
# - `x`: If not provided, x from xz will be used
# - `xz`: Full state
# """
# function calc_xz(get_x_z::Function, build_xz, g, xz::AbstractArray, u, p, t, xi=get_x_z(xz)[1])
#     _, z0 = get_x_z(xz) # use previous z as initial guess for root finder
#     sol = solve(NonlinearProblem{false}((z,_)->g(xi, z, u, p, t), z0), SimpleNewtonRaphson(), reltol=1e-10) # function takes parameter as second arg
#     nr = norm(sol.resid)
#     nr < 1e-3 || @warn "Root solving residual was large $nr" maxlog=10
#     zi = sol.u
#     build_xz(xi, zi)
# end

# calc_xz(ukf::DAEUnscentedKalmanFilter, args...) = 
#     calc_xz(ukf.get_x_z, ukf.build_xz, ukf.g, args...)

# function predict!(ukf::DAEUnscentedKalmanFilter, u, p = parameters(ukf), t = index(ukf)*ukf.Ts; R1 = get_mat(ukf.R1, ukf.x, u, p, t))
#     @unpack dynamics,measurement,x,xs,xz,xzs,R,g,build_xz,get_x_z = ukf
#     ns = length(xs)
#     sigmapoints!(xs,x,R) # generate only for x
#     if ukf.threads
#         @batch for i in eachindex(xs)
#             # generate z
#             xzi = calc_xz(ukf, xzs[i], u, p, t, xs[i])
#             xzs[i] = dynamics(xzi, u, p, t) # here they must be the same and in the correct order
#             xs[i],_ = get_x_z(xzs[i])
#         end
#     else
#         for i in eachindex(xs)
#             # generate z
#             xzi = calc_xz(ukf, xzs[i], u, p, t, xs[i])
#             xzs[i] = dynamics(xzi, u, p, t) # here they must be the same and in the correct order
#             xs[i],_ = get_x_z(xzs[i])
#         end
#     end
#     ukf.x = mean(xs) # xz or xs here? Answer: Covariance is associated only with x
#     xz .= calc_xz(ukf, xz, u, p, t, x)
#     ukf.R = symmetrize(cov(xs)) + get_mat(R1, x, u, p, t)
#     ukf.t += 1
# end

# function correct!(ukf::DAEUnscentedKalmanFilter, u, y, p = parameters(ukf), t = index(ukf)*ukf.Ts; R2 = get_mat(ukf.R2, ukf.x, u, p, t))
#     @unpack measurement,x,xs,xz,xzs,R,R1,g,get_x_z,build_xz  = ukf
#     n = size(R1,1)
#     p = size(R2,1)
#     ns = length(xs)
#     sigmapoints!(xs,x,R) # Update sigmapoints here since untransformed points required
#     for i in eachindex(xs)
#         xzs[i] = calc_xz(ukf, xzs[i], u, p, t, xs[i])
#     end
#     C = @SMatrix zeros(n,p)
#     ys = map(xzs) do xzi
#         measurement(xzi, u, p, t)
#     end
#     ym = mean(ys)
#     @inbounds for i in eachindex(ys) # Cross cov between x and y
#         d   = ys[i]-ym
#         xi,_ = get_x_z(xzs[i])
#         ca  = (xi-x)*d'
#         C  += ca
#     end
#     e   = y .- ym
#     S   = symmetrize(cov(ys)) + R2 # cov of y
#     Sᵪ = cholesky(S)
#     K   = (C./ns)/Sᵪ # ns normalization to make it a covariance matrix
#     ukf.x += K*e
#     xz .= calc_xz(ukf, xz, u, p, t, x)
#     # mul!(x, K, e, 1, 1) # K and e will be SVectors if ukf correctly initialized
#     if R isa SMatrix
#         ukf.R = symmetrize(R - K*S*K')
#     else
#         RmKSKT!(R, K, S)
#     end
#     ll = extended_logpdf(SimpleMvNormal(PDMat(S,Sᵪ)), e) #- 1/2*logdet(S) # logdet is included in logpdf
#     (; ll, e, S, Sᵪ, K)
# end


function unscentedplot end
function unscentedplot! end