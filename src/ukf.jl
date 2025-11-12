abstract type UTParams end
"""
    WikiParams(; α = 1.0, β = 0.0, κ = 1.0)
    WikiParams(; ακ = 1.0, β = 0.0) # Simplified interface with only one parameter for ακ

Unscented transform parameters suggested at [Wiki: Kalman_filter#Sigma_points](https://en.wikipedia.org/wiki/Kalman_filter#Sigma_points).

- `α`: Scaling parameter (0,1] for the spread of the sigma points. Reduce `α` to reduce the spread.
- `β`: Incorporates prior knowledge of the distribution of the state.
- `κ`: Secondary scaling parameter that is usually set to 3nx/2 or 1. Increase `κ` to increase the spread of the sigma points.

If ``α^2 κ < L`` where ``L`` is the dimension ofthe sigma points, the center mean weight is negative. This is allowed, but may in some cases lead to an indefinite covariance matrix.

The spread of the points are ``α^2 κ``, that is, independent on the point dimension. Visualize the spread by
```julia
using Plots
μ = [0.0, 0.0]
Σ = [1.0 0.0; 0.0 1.0]
pars = LowLevelParticleFilters.WikiParams(α = 1.0, β = 0.0, κ = 1.0)
xs = LowLevelParticleFilters.sigmapoints(μ, Σ, pars)
unscentedplot(xs, pars)
```

A simplified tuning rule 
- If a decrease in the spread of the sigma points is desired, use ``κ = 1`` and ``α < 1``.
- If an increase in the spread of the sigma points is desired, use ``κ > 1`` and ``α = 1``.

This rule may be used when using the interface with only a single function argument ``ακ``. See Nielsen, K. et al., 2021, "UKF Parameter Tuning for Local Variation Smoothing" for more details.

See also [`MerweParams`](@ref) and [`TrivialParams`](@ref)
"""
struct WikiParams{T} <: UTParams
    α::T
    β::T
    κ::T
    function WikiParams(; α = 1.0, β = 0.0, κ = 1.0, ακ = nothing)
        T = float(promote_type(typeof(α), typeof(β), typeof(κ)))
        if ακ !== nothing
            ακ > 0 || throw(ArgumentError("ακ must be positive"))
            if ακ < 1
                α = ακ
                κ = T(1.0)
            else
                α = T(1.0)
                κ = ακ
            end
        else
            α > 0 || throw(ArgumentError("α must be positive"))
            κ == 0 && throw(ArgumentError("κ must be non-zero"))
        end
        # β >= 0 || throw(ArgumentError("β must be non-negative"))
        new{T}(α, β, κ)
    end
end

"""
    MerweParams(; α = 1.0, β = 2.0, κ = 0.0)
    MerweParams(; ακ = 1.0, β = 2.0) # Simplified interface with only one parameter for ακ

Unscented transform parameters suggested by van der Merwe et al.

- `α`: Scaling parameter (0,1] for the spread of the sigma points. Reduce `α` to reduce the spread.
- `β`: Incorporates prior knowledge of the distribution of the state.
- `κ`: Secondary scaling parameter that is usually set to 0. Increase `κ` to increase the spread of the sigma points.

If ``α^2 (L + κ) < L`` where ``L`` is the dimension of the sigma points, the center mean weight is negative. This is allowed, but may in some cases lead to an indefinite covariance matrix.

The spread of the points are ``α^2 (L + κ)`` where ``L`` is the dimension of each point. Visualize the spread by
```julia
using Plots
μ = [0.0, 0.0]
Σ = [1.0 0.0; 0.0 1.0]
pars = LowLevelParticleFilters.MerweParams(α = 1e-3, β = 2.0, κ = 0.0)
xs = LowLevelParticleFilters.sigmapoints(μ, Σ, pars)
unscentedplot(xs, pars)
```

A simplified tuning rule 
- If a decrease in the spread of the sigma points is desired, use ``κ = 0`` and ``α < 1``.
- If an increase in the spread of the sigma points is desired, use ``κ > 0`` and ``α = 1``.

This rule may be used when using the interface with only a single function argument ``ακ``. See Nielsen, K. et al., 2021, "UKF Parameter Tuning for Local Variation Smoothing" for more details.

See also [`WikiParams`](@ref) and [`TrivialParams`](@ref)
"""
struct MerweParams{T} <: UTParams
    α::T
    β::T
    κ::T
    function MerweParams(; α = 1e-3, β = 2.0, κ = 0.0, ακ = nothing)
        T = float(promote_type(typeof(α), typeof(β), typeof(κ)))
        if ακ !== nothing
            ακ > 0 || throw(ArgumentError("ακ must be positive"))
            if ακ < 1
                α = ακ
                κ = T(0.0)
            else
                α = T(1.0)
                κ = ακ
            end
        else
            α > 0 || throw(ArgumentError("α must be positive"))
            # κ == 0 && throw(ArgumentError("κ must be non-zero"))
        end
        # β >= 0 || throw(ArgumentError("β must be non-negative"))
        new{T}(α, β, κ)
    end
end

"""
    TrivialParams()

Unscented transform parameters representing a trivial choice of weights, where all weights are equal.

See also [`WikiParams`](@ref) and [`MerweParams`](@ref)
"""
struct TrivialParams <: UTParams end


"""
    UKFWeights

Weights for the Unscented Transform.

Sigmapoints are by convention ordered such that the center (mean) point is first.

# Fields
- `wm`: center weight when computing mean
- `wc`: center weight when computing covariance
- `wmi`: off-center weight when computing mean
- `wci`: off-center weight when computing covariance
- `W`: Cholesky weight
"""
struct UKFWeights{T}
    "mean center weight"
    wm::T
    "covariance center weight"
    wc::T
    "off-center mean weight"
    wmi::T
    "off-center cov weight"
    wci::T
    "Cholesky weight"
    W::T
end

ns2L(n) = (n-1) ÷ 2

function UKFWeights(W::WikiParams, L::Integer)
    (; α, β, κ) = W
    α2κ = α^2 * κ
    wm = (α2κ - L) / (α2κ)
    wc = wm + 1 - α^2 + β
    wi = 1 / (2 * α2κ)
    WC = α^2*κ # To be applied on the input of Cholesky, not on the output as in wiki page
    isfinite(wm) || error("wm is not finite")
    isfinite(wi) || error("wi is not finite")
    isfinite(WC) || error("WC is not finite")
    UKFWeights(wm, wc, wi, wi, WC)
end

function UKFWeights(W::MerweParams, L::Integer)
    (; α, β, κ) = W
    λ = α^2 * (L + κ) - L
    wm = λ / (L + λ)
    wc = wm + 1 - α^2 + β
    wi = 1 / (2 * (L + λ))
    WC = L + λ
    isfinite(wm) || error("wm is not finite")
    isfinite(wi) || error("wi is not finite")
    isfinite(WC) || error("WC is not finite")
    UKFWeights(wm, wc, wi, wi, WC)
end

function UKFWeights(::TrivialParams, L::Integer)
    N = (2L+1)
    wm = 1 / N
    wc = 1 / (N-1)
    UKFWeights(wm, wc, wm, wc, typeof(wm)(L))
end





"""
    sigmapoints(m, Σ)

Return a vector of (2n+1) static vectors, where `n` is the length of `m`, representing sigma points with mean `m` and covariance `Σ`.
"""
@inline function sigmapoints(m, Σ, weight_params=TrivialParams(); static = true, cholesky! = cholesky!)
    T = promote_type(eltype(m), eltype(Σ))
    n = max(length(m), size(Σ,1))
    if static
        xs = [@SVector zeros(T, n) for _ in 1:(2n+1)]
    else
        xs = [zeros(T, n) for _ in 1:(2n+1)]
    end
    sigmapoints!(xs,m,Σ,weight_params,cholesky!)
end

function sigmapoints!(xs, m, Σ::AbstractMatrix, weight_params, cholesky! = cholesky!)
    n = length(xs[1])
    @assert n == length(m)
    @assert length(xs) == 2n+1
    W = UKFWeights(weight_params, n)
    CI = Symmetric(W.W*Σ)
    if cholesky! === LinearAlgebra.cholesky!
        # X = sqrt(CI) # 2.184 μs (16 allocations: 2.27 KiB)
        XX = cholesky!(CI, check=false) # 170.869 ns (3 allocations: 176 bytes)
        if !issuccess(XX)
            xs[1] = NaN*xs[1]
            return xs
        end
        X = XX.L
    else
        X = cholesky!(CI).L
    end
    xs[1] = m
    @inbounds @views for i in 2:n+1
        @bangbang xs[i] .= X[:,i-1]
        @bangbang xs[i+n] .= .-xs[i] .+ m
        @bangbang xs[i] .= xs[i] .+ m
    end
    xs
end

# UKF ==========================================================================

abstract type AbstractUnscentedKalmanFilter <: AbstractKalmanFilter end

mutable struct UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM,DT,MT,R1T,D0T,SPC,XT,RT,P,RJ,SMT,SCT,CH,WP,R1XT} <: AbstractUnscentedKalmanFilter
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
    weight_params::WP
    R1x::R1XT
end

function Base.getproperty(ukf::UnscentedKalmanFilter{<:Any, <:Any, AUGD}, s::Symbol) where AUGD
    s ∈ fieldnames(typeof(ukf)) && return getfield(ukf, s)
    mm = getfield(ukf, :measurement_model)
    if s ∈ fieldnames(typeof(mm))
        return getfield(mm, s)
    elseif s === :nx
        return length(getfield(ukf, :x))
    elseif s === :nw
        nx = length(getfield(ukf, :x))
        return AUGD ? length(getfield(ukf, :predict_sigma_point_cache).x0[1])-nx : nx
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
    UnscentedKalmanFilter(dynamics, measurement, R1, R2, d0=MvNormal(Matrix(R1)); p = NullParameters(), ny, nu, weight_params)
    UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}(dynamics, measurement_model::AbstractMeasurementModel, R1, d0=SimpleMvNormal(R1); p=NullParameters(), nu, weight_params)

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
where `w ~ N(0, R1)`, `e ~ N(0, R2)` and `x(0) ~ d0`. The former (default) assums that the noise is additive and added _after_ the dynamics and measurement updates, while the latter assumes that the dynamics functions take an additional argument corresponding to the noise term. The latter form (sometimes refered to as the "augmented" form) is useful when the noise is multiplicative or when the noise is added _before_ the dynamics and measurement updates. See "Augmented UKF" below for more details on how to use this form. In both cases should the noise be modeled as discrete-time white noise, see Discretization: [Covariance matrices](@ref).

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

# Weight tuning
The spread of the sigma points is controlled by `weight_params::UTParams`. See [Docs: Unscented transform](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/ut/) for a tutorial. The default is [`TrivialParams`](@ref) for unweighted sigma points, other options are [`WikiParams`](@ref) and [`MerweParams`](@ref).

# Sigma-point rejection
For problems with challenging dynamics, a mechanism for rejection of sigma points after the dynamics update is provided. A function `reject(x) -> Bool` can be provided through the keyword argument `reject` that returns `true` if a sigma point for ``x(t+1)`` should be rejected, e.g., if an instability or non-finite number is detected. A rejected point is replaced by the propagated mean point (the mean point cannot be rejected). This function may be provided either to the constructor of the UKF or passed to the [`predict!`](@ref) function.

# Enforcing contraints using sigma-point projection
Constraints on the state (or output) may be enforced by projecting the sigma points onto the constraint set during the dynamics (or measurement) update. In general, two projections per update are required, one after the generation of the sigma points but before the dynamics is applied, and one after the dynamics update. No functionality for this is provided in this package, but the projection may be readibly implemented manually in the dynamics function, e.g.,
```julia
function dynamics(x, u, p, t)
    x  = project(x)  # Sigma points may have been generated outside the constraint set
    xp = f(x, u, p, t)
    xp = project(xp) # The dynamics may have moved the points outside the constraint set
    return xp
end
```

Equality constraints can alternatively be handled by making use of a pseudo-measurement ``0 = C_{con}x`` with close to zero covariance.

# Custom measurement models

By default, standard arithmetic mean and `e(y, yh) = y - yh` are used as mean and innovation functions.

By passing and explicitly created [`UKFMeasurementModel`](@ref), one may provide custom functions that compute the mean, the covariance and the innovation. This is useful in situations where the state or a measurement lives on a manifold. One may further override the mean and covariance functions for the state sigma points by passing the keyword arguments `state_mean` and `state_cov` to the constructor.


- `state_mean(xs::AbstractVector{<:AbstractVector}, w::UKFWeights)` computes the weighted mean of the vector of vectors of state sigma points.
- `state_cov(xs::AbstractVector{<:AbstractVector}, m, w::UKFWeights)` where the first argument represent state sigma points and the second argument, represents the weighted mean of those points. The function should return the covariance matrix of the state sigma points weighted by `w`.

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
function UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}(dynamics, measurement_model::AbstractMeasurementModel, R1, d0=SimpleMvNormal(R1); Ts=1.0, p=NullParameters(), nu::Int, ny=measurement_model.ny, nw = nothing, reject=nothing, state_mean=weighted_mean, state_cov=weighted_cov, cholesky! = cholesky!, names=default_names(length(d0), nu, ny, "UKF"), weight_params = TrivialParams(), R1x=nothing, kwargs...) where {IPD,IPM,AUGD,AUGM}
    nx = length(d0)
    
    T = eltype(d0)

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
    if !hasmethod(state_mean, Tuple{AbstractVector, UKFWeights})
        weight_params isa TrivialParams || error("Unweighted state mean may not be used with custom weights")
        user_mean = state_mean
        state_mean = (xs, w) -> user_mean(xs)
    end
    if !hasmethod(state_cov, Tuple{AbstractVector, AbstractVector, UKFWeights})
        weight_params isa TrivialParams || error("Unweighted state covariance may not be used with custom weights")
        user_cov = state_cov
        state_cov = (xs, m, w) -> user_cov(xs)
    end

    R = convert_cov_type(R1, d0.Σ)
    x0 = eltype(predict_sigma_point_cache.x1)(convert_x0_type(d0.μ))
    UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM,typeof(dynamics),typeof(measurement_model),typeof(R1),typeof(d0),
        typeof(predict_sigma_point_cache),typeof(x0),typeof(R),typeof(p),typeof(reject),typeof(state_mean),typeof(state_cov), typeof(cholesky!), typeof(weight_params), typeof(R1x)}(
        dynamics, measurement_model, R1, d0, predict_sigma_point_cache, x0, R, 0, Ts, ny, nu, p, reject, state_mean, state_cov, cholesky!, names, weight_params, R1x)
end

function UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}(dynamics, measurement, R1, R2, d0=SimpleMvNormal(R1), args...; Ts = 1.0, p = NullParameters(), ny, nu, reject=nothing, state_mean=weighted_mean, state_cov=weighted_cov, cholesky! = cholesky!, kwargs...) where {IPD,IPM,AUGD,AUGM}
    nx = length(d0)
    T = eltype(d0)
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

function sample_state(kf::UnscentedKalmanFilter{false,<:Any,true,<:Any}, x, u, p=parameters(kf), t=index(kf)*kf.Ts; noise=true)
    kf.dynamics(x,u,p,t, noise.*rand(SimpleMvNormal(get_mat(kf.R1, x, u, p, t))))
end

sample_measurement(kf::AbstractUnscentedKalmanFilter, x, u, p=parameters(kf), t=index(kf)*kf.Ts; noise=true) = kf.measurement(x, u, p, t) .+ noise.*rand(SimpleMvNormal(get_mat(kf.R2, x, u, p, t)))
function sample_measurement(kf::UnscentedKalmanFilter{<:Any, <:Any, <:Any, true}, x, u, p=parameters(kf), t=index(kf)*kf.Ts; noise=true)
    e = noise.*rand(SimpleMvNormal(get_mat(kf.R2, x, u, p, t)))
    kf.measurement(x, u, p, t, e)
end
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
        R1 = get_mat(ukf.R1, ukf.x, u, p, t), reject = ukf.reject, mean::MF = ukf.state_mean, cov::CF = ukf.state_cov, dynamics = ukf.dynamics) where {IPD,IPM,AUGD,AUGM,MF,CF}
    (; dynamics,x,R,weight_params) = ukf
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
        ukf.x = mean_with_weights(mean, xsd,weight_params)[xinds]
        @bangbang ukf.R .= symmetrize(cov_with_weights(cov, [x[xinds] for x in xsd], ukf.x, weight_params)) #+ 1e-16I # TODO: optimize
    else
        ukf.x = mean_with_weights(mean, xsd, weight_params)
        @bangbang ukf.R .= symmetrize(cov_with_weights(cov, xsd, ukf.x, weight_params)) .+ R1
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
    ukf.R1x === nothing || (ukf.R += ukf.R1x)# # Ability to regularize the state covariance when R1 applies to explicit disturbance inputs
    Raug = cat(ukf.R, R1, dims=(1,2))
    sigmapoints!(xsd0, m, Raug, ukf.weight_params, ukf.cholesky!)
    isnan(xsd0[1][1]) && error("Cholesky factorization of augmented state covariance failed at time step $(ukf.t), see https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/parameter_estimation/#Troubleshooting-Kalman-filters for more help. Got Raug = $(printarray(Raug))")
    nothing
end

function sigmapoints_p!(ukf::UnscentedKalmanFilter{<:Any,<:Any,false}, R1)
    sigma_point_cache = ukf.predict_sigma_point_cache
    xsd0 = sigma_point_cache.x0
    sigmapoints!(xsd0, ukf.x, ukf.R, ukf.weight_params, ukf.cholesky!)
    isnan(xsd0[1][1]) && error("Cholesky factorization of state covariance failed at time step $(ukf.t), see https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/parameter_estimation/#Troubleshooting-Kalman-filters for more help. Got R = $(printarray(ukf.R))")
    nothing
end

# The functions below are JET-safe from dynamic dispatch if called with static arrays
function weighted_mean(xs, W::UKFWeights)
    m = xs[1]*W.wm
    for i = 2:length(xs)
        @bangbang m .+= W.wmi .* xs[i]
    end
    m
end

function weighted_cov(xs, m, W::UKFWeights)
    @assert length(m) == length(xs[1])
    X = reduce(hcat, xs) .- m
    @views X[:, 2:end] .*= sqrt(W.wci)
    if W.wc < 0 # In this case we cannot square root the weight parameter
        d1 = (xs[1] .- m)
        R = d1*d1'
        R .*= W.wc
        @views X[:, 1] .= 0
        mul!(R, X, X', true, true)
    else
        @views X[:, 1] .*= sqrt(W.wc)
        R = X * X'
    end
    R
end

function weighted_cov(xs::Vector{<:SVector{N}}, m, W::UKFWeights) where N
    @assert length(m) == length(xs[1])
    if N > 8
        return invoke(weighted_cov, Tuple{Any, Any, typeof(W)}, xs, m, W)
    else
        e = (xs[1] .- m)
        P = (W.wc*e)*e'
        for i in 2:length(xs)
            e = (xs[i] .- m) 
            P += (W.wci*e)*e'
        end
        P
    end
end

function mean_with_weights(mean, xs, weight_params)
    n = length(xs)
    W = UKFWeights(weight_params, ns2L(n))
    mean(xs, W)
end

function cov_with_weights(cov, xs, m, weight_params)
    n = length(xs)
    W = UKFWeights(weight_params, ns2L(n))
    cov(xs, m, W)
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
- `mean`: The function that computes the weighted mean of the output sigma points.
- `cross_cov`: The function that computes the weighted cross-covariance of the state and output sigma points.
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
    mean::MF = measurement_model.mean,
    cross_cov::CCF = measurement_model.cross_cov,
    innovation::IF = measurement_model.innovation,
    measurement = measurement_model.measurement,
) where {MF, CCF, IF}
    sigma_point_cache = measurement_model.cache
    xsm = sigma_point_cache.x0
    ys = sigma_point_cache.x1
    (; x, R) = kf
    sigmapoints_c!(kf, measurement_model, R2) # TODO: should this take other arguments?
    propagate_sigmapoints_c!(kf, u, p, t, R2, measurement_model)
    ym = mean_with_weights(mean, ys, measurement_model.weight_params)
    C  = cross_cov_with_weights(cross_cov, xsm, x, ys, ym, measurement_model.weight_params)
    e  = innovation(y, ym)
    S  = compute_S(measurement_model, R2, ym)
    Sᵪ = cholesky(Symmetric(S); check = false)
    issuccess(Sᵪ) ||
        error("Cholesky factorization of innovation covariance failed at time step $(kf.t), see https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/parameter_estimation/#Troubleshooting-Kalman-filters for more help. Got S = $(printarray(S))")
    K = C / Sᵪ # ns normalization to make it a covariance matrix
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
    chol = hasproperty(kf, :cholesky!) ? kf.cholesky! : cholesky!
    sigmapoints!(xsm, eltype(xsm)(kf.x), kf.R, measurement_model.weight_params, chol)
    isnan(xsm[1][1]) && error("Cholesky factorization of state covariance failed at time step $(kf.t), see https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/parameter_estimation/#Troubleshooting-Kalman-filters for more help. Got R = $(printarray(kf.R))")
    nothing
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
    sigmapoints!(xsm, xm, Raug, measurement_model.weight_params, kf.cholesky!)
    isnan(xsm[1][1]) && error("Cholesky factorization of augmented state covariance failed at time step $(kf.t), see https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/parameter_estimation/#Troubleshooting-Kalman-filters for more help. Got R = $(printarray(R))")
    nothing
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
    nothing
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
    nothing
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
    nothing
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
    nothing
end

function compute_S(measurement_model::UKFMeasurementModel{<:Any, AUGM}, R2, ym) where AUGM
    sigma_point_cache = measurement_model.cache
    ys = sigma_point_cache.x1
    cov = measurement_model.cov
    S = symmetrize(cov_with_weights(cov, ys, ym, measurement_model.weight_params))
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
    cross_cov(xsm, x, ys, y, w::UKFWeights)

Default `measurement_cov` function for `UnscentedKalmanFilter`. Computes the weighted cross-covariance between the state and output sigma points.
"""
function cross_cov(xsm, x, ys, y, W::UKFWeights)
    T = eltype(x)
    nx = length(x)
    ny = length(y)
    xinds = 1:nx
    if x isa SVector
        C = @SMatrix zeros(T,nx,ny)
    else
        C = zeros(T,nx,ny)
    end
    d   = (ys[1]-y) * W.wc
    add_to_C!(C, xsm[1], x, d, xinds)
    @inbounds for i in 2:length(ys) # Cross cov between x and y
        d   = (ys[i]-y) * W.wci
        C = add_to_C!(C, xsm[i], x, d, xinds)
    end
    C
end

function cross_cov_with_weights(cross_cov, xsm, x, ys, y, weight_params)
    n = length(xsm)
    W = UKFWeights(weight_params, ns2L(n))
    cross_cov(xsm, x, ys, y, W)
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
    @views if length(xinds) == length(xsm)
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
        R1 = get_mat(kf.R1, xt[t], u[t], p, tt)
        m = xt[t]
        m̃ = if AUGD
            [m; zeros(size(R1, 1))]
        else
            [m; 0*m]
        end
        P̃ = cat(Rt[t], R1, dims=(1,2))
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
    KalmanSmoothingSolution(sol, xT, RT)
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

