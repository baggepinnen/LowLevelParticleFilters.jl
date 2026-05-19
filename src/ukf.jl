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


"""
    DAEUnscentedKalmanFilter

A nonlinear state estimator for index-1 differential-algebraic (DAE)
systems of the form
```math
ẋ = f(x, z, u, p, t) + w,   w ~ N(0, R₁)
0 = g(x, z, u, p, t)
y = h(x, z, u, p, t) + e,   e ~ N(0, R₂)
```
where `x` is the differential state, `z` is the algebraic state, and `g`
is an index-1 constraint that determines `z` from `x`.

Sigma points are drawn only on the differential state `x`; for each sigma
point, `z` is reconstructed by solving `g(x, z) = 0`, so all sigma points
lie on the constraint manifold. The propagated descriptor is then run
through the user-supplied `dynamics` function. Filter uncertainty
(`R`) is carried over the differential state alone; the full descriptor
`xz` is stored on-manifold at `kf.xz`.

Implementation follows Mandela, Rengaswamy, Narasimhan (2010),
"Nonlinear State Estimation of Differential Algebraic Systems",
*Industrial & Engineering Chemistry Research* 49(11). Process noise is
additive on the differential equation (AUGD=false), and after each
prediction step the sigma points are regenerated from the inflated
covariance `P^{xx} + R₁` before the measurement update (the
`regenerate` field).

# Algorithm
**Predict** (`predict!`)
1. Sample sigma points for `x` from `N(x̂, P^{xx})`.
2. For each, solve `g(xᵢ, z) = 0` to get `zᵢ`; assemble the descriptor.
3. Propagate each descriptor through `dynamics`.
4. Compute the predicted mean and covariance on `x`; add `R₁`.
5. If `regenerate`, redraw sigma points from the inflated covariance and
   re-solve `g` for each.

**Correct** (`correct!`)
6. Apply `h` to each descriptor sigma point to get measurement sigmas.
7. Assemble the innovation covariance `S` and the *augmented*
   cross-covariance `P^{xz,y}` over the full descriptor.
8. Take the differential rows of the gain, update `x` and `P^{xx}`.
9. Reproject `kf.xz` from the updated `x` via the constraint.

# Type parameters
- `IPD`, `IPM`: whether the user-supplied `dynamics` and `measurement`
  functions are in-place (`true`) or out-of-place (`false`); auto-detected
  by the untyped constructor.
- `AUGD`: locked to `false` in the current implementation. AUGD=true (the
  augmented-dynamics variant) is a strictly more general formulation that
  is not Mandela's algorithm; reserved for a future extension.
- `AUGM`: accepted as a type parameter but currently unsupported in
  `correct!` / `sample_measurement` (will error). Plumbing exists for a
  future implementation.

See the `DAEUnscentedKalmanFilter` constructor for keyword arguments and
field semantics.

See also [`UnscentedKalmanFilter`](@ref), [`predict!`](@ref),
[`correct!`](@ref), [`simulate`](@ref).
"""
mutable struct DAEUnscentedKalmanFilter{IPD,IPM,AUGD,AUGM,DT,MT,R1T,D0T,SPC,XT,RT,P,SMT,SCT,CH,WP,XZT,RF,GXZ,BXZ,SOLV,SOLK} <: AbstractUnscentedKalmanFilter
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
    state_mean::SMT
    state_cov::SCT
    cholesky!::CH
    names::SignalNames
    weight_params::WP
    # DAE-specific
    xz::XZT
    xz_sigma_points::Vector{XZT}
    residual::RF
    get_x_z::GXZ
    build_xz::BXZ
    constraint_solve_alg::SOLV
    constraint_solve_kwargs::SOLK
    regenerate::Bool
end

"""
    DAEUnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}(
        dynamics, measurement, residual, get_x_z, build_xz,
        R1, R2, d0;
        xz0, nu, ny, constraint_solve_alg,
        Ts = 1.0,
        constraint_solve_kwargs = (; reltol = 1e-10),
        regenerate = true,
        kwargs...,
    )

Construct a `DAEUnscentedKalmanFilter` with explicit type-parameter flags.
Most users should reach for the untyped form
`DAEUnscentedKalmanFilter(dynamics, measurement, …)`, which auto-detects
`IPD` / `IPM` from the function signatures.

# Callbacks
All operate on the full descriptor `xz = [x; z]` except `residual`, which
sees the decomposed `(x, z)`.

- `dynamics(xz, u, p, t) -> xz_next` (or `(buf, xz, u, p, t)` if `IPD`).
  Advances the descriptor one timestep with the constraint enforced.
- `measurement(xz, u, p, t) -> y` (or in-place if `IPM`).
- `residual(x, z, u, p, t) -> g`. Used to reproject `z` from `x`.
- `get_x_z(xz) -> (x, z)` and `build_xz(x, z) -> xz`.

# Required keyword arguments
- `xz0`: initial descriptor, must satisfy `residual(x̂₀, ẑ₀) ≈ 0`.
- `nu`, `ny`: control and measurement dimensions.
- `constraint_solve_alg`: any SciML-compatible nonlinear algorithm used to
  reproject `z` from `x` (e.g. `SimpleNewtonRaphson()` from
  `SimpleNonlinearSolve`, or `NewtonRaphson()` / `TrustRegion()` from
  `NonlinearSolve`). The package does *not* take a nonlinear-solver
  package as a runtime dependency — the algorithm and any solver-side
  imports are the caller's responsibility.

# Selected keyword arguments
- `constraint_solve_kwargs`: NamedTuple forwarded to `solve`, e.g.
  `(; reltol = 1e-10)`. The reltol should be tighter than the truncation
  error of your `dynamics`.
- `regenerate`: see `DAEUnscentedKalmanFilter` (struct docstring).

See [`DAEUnscentedKalmanFilter`](@ref) for the algorithm, type-parameter
semantics, and field meanings.
"""
function DAEUnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}(
                                  dynamics, # (xz, u, p, t) -> xz_next via DAE solve
                                  measurement, # (xz, u, p, t) -> y
                                  residual, # (x, z, u, p, t) -> constraint residual
                                  get_x_z, # xz -> (x, z)
                                  build_xz, # (x, z) -> xz
                                  R1, # process noise covariance (over x)
                                  R2, # measurement noise covariance (over y)
                                  d0; # initial distribution of differential state x
                                  xz0, # initial full descriptor; must satisfy residual(x̂₀, ẑ₀) ≈ 0
                                  nu::Int, # control dimension
                                  ny::Int, # measurement dimension
                                  Ts::Real = 1.0, # timestep
                                  p = NullParameters(),
                                  state_mean = weighted_mean,
                                  state_cov  = weighted_cov,
                                  cholesky!  = cholesky!,
                                  weight_params = TrivialParams(),
                                  names = default_names(length(d0), nu, ny, "DAEUKF"),
                                  constraint_solve_alg,
                                  constraint_solve_kwargs = (; reltol = 1e-10),
                                  regenerate = true,
                                  kwargs...) where {IPD,IPM,AUGD,AUGM}
    nx = length(d0)
    T  = eltype(d0)
    static = nx ≤ 50
    predict_sigma_point_cache = SigmaPointCache{T}(nx, 0, nx, nx, static)

    # Mirror UKF's unweighted-fallback shim so users can pass mean/cov functions
    # that only take the sigma points; see UnscentedKalmanFilter constructor.
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

    # The measurement model's sigma cache must hold 2·nx+1 entries (matching
    # the differential-state sigma count used by predict!). The cache.x0
    # buffer goes unused for DAE-UKF — we route xz-sized sigma points through
    # the filter's own `xz_sigma_points` field — so its dimension is moot;
    # only the cache.x1 (length-ny, count 2·nx+1) matters for correct!.
    measurement_model = UKFMeasurementModel{T, IPM, AUGM}(
        measurement, R2; nx, ny, kwargs...)

    R  = convert_cov_type(R1, d0.Σ)
    x0 = eltype(predict_sigma_point_cache.x1)(convert_x0_type(d0.μ))

    xz_init = copy(xz0)
    XZT     = typeof(xz_init)
    xz_sigma_points = XZT[copy(xz_init) for _ in 1:(2nx + 1)]

    # Specify *every* type parameter so this dispatches to the default
    # field-by-field inner constructor, not back to this outer method.
    DAEUnscentedKalmanFilter{
        IPD,IPM,AUGD,AUGM,
        typeof(dynamics), typeof(measurement_model), typeof(R1), typeof(d0),
        typeof(predict_sigma_point_cache), typeof(x0), typeof(R), typeof(p),
        typeof(state_mean), typeof(state_cov), typeof(cholesky!),
        typeof(weight_params), typeof(xz_init),
        typeof(residual), typeof(get_x_z), typeof(build_xz),
        typeof(constraint_solve_alg), typeof(constraint_solve_kwargs),
    }(
        dynamics, measurement_model, R1, d0,
        predict_sigma_point_cache, x0, R,
        0, Float64(Ts), ny, nu, p,
        state_mean, state_cov, cholesky!, names, weight_params,
        xz_init, xz_sigma_points,
        residual, get_x_z, build_xz,
        constraint_solve_alg, constraint_solve_kwargs, regenerate,
    )
end

"""
    DAEUnscentedKalmanFilter(dynamics, measurement, residual, get_x_z, build_xz, R1, R2, d0; xz0, nu, ny, kwargs...)

Untyped entry point. Auto-detects `IPD` / `IPM` from the function signatures
and sets `AUGD = AUGM = false`, then forwards to the typed constructor for
the full contract.
"""
function DAEUnscentedKalmanFilter(dynamics, measurement, args...; kwargs...)
    IPD  = !has_oop(dynamics)
    IPM  = !has_oop(measurement)
    AUGD = false
    AUGM = false
    DAEUnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}(dynamics, measurement, args...; kwargs...)
end

# ==============================================================================
# DAE-UKF: constraint reprojection, reset!, predict!, correct!
# Mandela 2010, "Nonlinear State Estimation of Differential Algebraic Systems"
# Additive process noise (AUGD=false) + regenerate. AUGM is a free parameter.
# ==============================================================================

"""
    calc_xz(kf::DAEUnscentedKalmanFilter, xz, u, p, t, xi=get_x_z(xz)[1])
    calc_xz(get_x_z, build_xz, residual, alg, alg_kwargs, xz, u, p, t, xi)

Given a differential state `xi`, solve `residual(xi, z, u, p, t) = 0` for the
algebraic state `z`, and return the full descriptor `build_xz(xi, z)`. The z
slice of the input `xz` is the warm-start guess for the nonlinear solve.
"""
function calc_xz(get_x_z::Function, build_xz, residual, alg, alg_kwargs,
                 xz::AbstractArray, u, p, t, xi)
    _, z0 = get_x_z(xz)
    sol = solve(NonlinearProblem{false}((z,_)->residual(xi, z, u, p, t), z0), alg; alg_kwargs...)
    nr = norm(sol.resid)
    nr < 1e-3 || @warn "DAE-UKF constraint solve residual was large: $nr" maxlog=10
    build_xz(xi, sol.u)
end

calc_xz(kf::DAEUnscentedKalmanFilter, xz, u, p, t, xi = kf.get_x_z(xz)[1]) =
    calc_xz(kf.get_x_z, kf.build_xz, kf.residual,
            kf.constraint_solve_alg, kf.constraint_solve_kwargs,
            xz, u, p, t, xi)


# Forward unknown properties to the measurement model so that calls like
# `kf.R2` work in generic AbstractKalmanFilter code (e.g., forward_trajectory).
# Mirrors the UKF method at the top of this file (~line 254).
function Base.getproperty(ukf::DAEUnscentedKalmanFilter{<:Any,<:Any,AUGD}, s::Symbol) where AUGD
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


"""
    reset!(kf::DAEUnscentedKalmanFilter; x0 = kf.d0.μ, t = 0, xz0 = kf.xz, u = zeros(kf.nu), p = parameters(kf))

Restore the filter to an initial state. The differential mean is reset to
`x0`, the differential covariance to `kf.d0.Σ`, and the time index to `t`.
The algebraic state is then solved for via [`calc_xz`](@ref) using `xz0`
as the warm-start guess, and every entry of `kf.xz_sigma_points` is set
to the resulting on-manifold descriptor.

`u` and `p` are forwarded to the constraint solve and only matter if
`residual` depends on them at the initial step (uncommon for index-1
DAEs).
"""
function reset!(kf::DAEUnscentedKalmanFilter;
                x0 = kf.d0.μ, t = 0, xz0 = kf.xz,
                u = zeros(kf.nu), p = parameters(kf))
    kf.x = convert_x0_type(x0)
    kf.R = convert_cov_type(kf.R1, kf.d0.Σ)
    kf.xz = calc_xz(kf, xz0, u, p, Float64(t), kf.x)
    # Populate sigma points reflecting kf.R so that a `correct!` issued
    # immediately after reset (e.g., the first step of `forward_trajectory`,
    # which calls correct! before predict!) sees a non-degenerate ensemble.
    xs_diff = kf.predict_sigma_point_cache.x0
    sigmapoints!(xs_diff, kf.x, kf.R, kf.weight_params, kf.cholesky!)
    for i in eachindex(xs_diff)
        kf.xz_sigma_points[i] = calc_xz(kf, kf.xz, u, p, Float64(t), xs_diff[i])
    end
    kf.t = t
    nothing
end


"""
    predict!(kf::DAEUnscentedKalmanFilter, u, p = parameters(kf), t = index(kf)*kf.Ts; R1)

Advance the filter one prediction step under control `u` at time `t`.
Mutates `kf` in place: updates `kf.x`, `kf.R`, `kf.xz`, `kf.xz_sigma_points`,
and increments `kf.t`. Returns `nothing`.

# Keyword arguments
- `R1`: process-noise covariance for this step. Defaults to
  `get_mat(kf.R1, kf.x, u, p, t)` so that filters stored with a
  time-varying `R1` work transparently.

See [`DAEUnscentedKalmanFilter`](@ref) for the prediction-step algorithm.
"""
function predict!(kf::DAEUnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}, u,
                  p = parameters(kf), t::Real = index(kf)*kf.Ts;
                  R1 = get_mat(kf.R1, kf.x, u, p, t)) where {IPD,IPM,AUGD,AUGM}
    cache    = kf.predict_sigma_point_cache
    xs_diff  = cache.x0     # differential-state sigma points (nx-sized)
    xs_prop  = cache.x1     # propagated differential-state sigma points
    xzs      = kf.xz_sigma_points

    # Step 1: sigma points on the differential state x.
    sigmapoints!(xs_diff, kf.x, kf.R, kf.weight_params, kf.cholesky!)

    # Step 2: reproject each sigma point onto the constraint manifold.
    for i in eachindex(xs_diff)
        xzs[i] = calc_xz(kf, xzs[i], u, p, t, xs_diff[i])
    end

    # Step 3: propagate the full descriptor through the DAE dynamics.
    if IPD
        for i in eachindex(xzs)
            buf = similar(xzs[i])
            kf.dynamics(buf, xzs[i], u, p, t)
            xzs[i] = buf
        end
    else
        for i in eachindex(xzs)
            xzs[i] = kf.dynamics(xzs[i], u, p, t)
        end
    end

    # Step 4: extract differential parts into the propagated cache.
    for i in eachindex(xzs)
        xs_prop[i] = kf.get_x_z(xzs[i])[1]
    end

    # Step 5: weighted mean / covariance, then add R1 (additive form).
    kf.x = mean_with_weights(kf.state_mean, xs_prop, kf.weight_params)
    @bangbang kf.R .= symmetrize(cov_with_weights(kf.state_cov, xs_prop, kf.x, kf.weight_params)) .+ R1

    # Step 5.5: regenerate sigma points from the inflated covariance R = R̃ + R1
    # and re-reproject z for each. Without this, `correct!` would compute a
    # cross-cov against xz_sigma_points that are inconsistent with the inflated
    # kf.R (Mandela 2010, §3.2).
    if kf.regenerate
        sigmapoints!(xs_diff, kf.x, kf.R, kf.weight_params, kf.cholesky!)
        for i in eachindex(xs_diff)
            xzs[i] = calc_xz(kf, xzs[i], u, p, t, xs_diff[i])
        end
    end

    # Step 6: update the stored descriptor to be on-manifold at x̂⁺.
    kf.xz = calc_xz(kf, kf.xz, u, p, t, kf.x)

    # Step 7: advance time index.
    kf.t += 1
    nothing
end


"""
    (; ll, e, S, Sᵪ, K) = correct!(kf::DAEUnscentedKalmanFilter, u, y, p = parameters(kf), t = index(kf)*kf.Ts; R2)

Run a measurement update against observation `y` and control `u`. Expects
the descriptor sigma points (`kf.xz_sigma_points`) to be populated by a
prior [`predict!`](@ref) call. Mutates `kf.x`, `kf.R`, and `kf.xz` in
place.

# Keyword arguments
- `R2`: measurement-noise covariance for this step. Defaults to
  `get_mat(kf.measurement_model.R2, kf.xz, u, p, t)` to honor time-varying
  `R2`.

# Returns
A NamedTuple `(; ll, e, S, Sᵪ, K)` with:
- `ll`: per-step log-likelihood of `y` under the predicted measurement
  distribution.
- `e`: innovation `y - ŷ`.
- `S`, `Sᵪ`: innovation covariance and its Cholesky factorization.
- `K`: differential-row slice of the augmented Kalman gain (size
  `nx × ny`), suitable for downstream smoothers.

Calling with a filter whose `AUGM` type parameter is `true` errors —
augmented measurements are not yet supported.

See [`DAEUnscentedKalmanFilter`](@ref) for the correction-step algorithm.
"""
function correct!(kf::DAEUnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}, u, y,
                  p = parameters(kf), t::Real = index(kf)*kf.Ts;
                  R2 = get_mat(kf.measurement_model.R2, kf.xz, u, p, t)) where {IPD,IPM,AUGD,AUGM}
    AUGM && error("AUGM=true is not yet supported for DAEUnscentedKalmanFilter")

    mm  = kf.measurement_model
    ys  = mm.cache.x1               # reuse the measurement cache as the y_i buffer
    xzs = kf.xz_sigma_points        # already populated by predict!

    # Step 1: measurement sigma points y_i = h(xz_i, u, p, t).
    if IPM
        for i in eachindex(xzs, ys)
            mm.measurement(ys[i], xzs[i], u, p, t)
        end
    else
        for i in eachindex(xzs, ys)
            ys[i] = mm.measurement(xzs[i], u, p, t)
        end
    end

    # Step 2: predicted measurement mean.
    ym = mean_with_weights(mm.mean, ys, mm.weight_params)

    # Step 3: innovation.
    e = mm.innovation(y, ym)

    # Step 4: innovation covariance S = cov({y_i}) + R2 (AUGM=false branch).
    S = compute_S(mm, R2, ym)

    # Step 5: Cholesky factor of S.
    Sᵪ = cholesky(Symmetric(S); check = false)
    issuccess(Sᵪ) ||
        error("Cholesky factorization of innovation covariance failed at DAE-UKF time step $(kf.t). Got S = $(printarray(S))")

    # Step 6: augmented cross-cov over the full descriptor xz. xz mean and
    # xz_sigma_points are length (nx+nz); add_to_C! handles the equal-length
    # case at src/ukf.jl:843, so C ends up shape (nx+nz, ny) automatically.
    C = cross_cov_with_weights(mm.cross_cov, xzs, kf.xz, ys, ym, mm.weight_params)

    # Step 7: slice gain to differential rows.
    nx = length(kf.x)
    Kx = C[1:nx, :] / Sᵪ

    # Step 8: state and covariance update.
    kf.x = kf.x + Kx * e
    kf.R = symmetrize(kf.R - Kx * S * Kx')

    # Step 9: reproject kf.xz from the updated differential state.
    kf.xz = calc_xz(kf, kf.xz, u, p, t, kf.x)

    # Step 10: log-likelihood (mirrors UKF correct! at src/ukf.jl:669).
    ll = extended_logpdf(SimpleMvNormal(PDMat(S, Sᵪ)), e)

    # Step 11: return the AbstractKalmanFilter correct! contract.
    (; ll, e, S, Sᵪ, K = Kx)
end


# ------------------------------------------------------------------------------
# `simulate` support: sample_state / sample_measurement for DAE-UKF.
# The generic UKF versions at src/ukf.jl:435–446 assume the filter's "state" is
# the plain ODE state; here it's the full descriptor xz. We override both, and
# in the propagation case route noise through the constraint reprojection to
# preserve g(x,z)=0 on every realized step.
# ------------------------------------------------------------------------------

"""
    sample_state(kf::DAEUnscentedKalmanFilter, p = parameters(kf); noise = true)

Draw an initial full-descriptor state for use by [`simulate`](@ref).
Samples (or takes the mean of) the differential prior `kf.d0`, then
projects onto the constraint manifold via [`calc_xz`](@ref). Returns the
full state `xz`.

`u = 0` and `t = 0` are passed to the constraint solve; if your
`residual` depends meaningfully on either at the initial step, sample
manually rather than through `simulate`.
"""
function sample_state(kf::DAEUnscentedKalmanFilter, p = parameters(kf); noise = true)
    x = noise ? rand(kf.d0) : mean(kf.d0)
    # `simulate` doesn't supply a u at the initial step; use a zero placeholder.
    # The z-slice of kf.xz is the warm-start guess for the constraint solve.
    calc_xz(kf, kf.xz, zero(SVector{kf.nu, eltype(x)}), p, 0.0, x)
end

"""
    sample_state(kf::DAEUnscentedKalmanFilter, xz, u, p = parameters(kf), t = index(kf)*kf.Ts; noise = true)

Sample the next full-descriptor state given the current `xz`, control
`u`, and time `t`. Runs `kf.dynamics`, decomposes the result, adds
Gaussian noise `w ∼ N(0, R₁)` to the differential slice when `noise=true`,
then projects back onto the constraint manifold via [`calc_xz`](@ref) so
the returned `xz_next` satisfies `g(x, z) = 0` exactly.

This matches the additive-noise model the filter itself assumes
(Mandela's path): noise lives on the differential equation, never on the
constraint.
"""
function sample_state(kf::DAEUnscentedKalmanFilter, xz, u, p = parameters(kf),
                      t = index(kf)*kf.Ts; noise = true)
    xz_next = kf.dynamics(xz, u, p, t)
    x_next  = kf.get_x_z(xz_next)[1]
    if noise
        w = rand(SimpleMvNormal(get_mat(kf.R1, x_next, u, p, t)))
        x_next = x_next + w
    end
    calc_xz(kf, xz_next, u, p, t, x_next)
end

"""
    sample_measurement(kf::DAEUnscentedKalmanFilter, xz, u, p = parameters(kf), t = index(kf)*kf.Ts; noise = true)

Apply `kf.measurement_model.measurement` to the full descriptor `xz` and
add Gaussian noise `v ∼ N(0, R₂)` when `noise=true`. Used by
[`simulate`](@ref) to synthesize measurements along a trajectory.

Errors if `kf` was constructed with `AUGM = true` — augmented measurements
are not yet supported.
"""
function sample_measurement(kf::DAEUnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}, xz, u,
                            p = parameters(kf), t = index(kf)*kf.Ts;
                            noise = true) where {IPD,IPM,AUGD,AUGM}
    AUGM && error("AUGM=true is not yet supported for DAEUnscentedKalmanFilter")
    R2 = get_mat(kf.measurement_model.R2, xz, u, p, t)
    kf.measurement_model.measurement(xz, u, p, t) .+ noise .* rand(SimpleMvNormal(R2))
end
