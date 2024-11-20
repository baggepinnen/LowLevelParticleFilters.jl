"""
    sigmapoints(m, Σ)

Return a vector of (2n+1) static vectors, where `n` is the length of `m`, representing sigma points with mean `m` and covariance `Σ`.
"""
@inline function sigmapoints(m, Σ; static = true)
    T = promote_type(eltype(m), eltype(Σ))
    n = max(length(m), size(Σ,1))
    if static
        xs = [@SVector zeros(T, n) for _ in 1:(2n+1)]
    else
        xs = [zeros(T, n) for _ in 1:(2n+1)]
    end
    sigmapoints!(xs,m,Σ)
end

function sigmapoints!(xs, m, Σ::AbstractMatrix)
    n = length(xs[1])
    @assert n == length(m)
    # X = sqrt(Symmetric(n*Σ)) # 2.184 μs (16 allocations: 2.27 KiB)
    X = cholesky!(Symmetric(n*Σ)).L # 170.869 ns (3 allocations: 176 bytes)
    @inbounds @views for i in 1:n
        xs[i] = X[:,i]
        xs[i+n] = -xs[i] .+ m
        xs[i] = xs[i] .+ m
    end
    xs[end] = m
    xs
end

# UKF ==========================================================================

abstract type AbstractUnscentedKalmanFilter <: AbstractKalmanFilter end

@with_kw mutable struct UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM,DT,MT,R1T,R2T,D0T,XD,XD0,XM,Y,XT,RT,P,RJ} <: AbstractUnscentedKalmanFilter
    dynamics::DT
    measurement::MT
    R1::R1T
    R2::R2T
    d0::D0T
    "Sigma points after dynamics update"
    xsd::Vector{XD}
    "Sigma points before dynamics update"
    xsd0::Vector{XD0}
    "Sigma points before measurement update"
    xsm::Vector{XM}
    "Sigma points after measurement update"
    ys::Vector{Y}
    x::XT
    R::RT
    t::Int = 0
    Ts::Float64 = 1.0
    ny::Int
    nu::Int
    p::P
    reject::RJ = nothing
end


"""
    UnscentedKalmanFilter(dynamics, measurement, R1, R2, d0=MvNormal(Matrix(R1)); p = NullParameters(), ny, nu)

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
UnscentedKalmanFilter{inplace_dynamics,inplace_measurement,augmented_dynamics,augmnented_measurement}(...)
```
is used, where the Boolean type parameters have the following meaning
- `inplace_dynamics`: If `true`, the dynamics function operates in-place, i.e., it modifies the first argument in `dynamics(dx, x, u, p, t)`. Default is `false`.
- `inplace_measurement`: If `true`, the measurement function operates in-place, i.e., it modifies the first argument in `measurement(y, x, u, p, t)`. Default is `false`.
- `augmented_dynamics`: If `true` the dynamics function is augmented with an additional noise input `w`, i.e., `dynamics(x, u, p, t, w)`. Default is `false`.
- `augmnented_measurement`: If `true` the measurement function is agumented with an additional noise input `e`, i.e., `measurement(x, u, p, t, e)`. Default is `false`.

Use of augmented dynamics incurs extra computational cost. The number of sigma points used is `2L+1` where `L` is the length of the augmented state vector. Without augmentation, `L = nx`, with augmentation `L = nx + nw` and `L = nx + ne` for dynamics and measurement, respectively.

# Sigma-point rejection
For problems with challenging dynamics, a mechanism for rejection of sigma points after the dynamics update is provided. A function `reject(x) -> Bool` can be provided through the keyword argument `reject` that returns `true` if a sigma point for ``x(t+1)`` should be rejected, e.g., if an instability or non-finite number is detected. A rejected point is replaced by the propagated mean point (the mean point cannot be rejected). This function may be provided either to the constructor of the UKF or passed to the [`predict!`](@ref) function.
"""
function UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}(dynamics,measurement,R1,R2,d0=SimpleMvNormal(R1); Ts = 1.0, p = NullParameters(), nu::Int, ny::Int, reject=nothing) where {IPD,IPM,AUGD,AUGM}
    nx = length(d0)
    nw = size(R1, 1) # nw may be smaller than nx for augmented dynamics
    ne = size(R2, 1)
    T = promote_type(eltype(d0), eltype(R1), eltype(R2))
    if AUGD
        L = nx + nw
        xsd0 = [@SVector zeros(T, nx+nw) for _ in 1:2L+1]
        if IPD
            xsd = [zeros(T, nx) for _ in 1:2L+1]
        else
            xsd = [@SVector zeros(T, nx) for _ in 1:2L+1]
        end
    else
        L = nx
        xsd0 = [@SVector zeros(T, nx) for _ in 1:2L+1]
        if IPD
            xsd = Vector.(xsd0)
        else
            xsd = xsd0
        end
    end
    if AUGM
        L = nx + ne
        xsm = [@SVector zeros(T, nx+ne) for _ in 1:2L+1]
    else
        L = nx
        xsm = [@SVector zeros(T, nx) for _ in 1:2L+1]
    end
    if IPM
        ys = [zeros(T, ny) for _ in 1:2L+1]
    else
        ys = [@SVector zeros(T, ny) for _ in 1:2L+1]
    end
    R = convert_cov_type(R1, d0.Σ)
    x0 = convert_x0_type(d0.μ)
    UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM, typeof(dynamics), typeof(measurement), typeof(R1), typeof(R2), typeof(d0),
        typeof(xsd[1]), typeof(xsd0[1]), typeof(xsm[1]), typeof(ys[1]),
        typeof(x0), typeof(R), typeof(p), typeof(reject)}(
            dynamics,measurement,R1,R2, d0, xsd,xsd0,xsm,ys, x0, R, 0, Ts, ny, nu, p, reject)
end

function UnscentedKalmanFilter(dynamics,measurement,args...;kwargs...)
    IPD = has_ip(dynamics)
    IPM = has_ip(measurement)
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

function predict!(ukf::UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}, u, p = parameters(ukf), t::Real = index(ukf)*ukf.Ts; R1 = get_mat(ukf.R1, ukf.x, u, p, t), reject = ukf.reject) where {IPD,IPM,AUGD,AUGM}
    @unpack dynamics,measurement,x,xsd,xsd0,R = ukf
    xtyped = eltype(xsd)(x)
    nx = length(x)
    nw = size(R1, 1) # nw may be smaller than nx for augmented dynamics
    if AUGD
        xinds = 1:nx
        winds = nx+1:nx+nw
        m = [xtyped; 0*R1[:, 1]]
        S = cat(R, R1, dims=(1,2))
        sigmapoints!(xsd0,m,S)
    else
        sigmapoints!(xsd0,xtyped,R)
    end
    if IPD
        AUGD && error("IPD and AUGD not yet supported")
        xp = similar(xsd[1])
        for i in eachindex(xsd)
            xp .= 0
            dynamics(xp, xsd0[i], u, p, t)
            xsd[i] .= xp
        end
    else
        for i in eachindex(xsd)
            if AUGD
                xsd[i] = dynamics(xsd0[i][xinds], u, p, t, xsd0[i][winds])
            else
                xsd[i] = dynamics(xsd0[i], u, p, t)
            end
        end
    end
    if reject !== nothing
        for i = 2:length(xsd)
            if reject(xsd[i])
                # @info "rejecting $(xsd[i]) at time $t"
                @bangbang xsd[i] .= xsd[1]
            end
        end
    end
    if AUGD
        ukf.x = safe_mean(xsd)[xinds]
        @bangbang ukf.R .= symmetrize(safe_cov(xsd)[xinds,xinds]) # TODO: optimize
    else
        ukf.x = safe_mean(xsd)
        @bangbang ukf.R .= symmetrize(safe_cov(xsd)) .+ R1
    end
    ukf.t += 1
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

safe_cov(xs) = cov(xs)
function safe_cov(xs::Vector{<:SVector})
    m = safe_mean(xs)
    P = 0 .* m*m'
    for i in eachindex(xs)
        e = xs[i] .- m
        P += e*e'
    end
    c = P ./ (length(xs) - 1)
    c
end



function correct!(ukf::UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}, u, y, p=parameters(ukf), t::Real = index(ukf)*ukf.Ts; R2 = get_mat(ukf.R2, ukf.x, u, p, t)) where {IPD,IPM,AUGD,AUGM}
    @unpack measurement,x,xsm,ys,R,R1 = ukf
    nx = length(x)
    L = length(xsm[1])
    nv = size(R2, 1)
    ny = length(y)
    ns = length(xsm)
    if AUGM
        xinds = 1:nx
        vinds = nx+1:nx+nv
        xm = [x; 0*R2[:, 1]]
        S = cat(R, R2, dims=(1,2))
        sigmapoints!(xsm,xm,S)
    else
        xm = x
        sigmapoints!(xsm,eltype(xsm)(x),R) # Update sigmapoints here since untransformed points required
    end
    for i = eachindex(xsm,ys)
        if IPM
            measurement(ys[i], xsm[i], u, p, t)
        else
            if AUGM
                ys[i] = measurement(xsm[i][xinds], u, p, t, xsm[i][vinds])
            else
                ys[i] = measurement(xsm[i], u, p, t)
            end
        end
    end
    ym = safe_mean(ys)
    C = @SMatrix zeros(nx,ny)
    @inbounds for i in eachindex(ys) # Cross cov between x and y
        d   = ys[i]-ym
        if AUGM
            ca  = (xsm[i][xinds]-x)*d'
        else
            ca  = (xsm[i]-x)*d'
        end
        C  += ca
    end
    e   = y .- ym
    if AUGM
        S   = symmetrize(safe_cov(ys))
    else
        S   = symmetrize(safe_cov(ys)) + R2 # cov of y
    end
    Sᵪ  = cholesky(S)
    K   = (C./(ns-1))/Sᵪ # ns normalization to make it a covariance matrix
    ukf.x += K*e
    # mul!(x, K, e, 1, 1) # K and e will be SVectors if ukf correctly initialized
    if R isa SMatrix
        ukf.R = symmetrize(R - K*S*K')
    else
        RmKSKT!(R, K, S)
    end
    ll = extended_logpdf(SimpleMvNormal(PDMat(S,Sᵪ)), e) #- 1/2*logdet(S) # logdet is included in logpdf
    (; ll, e, S, Sᵪ, K)
end

@inline function RmKSKT!(R, K, S)
    R .-= K*S*K'
    symmetrize(R)
    nothing
end


function smooth(sol::KalmanFilteringSolution, kf::UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}, u::AbstractVector, y::AbstractVector, p=parameters(kf)) where {IPD,IPM,AUGD,AUGM}
    # ref: "Unscented Rauch–Tung–Striebel Smoother" Simo Särkkä
    (; x,xt,R,Rt,ll) = sol
    T            = length(y)
    xT           = similar(xt)
    RT           = similar(Rt)
    xT[end]      = xt[end]      |> copy
    RT[end]      = Rt[end]      |> copy
    ny = kf.ny
    nx = length(x[1])
    xi = 1:nx
    for t = T-1:-1:1
        tt = (t-1)*kf.Ts
        m = xt[t]
        m̃ = [m; 0*m]
        P̃ = cat(Rt[t], get_mat(kf.R1, xt[t], u[t], p, tt), dims=(1,2))
        X̃ = sigmapoints(m̃, P̃)
        X̃⁻ = map(X̃) do xq
            if AUGD
                if IPD
                    xd = similar(xq[xi]) .= 0
                    kf.dynamics(xd, xq[xi], u[t], p, tt, xq[nx+1:end])
                    xd
                else
                    kf.dynamics(xq[xi], u[t], p, tt, xq[nx+1:end])
                end
            else
                if IPD
                    xd = similar(xq) .= 0
                    kf.dynamics(xd, xq[xi], u[t], p, tt) + xq[nx+1:end]
                    xd
                else
                    kf.dynamics(xq[xi], u[t], p, tt) + xq[nx+1:end]
                end
            end
        end
        m⁻ = mean(X̃⁻)
        P⁻ = @SMatrix zeros(nx,nx)
        for i in eachindex(X̃⁻)
            e = (X̃⁻[i] - m⁻)
            P⁻ += e*e'
        end
        ns = length(X̃⁻)-1
        P⁻ = P⁻ ./ ns
        C = @SMatrix zeros(nx,nx)
        for i in eachindex(X̃⁻)
            C += (X̃[i][xi] - m)*(X̃⁻[i][xi] - m⁻)'
        end
        C = C ./ ns
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
