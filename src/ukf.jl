function sigmapoints(m, Σ)
    T = promote_type(eltype(m), eltype(Σ))
    n = max(length(m), size(Σ,1))
    xs = [@SVector zeros(T, n) for _ in 1:(2n+1)]
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

@with_kw struct UnscentedKalmanFilter{DT,MT,R1T,R2T,R2DT,D0T,VT,XT,RT} <: AbstractUnscentedKalmanFilter
    dynamics::DT
    measurement::MT
    R1::R1T
    R2::R2T
    R2d::R2DT
    d0::D0T
    xs::Vector{VT}
    x::XT
    R::RT
    t::Ref{Int} = Ref(1)
end


"""
    UnscentedKalmanFilter(dynamics,measurement,R1,R2,d0=MvNormal(Matrix(R1)))
"""
function UnscentedKalmanFilter(dynamics,measurement,R1,R2,d0=MvNormal(Matrix(R1)))
    try
        cR1 = cond(R1)
        cR2 = cond(R2)
        (cond(cR1) > 1e8 || cond(cR2) > 1e8) && @warn("Covariance matrices are poorly conditioned")
    catch
        nothing
    end
    n = size(R1,1)
    p = size(R2,1)
    R1s = SMatrix{n,n}(R1)
    R2s = SMatrix{p,p}(R2)
    xs = sigmapoints(mean(d0), cov(d0))
    UnscentedKalmanFilter(dynamics,measurement,R1s,R2s,MvNormal(Matrix(R2s)), d0, xs, Vector(d0.μ), Matrix(d0.Σ), Ref(1))
end

sample_state(kf::AbstractUnscentedKalmanFilter) = rand(kf.d0)
sample_state(kf::AbstractUnscentedKalmanFilter, x, u, t) = kf.dynamics(x,u,t) .+ rand(MvNormal(Matrix(kf.R1)))
sample_measurement(kf::AbstractUnscentedKalmanFilter, x, u, t) = kf.measurement(x, u, t) .+ rand(MvNormal(Matrix(kf.R2)))
measurement(kf::AbstractUnscentedKalmanFilter) = kf.measurement
dynamics(kf::AbstractUnscentedKalmanFilter) = kf.dynamics


function predict!(ukf::UnscentedKalmanFilter, u, t::Integer = index(ukf))
    @unpack dynamics,measurement,x,xs,R,R1 = ukf
    ns = length(xs)
    sigmapoints!(xs,x,R) # TODO: these are calculated in the update step
    for i in eachindex(xs)
        xs[i] = dynamics(xs[i], u, t)
    end
    x .= mean(xs)
    R .= symmetrize(cov(xs)) + R1
    ukf.t[] += 1
end



correct!(ukf::UnscentedKalmanFilter, y, t::Integer = index(ukf)) = correct!(ukf, y, 0, t)
function correct!(ukf::UnscentedKalmanFilter, u, y, t::Integer = index(ukf))
    @unpack measurement,x,xs,R,R1,R2,R2d = ukf
    n = size(R1,1)
    p = size(R2,1)
    ns = length(xs)
    sigmapoints!(xs,x,R) # Update sigmapoints here since untransformed points required
    C = @SMatrix zeros(n,p)
    ys = map(xs) do x
        measurement(x, u, t)
    end
    ym = mean(ys)
    @inbounds for i in eachindex(ys) # Cross cov between x and y
        d   = ys[i]-ym
        ca  = (xs[i]-x)*d'
        C  += ca
    end
    e   = y .- ym
    S   = symmetrize(cov(ys)) + R2 # cov of y
    Sᵪ  = cholesky(S)
    K   = (C./ns)/Sᵪ # ns normalization to make it a covariance matrix
    x .+= K*e
    # mul!(x, K, e, 1, 1) # K and e will be SVectors if ukf correctly initialized
    RmKSKT!(R, K, S)
    ll = logpdf(MvNormal(PDMat(S,Sᵪ)), e) #- 1/2*logdet(S) # logdet is included in logpdf
    ll, e
end

@inline function RmKSKT!(R, K, S)
    R .-= K*S*K'
    symmetrize(R)
    nothing
end


## DAE UKF
#= 
Nonlinear State Estimation of Differential Algebraic Systems

Ravi K. Mandela, Raghunathan Rengaswamy, Shankar Narasimhan

First, unscented samples are chosen for the differential
states. The unscented samples for the algebraic states
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

using NonlinearSolve
abstract type AbstractUnscentedKalmanFilter <: AbstractKalmanFilter end

@with_kw struct DAEUnscentedKalmanFilter{DT,MT,R1T,R2T,R2DT,D0T,VT,XT,RT,G,GXZ,BXZ,XZT,VZT} <: AbstractUnscentedKalmanFilter
    ukf::UnscentedKalmanFilter{DT,MT,R1T,R2T,R2DT,D0T,VT,XT,RT}
    g::G
    get_x_z::GXZ
    build_xz::BXZ
    xz::XZT
    xzs::Vector{VZT}
    nu::Int
    threads::Bool
    # TODO: root solver options
end



"""
    DAEUnscentedKalmanFilter(ukf; g, get_x_z, build_xz, xz0, threads=false)

An Unscented Kalman filter for differential-algebraic systems (DAE).

Ref: "Nonlinear State Estimation of Differential Algebraic Systems", 
Mandela, Rengaswamy, Narasimhan

# Arguments
- `ukf` is a regular [`UnscentedKalmanFilter`](@ref) that contains `dynamics(xz, u, t)` that propagates the combined state `xz(k)` to `xz(k+1)` and a measurement function with signature `(xz, u, t)`
- `g(x, z, u, t)` is a function that should fulfill `g(x, z, u, t) = 0`
_ `get_x_z(xz) -> x, z` is a function that decomposes `xz` into `x` and `z`
- `build_xz(x, z)` is the inverse of `get_x_z`
- `xz0` the initial full state.
- `threads`: If true, evaluates dynamics on Sigma points in parallel. This typically requires the dynamics to be non-allocating (use StaticArrays) to improve performance. 

# Assumptions
- The DAE dynamics is index 1 and can be written on the form 
```
ẋ = f(x, z, u, t) # Differential equations
0 = g(x, z, u, t) # Algebraic equations
y = h(x, z, u, t) # Measurements
```
the measurements may be functions of both differential states `x` and algebraic variables `z`. Note, the actual dynamcis and measurement functions stored in the internal `ukf` should have signatures `(xz, u, t)`, i.e., they take the combined state containing both `x` and `z` in a single vector as dictated by the function `build_xz`. It is only the function `g` that is assumed to actually have the signature `g(x,z,u,t)`.
"""
function DAEUnscentedKalmanFilter(ukf; g, get_x_z, build_xz, xz0, nu::Int, threads::Bool=false)
    T = eltype(ukf.xs[1])
    n = length(ukf.x)
    xzs = [@SVector zeros(T, length(xz0)) for _ in 1:(2n+1)] # These vectors have the length of xz0 but the number of them is determined by the dimension of x only
    DAEUnscentedKalmanFilter(ukf, g, get_x_z, build_xz, copy(xz0), xzs, nu, threads)
end


function Base.getproperty(ukf::DAEUnscentedKalmanFilter, s::Symbol)
    s ∈ fieldnames(typeof(ukf)) && return getfield(ukf, s)
    getproperty(getfield(ukf, :ukf), s) # Forward to inner filter
end

Base.propertynames(ukf::DAEUnscentedKalmanFilter) = (fieldnames(typeof(ukf))..., propertynames(getfield(ukf, :ukf))...)

state(ukf::DAEUnscentedKalmanFilter) = ukf.xz

function sample_state(kf::DAEUnscentedKalmanFilter)
    @unpack get_x_z, build_xz, xz, g, dynamics, R1, nu = kf
    xh = rand(kf.d0)
    calc_xz(get_x_z, build_xz, g, xz, zeros(nu), 0, xh)
end
function sample_state(kf::DAEUnscentedKalmanFilter, x, u, t)
    @unpack get_x_z, build_xz, xz, g, dynamics, R1 = kf
    xh = get_x_z(dynamics(x,u,t))[1] .+ rand(MvNormal(Matrix(R1)))
    calc_xz(get_x_z, build_xz, g, xz, u, t, xh)
end

"""
    calc_xz(dae_ukf, xz, u, t, x=get_x_z(xz)[1])
    calc_xz(get_x_z, build_xz, g, xz, u, t, x=get_x_z(xz)[1])

Find `z` such that g(x, z) = 0 (zeros of length(x) + length(z))
The z part of xz is used as initial guess

# Arguments:
- `x`: If not provided, x from xz will be used
- `xz`: Full state
"""
function calc_xz(get_x_z::Function, build_xz, g, xz::AbstractArray, u, t, xi=get_x_z(xz)[1])
    _, z0 = get_x_z(xz) # use previous z as initial guess for root finder
    sol = solve(NonlinearProblem{false}((z,_)->g(xi, z, u, t), z0), NewtonRaphson(), tol=1e-9) # function takes parameter as second arg
    nr = norm(sol.resid)
    nr < 1e-3 || @warn "Root solving residual was large $nr" maxlog=10
    zi = sol.u
    build_xz(xi, zi)
end

calc_xz(ukf::DAEUnscentedKalmanFilter, args...) = 
    calc_xz(ukf.get_x_z, ukf.build_xz, ukf.g, args...)
using Polyester
function predict!(ukf::DAEUnscentedKalmanFilter, u, t::Integer = index(ukf))
    @unpack dynamics,measurement,x,xs,xz,xzs,R,R1,g,build_xz,get_x_z = ukf
    ns = length(xs)
    sigmapoints!(xs,x,R) # generate only for x
    if ukf.threads
        @batch for i in eachindex(xs)
            # generate z
            xzi = calc_xz(ukf, xzs[i], u, t, xs[i])
            xzs[i] = dynamics(xzi, u, t) # here they must be the same and in the correct order
            xs[i],_ = get_x_z(xzs[i])
        end
    else
        for i in eachindex(xs)
            # generate z
            xzi = calc_xz(ukf, xzs[i], u, t, xs[i])
            xzs[i] = dynamics(xzi, u, t) # here they must be the same and in the correct order
            xs[i],_ = get_x_z(xzs[i])
        end
    end
    x .= mean(xs) # xz or xs here? Answer: Covariance is associated only with x
    xz .= calc_xz(ukf, xz, u, t, x)
    R .= symmetrize(cov(xs)) + R1
    ukf.t[] += 1
end

function correct!(ukf::DAEUnscentedKalmanFilter, u, y, t::Integer = index(ukf))
    @unpack measurement,x,xs,xz,xzs,R,R1,R2,R2d,g,get_x_z,build_xz  = ukf
    n = size(R1,1)
    p = size(R2,1)
    ns = length(xs)
    sigmapoints!(xs,x,R) # Update sigmapoints here since untransformed points required
    for i in eachindex(xs)
        xzs[i] = calc_xz(ukf, xzs[i], u, t, xs[i])
    end
    C = @SMatrix zeros(n,p)
    ys = map(xzs) do xzi
        measurement(xzi, u, t)
    end
    ym = mean(ys)
    @inbounds for i in eachindex(ys) # Cross cov between x and y
        d   = ys[i]-ym
        xi,_ = get_x_z(xzs[i])
        ca  = (xi-x)*d'
        C  += ca
    end
    e   = y .- ym
    S   = symmetrize(cov(ys)) + R2 # cov of y
    Sᵪ = cholesky(S)
    K   = (C./ns)/Sᵪ # ns normalization to make it a covariance matrix
    x .+= K*e
    xz .= calc_xz(ukf, xz, u, t, x)
    # mul!(x, K, e, 1, 1) # K and e will be SVectors if ukf correctly initialized
    RmKSKT!(R, K, S)
    ll = logpdf(MvNormal(PDMat(S,Sᵪ)), e) #- 1/2*logdet(S) # logdet is included in logpdf
    ll, e
end