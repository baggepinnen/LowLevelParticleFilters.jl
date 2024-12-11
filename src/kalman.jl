abstract type AbstractKalmanFilter <: AbstractFilter end

function convert_cov_type(R1, R)
    if !(eltype(R) <: AbstractFloat)
        R = float.(R)
    end
    if R isa SMatrix || R isa Matrix
        return copy(R)
    elseif R1 isa SMatrix && size(R) == size(R1)
        return SMatrix{size(R1,1),size(R1,2)}(R)
    elseif R1 isa Matrix
        return Matrix(R)
    else
        return Matrix(R)
    end
end
function convert_x0_type(μ)
    if μ isa Vector || μ isa SVector
        return copy(μ)
    else
        return Vector(μ)
    end
end

mutable struct KalmanFilter{AT,BT,CT,DT,R1T,R2T,D0T,XT,RT,TS,P,αT} <: AbstractKalmanFilter
    A::AT
    B::BT
    C::CT
    D::DT
    R1::R1T
    R2::R2T
    d0::D0T
    x::XT
    R::RT
    t::Int
    Ts::TS
    p::P
    α::αT
end


"""
    KalmanFilter(A,B,C,D,R1,R2,d0=MvNormal(R1); p = NullParameters(), α=1, check=true)

The matrices `A,B,C,D` define the dynamics
```
x' = Ax + Bu + w
y  = Cx + Du + e
```
where `w ~ N(0, R1)`, `e ~ N(0, R2)` and `x(0) ~ d0`

The matrices can be time varying such that, e.g., `A[:, :, t]` contains the ``A`` matrix at time index `t`.
They can also be given as functions on the form
```
Afun(x, u, p, t) -> A
```
For maximum performance, provide statically sized matrices from StaticArrays.jl

α is an optional "forgetting factor", if this is set to a value > 1, such as 1.01-1.2, the filter will, in addition to the covariance inflation due to ``R_1``, exhibit "exponential forgetting" similar to a [Recursive Least-Squares (RLS) estimator](https://en.wikipedia.org/wiki/Recursive_least_squares_filter). It is thus possible to get a RLS-like algorithm by setting ``R_1=0, R_2 = 1/α`` and ``α > 1`` (``α`` is the inverse of the traditional RLS parameter ``α = 1/λ``). The exact form of the covariance update is
```math
R(t+1|t) = α AR(t)A^T + R_1
```

If `check = true (default)` the function will check that the eigenvalues of `A` are less than 2 in absolute value. Large eigenvalues may be an indication that the system matrices are representing a continuous-time system and the user has forgotten to discretize it. Turn off this check by setting `check = false`.

# Tutorials on Kalman filtering
The tutorial ["How to tune a Kalman filter"](https://juliahub.com/pluto/editor.html?id=ad9ecbf9-bf83-45e7-bbe8-d2e5194f2240) details how to figure out appropriate covariance matrices for the Kalman filter, as well as how to add disturbance models to the system model. See also the [tutorial in the documentation](https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/adaptive_kalmanfilter/)
"""
function KalmanFilter(A,B,C,D,R1,R2,d0=SimpleMvNormal(Matrix(R1)); Ts = 1, p = NullParameters(), α = 1.0, check = true)
    if check
        α ≥ 1 || @warn "α should be > 1 for exponential forgetting. An α < 1 will lead to exponential loss of adaptation over time."
        maximum(abs, eigvals(A isa SMatrix ? Matrix(A) : A)) ≥ 2 && @warn "The dynamics matrix A has eigenvalues with absolute value ≥ 2. This is either a highly unstable system, or you have forgotten to discretize a continuous-time model. If you are sure that the system is provided in discrete time, you can disable this warning by setting check=false." maxlog=1
    end
    if D == 0
        D = zeros(eltype(A), size(C,1), size(B,2))
    end
    R = convert_cov_type(R1, d0.Σ)
    x0 = convert_x0_type(d0.μ)
    KalmanFilter(A,B,C,D,R1,R2, d0, x0, R, 0, Ts, p, α)
end

function Base.propertynames(kf::KF, private::Bool=false) where KF <: AbstractKalmanFilter
    return fieldnames(KF)
end


function Base.getproperty(kf::AbstractKalmanFilter, s::Symbol)
    s ∈ fieldnames(typeof(kf)) && return getfield(kf, s)
    if s === :nu
        return size(kf.B, 2)
    elseif s === :ny
        return size(kf.R2, 1)
    elseif s === :nx
        return size(kf.R1, 1)
    else
        throw(ArgumentError("$(typeof(kf)) has no property named $s"))
    end
end

sample_state(kf::AbstractKalmanFilter, p=parameters(kf); noise=true) = noise ? rand(kf.d0) : mean(kf.d0)
sample_state(kf::AbstractKalmanFilter, x, u, p=parameters(kf), t=0; noise=true) = kf.A*x .+ kf.B*u .+ noise*rand(SimpleMvNormal(get_mat(kf.R1, x, u, p, t)))
sample_measurement(kf::AbstractKalmanFilter, x, u, p=parameters(kf), t=0; noise=true) = kf.C*x .+ kf.D*u .+ noise*rand(SimpleMvNormal(get_mat(kf.R2, x, u, p, t)))
particletype(kf::AbstractKalmanFilter) = typeof(kf.x)
covtype(kf::AbstractKalmanFilter)      = typeof(kf.R)
state(kf::AbstractKalmanFilter)        = kf.x
covariance(kf::AbstractKalmanFilter)   = kf.R
function measurement(kf::AbstractKalmanFilter)
    function (x,u,p,t)
        y = get_mat(kf.C, x, u, p, t)*x
        if !(isa(kf.D, Union{Number, AbstractArray}) && iszero(kf.D))
            y .+= get_mat(kf.D, x, u, p, t)*u
        end
        y
    end
end

# This helper struct is used to return a oop measurement function regardless of how the measurement function is defined
struct MeasurementOop
    kf::AbstractKalmanFilter
end

function (kfm::MeasurementOop)(x,u,p,t)
    kf = kfm.kf
    mfun = measurement(kf)
    if kf isa UnscentedKalmanFilter{<:Any,true} || kf isa ExtendedKalmanFilter{<:Any,true}
        y = zeros(kf.ny)
        mfun(y,x,u,p,t)
        return y
    else
        return mfun(x,u,p,t)
    end
end

function measurement_oop(kf::AbstractKalmanFilter)
    MeasurementOop(kf)
end

function dynamics(kf::AbstractKalmanFilter)
    (x,u,p,t) -> get_mat(kf.A, x, u, p, t)*x + get_mat(kf.B, x, u, p, t)*u
end

"""
    reset!(kf::AbstractKalmanFilter; x0)

Reset the initial distribution of the state. Optionally, a new mean vector `x0` can be provided.
"""
function reset!(kf::AbstractKalmanFilter; x0 = kf.d0.μ)
    kf.x = convert_x0_type(x0)
    kf.R = convert_cov_type(kf.R1, kf.d0.Σ)# typeof(kf.R1)(kf.d0.Σ)
    kf.t = 0
end
