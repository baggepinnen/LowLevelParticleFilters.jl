
const StaticCovMat = Union{SMatrix, UpperTriangular{<:Any, <:SMatrix}}

function convert_cov_type(R1, R)
    if !(eltype(R) <: AbstractFloat)
        R = float.(R)
    end
    if (R isa StaticCovMat) || (R isa Matrix)
        return copy(R)
    elseif (R1 isa StaticCovMat) && size(R) == size(R1)
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
    nx::Int
    nu::Int
    ny::Int
    names::SignalNames
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
When providing functions, the dimensions of the state, input and output, `nx, nu, ny` must be provided as keyword arguments to the `KalmanFilter` constructor since these cannot be inferred from the function signature.
For maximum performance, provide statically sized matrices from StaticArrays.jl

α is an optional "forgetting factor", if this is set to a value > 1, such as 1.01-1.2, the filter will, in addition to the covariance inflation due to ``R_1``, exhibit "exponential forgetting" similar to a [Recursive Least-Squares (RLS) estimator](https://en.wikipedia.org/wiki/Recursive_least_squares_filter). It is thus possible to get a RLS-like algorithm by setting ``R_1=0, R_2 = 1/α`` and ``α > 1`` (``α`` is the inverse of the traditional RLS parameter ``α = 1/λ``). The exact form of the covariance update is
```math
R(t+1|t) = α AR(t)A^T + R_1
```

If `check = true (default)` the function will check that the eigenvalues of `A` are less than 2 in absolute value. Large eigenvalues may be an indication that the system matrices are representing a continuous-time system and the user has forgotten to discretize it. Turn off this check by setting `check = false`.

# Tutorials on Kalman filtering
The tutorial ["How to tune a Kalman filter"](https://juliahub.com/pluto/editor.html?id=ad9ecbf9-bf83-45e7-bbe8-d2e5194f2240) details how to figure out appropriate covariance matrices for the Kalman filter, as well as how to add disturbance models to the system model. See also the [tutorial in the documentation](https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/adaptive_kalmanfilter/)
"""
function KalmanFilter(A,B,C,D,R1,R2,d0=SimpleMvNormal(R1); Ts = 1, p = NullParameters(), α = 1.0, check = true, nx = length(d0), ny = size(C,1), nu = size(B,2), names = default_names(length(d0), nu, ny, "KF"))
    if check
        α ≥ 1 || @warn "α should be > 1 for exponential forgetting. An α < 1 will lead to exponential loss of adaptation over time."
        (A isa AbstractMatrix) && maximum(abs, eigvals(A isa SMatrix ? Matrix(A) : A)) ≥ 2 && @warn "The dynamics matrix A has eigenvalues with absolute value ≥ 2. This is either a highly unstable system, or you have forgotten to discretize a continuous-time model. If you are sure that the system is provided in discrete time, you can disable this warning by setting check=false." maxlog=1
    end
    R = convert_cov_type(R1, d0.Σ)
    x0 = convert_x0_type(d0.μ)
    if D == 0
        D = zeros(eltype(x0), ny, nu)
    end
    KalmanFilter(A,B,C,D,R1,R2, d0, x0, R, 0, Ts, p, α, nx, nu, ny, names)
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
sample_state(kf::AbstractKalmanFilter, x, u, p=parameters(kf), t=0; noise=true) = get_mat(kf.A, u,u,p,t)*x .+ get_mat(kf.B, u,u,p,t)*u .+ noise*rand(SimpleMvNormal(get_mat(kf.R1, x, u, p, t)))
sample_measurement(kf::AbstractKalmanFilter, x, u, p=parameters(kf), t=0; noise=true) = get_mat(kf.C, u,u,p,t)*x .+ get_mat(kf.D, u,u,p,t)*u .+ noise*rand(SimpleMvNormal(get_mat(kf.R2, x, u, p, t)))
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
    kf::AbstractFilter
end

function (kfm::MeasurementOop)(x,u,p,t)
    kf = kfm.kf
    mfun = measurement(kf)
    if kf isa UnscentedKalmanFilter{<:Any,true, <:Any, true} # augmented inplace
        y = zeros(kf.ny)
        mfun(y,x,u,p,t,0)
        return y
    elseif kf isa UnscentedKalmanFilter{<:Any,false, <:Any, true} # augmented oop
        return mfun(x,u,p,t,0)
    elseif kf isa UnscentedKalmanFilter{<:Any,true} || kf isa ExtendedKalmanFilter{<:Any,true} ||  kf isa SqExtendedKalmanFilter{<:Any,true}
        y = zeros(kf.ny)
        mfun(y,x,u,p,t)
        return y
    else
        return mfun(x,u,p,t)
    end
end

function measurement_oop(kf)
    MeasurementOop(kf)
end

function dynamics(kf::AbstractKalmanFilter)
    (x,u,p,t) -> get_mat(kf.A, x, u, p, t)*x + get_mat(kf.B, x, u, p, t)*u
end

"""
    reset!(kf::AbstractKalmanFilter; x0)

Reset the initial distribution of the state. Optionally, a new mean vector `x0` can be provided.
"""
function reset!(kf::AbstractKalmanFilter; x0 = kf.d0.μ, t=0)
    kf.x = convert_x0_type(x0)
    kf.R = convert_cov_type(kf.R1, kf.d0.Σ)# typeof(kf.R1)(kf.d0.Σ)
    kf.t = t
    nothing
end



"""
    project_bound(μ, R, idx; lower=-Inf, upper=Inf, tol=1e-9)

Project (μ,R) onto the bound lower ≤ x[idx] ≤ upper by minimizing
(x-μ)'*inv(R)*(x-μ). If μ[idx] is inside the interval (with tol), no change.

If a bound is violated, treats the active inequality as the equality
x[idx] = bound and applies the closed-form projection:
x* = μ - K * (μ[idx] - d),  R* = R - K * (R[idx,:]),
where K = R[:,idx] / R[idx,idx] and d is the active bound.

Returns `(x_proj, R_proj)`.
"""
function project_bound(μ::AbstractVector, P::AbstractMatrix, idx::Integer;
                       lower=-Inf, upper=Inf, tol=1e-9)

    @assert length(μ) == size(P,1) == size(P,2)

    x = copy(μ)
    Σ = copy(P)

    # Decide which (if any) bound is active
    d = if x[idx] < lower - tol
        lower
    elseif x[idx] > upper + tol
        upper
    else
        # Already feasible
        return x, Σ
    end

    Σii = Σ[idx, idx]
    if !isfinite(Σii) || Σii <= 0
        # Degenerate variance; safest fallback: clamp mean, leave covariance
        x[idx] = d
        return x, Σ
    end

    # Rank-1 “Kalman gain” for the hyperplane x[idx] = d
    Σi = Σ[:, idx]
    K = Σi / Σii
    Δ = x[idx] - d

    @bangbang x .-= K .* Δ
    @bangbang Σ .-= K * Σi'

    return x, symmetrize_psd(Σ)
end

# Helper: re-symmetrize and small PSD clip
function symmetrize_psd(A::AbstractMatrix; eps=1e-12)
    S = symmetrize(A)
    # Tiny eigenvalue floor to avoid numerical negatives
    vals, vecs = eigen(Symmetric(S))
    @bangbang vals .= max.(vals, eps)
    return vecs * Diagonal(vals) * vecs'
end

"""
    truncated_moment_match(μ, Σ, idx; lower=-Inf, upper=Inf, tol=1e-12, var_floor=1e-12)

Moment-match a Gaussian (μ, Σ) to enforce `lower ≤ x[idx] ≤ upper` by replacing the
marginal of x[idx] with a truncated-normal and adjusting the rest via the regression
identity:
    μ_-i' = μ_-i + A (m' - m),    Σ' = Σ + (s2' - s2) * (A * A'),
where A = Σ[:,idx] / Σ[idx,idx], m = μ[idx], s2 = Σ[idx,idx], and (m', s2') are the
mean/variance of the truncated scalar N(m, s2) on [lower, upper].

Returns `(μ_proj, Σ_proj)`.

Notes:
- Works for one-sided (lower or upper) and two-sided bounds.
- If the feasible probability mass is numerically ~0, falls back to projecting onto the
  nearest active bound (rank-1 “equality” projection).
"""
function truncated_moment_match(μ::AbstractVector, Σ::AbstractMatrix, idx::Integer;
                                lower=-Inf, upper=Inf, tol=1e-12, var_floor=1e-12)

    @assert length(μ) == size(Σ,1) == size(Σ,2)

    x = copy(μ)
    C = copy(Σ)

    s2 = C[idx, idx]
    if !isfinite(s2) || s2 <= 0
        # Degenerate variance: safest fallback — clamp mean, keep covariance
        x[idx] = clamp(x[idx], lower, upper)
        return x, C
    end

    m  = x[idx]
    s  = sqrt(s2)

    # Truncated scalar moments for N(m, s2) on [lower, upper]
    m′, s2′, ok = truncated_scalar_moments(m, s, lower, upper; tol=tol)
    if !ok
        # Fallback: equality-projection onto nearest active bound
        d = if m < lower - tol
            lower
        elseif m > upper + tol
            upper
        else
            # Interval too narrow or numerically ill-conditioned with m inside:
            # pin to closest endpoint
            abs(m - lower) < abs(upper - m) ? lower : upper
        end
        # rank-1 projection onto x[idx] = d
        Σi = C[:, idx]
        K  = Σi / s2
        Δ  = m - d
        @bangbang x .-= K .* Δ
        @bangbang C .-= K * Σi'
        return x, symmetrize_psd(C)
    end

    # Regression vector A = Σ[:,i] / s2 (keeps conditional x_-i | x_i the same)
    A = C[:, idx] / s2

    # Mean update: μ' = μ + A (m' - m)
    shift = (m′ - m)
    @bangbang x .+= A .* shift

    # Covariance update: Σ' = Σ + (s2' - s2) * (A * A')
    @bangbang C .+= (s2′ - s2) * (A * A')  # rank-1 symmetric update

    return x, symmetrize_psd(C; eps=var_floor)
end

# --- helpers ---------------------------------------------------------------

# Numerically-stable pdf/cdf for standard normal
normpdf(z::T) where T = exp(-(z*z)/2) / sqrt(T(2π))
normcdf(z::T) where T = SpecialFunctions.erfc(-z / sqrt(T(2)))/2        # stable in both tails
normccdf(z::T) where T = SpecialFunctions.erfc(z / sqrt(T(2)))/2        # 1 - Φ(z), stable for large z

"""
    truncated_scalar_moments(m, s, a, b; tol=1e-12)

Return (m′, s2′, ok) for X ~ N(m, s^2) truncated to [a,b].
If the normalizing mass Z is < tol, returns ok=false.
"""
function truncated_scalar_moments(m::Real, s::Real, a::Real, b::Real; tol=1e-12)
    if !isfinite(s) || s <= 0
        return m, 0.0, false
    end
    # Handle invalid / collapsed intervals
    if a >= b
        # treat as equality at nearest endpoint
        d = clamp(m, a, b)
        return d, 0.0, false
    end

    α = isfinite(a) ? (a - m)/s : -Inf
    β = isfinite(b) ? (b - m)/s :  Inf

    # One-sided cases (most stable)
    if !isfinite(β) && isfinite(α)
        # lower truncation [a, ∞)
        λ = normpdf(α) / max(normccdf(α), tol)      # Mills ratio for tail
        m′  = m + s * λ
        s2′ = (s^2) * (1 - λ*(λ - α))
        s2′ = max(s2′, 0.0)
        return m′, s2′, true
    elseif !isfinite(α) && isfinite(β)
        # upper truncation (-∞, b]
        λ = normpdf(β) / max(normcdf(β), tol)
        m′  = m - s * λ
        s2′ = (s^2) * (1 - λ*(λ + β))
        s2′ = max(s2′, 0.0)
        return m′, s2′, true
    else
        # two-sided [a,b]
        ϕα = isfinite(α) ? normpdf(α) : 0.0
        ϕβ = isfinite(β) ? normpdf(β) : 0.0
        Φα = isfinite(α) ? normcdf(α) : 0.0
        Φβ = isfinite(β) ? normcdf(β) : 1.0
        Z  = Φβ - Φα
        if !(Z > tol)   # numerically empty mass
            return m, 0.0, false
        end
        num = ϕα - ϕβ
        μshift = num / Z
        m′  = m + s * μshift
        s2′ = (s^2) * (1 + (α*ϕα - β*ϕβ)/Z - μshift^2)
        s2′ = max(s2′, 0.0)
        return m′, s2′, true
    end
end