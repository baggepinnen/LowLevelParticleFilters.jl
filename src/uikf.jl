"""
    UIKalmanFilter(kf::KalmanFilter, G)
    UIKalmanFilter(A, B, C, D, G, R1, R2, d0; kwargs...)

An Unknown Input Kalman Filter for estimating both state and unknown inputs in linear systems.

!!! warning "Experimental"
    This filter is currently considered experimental and the user interface may change in the future without respecting semantic versioning.

This filter implements the algorithm from Gillijns & De Moor (2007), "Unbiased minimum-variance
input and state estimation for linear discrete-time systems", which provides optimal
minimum-variance unbiased estimates of both the state `x` and unknown input `d`,
without augmenting the state vector.

The system is assumed to be on the form:
```
x(k+1) = A*x(k) + B*u(k) + G*d(k) + w(k)
y(k)   = C*x(k) + D*u(k) + e(k)
```
where `d(k)` is an unknown input vector, `w ~ N(0, R1)`, and `e ~ N(0, R2)`.

# Arguments:
- `kf::KalmanFilter`: An existing Kalman filter containing A, B, C, D, R1, R2, and initial state
- `G`: Unknown input matrix (nx × nd) or function `G(x,u,p,t)` returning such a matrix
- `A, B, C, D, R1, R2, d0`: Standard Kalman filter parameters (see [`KalmanFilter`](@ref))

# Requirements:
The filter requires that `rank(C*G) = size(G,2)` (full column rank). This ensures that the
unknown input can be uniquely estimated from the measurements. If this does not hold, consider an augmented state Kalman filter instead, for which a looser observability condition holds.

# Returns from correct!:
In addition to the standard Kalman filter outputs `(ll, e, S, Sᵪ, K)`, the `correct!` function
also returns:
- `d`: The estimate of the unknown input d(k-1)
- `M`: The matrix M used for unknown input estimation (weighted least squares matrix)

# Example:
```julia
sol = forward_trajectory(uikf, u, y)
sol.extra.d # Estimated unknown inputs over time
```

# Reference:
Gillijns, S., & De Moor, B. (2007). Unbiased minimum-variance input and state estimation
for linear discrete-time systems. Automatica, 43(1), 111-116.

See also [`KalmanFilter`](@ref), [`ExtendedKalmanFilter`](@ref)
"""
struct UIKalmanFilter{KF <: KalmanFilter, GT} <: AbstractKalmanFilter
    kf::KF
    G::GT
end

# Constructor from scratch
function UIKalmanFilter(A, B, C, D, G, R1, R2, d0=SimpleMvNormal(R1);
                       Ts=1.0, p=NullParameters(), α=1.0, check=true,
                       nx=length(d0), nu=size(B,2), ny=size(C,1))
    kf = KalmanFilter(A, B, C, D, R1, R2, d0; Ts, p, α, check, nx, nu, ny)
    UIKalmanFilter(kf, G)
end

# Property access - delegate to inner filter
function Base.getproperty(uikf::UIKalmanFilter, s::Symbol)
    s ∈ fieldnames(UIKalmanFilter) && return getfield(uikf, s)
    kf = getfield(uikf, :kf)
    return getproperty(kf, s)
end

function Base.setproperty!(uikf::UIKalmanFilter, s::Symbol, val)
    s ∈ fieldnames(UIKalmanFilter) && return setfield!(uikf, s, val)
    setproperty!(getfield(uikf, :kf), s, val)
end

function Base.propertynames(uikf::UIKalmanFilter, private::Bool=false)
    return (fieldnames(UIKalmanFilter)..., propertynames(uikf.kf, private)...)
end

function predict!(uikf::UIKalmanFilter, u, p=parameters(uikf), t::Real=index(uikf)*uikf.Ts; kwargs...)
    predict!(uikf.kf, u, p, t; kwargs...)
end

function correct!(uikf::UIKalmanFilter, u, y, p=parameters(uikf), t::Real=index(uikf)*uikf.Ts;
                  R2=get_mat(uikf.kf.R2, uikf.kf.x, u, p, t))
    (; kf, G) = uikf
    (; x, R) = kf

    C = get_mat(kf.C, x, u, p, t)
    D = get_mat(kf.D, x, u, p, t)
    G_mat = get_mat(G, x, u, p, t)

    # Compute innovation (equation 7 in paper)
    # ỹ(k) = y(k) - C(k) * x̂(k|k-1)
    e = y .- C * x
    if !iszero(D)
        e -= D*u
    end
    # Compute innovation covariance (equation 12)
    # R̃(k) = C(k) * P(k|k-1) * C'(k) + R2(k)
    R̃ = symmetrize(C * R * C') + R2
    R̃_chol = cholesky(Symmetric(R̃); check=false)
    issuccess(R̃_chol) || error("Cholesky factorization of innovation covariance failed at time $t, got R̃ = $(printarray(R̃))")

    # Compute F matrix: F(k) = C(k) * G(k-1)
    F = C * G_mat

    # Compute M matrix (equation 13) - Weighted Least Squares solution
    # M(k) = (F'(k) * R̃⁻¹(k) * F(k))⁻¹ * F'(k) * R̃⁻¹(k)
    FtRinv = F' / R̃_chol
    FtRinvF = FtRinv * F
    FtRinvF_chol = cholesky(Symmetric(FtRinvF); check=false)
    issuccess(FtRinvF_chol) || error("Cholesky factorization of F'*R̃⁻¹*F failed at time $t. Check that rank(C*G) = size(G,2)")
    M = (FtRinvF_chol \ FtRinv)

    # Estimate unknown input (equation 4)
    # d̂(k-1) = M(k) * ỹ(k)
    d = M * e

    # Update state with unknown input estimate (equation 5)
    # x̂*(k|k) = x̂(k|k-1) + G(k-1) * d̂(k-1)
    kf.x = x + G_mat * d

    # Compute modified covariance P*(k|k) using equation (25)
    # P*(k|k) = (I - G*M*C) * P(k|k-1) * (I - G*M*C)' + G*M*R2*M'*G'
    I_GMC = I - G_mat * M * C
    kf.R = symmetrize(I_GMC * R * I_GMC') + G_mat * M * R2 * M' * G_mat'

    # Compute innovation for the corrected state (equation 6)
    # e*(k) = y(k) - C(k) * x̂*(k|k)
    eˣ = y .- C * kf.x
    if !iszero(D)
        eˣ -= D*u
    end

    # Compute S* = correlation between state estimation error and measurement noise
    # S* = E[x̃* v'] = -G*M*R2  (equation 29 from paper)
    Sˣ = -G_mat * M * R2

    # Compute modified innovation covariance R̃* (equation 28-30)
    # Note: R̃* is SINGULAR with rank = ny - nd (proven in Lemma 6)
    I_CGM = I - C * G_mat * M
    R̃ˣ = symmetrize(I_CGM * R̃ * I_CGM')

    # Compute α using Theorem 8: α = [0 I_r] * U' * S̃^{-1}
    # where U contains left singular vectors of S̃^{-1} * C * G
    nd = size(G_mat, 2)
    r = kf.ny - nd  # rank of R̃*

    # Compute SVD of R̃^{-1/2} * C * G
    S̃_sqrt = R̃_chol.U  # R̃^{1/2}
    F_normalized = S̃_sqrt \ F
    U_svd = svd(F_normalized).U

    # Select α according to Theorem 8: use last r rows of U'S̃^{-1}
    α = U_svd[:, (nd+1):end]' / S̃_sqrt

    # Compute Kalman gain using equation (32):
    # K = (P*_{k|k} * C' + S*)' * α' * (α * R̃* * α')^{-1} * α
    Vˣ = kf.R * C' + Sˣ

    # Compute α * R̃* * α' (this should be invertible)
    α_Rˣ_αT = α * R̃ˣ * α'
    α_Rˣ_αT_chol = cholesky(Symmetric(α_Rˣ_αT); check=false)
    issuccess(α_Rˣ_αT_chol) || error("Cholesky factorization of α*R̃*α' failed at time $t")

    # Compute Kalman gain
    K = Vˣ * (α' / α_Rˣ_αT_chol) * α

    # Final state update (equation 6)
    # x̂(k|k) = x̂*(k|k) + K * e*(k)
    kf.x = kf.x + K * eˣ

    # Covariance update: P(k|k) = P*_{k|k} - K * V*_{k}'
    # where V* = P*_{k|k} * C' + S*
    kf.R = symmetrize(kf.R - K * Vˣ')

    # Compute log-likelihood using original innovation
    ll = extended_logpdf(SimpleMvNormal(PDMat(R̃, R̃_chol)), e)[]

    # Return standard outputs plus unknown input estimate and M matrix
    (; ll, e=eˣ, S=R̃, Sᵪ=R̃_chol, K, d, M)
end

# Helper methods - delegate to inner filter
sample_state(uikf::UIKalmanFilter, p=parameters(uikf); kwargs...) =
    sample_state(uikf.kf, p; kwargs...)

sample_state(uikf::UIKalmanFilter, x, u, p, t; kwargs...) =
    sample_state(uikf.kf, x, u, p, t; kwargs...)

sample_measurement(uikf::UIKalmanFilter, x, u, p, t; kwargs...) =
    sample_measurement(uikf.kf, x, u, p, t; kwargs...)

measurement(uikf::UIKalmanFilter) = measurement(uikf.kf)
dynamics(uikf::UIKalmanFilter) = dynamics(uikf.kf)

# Reset filter
reset!(uikf::UIKalmanFilter; kwargs...) = reset!(uikf.kf; kwargs...)

# Access filter properties
state(uikf::UIKalmanFilter) = state(uikf.kf)
covariance(uikf::UIKalmanFilter) = covariance(uikf.kf)
parameters(uikf::UIKalmanFilter) = parameters(uikf.kf)
index(uikf::UIKalmanFilter) = index(uikf.kf)

function forward_trajectory(kf::UIKalmanFilter, u::AbstractVector, y::AbstractVector, args...; kwargs...)
    d = Vector{Float64}[]
    post_correct_cb(kf, p, ret) = push!(d, ret.d)
    sol = invoke(forward_trajectory, Tuple{AbstractKalmanFilter, typeof(u), typeof(y), typeof.(args)...}, kf, u, y, args...; post_correct_cb, kwargs...)

    let (; f,u,y,x,xt,R,Rt,ll,e,K,S,t) = sol
        extra = (; d)
        return KalmanFilteringSolution(f,u,y,x,xt,R,Rt,ll,e,K,S,extra,t)
    end
end