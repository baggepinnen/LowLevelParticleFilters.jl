@kwdef mutable struct SqKalmanFilter{AT,BT,CT,DT,R1T,R2T,D0T,XT,RT,TS,P,αT} <: AbstractKalmanFilter
    A::AT
    B::BT
    C::CT
    D::DT
    R1::R1T
    R2::R2T
    d0::D0T
    x::XT
    R::RT
    t::Int = 0
    Ts::TS = 1
    p::P = NullParameters()
    α::αT = 1.0
    names::SignalNames = names=default_names(length(d0), size(B,2), size(C,1), name="SqKF")
end


"""
    SqKalmanFilter(A,B,C,D,R1,R2,d0=MvNormal(R1); p = NullParameters(), α=1)


A standard Kalman filter on square-root form. This filter may have better numerical performance when the covariance matrices are ill-conditioned.

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

The covariance matrices `R1` and `R2` are the covariance matrices of the process noise and measurement noise, respectively. They can be provided as a matrix, as an `UpperTriangular` matrix representing the Cholesky factor. If `R1` or `R2` is a function, it must return the upper triangular Cholesky factor of the covariance matrix at the given time index.

The internal fields storing covariance matrices are for this filter storing the upper-triangular Cholesky factor.

α is an optional "forgetting factor", if this is set to a value > 1, such as 1.01-1.2, the filter will, in addition to the covariance inflation due to ``R_1``, exhibit "exponential forgetting" similar to a [Recursive Least-Squares (RLS) estimator](https://en.wikipedia.org/wiki/Recursive_least_squares_filter). It is thus possible to get a RLS-like algorithm by setting ``R_1=0, R_2 = 1/α`` and ``α > 1`` (``α`` is the inverse of the traditional RLS parameter ``α = 1/λ``). The form of the covariance update is
```math
R(t+1|t) = α AR(t)A^T + R_1
```

Ref: "A Square-Root Kalman Filter Using Only QR Decompositions", Kevin Tracy https://arxiv.org/abs/2208.06452
"""
function SqKalmanFilter(A,B,C,D,R1,R2,d0=SimpleMvNormal(Matrix(R1)); p = NullParameters(), α = 1.0, check = true, Ts=1, names=default_names(length(d0), size(B,2), size(C,1), "SqKF"))
    α ≥ 1 || @warn "α should be > 1 for exponential forgetting. An α < 1 will lead to exponential loss of adaptation over time."
    if check
        maximum(abs, eigvals(A isa SMatrix ? Matrix(A) : A)) ≥ 2 && @warn "The dynamics matrix A has eigenvalues with absolute value ≥ 2. This is either a highly unstable system, or you have forgotten to discretize a continuous-time model. If you are sure that the system is provided in discrete time, you can disable this warning by setting check=false." maxlog=1
    end
    R = UpperTriangular(convert_cov_type(R1, cholesky(d0.Σ).U))
    if !(R1 isa UpperTriangular)
        R1 = cholesky(R1).U
    end
    if !(R2 isa UpperTriangular)
        R2 = cholesky(R2).U
    end
    
    x0 = convert_x0_type(d0.μ)

    SqKalmanFilter(A,B,C,D,R1,R2, d0, x0, R, 0, Ts, p, α, names)
end



function Base.getproperty(kf::SqKalmanFilter, s::Symbol)
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

sample_state(kf::SqKalmanFilter, p=parameters(kf); noise=true) = noise ? rand(kf.d0) : mean(kf.d0)
sample_state(kf::SqKalmanFilter, x, u, p=parameters(kf), t=0; noise=true) = kf.A*x .+ kf.B*u .+ noise*get_mat(kf.R1, x, u, p, t)*rand(kf.nx)
sample_measurement(kf::SqKalmanFilter, x, u, p=parameters(kf), t=0; noise=true) = kf.C*x .+ kf.D*u .+ noise*get_mat(kf.R2, x, u, p, t)*rand(kf.ny)
covariance(kf::SqKalmanFilter)   = kf.R'kf.R
covtype(kf::SqKalmanFilter) = typeof(kf.R.data)

"""
    reset!(kf::SqKalmanFilter; x0)

Reset the initial distribution of the state. Optionally, a new mean vector `x0` can be provided.
"""
function reset!(kf::SqKalmanFilter; x0 = kf.d0.μ)
    kf.x = convert_x0_type(x0)
    kf.R = UpperTriangular(convert_cov_type(kf.R1, cholesky(kf.d0.Σ).U))
    kf.t = 0
end

"""
    predict!(kf::SqKalmanFilter, u, p = parameters(kf), t::Real = index(kf); R1 = get_mat(kf.R1, kf.x, u, p, t), α = kf.α)

For the square-root Kalman filter, a custom provided `R1` must be the upper triangular Cholesky factor of the covariance matrix of the process noise.
"""
function predict!(kf::SqKalmanFilter, u, p=parameters(kf), t::Real = index(kf)*kf.Ts; At = get_mat(kf.A, kf.x, u, p, t), Bt = get_mat(kf.B, kf.x, u, p, t), R1 = get_mat(kf.R1, kf.x, u, p, t), α = kf.α)
    (; x,R) = kf
    if u === nothing || length(u) == 0
        # Special case useful since empty input is common special case
        kf.x = At*x
    else
        kf.x = At*x .+ Bt*u |> vec
    end
    if α == 1
        M1 = [R*At';R1]
        if R.data isa SMatrix
            kf.R = UpperTriangular(qr(M1).R)
        else
            kf.R = UpperTriangular(qr!(M1).R)
        end
    else
        M = [sqrt(α)*R*At';R1]
        if R.data isa SMatrix
            kf.R = UpperTriangular(qr(M).R) # symmetrize(α*At*R*At') + R1
        else
            kf.R = UpperTriangular(qr!(M).R) # symmetrize(α*At*R*At') + R1
        end
    end
    kf.t += 1
end


"""
    correct!(kf::SqKalmanFilter, u, y, p = parameters(kf), t::Real = index(kf); R2 = get_mat(kf.R2, kf.x, u, p, t))

For the square-root Kalman filter, a custom provided `R2` must be the upper triangular Cholesky factor of the covariance matrix of the measurement noise.
"""
function correct!(kf::SqKalmanFilter, u, y, p=parameters(kf), t::Real = index(kf)*kf.Ts; R2 = get_mat(kf.R2, kf.x, u, p, t))
    (; C,D,x,R) = kf
    Ct = get_mat(C, x, u, p, t)
    Dt = get_mat(D, x, u, p, t)
    e   = y .- Ct*x
    if !iszero(D)
        e -= Dt*u
    end
    S0 = qr([R*Ct';R2]).R
    S = UpperTriangular(S0)
    S0 = signdet!(S0, S)
    K   = ((R'*(R*Ct'))/S)/(S')
    kf.x += K*e
    M = [R*(I - K*Ct)';R2*K']
    if R.data isa SMatrix
        kf.R = UpperTriangular(qr(M).R)
    else
        kf.R = UpperTriangular(qr!(M).R)
    end
    SS = S'S
    Sᵪ = Cholesky(S0, 'U', 0)
    ll = extended_logpdf(SimpleMvNormal(PDMat(SS, Sᵪ)), e)# - 1/2*logdet(S) # logdet is included in logpdf
    (; ll, e, SS, Sᵪ, K)
end

@inline function signdet!(S0, S)
    @inbounds for rc in axes(S0, 1)
        # In order to get a well-defined logdet, we need to enforce a positive diagonal of the R factor
        if S0[rc,rc] < 0
            for c = rc:size(S0, 2)
                S0[rc, c] = -S0[rc,c]
            end
        end
    end
    S0
end

@inline function signdet!(S0::SMatrix, S)
    Stemp = similar(S0) .= S0
    signdet!(Stemp, S)
    SMatrix(Stemp)
end
