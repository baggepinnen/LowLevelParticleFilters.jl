"""
Square-root Extended Kalman Filter implementation for improved numerical stability.
This filter combines the nonlinear capabilities of the Extended Kalman Filter with
the numerical robustness of the square-root formulation.
"""

struct SqExtendedKalmanFilter{IPD, IPM, KF <: SqKalmanFilter, F, G, A} <: AbstractExtendedKalmanFilter{IPD,IPM}
    kf::KF
    dynamics::F
    measurement_model::G
    Ajac::A
    names::SignalNames
end

"""
    SqExtendedKalmanFilter(kf, dynamics, measurement; Ajac, Cjac)
    SqExtendedKalmanFilter(dynamics, measurement, R1,R2,d0=MvNormal(Matrix(R1)); nu::Int, p = NullParameters(), α = 1.0, check = true)

A nonlinear state estimator propagating uncertainty using linearization with square-root covariance representation.

This filter combines the Extended Kalman Filter's ability to handle nonlinear dynamics with the
Square-root Kalman Filter's numerical stability. It maintains the covariance in Cholesky factorized
form, ensuring positive definiteness and improved numerical conditioning.

The constructor takes dynamics and measurement functions, and either covariance matrices or a [`SqKalmanFilter`](@ref).
If the former constructor is used, the number of inputs to the system dynamics, `nu`, must be explicitly provided.

By default, the filter will internally linearize the dynamics using ForwardDiff. User provided Jacobian functions
can be provided as keyword arguments `Ajac` and `Cjac`. These functions should have the signature
`(x,u,p,t)::AbstractMatrix` where `x` is the state, `u` is the input, `p` is the parameters, and `t` is the time.

The dynamics and measurement function are on the following form:
```
x(t+1) = dynamics(x, u, p, t) + w
y      = measurement(x, u, p, t) + e
```
where `w ~ N(0, R1)`, `e ~ N(0, R2)` and `x(0) ~ d0`

For the square-root formulation, `R1` and `R2` can be provided as:
- Regular covariance matrices (will be converted to Cholesky factors internally)
- `UpperTriangular` matrices representing the Cholesky factors

See also [`ExtendedKalmanFilter`](@ref) for the standard formulation and [`SqKalmanFilter`](@ref) for the linear square-root filter.
"""
SqExtendedKalmanFilter

function SqExtendedKalmanFilter(dynamics, measurement_model::AbstractMeasurementModel, R1, d0=SimpleMvNormal(R1);
                                nu::Int, ny=measurement_model.ny, nx=length(d0), Ts = 1.0, p = NullParameters(),
                                α = 1.0, check = true, Ajac = nothing, kwargs...)
    T = eltype(d0)
    R2 = measurement_model.R2

    # Ensure R1 and R2 are in Cholesky factor form
    if !(R1 isa UpperTriangular)
        R1 = cholesky(R1).U
    end
    if !(R2 isa UpperTriangular)
        R2 = cholesky(R2).U
    end

    if R1 isa SMatrix
        x = @SVector zeros(T, nx)
        u = @SVector zeros(T, nu)
    else
        x = zeros(T, nx)
        u = zeros(T, nu)
    end

    t = zero(T)
    # Dummy matrices for SqKalmanFilter (never used)
    A = zeros(nx, nx)
    B = zeros(nx, nu)
    C = zeros(ny, nx)
    D = zeros(ny, nu)

    kf = SqKalmanFilter(A,B,C,D,R1,R2,d0; Ts, p, α, check)

    return SqExtendedKalmanFilter(kf, dynamics, measurement_model; Ajac, kwargs...)
end

function SqExtendedKalmanFilter(dynamics, measurement, R1, R2, d0=SimpleMvNormal(R1);
                                nu::Int, ny=size(R2,1), nx::Int=size(R1,1), Cjac = nothing, kwargs...)
    IPM = !has_oop(measurement)
    T = promote_type(eltype(R1), eltype(R2), eltype(d0))
    measurement_model = EKFMeasurementModel{T, IPM}(measurement, R2; nx, ny, Cjac)
    return SqExtendedKalmanFilter(dynamics, measurement_model, R1, d0; nu, nx, ny, kwargs...)
end

function SqExtendedKalmanFilter(kf::SqKalmanFilter, dynamics, measurement;
                                Ajac = nothing, Cjac = nothing,
                                names=default_names(kf.nx, kf.nu, kf.ny, "SqEKF"))
    IPD = !has_oop(dynamics)
    if measurement isa AbstractMeasurementModel
        measurement_model = measurement
        IPM = isinplace(measurement_model)
    else
        IPM = has_ip(measurement)
        T = promote_type(eltype(kf.R1), eltype(kf.R2), eltype(kf.d0))
        measurement_model = EKFMeasurementModel{T, IPM}(measurement, kf.R2; kf.nx, kf.ny, Cjac)
    end

    if Ajac === nothing
        if IPD
            outx = zeros(eltype(kf.d0), kf.nx)
            jacx = zeros(eltype(kf.d0), kf.nx, kf.nx)
            Ajac = (x,u,p,t) -> ForwardDiff.jacobian!(jacx, (xd,x)->dynamics(xd,x,u,p,t), outx, x)
        else
            Ajac = (x,u,p,t) -> ForwardDiff.jacobian(x->dynamics(x,u,p,t), x)
        end
    end

    return SqExtendedKalmanFilter{IPD,IPM,typeof(kf),typeof(dynamics),typeof(measurement_model),typeof(Ajac)}(
        kf, dynamics, measurement_model, Ajac, names)
end

# Property access delegation (same as ExtendedKalmanFilter)
function Base.getproperty(sekf::SEKF, s::Symbol) where SEKF <: SqExtendedKalmanFilter
    s ∈ fieldnames(SEKF) && return getfield(sekf, s)
    mm = getfield(sekf, :measurement_model)
    if s ∈ fieldnames(typeof(mm))
        return getfield(mm, s)
    elseif s === :measurement
        return measurement(mm)
    end
    kf = getfield(sekf, :kf)
    if s ∈ fieldnames(typeof(kf))
        return getproperty(kf, s)
    end
    if s ∈ (:nx, :nu, :ny)
        return getproperty(kf, s)
    end
    error("$(typeof(sekf)) has no property named $s")
end

function Base.setproperty!(sekf::SqExtendedKalmanFilter, s::Symbol, val)
    s ∈ fieldnames(typeof(sekf)) && return setproperty!(sekf, s, val)
    setproperty!(getfield(sekf, :kf), s, val) # Forward to inner filter
end

function Base.propertynames(sekf::SEKF, private::Bool=false) where SEKF <: SqExtendedKalmanFilter
    return (fieldnames(SEKF)..., propertynames(sekf.kf, private)...)
end

"""
    predict!(kf::SqExtendedKalmanFilter, u, p, t; R1, α)

Prediction step for the Square-root Extended Kalman Filter.
Linearizes the dynamics and updates the state and Cholesky factor of covariance using QR decomposition.
"""
function predict!(kf::SqExtendedKalmanFilter{IPD}, u, p = parameters(kf), t::Real = index(kf)*kf.Ts;
                  R1 = get_mat(kf.R1, kf.x, u, p, t), α = kf.α) where IPD
    (; x, R) = kf
    A = kf.Ajac(x, u, p, t)

    # Nonlinear state prediction
    if IPD
        xp = similar(x)
        kf.dynamics(xp, x, u, p, t)
        kf.x = xp
    else
        kf.x = kf.dynamics(x, u, p, t)
    end

    # Ensure R1 is an UpperTriangular Cholesky factor
    if !(R1 isa UpperTriangular)
        R1 = cholesky(R1).U
    end

    # Square-root covariance update using QR decomposition
    if α == 1
        M = [R*A'; R1]
    else
        M = [sqrt(α)*R*A'; R1]
    end

    if R.data isa SMatrix
        kf.R = UpperTriangular(qr(M).R)
    else
        kf.R = UpperTriangular(qr!(M).R)
    end

    kf.t += 1
end

"""
    correct!(kf::SqExtendedKalmanFilter, u, y, p, t; R2)

Correction step for the Square-root Extended Kalman Filter.
Linearizes the measurement and updates the state and Cholesky factor of covariance using QR decomposition.
"""
function correct!(kf::SqExtendedKalmanFilter, u, y, p, t::Real; kwargs...)
    measurement_model = kf.measurement_model
    correct!(kf, measurement_model, u, y, p, t; kwargs...)
end

function correct!(kf::SqExtendedKalmanFilter, measurement_model::EKFMeasurementModel{IPM}, u, y,
                  p = parameters(kf), t::Real = index(kf)*kf.Ts;
                  R2 = get_mat(measurement_model.R2, kf.x, u, p, t)) where IPM
    (; x, R) = kf
    (; measurement, Cjac) = measurement_model
    C = Cjac(x, u, p, t)

    # Compute innovation
    if IPM
        e = zeros(length(y))
        measurement(e, x, u, p, t)
        e .= y .- e
    else
        e = y .- measurement(x, u, p, t)
    end

    # Ensure R2 is an UpperTriangular Cholesky factor
    if !(R2 isa UpperTriangular)
        R2 = cholesky(R2).U
    end

    # Innovation covariance Cholesky factor
    S0 = qr([R*C'; R2]).R
    S = UpperTriangular(S0)
    S0 = signdet!(S0, S)  # Ensure positive diagonal

    # Kalman gain
    K = ((R'*(R*C'))/S)/(S')

    # State update
    kf.x += K*e

    # Square-root covariance update using QR decomposition
    M = [R*(I - K*C)'; R2*K']
    if R.data isa SMatrix
        kf.R = UpperTriangular(qr(M).R)
    else
        kf.R = UpperTriangular(qr!(M).R)
    end

    # Compute log-likelihood
    SS = S'S
    Sᵪ = Cholesky(S0, 'U', 0)
    ll = extended_logpdf(SimpleMvNormal(PDMat(SS, Sᵪ)), e)

    (; ll, e, SS, Sᵪ, K)
end

"""
    smooth(sol, kf::SqExtendedKalmanFilter, u, y, p)

Performs Rauch-Tung-Striebel smoothing for the Square-root Extended Kalman Filter.
Returns smoothed states and covariance matrices (converted from Cholesky factors).
"""
function smooth(sol, kf::SqExtendedKalmanFilter, u::AbstractVector=sol.u, y::AbstractVector=sol.y,
p=parameters(kf))
    T = length(y)
    (; x, xt, R, Rt, ll) = sol
    xT = similar(xt)

    RT0 = Rt[1]'Rt[1]
    RT = [zero(RT0) for i in eachindex(Rt)]
    xT[end] = xt[end] |> copy
    RT[end] = Rt[end]'*Rt[end]  # Convert to covariance for output compatibility

    for t = T-1:-1:1
        A = kf.Ajac(xT[t+1], u[t+1], p, ((t+1)-1)*kf.Ts)

        # Use Cholesky factors directly
        Rt_cov = Rt[t]'*Rt[t]
        C = Rt_cov*A'/Cholesky(R[t+1], 'U', 0)

        Ce = C*(xT[t+1] .- x[t+1])
        @bangbang Ce .+= xt[t]
        xT[t] = Ce

        R_cov = R[t+1]'*R[t+1]
        RD = RT[t+1] .- R_cov
        RDC = RD*C'
        if RD isa SMatrix
            R0 = symmetrize(C*RDC)
        else
            CRDC = RD
            mul!(CRDC, C, RDC)
            R0 = symmetrize(CRDC)
        end
        RT[t] = @bangbang R0 .+= Rt_cov
    end

    KalmanSmoothingSolution(sol, xT, RT)
end

function smooth(kf::SqExtendedKalmanFilter, args...)
    reset!(kf)
    sol = forward_trajectory(kf, args...)
    smooth(sol, kf, args...)
end

# Reuse sampling functions from ExtendedKalmanFilter
sample_state(kf::SqExtendedKalmanFilter, p=parameters(kf); noise=true) = noise ? rand(kf.d0) : mean(kf.d0)
sample_state(kf::SqExtendedKalmanFilter, x, u, p, t; noise=true) = kf.dynamics(x, u, p, t) .+ noise*rand(SimpleMvNormal(get_mat(kf.R1, x, u, p, t)))
sample_measurement(kf::SqExtendedKalmanFilter, x, u, p, t; noise=true) = kf.measurement(x, u, p, t) .+ noise*rand(SimpleMvNormal(get_mat(kf.R2, x, u, p, t)))
measurement(kf::SqExtendedKalmanFilter) = kf.measurement
dynamics(kf::SqExtendedKalmanFilter) = kf.dynamics

# covariance(kf::SqExtendedKalmanFilter)   = kf.R'kf.R

# Helper function from sq_kalman.jl
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

function reset!(kf::SqExtendedKalmanFilter; x0 = kf.d0.μ)
    kf.x = convert_x0_type(x0)
    kf.R = UpperTriangular(convert_cov_type(kf.R1, cholesky(kf.d0.Σ).U))
    kf.t = 0
end