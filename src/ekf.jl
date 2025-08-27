abstract type AbstractExtendedKalmanFilter{IPD,IPM} <: AbstractKalmanFilter end
struct ExtendedKalmanFilter{IPD, IPM, KF <: KalmanFilter, F, G, A} <: AbstractExtendedKalmanFilter{IPD,IPM}
    kf::KF
    dynamics::F
    measurement_model::G
    Ajac::A
    names::SignalNames
end

"""
    ExtendedKalmanFilter(kf, dynamics, measurement; Ajac, Cjac)
    ExtendedKalmanFilter(dynamics, measurement, R1,R2,d0=MvNormal(Matrix(R1)); nu::Int, p = NullParameters(), α = 1.0, check = true)

A nonlinear state estimator propagating uncertainty using linearization.

The constructor to the extended Kalman filter takes dynamics and measurement functions, and either covariance matrices, or a [`KalmanFilter`](@ref). If the former constructor is used, the number of inputs to the system dynamics, `nu`, must be explicitly provided with a keyword argument.

By default, the filter will internally linearize the dynamics using ForwardDiff. User provided Jacobian functions can be provided as keyword arguments `Ajac` and `Cjac`. These functions should have the signature `(x,u,p,t)::AbstractMatrix` where `x` is the state, `u` is the input, `p` is the parameters, and `t` is the time.

The dynamics and measurement function are on the following form
```
x(t+1) = dynamics(x, u, p, t) + w
y      = measurement(x, u, p, t) + e
```
where `w ~ N(0, R1)`, `e ~ N(0, R2)` and `x(0) ~ d0`

The matrices `R1, R2` can be time varying such that, e.g., `R1[:, :, t]` contains the ``R_1`` matrix at time index `t`.
They can also be given as functions on the form
```
Rfun(x, u, p, t) -> R
```
This allows for, e.g., handling functions where the dynamics disturbance ``w`` is an input argument to the function, by linearizing the dynamics w.r.t. the disturbance input in a function for ``R_1``, like this (assuming the dynamics have the function signalture `f(x, u, p, t, w)`):
```
function R1fun(x,u,p,t)
    Bw = ForwardDiff.jacobian(w->f(x, u, p, t, w), zeros(length(w)))
    Bw * R1 * Bw'
end
```
When providing functions, the dimensions of the state, input and output, `nx, nu, ny` must be provided as keyword arguments to the `ExtendedKalmanFilter` constructor since these cannot be inferred from the function signature.
For maximum performance, provide statically sized matrices from StaticArrays.jl

See also [`UnscentedKalmanFilter`](@ref) which is typically more accurate than `ExtendedKalmanFilter`. See [`KalmanFilter`](@ref) for detailed instructions on how to set up a Kalman filter `kf`.
"""
ExtendedKalmanFilter

function ExtendedKalmanFilter(dynamics, measurement_model::AbstractMeasurementModel, R1,d0=SimpleMvNormal(R1); nu::Int, ny=measurement_model.ny, nx=length(d0), Ts = 1.0, p = NullParameters(), α = 1.0, check = true, Ajac = nothing, kwargs...)
    T = eltype(d0)
    R2 = measurement_model.R2
    if R1 isa SMatrix
        x = @SVector zeros(T, nx)
        u = @SVector zeros(T, nu)
    else
        x = zeros(T, nx)
        u = zeros(T, nu)
    end
    t = zero(T)
    A = zeros(nx, nx) # This one is never needed
    B = zeros(nx, nu) # This one is never needed
    C = zeros(ny, nx) # This one is never needed
    D = zeros(ny, nu) # This one is never needed
    kf = KalmanFilter(A,B,C,D,R1,R2,d0; Ts, p, α, check, nx, nu, ny)

    return ExtendedKalmanFilter(kf, dynamics, measurement_model; Ajac, kwargs...)
end

function ExtendedKalmanFilter(dynamics, measurement, R1,R2,d0=SimpleMvNormal(R1); nu::Int, ny=size(R2,1), nx::Int=size(R1,1), Cjac = nothing, kwargs...)
    IPM = !has_oop(measurement)
    T = promote_type(eltype(R1), eltype(R2), eltype(d0))
    measurement_model = EKFMeasurementModel{T, IPM}(measurement, R2; nx, ny, Cjac)
    return ExtendedKalmanFilter(dynamics, measurement_model, R1, d0; nu, nx, ny, kwargs...)
end


function ExtendedKalmanFilter(kf, dynamics, measurement; Ajac = nothing, Cjac = nothing, names=default_names(kf.nx, kf.nu, kf.ny, "EKF"))
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
        # if IPD
        #     inner! = (xd,x)->dynamics(xd,x,u,p,t)
        #     out = zeros(eltype(kf.d0), length(kf.x))
        #     cfg = ForwardDiff.JacobianConfig(inner!, out, x)
        #     Ajac = (x,u,p,t) -> ForwardDiff.jacobian!((xd,x)->dynamics(xd,x,u,p,t), out, x, cfg, Val(false))
        # else
        #     inner = x->dynamics(x,u,p,t)
        #     cfg = ForwardDiff.JacobianConfig(inner, kf.x)
        #     Ajac = (x,u,p,t) -> ForwardDiff.jacobian(x->dynamics(x,u,p,t), x, cfg, Val(false))
        # end

        if IPD
            outx = zeros(eltype(kf.d0), kf.nx)
            jacx = zeros(eltype(kf.d0), kf.nx, kf.nx)
            Ajac = (x,u,p,t) -> ForwardDiff.jacobian!(jacx, (xd,x)->dynamics(xd,x,u,p,t), outx, x)
        else
            Ajac = (x,u,p,t) -> ForwardDiff.jacobian(x->dynamics(x,u,p,t), x)
        end
    end

    return ExtendedKalmanFilter{IPD,IPM,typeof(kf),typeof(dynamics),typeof(measurement_model),typeof(Ajac)}(kf, dynamics, measurement_model, Ajac, names)
end

function Base.getproperty(ekf::EKF, s::Symbol) where EKF <: AbstractExtendedKalmanFilter
    s ∈ fieldnames(EKF) && return getfield(ekf, s)
    mm = getfield(ekf, :measurement_model)
    if s ∈ fieldnames(typeof(mm))
        return getfield(mm, s)
    elseif s === :measurement
        return measurement(mm)
    end
    kf = getfield(ekf, :kf)
    if s ∈ fieldnames(typeof(kf))
        return getproperty(kf, s)
    end
    if s ∈ (:nx, :nu, :ny)
        return getproperty(kf, s)
    end
    error("$(typeof(ekf)) has no property named $s")
end

function Base.setproperty!(ekf::ExtendedKalmanFilter, s::Symbol, val)
    s ∈ fieldnames(typeof(ekf)) && return setproperty!(ekf, s, val)
    setproperty!(getfield(ekf, :kf), s, val) # Forward to inner filter
end

function Base.propertynames(ekf::EKF, private::Bool=false) where EKF <: AbstractExtendedKalmanFilter
    return (fieldnames(EKF)..., propertynames(ekf.kf, private)...)
end


function predict!(kf::AbstractExtendedKalmanFilter{IPD}, u, p = parameters(kf), t::Real = index(kf)*kf.Ts; R1 = get_mat(kf.R1, kf.x, u, p, t), α = kf.α) where IPD
    (; x, R) = kf
    A = kf.Ajac(x, u, p, t)
    if IPD
        xp = similar(x)
        kf.dynamics(xp, x, u, p, t)
        kf.x = xp
    else
        kf.x = kf.dynamics(x, u, p, t)
    end
    if α == 1
        kf.R = symmetrize(A*R*A') + R1
    else
        kf.R = symmetrize(α*A*R*A') + R1
    end
    kf.t += 1
end

function correct!(ukf::AbstractExtendedKalmanFilter, u, y, p, t::Real; kwargs...)
    measurement_model = ukf.measurement_model
    correct!(ukf, measurement_model, u, y, p, t::Real; kwargs...)    
end

function correct!(kf::AbstractKalmanFilter,  measurement_model::EKFMeasurementModel{IPM}, u, y, p = parameters(kf), t::Real = index(kf); R2 = get_mat(measurement_model.R2, kf.x, u, p, t)) where IPM
    (; x,R) = kf
    (; measurement, Cjac) = measurement_model
    C = Cjac(x, u, p, t)
    if IPM
        e = zeros(length(y))
        measurement(e, x, u, p, t)
        e .= y .- e
    else
        e = y .- measurement(x, u, p, t)
    end
    S   = symmetrize(C*R*C') + R2
    Sᵪ  = cholesky(Symmetric(S); check=false)
    issuccess(Sᵪ) || error("Cholesky factorization of innovation covariance failed, got S = ", S)
    K   = (R*C')/Sᵪ
    kf.x += vec(K*e)
    kf.R  = symmetrize((I - K*C)*R) # WARNING against I .- A
    ll = extended_logpdf(SimpleMvNormal(PDMat(S, Sᵪ)), e)[]# - 1/2*logdet(S) # logdet is included in logpdf
    (; ll, e, S, Sᵪ, K)
end

# If smoothing blows up / explodes / diverges towards the beginning of the trajectory, try increasing the measurement noise covariance. Also try the smooth_mbf in case the state dimension is large
function smooth(sol, kf::AbstractExtendedKalmanFilter, u::AbstractVector=sol.u, y::AbstractVector=sol.y, p=parameters(kf))
    T            = length(y)
    (; x,xt,R,Rt,ll) = sol
    xT           = similar(xt)
    RT           = similar(Rt)
    xT[end]      = xt[end]      |> copy
    RT[end]      = Rt[end]      |> copy
    for t = T-1:-1:1
        A = kf.Ajac(xT[t+1],u[t+1],p,((t+1)-1)*kf.Ts)
        C     = Rt[t]*A'/cholesky(Symmetric(R[t+1]))
        Ce = C*(xT[t+1] .- x[t+1])
        @bangbang Ce .+= xt[t]
        xT[t] = Ce
        RD = RT[t+1] .- R[t+1]
        RDC = RD*C'
        if RD isa SMatrix
            R0 = symmetrize(C*RDC)
        else
            CRDC = RD # Just a rename
            mul!(CRDC, C, RDC)
            R0 = symmetrize(CRDC)
        end
        RT[t] = @bangbang R0 .+= Rt[t]
    end
    KalmanSmoothingSolution(sol, xT, RT)
end


function smooth(kf::AbstractExtendedKalmanFilter, args...)
    reset!(kf)
    sol = forward_trajectory(kf, args...)
    smooth(sol, kf, args...)
end

sample_state(kf::AbstractExtendedKalmanFilter, p=parameters(kf); noise=true) = noise ? rand(kf.d0) : mean(kf.d0)
sample_state(kf::AbstractExtendedKalmanFilter, x, u, p, t; noise=true) = kf.dynamics(x, u, p, t) .+ noise*rand(SimpleMvNormal(get_mat(kf.R1, x, u, p, t)))
sample_measurement(kf::AbstractExtendedKalmanFilter, x, u, p, t; noise=true) = kf.measurement(x, u, p, t) .+ noise*rand(SimpleMvNormal(get_mat(kf.R2, x, u, p, t)))
measurement(kf::AbstractExtendedKalmanFilter) = kf.measurement
dynamics(kf::AbstractExtendedKalmanFilter) = kf.dynamics


# For smooth_mbf
get_A(kf::ExtendedKalmanFilter, x, u, p, t) = kf.Ajac(x,u,p,t)
get_C(kf::ExtendedKalmanFilter, x, u, p, t) = kf.Cjac(x,u,p,t)