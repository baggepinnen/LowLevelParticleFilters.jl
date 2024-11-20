abstract type AbstractExtendedKalmanFilter <: AbstractKalmanFilter end
@with_kw struct ExtendedKalmanFilter{KF <: KalmanFilter, F, G, A, C} <: AbstractExtendedKalmanFilter
    kf::KF
    dynamics::F
    measurement::G
    Ajac::A
    Cjac::C
end

"""
    ExtendedKalmanFilter(kf, dynamics, measurement)
    ExtendedKalmanFilter(dynamics, measurement, R1,R2,d0=MvNormal(Matrix(R1)); nu::Int, p = NullParameters(), α = 1.0, check = true)

A nonlinear state estimator propagating uncertainty using linearization.

The constructor to the extended Kalman filter takes dynamics and measurement functions, and either covariance matrices, or a [`KalmanFilter`](@ref). If the former constructor is used, the number of inputs to the system dynamics, `nu`, must be explicitly provided with a keyword argument.

The filter will internally linearize the dynamics using ForwardDiff.

The dynamics and measurement function are on the following form
```
x(t+1) = dynamics(x, u, p, t) + w
y      = measurement(x, u, p, t) + e
```
where `w ~ N(0, R1)`, `e ~ N(0, R2)` and `x(0) ~ d0`

See also [`UnscentedKalmanFilter`](@ref) which is typically more accurate than `ExtendedKalmanFilter`. See [`KalmanFilter`](@ref) for detailed instructions on how to set up a Kalman filter `kf`.
"""
ExtendedKalmanFilter

function ExtendedKalmanFilter(dynamics, measurement, R1,R2,d0=SimpleMvNormal(Matrix(R1)); nu::Int, p = NullParameters(), α = 1.0, check = true, Ajac = nothing, Cjac = nothing)
    nx = size(R1,1)
    ny = size(R2,1)
    T = eltype(R1)
    if R1 isa SMatrix
        x = @SVector zeros(T, nx)
        u = @SVector zeros(T, nu)
    else
        x = zeros(T, nx)
        u = zeros(T, nu)
    end
    t = one(T)
    if Ajac === nothing
        Ajac = (x,u,p,t) -> ForwardDiff.jacobian(x->dynamics(x,u,p,t), x)
    end
    if Cjac === nothing
        Cjac = (x,u,p,t) -> ForwardDiff.jacobian(x->measurement(x,u,p,t), x)
    end
    A = Ajac(x,u,p,t)
    B = zeros(nx, nu) # This one is never needed
    C = Cjac(x,u,p,t)
    D = zeros(ny, nu) # This one is never needed
    kf = KalmanFilter(A,B,C,D,R1,R2,d0; p, α, check)
    return ExtendedKalmanFilter(kf, dynamics, measurement, Ajac, Cjac)
end

function ExtendedKalmanFilter(kf, dynamics, measurement; Ajac = nothing, Cjac = nothing)
    if Ajac === nothing
        Ajac = (x,u,p,t) -> ForwardDiff.jacobian(x->dynamics(x,u,p,t), x)
    end
    if Cjac === nothing
        Cjac = (x,u,p,t) -> ForwardDiff.jacobian(x->measurement(x,u,p,t), x)
    end
    return ExtendedKalmanFilter(kf, dynamics, measurement, Ajac, Cjac)
end

function Base.getproperty(ekf::EKF, s::Symbol) where EKF <: AbstractExtendedKalmanFilter
    s ∈ fieldnames(EKF) && return getfield(ekf, s)
    return getproperty(getfield(ekf, :kf), s)
end

function Base.setproperty!(ekf::ExtendedKalmanFilter, s::Symbol, val)
    s ∈ fieldnames(typeof(ekf)) && return setproperty!(ekf, s, val)
    setproperty!(getfield(ekf, :kf), s, val) # Forward to inner filter
end

function Base.propertynames(ekf::EKF, private::Bool=false) where EKF <: AbstractExtendedKalmanFilter
    return (fieldnames(EKF)..., propertynames(ekf.kf, private)...)
end


function predict!(kf::AbstractExtendedKalmanFilter, u, p = parameters(kf), t::Integer = index(kf); R1 = get_mat(kf.R1, kf.x, u, p, t), α = kf.α)
    @unpack x,R = kf
    A = kf.Ajac(x, u, p, t)
    kf.x = kf.dynamics(x, u, p, t)
    if α == 1
        kf.R = symmetrize(A*R*A') + R1
    else
        kf.R = symmetrize(α*A*R*A') + R1
    end
    kf.t += 1
end

function correct!(kf::AbstractExtendedKalmanFilter, u, y, p = parameters(kf), t::Integer = index(kf); R2 = get_mat(kf.R2, kf.x, u, p, t))
    @unpack x,R = kf
    C   = kf.Cjac(x, u, p, t)
    e   = y .- kf.measurement(x, u, p, t)
    S   = symmetrize(C*R*C') + R2
    Sᵪ  = cholesky(S)
    K   = (R*C')/Sᵪ
    kf.x += vec(K*e)
    kf.R  = symmetrize((I - K*C)*R) # WARNING against I .- A
    ll = extended_logpdf(SimpleMvNormal(PDMat(S, Sᵪ)), e)[]# - 1/2*logdet(S) # logdet is included in logpdf
    (; ll, e, S, Sᵪ, K)
end


function smooth(sol, kf::AbstractExtendedKalmanFilter, u::AbstractVector, y::AbstractVector, p=parameters(kf))
    T            = length(y)
    (; x,xt,R,Rt,ll) = sol
    xT           = similar(xt)
    RT           = similar(Rt)
    xT[end]      = xt[end]      |> copy
    RT[end]      = Rt[end]      |> copy
    for t = T-1:-1:1
        A = ForwardDiff.jacobian(x->kf.dynamics(x,u[t+1],p,t+1), xT[t+1])
        C     = Rt[t]*A'/R[t+1]
        xT[t] = xt[t] .+ C*(xT[t+1] .- x[t+1])
        RT[t] = Rt[t] .+ symmetrize(C*(RT[t+1] .- R[t+1])*C')
    end
    xT,RT,ll
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