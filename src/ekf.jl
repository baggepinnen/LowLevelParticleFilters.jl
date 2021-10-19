abstract type AbstractExtendedKalmanFilter <: AbstractKalmanFilter end
@with_kw struct ExtendedKalmanFilter{KF <: KalmanFilter, F, G} <: AbstractExtendedKalmanFilter
    kf::KF
    dynamics::F
    measurement::G
end

function Base.getproperty(ekf::EKF, s::Symbol) where EKF <: AbstractExtendedKalmanFilter
    s ∈ fieldnames(EKF) && return getfield(ekf, s)
    return getproperty(ekf.kf, s)
end

function Base.propertynames(ekf::EKF, private::Bool=false) where EKF <: AbstractExtendedKalmanFilter
    return (fieldnames(EKF)..., propertynames(ekf.kf, private)...)
end


function predict!(kf::AbstractExtendedKalmanFilter, u, t::Integer = index(kf))
    @unpack x,R,R1 = kf
    A = ForwardDiff.jacobian(x->kf.dynamics(x,u,t), x)
    x .= kf.dynamics(x, u, t)
    R .= symmetrize(A*R*A') + R1
    kf.t[] += 1
end

function correct!(kf::AbstractExtendedKalmanFilter, u, y, t::Integer = index(kf))
    @unpack x,R,R2 = kf
    C = ForwardDiff.jacobian(x->kf.measurement(x,u,t), x)
    e  = y .- kf.measurement(x,u,t)
    S   = symmetrize(C*R*C') + R2
    Sᵪ  = cholesky(S)
    K   = (R*C')/Sᵪ
    x .+= vec(K*e)
    R  .= symmetrize((I - K*C)*R) # WARNING against I .- A
    ll = logpdf(MvNormal(PDMat(S, Sᵪ)), e)[]# - 1/2*logdet(S) # logdet is included in logpdf
    ll, e
end


function smooth(kf::AbstractExtendedKalmanFilter, u::AbstractVector, y::AbstractVector)
    reset!(kf)
    T            = length(y)
    x,xt,R,Rt,ll = forward_trajectory(kf, u, y)
    xT           = similar(xt)
    RT           = similar(Rt)
    xT[end]      = xt[end]      |> copy
    RT[end]      = Rt[end]      |> copy
    for t = T-1:-1:1
        A = ForwardDiff.jacobian(x->kf.dynamics(x,u[t+1],t+1), xt[t+1])
        C     = Rt[t]*A/R[t+1]
        xT[t] = xt[t] .+ C*(xT[t+1] .- x[t+1])
        # xT[t][end] = clamp(xT[t][end], 0.01, 7.499)
        RT[t] = Rt[t] .+ symmetrize(C*(RT[t+1] .- R[t+1])*C')
    end
    xT,RT,ll
end

sample_state(kf::AbstractExtendedKalmanFilter) = rand(kf.d0)
sample_state(kf::AbstractExtendedKalmanFilter, x, u, t) = kf.dynamics(x, u, t) .+ rand(MvNormal(kf.R1))
sample_measurement(kf::AbstractExtendedKalmanFilter, x, u, t) = kf.measurement(x, u, t) .+ rand(MvNormal(kf.R2))
