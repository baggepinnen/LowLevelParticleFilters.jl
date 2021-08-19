using LowLevelParticleFilters, ForwardDiff

@with_kw struct ExtendedKalmanFilter <: AbstractKalmanFilter
    kf::KalmanFilter
    dynamics
    measurement
end

function Base.getproperty(ekf::ExtendedKalmanFilter, s::Symbol)
    s ∈ fieldnames(ExtendedKalmanFilter) && return getfield(ekf, s)
    return getproperty(ekf.kf, s)
end

function Base.propertynames(ekf::ExtendedKalmanFilter, private::Bool=false)
    return (fieldnames(ExtendedKalmanFilter)..., fieldnames(KalmanFilter)...)
end


function predict!(kf::ExtendedKalmanFilter, u, t::Integer = index(kf))
    @unpack x,R,R1 = kf
    A = ForwardDiff.jacobian(x->kf.dynamics(x,u), x)
    x .= kf.dynamics(x, u)
    R .= symmetrize(A*R*A') + R1
    kf.t[] += 1
end

function correct!(kf::ExtendedKalmanFilter, y, u, t::Integer = index(kf))
    @unpack x,R,R2 = kf
    C = ForwardDiff.jacobian(x->kf.measurement(x,u), x)
    e  = y .- kf.measurement(x,u)
    S   = symmetrize(C*R*C') + R2
    Sᵪ  = cholesky(S)
    K   = (R*C')/Sᵪ
    x .+= vec(K*e)
    R  .= symmetrize((I - K*C)*R) # WARNING against I .- A
    ll = logpdf(MvNormal(PDMat(S, Sᵪ)), e)[]# - 1/2*logdet(S) # logdet is included in logpdf
    ll, e
end


function smooth(kf::ExtendedKalmanFilter, u::AbstractVector, y::AbstractVector)
    reset!(kf)
    T            = length(y)
    x,xt,R,Rt,ll = forward_trajectory(kf, u, y)
    xT           = similar(xt)
    RT           = similar(Rt)
    xT[end]      = xt[end]      |> copy
    RT[end]      = Rt[end]      |> copy
    for t = T-1:-1:1
        A = ForwardDiff.jacobian(x->kf.dynamics(x,u[t+1]), xt[t+1])
        C     = Rt[t]*A/R[t+1]
        xT[t] = xt[t] .+ C*(xT[t+1] .- x[t+1])
        # xT[t][end] = clamp(xT[t][end], 0.01, 7.499)
        RT[t] = Rt[t] .+ symmetrize(C*(RT[t+1] .- R[t+1])*C')
    end
    xT,RT,ll
end

sample_state(kf::ExtendedKalmanFilter) = rand(kf.d0)
sample_state(kf::ExtendedKalmanFilter, x, u, t) = kf.dynamics(x,u) .+ rand(MvNormal(kf.R1))
sample_measurement(kf::ExtendedKalmanFilter, x, u, t) = kf.measurement(x, u) .+ rand(MvNormal(kf.R2))

d0 = MvNormal(randn(n),2.0)   # Initial state Distribution
du = MvNormal(2,1) # Control input distribution
kf = KalmanFilter(A, B, C, 0, 0.001eye(n), eye(p), d0)
ekf = LLPF.ExtendedKalmanFilter(kf, dynamics, measurement)