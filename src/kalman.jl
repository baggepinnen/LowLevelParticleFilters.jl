abstract type AbstractKalmanFilter <: AbstractFilter end

@with_kw struct KalmanFilter{AT,BT,CT,DT,R1T,R2T,R2DT,D0T,XT,RT} <: AbstractKalmanFilter
    A::AT
    B::BT
    C::CT
    D::DT
    R1::R1T
    R2::R2T
    R2d::R2DT
    d0::D0T
    x::XT
    R::RT
    t::Ref{Int} = Ref(1)
end


"""
KalmanFilter(A,B,C,D,R1,R2,d0=MvNormal(R1))
"""
function KalmanFilter(A,B,C,D,R1,R2,d0=MvNormal(R1))
    all(iszero, D) || throw(ArgumentError("Nonzero D matrix not supported yet"))
    KalmanFilter(A,B,C,D,R1,R2,MvNormal(R2), d0, Vector(d0.μ), Matrix(d0.Σ), Ref(1))
end


sample_state(kf::AbstractKalmanFilter) = rand(kf.d0)
sample_state(kf::AbstractKalmanFilter, x, u, t) = kf.A*x .+ kf.B*u .+ rand(MvNormal(kf.R1))
sample_measurement(kf::AbstractKalmanFilter, x, t) = kf.C*x .+ rand(MvNormal(kf.R2))
particletype(kf::AbstractKalmanFilter) = typeof(kf.x)
covtype(kf::AbstractKalmanFilter)      = typeof(kf.R)
state(kf::AbstractKalmanFilter)        = kf.x
covariance(kf::AbstractKalmanFilter)   = kf.R

function reset!(kf::AbstractKalmanFilter)
    kf.x .= Vector(kf.d0.μ)
    kf.R .= copy(Matrix(kf.d0.Σ))
    kf.t[] = 1
end

# UKF ==========================================================================

@with_kw struct UnscentedKalmanFilter{DT,MT,R1T,R2T,R2DT,D0T,VT,XT,RT} <: AbstractKalmanFilter
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
    UnscentedKalmanFilter(A,B,C,D,R1,R2,d0=MvNormal(R1))
"""
function UnscentedKalmanFilter(dynamics,measurement,R1,R2,d0=MvNormal(R1))
    n = size(R1,1)
    p = size(R2,1)
    R1 = SMatrix{n,n}(R1)
    R2 = SMatrix{n,n}(R2)
    xs = [@SVector zeros(n) for _ in 1:(2n+1)]
    xs = sigmapoints(mean(d0), cov(d0))
    UnscentedKalmanFilter(dynamics,measurement,R1,R2,MvNormal(Matrix(R2)), d0, xs, Vector(d0.μ), Matrix(d0.Σ), Ref(1))
end

sample_state(kf::UnscentedKalmanFilter) = rand(kf.d0)
sample_state(kf::UnscentedKalmanFilter, x, u, t) = kf.dynamics(x,u) .+ rand(MvNormal(Matrix(kf.R1)))
sample_measurement(kf::UnscentedKalmanFilter, x, t) = kf.measurement(x) .+ rand(MvNormal(Matrix(kf.R2)))

# function transform_moments!(S,X,m,L)
#     X .-= mean(X) # Normalize the sample
#     for i in eachindex(X)
#         S[i] = m .+ L*X[i]
#     end
#     S
# end

function sigmapoints(m,Σ)
    n = max(length(m), size(Σ,1))
    xs = [@SVector zeros(n) for _ in 1:(2n+1)]
    sigmapoints!(xs,m,Σ)
end

function sigmapoints!(xs, m, Σ::AbstractMatrix)
    n = length(xs[1])
    X = sqrt(Symmetric(n*Σ))
    for i in 1:n
        xs[i] = X[:,i]
        xs[i+n] = -xs[i] .+ m
        xs[i] = xs[i] .+ m
    end
    xs[end] = m
    xs
end


function predict!(ukf::UnscentedKalmanFilter, u, t = index(ukf))
    @unpack dynamics,measurement,x,xs,R,R1 = ukf
    ns = length(xs)
    sigmapoints!(xs,x,R) # TODO: these are calculated in the update step
    r = copy(R)
    for i in eachindex(xs)
        xs[i] = dynamics(xs[i],u)
    end
    x .= mean(xs)
    # for i in eachindex(xs)
    #     d = xs[i]-x
    #     r .+= Symmetric(d*d')
    # end
    # R .= Symmetric(r)./n + R1
    R .= cov(xs) + R1
    ukf.t[] += 1
end

function correct!(ukf::UnscentedKalmanFilter, y, t = index(ukf))
    @unpack measurement,x,xs,R,R1,R2,R2d = ukf
    n = size(R1,1)
    p = size(R2,1)
    ns = length(xs)
    sigmapoints!(xs,x,R) # Update sigmapoints here since untransformed points required
    # S = @SMatrix zeros(p,p)
    C = @SMatrix zeros(n,p)
    # ym = @SVector zeros(p) # y mean
    ys = map(measurement, xs)
    # for i in eachindex(xs)
    #     ys  = measurement(xs[i])
    #     ym += ys
    # end
    # ym  = ym ./ ns
    ym = mean(ys)
    for i in eachindex(ys)
        d   = ys[i]-ym
        # S .+= Symmetric(d*d')
        C  += Symmetric((xs[i]-x)*d')
    end
    e   = y .- ym
    S = cov(ys) + R2
    # S   = S./ns + R2
    K   = (C./ns)/S
    x .+= K*e
    R  .= R - K*S*K'
    0. #logpdf(MvNormal(Matrix(S)), e) - 1/2*logdet(S) # TODO: this is not correct
end
