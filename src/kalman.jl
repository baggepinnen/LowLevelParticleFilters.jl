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
