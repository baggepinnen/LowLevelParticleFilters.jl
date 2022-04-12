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
    t::Base.RefValue{Int} = Ref(1)
end


"""
    KalmanFilter(A,B,C,D,R1,R2,d0=MvNormal(R1))

The matrices `A,B,C,D` define the dynamics
```
x' = Ax + Bu + w
y  = Cx + Du + e
```
where `w ~ N(0, R1)`, `e ~ N(0, R2)` and `x(0) ~ d0`
"""
function KalmanFilter(A,B,C,D,R1,R2,d0=MvNormal(Matrix(R1)))
    try
        cR1 = cond(R1)
        cR2 = cond(R2)
        (cond(cR1) > 1e8 || cond(cR2) > 1e8) && @warn("Covariance matrices are poorly conditioned")
    catch
        nothing
    end
    
    KalmanFilter(A,B,C,D,R1,R2,MvNormal(Matrix(R2)), d0, Vector(d0.μ), Matrix(d0.Σ), Ref(1))
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
sample_state(kf::AbstractKalmanFilter, x, u, p=parameters(kf), t=0; noise=true) = kf.A*x .+ kf.B*u .+ noise*rand(MvNormal(kf.R1))
sample_measurement(kf::AbstractKalmanFilter, x, u, p=parameters(kf), t=0; noise=true) = kf.C*x .+ kf.D*u .+ noise*rand(MvNormal(kf.R2))
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

function dynamics(kf::AbstractKalmanFilter)
    (x,u,p,t) -> get_mat(kf.A, x, u, p, t)*x + get_mat(kf.B, x, u, p, t)*u
end

function reset!(kf::AbstractKalmanFilter)
    kf.x .= Vector(kf.d0.μ)
    kf.R .= copy(Matrix(kf.d0.Σ))
    kf.t[] = 1
end
