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

sample_state(kf::AbstractKalmanFilter) = rand(kf.d0)
sample_state(kf::AbstractKalmanFilter, x, u, t) = kf.A*x .+ kf.B*u .+ rand(MvNormal(kf.R1))
sample_measurement(kf::AbstractKalmanFilter, x, u, t) = kf.C*x .+ kf.D*u .+ rand(MvNormal(kf.R2))
particletype(kf::AbstractKalmanFilter) = typeof(kf.x)
covtype(kf::AbstractKalmanFilter)      = typeof(kf.R)
state(kf::AbstractKalmanFilter)        = kf.x
covariance(kf::AbstractKalmanFilter)   = kf.R
function measurement(kf::AbstractKalmanFilter)
    if ndims(kf.A) == 3
        function (x,u,t)
            y = kf.C[:,:,t]*x
            if kf.D != 0
                y .+= kf.D[:,:,t]*u
            end
            y
        end
    else
        function (x,u,t)
            y = kf.C*x
            if kf.D != 0
                y .+= kf.D*u
            end
            y
        end
    end
end

function dynamics(kf::AbstractKalmanFilter)
    if ndims(kf.A) == 3
        (x,u,t) -> kf.A[:,:,t]*x + kf.B[:,:,t]*u
    else
        (x,u,t) -> kf.A*x + kf.B*u
    end
end

function reset!(kf::AbstractKalmanFilter)
    kf.x .= Vector(kf.d0.μ)
    kf.R .= copy(Matrix(kf.d0.Σ))
    kf.t[] = 1
end



# SigmaFilter ==========================================================================

struct SigmaFilter{DT,MT,MLT,D0T,DFT,VT} <: AbstractParticleFilter
    dynamics::DT
    measurement::MT
    measurement_likelihood::MLT
    w::Vector{Float64}
    we::Vector{Float64}
    initial_density::D0T
    dynamics_density::DFT
    x::Vector{VT}
    xprev::Vector{VT}
    xm::Vector{Float64}
    R::Matrix{Float64}
    t::Ref{Int}
end


"""
    SigmaFilter(N,dynamics,measurement,measurement_likelihood,df,d0)
"""
function SigmaFilter(N,dynamics,measurement,measurement_likelihood,df,d0)
    @show n = length(d0)
    xs = rand(d0,N)
    xs = reinterpret(SVector{n,Float64}, xs)[:]
    SigmaFilter(dynamics,measurement,measurement_likelihood,
        zeros(length(xs)), # w
        zeros(length(xs)), # we
        d0, df,
        xs,                # x
        copy(xs),          # xprev
        Vector(d0.μ),      # xm
        Matrix(d0.Σ),      # R
        Ref(1))
end

sample_state(sf::SigmaFilter) = rand(sf.initial_density)
sample_state(sf::SigmaFilter, x, u, t) = sf.dynamics(x,u,t,true)
sample_measurement(sf::SigmaFilter, x, u, t) = sf.measurement(x,u,t,true)
measurement(kf::SigmaFilter) = kf.measurement
dynamics(kf::SigmaFilter) = kf.dynamics
num_particles(sf::SigmaFilter) = length(sf.x)
particles(sf::SigmaFilter) = sf.x
weights(sf::SigmaFilter) = sf.w
expweights(sf::SigmaFilter) = sf.we
state(sf::SigmaFilter) = sf
rng(sf::SigmaFilter) = Random.GLOBAL_RNG

function predict!(sf::SigmaFilter, u, t::Integer = index(sf))
    @unpack dynamics,measurement,x,xprev,xm,R,w,we = sf
    N = length(x)
    n = length(x[1])
    xm .= sum(x->x[1]*x[2], zip(x,we)) # Must update based on correction
    R .= 0
    for i in eachindex(x)
        δ = x[i]-xm
        R .+= (δ*δ')*we[i]
    end
    R ./= (1-sum(abs2,we))
    R .= (R + R')/2 + 1e-16I
    d = MvNormal(xm,R)
    noisevec = zeros(length(d))
    for i in eachindex(x)
        xi = SVector{n,Float64}(rand!(d, noisevec))
        x[i] = dynamics(xi, u, t, true)
    end
    w  .= -log(N)
    we .= 1/N
    xm .= mean(x)
    R  .= cov(x)
    sf.t[] += 1
end


Base.@propagate_inbounds propagate_particles!(sf::SigmaFilter, u, j::Vector{Int}, t::Int, noise=true) = propagate_particles!(sf, u, t, noise)

# Base.@propagate_inbounds function measurement_equation!(sf::SigmaFilter, y, t, w = weights(sf))
#     g = measurement_likelihood(sf)
#     any(ismissing.(y)) && return w
#     x = particles(sf)
#     @inbounds for i = 1:num_particles(sf)
#         w[i] += g(x[i],y,t)
#     end
#     w
# end
