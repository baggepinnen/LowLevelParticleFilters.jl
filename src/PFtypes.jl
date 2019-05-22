
abstract type AbstractParticleFilter <: AbstractFilter end


struct PFstate{PT<:AbstractArray, FT<:AbstractFloat}
    x::Vector{PT}
    xprev::Vector{PT}
    w::Vector{FT}
    we::Vector{FT}
    maxw::Ref{FT}
    j::Vector{Int64}
    bins::Vector{Float64}
    t::Ref{Int}
end

PFstate(N::Integer) = PFstate([zeros(N)],[zeros(N)],fill(-log(N), N),fill(1/N, N),Ref(0.),collect(1:N),zeros(N), Ref(1))

@with_kw struct ParticleFilter{ST,FT,GT,FDT,GDT,IDT,RST<:DataType,RNGT} <: AbstractParticleFilter
    state::ST
    dynamics::FT
    measurement::GT
    dynamics_density::FDT
    measurement_density::GDT
    initial_density::IDT
    resample_threshold::Float64 = 0.1
    resampling_strategy::RST = ResampleSystematic
    rng::RNGT = MersenneTwister()
end

struct AuxiliaryParticleFilter{T<:AbstractParticleFilter} <: AbstractParticleFilter
    pf::T
end

AuxiliaryParticleFilter(args...;kwargs...) = AuxiliaryParticleFilter(ParticleFilter(args...;kwargs...))

"""
ParticleFilter(num_particles, dynamics::Function, measurement::Function, dynamics_density, measurement_density, initial_density)
"""
function ParticleFilter(N::Integer, dynamics::Function, measurement::Function, dynamics_density, measurement_density, initial_density)
    xprev = Vector{SVector{length(initial_density),eltype(initial_density)}}([rand(initial_density) for n=1:N])
    x = deepcopy(xprev)
    w = fill(log(1/N), N)
    we = fill(1/N, N)
    s = PFstate(x,xprev,w,we,Ref(0.), collect(1:N), zeros(N),Ref(1))
    f = 3 ∉ numargs(dynamics) ? @inline(function (x,u,t) dynamics(x,u) end) : dynamics
    g = 2 ∉ numargs(measurement) ? (x,t) -> measurement(x) : measurement

    ParticleFilter(state = s, dynamics = f, measurement = g,
    dynamics_density=dynamics_density, measurement_density=measurement_density,
    initial_density=initial_density)
end

function ParticleFilter(s::PFstate, dynamics::Function, measurement::Function, dynamics_density, measurement_density, initial_density)
    f = 3 ∉ numargs(dynamics) ? @inline(function (x,u,t) dynamics(x,u) end) : dynamics
    g = 2 ∉ numargs(measurement) ? (x,t) -> measurement(x) : measurement

    ParticleFilter(state = s, dynamics = f, measurement = g,
    dynamics_density=dynamics_density, measurement_density=measurement_density,
    initial_density=initial_density)
end


Base.@propagate_inbounds function measurement_equation!(pf, y, t, w = pf.state.w, d=measurement_density(pf))
    x,g = particles(pf), measurement(pf)
    any(ismissing.(y)) && return w
    if d isa UnivariateDistribution && length(y) == 1
        for i = 1:num_particles(pf)
            w[i] += logpdf(d, (y-g(x[i],t))[1])
        end
    else
        for i = 1:num_particles(pf)
            w[i] += logpdf(d, y-g(x[i],t))
        end
    end
    w
end

Base.@propagate_inbounds function propagate_particles!(pf::AbstractParticleFilter,u,j::Vector{Int}, t, d=pf.dynamics_density)
    f = dynamics(pf)
    s = state(pf)
    x,xp = s.x, s.xprev
    VecT = eltype(s.x)
    D = length(VecT)
    noise = zeros(D)
    if d === nothing
        for i = eachindex(x)
            x[i] =  f(xp[j[i]] ,u, t)
        end
    else
        for i = eachindex(x)
            x[i] =  f(xp[j[i]] ,u, t) + VecT(rand!(pf.rng, d, noise))
        end
    end
    x
end

Base.@propagate_inbounds function propagate_particles!(pf::AbstractParticleFilter,u, t, d=pf.dynamics_density)
    f = pf.dynamics
    x,xp = pf.state.x, pf.state.xprev
    VecT = eltype(pf.state.x)
    D = length(VecT)
    noise = zeros(D)
    if d === nothing
        for i = eachindex(x)
            x[i] =  f(xp[i] ,u, t)
        end
    else
        for i = eachindex(x)
            x[i] =  f(xp[i] ,u, t) + VecT(rand!(pf.rng, d, noise))
        end
    end
    x
end


# AUX =================================================================================
@inline Random.rand!(_, d::Bool, args...) = 0 # To turn off noise in the dynamics update for pf aux

Base.@propagate_inbounds function add_noise!(pf::AbstractParticleFilter, d=pf.dynamics_density)
    x,xp = pf.state.x, pf.state.xprev
    VecT = eltype(pf.state.x)
    D = length(VecT)
    noise = zeros(D)

    for i = eachindex(x)
        x[i] += VecT(rand!(pf.rng, d, noise))
    end

    x
end
# Advanced =================================================================================



@with_kw struct AdvancedParticleFilter{ST,FT,GT,GLT,FDT,IDT,RST<:DataType,RNGT} <: AbstractParticleFilter
    state::ST
    dynamics::FT
    measurement::GT
    measurement_likelihood::GLT
    dynamics_density::FDT = Normal()
    initial_density::IDT
    resample_threshold::Float64 = 0.5
    resampling_strategy::RST = ResampleSystematic
    rng::RNGT = MersenneTwister()
end


"""
ParticleFilter(num_particles, dynamics::Function, measurement::Function, initial_density)
"""
function AdvancedParticleFilter(N::Integer, dynamics::Function, measurement::Function, measurement_likelihood, dynamics_density, initial_density)
    xprev = Vector{SVector{length(initial_density),eltype(initial_density)}}([rand(initial_density) for n=1:N])
    x  = deepcopy(xprev)
    w  = fill(log(1/N), N)
    we = fill(1/N, N)
    s = PFstate(x,xprev,w,we,Ref(0.), collect(1:N), zeros(N),Ref(1))

    AdvancedParticleFilter(state = s, dynamics = dynamics, measurement = measurement, measurement_likelihood=measurement_likelihood, dynamics_density=dynamics_density,
    initial_density=initial_density)
end


Base.@propagate_inbounds function measurement_equation!(pf::AdvancedParticleFilter, y, t, w = weights(pf))
    g = measurement_likelihood(pf)
    any(ismissing.(y)) && return w
    x = particles(pf)
    @inbounds for i = 1:num_particles(pf)
        w[i] += g(x[i],y,t)
    end
    w
end


Base.@propagate_inbounds function propagate_particles!(pf::AdvancedParticleFilter, u, j::Vector{Int}, t::Int, noise=true)
    noise === nothing && (noise = false)
    f = dynamics(pf)
    s = state(pf)
    x,xp = s.x, s.xprev
    @inbounds for i = eachindex(x)
        x[i] = f(xp[j[i]], u, t, noise) # TODO: lots of allocations here
    end
    x
end

Base.@propagate_inbounds function propagate_particles!(pf::AdvancedParticleFilter, u, t::Int, noise=true)
    noise === nothing && (noise = false)
    f = pf.dynamics
    x,xp = pf.state.x, pf.state.xprev
    @inbounds for i = eachindex(x)
        x[i] = f(xp[i], u, t, noise)
    end
    x
end


# ==========================================================================================



@forward ParticleFilter.state num_particles, weights, particles, particletype
@forward AdvancedParticleFilter.state num_particles, weights, particles, particletype

@forward AuxiliaryParticleFilter.pf state, particles, weights, expweights, reset!, simulate, weigthed_mean, measurement_equation!, sample_state, sample_measurement, index, num_particles, particletype, dynamics, measurement, dynamics_density, measurement_density, initial_density, resample_threshold, resampling_strategy, rng, mode_trajectory, mean_trajectory


sample_state(pf::AbstractParticleFilter) = rand(pf.rng, pf.initial_density)
sample_state(pf::ParticleFilter, x, u, t) = dynamics(pf)(x,u,t) + rand(pf.rng, pf.dynamics_density)
sample_state(pf::AdvancedParticleFilter, x, u, t) = dynamics(pf)(x,u,t,true)
sample_measurement(pf::AdvancedParticleFilter, x, t) = measurement(pf)(x, t, true)
sample_measurement(pf::AbstractParticleFilter, x, t) = measurement(pf)(x, t) .+ rand(pf.rng, pf.measurement_density)
expweights(pf::AbstractParticleFilter) = pf.state.we
weights(s::PFstate)                    = s.w
expweights(s::PFstate)                 = s.we
index(pf::AbstractParticleFilter)      = pf.state.t[]

num_particles(s::PFstate)              = num_particles(s.x)
num_particles(s::AbstractArray)        = length(s)
particles(s::PFstate)                  = s.x
particletype(s::PFstate)               = eltype(s.x)




@inline state(pf::AbstractParticleFilter) = pf.state
@inline dynamics(pf::AbstractParticleFilter) = pf.dynamics
@inline measurement(pf::AbstractParticleFilter) = pf.measurement
@inline measurement_likelihood(pf::AdvancedParticleFilter) = pf.measurement_likelihood
@inline dynamics_density(pf::AbstractParticleFilter) = pf.dynamics_density
@inline measurement_density(pf::AbstractParticleFilter) = pf.measurement_density
@inline initial_density(pf::AbstractParticleFilter) = pf.initial_density
@inline resample_threshold(pf::AbstractParticleFilter) = pf.resample_threshold
@inline resampling_strategy(pf::AbstractParticleFilter) = pf.resampling_strategy
@inline rng(pf::AbstractParticleFilter) = pf.rng
