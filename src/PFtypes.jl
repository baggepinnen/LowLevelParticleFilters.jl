
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
    ParticleFilter(num_particles, dynamics, measurement, dynamics_density, measurement_density, initial_density)
"""
function ParticleFilter(N::Integer, dynamics::Function, measurement::Function, dynamics_density, measurement_density, initial_density; kwargs...)
    xprev = Vector{SVector{length(initial_density),eltype(initial_density)}}([rand(initial_density) for n=1:N])
    x = deepcopy(xprev)
    w = fill(log(1/N), N)
    we = fill(1/N, N)
    s = PFstate(x,xprev,w,we,Ref(0.), collect(1:N), zeros(N),Ref(1))


    ParticleFilter(; state = s, dynamics, measurement,
    dynamics_density, measurement_density,
    initial_density, kwargs...)
end

function ParticleFilter(s::PFstate, dynamics::Function, measurement::Function, dynamics_density, measurement_density, initial_density; kwargs...)
    ParticleFilter(; state = s, dynamics, measurement,
    dynamics_density, measurement_density,
    initial_density, kwargs...)
end


function Base.getproperty(pf::AbstractParticleFilter, s::Symbol)
    s ∈ fieldnames(typeof(pf)) && return getfield(pf, s)
    if s === :nx
        return length(pf.dynamics_density)
    elseif s === :ny
        return length(pf.measurement_density)
    elseif s === :nu
        return error("Input length unknown")
    else
        throw(ArgumentError("$(typeof(pf)) has no property named $s"))
    end
end



Base.@propagate_inbounds function measurement_equation!(pf::ParticleFilter, u, y, t, w = pf.state.w, d=measurement_density(pf))
    x,g = particles(pf), measurement(pf)
    any(ismissing, y) && return w
    if d isa UnivariateDistribution && length(y) == 1
        for i = 1:num_particles(pf)
            w[i] += logpdf(d, (y-g(x[i],u,t))[1])
        end
    else
        for i = 1:num_particles(pf)
            w[i] += logpdf(d, y-g(x[i],u,t))
        end
    end
    w
end

Base.@propagate_inbounds function propagate_particles!(pf::ParticleFilter,u,j::Vector{Int}, t::Int, d=pf.dynamics_density)
    f = dynamics(pf)
    s = state(pf)
    x,xp = s.x, s.xprev
    VecT = eltype(s.x)
    D = length(VecT)
    noise = zeros(D)
    if d === nothing
        for i = eachindex(x)
            x[i] =  f(xp[j[i]], u, t)
        end
    else
        for i = eachindex(x)
            x[i] =  f(xp[j[i]], u, t) + VecT(rand!(pf.rng, d, noise))
        end
    end
    x
end

Base.@propagate_inbounds function propagate_particles!(pf::ParticleFilter, u, t::Int, d=pf.dynamics_density)
    f = pf.dynamics
    x,xp = pf.state.x, pf.state.xprev
    VecT = eltype(pf.state.x)
    D = length(VecT)
    noise = zeros(D)
    if d === nothing
        for i = eachindex(x)
            x[i] =  f(xp[i], u, t)
        end
    else
        for i = eachindex(x)
            x[i] =  f(xp[i], u, t) + VecT(rand!(pf.rng, d, noise))
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
    AdvancedParticleFilter(Nparticles, dynamics, measurement, measurement_likelihood, dynamics_density, initial_density; kwargs...)
"""
function AdvancedParticleFilter(N::Integer, dynamics::Function, measurement::Function, measurement_likelihood, dynamics_density, initial_density; kwargs...)
    r1 = rand(initial_density)
    xprev = Vector{SVector{length(initial_density),eltype(r1)}}([rand(initial_density) for n=1:N])
    x  = deepcopy(xprev)
    w  = fill(log(1/N), N)
    we = fill(1/N, N)
    s = PFstate(x,xprev,w,we,Ref(0.), collect(1:N), zeros(N),Ref(1))

    AdvancedParticleFilter(; state = s, dynamics, measurement, measurement_likelihood, dynamics_density,
    initial_density, kwargs...)
end


Base.@propagate_inbounds function measurement_equation!(pf::AbstractParticleFilter, u, y, t, w = weights(pf))
    g = measurement_likelihood(pf)
    any(ismissing.(y)) && return w
    x = particles(pf)
    Threads.@threads for i = 1:num_particles(pf)
        @inbounds w[i] += g(x[i], u, y, t)
    end
    w
end


Base.@propagate_inbounds function propagate_particles!(pf::AdvancedParticleFilter, u, j::Vector{Int}, t::Int, noise=true)
    noise === nothing && (noise = false)
    f = dynamics(pf)
    s = state(pf)
    x,xp = s.x, s.xprev
    Threads.@threads for i = eachindex(x)
        @inbounds x[i] = f(xp[j[i]], u, t, noise) # TODO: lots of allocations here
    end
    x
end

Base.@propagate_inbounds function propagate_particles!(pf::AbstractParticleFilter, u, t::Int, noise=true)
    noise === nothing && (noise = false)
    f = pf.dynamics
    x,xp = particles(pf), state(pf).xprev
    Threads.@threads for i = eachindex(x)
        @inbounds x[i] = f(xp[i], u, t, noise)
    end
    x
end


# ==========================================================================================



@forward ParticleFilter.state num_particles, weights, particles, particletype
@forward AdvancedParticleFilter.state num_particles, weights, particles, particletype

@forward AuxiliaryParticleFilter.pf state, particles, weights, expweights, reset!, weigthed_mean, measurement_equation!, sample_state, sample_measurement, index, num_particles, particletype, dynamics, measurement, dynamics_density, measurement_density, initial_density, resample_threshold, resampling_strategy, rng, mode_trajectory, mean_trajectory


sample_state(pf::AbstractParticleFilter) = rand(pf.rng, pf.initial_density)
sample_state(pf::ParticleFilter, x, u, t) = dynamics(pf)(x,u,t) + rand(pf.rng, pf.dynamics_density)
sample_state(pf::AdvancedParticleFilter, x, u, t) = dynamics(pf)(x,u,t,true)
sample_measurement(pf::AdvancedParticleFilter, x, u, t) = measurement(pf)(x, u, t, true)
sample_measurement(pf::AbstractParticleFilter, x, u, t) = measurement(pf)(x, u, t) .+ rand(pf.rng, pf.measurement_density)

sample_measurement(f::AbstractFilter, u) = sample_measurement(f, state(f), u, f.t[])

expweights(pf::AbstractParticleFilter) = pf.state.we
weights(s::PFstate)                    = s.w
expweights(s::PFstate)                 = s.we
index(pf::AbstractParticleFilter)      = pf.state.t[]

num_particles(s::PFstate)              = num_particles(s.x)
num_particles(s::AbstractArray)        = length(s)
particles(s::PFstate)                  = s.x
particletype(s::PFstate)               = eltype(s.x)
particletype(pf::AbstractFilter)       = eltype(state(pf).x)




@inline state(pf::AbstractParticleFilter) = pf.state
@inline dynamics(pf::AbstractParticleFilter) = pf.dynamics
@inline measurement(pf::AbstractParticleFilter) = pf.measurement
@inline measurement_likelihood(pf::AbstractParticleFilter) = pf.measurement_likelihood
@inline dynamics_density(pf::AbstractParticleFilter) = pf.dynamics_density
@inline measurement_density(pf::AbstractParticleFilter) = pf.measurement_density
@inline initial_density(pf::AbstractParticleFilter) = pf.initial_density
@inline resample_threshold(pf::AbstractParticleFilter) = pf.resample_threshold
@inline resampling_strategy(pf::AbstractParticleFilter) = pf.resampling_strategy
@inline rng(pf::AbstractParticleFilter) = pf.rng
