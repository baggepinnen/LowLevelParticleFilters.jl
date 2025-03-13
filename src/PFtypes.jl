
abstract type AbstractParticleFilter <: AbstractFilter end

function parameters(f::AbstractFilter)
    hasproperty(f, :p) ? getproperty(f, :p) : NullParameters()
end

struct PFstate{PT<:AbstractArray, FT<:AbstractFloat}
    x::Vector{PT}
    xprev::Vector{PT}
    w::Vector{FT}
    we::Vector{FT}
    maxw::Base.RefValue{FT}
    j::Vector{Int64}
    bins::Vector{Float64}
    t::Base.RefValue{Int}
end

PFstate(N::Integer) = PFstate([zeros(N)],[zeros(N)],fill(-log(N), N),fill(1/N, N),Ref(0.),collect(1:N),zeros(N), Ref(0))

@with_kw struct ParticleFilter{ST,FT,GT,FDT,GDT,IDT,RST<:DataType,RNGT,P} <: AbstractParticleFilter
    state::ST
    dynamics::FT
    measurement::GT
    dynamics_density::FDT
    measurement_density::GDT
    initial_density::IDT
    resample_threshold::Float64 = 0.1
    resampling_strategy::RST = ResampleSystematic
    rng::RNGT = Xoshiro()
    p::P = NullParameters()
    threads::Bool = false
    Ts::Float64 = 1.0
end

struct AuxiliaryParticleFilter{T<:AbstractParticleFilter} <: AbstractParticleFilter
    pf::T
end

parameters(pf::AuxiliaryParticleFilter) = parameters(pf.pf)

"""
    AuxiliaryParticleFilter(args...; kwargs...)

Takes exactly the same arguments as [`ParticleFilter`](@ref), or an instance of [`ParticleFilter`](@ref).
"""
AuxiliaryParticleFilter(args...;kwargs...) = AuxiliaryParticleFilter(ParticleFilter(args...;kwargs...))

"""
    ParticleFilter(N::Integer, dynamics, measurement, dynamics_density, measurement_density, initial_density; threads = false, p = NullParameters(), kwargs...)


See the docs for more information: https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/#Particle-filter-1

# Arguments:
- `N`: Number of particles
- `dynamics`: A discrete-time dynamics function `(x, u, p, t) -> x⁺`
- `measurement`: A measurement function `(x, u, p, t) -> y`
- `dynamics_density`: A probability-density function for additive noise in the dynamics. Use [`AdvancedParticleFilter`](@ref) for non-additive noise.
- `measurement_density`: A probability-density function for additive measurement noise. Use [`AdvancedParticleFilter`](@ref) for non-additive noise.
- `initial_density`: Distribution of the initial state.
"""
function ParticleFilter(N::Integer, dynamics::Function, measurement::Function, dynamics_density, measurement_density, initial_density; kwargs...)
    xprev = Vector{SVector{length(initial_density),eltype(initial_density)}}([rand(initial_density) for n=1:N])
    x = deepcopy(xprev)
    w = fill(log(1/N), N)
    we = fill(1/N, N)
    s = PFstate(x,xprev,w,we,Ref(0.), collect(1:N), zeros(N),Ref(0))

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
        return length(pf.state.x[1])
    elseif s === :ny
        return length(measurement_density(pf))
    elseif s === :nu
        return error("Input length unknown")
    else
        throw(ArgumentError("$(typeof(pf)) has no property named $s"))
    end
end

function Base.getproperty(pf::AuxiliaryParticleFilter, s::Symbol)
    s === :pf && return getfield(pf, :pf)
    getproperty(getfield(pf, :pf), s)
end


Base.@propagate_inbounds function measurement_equation!(pf::ParticleFilter, u, y, p, t, w = pf.state.w, d=measurement_density(pf); g = measurement(pf))
    x = particles(pf)
    any(ismissing, y) && return w
    # if length(d) == 1 && length(y) == 1
    #     for i = 1:num_particles(pf)
    #         w[i] += extended_logpdf(d, (y-g(x[i],u,p,t))[1])
    #     end
    # else
        for i = 1:num_particles(pf)
            w[i] += extended_logpdf(d, y-g(x[i],u,p,t))
        end
    # end
    w
end

Base.@propagate_inbounds function propagate_particles!(pf::ParticleFilter,u,j::Vector{Int}, p, t::Real, d=pf.dynamics_density)
    f = dynamics(pf)
    s = state(pf)
    x,xp = s.x, s.xprev
    VecT = eltype(s.x)
    D = length(VecT)
    noise = zeros(D)
    if d === nothing
        for i = eachindex(x)
            x[i] =  f(xp[j[i]], u, p, t)
        end
    else
        for i = eachindex(x)
            x[i] =  f(xp[j[i]], u, p, t) + VecT(rand!(pf.rng, d, noise))
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



@with_kw struct AdvancedParticleFilter{ST,FT,GT,GLT,FDT,IDT,RST<:DataType,RNGT,P} <: AbstractParticleFilter
    state::ST
    dynamics::FT
    measurement::GT
    measurement_likelihood::GLT
    dynamics_density::FDT = SimpleMvNormal()
    initial_density::IDT
    resample_threshold::Float64 = 0.5
    resampling_strategy::RST = ResampleSystematic
    rng::RNGT = Xoshiro()
    p::P = NullParameters()
    threads::Bool = false
    Ts::Float64 = 1.0
end


"""
    AdvancedParticleFilter(N::Integer, dynamics::Function, measurement::Function, measurement_likelihood, dynamics_density, initial_density; p = NullParameters(), threads = false, kwargs...)

This type represents a standard particle filter but affords extra flexibility compared to the [`ParticleFilter`](@ref) type, e.g., non-additive noise in the dynamics and measurement functions.

See the docs for more information: https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/#AdvancedParticleFilter-1

# Arguments:
- `N`: Number of particles
- `dynamics`: A discrete-time dynamics function `(x, u, p, t, noise=false) -> x⁺`. It's important that the `noise` argument defaults to `false`.
- `measurement`: A measurement function `(x, u, p, t, noise=false) -> y`. It's important that the `noise` argument defaults to `false`.
- `measurement_likelihood`: A function `(x, u, y, p, t)->logl` to evaluate the log-likelihood of a measurement.
- `dynamics_density`: This field is not used by the advanced filter and can be set to `nothing`.
- `initial_density`: The distribution of the initial state.
- `threads`: use threads to propagate particles in parallel. Only activate this if your dynamics is thread-safe. `SeeToDee.SimpleColloc` is not thread-safe by default due to the use of internal caches, but `SeeToDee.Rk4` is.

# Extended help
## Multiple measurement models
The `measurement_likelihood` function is used to evaluate the likelihood of a measurement. If you have multiple sensors and want to perform individual `correct!` steps for each, call `correct!(..., g = custom_likelihood_function)`.
"""
function AdvancedParticleFilter(N::Integer, dynamics::Function, measurement::Function, measurement_likelihood, dynamics_density, initial_density; kwargs...)
    r1 = rand(initial_density)
    xprev = Vector{SVector{length(initial_density),eltype(r1)}}([rand(initial_density) for n=1:N])
    x  = deepcopy(xprev)
    w  = fill(log(1/N), N)
    we = fill(1/N, N)
    s = PFstate(x,xprev,w,we,Ref(0.), collect(1:N), zeros(N),Ref(0))

    AdvancedParticleFilter(; state = s, dynamics, measurement, measurement_likelihood, dynamics_density,
    initial_density, kwargs...)
end


"""
    measurement_equation!(pf::AbstractParticleFilter, u, y, p, t, w = weights(pf); g = measurement_likelihood(pf))

The main step of [`correct!`](@ref) for [`AdvancedParticleFilter`](@ref). This function can be called directly in order to pass a custom likelihood function to the filter.

# Arguments:
- `u`: The input
- `y`: The measured output
- `p`: The parameters
- `t`: The time
- `w`: The weights of the particles
- `g`: The likelihood function. Defaults to the one stored in the filter. Pass a custom function when updating with, e.g., a subset of sensors.
"""
Base.@propagate_inbounds function measurement_equation!(pf::AbstractParticleFilter, u, y, p, t, w = weights(pf); g = measurement_likelihood(pf))
    any(ismissing.(y)) && return w
    x = particles(pf)
    if pf.threads
        Threads.@threads :static for i = 1:num_particles(pf)
            @inbounds w[i] += g(x[i], u, y, p, t)
        end
    else
        for i = 1:num_particles(pf)
            @inbounds w[i] += g(x[i], u, y, p, t)
        end
    end
    w
end


Base.@propagate_inbounds function propagate_particles!(pf::AdvancedParticleFilter, u, j::Vector{Int}, p, t::Real, noise::Union{Bool, Nothing}=true)
    noise === nothing && (noise = false)
    f = dynamics(pf)
    s = state(pf)
    x,xp = s.x, s.xprev
    if pf.threads
        let f = f, xp=xp, j=j, u=u, p=p, t=t, noise=noise
            Threads.@threads :static for i = eachindex(x)
                @inbounds x[i] = f(xp[j[i]], u, p, t, noise) # TODO: lots of allocations here
            end
        end
    else
        for i = eachindex(x)
            @inbounds x[i] = f(xp[j[i]], u, p, t, noise) 
        end
    end
    x
end

Base.@propagate_inbounds function propagate_particles!(pf::AbstractParticleFilter, u, p, t::Real, noise::Nothing)
    f = pf.dynamics
    x,xp = particles(pf), state(pf).xprev
    if pf.threads
        Threads.@threads :static for i = eachindex(x)
            @inbounds x[i] = f(xp[i], u, p, t)
        end
    else
        for i = eachindex(x)
            @inbounds x[i] = f(xp[i], u, p, t)
        end
    end
    x
end

Base.@propagate_inbounds function propagate_particles!(pf::AbstractParticleFilter, u, p, t::Real, noise::Bool=true)
    f = pf.dynamics
    x,xp = particles(pf), state(pf).xprev
    if pf.threads
        Threads.@threads :static for i = eachindex(x)
            @inbounds x[i] = f(xp[i], u, p, t, noise)
        end
    else
        for i = eachindex(x)
            @inbounds x[i] = f(xp[i], u, p, t, noise)
        end
    end
    x
end


# ==========================================================================================



@forward ParticleFilter.state num_particles, weights, particles, particletype
@forward AdvancedParticleFilter.state num_particles, weights, particles, particletype

@forward AuxiliaryParticleFilter.pf state, particles, weights, expweights, reset!, weighted_mean, measurement_equation!, sample_state, sample_measurement, index, num_particles, particletype, dynamics, measurement, dynamics_density, measurement_density, initial_density, resample_threshold, resampling_strategy, rng, mode_trajectory, mean_trajectory


sample_state(pf::AbstractParticleFilter, p=parameters(pf); noise=true) = noise ? rand(pf.rng, initial_density(pf)) : mean(initial_density(pf))
sample_state(pf::ParticleFilter, x, u, p, t; noise=true) = dynamics(pf)(x,u,p,t) + noise*rand(pf.rng, dynamics_density(pf))
sample_state(pf::AdvancedParticleFilter, x, u, p, t; noise=true) = dynamics(pf)(x,u,p,t,noise)
sample_measurement(pf::AdvancedParticleFilter, x, u, p, t; noise=true) = measurement(pf)(x, u, p, t, noise)
sample_measurement(pf::AbstractParticleFilter, x, u, p, t; noise=true) = measurement(pf)(x, u, p, t) .+ noise*rand(pf.rng, pf.measurement_density)

sample_measurement(f::AbstractFilter, u, p=parameters(pf), t=pf.t[]; noise=true) = sample_measurement(f, state(f), u, p, t; noise)

expweights(pf::AbstractParticleFilter) = pf.state.we
weights(s::PFstate)                    = s.w
expweights(s::PFstate)                 = s.we
index(pf::AbstractParticleFilter)      = pf.state.t[]

num_particles(pf::AbstractParticleFilter) = num_particles(state(pf))
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
