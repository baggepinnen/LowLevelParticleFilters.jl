module LowLevelParticleFilters

export ParticleFilter, PFstate, num_particles

using StatsBase, Plots, Distributions, StaticArrays, TimerOutputs, Parameters


struct PFstate{PT<:AbstractArray, FT<:AbstractFloat}
    x::Vector{PT}
    xprev::Vector{PT}
    w::Vector{FT}
    j::Vector{Int64}
    bins::Vector{Float64}
end

@with_kw struct ParticleFilter{ST,FT,GT}
    state::ST
    dynamics::FT
    measurement::GT
    resample_threshold::Float64 = 0.1
end

num_particles(s::PFstate) = length(s.x)
num_particles(pf::ParticleFilter) = num_particles(pf.state)

function ParticleFilter(N::Integer, p0::Distribution, dynamics_function::Function, measurement_function::Function)
    xprev = Vector{SVector{length(p0),eltype(p0)}}([rand(p0) for n=1:N])
    x = similar(xprev)
    w = fill(log(1/N), N)
    s = PFstate(x,xprev,w, Vector{Int}(N), Vector{Float64}(N))
    ParticleFilter(state = s, dynamics = dynamics_function, measurement = measurement_function)
end

function (pf::ParticleFilter)(u, y)
    s = pf.state
    f = pf.dynamics
    g = pf.measurement
    N = num_particles(s)
    if shouldresample(s.w)
        j = resample(s)
        f(s.x, s.xprev, u, j)
        fill!(s.w, log(1/N))
    else # Resample not needed
        f(s.x, s.xprev, u, 1:N)
    end
    g(s.w, s.x, y)
    logsumexp!(s.w)
    copy!(s.xprev, s.x)
    s.x
end


include("utils.jl")
end # module
