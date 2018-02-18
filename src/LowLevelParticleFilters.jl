module LowLevelParticleFilters

export ParticleFilter, PFstate, num_particles, weights, particles, particletype, particle_smooth

using StatsBase, Plots, Distributions, StaticArrays, Parameters, Lazy

abstract type ResamplingStrategy end

struct ResampleSystematic <: ResamplingStrategy end
struct ResampleSystematicExp <: ResamplingStrategy end



struct PFstate{PT<:AbstractArray, FT<:AbstractFloat}
    x::Vector{PT}
    xprev::Vector{PT}
    w::Vector{FT}
    j::Vector{Int64}
    bins::Vector{Float64}
end

@with_kw struct ParticleFilter{ST,FT,GT,RST<:DataType}
    state::ST
    dynamics::FT
    measurement::GT
    resample_threshold::Float64 = 0.1
    resampling_strategy::RST = ResampleSystematic
end

num_particles(s::AbstractArray) = length(s)
num_particles(s::PFstate) = num_particles(s.x)
weights(s::PFstate) = s.w
particles(s::PFstate) = s.x
particletype(s::PFstate) = eltype(s.x)

@forward ParticleFilter.state num_particles, weights, particles, particletype

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
        j = resample(pf.resampling_strategy, s)
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


function forward_trajectory(pf, u, y)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(N,T)
    w = Array{Float64}(N,T)
    for t = 1:T
        pf(u[t], y[t])
        x[:,t] .= particles(pf)
        w[:,t] .= weights(pf)
    end
    x,w
end

function particle_smooth(pf, M, u, y, f_density)
    T = length(y)
    N = num_particles(pf)
    xf,wf = forward_trajectory(pf, u, y)

    xb = Array{particletype(pf)}(M,T)
    wexp = exp.(wf)
    j = resample(ResampleSystematicExp, wexp[:,T], M)
    xb[:,T] = xf[j, T]
    wb = Vector{Float64}(N)
    for t = T-1:-1:1
        for m = 1:M
            for n = 1:N
                wb[n] = wexp[n,t]*pdf(f_density, Vector(xb[m,t+1] .- xf[n,t])) + eps()
            end
            i = draw_one_categorical(wb)
            xb[m,t] = xf[i, t]
        end
    end
    return xb
end




include("resample.jl")
include("utils.jl")
end # module
