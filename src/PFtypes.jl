struct PFstate{PT<:AbstractArray, FT<:AbstractFloat}
    x::Vector{PT}
    xprev::Vector{PT}
    w::Vector{FT}
    j::Vector{Int64}
    bins::Vector{Float64}
    t::Ref{Int}
end


abstract type AbstractParticleFilter end

@with_kw struct ParticleFilter{ST,FT,GT,FDT,GDT,IDT,RST<:DataType} <: AbstractParticleFilter
    state::ST
    dynamics::FT
    measurement::GT
    dynamics_density::FDT
    measurement_density::GDT
    initial_density::IDT
    resample_threshold::Float64 = 0.1
    resampling_strategy::RST = ResampleSystematic
end


"""
ParticleFilter(num_particles, dynamics::Function, measurement::Function, dynamics_density, measurement_density, initial_density)
"""
function ParticleFilter(N::Integer, dynamics::Function, measurement::Function, dynamics_density, measurement_density, initial_density)
    xprev = Vector{SVector{length(initial_density),eltype(initial_density)}}([rand(initial_density) for n=1:N])
    x = deepcopy(xprev)
    w = fill(log(1/N), N)
    s = PFstate(x,xprev,w, Vector{Int}(N), Vector{Float64}(N),Ref(1))
    nf = numargs(dynamics)
    if nf < 3
        f = (x,u,t,p) -> dynamics(x,u)
    elseif nf < 4
        f = (x,u,t,p) -> dynamics(x,u,t)
    else
        f = dynamics
    end

    ng = numargs(measurement)
    if ng < 3
        g = (x,u,t,p) -> measurement(x,u)
    elseif ng < 4
        g = (x,u,t,p) -> measurement(x,u,t)
    else
        g = measurement
    end

    ParticleFilter(state = s, dynamics = f, measurement = g,
    dynamics_density=dynamics_density, measurement_density=measurement_density,
    initial_density=initial_density)
end


function measurement_equation!(pf, y, t, d=pf.measurement_density)
    g = pf.measurement
    @inbounds for i = 1:num_particles(pf)
        w[i] += logpdf(d, Vector(y-g(x[i],t)))
        w[i] = ifelse(w[i] < -1000, -1000, w[i])
    end
    w
end

function propagate_particles!(pf::ParticleFilter,u,j::Vector{Int}, t, d=pf.dynamics_density)
    f = pf.dynamics
    x,xp = pf.state.x, pf.state.xprev
    noise = zeros(length(x[1]))
    @inbounds for i = eachindex(x)
        x[i] =  f(xp[j[i]] ,u, t) + rand!(d, noise)
    end
    x
end

function propagate_particles!(pf::ParticleFilter,u, t, d=pf.dynamics_density)
    f = pf.dynamics
    x,xp = pf.state.x, pf.state.xprev
    noise = zeros(length(x[1]))
    @inbounds for i = eachindex(x)
        x[i] =  f(xp[i] ,u, t) + rand!(d, noise)
    end
    x
end


function propagate_particles(pf::ParticleFilter,u, t, d=pf.dynamics_density)
    f= pf.dynamics
    xp = pf.state.xprev
    x = similar(xp)
    noise = zeros(length(x[1]))
    @inbounds for i = eachindex(x)
        x[i] =  f(xp[i] ,u, t) + rand!(d, noise)
    end
    x
end


sample_measurement(pf, x, t) = pf.measurement(x, t) .+ rand(pf.measurement_density)


# Advanced =================================================================================



@with_kw struct AdvancedParticleFilter{ST,FT,GT,IDT,RST<:DataType} <: AbstractParticleFilter
    state::ST
    dynamics::FT
    measurement::GT
    initial_density::IDT
    resample_threshold::Float64 = 0.1
    resampling_strategy::RST = ResampleSystematic
end


"""
ParticleFilter(num_particles, dynamics::Function, measurement::Function, initial_density)
"""
function AdvancedParticleFilter(N::Integer, dynamics::Function, measurement::Function, initial_density)
    xprev = Vector{SVector{length(initial_density),eltype(initial_density)}}([rand(initial_density) for n=1:N])
    x = deepcopy(xprev)
    w = fill(log(1/N), N)
    s = PFstate(x,xprev,w, Vector{Int}(N), Vector{Float64}(N),Ref(1))

    AdvancedParticleFilter(state = s, dynamics = dynamics, measurement = measurement,
    initial_density=initial_density)
end


function measurement_equation!(pf::AdvancedParticleFilter, y, t)
    g = pf.measurement
    w = weights(pf)
    x = particles(pf)
    @inbounds for i = 1:num_particles(pf)
        w[i] += g(x[i],y,t)
        w[i] = ifelse(w[i] < -1000, -1000, w[i])
    end
    w
end

function propagate_particles!(pf::AdvancedParticleFilter, u, j, t::Int, noise::Bool=true)
    f = pf.dynamics
    x,xp = pf.state.x, pf.state.xprev
    @inbounds for i = eachindex(x)
        x[i] =  f(xp[j[i]] ,u, t, noise)
    end
    x
end

function propagate_particles!(pf::AdvancedParticleFilter, u, t::Int, noise::Bool=true)
    f = pf.dynamics
    x,xp = pf.state.x, pf.state.xprev
    @inbounds for i = eachindex(x)
        x[i] =  f(xp[i] ,u, t, noise)
    end
    x
end


function propagate_particles(pf::AdvancedParticleFilter, u, t::Int, noise::Bool=true)
    f = pf.dynamics
    xp = pf.state.xprev
    x = similar(xp)
    @inbounds for i = eachindex(x)
        x[i] =  f(xp[i] ,u, t, noise)
    end
    x
end

sample_measurement(pf::AdvancedParticleFilter, x, t) = pf.measurement(x, t)

# ==========================================================================================

num_particles(s::AbstractArray) = length(s)
num_particles(s::PFstate) = num_particles(s.x)
weights(s::PFstate) = s.w
particles(s::PFstate) = s.x
particletype(s::PFstate) = eltype(s.x)

@forward ParticleFilter.state num_particles, weights, particles, particletype
@forward AdvancedParticleFilter.state num_particles, weights, particles, particletype
