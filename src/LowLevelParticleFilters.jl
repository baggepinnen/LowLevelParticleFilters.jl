module LowLevelParticleFilters

export ParticleFilter, PFstate, num_particles, weights, particles, particletype, particle_smooth, sample_measurement, simulate, loglik, negative_log_likelihood_fun, forward_trajectory, mean_trajectory, reset!

using StatsBase, Parameters, Lazy, Reexport
@reexport using Distributions
@reexport using StatPlots
@reexport using StaticArrays

abstract type ResamplingStrategy end

struct ResampleSystematic <: ResamplingStrategy end
struct ResampleSystematicExp <: ResamplingStrategy end


struct PFstate{PT<:AbstractArray, FT<:AbstractFloat}
    x::Vector{PT}
    xprev::Vector{PT}
    w::Vector{FT}
    j::Vector{Int64}
    bins::Vector{Float64}
    t::Ref{Int}
end

@with_kw struct ParticleFilter{ST,FT,GT,FDT,GDT,IDT,RST<:DataType}
    state::ST
    dynamics::FT
    measurement::GT
    dynamics_density::FDT
    measurement_density::GDT
    initial_density::IDT
    resample_threshold::Float64 = 0.1
    resampling_strategy::RST = ResampleSystematic
end

num_particles(s::AbstractArray) = length(s)
num_particles(s::PFstate) = num_particles(s.x)
weights(s::PFstate) = s.w
particles(s::PFstate) = s.x
particletype(s::PFstate) = eltype(s.x)

@forward ParticleFilter.state num_particles, weights, particles, particletype

"""
ParticleFilter(num_particles, dynamics::Function, measurement::Function, dynamics_density, measurement_density, initial_density)
"""
function ParticleFilter(N::Integer, dynamics::Function, measurement::Function, dynamics_density, measurement_density, initial_density)
    xprev = Vector{SVector{length(initial_density),eltype(initial_density)}}([rand(initial_density) for n=1:N])
    x = deepcopy(xprev)
    w = fill(log(1/N), N)
    s = PFstate(x,xprev,w, Vector{Int}(N), Vector{Float64}(N),Ref(1))
    ParticleFilter(state = s, dynamics = dynamics, measurement = measurement,
    dynamics_density=dynamics_density, measurement_density=measurement_density,
    initial_density=initial_density)
end

function reset!(pf)
    s = pf.state
    for i = eachindex(s.xprev)
        s.xprev[i] = rand(pf.initial_density)
        s.x[i] = copy(s.xprev[i])
    end
    fill!(s.w, log(1/num_particles(pf)))
    s.t[] = 1
end


function predict!(pf,u, t = pf.state.t[])
    s = pf.state
    f,df = pf.dynamics, pf.dynamics_density
    N = num_particles(s)
    if shouldresample(s.w)
        j = resample(pf.resampling_strategy, s)
        propagate_particles!(s.x, s.xprev, u, j, f, df, t)
        fill!(s.w, log(1/N))
    else # Resample not needed
        propagate_particles!(s.x, s.xprev, u, f, df, t)
    end
    pf.state.t[] += 1
end

function correct!(pf,y, t = pf.state.t[])
    s = pf.state
    g,dg = pf.measurement, pf.measurement_density
    measurement_equation!(s.w, s.x, y, g, dg, t)
    loklik = logsumexp!(s.w)
end


function (pf::ParticleFilter)(u, y, t = pf.state.t[])
    s = pf.state
    predict!(pf,u, t)
    loklik = correct!(pf,y, t)
    copy!(s.xprev, s.x)
    loklik
end

"""
x,w,ll = forward_trajectory(pf, u::Vector{Vector}, y::Vector{Vector})

This function resets the particle filter to the initial state distribution upon start
"""
function forward_trajectory(pf, u::Vector{Vector}, y::Vector{Vector})
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(N,T)
    w = Array{Float64}(N,T)
    ll = 0.0
    for t = 1:T
        ll += pf(u[t], y[t], t)
        x[:,t] .= particles(pf)
        w[:,t] .= weights(pf)
    end
    x,w,ll
end

"""
x,ll = mean_trajectory(pf, u::Vector{Vector}, y::Vector{Vector})

This function resets the particle filter to the initial state distribution upon start
"""
function mean_trajectory(pf, u::Vector, y::Vector)
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(T)
    ll = 0.0
    for t = 1:T
        ll += pf(u[t], y[t], t)
        x[t] = weigthed_mean(pf)
    end
    x,ll
end

"""
xb = particle_smooth(pf, M, u, y)
"""
function particle_smooth(pf, M, u, y)
    T = length(y)
    N = num_particles(pf)
    xf,wf,ll = forward_trajectory(pf, u, y)
    @assert M <= N "Must extend cache size of bins and j to allow this"
    xb = Array{particletype(pf)}(M,T)
    wexp = exp.(wf)
    j = resample(ResampleSystematicExp, wexp[:,T], M)
    for i = 1:M
        xb[i,T] = xf[j[i], T]
    end
    wb = Vector{Float64}(N)
    for t = T-1:-1:1
        for m = 1:M
            for n = 1:N
                wb[n] = wexp[n,t]*pdf(pf.dynamics_density, Vector(xb[m,t+1] .- xf[n,t])) + eps()
            end
            i = draw_one_categorical(wb)
            xb[m,t] = xf[i, t]
        end
    end
    return xb
end

function loglik(pf,u,y)
    reset!(pf)
    ll = sum((x)->pf(x[1],x[2]), zip(u, y))
end

"""
nll(θ) = negative_log_likelihood_fun(filter_from_parameters(θ::Vector)::Function, priors::Vector{Distribution}, u, y, averaging=1)
"""
function negative_log_likelihood_fun(filter_from_parameters,priors,u,y,mc=1)
    function (θ)
        lls = map(1:mc) do j
            pf = filter_from_parameters(θ)
            -loglik(pf,u,y)  - sum(i->logpdf(priors[i], θ[i]), eachindex(priors))
        end
        median(lls)
    end
end

function measurement_equation!(w,x,y,g::Function,dg::Distribution, t)
    @inbounds for i = 1:length(w)
        w[i] += logpdf(dg, Vector(y-g(x[i],t)))
        w[i] = ifelse(w[i] < -1000, -1000, w[i])
    end
    w
end

function propagate_particles!(x,xp,u,j, f::Function, df::Distribution, t)
    @inbounds for i = eachindex(x)
        x[i] =  f(xp[j[i]] ,u, t) + rand(df)
    end
    x
end

function propagate_particles!(x,xp,u, f::Function, df::Distribution, t)
    @inbounds for i = eachindex(x)
        x[i] =  f(xp[i] ,u, t) + rand(df)
    end
    x
end

function propagate_particles!(x,xp,u,f::Function, t)
    @inbounds for i = eachindex(x)
        x[i] =  f(xp[i] ,u, t)
    end
    x
end

function propagate_particles(xp,u, f::Function, df::Distribution, t)
    x = similar(xp)
    @inbounds for i = eachindex(x)
        x[i] =  f(xp[i] ,u, t) + rand(df)
    end
    x
end

function propagate_particles(xp,u,f::Function, t)
    x = similar(xp)
    @inbounds for i = eachindex(x)
        x[i] =  f(xp[i] ,u, t)
    end
    x
end

sample_measurement(pf,x, t) = pf.measurement(x, t) .+ rand(pf.measurement_density)

function simulate(pf,T,du)
    u = [rand(du) for t=1:T]
    y = Vector{Vector{Float64}}(T)
    x = Vector{Vector{Float64}}(T)
    x[1] = rand(pf.initial_density)
    for t = 1:T-1
        y[t] = sample_measurement(pf,x[t], t)
        x[t+1] = pf.dynamics(x[t],u[t], t) + rand(pf.dynamics_density)
    end
    y[T] = sample_measurement(pf,x[T], t)
    x,u,y
end

include("resample.jl")
include("utils.jl")
end # module
