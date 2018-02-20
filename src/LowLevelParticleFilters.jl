module LowLevelParticleFilters

export ParticleFilter, AdvancedParticleFilter, PFstate, num_particles, weights, particles, particletype, particle_smooth, sample_measurement, simulate, loglik, negative_log_likelihood_fun, forward_trajectory, mean_trajectory, reset!

using StatsBase, Parameters, Lazy, Reexport
@reexport using Distributions
@reexport using StatPlots
@reexport using StaticArrays

abstract type ResamplingStrategy end

struct ResampleSystematic <: ResamplingStrategy end
struct ResampleSystematicExp <: ResamplingStrategy end

include("PFtypes.jl")

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
    N = num_particles(s)
    if shouldresample(s.w)
        j = resample(pf)
        propagate_particles!(pf, u, j, t)
        fill!(s.w, log(1/N))
    else # Resample not needed
        propagate_particles!(pf, u, t)
    end
    pf.state.t[] += 1
end

function correct!(pf, y, t = pf.state.t[])
    measurement_equation!(pf, y, t)
    loklik = logsumexp!(pf.state.w)
end

function update!(pf::AbstractParticleFilter, u, y, t = pf.state.t[])
    s = pf.state
    predict!(pf, u, t)
    loklik = correct!(pf, y, t)
    copy!(s.xprev, s.x)
    loklik
end


(pf::ParticleFilter)(u, y, t = pf.state.t[]) =  update!(pf, u, y, t)
(pf::AdvancedParticleFilter)(u, y, t = pf.state.t[]) =  update!(pf, u, y, t)


"""
x,w,ll = forward_trajectory(pf, u::Vector{Vector}, y::Vector{Vector})

This function resets the particle filter to the initial state distribution upon start
"""
function forward_trajectory(pf, u::Vector, y::Vector)
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

# Catch-all method that assumes additive noise
Distributions.pdf(d::Distribution,x,xp,t) = pdf(d,Vector(x.-xp))

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
                wb[n] = wexp[n,t]*pdf(pf.dynamics_density, xb[m,t+1], xf[n,t], t) + eps()
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
