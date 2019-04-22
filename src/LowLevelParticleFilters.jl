module LowLevelParticleFilters

export KalmanFilter, ParticleFilter, AdvancedParticleFilter, PFstate, index, state, covariance, num_particles, weights, expweights, particles, particletype, smooth, sample_measurement, simulate, loglik, log_likelihood_fun, forward_trajectory, mean_trajectory, reset!, metropolis

using StatsBase, Parameters, Lazy, Yeppp, Random, LinearAlgebra
import PDMats # To extend some methods on static arrays
using StaticArrays
using Distributions
using StatsPlots

abstract type ResamplingStrategy end

struct ResampleSystematic <: ResamplingStrategy end
struct ResampleSystematicExp <: ResamplingStrategy end

include("PFtypes.jl")

# Base.Vector(v::Distributions.ZeroVector) = zeros(eltype(v), length(v))

function reset!(kf::AbstractKalmanFilter)
    kf.x .= Vector(kf.d0.μ)
    kf.R .= copy(Matrix(kf.d0.Σ))
    kf.t[] = 1
end

"""
    Reset the filter to initial state and covariance/distribution
"""
function reset!(pf)
    s = pf.state
    for i = eachindex(s.xprev)
        s.xprev[i] = rand(pf.rng, pf.initial_density)
        s.x[i] = copy(s.xprev[i])
    end
    fill!(s.w, log(1/num_particles(pf)))
    fill!(s.we, 1/num_particles(pf))
    s.t[] = 1
end

function predict!(kf::AbstractKalmanFilter, u, t = index(kf))
    @unpack A,B,x,R,R1 = kf
    if ndims(A) == 3
        At = A[:,:,t]
        Bt = B[:,:,t]
    else
        At = A
        Bt = B
    end
    x .= At*x .+ Bt*u
    R .= At*R*At' + R1
    kf.t[] += 1
end

function correct!(kf::AbstractKalmanFilter, y, t = index(kf))
    @unpack C,x,R,R1,R2,R2d = kf
    if ndims(C) == 3
        Ct = C[:,:,t]
    else
        Ct = C
    end
    e   = y .- Ct*x
    F   = Ct*R*Ct'
    F   = 0.5(F+F')
    K   = (R*Ct')/(F + R2) # Do not use .+ if R2 is I
    x .+= K*e
    R  .= (I - K*Ct)*R # warning against I .- A
    logpdf(MvNormal(F), e) - 1/2*logdet(F)
end

"""
    predict!(f,u, t = index(f))
Move filter state forward in time using dynamics equation and input vector `u`.
"""
function predict!(pf,u, t = index(pf))
    s = pf.state
    N = num_particles(s)
    if shouldresample(pf)
        j = resample(pf)
        propagate_particles!(pf, u, j, t)
        reset_weights!(s)
    else # Resample not needed
        propagate_particles!(pf, u, t)
    end
    copyto!(s.xprev, s.x)
    pf.state.t[] += 1
end

"""
 ll = correct!(f, y, t = index(f))
Update state/covariance/weights based on measurement `y`,  returns loglikelihood.
"""
function correct!(pf, y, t = index(pf))
    measurement_equation!(pf, y, t)
    loklik = logsumexp!(pf.state)
end

"""
ll = update!(f::AbstractFilter, u, y, t = index(f))
Perform one step of `predict!` and `correct!`, returns loglikelihood.
"""
function update!(f::AbstractFilter, u, y, t = index(f))
    predict!(f, u, t)
    loklik = correct!(f, y, t)
end


(kf::KalmanFilter)(u, y, t = index(kf)) =  update!(kf, u, y, t)
(pf::ParticleFilter)(u, y, t = index(pf)) =  update!(pf, u, y, t)
(pf::AdvancedParticleFilter)(u, y, t = index(pf)) =  update!(pf, u, y, t)


"""
x,xt,R,Rt,ll = forward_trajectory(kf, u::Vector{Vector}, y::Vector{Vector})
x,w,ll       = forward_trajectory(pf, u::Vector{Vector}, y::Vector{Vector})

This Function resets the filter to the initial state distribution upon start
"""
function forward_trajectory(kf::AbstractKalmanFilter, u::Vector, y::Vector)
    reset!(kf)
    T     = length(y)
    x     = Array{particletype(kf)}(undef,T)
    xt    = Array{particletype(kf)}(undef,T)
    R     = Array{covtype(kf)}(undef,T)
    Rt    = Array{covtype(kf)}(undef,T)
    x[1]  = state(kf)       |> copy
    R[1]  = covariance(kf)  |> copy
    ll    = correct!(kf, y[1], 1)
    xt[1] = state(kf)       |> copy
    Rt[1] = covariance(kf)  |> copy
    for t = 2:T
        predict!(kf, u[t-1], t-1)
        x[t]   = state(kf)              |> copy
        R[t]   = covariance(kf)         |> copy
        ll    += correct!(kf, y[t], t)
        xt[t]  = state(kf)              |> copy
        Rt[t]  = covariance(kf)         |> copy
    end
    x,xt,R,Rt,ll
end

"""
xT,RT,ll = smooth(kf::AbstractKalmanFilter, u::Vector, y::Vector)
Returns smoothed estimates of state `x` and covariance `R` given all input output data `u,y`
"""
function smooth(kf::AbstractKalmanFilter, u::AbstractVector, y::AbstractVector)
    reset!(kf)
    T            = length(y)
    x,xt,R,Rt,ll = forward_trajectory(kf, u, y)
    xT           = similar(xt)
    RT           = similar(Rt)
    xT[end]      = xt[end]      |> copy
    RT[end]      = Rt[end]      |> copy
    for t = T-1:-1:1
        C     = Rt[t]/R[t+1]
        xT[t] = xt[t] .+ C*(xT[t+1] .- x[t+1])
        RT[t] = Rt[t] .+ C*(RT[t+1] .- R[t+1])*C'
    end
    xT,RT,ll
end


function forward_trajectory(pf, u::AbstractVector, y::AbstractVector)
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(undef,N,T)
    w = Array{Float64}(undef,N,T)
    we = Array{Float64}(undef,N,T)
    ll = 0.0
    for t = 1:T
        ll += pf(u[t], y[t], t)
        x[:,t] .= particles(pf)
        w[:,t] .= weights(pf)
        we[:,t] .= expweights(pf)
    end
    x,w,we,ll
end



"""
x,ll = mean_trajectory(pf, u::Vector{Vector}, y::Vector{Vector})

This Function resets the particle filter to the initial state distribution upon start
"""
function mean_trajectory(pf, u::Vector, y::Vector)
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(undef,T)
    ll = 0.0
    for t = 1:T
        ll += pf(u[t], y[t], t)
        x[t] = weigthed_mean(pf)
    end
    x,ll
end



# Some methods to speed up distributions using static arrays

@inline Base.:(-)(x::StaticArray, ::Distributions.ZeroVector) = x
Distributions.logpdf(d::Distribution,x,xp,t) = logpdf(d,x-xp)
Distributions.sqmahal(d::MvNormal, x::StaticArray) = Distributions.invquad(d.Σ, x - d.μ)
@inline PDMats.invquad(a::PDMats.ScalMat, x::StaticVector) = dot(x,x) * a.inv_value
PDMats.invquad(a::PDMats.PDMat, x::StaticVector) = dot(x, a \ x) # \ not implemented
Base.:(\)(a::PDMats.PDMat, x::StaticVector) = a.chol \ x
PDMats.invquad(a::PDMats.PDiagMat, x::StaticVector) = PDMats.wsumsq(a.inv_diag, x)

"""
xb,ll = smooth(pf, M, u, y)
"""
function smooth(pf::AbstractParticleFilter, M, u, y)
    T = length(y)
    N = num_particles(pf)
    xf,wf,wef,ll = forward_trajectory(pf, u, y)
    @assert M <= N "Must extend cache size of bins and j to allow this"
    xb = Array{particletype(pf)}(undef,M,T)
    j = resample(ResampleSystematic, wef[:,T], M)
    for i = 1:M
        xb[i,T] = xf[j[i], T]
    end
    wb = Vector{Float64}(undef,N)
    for t = T-1:-1:1
        for m = 1:M
            for n = 1:N
                wb[n] = wf[n,t] + logpdf(pf.dynamics_density, xb[m,t+1], xf[n,t], t)
            end
            i = draw_one_categorical(pf,wb)
            xb[m,t] = xf[i, t]
        end
    end
    return xb,ll
end

function loglik(f,u,y)
    reset!(f)
    ll = sum(x->f(x[1],x[2]), zip(u, y))
end

"""
ll(θ) = log_likelihood_fun(filter_from_parameters(θ::Vector)::Function, priors::Vector{Distribution}, u, y)

returns function θ -> p(y|θ)p(θ)
"""
function log_likelihood_fun(filter_from_parameters,priors::Vector{<:Distribution},u,y)
    function (θ)
        length(θ) == length(priors) || throw(ArgumentError("Input must have same length as priors"))
        ll = sum(i->logpdf(priors[i], θ[i]), eachindex(priors))
        isfinite(ll) || return Inf
        pf = filter_from_parameters(θ)
        ll += loglik(pf,u,y)
    end
end

function naive_sampler(θ₀)
    !any(iszero.(θ₀)) || throw(ArgumentError("Naive sampler does not work if initial parameter vector contains zeros (it was going to return θ -> θ .+ rand(MvNormal(0.1abs.(θ₀))), but that is not a good ideas if θ₀ is zero."))
    θ -> θ .+ rand(MvNormal(0.1abs.(θ₀)))
end

"""
    metropolis(ll::Function(θ), R::Int, θ₀::Vector, draw::Function(θ) = naive_sampler(θ₀))

Performs MCMC sampling using the marginal Metropolis (-Hastings) algorithm
`draw = θ -> θ'` samples a new parameter vector given an old parameter vector. The distribution must be symmetric, e.g., a Gaussian. `R` is the number of iterations.
See `log_likelihood_fun`

# Example:
```julia
filter_from_parameters(θ) = ParticleFilter(N, dynamics, measurement, MvNormal(n,exp(θ[1])), MvNormal(p,exp(θ[2])), d0)
priors = [Normal(0,0.1),Normal(0,0.1)]
ll     = log_likelihood_fun(filter_from_parameters,priors,u,y,1)
θ₀ = log.([1.,1.]) # Initial point
draw = θ -> θ .+ rand(MvNormal(0.1ones(2))) # Function that proposes new parameters (has to be symmetric)
burnin = 200 # If using threaded call, provide number of burnin iterations
# @time theta, lls = metropolis(ll, 2000, θ₀, draw) # Run single threaded
# thetam = reduce(hcat, theta)'
@time thetalls = LowLevelParticleFilters.metropolis_threaded(burnin, ll, 5000, θ₀, draw) # run on all threads, will provide (2000-burnin)*nthreads() samples
histogram(exp.(thetalls[:,1:2]), layout=3)
plot!(thetalls[:,3], subplot=3) # if threaded call, log likelihoods are in the last column
```
"""
function metropolis(ll, R, θ₀, draw = naive_sampler(θ₀))
    params    = Vector{typeof(θ₀)}(undef,R)
    lls       = Vector{Float64}(undef,R)
    params[1] = θ₀
    lls[1]    = ll(θ₀)
    for i = 2:R
        θ = draw(params[i-1])
        lli = ll(θ)
        if rand() < exp(lli-lls[i-1])
            params[i] = θ
            lls[i] = lli
        else
            params[i] = params[i-1]
            lls[i] = lls[i-1]
        end
    end
    params, lls
end

function metropolis_threaded(burnin, args...)
    res = []
    mtx = Threads.Mutex()
    Threads.@threads for i = 1:Threads.nthreads()
        p,l = metropolis(args...)
        resi = [reduce(hcat,p)' l]
        resi = resi[burnin+1:end,:]
        lock(mtx)
        push!(res, resi)
        unlock(mtx)
    end
    reduce(vcat,res)
end

"""
x,u,y = simulate(f::AbstractFilter,T::Int,du::Distribution)
Simulate dynamical system forward in time, returns state sequence, inputs and measurements
`du` is a distribution of random inputs
"""
function simulate(f::AbstractFilter,T::Int,du::Distribution)
    u = [rand(du) for t=1:T]
    y = Vector{Vector{Float64}}(undef,T)
    x = Vector{Vector{Float64}}(undef,T)
    x[1] = sample_state(f)
    for t = 1:T-1
        y[t] = sample_measurement(f,x[t], t)
        x[t+1] = sample_state(f, x[t], u[t], t)
    end
    y[T] = sample_measurement(f,x[T], T)
    x,u,y
end

include("resample.jl")
include("utils.jl")
end # module
