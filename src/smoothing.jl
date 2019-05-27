
"""
xT,RT,ll = smooth(kf::KalmanFilter, u::Vector, y::Vector)
Returns smoothed estimates of state `x` and covariance `R` given all input output data `u,y`
"""
function smooth(kf::KalmanFilter, u::AbstractVector, y::AbstractVector)
    reset!(kf)
    T            = length(y)
    x,xt,R,Rt,ll = forward_trajectory(kf, u, y)
    xT           = similar(xt)
    RT           = similar(Rt)
    xT[end]      = xt[end]      |> copy
    RT[end]      = Rt[end]      |> copy
    for t = T-1:-1:1
        C     = Rt[t]*kf.A/R[t+1]
        xT[t] = xt[t] .+ C*(xT[t+1] .- x[t+1])
        RT[t] = Rt[t] .+ C*(RT[t+1] .- R[t+1])*C'
    end
    xT,RT,ll
end

function smooth(pf::AbstractParticleFilter, M, u, y)
    xf,wf,wef,ll = forward_trajectory(pf, u, y)
    smooth(pf::AbstractParticleFilter, xf, wf, wef, ll, M, u, y)
end

"""
    xb,ll = smooth(pf, M, u, y)

Perform particle smoothing using forward-filtering, backward simulation. Return smoothed particles and loglikelihood.
See also `smoothed_trajs`, `smoothed_mean`, `smoothed_cov`
"""
function smooth(pf::AbstractParticleFilter, xf, wf, wef, ll, M, u, y)
    T = length(y)
    N = num_particles(pf)
    f = dynamics(pf)
    df = dynamics_density(pf)
    @assert M <= N "Must extend cache size of bins and j to allow this"
    xb = Array{particletype(pf)}(undef,M,T)
    j = resample(ResampleSystematic, wef[:,T], M)
    # @show Set(j)
    for i = 1:M
        xb[i,T] = xf[j[i], T]
    end
    wb = Vector{Float64}(undef,N)
    @inbounds for t = T-1:-1:1
        # tset = Set{Int}()
        for m = 1:M
            for n = 1:N
                wb[n] = wf[n,t] + logpdf(df, xb[m,t+1], f(xf[n,t],u[t],t), t)
            end
            i = draw_one_categorical(pf,wb)
            # push!(tset, i)
            xb[m,t] = xf[i, t]
        end
        # @show tset
    end
    return xb,ll
end




"""
    ll = loglik(filter,u,y)
Calculate loglikelihood for entire sequences `u,y`
"""
function loglik(f,u,y)
    reset!(f)
    ll = sum(x->f(x[1],x[2]), zip(u, y))
end

function loglik(pf::AuxiliaryParticleFilter,u,y)
    reset!(pf)
    ll = sum(t->pf(u[t],y[t],y[t+1],t), 1:length(u)-1)
    ll + pf.pf(u[end],y[end], length(u))
end

"""
ll(θ) = log_likelihood_fun(filter_from_parameters(θ::Vector)::Function, priors::Vector{Distribution}, u, y)

returns function θ -> p(y|θ)p(θ)
"""
function log_likelihood_fun(filter_from_parameters,priors::Vector{<:Distribution},u,y)
    n = numargs(filter_from_parameters)
    pf = nothing
    function (θ)
        pf === nothing && (pf = filter_from_parameters(θ))
        length(θ) == length(priors) || throw(ArgumentError("Input must have same length as priors"))
        ll = sum(i->logpdf(priors[i], θ[i]), eachindex(priors))
        isfinite(ll) || return -Inf
        pf = filter_from_parameters(θ,pf)
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
    smoothed_mean(xb)
Helper function to calculate the mean of smoothed particle trajectories
"""
function smoothed_mean(xb)
    M,T = size(xb)
    n = length(xb[1])
    xbm = vec(mean(xb,dims=1))
    reduce(hcat, xbm)
end

"""
    smoothed_cov(xb)
Helper function to calculate the covariance of smoothed particle trajectories
"""
function smoothed_cov(xb)
    M,T = size(xb)
    n = length(xb[1])
    xbc = [cov(copy(reshape(reinterpret(Float64, xb[:,t]),n,M)),dims=2) for t = 1:T]
end

"""
    smoothed_trajs(xb)
Helper function to get particle trajectories as a 3-dimensions array (N,M,T) instead of matrix of vectors.
"""
function smoothed_trajs(xb)
    M,T = size(xb)
    n = length(xb[1])
    copy(reshape(reinterpret(Float64, xb), n,M,T))
end
