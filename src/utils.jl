export weigthed_mean, weigthed_cov, plot_trajectories, scatter_particles, logsumexp!, smoothed_mean, smoothed_cov, smoothed_trajs, plot_priors

"""
ll = logsumexp!(w, we [, maxw])
Normalizes the weight vector `w` and returns the weighted log-likelihood

https://arxiv.org/pdf/1412.8695.pdf eq 3.8 for p(y)
https://discourse.julialang.org/t/fast-logsumexp/22827/7?u=baggepinnen for stable logsumexp
"""
function logsumexp!(w,we,maxw=Ref(zero(eltype(w))))
    offset,maxind = findmax(w)
    w  .-= offset
    Yeppp.exp!(we,w)
    s    = sum_all_but(we,maxind) # s = ∑wₑ-1
    we .*= 1/(s+1)
    w  .-= log1p(s)
    maxw[] += offset
    log1p(s) + maxw[] - log(length(w))
end

logsumexp!(s) = logsumexp!(s.w,s.we,s.maxw)
logsumexp!(pf::AbstractParticleFilter) = logsumexp!(pf.state)
# function logsumexp!(w,we)
#     offset = maximum(w)
#     # w  .-= offset
#     we  .= exp.(w .- offset)
#     s    = sum(we)
#     we .*= 1/s
#     w  .-= (log(s) + offset)
#     s/exp(-offset) - log(length(w))
# end


function sum_all_but(w,i)
    w[i] -= 1
    s = sum(w)
    w[i] += 1
    s
end

function reset_weights!(s)
    N = num_particles(s)
    fill!(s.w, log(1/N))
    fill!(s.we, 1/N)
    s.maxw[] = 0
end
reset_weights!(pf::AbstractParticleFilter) = reset_weights!(state(pf))

"""
    x̂ = weigthed_mean(x,we)
Calculated weighted mean of particle trajectories. `we` are expweights.
"""
function weigthed_mean(x,we::AbstractVector)
    @assert sum(we) ≈ 1
    xh = zeros(size(x[1]))
    @inbounds @simd  for i = eachindex(x)
        xh .+= x[i].*we[i]
    end
    return xh
end
function weigthed_mean(x,we::AbstractMatrix)
    @assert sum(we) ≈ 1
    N,T = size(x)
    xh = zeros(eltype(x), T)
    for t = 1:T
        @inbounds @simd for i = 1:N
            xh[t] += x[i,t].*we[i,t]
        end
    end
    return xh
end

"""
    x̂ = weigthed_mean(pf)
    x̂ = weigthed_mean(s::PFstate)
"""
weigthed_mean(s) = weigthed_mean(s.x,s.we)
weigthed_mean(pf::AbstractParticleFilter) = weigthed_mean(pf.state)
"""
    Similar to `weigthed_mean`, but returns covariances
"""
function weigthed_cov(x,we)
    N,T = size(x)
    n = length(x[1])
    [cov(copy(reshape(reinterpret(Float64, x[:,t]),n,N)),ProbabilityWeights(we[:,t]), dims=2, corrected=true) for t = 1:T]
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

# function plot_trajectories(pf,y,xt)
#     xa = reinterpret(Float64, particles(pf), (length(particles(pf)[1]), length(particles(pf))))
#     scatter(xa[1,:],xa[2,:], title="Particles", reuse=true, xlims=(-15,15), ylims=(-15,15), grid=false, size=(1000,1000))
#     scatter!([y[1]], [y[2]], m = (:red, 5))
#     scatter!([xt[1]], [xt[2]], m = (:green, 5))
#     sleep(0.2)
# end

# function scatter_particles(pf,xt,t; kwargs...)
#     dim, T = size(xt)
#     np = num_particles(pf)
#     xa = reinterpret(Float64, pf.s.x, (dim, np))
#     plot(xt', title="Particles", reuse=true,  grid=false, layout=dim, kwargs...)
#     plot!(y', l = (:red, 2))
#     I = t*ones(np)
#     for i = 1:dim
#         scatter!(I, xa[i,:], subplot=i)
#     end
#     sleep(0.2)
# end

function plot_priors(priors; kwargs...)
    fig = plot(priors[1]; layout=length(priors), kwargs...)
    for i = 2:length(priors)
        plot!(priors[i], subplot=i)
    end
    fig
end

"""
    numparameters(f)
Returns the number of parameters of `f` for the method which has the most parameters. This function is shamefully borrowed from [DiffEqBase.jl](https://github.com/JuliaDiffEq/DiffEqBase.jl/blob/master/src/utils.jl#L6)
"""
function numargs(f)
    numparam = maximum([num_types_in_tuple(m.sig) for m in methods(f)])
    return (numparam-1) #-1 in v0.5 since it adds f as the first parameter
end

function num_types_in_tuple(sig)
    length(sig.parameters)
end

function num_types_in_tuple(sig::UnionAll)
    length(Base.unwrap_unionall(sig).parameters)
end
