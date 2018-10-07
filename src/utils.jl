export weigthed_mean, weigthed_cov, plot_trajectories, scatter_particles, logsumexp!, smoothed_mean, smoothed_cov, smoothed_trajs, plot_priors

"""
ll = logsumexp!(w)
Normalizes the weight vector `w` and returns the weighted log-likelihood
"""
function logsumexp!(w)
    offset = maximum(w)
    nc = sum(w->exp(w-offset), w)
    nc = log(nc) + offset
    w .-= nc
    nc - log(length(w))
end

function weigthed_mean(x,w::AbstractVector)
    xh = zeros(size(x[1]))
    @inbounds @simd  for i = eachindex(x)
        xh .+= x[i].*exp(w[i])
    end
    return xh
end
function weigthed_mean(x,w::AbstractMatrix)
    N,T = size(x)
    xh = zeros(eltype(x), T)
    for t = 1:T
        @inbounds @simd for i = 1:N
            xh[t] = xh[t] + x[i,t].*exp(w[i,t])
        end
    end
    return xh
end
weigthed_mean(s) = weigthed_mean(s.x,s.w)
weigthed_mean(pf::AbstractParticleFilter) = weigthed_mean(pf.state)

function weigthed_cov(x,w)
    N,T = size(x)
    n = length(x[1])
    [cov(copy(reshape(reinterpret(Float64, x[:,t]),n,N)),ProbabilityWeights(exp.(w[:,t])), dims=2, corrected=true) for t = 1:T]
end

function smoothed_mean(xb)
    M,T = size(xb)
    n = length(xb[1])
    xbm = sum(xb,dims=1)[:] ./ M
    copy(reshape(reinterpret(Float64, xbm), n,T))
end

function smoothed_cov(xb)
    M,T = size(xb)
    n = length(xb[1])
    xbc = [cov(copy(reshape(reinterpret(Float64, xb[:,t]),n,M)),dims=2) for t = 1:T]
end

function smoothed_trajs(xb)
    M,T = size(xb)
    n = length(xb[1])
    copy(reshape(reinterpret(Float64, xb), n,M,T))
end

function plot_trajectories(pf,y,xt)
    xa = reinterpret(Float64, particles(pf), (length(particles(pf)[1]), length(particles(pf))))
    scatter(xa[1,:],xa[2,:], title="Particles", reuse=true, xlims=(-15,15), ylims=(-15,15), grid=false, size=(1000,1000))
    scatter!([y[1]], [y[2]], m = (:red, 5))
    scatter!([xt[1]], [xt[2]], m = (:green, 5))
    sleep(0.2)
end

function scatter_particles(pf,xt,t; kwargs...)
    dim, T = size(xt)
    np = num_particles(pf)
    xa = reinterpret(Float64, pf.s.x, (dim, np))
    plot(xt', title="Particles", reuse=true,  grid=false, layout=dim, kwargs...)
    plot!(y', l = (:red, 2))
    I = t*ones(np)
    for i = 1:dim
        scatter!(I, xa[i,:], subplot=i)
    end
    sleep(0.2)
end

function plot_priors(priors; kwargs...)
    fig = plot(priors[1]; layout=length(priors), kwargs...)
    for i = 2:length(priors)
        plot!(priors[i], subplot=i)
    end
    fig
end

"""
`numparameters(f)`
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
