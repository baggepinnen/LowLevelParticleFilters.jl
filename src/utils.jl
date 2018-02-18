export weigthed_mean, plot_trajectories, scatter_particles, logsumexp!

function logsumexp!(w)
    offset = maximum(w)
    normConstant = sum(w->exp(w-offset), w)
    w .-= log(normConstant) + offset
end

function weigthed_mean(x,w)
    xh = zeros(size(x[1]))
    @inbounds @simd  for i = eachindex(x)
        xh .+= x[i].*exp(w[i])
    end
    return xh
end
weigthed_mean(s) = weigthed_mean(s.x,s.w)
weigthed_mean(pf::ParticleFilter) = weigthed_mean(pf.state)

function Base.mean(xb::Matrix{SVector})
    M,T = size(xb,2)
    n = length(xb[1])
    xbm = sum(xb,1)[:] ./ M
    reinterpret(Float64, xbm, (n,T))
end

function plot_trajectories(pf,y,xt)
    xa = reinterpret(Float64, pf.s.x, (length(pf.s.x[1]), length(pf.s.x)))
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
