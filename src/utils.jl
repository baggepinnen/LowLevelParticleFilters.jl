export weigthed_mean, resample, plot_trajectories, scatter_particles

function logsumexp!(w)
    offset = maximum(w)
    normConstant = zero(eltype(w))
    for i = eachindex(w)
        normConstant += exp(w[i]-offset)
    end
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

shouldresample(w) = rand() < 0.5

function resample(w)
    N = length(w)
    j = Array{Int64}(N)
    bins = Array{Float64}(N)
    bins[1] = exp(w[1])
    for i = 2:N
        bins[i] = bins[i-1] + exp(w[i])
    end
    s = (rand()/N):(1/N):bins[end]
    bo = 1
    for i = 1:N
        @inbounds for b = bo:N
            if s[i] < bins[b]
                j[i] = b
                bo = b
                break
            end
        end
    end
    return j
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
