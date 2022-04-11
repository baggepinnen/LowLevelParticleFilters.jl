
import .MonteCarloMeasurements: Particles
# From simulation result to particles

vv2m(x) = copy(reduce(hcat, x)')
function simulate(pf, n, du, p, Npart::Int)
    sims = map(_->vv2m.(simulate(pf, n, du, p)), 1:Npart)
    ns = length(sims[1]) # number of time series, usually x,u,y
    ntuple(ns) do s
        xes = getindex.(sims,s)
        M = reduce((x,y)->cat(x,y,dims=3), xes)
        M = permutedims(M, (3,2,1))
        copy(dropdims(mapslices(Particles, M, dims=(1,2)), dims=2)')
    end
end

# From filter result to particles
function MonteCarloMeasurements.Particles(x::AbstractMatrix{<:AbstractVector},we=nothing) # Helper function
    xp = copy(vv2m([Particles(vv2m(c)) for c in eachcol(x)]))
    we === nothing && return xp
    ## Perform resampling so that all particles have the same weight
    choices = map(LowLevelParticleFilters.resample, eachcol(we))
    for j in 1:size(xp,2), i in 1:size(xp,1)
        xp[i,j].particles .= xp[i,j].particles[choices[j]]
    end
    xp
end
