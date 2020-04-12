
# From simulation result to particles
function simulate(pf, n, du, Npart)
    sims = map(_->vecvec2mat.(simulate(pf, n, du)), 1:Npart)
    ns = length(sims[1]) # number of time series, usually x,u,y
    series = map(1:ns) do s
        xes = getindex.(sims,s)
        M = reduce((x,y)->cat(x,y,dims=3), xes)
        M = permutedims(M, (3,2,1))
        copy(dropdims(mapslices(Particles, M, dims=(1,2)), dims=2)')
    end
    (series...,)
end

# From filter result to particles
function MonteCarloMeasurements.Particles(x::AbstractMatrix{<:AbstractVector},we=nothing) # Helper function
    xp = copy(reduce(hcat,[Particles(vecvec2mat(c)) for c in eachcol(x)])')
    we === nothing && return xp
    ## Perform resampling so that all particles have the same weight
    choices = map(LowLevelParticleFilters.resample, eachcol(we))
    for j in 1:size(xp,2), i in 1:size(xp,1)
        xp[i,j].particles .= xp[i,j].particles[choices[j]]
    end
    xp
end
