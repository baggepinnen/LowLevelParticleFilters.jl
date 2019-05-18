# module ParticlePlot
#
# export kde, heatboxplot, densityplot, pplot, pploti
# using RecipesBase, StatsBase
"""
Weighted kernel density estimate of the data `x` ∈ ℜN with weights `w` ∈ ℜN
`xi, densityw, density = kde(x,w)`
The number of grid points is chosen automatically and will approximately be equal to N/3
The bandwidth of the Gaussian kernel is chosen based on Silverman's rule of thumb
returns both weighted and non-weighted densities `densityw, density`
"""
function kde(x,w::AbstractVector)
    @assert all(>=(0), w) "All weights must be non-negative"
    e   = StatsBase.histrange(x,ceil(Int,length(x)/3))
    nb  = length(e)-1
    np  = length(x)
    I   = sortperm(x)
    s   = (rand()/nb+0):1/nb:e[end]
    j   = zeros(Float64,nb)
    bo = 1
    ilast = 0
    for i = 1:np
        ii = I[i]
        for b = bo:nb
            if x[ii] <= e[b+1]
                j[b] += w[ii]
                bo = b
                ilast = i
                break
            end
        end
    end
    j[end] += sum(w[I[ilast+1:np]])
    @assert all(j .>= 0) "j"
    xi      = e[1:end-1] .+ 0.5*(e[2]-e[1])
    σ       = std(x)
    if σ == 0 # All mass on one point
        return x,ones(x),ones(x)
    end
    h        = 1.06σ*np^(-1/5)
    K(x)     = exp(-(x/h).^2/2) / √(2π)
    densityw = [1/h*sum(j.*K.(xi.-xi[i])) for i=1:nb] # This density is effectively normalized by nb due to sum(j) = 1
    density  = [1/(np*h)*sum(K.(x.-xi[i])) for i=1:nb]

    # @assert all(densityw .>= 0) "densityw"
    # @assert all(density  .>= 0) "density"

    return xi, densityw, density

end

@userplot DensityPlot
@recipe function f(dp::DensityPlot)
    x = dp.args[1]
    w = length(dp.args) > 1 ? dp.args[2] : ones(x)/length(x)
    xi, densityw, density = kde(x,w)
    if maximum(x)-minimum(x) > 0
        title --> "Kernel density estimate"
        seriestype := :path
        @series begin
            label --> "Weighted density"
            linecolor --> :blue
            xi,densityw
        end
        @series begin
            label --> "Non-weighted density"
            linecolor --> :red
            xi,density
        end
    end
end

@userplot HeatboxPlot
@recipe function f(p::HeatboxPlot; nbinsy=30)
    x,t = p.args[1:2]
    seriestype := :histogram2d
    nbins --> (1,nbinsy)
    if length(p.args) >= 3
        weights --> p.args[3]
    end
    fill(t, length(x)), x
end


"""
    pplot(x, w, y, yhat, N, a, t, pdata)
    pplot(pf, y, pdata)
To be called inside a particle filter, plots either particle density (`density=true`) or individual particles (`density=false`) \n
Will plot all the real states in `xIndices` as well as the expected vs real measurements of `yIndices`.
Arguments: \n
* `x`: `Array(N)`. The states for each particle where `M` number of states, `N` number of Particles
* `w`: `Array(N)`. weight of each particle
* `y`: `Array(T)`. All true outputs. `R` is number of outputs, `T` is total number of time steps (will only use index `t`)
* `yhat`: `Array(N)` The expected output per particle. `R` is number of outputs, `N` number of Particles
* `N`, Number of particles
* `a`, `Array(N)`, reorderng of particles (e.g. `1:N`)
* `t`, Current time step
* `xreal`: `Array(T)`. All true states. `R` is number of states, `T` is total number of time steps (will only use index `t`)
* `xhat`: Not used
* `xOld`: Same as `x`, but for previous time step, only used when `!density` to show states origins
* `pdata`: Persistant data for plotting. Set to `nothing` in first call and pdataOut on remaining \n
* `density = true` To only plot the particle trajectories, set (`leftonly=false`)\n
* `leftonly = false`\n
* `xIndices = 1:size(x,1)`\n
* `yIndices = 1:size(y,1)`\n
Returns: `pdataOut`
"""
function pplot(pf, y, args...; kwargs...)
    s = state(pf)
    pplot(s.x, s.we, y, LowLevelParticleFilters.measurement(pf).(s.x, s.t[]), s.j, s.t[], args...; kwargs...)
end
function pplot(x, w, y, yhat, a, t, pdata; xreal=nothing, xhat=nothing, xOld=nothing,  density = true, leftonly = false, xIndices = 1:length(x[1]), yIndices = 1:length(y[1]), slidef=0.9)

    T = length(y)
    N = length(x)
    D = length(x[1])
    x = reduce(hcat, x)
    y = reduce(hcat, y)
    yhat = reduce(hcat, yhat)

    cols = leftonly ? 1 : 2
    grd = (r,c) -> (r-1)*cols+c
    println("Surviving: "*string((N-length(setdiff(Set(1:N),Set(a))))/N))
    plotvals = [x;yhat]
    realVals = xreal === nothing ? [fill(Inf,D,T);y] : [reduce(hcat,xreal);y] # Inf turns plotting off
    if !density
        plotvalsOld = xOld === nothing ? [Inf*x;yhat] : [reduce(hcat, xOld);yhat]
    end

    plotindices = [xIndices; size(x,1) .+ yIndices]
    if pdata === nothing
        pdata = plot(layout=(length(plotindices),cols)), zeros(length(plotindices),2)
    end
    p, minmax = pdata
    dataMin = minimum(plotvals[plotindices,:])
    dataMax = maximum(plotvals[plotindices,:])
    minmax = [min.(minmax[:,1], dataMin)*slidef .+ (1-slidef)*dataMin max.(minmax[:,2], dataMax)*slidef .+ (1-slidef)*dataMax]

    for (i, pind) in enumerate(plotindices)
        #Plot the heatmap on the left plot
        if density
            heatboxplot!(plotvals[pind,:], t, w, ylims=minmax[i,:], subplot=grd(i,1), reuse=false)
            if !leftonly
                densityplot!( plotvals[pind,:], w , ylims = tuple(minmax[i,:]...), c=:blue, subplot=grd(i,2), reuse=true, legend=false)
            end
        else
            #Plot the line on the left plot
            plot!(repeat([t-1.5 t-1], N)', [plotvalsOld[pind,:] plotvalsOld[pind,:]]', legend=false, subplot=grd(i,1))
            plot!(repeat([t-1 t-0.5], N)', [plotvalsOld[pind,a] plotvals[pind,:]]', legend=false, subplot=grd(i,1))
        end
        #Plot Real State Here
        scatter!( [t+0.5], [realVals[pind,t]], subplot=grd(i,1), legend=false, m=(:cyan,))
    end
    # gui(p)
    (p, minmax)
end


function commandplot(f)
    res = f(nothing)
    while true
        print("Waiting for command. q to Quit, ^D to run all, s NN to skip NN steps:\n")
        line = readline(STDIN)
        if line == "q\n"
            return
        elseif contains(line, "s")
            ss = split(strip(line,'\n'))
            skip = parse(Int,ss[2])
            foreach(1:skip) do i
                res = f(res)
            end
        end
    end
end
# end # module
