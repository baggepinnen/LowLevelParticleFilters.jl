"""
    kde(x,w)
Weighted kernel density estimate of the data `x` ∈ ℜN with weights `w` ∈ ℜN
`xi, densityw, density = kde(x,w)`
The number of grid points is chosen automatically and will approximately be equal to N/3
The bandwidth of the Gaussian kernel is chosen based on Silverman's rule of thumb
returns both weighted and non-weighted densities `densityw, density`
"""
function kde(x,w::AbstractVector)
    @assert all(x->x>0, w) "All weights must be non-negative"
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
        if length(dp.args) > 1
            @series begin
                label --> "Weighted density"
                linecolor --> :blue
                xi,densityw
            end
        end
        @series begin
            label --> "Non-weighted density"
            linecolor --> :red
            xi,density
        end
    end
end

"""
    densityplot(x,[w])
Plot (weighted) particles densities
"""
densityplot

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
    pdata = pplot(x, w, y, yhat, a, t, pdata; kwargs...)
    pdata = pplot(pf, y, pdata; kwargs...)

To be called inside a particle filter, plots either particle density (`density=true`) or individual particles (`density=false`) \n
Will plot all the real states in `xindices` as well as the expected vs real measurements of `yindices`.
# Arguments:
- `x`: `Vector{Vector}(N)`. The states for each particle where `N` number of Particles
- `w`: `Vector(N)`. weight of each particle
- `y`: `Vector{Vector}(T)`. All true outputs. `T` is total number of time steps (will only use index `t`)
- `yhat`: `Vector{Vector}(N)` The expected output per particle. `N` number of Particles
- `a`, `Vector(N)`, reorderng of particles (e.g. `1:N`)
- `t`, Current time step
- `xreal`: `Vector{Vector}(T)`. All true states if available. `T` is total number of time steps (will only use index `t`)
- `xprev`: Same as `x`, but for previous time step, only used when `!density` to show states origins
- `pdata`: Persistant data for plotting. Set to `nothing` in first call and pdata on remaining \n
- `density = true` To only plot the particle trajectories, set (`leftonly=false`)\n
- `leftonly = true`: only plot the left column\n
- `xindices = 1:n_states`\n
- `yindices = 1:n_measurements`\n
Returns: `pdata`
"""
function pplot(pf, y, args...; kwargs...)
    s = state(pf)
    pplot(s.x, s.we, y, LowLevelParticleFilters.measurement(pf).(s.x, s.t[]), s.j, s.t[], args...; xprev=s.xprev, kwargs...)
end

function pplot(x, w, y, yhat, a, t, pdata; xreal=nothing, xprev=nothing,  density = true, leftonly = true, xindices = 1:length(x[1]), yindices = 1:length(y[1]), lowpass=0.9)


    (t == 1 || t % 35 == 0) &&  @printf "Time     Surviving    Effective nbr of particles\n--------------------------------------------------------------\n"

    T = length(y)
    N = length(x)
    D = length(x[1])
    x = reduce(hcat, x)
    y = reduce(hcat, y)
    yhat = reduce(hcat, yhat)

    cols = leftonly ? 1 : 2
    grd = (r,c) -> (r-1)*cols+c
    @printf("t: %5d %7.3f %9.1f\n", t, (N-length(setdiff(Set(1:N),Set(a))))/N, effective_particles(w))
    plotvals = [x;yhat]
    realVals = xreal === nothing ? [fill(Inf,D,T);y] : [reduce(hcat,xreal);y] # Inf turns plotting off
    if !density
        plotvalsOld = xprev === nothing ? [Inf*x;yhat] : [reduce(hcat, xprev);yhat]
    end

    plotindices = [xindices; size(x,1) .+ yindices]
    if pdata === nothing
        pdata = plot(layout=(length(plotindices),cols)), zeros(length(plotindices),2)
    end
    p, minmax = pdata
    dataMin = minimum(plotvals[plotindices,:])
    dataMax = maximum(plotvals[plotindices,:])
    minmax = [min.(minmax[:,1], dataMin)*lowpass .+ (1-lowpass)*dataMin max.(minmax[:,2], dataMax)*lowpass .+ (1-lowpass)*dataMax]

    for (i, pind) in enumerate(plotindices)
        #Plot the heatmap on the left plot
        if density
            heatboxplot!(plotvals[pind,:], t, w, ylims=minmax[i,:], subplot=grd(i,1), reuse=false)
            if !leftonly
                densityplot!( plotvals[pind,:], w , ylims = tuple(minmax[i,:]...), c=:blue, subplot=grd(i,2), reuse=true, legend=false)
            end
        else
            #Plot the line on the left plot
            plot!(repeat([t-0.5 t+0.5], N)', [plotvalsOld[pind,:] plotvalsOld[pind,:]]', legend=false, subplot=grd(i,1), l=(:black, 0.1))
            # plot!(repeat([t-1 t-0.5], N)', [plotvalsOld[pind,a] plotvals[pind,:]]', legend=false, subplot=grd(i,1), l=(:black, 0.1))
        end
        #Plot Real State Here
        scatter!( [t+0.5], [realVals[pind,t]], subplot=grd(i,1), legend=false, m=(:cyan,))
    end
    # gui(p)
    (p, minmax)
end


function commandplot(f)
    res = f(nothing)
    display(res[1])
    while true
        print("Waiting for command. q to Quit, s NN to skip NN steps:\n")
        line = readline()
        skip = 1
        if line[1] == 'q'
            return
        elseif occursin("s", line)
            ss = split(strip(line,'\n'))
            skip = parse(Int,ss[2])
        end
        foreach(1:skip) do i
            res = f(res)
        end
        display(res[1])
    end
end


"""
    commandplot(pf, u, y; kwargs...)

Produce a helpful plot. For customization options (`kwargs...`), see `?pplot`.
After each time step, a command from the user is requested.
- q: quit
- s n: step `n` steps
"""
function commandplot(pf, u, y; kwargs...)
    # pdata = nothing
    reset!(pf)
    pfp = pf isa AuxiliaryParticleFilter ? pf.pf : pf
    commandplot() do pdata
        pdata = pplot(pfp, y, pdata; kwargs...)
        t = index(pf)
        LowLevelParticleFilters.update!(pf,u[t],y[t])
        pdata
    end
end


"""
    debugplot(pf, u, y; runall=false, kwargs...)

Produce a helpful plot. For customization options (`kwargs...`), see `?pplot`.
- ` runall=false:` if true, runs all time steps befor displaying (faster), if false, displays the plot after each time step.

The generated plot becomes quite heavy. Initially, try limiting your input to 100 time steps to verify that it doesn't crash.
"""
function debugplot(pf, u, y; runall=false, kwargs...)
    pdata = nothing
    reset!(pf)
    pfp = pf isa AuxiliaryParticleFilter ? pf.pf : pf
    for i = 1:length(y)
        pdata = pplot(pfp, y, pdata; kwargs...)
        t = index(pf)
        LowLevelParticleFilters.update!(pf,u[t],y[t])
        runall || display(pdata[1])
    end
    display(pdata[1])
end



@userplot TrajectoryDensity
@recipe function f(p::TrajectoryDensity; nbinsy=30, xreal=nothing)
    pf,x,w,y = p.args[1:4]
    N,T = size(x)
    D = length(x[1])
    P = length(y[1])
    if sum(w) ≉ T
        w = exp.(w)
        w ./= sum(w, dims=1)
    end
    layout := D+P
    label := ""
    markercolor --> :cyan
    title --> reshape([["State $d" for d = 1:D];["Measurement $d" for d = 1:P]], 1, :)
    for d = 1:D
        subplot := d
        @series begin
            seriestype := :histogram2d
            bins --> (0.5:1:T+0.5,nbinsy)
            weights --> vec(w)
            repeat((1:T)' .-0.5,N)[:], vec(getindex.(x,d))
        end
        xreal === nothing || @series begin
            seriestype := :scatter
            1:T, getindex.(xreal,d)
        end
    end
    yhat = measurement(pf).(x,0)
    for d = 1:P
        subplot := d+D
        @series begin
            seriestype := :histogram2d
            bins --> (0.5:1:T+0.5,nbinsy)
            weights --> vec(w)
            repeat((1:T)' .-0.5,N)[:], vec(getindex.(yhat,d))
        end
        @series begin
            seriestype := :scatter
            1:T, getindex.(y,d)
        end
    end
end

@userplot DimensionDensity
@recipe function f(p::DimensionDensity; nbinsy=30, xreal=nothing)
    length(p.args) >=5 || throw(ArgumentError("Supply arguments: pf,x,w,y,dimension::Int"))
    pf,x,w,y,d = p.args[1:5]
    N,T = size(x)
    D = length(x[1])
    P = length(y[1])

    if sum(w) ≉ T
        w = exp.(w)
        w ./= sum(w, dims=1)
    end

    label := ""
    markercolor --> :cyan
    if d <= D
        @series begin
            seriestype := :histogram2d
            bins --> (0.5:1:T+0.5,nbinsy)
            weights --> vec(w)
            repeat((1:T)' .-0.5,N)[:], vec(getindex.(x,d))
        end
        xreal === nothing || @series begin
            seriestype := :scatter
            1:T, getindex.(xreal,d)
        end
    else
        d -= D
        yhat = measurement(pf).(x,0)
        @series begin
            seriestype := :histogram2d
            bins --> (0.5:1:T+0.5,nbinsy)
            weights --> vec(w)
            repeat((1:T)' .-0.5,N)[:], vec(getindex.(yhat,d))
        end
        @series begin
            seriestype := :scatter
            1:T, getindex.(y,d)
        end

    end

end

"""
    dimensiondensity(pf,x,we,y, dimension, nbinsy=30, xreal=nothing)
Same as trajectorydensity but only plots subplot `dimension`.
"""
dimensiondensity

"""
    trajectorydensity(pf,x,we,y, nbinsy=30, xreal=nothing)
Plots particle densities along trajectory. See the readme for an example.
"""
trajectorydensity
