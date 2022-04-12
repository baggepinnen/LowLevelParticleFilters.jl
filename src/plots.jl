using .Plots

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
function pplot(pf::AbstractFilter, u, y, p, args...; kwargs...)
    s = state(pf)
    t = s.t[]
    pplot(s.x, s.we, u, y, LowLevelParticleFilters.measurement(pf).(s.x, Ref(u[t]), Ref(p), t), s.j, t, args...; xprev=s.xprev, kwargs...) 
end

function pplot(x, w, u, y, yhat, a, t, pdata; xreal=nothing, xprev=nothing,  density = true, leftonly = true, xindices = 1:length(x[1]), yindices = 1:length(y[1]), lowpass=0.9)


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
    # dataMin = minimum(plotvals[plotindices,:])
    # dataMax = maximum(plotvals[plotindices,:])
    # minmax = [min.(minmax[:,1], dataMin)*lowpass .+ (1-lowpass)*dataMin max.(minmax[:,2], dataMax)*lowpass .+ (1-lowpass)*dataMax]

    for (i, pind) in enumerate(plotindices)
        #Plot the heatmap on the left plot
        if density
            heatboxplot!(plotvals[pind,:], t, w, subplot=grd(i,1), reuse=false) #ylims=minmax[i,:],
            if !leftonly
                densityplot!( plotvals[pind,:], w , c=:blue, subplot=grd(i,2), reuse=true, legend=false) #ylims = tuple(minmax[i,:]...)
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
            if length(line) <= 2
                skip = 1
            else
                ss = split(strip(line,'\n'))
                skip = parse(Int,ss[2])
            end
        end
        foreach(1:skip) do i
            res = f(res)
        end
        display(res[1])
    end
end


"""
    commandplot(pf, u, y, p=parameters(pf); kwargs...)

Produce a helpful plot. For customization options (`kwargs...`), see `?pplot`.
After each time step, a command from the user is requested.
- q: quit
- s n: step `n` steps
"""
function commandplot(pf, u, y, p=parameters(pf); kwargs...)
    # pdata = nothing
    reset!(pf)
    pfp = pf isa AuxiliaryParticleFilter ? pf.pf : pf
    commandplot() do pdata
        pdata = pplot(pfp, u, y, p, pdata; kwargs...)
        t = index(pf)
        LowLevelParticleFilters.update!(pf,u[t],y[t],p)
        pdata
    end
end


"""
    debugplot(pf, u, y, p=parameters(pf); runall=false, kwargs...)

Produce a helpful plot. For customization options (`kwargs...`), see `?pplot`.
- ` runall=false:` if true, runs all time steps befor displaying (faster), if false, displays the plot after each time step.

The generated plot becomes quite heavy. Initially, try limiting your input to 100 time steps to verify that it doesn't crash.
"""
function debugplot(pf, u, y, p=parameters(pf); runall=false, kwargs...)
    pdata = nothing
    reset!(pf)
    pfp = pf isa AuxiliaryParticleFilter ? pf.pf : pf
    for i = 1:length(y)
        pdata = pplot(pfp, u, y, p, pdata; kwargs...)
        t = index(pf)
        LowLevelParticleFilters.update!(pf,u[t],y[t], p)
        runall || display(pdata[1])
    end
    display(pdata[1])
end
