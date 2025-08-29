module LowLevelParticleFiltersPlotsExt
using LowLevelParticleFilters
using LowLevelParticleFilters: AbstractFilter, parameters, pplot
using Plots
using Printf
using Statistics: mean, cov
using LinearAlgebra: eigen, diagm, cholesky


function LowLevelParticleFilters.pplot(pf::AbstractFilter, u, y, p, args...; kwargs...)
    s = state(pf)
    t = s.t[]
    LowLevelParticleFilters.pplot(s.x, s.we, u, y, LowLevelParticleFilters.measurement(pf).(s.x, Ref(u[t]), Ref(p), t), s.j, t, args...; xprev=s.xprev, kwargs...) 
end

function LowLevelParticleFilters.pplot(x, w, u, y, yhat, a, t, pdata; xreal=nothing, xprev=nothing,  density = true, leftonly = true, xindices = 1:length(x[1]), yindices = 1:length(y[1]), lowpass=0.9)


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


function LowLevelParticleFilters.commandplot(f)
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



function LowLevelParticleFilters.commandplot(pf, u, y, p=parameters(pf); kwargs...)
    # pdata = nothing
    reset!(pf)
    pfp = pf isa AuxiliaryParticleFilter ? pf.pf : pf
    commandplot() do pdata
        pdata = LowLevelParticleFilters.pplot(pfp, u, y, p, pdata; kwargs...)
        t = index(pf)
        LowLevelParticleFilters.update!(pf,u[t],y[t],p)
        pdata
    end
end



function LowLevelParticleFilters.debugplot(pf, u, y, p=parameters(pf); runall=false, kwargs...)
    pdata = nothing
    reset!(pf)
    pfp = pf isa AuxiliaryParticleFilter ? pf.pf : pf
    for i = 1:length(y)
        pdata = LowLevelParticleFilters.pplot(pfp, u, y, p, pdata; kwargs...)
        t = index(pf)
        LowLevelParticleFilters.update!(pf,u[t],y[t], p)
        runall || display(pdata[1])
    end
    display(pdata[1])
end

function gaussian_chol(μ, Σ; n_std::Real, dims=1:2)
    # λ, U = eigen(Σ[dims, dims])
    # μ[dims], (n_std * U * diagm(.√λ))
    U = cholesky(Σ[dims, dims]).L
    μ[dims], (n_std * U)
end

import LowLevelParticleFilters: unscentedplot, unscentedplot!, covplot, covplot!
@userplot UnscentedPlot

@recipe function f(c::UnscentedPlot; n_std = 2, N = 100, inds=nothing, dims=nothing)
    if c.args[1] isa UnscentedKalmanFilter
        ukf = c.args[1]
        sigmapoints = ukf.predict_sigma_point_cache.x1
        pars = ukf.weight_params
    else
        sigmapoints = c.args[1]
        pars = length(c.args) >= 2 ? c.args[2] : TrivialParams()
    end
    if length(sigmapoints[1]) < 2
        error("1D sigma points are not supported")
    elseif length(sigmapoints[1]) == 2
        dims = 1:2
    else
        if dims === nothing
            @warn("UnscentedPlot only supports 2D inputs, plotting the first two dimensions. Select dimensions using dims=[dim1, dim2].")
            dims = 1:2
        end
    end
    μ = LowLevelParticleFilters.mean_with_weights(LowLevelParticleFilters.weighted_mean, sigmapoints, pars)
    S = LowLevelParticleFilters.cov_with_weights(LowLevelParticleFilters.weighted_cov, sigmapoints, μ, pars)
    μ, S = gaussian_chol(μ, S; n_std, dims)

    θ = range(0, 2π; length = N)
    A = S * [cos.(θ)'; sin.(θ)']

    xguide --> "x$(dims[1])"
    yguide --> "x$(dims[2])"
    @series begin
        seriestype --> :scatter
        markersize --> 2
        label --> "Sigma points"
        SP = reduce(hcat, sigmapoints)
        SP[dims[1], :], SP[dims[2], :]
    end
    @series begin
        primary := false
        seriesalpha --> 0.3
        Plots.Shape(μ[1] .+ A[1, :], μ[2] .+ A[2, :])
    end
    @series begin
        primary := false
        seriestype --> :scatter
        markershape --> :xcross
        markersize --> 5
        [μ[1]], [μ[2]]
    end

    
end

Plots.@userplot CovPlot

"""
    covplot(μ, R; n_std = 2, N = 100, dims=1:2)
    covplot(f;    n_std = 2, N = 100, dims=1:2)

Plot the covariance ellipse of the state `μ` and covariance `R`. `dims` indicate the two dimensions to plot, and defaults to the first two dimensions.

If a filter `f` is passed, the state and covariance are extracted from the filter.
"""
covplot
Plots.@recipe function f(c::CovPlot; n_std = 2, N = 100, dims=nothing, mean=false)
    if dims === nothing
        dims = 1:2
    end
    if c.args[1] isa AbstractFilter
        kf = c.args[1]
        μ, R = state(kf), covariance(kf)
        xguide --> kf.names.x[dims[1]]
        yguide --> kf.names.x[dims[2]]
    else
        xguide --> "$(dims[1])"
        yguide --> "$(dims[2])"
        μ, R = c.args[1:2]
    end
    μ, S = gaussian_chol(μ, R; n_std, dims)

    θ = range(0, 2π; length = N)
    A = S * [cos.(θ)'; sin.(θ)']


    Plots.@series begin
        seriesalpha --> 0.3
        Plots.Shape(μ[1] .+ A[1, :], μ[2] .+ A[2, :])
    end
    if mean
        Plots.@series begin
            primary := false
            seriestype --> :scatter
            markersize --> 5
            [μ[1]], [μ[2]]
        end
    end
end

end