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
    bins --> (1,nbinsy)
    if length(p.args) >= 3
        w = p.args[3]
        valid = (w .!= -Inf) .& (w .!= 0)
        weights --> w[valid]
        x = x[valid]
    end
    fill(t, length(x)), x
end


@userplot TrajectoryDensity
@recipe function f(p::TrajectoryDensity; nbinsy=30, xreal=nothing)
    pf,x,w,u,y = p.args[1:5]
    N,T = size(x)
    D = length(x[1])
    P = length(y[1])
    if sum(w) ≉ T
        w = exp.(w)
        w ./= sum(w, dims=1)
    end

    w = vec(w)
    vx = vec(x)
    background_color --> :black

    layout := D+P
    label := ""
    markercolor --> :cyan
    title --> reshape([["State $d" for d = 1:D];["Measurement $d" for d = 1:P]], 1, :)
    for d = 1:D
        subplot := d
        @series begin
            seriestype := :histogram2d
            bins --> (0.5:1:T+0.5,nbinsy)
            weights --> w
            repeat((1:T)' .-0.5,N)[:], vec(getindex.(vx,d))
        end
        xreal === nothing || @series begin
            seriestype := :scatter
            1:T, getindex.(xreal,d)
        end
    end
    yhat = measurement(pf).(x,u',0) |> vec
    for d = 1:P
        subplot := d+D
        @series begin
            seriestype := :histogram2d
            bins --> (0.5:1:T+0.5,nbinsy)
            weights --> w
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
    pf,x,w,u,y,d = p.args[1:6]
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
        yhat = measurement(pf).(x,u',0) |> vec
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
