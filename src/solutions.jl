abstract type AbstractFilteringSolution end

"""
    KalmanFilteringSolution{Tx,Txt,TR,TRt,Tll} <: AbstractFilteringSolution

# Fields
- `x`: predictions ``x(t+1|t)`` (plotted if `plotx=true`)
- `xt`: filtered estimates ``x(t|t)`` (plotted if `plotxt=true`)
- `R`: predicted covariance matrices ``R(t+1|t)`` (plotted if `plotR=true`)
- `Rt`: filter covariances ``R(t|t)`` (plotted if `plotRt=true`)
- `ll`: loglikelihood
- `e`: prediction errors ``e(t|t-1) = y - ŷ(t|t-1)`` (plotted if `plote=true`)

# Plot
The solution object can be plotted
```
plot(sol, plotx=true, plotxt=true, plotR=true, plotRt=true, plote=true, plotu=true, ploty=true, plotyh=true, plotyht=true, name="")
```
where
- `plotx`: Plot the predictions `x(t|t-1)`
- `plotxt`: Plot the filtered estimates `x(t|t)`
- `plotR`: Plot the predicted covariances `R(t|t-1)` as ribbons at ±2σ (1.96 σ to be precise)
- `plotRt`: Plot the filter covariances `R(t|t)` as ribbons at ±2σ (1.96 σ to be precise)
- `plote`: Plot the prediction errors `e(t|t-1) = y - ŷ(t|t-1)`
- `plotu`: Plot the input
- `ploty`: Plot the measurements
- `plotyh`: Plot the predicted measurements `ŷ(t|t-1)`
- `plotyht`: Plot the filtered measurements `ŷ(t|t)`
- `name`: a string that is prepended to the labels of the plots, which is useful when plotting multiple solutions in the same plot.

To modify the signal names used in legend entries, construct an instance of [`SignalNames`](@ref) and pass this to the filter (or directly to the plot command) using the `names` keyword argument.
"""
struct KalmanFilteringSolution{F,Tu,Ty,Tx,Txt,TR,TRt,Tll,Te,Et} <: AbstractFilteringSolution
    f::F
    u::Tu
    y::Ty
    x::Tx
    xt::Txt
    R::TR
    Rt::TRt
    ll::Tll
    e::Te
    extra::Et
end

KalmanFilteringSolution(f,u,y,x,xt,R,Rt,ll,e) = KalmanFilteringSolution(f,u,y,x,xt,R,Rt,ll,e,nothing)

function Base.show(io::IO, sol::KalmanFilteringSolution)
    println(io, "KalmanFilteringSolution:")
    println(io, "  Filter: ", sol.f.names.name, " ", typeof(sol.f))
    l = length(sol.x)
    println(io, "  length: ", l)
    if l >= 1
        println(io, "  nx: ", length(sol.x[1]))
    end
    println(io, "  ll: ", sol.ll)
end

@recipe function plot(timevec::AbstractVector{<:Real}, sol::KalmanFilteringSolution; plotx = true, plotxt=true, plotu=true, ploty=true, plotyh=true, plotyht=false, plote=false, plotR=false, plotRt=false, names = sol.f.names, name = names.name)
    isempty(name) || (name = name*" ")
    kf = sol.f
    nx, nu, ny = length(sol.x[1]), length(sol.u[1]), length(sol.y[1])
    layout --> nx*(plotx || plotxt) + plotu*nu + (ploty || plotyh || plotyht || plote)*ny
    xnames = names.x
    if plotx
        m = reduce(hcat, sol.x)'
        twoσ = 1.96 .* sqrt.(reduce(hcat, diag.(sol.R))')
        for i = 1:nx
            @series begin
                label --> "$(name)$(xnames[i])(t|t-1)"
                subplot --> i
                if plotR
                    ribbon := twoσ[:,i]
                end
                timevec, m[:,i]
            end
        end
    end
    if plotxt
        m = reduce(hcat, sol.xt)'
        twoσ = 1.96 .* sqrt.(reduce(hcat, diag.(sol.Rt))')
        for i = 1:nx
            @series begin
                label --> "$(name)$(xnames[i])(t|t)"
                subplot --> i
                if plotRt
                    ribbon := twoσ[:,i]
                end
                timevec, m[:,i]
            end
        end
    end
    if plotu && nu > 0
        series = reduce(hcat, sol.u)'
        unames = names.u
        for i = 1:nu
            @series begin
                label --> "$(unames[i])"
                subplot --> i + nx*(plotx || plotxt)
                timevec, series[:, i]
            end
        end
    end
    ynames = names.y
    if ploty
        series = reduce(hcat, sol.y)'
        for i = 1:ny
            @series begin
                label --> "$(ynames[i])"
                subplot --> i + (nx*(plotx || plotxt) + nu*plotu)
                timevec, series[:, i]
            end
        end
    end
    if plotyh
        series = reduce(hcat, measurement_oop(kf).(sol.x, sol.u, Ref(kf.p), timevec))'
        for i = 1:ny
            @series begin
                label -->"$(name)ŷ$(i)(t|t-1)" 
                subplot --> i + (nx*(plotx || plotxt) + nu*plotu)
                linestyle --> :dash
                
                timevec, series[:, i]
            end
        end
    end
    if plotyht
        series = reduce(hcat, measurement_oop(kf).(sol.xt, sol.u, Ref(kf.p), timevec))'
        for i = 1:ny
            @series begin
                label --> "$(name)ŷ$(i)(t|t)"
                subplot --> i + (nx*(plotx || plotxt) + nu*plotu)
                linestyle --> :dash
                timevec, series[:, i]
            end
        end
    end
    if plote
        series = reduce(hcat, sol.e)'
        for i = 1:ny
            @series begin
                label --> "$(name)e$(i)(t|t-1)"
                subplot --> i + (nx*(plotx || plotxt) + nu*plotu)
                linestyle --> :dash
                timevec, series[:, i]
            end
        end
    end
end

@recipe function plot(sol::KalmanFilteringSolution)
    timevec = (0:length(sol.y)-1)*sol.f.Ts
    @series timevec, sol
end

"""
    ParticleFilteringSolution{F, Tu, Ty, Tx, Tw, Twe, Tll} <: AbstractFilteringSolution

# Fields:
- `f`: The filter used to produce the solution.
- `u`: Input
- `y`: Output / measurements
- `x`: Particles, the size of this array is `(N,T)`, where `N` is the number of particles and `T` is the number of time steps. Each element represents a weighted state hypothesis with weight given by `we`.
- `w`: Weights (log space). These are used for internal computations.
- `we`: Weights (exponentiated / original space). These are the ones to use to compute weighted means etc., they sum to one for each time step.
- `ll`: Log likelihood

# Plot
The solution object can be plotted
```
plot(sol; nbinsy=30, xreal=nothing, dim=nothing, ploty=true, q=nothing)
```

By default, a weighted 2D histogram is plotted, one for each state variable. If a vector of quantiles are provided in `q`, the quantiles are plotted instead of the histogram. If `xreal` is provided, the true state is plotted as a scatter plot on top of the histogram. If `dim` is provided, only the specified dimension is plotted. If `ploty` is true, the measurements are plotted as well.
"""
struct ParticleFilteringSolution{F,Tu,Ty,Tx,Tw,Twe,Tll} <: AbstractFilteringSolution
    f::F
    u::Tu
    y::Ty
    x::Tx
    w::Tw
    we::Twe
    ll::Tll
end

function td_getargs(sol::ParticleFilteringSolution, d::Int=1)
    (; f,x,w,u,y) = sol
    f,x,w,u,y,d
end

td_getargs(f,x,w,u,y,d::Int=1) = f,x,w,u,y,d

@recipe function plot(sol::ParticleFilteringSolution; nbinsy=30, xreal=nothing, dim=nothing, ploty=true, q=nothing)
    f = sol.f
    timevec = (0:size(sol.y,1)-1)*f.Ts
    timevecm05 = timevec .- 0.5f.Ts
    timevecp05 = timevec .+ 0.5f.Ts
    names = hasproperty(f, :names) ? f.names : default_names(f.nx, length(sol.u[1]), length(sol.y[1]))
    name = names.name
    isempty(name) || (name = name*" ")
    if dim === nothing || dim === (:)
        (; f,x,we,u,y) = sol
        p = parameters(f)
        N,T = size(x)
        D = length(x[1])
        P = length(y[1])
        if sum(we) ≉ T
            we ./= sum(we, dims=1)
        end

        w = vec(we)
        vx = vec(x)
        background_color --> :black

        layout --> D+P*ploty
        markercolor --> :cyan
        title --> reshape([["$(name)$(names.x[d])" for d = 1:D];["$(names.y[d])" for d = 1:P]], 1, :)
        if q isa Number
            q = [q]
        end
        if q isa AbstractArray
            Qs = map(q) do q
                weighted_quantile(sol, q)
            end
        end

        for d = 1:D
            subplot := d
            if q isa AbstractArray
                for qi in eachindex(Qs)
                    @series begin
                        label --> "q = $(q[qi])"
                        timevec, getindex.(Qs[qi], d)
                    end
                end
            else
                @series begin
                    seriestype := :histogram2d
                    bins --> (timevecp05,nbinsy)
                    weights --> w
                    repeat(timevecm05',N)[:], vec(getindex.(vx,d))
                end
            end
            xreal === nothing || @series begin
                label --> ""
                seriestype := :scatter
                timevec, (xreal isa AbstractVector{<:AbstractArray} ? getindex.(xreal,d) : xreal[:, d]) # Handle both vec of vec and matrix
            end
        end
        if ploty
            yhat = measurement(f).(x, permutedims(u), Ref(p), (timevec)') |> vec
            for d = 1:P
                subplot := d+D
                @series begin
                    label --> ""
                    seriestype := :histogram2d
                    bins --> (timevecp05,nbinsy)
                    weights --> w
                    repeat(timevecm05',N)[:], vec(getindex.(yhat,d))
                end
                @series begin
                    label --> ""
                    seriestype := :scatter
                    timevec, getindex.(y,d)
                end
            end
        end
    else
        (; f,x,w,u,y) = sol
        d = dim
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
                bins --> (timevecp05,nbinsy)
                weights --> vec(w)
                repeat(timevecm05',N)[:], vec(getindex.(x,d))
            end
            xreal === nothing || @series begin
                seriestype := :scatter
                timevec, (xreal isa AbstractVector{<:AbstractArray} ? getindex.(xreal,d) : xreal[:, d]) # Handle both vec of vec and matrix
            end
        else
            d -= D
            yhat = measurement(f).(x,u',0) |> vec
            @series begin
                seriestype := :histogram2d
                bins --> (timevecp05,nbinsy)
                weights --> vec(w)
                repeat(timevecm05',N)[:], vec(getindex.(yhat,d))
            end
            @series begin
                seriestype := :scatter
                timevec, getindex.(y,d)
            end
    
        end
    end
end

# function Base.getproperty(sol::AbstractFilteringSolution, s::Symbol)
#     s ∈ fieldnames(typeof(sol)) && return getfield(sol, s)
#     if s === :retcode
#         return :success
#     elseif s === :interp
#         return nothing
#     elseif s === :dense
#         return false
#     elseif s === :prob
#         return sol
#     elseif s === :t
#         return 1:length(getfield(sol, :x))
#     elseif s === :tslocation
#         return 0
#     else
#         throw(ArgumentError("$(typeof(sol)) has no property named $s"))
#     end
# end

# SciMLBase.has_syms(pf::AbstractFilter) = false