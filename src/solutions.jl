abstract type AbstractFilteringSolution end

"""
    KalmanFilteringSolution{Tx,Txt,TR,TRt,Tll} <: AbstractFilteringSolution

# Fields
- `x`: predictions ``x(t+1|t)``
- `xt`: filtered estimates ``x(t|t)``
- `R`: predicted covariance matrices ``R(t+1|t)``
- `Rt`: filter covariances ``R(t|t)``
- `ll`: loglikelihood
- `e`: prediction errors

# Plot
The solution object can be plotted
```
plot(sol, plotx=true, plotxt=true, plotu=true, ploty=true, plotyh=true, plotyht=false, name="")
```
where `plotx`, `plotxt`, `plotu`, `ploty`, `plotyh`, `plotyht` are booleans that control which plots are shown. `name` is a string that is prepended to the labels of the plots, which is useful when plotting multiple solutions in the same plot.
"""
struct KalmanFilteringSolution{F,Tu,Ty,Tx,Txt,TR,TRt,Tll,Te} <: AbstractFilteringSolution
    f::F
    u::Tu
    y::Ty
    x::Tx
    xt::Txt
    R::TR
    Rt::TRt
    ll::Tll
    e::Te
end

@recipe function plot(timevec::AbstractVector{<:Real}, sol::KalmanFilteringSolution; plotx = true, plotxt=true, plotu=true, ploty=true, plotyh=true, plotyht=false, plote=false, name = "")
    isempty(name) || (name = name*" ")
    kf = sol.f
    nx, nu, ny = length(sol.x[1]), length(sol.u[1]), length(sol.y[1])
    layout --> nx*(plotx || plotxt) + plotu*nu + (ploty || plotyh || plotyht || plote)*ny
    plotx && @series begin
        label --> ["$(name)x$(i)(t|t-1)" for i in 1:nx] |> permutedims
        subplot --> (1:nx)'
        timevec, reduce(hcat, sol.x)'
    end
    plotxt && @series begin
        label --> ["$(name)x$(i)(t|t)" for i in 1:nx] |> permutedims
        subplot --> (1:nx)'
        timevec, reduce(hcat, sol.xt)'
    end
    plotu && nu > 0 && @series begin
        label --> ["u$(i)" for i in 1:nu] |> permutedims
        subplot --> (1:nu)' .+ nx*(plotx || plotxt)
        timevec, reduce(hcat, sol.u)'
    end
    ploty && @series begin
        label --> ["y$(i)" for i in 1:ny] |> permutedims
        subplot --> (1:ny)' .+ (nx*(plotx || plotxt) + nu*plotu)
        timevec, reduce(hcat, sol.y)'
    end
    plotyh && @series begin
        label --> ["$(name)ŷ$(i)(t|t-1)" for i in 1:ny] |> permutedims
        subplot --> (1:ny)' .+ (nx*(plotx || plotxt) + nu*plotu)
        linestyle --> :dash
        yh = measurement_oop(kf).(sol.x, sol.u, Ref(kf.p), timevec)
        timevec, reduce(hcat, yh)'
    end
    plotyht && @series begin
        label --> ["$(name)ŷ$(i)(t|t)" for i in 1:ny] |> permutedims
        subplot --> (1:ny)' .+ (nx*(plotx || plotxt) + nu*plotu)
        linestyle --> :dash
        yht = measurement_oop(kf).(sol.xt, sol.u, Ref(kf.p), timevec)
        timevec, reduce(hcat, yht)'
    end
    plote && @series begin
        label --> ["$(name)e$(i)(t|t-1)" for i in 1:ny] |> permutedims
        subplot --> (1:ny)' .+ (nx*(plotx || plotxt) + nu*plotu)
        linestyle --> :dash
        timevec, reduce(hcat, sol.e)'
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
plot(sol; nbinsy=30, xreal=nothing, dim=nothing, ploty=true)
```
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

@recipe function plot(sol::ParticleFilteringSolution; nbinsy=30, xreal=nothing, dim=nothing, ploty=true)
    timevec = (0:size(sol.y,1)-1)*sol.f.Ts
    timevecm05 = timevec .- 0.5sol.f.Ts
    timevecp05 = timevec .+ 0.5sol.f.Ts
    if dim === nothing || dim === (:)
        (; f,x,w,u,y) = sol
        p = parameters(f)
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

        layout := D+P*ploty
        label := ""
        markercolor --> :cyan
        title --> reshape([["State $d" for d = 1:D];["Measurement $d" for d = 1:P]], 1, :)
        for d = 1:D
            subplot := d
            @series begin
                seriestype := :histogram2d
                bins --> (timevecp05,nbinsy)
                weights --> w
                repeat(timevecm05',N)[:], vec(getindex.(vx,d))
            end
            xreal === nothing || @series begin
                seriestype := :scatter
                timevec, (xreal isa AbstractVector{<:AbstractArray} ? getindex.(xreal,d) : xreal[:, d]) # Handle both vec of vec and matrix
            end
        end
        if ploty
            yhat = measurement(f).(x, permutedims(u), Ref(p), (timevec)') |> vec
            for d = 1:P
                subplot := d+D
                @series begin
                    seriestype := :histogram2d
                    bins --> (timevecp05,nbinsy)
                    weights --> w
                    repeat(timevecm05',N)[:], vec(getindex.(yhat,d))
                end
                @series begin
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