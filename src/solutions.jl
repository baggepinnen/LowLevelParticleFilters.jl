abstract type AbstractFilteringSolution end
abstract type AbstractKalmanFilteringSolution <: AbstractFilteringSolution end

Base.length(sol::AbstractFilteringSolution) = length(getfield(sol, :u))

function Base.getproperty(sol::AbstractFilteringSolution, s::Symbol)
    s ∈ fieldnames(typeof(sol)) && return getfield(sol, s)
    throw(ArgumentError("$(typeof(sol)) has no property named $s"))
end

Base.propertynames(sol::AbstractFilteringSolution) = fieldnames(typeof(sol))


"""
    KalmanFilteringSolution <: AbstractKalmanFilteringSolution

# Fields
- `x`: predictions ``x(t+1|t)`` (plotted if `plotx=true`)
- `xt`: filtered estimates ``x(t|t)`` (plotted if `plotxt=true`)
- `R`: predicted covariance matrices ``R(t+1|t)`` (plotted if `plotR=true`)
- `Rt`: filter covariances ``R(t|t)`` (plotted if `plotRt=true`)
- `ll`: loglikelihood
- `e`: prediction errors ``e(t|t-1) = y - ŷ(t|t-1)`` (plotted if `plote=true`)
- `K`: Kalman gain
- `S`: Cholesky factorization of innovation covariance

# Plot
The solution object can be plotted
```
plot(sol, plotx=true, plotxt=true, plotR=true, plotRt=true, plote=true, plotu=true, ploty=true, plotyh=true, plotyht=true, plotSt=false, name="")
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
- `plotS`: Plot the innovation covariances `S(t|t-1)` as ribbons at ±2σ on predicted measurements `ŷ(t|t-1)` (requires `plotyh=true`)
- `plotSt`: Plot the filtered output covariances `St = C*Rt*C'` as ribbons at ±2σ on filtered measurements `ŷ(t|t)` (requires `plotyht=true`, not supported for UnscentedKalmanFilter)
- `name`: a string that is prepended to the labels of the plots, which is useful when plotting multiple solutions in the same plot.
- `σ = 1.96` The number of standard deviations covered by covariance ribbons

To modify the signal names used in legend entries, construct an instance of [`SignalNames`](@ref) and pass this to the filter (or directly to the plot command) using the `names` keyword argument.
"""
struct KalmanFilteringSolution{F,Tu,Ty,Tx,Txt,TR,TRt,Tll,Te,TK,TS,Et,Tt} <: AbstractKalmanFilteringSolution
    f::F
    u::Tu
    y::Ty
    x::Tx
    xt::Txt
    R::TR
    Rt::TRt
    ll::Tll
    e::Te
    K::TK
    S::TS
    extra::Et
    t::Tt
end

KalmanFilteringSolution(f,u,y,x,xt,R,Rt,ll,e,K,S) = KalmanFilteringSolution(f,u,y,x,xt,R,Rt,ll,e,K,S,nothing)
KalmanFilteringSolution(f,u,y,x,xt,R,Rt,ll,e,K,S,extra) = KalmanFilteringSolution(f,u,y,x,xt,R,Rt,ll,e,K,S,extra,range(0, step=f.Ts, length=length(x)))

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

cov_diag(R::AbstractMatrix) = diag(R)
cov_diag(U::UpperTriangular) = diag(U'U)
cov_diag(Sᵪ::Cholesky) = diag(Sᵪ.U'Sᵪ.U)

@recipe function plot(timevec::AbstractVector{<:Real}, sol::KalmanFilteringSolution; plotx = true, plotxt=true, plotu=true, ploty=true, plotyh=true, plotyht=false, plote=false, plotR=false, plotRt=false, plotS=false, plotSt=false, names = sol.f.names, name = names.name, σ=1.96, always_include_x=false)
    isempty(name) || (name = name*" ")
    kf = sol.f

    kf isa UnscentedKalmanFilter && plotSt && error("Output covariance plotting (plotSt) is not yet supported for UnscentedKalmanFilter")

    nx, nu, ny = length(sol.x[1]), length(sol.u[1]), length(sol.y[1])
    lay = nx*(plotx || plotxt || always_include_x) + plotu*nu + (ploty || plotyh || plotyht || plote)*ny
    layout --> lay
    xnames = names.x
    if plotx
        m = reduce(hcat, sol.x)'
        twoσ = σ .* sqrt.(reduce(hcat, cov_diag.(sol.R))')
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
        twoσ = σ .* sqrt.(reduce(hcat, cov_diag.(sol.Rt))')
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
                subplot --> i + nx*(plotx || plotxt || always_include_x)
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
                subplot --> i + (nx*(plotx || plotxt || always_include_x) + nu*plotu)
                timevec, series[:, i]
            end
        end
    end
    if plotyh
        series = reduce(hcat, measurement_oop(kf).(sol.x, sol.u, Ref(kf.p), timevec))'
        if plotS && !isempty(sol.S) && !isnothing(sol.S[1])
            twoσ = σ .* sqrt.(reduce(hcat, cov_diag.(sol.S))')
        end
        for i = 1:ny
            @series begin
                label -->"$(name)ŷ$(i)(t|t-1)" 
                subplot --> i + (nx*(plotx || plotxt || always_include_x) + nu*plotu)
                linestyle --> :dash
                if plotS && !isempty(sol.S) && !isnothing(sol.S[1])
                    ribbon := twoσ[:,i]
                end
                timevec, series[:, i]
            end
        end
    end
    if plotyht
        series = reduce(hcat, measurement_oop(kf).(sol.xt, sol.u, Ref(kf.p), timevec))'
        if plotSt
            # Compute filtered output covariances: St = C * Rt * C'
            twoσ0 = map(1:length(sol.xt)) do i
                C = get_C(kf, sol.xt[i], sol.u[i], kf.p, timevec[i])
                R = eltype(sol.Rt) <: UpperTriangular ? sol.Rt[i]'sol.Rt[i] : sol.Rt[i]
                St = C * R * C'
                σ .* sqrt.(diag(St))
            end
            twoσ = reduce(hcat, twoσ0)'
        end
        for i = 1:ny
            @series begin
                label --> "$(name)ŷ$(i)(t|t)"
                subplot --> i + (nx*(plotx || plotxt || always_include_x) + nu*plotu)
                linestyle --> :dash
                if plotSt
                    ribbon := twoσ[:,i]
                end
                timevec, series[:, i]
            end
        end
    end
    if plote
        series = reduce(hcat, sol.e)'
        for i = 1:ny
            @series begin
                label --> "$(name)e$(i)(t|t-1)"
                subplot --> i + (nx*(plotx || plotxt || always_include_x) + nu*plotu)
                linestyle --> :dash
                timevec, series[:, i]
            end
        end
    end
end

@recipe function plot(sol::KalmanFilteringSolution)
    @series sol.t, sol
end


"""
    struct KalmanSmoothingSolution

A structure representing the solution to a Kalman smoothing problem.

# Fields
- `sol`: A solution object containing the results of the _filtering_ process.
- `xT`: The smoothed state estimate.
- `RT`: The smoothed state covariance.

The solution object can be plotted
```
plot(sol; plotxT=true, plotRT=true, plotyhT=false, plotST=false, kwargs...)
```
where
- `plotxT`: Plot the smoothed estimates `x(t|T)`
- `plotRT`: Plot the smoothed covariances `R(t|T)` as ribbons at ±2σ (1.96 σ to be precise)
- `plotyhT`: Plot the smoothed output estimates `ŷ(t|T) = C*x(t|T)`
- `plotST`: Plot the smoothed output covariances `ST = C*RT*C'` as ribbons at ±2σ on smoothed measurements `ŷ(t|T)` (requires `plotyhT=true`, not supported for UnscentedKalmanFilter)
- The rest of the keyword arguments are the same as for [`KalmanFilteringSolution`](@ref)

When plotting a smoothing solution, the filtering solution is also plotted. The same keyword arguments as for [`KalmanFilteringSolution`](@ref) may be used to control which signals are plotted
"""
struct KalmanSmoothingSolution <: AbstractKalmanFilteringSolution
    sol
    xT
    RT
end

function Base.getproperty(sol::KalmanSmoothingSolution, s::Symbol)
    s ∈ fieldnames(typeof(sol)) && return getfield(sol, s)
    return getproperty(getfield(sol, :sol), s)
end

Base.iterate(r::KalmanSmoothingSolution)               = (r.xT, Val(:RT))
Base.iterate(r::KalmanSmoothingSolution, ::Val{:RT})   = (r.RT, Val(:ll))
Base.iterate(r::KalmanSmoothingSolution, ::Val{:ll})   = (r.x, Val(:done))
Base.iterate(r::KalmanSmoothingSolution, ::Val{:done}) = nothing


@recipe function plot(timevec::AbstractVector{<:Real}, sol::KalmanSmoothingSolution; plotx = true, plotxt=true, plotu=true, ploty=true, plotyh=true, plotyht=false, plote=false, plotxT = true, plotRT=true, plotyhT=false, plotST=false, names = sol.f.names, name = names.name, σ = 1.96)
    isempty(name) || (name = name*" ")
    kf = sol.f

    kf isa UnscentedKalmanFilter && plotST && error("Output covariance plotting (plotST) is not supported for UnscentedKalmanFilter")

    nx, nu, ny = length(sol.x[1]), length(sol.u[1]), length(sol.y[1])
    xnames = names.x

    # The mess of replicating all plotx kwargs in this recipe is due to an obscure bug in Plots that causes the layout that is set in the lower KalmanFilteringSolution recipe to only take effect if anything is actually drawn in the recipe. When the user wants to plot only xT, the lower level recipe only sets the layout but plots nothing, and then we get an indexing error here due to there not being any layout > 1 subplot set.
    lay = nx*(plotx || plotxt || plotxT) + plotu*nu + (ploty || plotyh || plotyht || plote || plotyhT)*ny
    layout --> lay
    @series begin
        # This is unfortunately also required
        always_include_x := plotxT
        plotx := plotx
        plotxt := plotxt
        plotu := plotu
        ploty := ploty
        plotyh := plotyh
        plotyht := plotyht
        plote := plote
        sol.sol
    end

    if plotxT
        m = reduce(hcat, sol.xT)'
        if plotRT
            twoσ = σ .* sqrt.(reduce(hcat, cov_diag.(sol.RT))')
        end
        for i = 1:nx
            @series begin
                label --> "$(name)$(xnames[i])(t|T)"
                subplot --> i
                if plotRT
                    ribbon := twoσ[:,i]
                end
                timevec, m[:,i]
            end
        end
    end
    if plotyhT
        ynames = names.y
        series = reduce(hcat, measurement_oop(kf).(sol.xT, sol.u, Ref(kf.p), timevec))'
        if plotST
            # Compute smoothed output covariances: ST = C * RT * C'
            twoσ0 = map(1:length(sol.xT)) do i
                C = get_C(kf, sol.xT[i], sol.u[i], kf.p, timevec[i])
                ST = C * sol.RT[i] * C'
                σ .* sqrt.(diag(ST))
            end
            twoσ = reduce(hcat, twoσ0)'
        end
        for i = 1:ny
            @series begin
                label --> "$(name)ŷ$(i)(t|T)"
                subplot --> i + (nx*(plotx || plotxt || plotxT) + nu*plotu)
                linestyle --> :dot
                if plotST
                    ribbon := twoσ[:,i]
                end
                timevec, series[:, i]
            end
        end
    end
end


@recipe function plot(sol::KalmanSmoothingSolution)
    @series sol.t, sol
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
struct ParticleFilteringSolution{F,Tu,Ty,Tx,Tw,Twe,Tll,Tt} <: AbstractFilteringSolution
    f::F
    u::Tu
    y::Ty
    x::Tx
    w::Tw
    we::Twe
    ll::Tll
    t::Tt
end

ParticleFilteringSolution(f,u,y,x,w,we,ll) = ParticleFilteringSolution(f,u,y,x,w,we,ll,range(0, step=f.Ts, length=size(x,2)))

function td_getargs(sol::ParticleFilteringSolution, d::Int=1)
    (; f,x,w,u,y) = sol
    f,x,w,u,y,d
end

td_getargs(f,x,w,u,y,d::Int=1) = f,x,w,u,y,d

@recipe function plot(sol::ParticleFilteringSolution; nbinsy=30, xreal=nothing, dim=nothing, ploty=true, q=nothing)
    f = sol.f
    timevec = sol.t
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
                eltype(vx) <: RBParticle && @warn "The estimated quantiles for the linear sub state does not take the covariance matrix of each particle into account. Interpret plot with caution."
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
                    xs = vec(getindex.(vx,d))
                    if eltype(vx) <: RBParticle && d > length(vx[1].xn) 
                        # @show Particles(10, permute=false).particles
                        systematic_normal_sample = [-1.6448536269514729, -1.0364333894937896, -0.6744897501960818, -0.3853204664075677, -0.12566134685507402, 0.12566134685507416, 0.3853204664075677, 0.6744897501960818, 1.0364333894937896, 1.6448536269514717]
                        nxn = length(vx[1].xn)
                        # In this case we sample from the Gaussian distribution of the particle as well
                        Nsamples = length(systematic_normal_sample)
                        perturbations = reduce(hcat, [sqrt(x.R[d-nxn,d-nxn])*systematic_normal_sample for x in vx])'
                        xs = vec((xs .+ perturbations)')
                        ws = vec(repeat(w, 1, Nsamples)')

                    else
                        ws = w
                        Nsamples = 1
                    end
                    weights --> ws

                    repeat(timevecm05, inner=N*Nsamples)[:], xs
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