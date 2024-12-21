# Interacting multiple models

mutable struct IMM{MT, PT, XT, RT, μT, PAT, NX, NR} <: AbstractFilter
    models::MT
    P::PT
    x::XT
    R::RT
    μ::μT
    μ0::μT
    p::PAT
    new_x::NX
    new_R::NR
    interact::Bool
end


"""
    IMM(models, P, μ; check = true, p = NullParameters(), interact = true)

Interacting Multiple Model (IMM) filter. This filter is a combination of multiple Kalman-type filters, each with its own state and covariance. The IMM filter is a probabilistically weighted average of the states and covariances of the individual filters. The weights are determined by the probability matrix `P` and the mixing probabilities `μ`.

!!! warning "Experimental"
    This filter is currently considered experimental and the user interface may change in the future without respecting semantic versioning.

In addition to the [`predict!`](@ref) and [`correct!`](@ref) steps, the IMM filter has an [`interact!`](@ref) method that updates the states and covariances of the individual filters based on the mixing probabilities. The [`combine!`](@ref) method combines the states and covariances of the individual filters into a single state and covariance. These four functions are typically called in either of the orders
- `correct!, combine!, interact!, predict!` (as is done in [`update!`](@ref))
- `interact!, predict!, correct!, combine!` (as is done in the reference cited below)

These two orders are cyclic permutations of each other, and the order used in [`update!`](@ref) is chosen to align with the order used in the other filters, where the initial condition is corrected using the first measurement, i.e., we assume the first measurement updates ``x(0|-1)`` to ``x(0|0)``.

The initial (combined) state and covariance of the IMM filter is made up of the weighted average of the states and covariances of the individual filters. The weights are the initial mixing probabilities `μ`.

Ref: "Interacting multiple model methods in target tracking: a survey", E. Mazor; A. Averbuch; Y. Bar-Shalom; J. Dayan

# Arguments:
- `models`: An array of Kalman-type filters, such as [`KalmanFilter`](@ref), [`ExtendedKalmanFilter`](@ref), [`UnscentedKalmanFilter`](@ref), etc. The state of each model must have the same meaning, such that forming a weighted average makes sense.
- `P`: The mode-transition probability matrix. `P[i,j]` is the probability of transitioning from mode `i` to mode `j` (each row must sum to one).
- `μ`: The initial mixing probabilities. `μ[i]` is the probability of being in mode `i` at the initial contidion (must sum to one).
- `check`: If `true`, check that the inputs are valid. If `false`, skip the checks.
- `p`: Parameters for the filter. NOTE: this `p` is shared among all internal filters. The internal `p` of each filter will be overridden by this one.
- `interact`: If `true`, the filter will run the interaction as part of [`update!`](@ref) and [`forward_trajectory`](@ref). If `false`, the filter will not run the interaction step. This choice can be overridden by passing the keyword argument `interact` to the respective functions.
"""
function IMM(models, P::AbstractMatrix, μ::AbstractVector; check=true, p = NullParameters(), interact = true)
    if check
        N = length(models)
        length(μ) == N || throw(ArgumentError("μ must have the same length as the number of models"))
        LinearAlgebra.checksquare(P) == N || throw(ArgumentError("P must be square with side length same as the number of models"))
        sum(μ) ≈ 1.0 || throw(ArgumentError("μ must sum to 1.0"))
        all(x ≈ 1 for x in sum(P, dims=2)) || throw(ArgumentError("P must sum to 1.0 along rows"))
        allequal(typeof(m.x) for m in models) || @warn("The list of models have different type of their state vector x, this leads to poor performance. Turn off this warining by passing IMM(..., check=false)")
        allequal(typeof(m.R) for m in models) || @warn("The list of models have different type of their state vector x, this leads to poor performance. Turn off this warining by passing IMM(..., check=false)")
        allequal(m.Ts for m in models) || throw(ArgumentError("All models must have the same sampling time Ts"))
    end
    T = eltype(models[1].d0)
    μT = T.(μ)
    x = sum(i->μT[i]*models[i].x, eachindex(models))
    R = sum(i->μT[i]*models[i].R, eachindex(models))
    new_x = [copy(model.x) for model in models]
    new_R = [copy(model.R) for model in models]
    IMM(models, P, x, R, μT, copy(μT), p, new_x, new_R, interact)
end

function Base.getproperty(imm::IMM, s::Symbol)
    s ∈ fieldnames(typeof(imm)) && return getfield(imm, s)
    if s === :Ts
        return getfield(imm, :models)[1].Ts
    else
        throw(ArgumentError("$(typeof(imm)) has no property named $s"))
    end
end


"""
    interact!(imm::IMM)

The interaction step of the IMM filter updates the state and covariance of each internal model based on the mixing probabilities `imm.μ` and the transition probability matrix `imm.P`.

Models with small mixing probabilities will have their states and covariances updated more towards the states and covariances of models with higher mixing probabilities, and vice versa.
"""
function interact!(imm::IMM)
    (; μ, P, models, new_x, new_R) = imm
    @assert sum(μ) ≈ 1.0
    cj = P'μ
    for j in eachindex(models)
        if iszero(cj[j]) # Filter has died, we let it evolve on its own
            new_x[j] = models[j].x
            new_R[j] = models[j].R
            continue
        else
            @bangbang new_x[j] .= 0 .* new_x[j]
            @bangbang new_R[j] .= 0 .* new_R[j]
        end
        for i = eachindex(models)
            μij = calc_μij(P[i,j], μ[i], cj[j])
            @bangbang new_x[j] .+= μij .* models[i].x
        end
        for i = eachindex(models)
            μij = calc_μij(P[i,j], μ[i], cj[j])
            if !iszero(μij)
                d = models[i].x - new_x[j]
                @bangbang new_R[j] .+= symmetrize(μij .* (d * d' .+  models[i].R))
            end
        end
    end
    for (model, x, R) in zip(models, new_x, new_R)
        @bangbang model.x .= x
        @bangbang model.R .= R
    end

    nothing
end

function calc_μij(P, μ, cj)
    P*μ/cj # This overflows for Dual numbers when cj has extremely small values
end

function predict!(imm::IMM, args...; kwargs...)
    for (model,μ) in zip(imm.models, imm.μ)
        # iszero(μ) && continue # Filter has died
        predict!(model, args...; kwargs...)
    end
end



"""
    ll, lls, rest = correct!(imm::IMM, u, y, args; kwargs)

The correct step of the IMM filter corrects each model with the measurements `y` and control input `u`. The mixing probabilities `imm.μ` are updated based on the likelihood of each model given the measurements and the transition probability matrix `P`.

The returned tuple consists of the sum of the log-likelihood of all models, the vector of individual log-likelihoods and an array of the rest of the return values from the correct step of each model.
"""
function correct!(imm::IMM, u, y, args...; expnormalize = true, kwargs...)
    (; μ, P, models) = imm
    lls = zeros(eltype(imm.x), length(models))
    rest = []
    for (j, model) in enumerate(models)
        # if iszero(μ[j]) # Filter has died
        #     # QUESTION: we may want to keep updating the filter, maybe make it an option?
        #     lls[j] = eltype(imm.x)(-Inf)
        #     push!(rest, [])
        #     continue
        # end
        lls[j], others... = correct!(model, u, y, args...; kwargs...)
        push!(rest, others)
    end
    μP = P'μ

    # Naive formulas
    # new_μ = exp.(lls) .* μP
    # μ .= new_μ ./ sum(new_μ)

    # Rewrite exp.(lls) .* μP as exp.(lls .+ log.(μP)), after which we can identify lls .+ log.(μP) as w and μ as we from the particle-filter implementation. This may improve the numerics
    w = lls .+ log.(μP)
    ll = logsumexp!(w, μ)
    return ll, lls, rest
end

"""
    combine!(imm::IMM)

Combine the models of the IMM filter into a single state `imm.x` and covariance `imm.R`. This is done by taking a weighted average of the states and covariances of the individual models, where the weights are the mixing probabilities `μ`.
"""
function combine!(imm::IMM)
    (; μ, x, R, models) = imm
    @assert sum(μ) ≈ 1.0 "sum(μ) = $(sum(μ))"

    @bangbang x .= 0 .* x
    @bangbang R .= 0 .* R

    for (j, model) in enumerate(models)
        @bangbang x .+= μ[j] .* model.x
    end

    for (j, model) in enumerate(models)
        iszero(μ[j]) && continue
        d = model.x .- x
        @bangbang R .+= symmetrize(μ[j] .* (model.R .+ d * d'))
    end

    imm.x = x
    imm.R = R
    nothing
end


"""
    update!(imm::IMM, u, y, p, t; correct_kwargs = (;), predict_kwargs = (;), interact = true)

The combined udpate for an [`IMM`](@ref) filter performs the following steps:
1. Correct each model with the measurements `y` and control input `u`.
2. Combine the models into a single state and covariance.
3. Interact the models to update their respective state and covariance.
4. Predict each model to the next time step.

This differs slightly from the udpate step of other filters, where at the end of an update the state of the filter is the one-step ahead _predicted_ value, whereas here each individual filter has a predicted state, but the [`combine!`](@ref) step of the IMM filter hasn't been performed on the predictions yet. The state of the IMM filter is thus ``x(t|t)`` and not ``x(t+1|t)`` like it is for other filters, and each filter internal to the IMM.

# Arguments:
- `correct_kwargs`: An optional named tuple of keyword arguments that are sent to [`correct!`](@ref).
- `predict_kwargs`: An optional named tuple of keyword arguments that are sent to [`predict!`](@ref).
- `interact`: Whether or not to run the interaction step.
"""
function update!(imm::IMM, u, y, args...; correct_kwargs = (;), predict_kwargs = (;), interact = true)
    ll, rest = correct!(imm, u, y, args...; correct_kwargs...)
    combine!(imm)
    interact && interact!(imm)
    predict!(imm, u, args...; predict_kwargs...)
    ll, rest
end

(imm::IMM)(args...; kwargs...) = update!(imm::IMM, args...; kwargs...)

function reset!(imm::IMM)
    (; models, μ0) = imm
    for model in models
        reset!(model)
    end
    imm.x = sum(i->μ0[i]*models[i].x, eachindex(models))
    imm.R = sum(i->μ0[i]*models[i].R, eachindex(models))
    imm.μ = copy(imm.μ0)
    nothing
end


function sample_state(imm::IMM, p=parameters(kf); noise=true)
    sum(i->imm.μ[i]*sample_state(imm.models[i], p; noise=noise), eachindex(imm.models))
end
function sample_state(imm::IMM, x, u, p, t; noise=true)
    sum(i->imm.μ[i]*sample_state(imm.models[i], x, u, p, t; noise=noise), eachindex(imm.models))
end
function sample_measurement(imm::IMM, x, u, p, t; noise=true)
    sum(i->imm.μ[i]*sample_measurement(imm.models[i], x, u, p, t; noise=noise), eachindex(imm.models))
end
function measurement(imm::IMM)
    function imm_measurement(args...)
        sum(i->imm.μ[i]*measurement(imm.models[i])(args...), eachindex(imm.models))
    end
end
function dynamics(imm::IMM)
    function imm_dynamics(args...)
        sum(i->imm.μ[i]*dynamics(imm.models[i])(args...), eachindex(imm.models))
    end
end

particletype(imm::IMM) = typeof(imm.x)
covtype(imm::IMM) = typeof(imm.R)
state(imm::IMM) = imm.x
covariance(imm::IMM) = imm.R


"""
    forward_trajectory(imm::IMM, u, y, p = parameters(imm); interact = true)

When performing batch filtering using an [`IMM`](@ref) filter, one may
- Override the `interact` parameter of the filter
- Access the mode probabilities along the trajectory as the `sol.extra` field. This is a matrix of size `(n_modes, T)` where `T` is the length of the trajectory (length of `u` and `y`).

The returned solution object is of type [`KalmanFilteringSolution`](@ref) and has the following fields:
"""
function forward_trajectory(imm::IMM, u::AbstractVector, y::AbstractVector, p=parameters(imm); interact = true)
    reset!(imm)
    T    = length(y)
    x    = Array{particletype(imm)}(undef,T)
    xt   = Array{particletype(imm)}(undef,T)
    R    = Array{covtype(imm)}(undef,T)
    Rt   = Array{covtype(imm)}(undef,T)
    μ    = zeros(length(imm.μ), T)
    e    = similar(y)
    ll   = zero(eltype(particletype(imm)))
    for t = 1:T
        ti = (t-1)*imm.Ts
        x[t]  = state(imm)      |> copy
        R[t]  = covariance(imm) |> copy
        lli, _ = correct!(imm, u[t], y[t], p, ti)
        ll += lli
        μ[:, t] .= imm.μ
        combine!(imm)
        yh = measurement(imm)(state(imm), u[t], p, ti)
        e[t] = y[t] .- yh
        interact && interact!(imm)
        xt[t] = state(imm)      |> copy
        Rt[t] = covariance(imm) |> copy
        predict!(imm, u[t], p, ti)
    end
    KalmanFilteringSolution(imm,u,y,x,xt,R,Rt,ll,e,μ)
end