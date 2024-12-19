# Interacting multiple models

mutable struct IMM{MT, PT, XT, RT, μT} <: AbstractFilter
    models::MT
    P::PT
    x::XT
    R::RT
    μ::μT
end


"""
    IMM(models, P, μ; check = true)

Interacting Multiple Model (IMM) filter. This filter is a combination of multiple Kalman-type filters, each with its own state and covariance. The IMM filter is a probabilistically weighted average of the states and covariances of the individual filters. The weights are determined by the probability matrix `P` and the mixing probabilities `μ`.

!!! warning "Experimental"
    This filter is currently considered experimental and the user interface may change in the future without respecting semantic versioning.

# Arguments:
- `models`: An array of Kalman-type filters, such as [`KalmanFilter`](@ref), [`ExtendedKalmanFilter`](@ref), [`UnscentedKalmanFilter`](@ref), etc. The state of each model must have the same meaning, such that forming a weighted average makes sense.
- `P`: The mode-transition probability matrix. `P[i,j]` is the probability of transitioning from mode `i` to mode `j` (each row must sum to one).
- `μ`: The initial mixing probabilities. `μ[i]` is the probability of being in mode `i` at the initial contidion (must sum to one).
- `check`: If `true`, check that the inputs are valid. If `false`, skip the checks.
"""
function IMM(models, P, μ; check=true)
    if check
        N = length(models)
        length(μ) == N || throw(ArgumentError("μ must have the same length as the number of models"))
        LinearAlgebra.checksquare(P) == N || throw(ArgumentError("P must be square with side length same as the number of models"))
        sum(μ) ≈ 1.0 || throw(ArgumentError("μ must sum to 1.0"))
        sum(P, dims=2) ≈ ones(N) || throw(ArgumentError("P must sum to 1.0 along rows"))
        allequal(typeof(m.x) for m in models) || @warn("The list of models have different type of their state vector x, this leads to poor performance. Turn off this warining by passing IMM(..., check=false)")
        allequal(typeof(m.R) for m in models) || @warn("The list of models have different type of their state vector x, this leads to poor performance. Turn off this warining by passing IMM(..., check=false)")
        allequal(m.Ts for m in models) || throw(ArgumentError("All models must have the same sampling time Ts"))
    end
    x = sum(i->μ[i]*models[i].x, eachindex(models))
    R = sum(i->μ[i]*models[i].R, eachindex(models))
    IMM(models, P, x, R, μ)
end

function Base.getproperty(imm::IMM, s::Symbol)
    s ∈ fieldnames(typeof(imm)) && return getfield(imm, s)
    if s === :Ts
        return getfield(imm, :models)[1].Ts
    else
        throw(ArgumentError("$(typeof(imm)) has no property named $s"))
    end
end



function interact!(imm::IMM)
    (; μ, P, models) = imm
    @assert sum(μ) ≈ 1.0
    cj = P'μ
    new_x = [0*model.x for model in models]
    new_R = [0*model.R for model in models]
    for j in eachindex(models)
        for i = 1:length(models)
            μij = P[i,j] * μ[i] / cj[j]
            @bangbang new_x[j] .+= μij .* models[i].x
        end
        for i = eachindex(models)
            μij = P[i,j] * μ[i] / cj[j]
            d = models[i].x - new_x[j]
            @bangbang new_R[j] .+= μij .* (d * d' .+  models[i].R)
        end
    end
    for (model, x, R) in zip(models, new_x, new_R)
        model.x = x
        model.R = R
    end

    nothing
end

function predict!(imm::IMM, args...; kwargs...)
    for model in imm.models
        predict!(model, args...; kwargs...)
    end
end

function correct!(imm::IMM, u, y, t, args...; kwargs...)
    (; μ, P, models) = imm
    lls = zeros(eltype(imm.x), length(models))
    rest = []
    for (j, model) in enumerate(models)
        lls[j], others... = correct!(model, u, y, args...; kwargs...)
        push!(rest, others)
    end
    μP = P'μ # TODO: verify order we want for P
    new_μ = exp.(lls) .* μP
    μ .= new_μ ./ sum(new_μ)

    sum(lls), rest
end

function combine!(imm::IMM)
    (; μ, x, R, models) = imm
    @assert sum(μ) ≈ 1.0

    x = 0*x
    R = 0*R

    for (j, model) in enumerate(models)
        @bangbang x .+= μ[j] .* model.x
    end

    for (j, model) in enumerate(models)
        d = model.x .- x
        @bangbang R .+= μ[j] .* (model.R .+ d * d')
    end

    imm.x = x
    imm.R = R
    nothing
end


function update!(imm::IMM, args...; kwargs...)
    ll, rest = correct!(imm, args...; kwargs...)
    combine!(imm)
    interact!(imm)
    predict!(imm, args...; kwargs...)
    ll, rest
end

function reset!(imm::IMM)
    for model in imm.models
        reset!(model)
    end
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

function forward_trajectory(imm::IMM, u::AbstractVector, y::AbstractVector, p=parameters(imm))
    reset!(imm)
    T    = length(y)
    x    = Array{particletype(imm)}(undef,T)
    xt   = Array{particletype(imm)}(undef,T)
    R    = Array{covtype(imm)}(undef,T)
    Rt   = Array{covtype(imm)}(undef,T)
    e    = similar(y)
    ll   = zero(eltype(particletype(imm)))
    for t = 1:T
        ti = (t-1)*imm.Ts
        x[t]  = state(imm)      |> copy
        R[t]  = covariance(imm) |> copy
        lli, ei = correct!(imm, u[t], y[t], p, ti)
        ll += lli
        combine!(imm)
        yh = measurement(imm)(state(imm), u[t], p, ti)
        e[t] = y[t] .- yh
        interact!(imm)
        xt[t] = state(imm)      |> copy
        Rt[t] = covariance(imm) |> copy
        predict!(imm, u[t], p, ti)
    end
    KalmanFilteringSolution(imm,u,y,x,xt,R,Rt,ll,e)
end