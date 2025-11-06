"""
Reset the filter to initial state and covariance/distribution
"""
function reset!(pf::AbstractParticleFilter)
    s = state(pf)
    rn = rng(pf)
    for i = eachindex(s.xprev)
        s.xprev[i] = rand(rn, initial_density(pf))
        s.x[i] = copy(s.xprev[i])
    end
    fill!(s.w, -log(num_particles(pf)))
    fill!(s.we, 1/num_particles(pf))
    s.t[] = 1
end

@inline get_mat(A::Nothing,x,u,p,t) = 0
@inline get_mat(A::Union{AbstractMatrix, Number},x,u,p,t) = A
@inline get_mat(A::AbstractArray{<:Any, 3},x,u,p,t) = @view A[:,:,t+1]
@inline get_mat(A::Function,x,u,p,t) = A(x,u,p,t)

"""
    get_mat(A::Union{AbstractMatrix, Number},x,u,p,t) = A
    get_mat(A::AbstractArray{<:Any, 3},x,u,p,t) = A[:,:,t+1]
    get_mat(A::Function,x,u,p,t) = A(x,u,p,t)

This is a helper function that makes it possible to supply any of
- A matrix
- A "time varying" matrix where time is in the last dimension (the array will be indexed at `t+1` since `t` starts at 0)
- A function of `x,u,p,t` that returns the matrix

This is useful to implement things like
- Time varying dynamics
- Nonlinear dynamics in the form of parameter-varying dynamics
- Time or state varying process noise
"""
get_mat

"""
    predict!(kf::AbstractKalmanFilter, u, p = parameters(kf), t::Integer = index(kf); R1, α = kf.α)

Perform the prediction step (updating the state estimate to ``x(t+1|t)``).
If `R1` stored in `kf` is a function `R1(x, u, p, t)`, this function is evaluated at the state *before* the prediction is performed.
The dynamics noise covariance matrix `R1` stored in `kf` can optionally be overridden by passing the argument `R1`, in this case `R1` must be a matrix.
"""
function predict!(kf::AbstractKalmanFilter, u, p=parameters(kf), t::Real = index(kf)*kf.Ts; R1 = get_mat(kf.R1, kf.x, u, p, t), α = kf.α, At = get_mat(kf.A, kf.x, u, p, t), Bt = get_mat(kf.B, kf.x, u, p, t))
    (; x,R) = kf
    if u === nothing || length(u) == 0
        # Special case useful since empty input is common special case
        kf.x = At*x
    else
        kf.x = At*x .+ Bt*u |> vec
    end
    if α == 1
        if R isa SMatrix
            kf.R = symmetrize(At*R*At') + R1
        else
            AtR = At*R
            mul!(R, AtR, At')
            symmetrize(R)
            R .+= R1
        end
    else
        Ru = symmetrize(α*At*R*At')
        @bangbang kf.R .= Ru .+ R1
    end
    kf.t += 1
end

@inline function symmetrize(x::SArray)
    x = 0.5 .* (x .+ x')
    x
end
@inline function symmetrize(x)
    n = size(x,1)
    @inbounds for i = 1:n, j = i+1:n
        x[i,j] = 0.5 * (x[i,j] + x[j,i])
        x[j,i] = x[i,j]
    end
    x
end

"""
    (; ll, e, S, Sᵪ, K) = correct!(kf::AbstractKalmanFilter, u, y, p = parameters(kf), t::Integer = index(kf), R2)

The correct step for a Kalman filter returns not only the log likelihood `ll` and the prediction error `e`, but also the covariance of the output `S`, its Cholesky factor `Sᵪ` and the Kalman gain `K`.

If `R2` stored in `kf` is a function `R2(x, u, p, t)`, this function is evaluated at the state *before* the correction is performed.
The measurement noise covariance matrix `R2` stored in the filter object can optionally be overridden by passing the argument `R2`, in this case `R2` must be a matrix.

# Extended help
To perform separate measurement updates for different sensors, see the ["Measurement models" in the documentation](@ref measurement_models).
"""
function correct!(kf::AbstractKalmanFilter, mm::LinearMeasurementModel, u, y, p=parameters(kf), t::Real = index(kf)*kf.Ts; R2 = get_mat(mm.R2, kf.x, u, p, t), Ct = get_mat(mm.C, kf.x, u, p, t), Dt = get_mat(mm.D, kf.x, u, p, t))
    (;x,R) = kf
    e   = y .- Ct*x
    if !iszero(Dt)
        e -= Dt*u
    end
    S = symmetrize(Ct*R*Ct')
    @bangbang S .+= R2
    Sᵪ  = cholesky(Symmetric(S); check = false)
    issuccess(Sᵪ) || error("Cholesky factorization of innovation covariance failed at time step $(kf.t), got S = $(printarray(S))")
    K   = (R*Ct')/Sᵪ
    kf.x += K*e
    kf.R  = symmetrize((I - K*Ct)*R) # WARNING against I .- A
    ll = extended_logpdf(SimpleMvNormal(PDMat(S, Sᵪ)), e)# - 1/2*logdet(S) # logdet is included in logpdf
    (; ll, e, S, Sᵪ, K)
end

function correct!(kf::AbstractKalmanFilter, u, y, p=parameters(kf), t::Real = index(kf)*kf.Ts; kwargs...)
    measurement_model = LinearMeasurementModel(kf.C, kf.D, kf.R2, length(y), nothing)
    correct!(kf, measurement_model, u, y, p, t; kwargs...)
end

"""
    predict!(f, u, p = parameters(f), t = index(f))

Move filter state forward in time using dynamics equation and input vector `u`.
"""
function predict!(pf, u, p = parameters(pf), t = index(pf)*pf.Ts)
    s = pf.state
    N = num_particles(s)
    if shouldresample(pf)
        j = resample(pf)
        propagate_particles!(pf, u, j, p, t)
        reset_weights!(s)
    else # Resample not needed
        s.j .= 1:N
        propagate_particles!(pf, u, p, t)
    end
    copyto!(s.xprev, s.x)
    pf.state.t[] += 1
end


"""
    ll, e = correct!(pf, u, y, p = parameters(f), t = index(f))
    
Update state/weights based on measurement `y`,  returns log-likelihood and prediction error (the error is always 0 for particle filters).

# Extended help
To perform separate measurement updates for different sensors, see the ["Measurement models" in the documentation](@ref measurement_models). For [`AdvancedParticleFilter`](@ref), this can be realized by passing a custom `measurement_likelihood` function as the keyword argument `g` to `correct!`, or by calling the lower-level function `measurement_equation!` with a custom `measurement_likelihood`.
"""
function correct!(pf, u, y, p = parameters(pf), t = index(pf)*pf.Ts, args...; kwargs...)
    measurement_equation!(pf, u, y, p, t, args...; kwargs...)
    ll = logsumexp!(state(pf))
    ll, 0
end

function correct!(pf::AuxiliaryParticleFilter, u, y, p = parameters(pf), t = index(pf)*pf.Ts, args...; kwargs...)
    # The measurement update is done already in the predict step for AUX filter
    ll = logsumexp!(state(pf))
    ll, 0
end

"""
    ll, e = update!(f::AbstractFilter, u, y, p = parameters(f), t = index(f))

Perform one step of `predict!` and `correct!`, returns log-likelihood and prediction error
"""
function update!(f::AbstractFilter, u, y, p = parameters(f), t = index(f)*f.Ts)
    ll_e = correct!(f, u, y, p, t)
    predict!(f, u, p, t)
    ll_e
end

function update!(pf::AuxiliaryParticleFilter, u, y, y1, p = parameters(pf), t = index(pf)*pf.Ts)
    ll_e = correct!(pf, u, y, p, t)
    predict!(pf, u, y1, p, t)
    ll_e
end



function predict!(pf::AuxiliaryParticleFilter, u, y1, p = parameters(pf), t = index(pf)*pf.Ts)
    g = measurement(pf)
    d = measurement_density(pf)
    s = state(pf)
    propagate_particles!(pf.pf, u, p, t, nothing)# Propagate without noise
    λ  = s.we
    λ .= 0
    measurement_equation!(pf.pf, u, y1, p, t, λ)
    s.w .+= λ # old w + new w, equivalent to old_we*new_we
    expnormalize!(s.w) # w used as buffer
    j = resample(ResampleSystematic, s.w , s.j, s.bins)
    # reset_weights!(s) # We don't do this to keep λ and instead assign into w directly
    permute_with_buffer!(s.x, s.xprev, j)
    add_noise!(pf.pf)
    lN = log(num_particles(pf))
    for i = 1:length(s.we)
        # note unresampled λ[i] instead of λ[j[i]]
        s.w[i] = λ[i] - lN
    end

    s.t[] += 1
    copyto!(s.xprev, s.x)
end

function predict!(pf::AuxiliaryParticleFilter{<:AdvancedParticleFilter},u, y, p=parameters(pf), t = index(pf)*pf.Ts)
    s = state(pf)
    propagate_particles!(pf.pf, u, p, t, nothing)# Propagate without noise
    λ  = s.we
    λ .= 0
    measurement_equation!(pf.pf, u, y, p, t, λ)
    s.w .+= λ
    expnormalize!(s.w) # w used as buffer
    j = resample(ResampleSystematic, s.w , s.j, s.bins)
    reset_weights!(s)
    # NOTE: this can be sped by letting the user implement `add_noise` and call that here, rather than propagating again
    propagate_particles!(pf.pf, u, j, p, t)# Propagate with noise and permutation

    s.t[] += 1
    copyto!(s.xprev, s.x)
end


(kf::AbstractKalmanFilter)(u, y, p = parameters(kf), t = index(kf)*kf.Ts) =  update!(kf, u, y, p, t)
(pf::ParticleFilter)(u, y, p = parameters(pf), t = index(pf)*pf.Ts) =  update!(pf, u, y, p, t)
(pf::AuxiliaryParticleFilter)(u, y, y1, p = parameters(pf), t = index(pf)*pf.Ts) =  update!(pf, u, y, y1, p, t)
(pf::AdvancedParticleFilter)(u, y, p = parameters(pf), t = index(pf)*pf.Ts) =  update!(pf, u, y, p, t)


"""
    sol = forward_trajectory(kf::AbstractKalmanFilter, u::Vector, y::Vector, p=parameters(kf); debug=false)

Run a Kalman filter forward to perform (offline / batch) filtering along an entire trajectory `u, y`.

Returns a KalmanFilteringSolution: with the following
- `x`: predictions ``x(t|t-1)``
- `xt`: filtered estimates ``x(t|t)``
- `R`: predicted covariance matrices ``R(t|t-1)``
- `Rt`: filter covariances ``R(t|t)``
- `ll`: loglik

`sol` can be plotted
```
plot(sol::KalmanFilteringSolution; plotx = true, plotxt=true, plotu=true, ploty=true)
```
See [`KalmanFilteringSolution`](@ref) for more details.

# Extended help
## Very large systems
If your system is very large, i.e., the dimension of the state is very large, and the arrays `u,y` are long, this function may use a lot of memory to store all covariance matrices `R, Rt`. If you do not need all the information retained by this function, you may opt to call one of the functions
- [`loglik`](@ref)
- [`LowLevelParticleFilters.sse`](@ref)
- [`LowLevelParticleFilters.prediction_errors!`](@ref)
That store significantly less information. The amount of computation performed by all of these functions is identical, the only difference lies in what is stored and returned.

## Callbacks
For advanced usage, such as implementing conditional resetting and adaptive covariance, one may make use of the callback functions
- `pre_correct_cb(kf, u, y, p, t)`: called before the correction step, returns either `nothing` or a covariance matrix `R2` to use in the correction step.
- `pre_predict_cb(kf, u, y, p, t, ll, e, S, Sᵪ)`: called before the prediction step, returns either `nothing` or a covariance matrix `R1` to use in the prediction step. The arguments to this callback are `filter, input, measurement, parameters, time, loglikelihood, prediction error, innovation covariance and Cholesky factor of the innovation covariance`, essentially all the information available after the correct step.

The filter loop consists of the following steps, in this order:
1. `pre_correct_cb`
2. `correct!`
3. `post_correct_cb` # This callback is considered internal, the signature is subject to change. This happens after correction, but before state and covariance is saved
4. `pre_predict_cb`
5. `predict!`
6. `post_predict_cb` # This happens after prediction, but before next iteration when the state and covariance is saved
"""
function forward_trajectory(kf::AbstractKalmanFilter, u::AbstractVector, y::AbstractVector, p=parameters(kf); t = range(0, step=kf.Ts, length=length(y)), debug=false, pre_correct_cb=(args...)->nothing, pre_predict_cb=(args...)->nothing, post_predict_cb=(args...)->nothing, post_correct_cb=(args...)->nothing)
    reset!(kf)
    T    = length(y)
    x    = Array{particletype(kf)}(undef,T)
    xt   = Array{particletype(kf)}(undef,T)
    R    = Array{covtype(kf)}(undef,T)
    Rt   = Array{covtype(kf)}(undef,T)
    e    = similar(y)
    ll   = zero(eltype(particletype(kf)))
    local k, S, K
    try
        for outer k = 1:T
            ti = t[k]
            x[k]  = state(kf)      |> copy
            R[k]  = covariance(kf) |> copy
            R2 = pre_correct_cb(kf, u[k], y[k], p, ti)
            ret = correct!(kf, u[k], y[k], p, ti; R2 = something(R2, get_mat(kf.R2, kf.x, u[k], p, ti)))
            lli, ei, Si, Sᵪi, Ki = ret
            post_correct_cb(kf, p, ret)
            ll += lli
            e[k] = ei
            xt[k] = state(kf)      |> copy
            Rt[k] = covariance(kf) |> copy

            if k == 1
                S = Vector{typeof(Sᵪi)}(undef, T)
                K = Vector{typeof(Ki)}(undef, T)
            end
            S[k] = Sᵪi
            K[k] = Ki
            R1 = pre_predict_cb(kf, u[k], y[k], p, ti, lli, ei, Si, Sᵪi)
            predict!(kf, u[k], p, ti; R1 = something(R1, get_mat(kf.R1, kf.x, u[k], p, ti)))
            post_predict_cb(kf, p)
        end
    catch err
        if debug
            k -= 1
            x, xt, R, Rt, e, u, y = x[1:k], xt[1:k], R[1:k], Rt[1:k], e[1:k], u[1:k], y[1:k]
            @error "State estimation failed, returning partial solution" err
        else
            @error "State estimation failed, pass `debug = true` to forward_trajectory to return a partial solution"
            rethrow()
        end
    end
    KalmanFilteringSolution(kf,u,y,x,xt,R,Rt,ll,e,K,S,nothing,t)
end


"""
    sol = forward_trajectory(pf, u::AbstractVector, y::AbstractVector, p=parameters(pf))

Run the particle filter for a sequence of inputs and measurements (offline / batch filtering). Return a solution with
`x,w,we,ll = particles, weights, expweights and loglikelihood`

If [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl) is loaded, you may transform the output particles to `Matrix{MonteCarloMeasurements.Particles}` using `Particles(x,we)`. Internally, the particles are then resampled such that they all have unit weight. This is conventient for making use of the [plotting facilities of MonteCarloMeasurements.jl](https://baggepinnen.github.io/MonteCarloMeasurements.jl/stable/#Plotting-1).

`sol` can be plotted
```
plot(sol::ParticleFilteringSolution; nbinsy=30, xreal=nothing, dim=nothing)
```
"""
function forward_trajectory(pf, u::AbstractVector, y::AbstractVector, p=parameters(pf); pre_correct_cb=(args...)->nothing, pre_predict_cb=(args...)->nothing, post_predict_cb=(args...)->nothing, post_correct_cb=(args...)->nothing)
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(undef,N,T)
    w = Array{Float64}(undef,N,T)
    we = Array{Float64}(undef,N,T)
    ll = 0.
    @inbounds for t = 1:T
        ti = (t-1)*pf.Ts
        pre_correct_cb(pf, u[t], y[t], p, ti)
        lli = correct!(pf, u[t], y[t], p, ti) |> first
        post_correct_cb(pf, u[t], y[t], p, ti, lli)
        ll += lli
        x[:,t] .= particles(pf)
        w[:,t] .= weights(pf)
        we[:,t] .= expweights(pf)
        pre_predict_cb(pf, u[t], y[t], p, ti, lli)
        predict!(pf, u[t], p, ti)
        post_predict_cb(pf, u[t], y[t], p, ti)
    end
    ParticleFilteringSolution(pf,u,y,x,w,we,ll)
end

function forward_trajectory(pf::AuxiliaryParticleFilter, u::AbstractVector, y::AbstractVector, p=parameters(pf))
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(undef,N,T)
    w = Array{Float64}(undef,N,T)
    we = Array{Float64}(undef,N,T)
    ll = 0.
    @inbounds for t = 1:T
        ti = (t-1)*pf.Ts
        ll += correct!(pf, u[t], y[t], p, ti) |> first
        x[:,t] .= particles(pf)
        w[:,t] .= weights(pf)
        we[:,t] .= expweights(pf)
        t < T && predict!(pf, u[t], y[t+1], p, ti)
    end
    ParticleFilteringSolution(pf,u,y,x,w,we,ll)
end



"""
    x,ll = mean_trajectory(pf, u::Vector{Vector}, y::Vector{Vector}, p=parameters(pf))

This method resets the particle filter to the initial state distribution upon start
"""
mean_trajectory(pf, u::Vector, y::Vector, p=pf.p; kwargs...) = reduce_trajectory(pf, u::Vector, y::Vector, weighted_mean, p; kwargs...)

mode_trajectory(pf, u::Vector, y::Vector, p=pf.p; kwargs...) = reduce_trajectory(pf, u::Vector, y::Vector, mode, p; kwargs...)

"""
    mean_trajectory(sol::ParticleFilteringSolution)
    mean_trajectory(x::AbstractMatrix, we::AbstractMatrix)

Compute the weighted mean along the trajectory of a particle-filter solution. Returns a matrix of size `T × nx`.
If `x` and `we` are supplied, the weights are expected to be in the original space (not log space).

See also [`mode_trajectory`](@ref)
"""
mean_trajectory(sol::ParticleFilteringSolution) = mean_trajectory(sol.x, sol.we)


"""
    mode_trajectory(sol::ParticleFilteringSolution)
    mode_trajectory(x::AbstractMatrix, we::AbstractMatrix)

Compute the mode (particle with largest weight) along the trajectory of a particle-filter solution. Returns a matrix of size `T × nx`.
"""
mode_trajectory(sol::ParticleFilteringSolution) = mode_trajectory(sol.x, sol.we)

function reduce_trajectory(pf, u::Vector, y::Vector, f::F, p=parameters(pf); pre_correct_cb=(args...)->nothing, post_predict_cb=(args...)->nothing) where F
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(undef,T)
    ll = correct!(pf,u[1],y[1],p,0) |> first
    x[1] = f(state(pf))
    for t = 2:T
        ti = (t-1)*pf.Ts
        pre_correct_cb(pf,u[t-1],y[t],p,ti)
        ll += pf(u[t-1], y[t], p, ti) |> first
        post_predict_cb(pf,u[t-1],y[t],p,ti)
        x[t] = f(pf)
    end
    x,ll
end

StatsBase.mode(pf::AbstractParticleFilter) = particles(pf)[findmax(expparticles(pf))[2]]

mode_trajectory(x::AbstractMatrix, we::AbstractMatrix) =  reduce(hcat,vec(x[findmax(we, dims=1)[2]]))'

function mean_trajectory(x::AbstractMatrix, we::AbstractMatrix)
    copy(reduce(hcat,vec(sum(x.*we,dims=1)))')
end


"""
    x,u,y = simulate(f::AbstractFilter, T::Int, du::Distribution, p=parameters(f), [N]; dynamics_noise=true, measurement_noise=true)
    x,u,y = simulate(f::AbstractFilter, u, p=parameters(f); dynamics_noise=true, measurement_noise=true)

Simulate dynamical system forward in time `T` steps, or for the duration of `u`. Returns state sequence, inputs and measurements.

- `u` is an input-signal trajectory, alternatively, `du` is a distribution of random inputs.

A simulation can be considered a draw from the prior distribution over the evolution of the system implied by the selected noise models. Such a simulation is useful in order to evaluate whether or not the noise models are reasonable.

If [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl) is loaded, the argument `N::Int` can be supplied, in which case `N` simulations are done and the result is returned in the form of `Vector{MonteCarloMeasurements.Particles}`.
"""
function simulate(f::AbstractFilter, T::Int, du, p=parameters(f); dynamics_noise=true, measurement_noise=true, sample_initial=false)
    u = [rand(du) for t=1:T]
    simulate(f, u, p; dynamics_noise, measurement_noise, sample_initial)
end

function simulate(f::AbstractFilter,u,p=parameters(f); dynamics_noise=true, measurement_noise=true, sample_initial=false)
    T = length(u)
    x1 = sample_state(f, p; noise=sample_initial)
    y1 = sample_measurement(f,x1, u[1], p, 0; noise=measurement_noise)
    y = Vector{typeof(y1)}(undef, T)
    x = Vector{typeof(x1)}(undef, T)
    y[1] = y1
    x[1] = x1
    for t = 1:T-1
        ti = (t-1)*f.Ts
        y[t] = sample_measurement(f,x[t], u[t], p, ti; noise=measurement_noise)
        x[t+1] = sample_state(f, x[t], u[t], p, ti; noise=dynamics_noise)
    end
    y[T] = sample_measurement(f,x[T], u[T], p, (T-1)*f.Ts; noise=measurement_noise)
    x,u,y
end

function rollout(f, x0::AbstractVector, u, p=nothing; Ts=f.Ts)
    x = [x0]
    for (i,u) in enumerate(u)
        push!(x, f(x[end], u, p, i*Ts))
    end
    x
end


"""
    x̂ = weighted_mean(x,we)

Calculated weighted mean of particle trajectories. `we` are expweights.
"""
function weighted_mean(x,we::AbstractVector)
    @assert sum(we) ≈ 1 "Weights must sum to one, did you pass log weights instead of exp weights?"
    xh = zeros(size(x[1]))
    @inbounds @simd  for i = eachindex(x)
        xh .+= x[i].*we[i]
    end
    return xh
end
function weighted_mean(x,we::AbstractMatrix)
    N,T = size(x)
    @assert sum(we) ≈ T
    xh = zeros(eltype(x), T)
    for t = 1:T
        @inbounds @simd for i = 1:N
            xh[t] += x[i,t].*we[i,t]
        end
    end
    return xh
end

"""
    x̂ = weighted_mean(pf)
    x̂ = weighted_mean(s::PFstate)

See also [`mean_trajectory`](@ref)
"""
weighted_mean(s) = weighted_mean(s.x,s.we)
weighted_mean(pf::AbstractParticleFilter) = weighted_mean(state(pf))

"""
    weighted_cov(x,we)

Similar to [`weighted_mean`](@ref), but returns covariances
"""
function weighted_cov(x,we)
    N,T = size(x)
    n = length(x[1])
    [cov(copy(reshape(reinterpret(Float64, x[:,t]),n,N)),ProbabilityWeights(we[:,t]), 2, corrected=true) for t = 1:T]
end

function weighted_cov(sol::ParticleFilteringSolution)
    weighted_cov(sol.x,sol.we)
end


"""
    weighted_quantile(x,we,q)
    weighted_quantile(sol,q)

Calculated weighted quantile `q` of particle trajectories. `we` are expweights. Returns a vector of length `size(x, 2)` where each entry has length `nx`. For a particle-filtering solution, this means the vector will be as long as the number of time steps in the solution.
"""
function weighted_quantile(x,we,q)
    [[quantile(getindex.(x[:, t], i),ProbabilityWeights(we[:,t]),q) for i = 1:length(x[1])] for t = 1:size(x,2)]
end
weighted_quantile(sol::ParticleFilteringSolution,q) = weighted_quantile(sol.x,sol.we,q)
