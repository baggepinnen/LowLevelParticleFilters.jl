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

@inline get_mat(A::Union{AbstractMatrix, Number},x,u,p,t) = A
@inline get_mat(A::AbstractArray{<:Any, 3},x,u,p,t) = @view A[:,:,t]
@inline get_mat(A::Function,x,u,p,t) = A(x,u,p,t)

"""
    get_mat(A::Union{AbstractMatrix, Number},x,u,p,t) = A
    get_mat(A::AbstractArray{<:Any, 3},x,u,p,t) = A[:,:,t]
    get_mat(A::Function,x,u,p,t) = A(x,u,p,t)

This is a helper function that makes it possible to supply any of
- A matrix
- A "time varying" matrix where time is in the last dimension
- A function of `x,u,p,t` that returns the matrix

This is useful to implement things like
- Time varying dynamics
- Nonlinear dynamics in the form of parameter-varying dynamics
- Time or state varying process noise
"""
get_mat

"""
    predict!(kf::AbstractKalmanFilter, u, p = parameters(kf), t::Integer = index(kf); R1)

Perform the prediction step (updating the state estimate to ``x(t+1|t)``).
If `R1` stored in `kf` is a function `R1(x, u, p, t)`, this function is evaluated at the state *before* the prediciton is performed.
The dynamics noise covariance matrix `R1` stored in `kf` can optionally be overridden by passing the argument `R1`, in this case `R1` must be a matrix.
"""
function predict!(kf::AbstractKalmanFilter, u, p=parameters(kf), t::Real = index(kf); R1 = get_mat(kf.R1, kf.x, u, p, t))
    @unpack A,B,x,R = kf
    At = get_mat(A, x, u, p, t)
    Bt = get_mat(B, x, u, p, t)
    kf.x = At*x .+ Bt*u |> vec
    if kf.α == 1
        Ru = symmetrize(At*R*At')
        kf.R = Ru + R1
    else
        Ru = symmetrize(kf.α*At*R*At')
        kf.R = Ru + R1
    end
    kf.t[] += 1
end

@inline function symmetrize(x::SArray)
    x = 0.5 .* (x .+ x')
    Symmetric(x)
end
@inline function symmetrize(x)
    n = size(x,1)
    @inbounds for i = 1:n, j = i+1:n
        x[i,j] = 0.5 * (x[i,j] + x[j,i])
        x[j,i] = x[i,j]
    end
    Symmetric(x)
end

"""
    (; ll, e, S, Sᵪ, K) = correct!(kf::AbstractKalmanFilter, u, y, p = parameters(kf), t::Integer = index(kf), R2)

The correct step for a Kalman filter returns not only the log likelihood `ll` and the prediction error `e`, but also the covariance of the output `S`, its Cholesky factor `Sᵪ` and the Kalman gain `K`.

If `R2` stored in `kf` is a function `R2(x, u, p, t)`, this function is evaluated at the state *before* the correction is performed.
The measurement noise covariance matrix `R2` stored in the filter object can optionally be overridden by passing the argument `R2`, in this case `R2` must be a matrix.
"""
function correct!(kf::AbstractKalmanFilter, u, y, p=parameters(kf), t::Real = index(kf); R2 = get_mat(kf.R2, kf.x, u, p, t))
    @unpack C,D,x,R = kf
    Ct = get_mat(C, x, u, p, t)
    Dt = get_mat(D, x, u, p, t)
    e   = y .- Ct*x
    if !iszero(D)
        e -= Dt*u
    end
    S   = symmetrize(Ct*R*Ct') + R2
    Sᵪ  = cholesky(S)
    K   = (R*Ct')/Sᵪ
    kf.x += K*e
    kf.R  = symmetrize((I - K*Ct)*R) # WARNING against I .- A
    ll = logpdf(MvNormal(PDMat(S, Sᵪ)), e)# - 1/2*logdet(S) # logdet is included in logpdf
    (; ll, e, S, Sᵪ, K)
end

"""
    predict!(f, u, p = parameters(f), t = index(f))

Move filter state forward in time using dynamics equation and input vector `u`.
"""
function predict!(pf, u, p = parameters(pf), t = index(pf))
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
    ll, e = correct!(f, u, y, p = parameters(f), t = index(f))
    
Update state/covariance/weights based on measurement `y`,  returns loglikelihood and prediction error (the error is always 0 for particle filters).
"""
function correct!(pf, u, y, p = parameters(pf), t = index(pf))
    measurement_equation!(pf, u, y, p, t)
    ll = logsumexp!(state(pf))
    ll, 0
end

"""
    ll, e = update!(f::AbstractFilter, u, y, p = parameters(f), t = index(f))

Perform one step of `predict!` and `correct!`, returns loglikelihood and prediction error
"""
function update!(f::AbstractFilter, u, y, p = parameters(f), t = index(f))
    ll_e = correct!(f, u, y, p, t)
    predict!(f, u, p, t)
    ll_e
end

function update!(pf::AuxiliaryParticleFilter, u, y, y1, p = parameters(pf), t = index(pf))
    ll_e = correct!(pf, u, y, p, t)
    predict!(pf, u, y1, p, t)
    ll_e
end



function predict!(pf::AuxiliaryParticleFilter, u, y1, p = parameters(pf), t = index(pf))
    s = state(pf)
    propagate_particles!(pf.pf, u, p, t, nothing)# Propagate without noise
    λ  = s.we
    λ .= 0
    measurement_equation!(pf.pf, u, y1, p, t, λ)
    s.w .+= λ
    expnormalize!(s.w) # w used as buffer
    j = resample(ResampleSystematic, s.w , s.j, s.bins)
    reset_weights!(s)
    permute_with_buffer!(s.x, s.xprev, j)
    add_noise!(pf.pf)

    s.t[] += 1
    copyto!(s.xprev, s.x)
end


function predict!(pf::AuxiliaryParticleFilter{<:AdvancedParticleFilter},u, y, p=parameters(pf), t = index(pf))
    s = state(pf)
    propagate_particles!(pf.pf, u, p, t, nothing)# Propagate without noise
    λ  = s.we
    λ .= 0
    measurement_equation!(pf.pf, u, y, p, t, λ)
    s.w .+= λ
    expnormalize!(s.w) # w used as buffer
    j = resample(ResampleSystematic, s.w , s.j, s.bins)
    reset_weights!(s)
    propagate_particles!(pf.pf, u, j, p, t)# Propagate with noise and permutation

    s.t[] += 1
    copyto!(s.xprev, s.x)
end


(kf::KalmanFilter)(u, y, p=parameters(kf), t = index(kf)) =  update!(kf, u, y, p, t)
(kf::AbstractUnscentedKalmanFilter)(u, y, p = parameters(kf), t = index(kf)) =  update!(kf, u, y, p, t)
(pf::ParticleFilter)(u, y, p = parameters(pf), t = index(pf)) =  update!(pf, u, y, p, t)
(pf::AuxiliaryParticleFilter)(u, y, y1, p = parameters(pf), t = index(pf)) =  update!(pf, u, y, y1, p, t)
(pf::AdvancedParticleFilter)(u, y, p = parameters(pf), t = index(pf)) =  update!(pf, u, y, p, t)


"""
    sol = forward_trajectory(kf::AbstractKalmanFilter, u::Vector, y::Vector, p=parameters(kf))

Run a Kalman filter forward

Returns a KalmanFilteringSolution: with the following
- `x`: predictions
- `xt`: filtered estimates
- `R`: predicted covariance matrices
- `Rt`: filter covariances
- `ll`: loglik

`sol` can be plotted
```
plot(sol::KalmanFilteringSolution; plotx = true, plotxt=true, plotu=true, ploty=true)
```
"""
function forward_trajectory(kf::AbstractKalmanFilter, u::AbstractVector, y::AbstractVector, p=parameters(kf))
    reset!(kf)
    T    = length(y)
    x    = Array{particletype(kf)}(undef,T)
    xt   = Array{particletype(kf)}(undef,T)
    R    = Array{covtype(kf)}(undef,T)
    Rt   = Array{covtype(kf)}(undef,T)
    ll   = zero(eltype(particletype(kf)))
    for t = 1:T
        x[t]  = state(kf)      |> copy
        R[t]  = covariance(kf) |> copy
        ll   += correct!(kf, u[t], y[t], p, t)[1]
        xt[t] = state(kf)      |> copy
        Rt[t] = covariance(kf) |> copy
        predict!(kf, u[t], p, t)
    end
    KalmanFilteringSolution(kf,u,y,x,xt,R,Rt,ll)
end


"""
    sol = forward_trajectory(pf, u::AbstractVector, y::AbstractVector, p=parameters(pf))

Run the particle filter for a sequence of inputs and measurements. Return a solution with
`x,w,we,ll = particles, weights, expweights and loglikelihood`

If [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl) is loaded, you may transform the output particles to `Matrix{MonteCarloMeasurements.Particles}` using `Particles(x,we)`. Internally, the particles are then resampled such that they all have unit weight. This is conventient for making use of the [plotting facilities of MonteCarloMeasurements.jl](https://baggepinnen.github.io/MonteCarloMeasurements.jl/stable/#Plotting-1).

`sol` can be plotted
```
plot(sol::ParticleFilteringSolution; nbinsy=30, xreal=nothing, dim=nothing)
```
"""
function forward_trajectory(pf, u::AbstractVector, y::AbstractVector, p=parameters(pf))
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(undef,N,T)
    w = Array{Float64}(undef,N,T)
    we = Array{Float64}(undef,N,T)
    ll = 0.
    @inbounds for t = 1:T
        ll += correct!(pf, u[t], y[t], p, t) |> first
        x[:,t] .= particles(pf)
        w[:,t] .= weights(pf)
        we[:,t] .= expweights(pf)
        predict!(pf, u[t], p, t)
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
        ll += correct!(pf, u[t], y[t], p, t) |> first
        x[:,t] .= particles(pf)
        w[:,t] .= weights(pf)
        we[:,t] .= expweights(pf)
        t < T && predict!(pf, u[t], y[t+1], p, t)
    end
    ParticleFilteringSolution(pf,u,y,x,w,we,ll)
end



"""
    x,ll = mean_trajectory(pf, u::Vector{Vector}, y::Vector{Vector}, p=parameters(pf))

This method resets the particle filter to the initial state distribution upon start
"""
mean_trajectory(pf, u::Vector, y::Vector) = reduce_trajectory(pf, u::Vector, y::Vector, weighted_mean)
mode_trajectory(pf, u::Vector, y::Vector) = reduce_trajectory(pf, u::Vector, y::Vector, mode)

"""
    mean_trajectory(sol::ParticleFilteringSolution)
    mean_trajectory(x::AbstractMatrix, we::AbstractMatrix)

Compute the weighted mean along the trajectory of a particle-filter solution. Returns a matrix of size `T × nx`.
If `x` and `we` are supplied, the weights are expected to be in the original space (not log space).
"""
mean_trajectory(sol::ParticleFilteringSolution) = mean_trajectory(sol.x, sol.we)
mode_trajectory(sol::ParticleFilteringSolution) = mode_trajectory(sol.x, sol.we)

function reduce_trajectory(pf, u::Vector, y::Vector, f::F, p=parameters(pf)) where F
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(undef,T)
    ll = correct!(pf,u[1],y[1],p,1) |> first
    x[1] = f(state(pf))
    for t = 2:T
        ll += pf(u[t-1], y[t], p, t) |> first
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

Simulate dynamical system forward in time `T` steps, or for the duration of `u`, returns state sequence, inputs and measurements
`du` is a distribution of random inputs.

A simulation can be considered a draw from the prior distribution over the evolution of the system implied by the selected noise models. Such a simulation is useful in order to evaluate whether or not the noise models are reasonable.

If [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl) is loaded, the argument `N::Int` can be supplied, in which case `N` simulations are done and the result is returned in the form of `Vector{MonteCarloMeasurements.Particles}`.
"""
function simulate(f::AbstractFilter, T::Int, du::Distribution, p=parameters(f); dynamics_noise=true, measurement_noise=true, sample_initial=false)
    u = [rand(du) for t=1:T]
    simulate(f, u, p; dynamics_noise, measurement_noise, sample_initial)
end

function simulate(f::AbstractFilter,u,p=parameters(f); dynamics_noise=true, measurement_noise=true, sample_initial=false)
    y = similar(u)
    x = similar(u)
    x[1] = sample_state(f, p; noise=sample_initial)
    T = length(u)
    for t = 1:T-1
        y[t] = sample_measurement(f,x[t], u[t], p, t; noise=measurement_noise)
        x[t+1] = sample_state(f, x[t], u[t], p, t; noise=dynamics_noise)
    end
    y[T] = sample_measurement(f,x[T], u[T], p, T; noise=measurement_noise)
    x,u,y
end

function rollout(f, x0::AbstractVector, u, p=nothing; Ts=1)
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
    @assert sum(we) ≈ 1
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
    [cov(copy(reshape(reinterpret(Float64, x[:,t]),n,N)),ProbabilityWeights(we[:,t]), dims=2, corrected=true) for t = 1:T]
end
