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

function predict!(kf::AbstractKalmanFilter, u, t::Integer = index(kf))
    @unpack A,B,x,R,R1 = kf
    if ndims(A) == 3
        At = A[:,:,t]
        Bt = B[:,:,t]
    else
        At = A
        Bt = B
    end
    x .= At*x .+ Bt*u |> vec
    R .= symmetrize(At*R*At') + R1
    kf.t[] += 1
end

@inline function symmetrize(x::SArray)
    x = 0.5 .* (x .+ x')
    Symmetric(x)
end
@inline function symmetrize(x)
    x .+= x'
    x .*= 0.5
    Symmetric(x)
end

correct!(kf::AbstractKalmanFilter, u, y, t::Integer = index(kf)) = correct!(kf, u, y, 0, t)

function correct!(kf::AbstractKalmanFilter, u, y, t::Integer = index(kf))
    @unpack C,D,x,R,R2 = kf
    if ndims(C) == 3
        Ct = C[:,:,t]
    else
        Ct = C
    end
    # Handle D separately in case it is 0
    if ndims(D) == 3
        Dt = D[:,:,t]
    else
        Dt = D
    end
    e   = y .- Ct*x
    if ! iszero(D)
        e .-= Dt*u
    end
    S   = symmetrize(Ct*R*Ct') + R2
    Sᵪ  = cholesky(S)
    K   = (R*Ct')/Sᵪ
    x .+= K*e
    R  .= symmetrize((I - K*Ct)*R) # WARNING against I .- A
    ll = logpdf(MvNormal(PDMat(S, Sᵪ)), e)# - 1/2*logdet(S) # logdet is included in logpdf
    ll, e
end

"""
    predict!(f, u, t = index(f))
Move filter state forward in time using dynamics equation and input vector `u`.
"""
function predict!(pf, u, t = index(pf))
    s = pf.state
    N = num_particles(s)
    if shouldresample(pf)
        j = resample(pf)
        propagate_particles!(pf, u, j, t)
        reset_weights!(s)
    else # Resample not needed
        s.j .= 1:N
        propagate_particles!(pf, u, t)
    end
    copyto!(s.xprev, s.x)
    pf.state.t[] += 1
end


"""
     ll = correct!(f, u, y, t = index(f))
Update state/covariance/weights based on measurement `y`,  returns loglikelihood.
"""
function correct!(pf, u, y, t = index(pf))
    measurement_equation!(pf, u, y, t)
    ll = logsumexp!(state(pf))
    ll, 0
end

"""
    ll, e = update!(f::AbstractFilter, u, y, t = index(f))
Perform one step of `predict!` and `correct!`, returns loglikelihood and prediction error
"""
function update!(f::AbstractFilter, u, y, t = index(f))
    ll_e = correct!(f, u, y, t)
    predict!(f, u, t)
    ll_e
end

function update!(pf::AuxiliaryParticleFilter, u, y, y1, t = index(pf))
    ll_e = correct!(pf, u, y, t)
    predict!(pf, u, y1, t)
    ll_e
end



function predict!(pf::AuxiliaryParticleFilter, u, y1, t = index(pf))
    s = state(pf)
    propagate_particles!(pf.pf, u, t, nothing)# Propagate without noise
    λ  = s.we
    λ .= 0
    measurement_equation!(pf.pf, u, y1, t, λ)
    s.w .+= λ
    expnormalize!(s.w) # w used as buffer
    j = resample(ResampleSystematic, s.w , s.j, s.bins)
    reset_weights!(s)
    permute_with_buffer!(s.x, s.xprev, j)
    add_noise!(pf.pf)

    s.t[] += 1
    copyto!(s.xprev, s.x)
end


function predict!(pf::AuxiliaryParticleFilter{<:AdvancedParticleFilter},u, y, t = index(pf))
    s = state(pf)
    propagate_particles!(pf.pf, u, t, nothing)# Propagate without noise
    λ  = s.we
    λ .= 0
    measurement_equation!(pf.pf, u, y, t, λ)
    s.w .+= λ
    expnormalize!(s.w) # w used as buffer
    j = resample(ResampleSystematic, s.w , s.j, s.bins)
    reset_weights!(s)
    propagate_particles!(pf.pf, u, j, t)# Propagate with noise and permutation

    s.t[] += 1
    copyto!(s.xprev, s.x)
end


(kf::KalmanFilter)(u, y, t = index(kf)) =  update!(kf, u, y, t)
(kf::UnscentedKalmanFilter)(u, y, t = index(kf)) =  update!(kf, u, y, t)
(pf::ParticleFilter)(u, y, t = index(pf)) =  update!(pf, u, y, t)
(pf::AuxiliaryParticleFilter)(u, y, y1, t = index(pf)) =  update!(pf, u, y, y1, t)
(pf::AdvancedParticleFilter)(u, y, t = index(pf)) =  update!(pf, u, y, t)
(pf::SigmaFilter)(u, y, t = index(pf)) =  update!(pf, u, y, t)


"""
    x,xt,R,Rt,ll = forward_trajectory(kf::AbstractKalmanFilter, u::Vector, y::Vector)

Run a Kalman filter forward

# Returns:
- `x`: predictions
- `xt`: filtered estimates
- `R`: predicted covariance matrices
- `Rt`: filter covariances
- `ll`: loglik
"""
function forward_trajectory(kf::AbstractKalmanFilter, u::AbstractVector, y::AbstractVector)
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
        ll   += correct!(kf, u[t], y[t], t)[1]
        xt[t] = state(kf)      |> copy
        Rt[t] = covariance(kf) |> copy
        predict!(kf, u[t], t)
    end
    x,xt,R,Rt,ll
end


"""
    x,w,we,ll = forward_trajectory(pf, u::AbstractVector, y::AbstractVector)
Run the particle filter for a sequence of inputs and measurements. Return particles, weights, expweights and loglikelihood

If [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl) is loaded, you may transform the output particles to `Matrix{MonteCarloMeasurements.Particles}` using `Particles(x,we)`. Internally, the particles are then resampled such that they all have unit weight. This is conventient for making use of the [plotting facilities of MonteCarloMeasurements.jl](https://baggepinnen.github.io/MonteCarloMeasurements.jl/stable/#Plotting-1).
"""
function forward_trajectory(pf, u::AbstractVector, y::AbstractVector)
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(undef,N,T)
    w = Array{Float64}(undef,N,T)
    we = Array{Float64}(undef,N,T)
    ll = 0.
    @inbounds for t = 1:T
        ll += correct!(pf, u[t], y[t], t) |> first
        x[:,t] .= particles(pf)
        w[:,t] .= weights(pf)
        we[:,t] .= expweights(pf)
        predict!(pf, u[t], t)
    end
    x,w,we,ll
end

function forward_trajectory(pf::AuxiliaryParticleFilter, u::AbstractVector, y::AbstractVector)
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(undef,N,T)
    w = Array{Float64}(undef,N,T)
    we = Array{Float64}(undef,N,T)
    ll = 0.
    @inbounds for t = 1:T
        ll += correct!(pf, u[t], y[t], t) |> first
        x[:,t] .= particles(pf)
        w[:,t] .= weights(pf)
        we[:,t] .= expweights(pf)
        t < T && predict!(pf, u[t], y[t+1], t)
    end
    x,w,we,ll
end



"""
    x,ll = mean_trajectory(pf, u::Vector{Vector}, y::Vector{Vector})

This Function resets the particle filter to the initial state distribution upon start
"""
mean_trajectory(pf, u::Vector, y::Vector) = reduce_trajectory(pf, u::Vector, y::Vector, weigthed_mean)
mode_trajectory(pf, u::Vector, y::Vector) = reduce_trajectory(pf, u::Vector, y::Vector, mode)

function reduce_trajectory(pf, u::Vector, y::Vector, f::F) where F
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(undef,T)
    ll = correct!(pf,u[1],y[1],1) |> first
    x[1] = f(state(pf))
    for t = 2:T
        ll += pf(u[t-1], y[t], t) |> first
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
    x,u,y = simulate(f::AbstractFilter,T::Int,du::Distribution, [N])
Simulate dynamical system forward in time, returns state sequence, inputs and measurements
`du` is a distribution of random inputs.

If [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl) is loaded, the argument `N::Int` can be supplied, in which case `N` simulations are done and the result is returned in the form of `Vector{MonteCarloMeasurements.Particles}`.
"""
function simulate(f::AbstractFilter,T::Int,du::Distribution)
    u = [rand(du) for t=1:T]
    simulate(f, u)
end

function simulate(f::AbstractFilter,u)
    y = similar(u)
    x = similar(u)
    x[1] = sample_state(f)
    T = length(u)
    for t = 1:T-1
        y[t] = sample_measurement(f,x[t], u[t], t)
        x[t+1] = sample_state(f, x[t], u[t], t)
    end
    y[T] = sample_measurement(f,x[T], u[T], T)
    x,u,y
end


"""
    x̂ = weigthed_mean(x,we)
Calculated weighted mean of particle trajectories. `we` are expweights.
"""
function weigthed_mean(x,we::AbstractVector)
    @assert sum(we) ≈ 1
    xh = zeros(size(x[1]))
    @inbounds @simd  for i = eachindex(x)
        xh .+= x[i].*we[i]
    end
    return xh
end
function weigthed_mean(x,we::AbstractMatrix)
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
    x̂ = weigthed_mean(pf)
    x̂ = weigthed_mean(s::PFstate)
"""
weigthed_mean(s) = weigthed_mean(s.x,s.we)
weigthed_mean(pf::AbstractParticleFilter) = weigthed_mean(state(pf))
weigthed_mean(s::SigmaFilter) = weigthed_mean(s.x,s.we)
"""
Similar to [`weigthed_mean`](@ref), but returns covariances
"""
function weigthed_cov(x,we)
    N,T = size(x)
    n = length(x[1])
    [cov(copy(reshape(reinterpret(Float64, x[:,t]),n,N)),ProbabilityWeights(we[:,t]), dims=2, corrected=true) for t = 1:T]
end
