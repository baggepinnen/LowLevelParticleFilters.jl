function reset!(kf::AbstractKalmanFilter)
    kf.x .= Vector(kf.d0.μ)
    kf.R .= copy(Matrix(kf.d0.Σ))
    kf.t[] = 1
end

"""
    Reset the filter to initial state and covariance/distribution
"""
function reset!(pf::AbstractParticleFilter)
    s = pf.state
    for i = eachindex(s.xprev)
        s.xprev[i] = rand(pf.rng, pf.initial_density)
        s.x[i] = copy(s.xprev[i])
    end
    fill!(s.w, -log(num_particles(pf)))
    fill!(s.we, 1/num_particles(pf))
    s.t[] = 1
end

function predict!(kf::AbstractKalmanFilter, u, t = index(kf))
    @unpack A,B,x,R,R1 = kf
    if ndims(A) == 3
        At = A[:,:,t]
        Bt = B[:,:,t]
    else
        At = A
        Bt = B
    end
    x .= At*x .+ Bt*u
    R .= At*R*At' + R1
    kf.t[] += 1
end

function correct!(kf::AbstractKalmanFilter, y, t = index(kf))
    @unpack C,x,R,R1,R2,R2d = kf
    if ndims(C) == 3
        Ct = C[:,:,t]
    else
        Ct = C
    end
    e   = y .- Ct*x
    F   = Ct*R*Ct'
    F   = 0.5(F+F')
    K   = (R*Ct')/(F + R2) # Do not use .+ if R2 is I
    x .+= K*e
    R  .= (I - K*Ct)*R # warning against I .- A
    logpdf(MvNormal(F), e) - 1/2*logdet(F)
end

"""
    predict!(f,u, t = index(f))
Move filter state forward in time using dynamics equation and input vector `u`.
"""
function predict!(pf,u, t = index(pf))
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
 ll = correct!(f, y, t = index(f))
Update state/covariance/weights based on measurement `y`,  returns loglikelihood.
"""
function correct!(pf, y, t = index(pf))
    measurement_equation!(pf, y, t)
    loklik = logsumexp!(pf.state)
end

"""
ll = update!(f::AbstractFilter, u, y, t = index(f))
Perform one step of `predict!` and `correct!`, returns loglikelihood.
"""
function update!(f::AbstractFilter, u, y, t = index(f))
    predict!(f, u, t)
    loklik = correct!(f, y, t)
end

function update!(pf::AuxiliaryParticleFilter,u, y, t = index(pf))
    s = state(pf)
    N = num_particles(s)
    propagate_particles!(pf.pf, u, t, nothing)# Propagate without noise
    λ  = s.we
    λ .= 0
    measurement_equation!(pf.pf, y, t, measurement_density(pf), λ)
    s.w .+= λ
    expnormalize!(s.bins,s.w)
    if effective_particles(s.bins) < resample_threshold(pf)*N
        j = resample(ResampleSystematicExp, s.w , s.j, s.bins)
        fill!(s.w, -log(N))
        s.w .-= λ[j]
        s.x .= s.x[j] # TODO: these lines allocate
    else
        s.j .= 1:N
        s.w .-= λ
    end
    add_noise!(pf.pf)
    # s.w .= s.w[j] # TODO: these lines allocate
    copyto!(s.xprev, s.x)
    s.t[] += 1

    # Correct step
    measurement_equation!(pf.pf, y, t)
    loklik = logsumexp!(s)
end


(kf::KalmanFilter)(u, y, t = index(kf)) =  update!(kf, u, y, t)
(pf::ParticleFilter)(u, y, t = index(pf)) =  update!(pf, u, y, t)
(pf::AuxiliaryParticleFilter)(u, y, t = index(pf)) =  update!(pf, u, y, t)
(pf::AdvancedParticleFilter)(u, y, t = index(pf)) =  update!(pf, u, y, t)



"""
x,xt,R,Rt,ll = forward_trajectory(kf, u::Vector{Vector}, y::Vector{Vector})
x,w,ll       = forward_trajectory(pf, u::Vector{Vector}, y::Vector{Vector})

This Function resets the filter to the initial state distribution upon start
"""
function forward_trajectory(kf::AbstractKalmanFilter, u::Vector, y::Vector)
    reset!(kf)
    T     = length(y)
    x     = Array{particletype(kf)}(undef,T)
    xt    = Array{particletype(kf)}(undef,T)
    R     = Array{covtype(kf)}(undef,T)
    Rt    = Array{covtype(kf)}(undef,T)
    x[1]  = state(kf)       |> copy
    R[1]  = covariance(kf)  |> copy
    ll    = correct!(kf, y[1], 1)
    xt[1] = state(kf)       |> copy
    Rt[1] = covariance(kf)  |> copy
    for t = 2:T
        predict!(kf, u[t-1], t-1)
        x[t]   = state(kf)              |> copy
        R[t]   = covariance(kf)         |> copy
        ll    += correct!(kf, y[t], t)
        xt[t]  = state(kf)              |> copy
        Rt[t]  = covariance(kf)         |> copy
    end
    x,xt,R,Rt,ll
end


"""
    x,w,we,ll = forward_trajectory(pf, u::AbstractVector, y::AbstractVector)
Run the particle filter for a sequence of inputs and measurements. Return particles, weights, expweights and loglikelihood
"""
function forward_trajectory(pf, u::AbstractVector, y::AbstractVector)
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(undef,N,T)
    w = Array{Float64}(undef,N,T)
    we = Array{Float64}(undef,N,T)
    ll = 0.0
    @inbounds for t = 1:T
        ll += pf(u[t], y[t], t)
        x[:,t] .= particles(pf)
        w[:,t] .= weights(pf)
        we[:,t] .= expweights(pf)
    end
    x,w,we,ll
end



"""
x,ll = mean_trajectory(pf, u::Vector{Vector}, y::Vector{Vector})

This Function resets the particle filter to the initial state distribution upon start
"""
function mean_trajectory(pf, u::Vector, y::Vector)
    reset!(pf)
    T = length(y)
    N = num_particles(pf)
    x = Array{particletype(pf)}(undef,T)
    ll = 0.0
    for t = 1:T
        ll += pf(u[t], y[t], t)
        x[t] = weigthed_mean(pf)
    end
    x,ll
end

function mean_trajectory(x::AbstractMatrix, we::AbstractMatrix)
    copy(reduce(hcat,vec(sum(x.*we,dims=1)))')
end


"""
    x,u,y = simulate(f::AbstractFilter,T::Int,du::Distribution)
Simulate dynamical system forward in time, returns state sequence, inputs and measurements
`du` is a distribution of random inputs
"""
function simulate(f::AbstractFilter,T::Int,du::Distribution)
    u = [rand(du) for t=1:T]
    y = Vector{Vector{Float64}}(undef,T)
    x = Vector{Vector{Float64}}(undef,T)
    x[1] = sample_state(f)
    for t = 1:T-1
        y[t] = sample_measurement(f,x[t], t)
        x[t+1] = sample_state(f, x[t], u[t], t)
    end
    y[T] = sample_measurement(f,x[T], T)
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
    @assert sum(we) ≈ 1
    N,T = size(x)
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
weigthed_mean(pf::AbstractParticleFilter) = weigthed_mean(pf.state)
"""
    Similar to `weigthed_mean`, but returns covariances
"""
function weigthed_cov(x,we)
    N,T = size(x)
    n = length(x[1])
    [cov(copy(reshape(reinterpret(Float64, x[:,t]),n,N)),ProbabilityWeights(we[:,t]), dims=2, corrected=true) for t = 1:T]
end
