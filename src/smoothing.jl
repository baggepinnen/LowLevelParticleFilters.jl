
"""
    sol = smooth(filtersol)
    sol = smooth(kf::AbstractKalmanFilter, u::Vector, y::Vector, p=parameters(kf))

Returns a [`KalmanSmoothingSolution`](@ref) with smoothed estimates of state `xT` and covariance `RT` given all input output data `u,y` or an existing filtering solution `filtersol` obtained from [`forward_trajectory`](@ref).

The return smoothing can be plotted using `plot(sol)`, see [`KalmanSmoothingSolution`](@ref) and [`KalmanFilteringSolution`](@ref) for details.
"""
function smooth(sol::KalmanFilteringSolution, kf::KalmanFilter, u::AbstractVector=sol.u, y::AbstractVector=sol.y,  p=parameters(kf))
    (; x,xt,R,Rt,ll) = sol
    T            = length(y)
    xT           = similar(xt)
    RT           = similar(Rt)
    xT[end]      = xt[end]      |> copy
    RT[end]      = Rt[end]      |> copy
    for t = T-1:-1:1
        C     = Rt[t]*get_mat(kf.A, xT[t+1], u[t+1], p, (t+1-1)*kf.Ts)'/cholesky(Symmetric(R[t+1]))
        xT[t] = xt[t] .+ C*(xT[t+1] .- x[t+1])
        RT[t] = Rt[t] .+ symmetrize(C*(RT[t+1] .- R[t+1])*C')
    end
    KalmanSmoothingSolution(sol, xT, RT)
end

smooth(sol::KalmanFilteringSolution) = smooth(sol, sol.f)

function smooth(kf::KalmanFilter, args...)
    reset!(kf)
    sol = forward_trajectory(kf, args...)
    smooth(sol, kf, args...)
end

# This smoother appears to have issues when there are missing measurements. It also requires more information to be stored from the forward pass, K and S. The benefit of this implementation is that it does not invert the state covariance matrix, instead, it inverts the residual covariance. Pick the smoother that inverts the smallest matrix.
"""
    xT,RT,ll,λ̃,λ̂,r = smooth_mbf(sol, kf)

Implements the "modified Bryson-Frazier smoother" which is a variant of the Rauch-Tung-Striebel smoother used in [`smooth`](@ref) that does not require the inversion of the state covariance matrix. The smoother is described in "New Kalman filter and smoother consistency tests" by Gibbs.
"""
function smooth_mbf(sol::KalmanFilteringSolution, kf::AbstractKalmanFilter, u::AbstractVector=sol.u, y::AbstractVector=sol.y,  p=parameters(kf))
    (; x,xt,R,Rt,ll) = sol
    T            = length(y)
    xT           = similar(xt)
    RT           = similar(Rt)
    xT[end]      = xt[end]      |> copy
    RT[end]      = Rt[end]      |> copy
    λ̃           = similar(xt)
    λ̂           = similar(xt)
    Λ̃           = similar(Rt)
    Λ̂           = similar(Rt) # Λ̂ is the covariance of λ̂
    Λ̂[end] = zero(Rt[end])
    λ̂[end] = zero(xt[end])
    r = similar(λ̂)
    for t = T:-1:1
        ti = ((t+1)-1)*kf.Ts
        # if t < T
        #     F = get_A(kf, xT[t+1], u[t+1], p, ti) 
        #     H = get_C(kf, xT[t+1], u[t+1], p, ti)
        # else
            F = get_A(kf, xt[t], u[t], p, ti) # NOTE: may be wrong state here, xT[t+1]?
            H = get_C(kf, xt[t], u[t], p, ti)
        # end
        if !isassigned(sol.K, t)
            xT[t] = xt[t]
            RT[t] = Rt[t]
            r[t] = zero(x[t])
            λ̂[t-1] = zero(xt[t])
            Λ̂[t-1] = zero(Rt[t])
            continue
        end

        K = sol.K[t]
        S = sol.S[t]
        C = I-K*H

        HTS = H'/S
        r[t] = C'λ̂[t]
        λ̃[t] = -HTS*sol.e[t] + C'λ̂[t] # Wikipedia wrong here, it should be residual instead of measurement
        Λ̃[t] = HTS*H + C'Λ̂[t]*C
        if t > 1
            λ̂[t-1] = F'*λ̃[t]
            Λ̂[t-1] = F'Λ̃[t]*F
        end
        
        xT[t] = xt[t] .- Rt[t]*λ̂[t]
        RT[t] = Rt[t] .- symmetrize(Rt[t]*Λ̂[t]*Rt[t])
        # if true
        #     # Project onto closest positive definite matrix
        #     eiv = eigen(RT[t])
        #     RT[t] = symmetrize(eiv.vectors * Diagonal(max.(eiv.values, 1e-3)) * eiv.vectors')
        # end

        # The alternative formulation below is bad when there are missing values in the measurement sequence
        # xT[t] = x[t] .- R[t]*λ̃[t]
        # RT[t] = R[t] .- symmetrize(R[t]*Λ̃[t]*R[t])
    end
    xT,RT,ll,λ̃,λ̂,r
end

get_A(kf::KalmanFilter, x, u, p, t) = kf.A
get_C(kf::KalmanFilter, x, u, p, t) = kf.C

function smooth(pf::AbstractParticleFilter, M, u, y, p=parameters(pf))
    sol = forward_trajectory(pf, u, y, p)
    smooth(pf::AbstractParticleFilter, sol.x, sol.w, sol.we, sol.ll, M, u, y, p)
end

"""
    xb,ll = smooth(pf, M, u, y, p=parameters(pf))
    xb,ll = smooth(pf, xf, wf, wef, ll, M, u, y, p=parameters(pf))

Perform particle smoothing using forward-filtering, backward simulation. Return smoothed particles and loglikelihood.
See also [`smoothed_trajs`](@ref), [`smoothed_mean`](@ref), [`smoothed_cov`](@ref)
"""
function smooth(pf::AbstractParticleFilter, xf, wf, wef, ll, M, u, y, p=parameters(pf))
    T = length(y)
    N = num_particles(pf)
    f = dynamics(pf)
    df = dynamics_density(pf)
    @assert M <= N "Must extend cache size of bins and j to allow this"
    xb = Array{particletype(pf)}(undef,M,T)
    j = resample(ResampleSystematic, wef[:,T], M)
    # @show Set(j)
    for i = 1:M
        xb[i,T] = xf[j[i], T]
    end
    wb = Vector{Float64}(undef,N)
    @inbounds for t = T-1:-1:1
        ti = (t-1)*pf.Ts
        # tset = Set{Int}()
        for m = 1:M
            for n = 1:N
                wb[n] = wf[n,t] + extended_logpdf(df, xb[m,t+1], f(xf[n,t],u[t],p,ti), ti)
            end
            i = draw_one_categorical(pf,wb)
            # push!(tset, i)
            xb[m,t] = xf[i, t]
        end
        # @show tset
    end
    return xb,ll
end

"""
    sse(f::AbstractFilter, u, y, p = parameters(pf), λ = 1; post_update_cb=(f, u, y, p, ll, e)->nothing)

Calculate the sum of squared errors ``\\sum dot(e, λ, e)``.
- `λ`: May be a weighting matrix. A commonly used metric is `λ = Diagonal(1 ./ (mag.^2))`, where `mag` is a vector of the "typical magnitude" of each output.

See also [`LowLevelParticleFilters.prediction_errors!`](@ref) which returns the prediction errors themselves rather than their sum of squares (for use with Gauss-Newton style optimization).
"""
function sse(f::AbstractFilter, u, y, p=parameters(f), λ=1; post_update_cb=(args...)->nothing)
    reset!(f)
    sum(zip(u, y)) do (u,y)
        ll, e = f(u,y,p)
        post_update_cb(f, u, y, p, ll, e)
        dot(e, λ, e)
    end
end

"""
    prediction_errors!(res, f::AbstractFilter, u, y, p = parameters(f), λ = 1)

Calculate the prediction errors and store the result in `res`. Similar to [`sse`](@ref), this function is useful for sum-of-squares optimization. In contrast to `sse`, this function returns the residuals themselves rather than their sum of squares. This is useful for Gauss-Newton style optimizers, such as [LeastSquaresOptim.LevenbergMarquardt](https://github.com/matthieugomez/LeastSquaresOptim.jl).

# Arguments:
- `res`: A vector of length `ny*length(y)`. Note, for each datapoint in `u` and `u`, there are `ny` outputs, and thus `ny` residuals.
- `f`: Any filter
- `λ`: A weighting factor to minimize `dot(e, λ, e`). A commonly used metric is `λ = Diagonal(1 ./ (mag.^2))`, where `mag` is a vector of the "typical magnitude" of each output. Internally, the square root of `W = sqrt(λ)` is calculated so that the residuals stored in `res` are `W*e`.

See example in [Solving using Gauss-Newton optimization](@ref).
"""
function prediction_errors!(res, f::AbstractFilter, u, y, p=parameters(f), λ=1)
    reset!(f)
    W = sqrt(λ)
    ny = f.ny
    length(u)*ny == length(y)*ny == length(res) || error("The residual vector must be of length ny*length(data) (there is one residual per output for each datapoint)")
    inds = 1:ny
    for (i, (u,y)) in enumerate(zip(u, y))
        ll, e = f(u,y,p)
        @views mul!(res[inds], W, e)
        inds = inds .+ ny
    end
    res
end


"""
    ll = loglik(filter, u, y, p=parameters(filter))

Calculate log-likelihood for entire sequences `u,y`.


See also [`loglik_x`](@ref) for Kalman-type filters when an accurate state sequence `x` is available.
"""
function loglik(f::AbstractFilter,u,y,p=parameters(f); kwargs...)
    reset!(f)
    sum(x->f(x[1],x[2],p; kwargs...)[1], zip(u, y))
end

function loglik(pf::AuxiliaryParticleFilter,u,y,p=parameters(pf))
    reset!(pf)
    ll = sum(t->pf(u[t],y[t],y[t+1],p,(t-1)*pf.Ts)[1], 1:length(u)-1)
    ll + pf.pf(u[end],y[end], p, (length(u)-1)*pf.Ts)[1]
end

"""
    ll = loglik_x(kf, u, y, x, p=parameters(kf))

For Kalman-type filters when an accurate state sequence `x` is available, such as when data is obtained from a simulation or in a lab setting, the log-likelihood can be calculated using the state prediction errors rather than the output prediction errors. In this case, `logpdf(f.R, x-x̂)` is used rather than `logpdf(S, y-ŷ)`.
"""
function loglik_x(f::AbstractKalmanFilter,u,y,x::AbstractVector,p=parameters(f); kwargs...)
    length(u) == length(y) == length(x) || throw(ArgumentError("u, y, and x must have the same length"))
    reset!(f)
    sum(1:length(u)-1) do i
        ui,yi,xi = u[i],y[i],x[i]
        xh = f.x
        xe = xi .- xh
        # The paper https://liu.diva-portal.org/smash/get/diva2:1641373/FULLTEXT01.pdf suggests performing the correct step before calculating the logpdf, but the example https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable/parameter_estimation/#Maximum-likelihood-estimation where the data is simulated with dynamics_noise=false
        # xs,u,y = simulate(pf,300,df, dynamics_noise=false)
        # suggests that one shall use the prediction errors rather than the filtering errors. With prediciton errors, ll increases as s gets smaller and decreases as s gets bigger. With filtering errors, ll plateaus for large s which is not what we want
        correct!(f,ui,yi,p; kwargs...)
        predict!(f,ui,p; kwargs...)
        ll = extended_logpdf(SimpleMvNormal(f.R), xe)
        ll
    end
end

"""
    ll(θ) = log_likelihood_fun(filter_from_parameters(θ::Vector)::Function, priors::Vector{Distribution}, u, y, p)
    ll(θ) = log_likelihood_fun(filter_from_parameters(θ::Vector)::Function, priors::Vector{Distribution}, u, y, x, p)

returns function θ -> p(y|θ)p(θ)
"""
function log_likelihood_fun(filter_from_parameters,priors::AbstractVector,args...)
    n = numargs(filter_from_parameters)
    pf = nothing
    function (θ)
        pf === nothing && (pf = filter_from_parameters(θ))
        length(θ) == length(priors) || throw(ArgumentError("Input must have same length as priors"))
        ll = sum(i->extended_logpdf(priors[i], θ[i]), eachindex(priors))
        isfinite(ll) || return eltype(θ)(-Inf)
        pf = filter_from_parameters(θ,pf)
        try
            return ll + loglik(pf,args...)
        catch
            return eltype(θ)(-Inf)
        end

    end
end

function naive_sampler(θ₀)
    any(iszero,θ₀) && throw(ArgumentError("Naive sampler does not work if initial parameter vector contains zeros (it was going to return θ -> θ .+ rand(MvNormal(0.1abs.(θ₀))), but that is not a good idea if θ₀ is zero."))
    θ -> θ .+ rand(SimpleMvNormal(0*θ₀, Diagonal(0.1abs.(θ₀))))
end

"""
    metropolis(ll::Function(θ), R::Int, θ₀::Vector, draw::Function(θ) = naive_sampler(θ₀))

Performs MCMC sampling using the marginal Metropolis (-Hastings) algorithm
`draw = θ -> θ'` samples a new parameter vector given an old parameter vector. The distribution must be symmetric, e.g., a Gaussian. `R` is the number of iterations.
See `log_likelihood_fun`

# Example:
```julia
filter_from_parameters(θ) = ParticleFilter(N, dynamics, measurement, MvNormal(n,exp(θ[1])), MvNormal(p,exp(θ[2])), d0)
priors = [Normal(0,0.1),Normal(0,0.1)]
ll     = log_likelihood_fun(filter_from_parameters,priors,u,y,1)
θ₀ = log.([1.,1.]) # Initial point
draw = θ -> θ .+ rand(MvNormal(0.1ones(2))) # Function that proposes new parameters (has to be symmetric)
burnin = 200 # If using threaded call, provide number of burnin iterations
# @time theta, lls = metropolis(ll, 2000, θ₀, draw) # Run single threaded
# thetam = reduce(hcat, theta)'
@time thetalls = LowLevelParticleFilters.metropolis_threaded(burnin, ll, 5000, θ₀, draw) # run on all threads, will provide (2000-burnin)*nthreads() samples
histogram(exp.(thetalls[:,1:2]), layout=3)
plot!(thetalls[:,3], subplot=3) # if threaded call, log likelihoods are in the last column
```
"""
function metropolis(ll, R, θ₀, draw = naive_sampler(θ₀))
    params    = Vector{typeof(θ₀)}(undef,R)
    lls       = Vector{Float64}(undef,R)
    params[1] = θ₀
    lls[1]    = ll(θ₀)
    for i = 2:R
        θ = draw(params[i-1])
        lli = ll(θ)
        if rand() < exp(lli-lls[i-1])
            params[i] = θ
            lls[i] = lli
        else
            params[i] = params[i-1]
            lls[i] = lls[i-1]
        end
    end
    params, lls
end

"""
    metropolis_threaded(burnin, args...; nthreads=Threads.nthreads())

Run `Threads.nthreads()` individual Markov chains. `args...` are the same as for [`metropolis`](@ref).
"""
function metropolis_threaded(burnin, args...; nthreads=Threads.nthreads())
    res = []
    mtx = ReentrantLock()
    Threads.@threads for i = 1:nthreads
        p,l = metropolis(args...)
        resi = [reduce(hcat,p)' l]
        resi = resi[burnin+1:end,:]
        lock(mtx)
        push!(res, resi)
        unlock(mtx)
    end
    reduce(vcat,res)
end



"""
    smoothed_mean(xb)

Helper function to calculate the mean of smoothed particle trajectories
"""
function smoothed_mean(xb)
    M,T = size(xb)
    n = length(xb[1])
    xbm = vec(mean(xb,dims=1))
    reduce(hcat, xbm)
end

"""
    smoothed_cov(xb)

Helper function to calculate the covariance of smoothed particle trajectories
"""
function smoothed_cov(xb)
    M,T = size(xb)
    n = length(xb[1])
    xbc = [cov(copy(reshape(reinterpret(Float64, xb[:,t]),n,M)),dims=2) for t = 1:T]
end

"""
    smoothed_trajs(xb)

Helper function to get particle trajectories as a 3-dimensions array (N,M,T) instead of matrix of vectors.
"""
function smoothed_trajs(xb)
    M,T = size(xb)
    n = length(xb[1])
    copy(reshape(reinterpret(Float64, xb), n,M,T))
end
