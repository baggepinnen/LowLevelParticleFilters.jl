# Fixed-interval smoothing for the IMM filter
#
# Two-filter IMM smoother following
#   R. E. Helmick, W. D. Blair and S. A. Hoffman, "Fixed-interval smoothing for Markovian
#   switching systems," IEEE Transactions on Information Theory, vol. 41, no. 6,
#   pp. 1845-1855, Nov. 1995, doi: 10.1109/18.476310.
#
# Notation (time index k, modes i, j):
#   M_k^j        Mode j is active on the interval (k-1, k], i.e., model j propagates x(k-1) -> x(k).
#                This matches the forward pass of `IMM`, where the model that is corrected at time k
#                is the one that was used in the prediction from k-1 to k.
#   P[i, j]      P(M_{k+1}^j | M_k^i), the mode transition probability matrix
#   π_k[j]       P(M_k^j), the prior mode probabilities, π_1 = P'μ0, π_{k+1} = P'π_k
#   x^j(k|k)     Forward filtered estimate conditioned on M_k^j
#   x̂^{b,i}(k)   Backward one-step prediction of x(k) given y(k+1:T), conditioned on M_{k+1}^i

"""
    ssol = smooth(imm::IMM, u, y, p = parameters(imm); t, interact, backward_init = :diffuse, diffuse_scale = 1e8)
    ssol = smooth(sol::KalmanFilteringSolution, imm::IMM, u = sol.u, y = sol.y, p = parameters(imm); kwargs...)

Fixed-interval smoothing for an [`IMM`](@ref) filter using the two-filter approach of Helmick, Blair and Hoffman (1995). A forward-time IMM filter and a backward-time IMM filter are run over the data, and the smoothed estimate at time `k` is formed by fusing the ``n_m^2`` combinations of forward filtered model-conditioned estimates (conditioned on the mode active on the interval ``(k-1, k]``) and backward predicted model-conditioned estimates (conditioned on the mode active on ``(k, k+1]``), followed by moment matching.

Returns a [`KalmanSmoothingSolution`](@ref) with smoothed (combined) states `xT` and covariances `RT`. The smoothed mode probabilities are available in the field `extraT`, a matrix of size `(n_modes, T)`. The filtered mode probabilities are, like for [`forward_trajectory`](@ref), available in `ssol.extra`.

When called with a filtering solution `sol`, the forward pass is recomputed since the smoother requires the model-conditioned estimates of each individual model, which are not stored by [`forward_trajectory`](@ref).

# Supported models
The backward filter propagates the model-conditioned estimates backwards in time through the inverse of the dynamics, the dynamics of each model must thus be invertible (which is always the case for dynamics obtained by discretizing a continuous-time system). The following model types are supported in the IMM:
- [`KalmanFilter`](@ref): The backward prediction is exact.
- [`ExtendedKalmanFilter`](@ref) (including [`IteratedExtendedKalmanFilter`](@ref)): The dynamics are inverted using Newton's method and linearized around the inverted point.
- [`UnscentedKalmanFilter`](@ref) with additive dynamics noise (`AUGD = false`): The backward prediction is performed using linearization (ForwardDiff) like for the EKF, while the measurement update uses the unscented transform.

The forward pass applies the forgetting factor `α` of the individual filters, while the backward pass does not.

# Keyword arguments
- `t`: Time vector, defaults to `range(0, step=imm.Ts, length=length(y))` (or `sol.t`).
- `interact`: Whether or not to run the interaction step in the forward pass. The smoother assumes that the forward filter is an IMM filter, i.e., `interact = true`.
- `backward_init`: How to initialize the backward filter at the final time step `T`.
    - `:diffuse` (default): The backward filter of each mode is initialized with the forward filtered mean and a covariance equal to `diffuse_scale` times the forward filtered covariance, and is then corrected with the measurement `y[T]`. This makes the backward filter approximate the likelihood ``p(y_{k+1:T} | x_k)`` rather than a posterior, which avoids double counting the information in `y[1:T]`. With this choice, the smoother reduces to the Rauch-Tung-Striebel smoother when the IMM contains a single linear model.
    - `:forward`: The backward filter is initialized with the forward filtered model-conditioned estimates at time `T`, as in the description of the algorithm in the EKF/UKF toolbox for Matlab (Hartikainen, Solin & Särkkä). This double counts the information in the forward filtered estimates at the end of the interval, leading to overconfident covariance estimates there.
- `diffuse_scale`: See `backward_init`.

The backward filter is initialized with the forward filtered mode probabilities at time `T`, and the backward mode transition probabilities are computed from the prior mode probabilities as in the reference.

!!! note "Approximations"
    In addition to the moment-matching approximations made by the forward and backward IMM filters, the smoothed mode probabilities are computed using the approximation of Helmick et al. where the future measurements ``y_{k+1:T}`` are replaced by the backward-predicted model-conditioned means and covariances, i.e., ``p(y_{k+1:T} | M_k^j, M_{k+1}^i, y_{1:k}) ≈ \\mathcal{N}(\\hat{x}^{b,i}_k - x^j_{k|k}; 0, \\hat{R}^{b,i}_k + R^j_{k|k})``.

# Example
```julia
imm  = IMM([kf1, kf2], P, μ)
ssol = smooth(imm, u, y)
ssol.xT      # Smoothed states
ssol.extraT  # Smoothed mode probabilities (n_modes × T)
ssol.extra   # Filtered mode probabilities (n_modes × T)
```
"""
function smooth(sol::KalmanFilteringSolution, imm::IMM, u::AbstractVector=sol.u, y::AbstractVector=sol.y, p=parameters(imm); t = sol.t, kwargs...)
    smooth(imm, u, y, p; t, kwargs...)
end

function smooth(imm::IMM, u::AbstractVector, y::AbstractVector, p=parameters(imm);
    t = range(0, step=imm.Ts, length=length(y)),
    interact = imm.interact,
    backward_init::Symbol = :diffuse,
    diffuse_scale::Real = 1e8,
)
    backward_init ∈ (:diffuse, :forward) || throw(ArgumentError("backward_init must be either :diffuse or :forward, got :$backward_init"))
    for model in imm.models
        _check_imm_smoothing_support(model)
    end
    T = length(y)
    T >= 1 || throw(ArgumentError("Cannot smooth an empty trajectory"))
    sol, fw = _forward_trajectory_imm(imm, u, y, p, t, interact, Val(true))
    (; xf, Rf, xm) = fw
    μf = sol.extra
    P  = imm.P
    nm = length(imm.models)
    FT = eltype(μf)

    # The backward filter is run on copies of the models so that the state of `imm` after
    # smoothing is the same as after a call to `forward_trajectory`.
    models = deepcopy(imm.models)

    # Prior mode probabilities π[:, k] = P(M_k), consistent with the forward pass where the first
    # correction step uses P'μ0 as the prior mode probabilities.
    π = zeros(FT, nm, T)
    π[:, 1] .= P'*imm.μ0
    for k = 2:T
        π[:, k] .= P'*π[:, k-1]
    end
    logP = log.(P)

    xT = similar(sol.xt)
    RT = similar(sol.Rt)
    μT = similar(μf)
    xT[T] = copy(sol.xt[T])
    RT[T] = copy(sol.Rt[T])
    μT[:, T] .= μf[:, T]

    # Initialization of the backward filter at time T
    xb = [copy(xf[j][T]) for j in 1:nm] # Backward filtered x^{b,j}(k|k:T), conditioned on M_k^j
    Rb = [copy(Rf[j][T]) for j in 1:nm]
    if backward_init === :diffuse
        for j in 1:nm
            models[j].x = copy(xf[j][T])
            models[j].R = symmetrize(diffuse_scale .* Rf[j][T])
            if !(y[T] === missing || ismissing_measurement(y[T]))
                correct!(models[j], u[T], y[T], p, t[T])
            end
            xb[j] = copy(models[j].x)
            Rb[j] = copy(models[j].R)
        end
    end
    μb = μf[:, T] # Backward mode probabilities P(M_k | y_{k:T})

    XT = eltype(xb)
    RTT = eltype(Rb)
    x̂b = Vector{XT}(undef, nm)          # Backward predictions x̂^{b,i}(k), conditioned on M_{k+1}^i
    R̂b = Vector{RTT}(undef, nm)
    xs_ji = Matrix{XT}(undef, nm, nm)   # Fused estimates conditioned on M_k^j and M_{k+1}^i
    Rs_ji = Matrix{RTT}(undef, nm, nm)
    logΛ  = zeros(FT, nm, nm)           # logΛ[j, i] ≈ log p(y_{k+1:T} | M_k^j, M_{k+1}^i, y_{1:k})
    logd  = zeros(FT, nm)
    logμs = zeros(FT, nm)
    logμb = zeros(FT, nm)
    w     = zeros(FT, nm)
    xs    = Vector{XT}(undef, nm)       # Model-conditioned smoothed estimates
    Rs    = Vector{RTT}(undef, nm)

    for k = T-1:-1:1
        # Backward prediction k+1 -> k through the inverse dynamics of each model
        for i in 1:nm
            x̂b[i], R̂b[i] = _imm_backward_predict(models[i], xb[i], Rb[i], xm[i][k], u[k], p, t[k])
        end

        # Fusion of forward filtered and backward predicted model-conditioned estimates
        for j in 1:nm, i in 1:nm
            xfj = xf[j][k]
            Rfj = Rf[j][k]
            D   = symmetrize(Rfj + R̂b[i])
            Dc  = cholesky(Symmetric(D))
            Δ   = x̂b[i] - xfj
            # The constant term -n/2*log(2π) is the same for all (i,j) and cancels in the normalizations below
            logΛ[j, i] = -(dot(Δ, Dc\Δ) + logdet(Dc))/2
            K = Rfj/Dc
            xs_ji[j, i] = xfj + K*Δ
            Rs_ji[j, i] = symmetrize(Rfj - K*Rfj) # = (Rf⁻¹ + R̂b⁻¹)⁻¹
        end

        # Smoothed mode probabilities μT[j, k] ∝ d_j μf[j, k], d_j = ∑_i P[j, i]Λ[j, i]
        for j in 1:nm
            logd[j] = _logsumexp(@views logP[j, :] .+ logΛ[j, :])
            logμs[j] = logd[j] + log(μf[j, k])
        end
        μT[:, k] .= exp.(logμs .- _logsumexp(logμs))

        # Model-conditioned smoothed estimates by moment matching over the mode at k+1, using the
        # conditional mode probabilities P(M_{k+1}^i | M_k^j, y_{1:T}) = P[j, i]Λ[j, i] / d_j
        for j in 1:nm
            if isfinite(logd[j])
                @views w .= exp.(logP[j, :] .+ logΛ[j, :] .- logd[j])
            else
                w .= P[j, :]
            end
            xs[j], Rs[j] = _moment_match(w, view(xs_ji, j, :), view(Rs_ji, j, :))
        end

        # Combined smoothed estimate
        xT[k], RT[k] = _moment_match(view(μT, :, k), xs, Rs)

        k == 1 && break

        # Backward IMM filter step, results in the backward filtered estimates at time k
        for j in 1:nm
            # Backward mode transition probabilities pb[i, j] = P(M_k^j | M_{k+1}^i) = P[j, i]π_k[j] / π_{k+1}[i]
            for i in 1:nm
                pbij = iszero(π[i, k+1]) ? zero(FT) : P[j, i]*π[j, k] / π[i, k+1]
                w[i] = pbij * μb[i]
            end
            aj = sum(w)
            if iszero(aj) # Mode j is not reachable from any mode with nonzero probability, let it evolve on its own
                x0, R0 = x̂b[j], R̂b[j]
                logμb[j] = FT(-Inf)
            else
                w ./= aj
                x0, R0 = _moment_match(w, x̂b, R̂b)
                logμb[j] = log(aj)
            end
            models[j].x = x0
            models[j].R = R0
            if !(y[k] === missing || ismissing_measurement(y[k]))
                ll, _ = correct!(models[j], u[k], y[k], p, t[k])
                logμb[j] += ll
            end
            xb[j] = copy(models[j].x)
            Rb[j] = copy(models[j].R)
        end
        μb = exp.(logμb .- _logsumexp(logμb))
    end
    KalmanSmoothingSolution(sol, xT, RT, μT)
end

function _logsumexp(v)
    m = maximum(v)
    isfinite(m) || return m
    m + log(sum(x -> exp(x - m), v))
end

# Moment matching of a Gaussian mixture with weights w, means xs and covariances Rs
function _moment_match(w, xs, Rs)
    x = zero(xs[1])
    for i in eachindex(w, xs)
        iszero(w[i]) && continue
        x = x + w[i]*xs[i]
    end
    R = zero(Rs[1])
    for i in eachindex(w, xs, Rs)
        iszero(w[i]) && continue
        d = xs[i] - x
        R = R + w[i]*(Rs[i] + d*d')
    end
    x, symmetrize(R)
end

const _IMMSmoothableNonlinear = Union{ExtendedKalmanFilter, UnscentedKalmanFilter{<:Any, <:Any, false}}

_check_imm_smoothing_support(::Union{KalmanFilter, _IMMSmoothableNonlinear}) = nothing
_check_imm_smoothing_support(model) = throw(ArgumentError("IMM smoothing is not supported for models of type $(nameof(typeof(model))). Supported model types are KalmanFilter, ExtendedKalmanFilter (including IteratedExtendedKalmanFilter) and UnscentedKalmanFilter with additive dynamics noise (AUGD = false)."))

"""
    x̂, R̂ = _imm_backward_predict(model, x, R, xlin, u, p, t)

Propagate the estimate `x, R` of the state at time `k+1` backwards in time to time `k` through the inverse of the dynamics of `model`, ``x_k = f^{-1}(x_{k+1} - w_k)`` with ``w_k ∼ N(0, R_1)``, i.e., the covariance is ``A^{-1}(R + R_1)A^{-T}``. `xlin` is the point at which time and state-dependent matrices are evaluated (the state the forward pass propagated from), and the initial guess for the inversion of nonlinear dynamics.
"""
function _imm_backward_predict(kf::KalmanFilter, x, R, xlin, u, p, t)
    A  = get_mat(kf.A, xlin, u, p, t)
    R1 = get_mat(kf.R1, xlin, u, p, t)
    xp = (u === nothing || length(u) == 0) ? x : x .- get_mat(kf.B, xlin, u, p, t)*u
    F  = lu(A)
    x̂  = F \ xp
    x̂, _inv_sandwich(F, R + R1)
end

function _imm_backward_predict(kf::_IMMSmoothableNonlinear, x, R, xlin, u, p, t; maxiters = 20)
    f  = _oop_dynamics(kf)
    R1 = get_mat(kf.R1, xlin, u, p, t)
    # Solve f(x̂, u, p, t) = x for x̂ using Newton's method, initialized at xlin
    x̂ = xlin
    tol = sqrt(eps(float(eltype(x̂))))
    converged = false
    for _ in 1:maxiters
        r  = x - f(x̂, u, p, t)
        A  = _dynamics_jacobian(kf, x̂, u, p, t)
        δ  = A \ r
        x̂  = x̂ + δ
        if norm(δ) <= tol*(1 + norm(x̂))
            converged = true
            break
        end
    end
    converged || @warn "Inversion of the dynamics did not converge in the backward pass of the IMM smoother at t = $t" maxlog=10
    A = _dynamics_jacobian(kf, x̂, u, p, t)
    x̂, _inv_sandwich(lu(A), R + R1)
end

# Computes A⁻¹ M A⁻ᵀ given F = lu(A) and a symmetric matrix M
function _inv_sandwich(F, M)
    AiM = F \ M         # A⁻¹M
    symmetrize(F \ AiM') # A⁻¹(A⁻¹M)ᵀ = A⁻¹MA⁻ᵀ since M is symmetric
end

function _oop_dynamics(kf::Union{ExtendedKalmanFilter{IPD}, UnscentedKalmanFilter{IPD}}) where IPD
    f = kf.dynamics
    if IPD
        return function (x, u, p, t)
            xp = similar(x)
            f(xp, x, u, p, t)
            xp
        end
    else
        return f
    end
end

_dynamics_jacobian(kf::ExtendedKalmanFilter, x, u, p, t) = copy(kf.Ajac(x, u, p, t))

function _dynamics_jacobian(kf::UnscentedKalmanFilter, x, u, p, t)
    f = _oop_dynamics(kf)
    ForwardDiff.jacobian(x -> f(x, u, p, t), x)
end
