"""
    MUKF(; dynamics, nl_measurement_model::RBMeasurementModel, An, kf::KalmanFilter, R1n, d0n, nu=kf.nu, Ts=1.0, p=NullParameters(), weight_params=MerweParams(), names=default_names(length(d0n.μ) + length(kf.d0.μ), nu, kf.ny, "MUKF"))

Marginalized Unscented Kalman Filter for mixed linear/nonlinear state-space models.

!!! warning "Experimental"
    This filter is currently considered experimental and the user interface may change in the future without respecting semantic versioning.

This filter combines the Unscented Kalman Filter (UKF) for the nonlinear substate with a bank of Kalman filters (one per sigma point) for the linear substate. This approach provides improved accuracy compared to linearization-based methods while remaining more efficient than a full particle filter. This filter is sometimes referred to as a Rao-Blackwellized Unscented Kalman Filter, similar to the [`RBPF`](@ref).

# Model structure
The filter assumes dynamics on the form:
```math
\\begin{aligned}
x_{t+1}^n &= f_n(x_t^n, u, p, t) + A_n x_t^l + w_t^n, \\quad &w_t^n \\sim \\mathcal{N}(0, R_1^n) \\\\
x_{t+1}^l &= A_l x_t^l + B_l u + w_t^l, \\quad &w_t^l \\sim \\mathcal{N}(0, R_1^l) \\\\
y_t &= g(x_t^n, u, p, t) + C_l x_t^l + e_t, \\quad &e_t \\sim \\mathcal{N}(0, R_2)
\\end{aligned}
```
where ``x^n`` is the nonlinear substate and ``x^l`` is the linear substate.

# Arguments
- `dynamics`: The nonlinear dynamics function ``f_n(x^n, u, p, t)``
- `nl_measurement_model`: An instance of [`RBMeasurementModel`](@ref) containing ``g`` and ``R_2``
- `An`: The coupling matrix from linear to nonlinear state (can be a matrix or function)
- `kf`: A [`KalmanFilter`](@ref) describing the linear substate dynamics (``A_l, B_l, C_l, R_1^l``)
- `R1n`: Process noise covariance for the nonlinear substate
- `d0n`: Initial distribution for the nonlinear substate (SimpleMvNormal)
- `nu`: Number of inputs (default: `kf.nu`)
- `Ts`: Sampling time (default: 1.0)
- `p`: Parameters (default: NullParameters())
- `weight_params`: Unscented transform parameters (default: MerweParams())
- `names`: Signal names for plotting

# Extended help
The MUKF uses sigma points to represent the distribution of the nonlinear substate.
For each sigma point, a separate Kalman filter tracks the linear substate conditioned
on that sigma point. This approach is particularly efficient when:
- The system has a large linear substate
- The nonlinear substate is low-dimensional
- Gaussian noise assumptions are appropriate

See also [`UnscentedKalmanFilter`](@ref), [`RBPF`](@ref), [`KalmanFilter`](@ref)

# References
Based on the Marginalized Unscented Kalman Filter described in the literature on
Rao-Blackwellized filtering techniques.
"""
mutable struct MUKF{IPD,IPM,DT,MMT,ANT,KFT,R1NT,D0NT,XNT,RNT,XLT,ΓT,RnlT,TS,P,WP,NT,SPC} <: AbstractKalmanFilter
    # Model functions and parameters
    dynamics::DT              # xⁿ_{k+1} = fₙ(xⁿ_k, u, p, t) + Aₙ xˡ_k + wⁿ_k
    nl_measurement_model::MMT # Nonlinear measurement model
    An::ANT                   # Coupling matrix (or function)
    kf::KFT                   # Template Kalman filter for linear substate

    # Noise covariances
    R1n::R1NT                 # Process noise covariance for nonlinear state
    d0n::D0NT                 # Initial distribution for nonlinear state

    # Sigma point cache
    sigma_point_cache::SPC    # Pre-allocated sigma point storage
    xn_sigma_points::Vector{XNT}  # Current sigma points for xn (for cross-covariance)

    # Current state estimates
    xn::XNT                   # Mean of nonlinear state
    Rn::RNT                   # Covariance of nonlinear state
    xl::Vector{XLT}           # Conditional mean of linear state per sigma point: E[xl|xn[i]]
    Γ::ΓT                     # Conditional covariance of linear state given nonlinear: Cov(xl|xn)
    Rnl::RnlT                 # Cross-covariance between nonlinear and linear states: Cov(xn, xl)

    # Metadata
    t::Int
    Ts::TS
    nu::Int
    ny::Int
    p::P
    weight_params::WP
    names::NT
end

function MUKF{IPD,IPM}(; dynamics, nl_measurement_model::RBMeasurementModel, An, kf::KalmanFilter,
              R1n, d0n, nu=kf.nu, Ts=1.0, p=NullParameters(),
              weight_params=MerweParams(),
              names=default_names(length(d0n.μ) + length(kf.d0.μ), nu, kf.ny, "MUKF")) where {IPD,IPM}

    nxn = length(d0n.μ)
    nxl = length(kf.d0.μ)
    ny  = kf.ny

    # Determine element type
    T = promote_type(eltype(d0n), eltype(kf.d0))

    # Number of sigma points (2*nxn + 1)
    L = nxn
    ns = 2*L + 1

    # Decide if we should use static arrays (like UKF does)
    static = !(IPD || L > 50)

    # Create sigma point cache
    sigma_point_cache = SigmaPointCache{T}(nxn, 0, nxn, L, static)

    # Initialize state estimates using type from d0 and kf
    xn0 = convert_x0_type(d0n.μ)
    Rn0 = convert_cov_type(R1n, d0n.Σ)
    xl0 = convert_x0_type(kf.d0.μ)
    Γ0 = convert_cov_type(kf.R1, kf.d0.Σ)  # Initial conditional covariance
    Rnl0 = zero(xn0) * zero(xl0)'  # Initial cross-covariance (zero for independent initial distributions)

    # Initialize conditional means per sigma point
    xl = [copy(xl0) for _ in 1:ns]

    # Initialize sigma points (will be updated in predict!/correct!)
    xn_sigma_points = [copy(xn0) for _ in 1:ns]
    sigmapoints!(xn_sigma_points, xn0, Rn0, weight_params)

    MUKF{IPD,IPM,typeof(dynamics), typeof(nl_measurement_model), typeof(An), typeof(kf),
         typeof(R1n), typeof(d0n), typeof(xn0), typeof(Rn0), typeof(xl0), typeof(Γ0), typeof(Rnl0),
         typeof(Ts), typeof(p), typeof(weight_params), typeof(names), typeof(sigma_point_cache)}(
        dynamics, nl_measurement_model, An, kf, R1n, d0n, sigma_point_cache, xn_sigma_points,
        xn0, Rn0, xl, Γ0, Rnl0,
        0, Ts, nu, ny, p, weight_params, names)
end

# Convenience constructor that infers IPD and IPM (default to false for now)
MUKF(args...; kwargs...) = MUKF{false,false}(args...; kwargs...)

# Convenience accessors
state(f::MUKF) = [f.xn; xl_mean(f)]
function covariance(f::MUKF)
    # Compute marginal covariances
    Rn = f.Rn
    Σxl = xl_cov(f)

    # Use stored cross-covariance
    Rnl = f.Rnl

    # Return full covariance matrix
    [Rn Rnl; Rnl' Σxl]
end
parameters(f::MUKF) = f.p
particletype(f::MUKF) = typeof([f.xn; f.xl[1]])
covtype(f::MUKF) = typeof(cat(f.Rn, f.Γ, dims=(1,2)))

# Simplified getproperty - only provide essential virtual properties
function Base.getproperty(f::MUKF, s::Symbol)
    s ∈ fieldnames(typeof(f)) && return getfield(f, s)
    if s === :x
        return state(f)
    elseif s === :R
        return covariance(f)
    elseif s === :nx
        return length(getfield(f, :xn)) + length(getfield(f, :xl)[1])
    else
        throw(ArgumentError("$(typeof(f)) has no property named $s"))
    end
end

"""
    xl_mean(f::MUKF)

Compute the marginal mean of the linear substate by averaging over sigma points.
"""
function xl_mean(f::MUKF)
    # Use the same weighted mean function as UKF (uses wm weights which sum to 1)
    return mean_with_weights(weighted_mean, f.xl, f.weight_params)
end

"""
    xl_cov(f::MUKF)

Compute the marginal covariance of the linear substate.

Using the Marginalized Unscented Transform (MUT), the linear substate follows a Gaussian
mixture: p(xl) ≈ Σᵢ wᵢ * N(xl | xl[i], Γ) where xl[i] are conditional means and Γ is
the conditional covariance (same for all sigma points).

The covariance of this mixture is:
Var[xl] = E[Var[xl|xn]] + Var[E[xl|xn]]
        = Γ + Σᵢ wᵢ * (xl[i] - μ)(xl[i] - μ)'
"""
function xl_cov(f::MUKF)
    W = UKFWeights(f.weight_params, length(f.xn))
    μ = xl_mean(f)

    # Start with conditional covariance
    Σ = f.Γ

    # Add variance of the conditional means
    @inbounds for i in eachindex(f.xl)
        w = i == 1 ? W.wm : W.wmi
        d = f.xl[i] .- μ
        Σ = Σ .+ w .* (d * d')
    end
    return Σ
end

"""
    xl_cross_cov(f::MUKF)

Compute the cross-covariance between the nonlinear state xn and linear state xl.

Based on Morelande & Moran (2007) "An Unscented Transformation for Conditionally
Linear Models", this computes:
Cov(xn, xl) = Σᵢ wᵢ * (xn[i] - μn)(xl[i] - μl)'

where xn[i] are the stored sigma points and xl[i] are the conditional means of xl given xn[i].
"""
function xl_cross_cov(f::MUKF)
    # Use the stored sigma points (updated in predict!/correct!)
    sp = f.xn_sigma_points

    W = UKFWeights(f.weight_params, length(f.xn))
    μn = f.xn
    μl = xl_mean(f)

    # Compute weighted cross-covariance using sigma points (use covariance weights wc, not mean weights wm)
    Rnl = zero(f.xn) * zero(f.xl[1])'
    @inbounds for i in eachindex(sp)
        w = i == 1 ? W.wc : W.wci
        dxn = sp[i] .- μn
        dxl = f.xl[i] .- μl
        Rnl = Rnl .+ w .* (dxn * dxl')
    end

    return Rnl
end

function reset!(f::MUKF, d0n=f.d0n, d0l=f.kf.d0)
    @bangbang f.xn .= d0n.μ
    @bangbang f.Rn .= d0n.Σ
    @bangbang f.Γ .= d0l.Σ
    @bangbang f.Rnl .= zero(f.xn) * zero(f.xl[1])'  # Reset cross-covariance to zero
    for i in eachindex(f.xl)
        @bangbang f.xl[i] .= d0l.μ
    end
    # Reinitialize sigma points
    sigmapoints!(f.xn_sigma_points, d0n.μ, d0n.Σ, f.weight_params)
    f.t = 0
    f
end

function predict!(f::MUKF{IPD}, u=zeros(f.nu), p=parameters(f), t::Real=index(f)*f.Ts) where IPD
    # Generate sigma points for nonlinear state using cache
    sp = f.sigma_point_cache.x0
    sigmapoints!(sp, f.xn, f.Rn, f.weight_params)

    # Get matrices
    Al = get_mat(f.kf.A, f.xn, u, p, t)
    Bl = get_mat(f.kf.B, f.xn, u, p, t)
    R1l = get_mat(f.kf.R1, f.xn, u, p, t)
    An = get_mat(f.An, f.xn, u, p, t)
    R1n_mat = f.R1n isa AbstractMatrix ? f.R1n : (f.R1n isa Function ? f.R1n(f.xn, u, p, t) : f.R1n.Σ)

    # Compute conditional means of linear state for each sigma point
    # xl_i = E[xl] + Rnl' * inv(Rn) * (xn_i - E[xn])
    xl_curr_mean = xl_mean(f)
    Rn_inv_Rnl = f.Rn \ f.Rnl
    xl_sigma = similar(f.xl)
    @inbounds for i in eachindex(sp)
        dxn = sp[i] .- f.xn
        xl_sigma[i] = xl_curr_mean .+ Rn_inv_Rnl' * dxn
    end

    # Propagate nonlinear state through dynamics using sigma points
    # Reuse cache for transformed points
    Xn_pred = f.sigma_point_cache.x1
    if IPD
        # In-place dynamics
        xp = similar(Xn_pred[1])
        @inbounds for i in eachindex(sp)
            xp .= 0
            f.dynamics(xp, sp[i], u, p, t)
            @bangbang xp .+= An * xl_sigma[i]
            Xn_pred[i] .= xp
        end
    else
        # Out-of-place dynamics
        @inbounds for i in eachindex(sp)
            Xn_pred[i] = f.dynamics(sp[i], u, p, t) .+ An * xl_sigma[i]
        end
    end

    xn_pred = mean_with_weights(weighted_mean, Xn_pred, f.weight_params)
    # MUT: Include coupling term An*Γ*An' in nonlinear state covariance (equation 36)
    Rn_pred = cov_with_weights(weighted_cov, Xn_pred, xn_pred, f.weight_params) .+ An * f.Γ * An' .+ R1n_mat

    # Propagate conditional means of linear substate (one per sigma point)
    Xl_pred = similar(f.xl)
    for i in eachindex(xl_sigma)
        Xl_pred[i] = Al * xl_sigma[i] .+ Bl * u
    end
    xl_pred = mean_with_weights(weighted_mean, Xl_pred, f.weight_params)

    # Propagate conditional covariance (MUT formula)
    Γ_pred = Al * f.Γ * Al' .+ R1l

    # Compute cross-covariance using sigma points
    W = UKFWeights(f.weight_params, length(f.xn))
    Rnl_pred = zero(f.xn) * zero(f.xl[1])'
    @inbounds for i in eachindex(Xn_pred)
        w = i == 1 ? W.wc : W.wci
        dxn = Xn_pred[i] .- xn_pred
        dxl = Xl_pred[i] .- xl_pred
        Rnl_pred = Rnl_pred .+ w .* (dxn * dxl')
    end
    # Add coupling term: An * Γ * Al' where Γ is conditional covariance
    # This creates cross-covariance from the conditional uncertainty propagation
    Rnl_pred = Rnl_pred .+ An * f.Γ * Al'

    # Update state estimates
    @bangbang f.xn .= xn_pred
    @bangbang f.Rn .= Rn_pred
    @bangbang f.Γ .= Γ_pred
    @bangbang f.Rnl .= Rnl_pred

    # Compute conditional means using cross-covariance: xl[i] = xl_pred + Rnl' * inv(Rn) * (xn[i] - xn_pred)
    # Generate new sigma points for updated distribution
    sigmapoints!(f.xn_sigma_points, xn_pred, Rn_pred, f.weight_params)
    Rn_inv_Rnl = Rn_pred \ Rnl_pred  # More stable than inv(Rn) * Rnl
    @inbounds for i in eachindex(f.xl)
        dxn = f.xn_sigma_points[i] .- xn_pred
        f.xl[i] = xl_pred .+ Rn_inv_Rnl' * dxn
    end

    f.t += 1
    return f
end

function correct!(f::MUKF{IPD,IPM}, u, y, p=parameters(f), t::Real=index(f)*f.Ts; R2=nothing) where {IPD,IPM}
    # Generate sigma points using cache
    sp = f.sigma_point_cache.x0
    sigmapoints!(sp, f.xn, f.Rn, f.weight_params)
    W = UKFWeights(f.weight_params, length(f.xn))

    g = f.nl_measurement_model.measurement
    Cl = get_mat(f.kf.C, f.xn, u, p, t)
    R2_mat = if R2 !== nothing
        R2
    else
        mm_R2 = f.nl_measurement_model.R2
        mm_R2 isa AbstractMatrix ? mm_R2 : (mm_R2 isa Function ? mm_R2(f.xn, u, p, t) : mm_R2.Σ)
    end

    # Compute conditional means of linear state for each sigma point
    # xl_i = E[xl] + Rnl' * inv(Rn) * (xn_i - E[xn])
    xl_curr_mean = xl_mean(f)
    Rn_inv_Rnl = f.Rn \ f.Rnl
    xl_sigma = similar(f.xl)
    @inbounds for i in eachindex(sp)
        dxn = sp[i] .- f.xn
        xl_sigma[i] = xl_curr_mean .+ Rn_inv_Rnl' * dxn
    end

    # Predicted measurements per sigma point: ŷᵢ = g(xⁿᵢ) + Cl xˡᵢ
    Y = Vector{typeof(y)}(undef, length(sp))

    if IPM
        # In-place measurement
        y_temp = similar(y)
        @inbounds for i in eachindex(sp)
            g(y_temp, sp[i], u, p, t)
            Y[i] = y_temp .+ Cl * xl_sigma[i]
        end
    else
        # Out-of-place measurement
        @inbounds for i in eachindex(sp)
            Y[i] = g(sp[i], u, p, t) .+ Cl * xl_sigma[i]
        end
    end

    yhat = mean_with_weights(weighted_mean, Y, f.weight_params)

    # Innovation covariance (MUT: add Γ term once, not summed over sigma points)
    Σy_extra = Cl * f.Γ * Cl'
    S = cov_with_weights(weighted_cov, Y, yhat, f.weight_params) .+ Σy_extra .+ R2_mat

    # Cross-covariance between nonlinear state and measurement
    Σny = zeros(length(f.xn), f.ny)
    @inbounds for i in eachindex(sp)
        w = i == 1 ? W.wc : W.wci
        dx = sp[i] .- f.xn
        dy = Y[i] .- yhat
        Σny .+= w .* (dx * dy')
    end

    # UKF-style update for nonlinear state
    Sᵪ = cholesky(Symmetric(S), check=false)
    if !issuccess(Sᵪ)
        error("Cholesky factorization of innovation covariance failed at time step $(f.t)")
    end

    innovation = y .- yhat

    # Compute Kalman gains
    Kn = Σny / Sᵪ

    # Cross-covariance between linear state and measurement
    Σly = zeros(length(f.xl[1]), f.ny)
    @inbounds for i in eachindex(sp)
        w = i == 1 ? W.wc : W.wci
        dx = xl_sigma[i] .- xl_curr_mean
        dy = Y[i] .- yhat
        Σly .+= w .* (dx * dy')
    end
    Kl = Σly / Sᵪ

    # Update each conditional mean using the Kalman gain and its individual innovation
    if IPM
        # In-place measurement
        y_temp = similar(y)
        @inbounds for i in eachindex(sp)
            g(y_temp, sp[i], u, p, t)
            innovation_i = y .- y_temp .- Cl * xl_sigma[i]
            f.xl[i] = xl_sigma[i] .+ Kl * innovation_i
        end
    else
        # Out-of-place measurement
        @inbounds for i in eachindex(sp)
            innovation_i = y .- g(sp[i], u, p, t) .- Cl * xl_sigma[i]
            f.xl[i] = xl_sigma[i] .+ Kl * innovation_i
        end
    end

    # Update state estimates
    @bangbang f.xn .= f.xn .+ Kn * innovation
    @bangbang f.Rn .= f.Rn .- Kn * S * Kn'
    symmetrize(f.Rn)

    # Update conditional covariance Γ using Kalman filter (MUT formula)
    @bangbang f.Γ .= f.Γ .- Kl * S * Kl'
    symmetrize(f.Γ)

    # Regenerate sigma points for updated nonlinear state
    sigmapoints!(f.xn_sigma_points, f.xn, f.Rn, f.weight_params)

    # Recompute Rnl from the updated xl[i] representation
    xl_m = xl_mean(f)
    Rnl_new = zero(f.xn) * zero(f.xl[1])'
    @inbounds for i in eachindex(f.xn_sigma_points)
        w = i == 1 ? W.wc : W.wci
        dxn = f.xn_sigma_points[i] .- f.xn
        dxl = f.xl[i] .- xl_m
        Rnl_new = Rnl_new .+ w .* (dxn * dxl')
    end
    @bangbang f.Rnl .= Rnl_new

    ll = extended_logpdf(SimpleMvNormal(PDMat(S, Sᵪ)), innovation)
    return (; ll, e=innovation, S, Sᵪ, K=Kn)
end

function update!(f::MUKF, u, y, p=parameters(f), t=index(f)*f.Ts)
    predict!(f, u, p, t-f.Ts)
    correct!(f, u, y, p, t)
end

# Make MUKF callable
(f::MUKF)(u, y, p=parameters(f), t=index(f)*f.Ts) = update!(f, u, y, p, t)

# Measurement function for MUKF
function measurement(mukf::MUKF)
    function (x, u, p, t)
        nxn = length(mukf.xn)
        xn = x[1:nxn]
        xl = x[nxn+1:end]

        g = mukf.nl_measurement_model.measurement
        Cl = get_mat(mukf.kf.C, xn, u, p, t)

        return g(xn, u, p, t) .+ Cl * xl
    end
end

# Dynamics function for MUKF
function dynamics(mukf::MUKF)
    function (x, u, p, t)
        nxn = length(mukf.xn)
        xn = x[1:nxn]
        xl = x[nxn+1:end]

        An = get_mat(mukf.An, xn, u, p, t)
        Al = get_mat(mukf.kf.A, xn, u, p, t)
        Bl = get_mat(mukf.kf.B, xn, u, p, t)

        xn_next = mukf.dynamics(xn, u, p, t) .+ An * xl
        xl_next = Al * xl .+ Bl * u

        return [xn_next; xl_next]
    end
end

# Simulation support
function sample_state(f::MUKF, p=parameters(f); noise=true)
    xn = noise ? rand(f.d0n) : f.d0n.μ
    xl = noise ? rand(f.kf.d0) : f.kf.d0.μ
    return [xn; xl]
end

function sample_state(f::MUKF{IPD}, x, u, p=parameters(f), t=index(f)*f.Ts; noise=true) where IPD
    nxn = length(f.xn)
    xn = x[1:nxn]
    xl = x[nxn+1:end]

    An = get_mat(f.An, xn, u, p, t)

    # Handle in-place vs out-of-place dynamics
    if IPD
        xn_next = similar(xn)
        f.dynamics(xn_next, xn, u, p, t)
        @bangbang xn_next .+= An * xl
    else
        xn_next = f.dynamics(xn, u, p, t) .+ An * xl
    end

    if noise
        R1n_mat = f.R1n isa AbstractMatrix ? f.R1n : (f.R1n isa Function ? f.R1n(xn, u, p, t) : f.R1n.Σ)
        xn_next = xn_next .+ rand(SimpleMvNormal(R1n_mat))
    end

    Al = get_mat(f.kf.A, xn, u, p, t)
    Bl = get_mat(f.kf.B, xn, u, p, t)
    xl_next = Al * xl .+ Bl * u
    if noise
        xl_next = xl_next .+ rand(SimpleMvNormal(get_mat(f.kf.R1, xn, u, p, t)))
    end

    return [xn_next; xl_next]
end

function sample_measurement(f::MUKF, x, u, p=parameters(f), t=index(f)*f.Ts; noise=true)
    nxn = length(f.xn)
    xn = x[1:nxn]
    xl = x[nxn+1:end]

    g = f.nl_measurement_model.measurement
    Cl = get_mat(f.kf.C, xn, u, p, t)

    y = g(xn, u, p, t) .+ Cl * xl
    if noise
        y = y .+ rand(f.nl_measurement_model.R2)
    end
    return y
end

# Custom forward_trajectory for MUKF
function forward_trajectory(mukf::MUKF, u::AbstractVector, y::AbstractVector, p=parameters(mukf); debug=false)
    reset!(mukf)
    T    = length(y)
    x    = Vector{Vector{Float64}}(undef, T)
    xt   = Vector{Vector{Float64}}(undef, T)
    R    = Vector{Matrix{Float64}}(undef, T)
    Rt   = Vector{Matrix{Float64}}(undef, T)
    e    = similar(y)
    ll   = 0.0
    local t, S, K

    try
        for outer t = 1:T
            ti = (t-1)*mukf.Ts
            x[t]  = state(mukf)      |> copy
            R[t]  = covariance(mukf) |> copy

            lli, ei, Si, Sᵪi, Ki = correct!(mukf, u[t], y[t], p, ti)
            ll += lli
            e[t] = ei
            xt[t] = state(mukf)      |> copy
            Rt[t] = covariance(mukf) |> copy

            if t == 1
                S = Vector{typeof(Sᵪi)}(undef, T)
                K = Vector{typeof(Ki)}(undef, T)
            end
            S[t] = Sᵪi
            K[t] = Ki

            predict!(mukf, u[t], p, ti)
        end
    catch err
        if debug
            t -= 1
            x, xt, R, Rt, e, u, y = x[1:t], xt[1:t], R[1:t], Rt[1:t], e[1:t], u[1:t], y[1:t]
            @error "State estimation failed, returning partial solution" err
        else
            @error "State estimation failed, pass `debug = true` to forward_trajectory to return a partial solution"
            rethrow()
        end
    end
    KalmanFilteringSolution(mukf, u, y, x, xt, R, Rt, ll, e, K, S)
end
