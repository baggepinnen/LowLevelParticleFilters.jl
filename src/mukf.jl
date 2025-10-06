"""
    MUKF(; dynamics, nl_measurement_model::RBMeasurementModel, An, kf::KalmanFilter, R1n, d0n, nu=kf.nu, Ts=1.0, p=NullParameters(), weight_params=MerweParams(), names=default_names(length(d0n.μ) + length(kf.d0.μ), nu, kf.ny, "MUKF"))

Marginalized Unscented Kalman Filter for mixed linear/nonlinear state-space models.

!!! warning "Experimental"
    This filter is currently considered experimental and the user interface may change in the future without respecting semantic versioning.

This filter combines the Unscented Kalman Filter (UKF) for the nonlinear substate with a
bank of Kalman filters (one per sigma point) for the linear substate. This approach provides
improved accuracy compared to linearization-based methods while remaining more efficient
than a full particle filter.

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
mutable struct MUKF{DT,MMT,ANT,KFT,R1NT,D0NT,XNT,RNT,XLT,RLT,TS,P,WP,NT} <: AbstractKalmanFilter
    # Model functions and parameters
    dynamics::DT              # xⁿ_{k+1} = fₙ(xⁿ_k, u, p, t) + Aₙ xˡ_k + wⁿ_k
    nl_measurement_model::MMT # Nonlinear measurement model
    An::ANT                   # Coupling matrix (or function)
    kf::KFT                   # Template Kalman filter for linear substate

    # Noise covariances
    R1n::R1NT                 # Process noise covariance for nonlinear state
    d0n::D0NT                 # Initial distribution for nonlinear state

    # Current state estimates
    xn::XNT                   # Mean of nonlinear state
    Rn::RNT                   # Covariance of nonlinear state
    xl::Vector{XLT}           # Mean of linear state per sigma point
    Rl::Vector{RLT}           # Covariance of linear state per sigma point

    # Metadata
    t::Int
    Ts::TS
    nu::Int
    ny::Int
    p::P
    weight_params::WP
    names::NT
end

function MUKF(; dynamics, nl_measurement_model::RBMeasurementModel, An, kf::KalmanFilter,
              R1n, d0n, nu=kf.nu, Ts=1.0, p=NullParameters(),
              weight_params=MerweParams(),
              names=default_names(length(d0n.μ) + length(kf.d0.μ), nu, kf.ny, "MUKF"))

    nxn = length(d0n.μ)
    nxl = length(kf.d0.μ)
    ny  = kf.ny

    # Initialize sigma points for the nonlinear state
    sp = sigmapoints(d0n.μ, d0n.Σ, weight_params)
    ns = length(sp)  # Number of sigma points (2*nxn + 1)

    # Initialize state estimates using type from d0 and kf
    xn0 = convert_x0_type(d0n.μ)
    Rn0 = convert_cov_type(R1n, d0n.Σ)
    xl0 = convert_x0_type(kf.d0.μ)
    Rl0 = convert_cov_type(kf.R1, kf.d0.Σ)

    # Initialize one Kalman filter per sigma point
    xl = [copy(xl0) for _ in 1:ns]
    Rl = [copy(Rl0) for _ in 1:ns]

    MUKF{typeof(dynamics), typeof(nl_measurement_model), typeof(An), typeof(kf),
         typeof(R1n), typeof(d0n), typeof(xn0), typeof(Rn0), typeof(xl0), typeof(Rl0),
         typeof(Ts), typeof(p), typeof(weight_params), typeof(names)}(
        dynamics, nl_measurement_model, An, kf, R1n, d0n,
        xn0, Rn0, xl, Rl,
        0, Ts, nu, ny, p, weight_params, names)
end

# Convenience accessors
state(f::MUKF) = [f.xn; xl_mean(f)]
covariance(f::MUKF) = cat(f.Rn, xl_cov(f), dims=(1,2))
parameters(f::MUKF) = f.p
particletype(f::MUKF) = typeof([f.xn; f.xl[1]])
covtype(f::MUKF) = typeof(cat(f.Rn, f.Rl[1], dims=(1,2)))

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
    sp = sigmapoints(f.xn, f.Rn, f.weight_params)
    W = UKFWeights(f.weight_params, length(f.xn))
    μ = zeros(eltype(f.xl[1]), length(f.xl[1]))
    @inbounds for i in eachindex(f.xl)
        w = i == 1 ? W.wm : W.wmi
        μ .+= w .* f.xl[i]
    end
    return μ
end

"""
    xl_cov(f::MUKF)

Compute the marginal covariance of the linear substate.
"""
function xl_cov(f::MUKF)
    sp = sigmapoints(f.xn, f.Rn, f.weight_params)
    W = UKFWeights(f.weight_params, length(f.xn))
    μ = xl_mean(f)
    nxl = length(μ)
    Σ = zeros(eltype(f.Rl[1]), nxl, nxl)

    @inbounds for i in eachindex(f.xl)
        w = i == 1 ? W.wc : W.wci
        d = f.xl[i] .- μ
        Σ .+= w .* (d * d' .+ f.Rl[i])
    end
    return Σ
end

function reset!(f::MUKF, d0n=f.d0n, d0l=f.kf.d0)
    f.xn .= d0n.μ
    f.Rn .= d0n.Σ
    sp = sigmapoints(f.xn, f.Rn, f.weight_params)
    for i in eachindex(f.xl)
        f.xl[i] .= d0l.μ
        f.Rl[i] .= d0l.Σ
    end
    f.t = 0
    f
end

function predict!(f::MUKF, u=zeros(f.nu), p=parameters(f), t::Real=index(f)*f.Ts)
    # Generate sigma points for nonlinear state
    sp = sigmapoints(f.xn, f.Rn, f.weight_params)
    W = UKFWeights(f.weight_params, length(f.xn))

    # Get matrices
    Al = get_mat(f.kf.A, f.xn, u, p, t)
    Bl = get_mat(f.kf.B, f.xn, u, p, t)
    R1l = get_mat(f.kf.R1, f.xn, u, p, t)
    An = get_mat(f.An, f.xn, u, p, t)
    R1n_mat = f.R1n isa AbstractMatrix ? f.R1n : (f.R1n isa Function ? f.R1n(f.xn, u, p, t) : f.R1n.Σ)

    # Propagate each Kalman filter (linear substate time update)
    for i in eachindex(f.xl)
        f.xl[i] = Al * f.xl[i] .+ Bl * u
        f.Rl[i] = Al * f.Rl[i] * Al' .+ R1l
    end

    # Propagate nonlinear state through dynamics using sigma points
    Xn_pred = similar(sp)
    @inbounds for i in eachindex(sp)
        Xn_pred[i] = f.dynamics(sp[i], u, p, t) .+ An * f.xl[i]
    end

    # Compute predicted mean and covariance for nonlinear state
    xn_pred = weighted_mean(Xn_pred, W)
    Rn_pred = weighted_cov(Xn_pred, xn_pred, W) .+ R1n_mat

    f.xn .= xn_pred
    f.Rn .= Rn_pred
    f.t += 1
    return f
end

function correct!(f::MUKF, u, y, p=parameters(f), t::Real=index(f)*f.Ts; R2=nothing)
    # Generate sigma points
    sp = sigmapoints(f.xn, f.Rn, f.weight_params)
    W = UKFWeights(f.weight_params, length(f.xn))

    g = f.nl_measurement_model.measurement
    Cl = get_mat(f.kf.C, f.xn, u, p, t)
    R2_mat = if R2 !== nothing
        R2
    else
        mm_R2 = f.nl_measurement_model.R2
        mm_R2 isa AbstractMatrix ? mm_R2 : (mm_R2 isa Function ? mm_R2(f.xn, u, p, t) : mm_R2.Σ)
    end

    # Predicted measurements per sigma point: ŷᵢ = g(xⁿᵢ) + Cl xˡᵢ
    Y = Vector{typeof(y)}(undef, length(sp))
    Σy_extra = zeros(f.ny, f.ny)

    @inbounds for i in eachindex(sp)
        Y[i] = g(sp[i], u, p, t) .+ Cl * f.xl[i]
        # Add contribution from linear state uncertainty
        Σy_extra .+= (i == 1 ? W.wc : W.wci) .* (Cl * f.Rl[i] * Cl')
    end

    yhat = weighted_mean(Y, W)

    # Innovation covariance
    S = weighted_cov(Y, yhat, W) .+ Σy_extra .+ R2_mat

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

    Kn = Σny / Sᵪ
    innovation = y .- yhat
    f.xn .= f.xn .+ Kn * innovation
    f.Rn .= f.Rn .- Kn * S * Kn'
    f.Rn .= (f.Rn .+ f.Rn') ./ 2  # Ensure symmetry

    # Update each linear Kalman filter with its residual
    @inbounds for i in eachindex(sp)
        r_i = y .- g(sp[i], u, p, t) .- Cl * f.xl[i]
        Si = Cl * f.Rl[i] * Cl' .+ R2_mat
        Ki = f.Rl[i] * Cl' / cholesky(Symmetric(Si))
        f.xl[i] .= f.xl[i] .+ Ki * r_i
        f.Rl[i] = (I - Ki * Cl) * f.Rl[i]
        f.Rl[i] .= (f.Rl[i] .+ f.Rl[i]') ./ 2  # Ensure symmetry
    end

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

function sample_state(f::MUKF, x, u, p=parameters(f), t=index(f)*f.Ts; noise=true)
    nxn = length(f.xn)
    xn = x[1:nxn]
    xl = x[nxn+1:end]

    An = get_mat(f.An, xn, u, p, t)
    xn_next = f.dynamics(xn, u, p, t) .+ An * xl
    if noise
        R1n_mat = f.R1n isa AbstractMatrix ? f.R1n : (f.R1n isa Function ? f.R1n(xn, u, p, t) : f.R1n.Σ)
        xn_next .+= rand(SimpleMvNormal(R1n_mat))
    end

    Al = get_mat(f.kf.A, xn, u, p, t)
    Bl = get_mat(f.kf.B, xn, u, p, t)
    xl_next = Al * xl .+ Bl * u
    if noise
        xl_next .+= rand(SimpleMvNormal(get_mat(f.kf.R1, xn, u, p, t)))
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
        y .+= rand(f.nl_measurement_model.R2)
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
