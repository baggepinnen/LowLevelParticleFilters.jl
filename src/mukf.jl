
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
x_{t+1}^n &= f_n(x_t^n, u, p, t) + A_n(x_t^n)\\, x_t^l + w_t^n, \\quad &w_t^n \\sim \\mathcal{N}(0, R_1^n) \\\\
x_{t+1}^l &= A_l(x_t^n)\\, x_t^l + B_l(x_t^n)\\, u + w_t^l, \\quad &w_t^l \\sim \\mathcal{N}(0, R_1^l) \\\\
y_t &= g(x_t^n, u, p, t) + C_l(x_t^n)\\, x_t^l + e_t, \\quad &e_t \\sim \\mathcal{N}(0, R_2)
\\end{aligned}
````

where `x^n` is the nonlinear substate and `x^l` is the linear substate.

# Arguments

* `dynamics`: The nonlinear dynamics function `f_n(x^n, u, p, t)`
* `nl_measurement_model`: An instance of [`RBMeasurementModel`](@ref) containing `g` and `R_2`
* `An`: The coupling matrix/function from linear to nonlinear state (matrix or function)
* `kf`: A [`KalmanFilter`](@ref) describing the linear substate dynamics (`A_l, B_l, C_l, R_1^l`)
* `R1n`: Process noise covariance for the nonlinear substate (matrix, `SimpleMvNormal`, or function)
* `d0n`: Initial distribution for the nonlinear substate (SimpleMvNormal)
* `nu`: Number of inputs (default: `kf.nu`)
* `Ts`: Sampling time (default: 1.0)
* `p`: Parameters (default: NullParameters())
* `weight_params`: Unscented transform parameters (default: MerweParams())
* `names`: Signal names for plotting
  """
mutable struct MUKF{IPD,IPM,DT,MMT,ANT,KFT,R1NT,D0NT,XNT,XLT,PT,TS,PARAMS,WP,NT,SPC} <:
               AbstractKalmanFilter

    # Model functions and parameters

    dynamics::DT              # xⁿ_{k+1} = fₙ(xⁿ_k, u, p, t) + Aₙ(xⁿ_k) xˡ_k + wⁿ_k
    nl_measurement_model::MMT # Nonlinear measurement model
    An::ANT                   # Coupling (matrix or function)
    kf::KFT                   # Template Kalman filter for linear substate

    # Noise covariances

    R1n::R1NT                 # Process noise covariance for nonlinear state
    d0n::D0NT                 # Initial distribution for nonlinear state

    # Sigma point cache

    sigma_point_cache::SPC
    xn_sigma_points::Vector{XNT}  # Current sigma points for xn (for cross-covariance)

    # Current state estimates

    xn::XNT                   # Mean of nonlinear state
    xl::Vector{XLT}           # Mean of linear state (all elements equal)
    P::PT                     # Joint covariance matrix [xn; xl]

    # Metadata

    t::Int
    Ts::TS
    nu::Int
    ny::Int
    p::PARAMS
    weight_params::WP
    names::NT
end

function MUKF{IPD,IPM}(;
    dynamics,
    nl_measurement_model::RBMeasurementModel,
    An,
    kf::KalmanFilter,
    R1n,
    d0n,
    nu = kf.nu,
    Ts = 1.0,
    p = NullParameters(),
    weight_params = MerweParams(),
    names = default_names(length(d0n.μ) + length(kf.d0.μ), nu, kf.ny, "MUKF"),
) where {IPD,IPM}

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
    xn0  = convert_x0_type(d0n.μ)
    xl0  = convert_x0_type(kf.d0.μ)

    # Initialize joint covariance P0 = blockdiag(Rn0, Γ0) with Rnl=0
    Rn0  = convert_cov_type(R1n, d0n.Σ)
    Γ0   = convert_cov_type(kf.R1, kf.d0.Σ)
    P0   = blockdiag(Rn0, Γ0)  # Start with zero cross-covariance

    # Initialize linear state means (all equal)
    xl = [copy(xl0) for _ = 1:ns]

    # Initialize sigma points (will be updated in predict!/correct!)
    xn_sigma_points = [copy(xn0) for _ = 1:ns]
    sigmapoints!(xn_sigma_points, xn0, Rn0, weight_params)

    MUKF{
        IPD,
        IPM,
        typeof(dynamics),
        typeof(nl_measurement_model),
        typeof(An),
        typeof(kf),
        typeof(R1n),
        typeof(d0n),
        typeof(xn0),
        typeof(xl0),
        typeof(P0),
        typeof(Ts),
        typeof(p),
        typeof(weight_params),
        typeof(names),
        typeof(sigma_point_cache),
    }(
        dynamics,
        nl_measurement_model,
        An,
        kf,
        R1n,
        d0n,
        sigma_point_cache,
        xn_sigma_points,
        xn0,
        xl,
        P0,
        0,
        Ts,
        nu,
        ny,
        p,
        weight_params,
        names,
    )

end

# Convenience constructor

MUKF(args...; kwargs...) = MUKF{false,false}(args...; kwargs...)

# Convenience accessors

state(f::MUKF) = [f.xn; xl_mean(f)]
covariance(f::MUKF) = f.P
parameters(f::MUKF) = f.p
particletype(f::MUKF) = typeof([f.xn; f.xl[1]])
covtype(f::MUKF) = typeof(f.P)

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
Return the mean of the linear substate. All xl[i] are equal by construction.
"""
function xl_mean(f::MUKF)
    return f.xl[1]  # All xl[i] are kept equal
end

"""
xl_cov(f::MUKF)
Extract the marginal covariance of the linear substate from joint covariance P.
"""
function xl_cov(f::MUKF)
    nxn = length(f.xn)
    return f.P[nxn+1:end, nxn+1:end]
end

"""
xl_cross_cov(f::MUKF)
Extract cross-covariance between xn and xl from joint covariance P.
"""
function xl_cross_cov(f::MUKF)
    nxn = length(f.xn)
    return f.P[1:nxn, nxn+1:end]
end

function reset!(f::MUKF, d0n = f.d0n, d0l = f.kf.d0)
    @bangbang f.xn .= d0n.μ
    for i in eachindex(f.xl)
        @bangbang f.xl[i] .= d0l.μ
    end
    # Reset to block diagonal covariance with zero cross-covariance
    @bangbang f.P .= blockdiag(d0n.Σ, d0l.Σ)
    sigmapoints!(f.xn_sigma_points, d0n.μ, d0n.Σ, f.weight_params)
    f.t = 0
    f
end

# --- Helper functions for MUT (Marginalized Unscented Transform) ---


"""
    partition_cov(P, nxn, nxl)

Partition joint covariance P into blocks: Pnn, Pnl, Pln, Pll
"""
function partition_cov(P, nxn, nxl)
    Pnn = P[1:nxn, 1:nxn]
    Pnl = P[1:nxn, nxn+1:end]
    Pln = P[nxn+1:end, 1:nxn]
    Pll = P[nxn+1:end, nxn+1:end]
    return Pnn, Pnl, Pln, Pll
end

"""
    cond_linear_params(Pnn, Pnl, Pln, Pll)

Compute conditional Gaussian parameters:
- L = Pln * inv(Pnn): regression matrix
- Γ = Pll - Pln * inv(Pnn) * Pnl: conditional covariance

For joint Gaussian [xn; xl] ~ N(μ, P), the conditional distribution is:
xl | xn ~ N(μl + L*(xn - μn), Γ)
"""
function cond_linear_params(Pnn, Pnl, Pln, Pll)
    # Compute L = Pln * inv(Pnn) stably using linear solve
    nxn = size(Pnn, 1)
    L = Pln * (Pnn \ I(nxn))
    # Compute conditional covariance
    Γ = Pll - Pln * (Pnn \ Pnl)
    return L, Γ
end

"""
    blockdiag(A, B)

Create block diagonal matrix [A 0; 0 B]
"""
function blockdiag(A::StaticMatrix{m,n}, B::StaticMatrix{p,q}) where {m,n,p,q}
    T = promote_type(eltype(A), eltype(B))
    return SMatrix{m+p,n+q,T}([A zeros(SMatrix{m,q,T}); zeros(SMatrix{p,n,T}) B])
end

function blockdiag(A::AbstractMatrix, B::AbstractMatrix)
    m, n = size(A)
    p, q = size(B)
    C = zeros(eltype(A), m+p, n+q)
    C[1:m, 1:n] = A
    C[m+1:end, n+1:end] = B
    return C
end

# --- MUT-based predict and correct functions ---

function predict!(
    f::MUKF{IPD},
    u = zeros(f.nu),
    p = parameters(f),
    t::Real = index(f)*f.Ts,
) where {IPD}
    nxn = length(f.xn)
    nxl = length(f.xl[1])
    nx = nxn + nxl

    # Get process noise covariances
    R1n_mat = f.R1n isa AbstractMatrix ? f.R1n :
              (f.R1n isa Function ? f.R1n(f.xn, u, p, t) : f.R1n.Σ)
    R1l_mat = get_mat(f.kf.R1, f.xn, u, p, t)  # May be state-dependent

    # Extract conditional parameters from current joint covariance
    Pnn, Pnl, Pln, Pll = partition_cov(f.P, nxn, nxl)
    L, Γ_curr = cond_linear_params(Pnn, Pnl, Pln, Pll)

    # Current means
    μn = f.xn
    μl = xl_mean(f)

    # Generate sigma points for nonlinear state only
    sp = f.sigma_point_cache.x0
    sigmapoints!(sp, μn, Pnn, f.weight_params)
    W = UKFWeights(f.weight_params, nxn)

    # Transform sigma points through dynamics
    # Yi = [fn(sp[i]) + An_i*νB_i; Al_i*νB_i + Bl_i*u]
    # where νB_i = μl + L*(sp[i] - μn) is the conditional mean of xl given xn=sp[i]

    # Preallocate with type preservation
    Y_proto = [f.xn; μl]  # Prototype full state vector
    Y = [similar(Y_proto) for _ in eachindex(sp)]
    G_proto = [f.An; f.kf.A]  # Prototype G matrix
    G_matrices = [similar(G_proto) for _ in eachindex(sp)]

    if IPD
        # In-place dynamics
        xp = similar(f.xn)
        @inbounds for i in eachindex(sp)
            # Get state-dependent matrices
            An_i = get_mat(f.An, sp[i], u, p, t)
            Al_i = get_mat(f.kf.A, sp[i], u, p, t)
            Bl_i = get_mat(f.kf.B, sp[i], u, p, t)

            # Conditional mean of xl given xn=sp[i]
            νB = μl .+ L * (sp[i] .- μn)

            # Nonlinear state dynamics
            xp .= 0
            f.dynamics(xp, sp[i], u, p, t)
            xp .+= An_i * νB

            # Linear state dynamics
            xl_i = Al_i * νB .+ Bl_i * u

            # Store full transformed state
            Y[i] = [xp; xl_i]

            # Store G matrix: [An_i; Al_i]
            G_matrices[i] = [An_i; Al_i]
        end
    else
        # Out-of-place dynamics
        @inbounds for i in eachindex(sp)
            An_i = get_mat(f.An, sp[i], u, p, t)
            Al_i = get_mat(f.kf.A, sp[i], u, p, t)
            Bl_i = get_mat(f.kf.B, sp[i], u, p, t)

            νB = μl .+ L * (sp[i] .- μn)

            xn_i = f.dynamics(sp[i], u, p, t) .+ An_i * νB
            xl_i = Al_i * νB .+ Bl_i * u

            Y[i] = [xn_i; xl_i]
            G_matrices[i] = [An_i; Al_i]
        end
    end

    # Compute predicted mean
    μ_pred = zero(Y[1])
    @inbounds for i in eachindex(sp)
        w = i == 1 ? W.wm : W.wmi
        μ_pred = μ_pred .+ w .* Y[i]
    end

    # Compute spread covariance from transformed sigma points
    P_spread = zero(f.P)
    @inbounds for i in eachindex(sp)
        w = i == 1 ? W.wc : W.wci
        δ = Y[i] .- μ_pred
        P_spread = P_spread .+ w .* (δ * δ')
    end

    # Compute weighted average of G matrices for analytic term
    G_avg = zero(G_matrices[1])
    @inbounds for i in eachindex(sp)
        w = i == 1 ? W.wm : W.wmi
        G_avg = G_avg .+ w .* G_matrices[i]
    end

    # Add analytic MUT term: G*Γ*G'
    P_analytic = G_avg * Γ_curr * G_avg'

    # Add process noise
    P_noise = blockdiag(R1n_mat, R1l_mat)

    # Full predicted covariance
    P_pred = P_spread .+ P_analytic .+ P_noise
    P_pred = symmetrize(P_pred)

    # Update state and covariance
    @bangbang f.xn .= μ_pred[1:nxn]
    μl_pred = μ_pred[nxn+1:end]
    @bangbang f.P .= P_pred

    # Update xl means - all xl[i] are set to marginal mean
    @inbounds for i in eachindex(f.xl)
        f.xl[i] = μl_pred
    end

    f.t += 1
    return f
end

function correct!(
    f::MUKF{IPD,IPM},
    u,
    y,
    p = parameters(f),
    t::Real = index(f)*f.Ts;
    R2 = nothing,
) where {IPD,IPM}
    nxn = length(f.xn)
    nxl = length(f.xl[1])
    nx = nxn + nxl
    ny = f.ny

    # Get measurement noise covariance
    g = f.nl_measurement_model.measurement
    R2_mat = if R2 !== nothing
        R2
    else
        mm_R2 = f.nl_measurement_model.R2
        mm_R2 isa AbstractMatrix ? mm_R2 :
        (mm_R2 isa Function ? mm_R2(f.xn, u, p, t) : mm_R2.Σ)
    end

    # Extract conditional parameters from current joint covariance
    Pnn, Pnl, Pln, Pll = partition_cov(f.P, nxn, nxl)
    L, Γ_curr = cond_linear_params(Pnn, Pnl, Pln, Pll)

    # Current means
    μn = f.xn
    μl = xl_mean(f)
    μ_full = [μn; μl]

    # Generate sigma points for nonlinear state only
    sp = f.sigma_point_cache.x0
    sigmapoints!(sp, μn, Pnn, f.weight_params)
    W = UKFWeights(f.weight_params, nxn)

    # Transform sigma points through measurement model
    # yi = g(sp[i]) + Cl_i*νB_i
    # where νB_i = μl + L*(sp[i] - μn)

    # Type-preserving allocations using prototypes
    y_proto = similar(y)
    Y_meas = [similar(y_proto) for _ in eachindex(sp)]
    x_proto = [f.xn; μl]
    X_full = [similar(x_proto) for _ in eachindex(sp)]
    Cl_proto = get_mat(f.kf.C, f.xn, u, p, t)
    Cl_matrices = [similar(Cl_proto) for _ in eachindex(sp)]

    if IPM
        # In-place measurement
        y_temp = similar(y)
        @inbounds for i in eachindex(sp)
            Cl_i = get_mat(f.kf.C, sp[i], u, p, t)
            νB = μl .+ L * (sp[i] .- μn)

            g(y_temp, sp[i], u, p, t)
            Y_meas[i] = y_temp .+ Cl_i * νB

            # Store full state vector for cross-covariance
            X_full[i] = [sp[i]; νB]
            Cl_matrices[i] = Cl_i
        end
    else
        # Out-of-place measurement
        @inbounds for i in eachindex(sp)
            Cl_i = get_mat(f.kf.C, sp[i], u, p, t)
            νB = μl .+ L * (sp[i] .- μn)

            Y_meas[i] = g(sp[i], u, p, t) .+ Cl_i * νB
            X_full[i] = [sp[i]; νB]
            Cl_matrices[i] = Cl_i
        end
    end

    # Compute predicted measurement mean
    yhat = zero(Y_meas[1])
    @inbounds for i in eachindex(sp)
        w = i == 1 ? W.wm : W.wmi
        yhat = yhat .+ w .* Y_meas[i]
    end

    # Compute innovation covariance: S = spread + Cl*Γ*Cl' + R2
    S = zero(R2_mat)
    @inbounds for i in eachindex(sp)
        w = i == 1 ? W.wc : W.wci
        δy = Y_meas[i] .- yhat
        S = S .+ w .* (δy * δy')
    end

    # Compute weighted average of Cl matrices for analytic term
    Cl_avg = zero(Cl_matrices[1])
    @inbounds for i in eachindex(sp)
        w = i == 1 ? W.wm : W.wmi
        Cl_avg = Cl_avg .+ w .* Cl_matrices[i]
    end

    # Add analytic MUT term to innovation covariance
    S = S .+ Cl_avg * Γ_curr * Cl_avg' .+ R2_mat
    S = symmetrize(S)

    # Compute cross-covariance Σxy with CRITICAL extra term from equation 16
    # Create proper nx x ny matrix prototype from outer product
    δx_proto = X_full[1] .- μ_full
    δy_proto = Y_meas[1] .- yhat
    Σxy = zero(δx_proto * δy_proto')
    @inbounds for i in eachindex(sp)
        w = i == 1 ? W.wc : W.wci
        δx = X_full[i] .- μ_full
        δy = Y_meas[i] .- yhat
        Σxy = Σxy .+ w .* (δx * δy')
    end

    # Add the missing term: [0; Γ*Cl_avg'] (from equation 16 in MUT paper)
    # This is the conditional covariance contribution that was causing negative Γ!
    Σxy[nxn+1:end, :] .+= Γ_curr * Cl_avg'

    # Factorize S and compute Kalman gain
    Sᵪ = cholesky(Symmetric(S), check = false)
    if !issuccess(Sᵪ)
        error("Cholesky factorization of innovation covariance failed at time step $(f.t)")
    end
    K = Σxy / Sᵪ

    # Innovation
    innovation = y .- yhat

    # Update full state and covariance
    μ_new = μ_full .+ K * innovation
    P_new = f.P .- K * S * K'
    P_new = symmetrize(P_new)

    # Update state and covariance
    @bangbang f.xn .= μ_new[1:nxn]
    μl_new = μ_new[nxn+1:end]
    @bangbang f.P .= P_new

    # Update xl means - all xl[i] are set to marginal mean
    @inbounds for i in eachindex(f.xl)
        f.xl[i] = μl_new
    end

    ll = extended_logpdf(SimpleMvNormal(PDMat(S, Sᵪ)), innovation)
    return (; ll, e = innovation, S, Sᵪ, K = K[1:nxn, :])  # Return only nonlinear gain for compatibility

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
