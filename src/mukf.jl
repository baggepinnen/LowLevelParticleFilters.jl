
"""
    MUKFCorrectCache

Cache for correction step arrays in MUKF to avoid allocations.

# Fields:
- `X_full`: Vector of full state vectors [xn; xl] per sigma point
- `Cl_matrices`: Vector of Cl measurement matrices per sigma point
- `y_temp`: Temporary vector for in-place measurement function
"""
struct MUKFCorrectCache{XF, CL, YT}
    X_full::XF
    Cl_matrices::CL
    y_temp::YT
end

"""
    MUKFCorrectCache(x_proto, Cl_proto, y_proto, ns, static)

Create correction cache with proper types.

# Arguments:
- `x_proto`: Prototype full state vector [xn; xl] (nx-dimensional)
- `Cl_proto`: Prototype Cl matrix (ny × nxl)
- `y_proto`: Prototype measurement vector (ny-dimensional)
- `ns`: Number of sigma points (2*nxn + 1)
- `static`: Whether to use static arrays
"""
function MUKFCorrectCache(x_proto, Cl_proto, y_proto, ns, static::Bool, IPM)
    if static
        # For static arrays, create mutable static arrays to allow in-place operations
        X_full = [SVector{length(x_proto)}(x_proto) for _ in 1:ns]
        Cl_matrices = [SMatrix{size(Cl_proto,1), size(Cl_proto,2)}(Cl_proto) for _ in 1:ns]
        y_temp = IPM ? similar(y_proto) : SVector{length(y_proto)}(y_proto)
    else
        # For regular arrays, use similar
        X_full = [similar(x_proto) for _ in 1:ns]
        Cl_matrices = [similar(Cl_proto) for _ in 1:ns]
        y_temp = similar(y_proto)
    end
    MUKFCorrectCache(X_full, Cl_matrices, y_temp)
end

"""
    MUKFPredictCache

Cache for prediction step arrays in MUKF to avoid allocations.

# Fields:
- `Y`: Vector of full state vectors [xn; xl] per sigma point
- `G_matrices`: Vector of G matrices [An; Al] per sigma point
- `xp`: Temporary vector for in-place dynamics function
"""
struct MUKFPredictCache{Y_VEC, G_MAT, XP}
    Y::Y_VEC
    G_matrices::G_MAT
    xp::XP
end

"""
    MUKFPredictCache(x_full_proto, G_proto, xp_proto, ns, static)

Create prediction cache with proper types.

# Arguments:
- `x_full_proto`: Prototype full state vector [xn; xl] (nx-dimensional)
- `G_proto`: Prototype G matrix [An; Al] (nx × nxl)
- `xp_proto`: Prototype xn vector for in-place dynamics (nxn-dimensional)
- `ns`: Number of sigma points (2*nxn + 1)
- `static`: Whether to use static arrays
"""
function MUKFPredictCache(x_full_proto, G_proto, xp_proto, ns, static::Bool)
    if static
        # For static arrays, create mutable static arrays
        Y = [SVector{length(x_full_proto)}(x_full_proto) for _ in 1:ns]
        G_matrices = [SMatrix{size(G_proto,1), size(G_proto,2)}(G_proto) for _ in 1:ns]
        xp = SVector{length(xp_proto)}(xp_proto)
    else
        # For regular arrays
        Y = [similar(x_full_proto) for _ in 1:ns]
        G_matrices = [similar(G_proto) for _ in 1:ns]
        xp = similar(xp_proto)
    end
    MUKFPredictCache(Y, G_matrices, xp)
end

"""
    MUKF(; dynamics, nl_measurement_model::RBMeasurementModel, A, Cl, R1, d0, nxn, nu, ny, Ts=1.0, p=NullParameters(), weight_params=MerweParams(), names=default_names(length(d0.μ), nu, ny, "MUKF"))

Marginalized Unscented Kalman Filter for mixed linear/nonlinear state-space models.

!!! warning "Experimental"
    This filter is currently considered experimental and the user interface may change in the future without respecting semantic versioning.

This filter combines the Unscented Kalman Filter (UKF) for the nonlinear substate with a bank of Kalman filters (one per sigma point) for the linear substate. This approach provides improved accuracy compared to linearization-based methods while remaining more efficient than a full particle filter. This filter is sometimes referred to as a Rao-Blackwellized Unscented Kalman Filter, similar to the [`RBPF`](@ref).

Ref: Morelande, M.R. & Moran, Bill. (2007). An Unscented Transformation for Conditionally Linear Models

# Model structure
The filter assumes dynamics on the form:
```math
\\begin{aligned}
x_{t+1}^n &= d_n(x_t^n, u, p, t) + A_n(x_t^n)\\, x_t^l + w_t^n \\\\
x_{t+1}^l &= d_l(x_t^n, u, p, t) + A_l(x_t^n)\\, x_t^l + w_t^l \\\\
w_t &= \\begin{bmatrix} w_t^n \\\\ w_t^l \\end{bmatrix} &\\sim \\mathcal{N}(0, R_1) \\\\
y_t &= g(x_t^n, u, p, t) + C_l(x_t^n)\\, x_t^l + e_t, \\quad &e_t \\sim \\mathcal{N}(0, R_2)
\\end{aligned}
```

where ``x^n`` is the nonlinear substate and ``x^`l` is the linear substate. This is the **conditionally linear** form from Morelande & Moran (2007), which allows the linear substate to depend on the nonlinear substate through both ``d_l(x^n)`` and the coupling matrices. Control input dependence can be encoded directly in the ``d_l(x^n, u)`` term.

# Arguments

* `dynamics`: Function returning nonlinear contribution to the dynamics `[dn(xn, u, p, t); dl(xn, u, p, t)]`. Control input dependence can be encoded directly in both `dn` and `dl`.
* `nl_measurement_model`: An instance of [`RBMeasurementModel`](@ref) containing `g` and `R_2`
* `A`: Combined coupling and dynamics matrix/function `[An(xn); Al(xn)]` (nx × nxl). The first nxn rows (An) couple the linear state to the nonlinear state dynamics, and the last nxl rows (Al) define the linear state dynamics matrix.
* `Cl`: Measurement matrix/function `Cl(xn, u, p, t)` for the linear substate
* `R1`: Full process noise covariance matrix (nx × nx) for the combined state [xn; xl] (matrix or function)
* `d0`: Initial normal distribution for the full state [xn; xl] (`LowLevelParticleFilters.SimpleMvNormal`)
* `nxn`: Dimension of the nonlinear substate
* `nu`: Number of inputs
* `ny`: Number of measurements
* `Ts`: Sampling time (default: 1.0)
* `p`: Parameters (default: NullParameters())
* `weight_params`: Unscented transform parameters (default: MerweParams())
* `names`: Signal names for plotting
  """
mutable struct MUKF{IPD,IPM,DT,MMT,AT,CLT,R1T,D0T,XNT,XT,RT,TS,PARAMS,WP,NT,SPC,SPC2,PRED_C,CORR_C,NINDS,LINDS} <:
               AbstractKalmanFilter

    # Model functions and parameters

    dynamics::DT              # Returns [dₙ(xⁿ, u, p, t); dₗ(xⁿ, u, p, t)] (uncoupled part of dynamics)
    nl_measurement_model::MMT # Nonlinear measurement model
    A::AT                     # Combined coupling and dynamics matrix [An; Al] (nx × nxl)
    Cl::CLT                   # Measurement matrix for linear state

    # Noise covariances and initial distributions

    R1::R1T                   # Full process noise covariance (nx × nx)
    d0::D0T                   # Initial distribution for full state [xn; xl]

    # Sigma point caches

    sigma_point_cache::SPC            # For predict step (sigma points only)
    correct_sigma_point_cache::SPC2   # For correct step (sigma points and Y_meas)
    predict_cache::PRED_C             # For predict step (Y, G_matrices, xp)
    correct_cache::CORR_C             # For correct step (X_full, Cl_matrices, y_temp)
    xn_sigma_points::Vector{XNT}      # Current sigma points for xn (for cross-covariance)

    # Current state estimates

    x::XT                     # Full state vector [xn; xl]
    R::RT                     # Joint covariance matrix [xn; xl]
    n_inds::NINDS             # Index vector for nonlinear state (1:nxn)
    l_inds::LINDS             # Index vector for linear state (nxn+1:nxn+nxl)

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
    A,
    Cl,
    R1,
    d0,
    nu,
    ny,
    nxn::Int,
    Ts = 1.0,
    p = NullParameters(),
    weight_params = MerweParams(),
    names = default_names(length(d0.μ), nu, ny, "MUKF"),
) where {IPD,IPM}

    nx = length(d0.μ)
    nxl = nx - nxn

    # Determine element type
    T = eltype(d0)

    # Number of sigma points (2*nxn + 1)
    L = nxn
    ns = 2*L + 1

    # Decide if we should use static arrays (like UKF does)
    static = !(IPD || L > 50)

    # Create sigma point cache for predict
    sigma_point_cache = SigmaPointCache{T}(nxn, 0, nxn, L, static)

    # Create sigma point cache for correct (x0: xn sigma points, x1: Y_meas)
    correct_sigma_point_cache = SigmaPointCache{T}(nxn, 0, ny, L, static)

    # Initialize state estimates from unified d0
    x0 = convert_x0_type(d0.μ)  # Full state vector

    # Create index vectors for partitioning (needed before using xn0)
    # We'll temporarily create these to extract xn0, then recreate with proper type below
    n_inds_temp = 1:nxn
    l_inds_temp = nxn+1:nx
    xn0 = x0[n_inds_temp]

    # Create prototypes for correct cache initialization
    x_proto = x0  # Full state prototype (nx-dimensional)

    # Get Cl prototype - need to evaluate at initial state
    Cl_proto = get_mat(Cl, xn0, zeros(T, nu), p, T(0))  # ny × nxl matrix

    # Create y prototype
    if static && ny <= 50
        y_proto = @SVector zeros(T, ny)
    else
        y_proto = zeros(T, ny)
    end

    # Create correction cache for MUKF-specific arrays
    correct_cache = MUKFCorrectCache(x_proto, Cl_proto, y_proto, ns, static, IPM)

    # Create predict cache for dynamics transformation
    x_full_proto = x_proto  # nx-dimensional (same as x_proto)
    G_proto = get_mat(A, xn0, zeros(T, nu), p, T(0))  # nx×nxl matrix (already concatenated [An; Al])
    xp_proto = x0  # nx-dimensional (full state) - dynamics returns [dn; dl]
    predict_cache = MUKFPredictCache(x_full_proto, G_proto, xp_proto, ns, static)

    # Initialize joint covariance from unified d0
    R0 = convert_cov_type(R1, d0.Σ)  # Use R1 as type reference

    # Create index vectors for efficient partition_cov (static if R0 is static)
    if R0 isa StaticArray
        n_inds = SVector{nxn}(1:nxn)
        l_inds = SVector{nxl}((1:nxl) .+ nxn)
    else
        n_inds = 1:nxn
        l_inds = nxn+1:nx
    end

    # Initialize sigma points (will be updated in predict!/correct!)
    # Extract nonlinear part of R0 for sigma point generation
    Rn0 = R0[n_inds, n_inds]
    xn_sigma_points = [copy(xn0) for _ = 1:ns]
    sigmapoints!(xn_sigma_points, xn0, Rn0, weight_params)

    MUKF{
        IPD,
        IPM,
        typeof(dynamics),
        typeof(nl_measurement_model),
        typeof(A),
        typeof(Cl),
        typeof(R1),
        typeof(d0),
        typeof(xn0),
        typeof(x0),
        typeof(R0),
        typeof(Ts),
        typeof(p),
        typeof(weight_params),
        typeof(names),
        typeof(sigma_point_cache),
        typeof(correct_sigma_point_cache),
        typeof(predict_cache),
        typeof(correct_cache),
        typeof(n_inds),
        typeof(l_inds),
    }(
        dynamics,
        nl_measurement_model,
        A,
        Cl,
        R1,
        d0,
        sigma_point_cache,
        correct_sigma_point_cache,
        predict_cache,
        correct_cache,
        xn_sigma_points,
        x0,
        R0,
        n_inds,
        l_inds,
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

function Base.show(io::IO, mukf::MUKF{IPD,IPM}) where {IPD,IPM}
    println(io, "MUKF{$IPD,$IPM}")
    println(io, "  Inplace dynamics: $IPD")
    println(io, "  Inplace measurement: $IPM")
    nxn = length(mukf.n_inds)
    nxl = length(mukf.l_inds)
    println(io, "  nxn: $nxn (nonlinear state)")
    println(io, "  nxl: $nxl (linear state)")
    println(io, "  nx: $(nxn + nxl)")
    println(io, "  nu: $(mukf.nu)")
    println(io, "  ny: $(mukf.ny)")
    println(io, "  Ts: $(mukf.Ts)")
    println(io, "  t: $(mukf.t)")
    println(io, "  weight_params: $(typeof(mukf.weight_params))")
    for field in fieldnames(typeof(mukf))
        field in (:ny, :nu, :Ts, :t, :nxn, :nxl, :weight_params) && continue
        if field in (:nl_measurement_model, :sigma_point_cache, :correct_sigma_point_cache, :predict_cache, :correct_cache, :A, :Cl, :xn_sigma_points, :x, :R, :n_inds, :l_inds)
            println(io, "  $field: $(fieldtype(typeof(mukf), field))")
        else
            println(io, "  $field: $(repr(getfield(mukf, field), context=:compact => true))")
        end
    end
end

# Convenience accessors

state(f::MUKF) = f.x
covariance(f::MUKF) = f.R
parameters(f::MUKF) = f.p
particletype(f::MUKF) = typeof(f.x)
covtype(f::MUKF) = typeof(f.R)

function Base.getproperty(f::MUKF, s::Symbol)
    s ∈ fieldnames(typeof(f)) && return getfield(f, s)
    if s === :nx
        return length(getfield(f, :x))
    elseif s === :xn
        return getfield(f, :x)[getfield(f, :n_inds)]
    elseif s === :xl
        return getfield(f, :x)[getfield(f, :l_inds)]
    elseif s === :R2
        return getfield(f, :nl_measurement_model).R2.Σ
    else
        throw(ArgumentError("$(typeof(f)) has no property named $s"))
    end
end


"""
xl_cov(f::MUKF)
Extract the marginal covariance of the linear substate from joint covariance R.
"""
function xl_cov(f::MUKF)
    return f.R[f.l_inds, f.l_inds]
end

"""
xl_cross_cov(f::MUKF)
Extract cross-covariance between xn and xl from joint covariance R.
"""
function xl_cross_cov(f::MUKF)
    return f.R[f.n_inds, f.l_inds]
end

function reset!(f::MUKF, d0 = f.d0)
    @bangbang f.x .= d0.μ
    @bangbang f.R .= d0.Σ
    # Extract nonlinear part for sigma points
    # xn0 = d0.μ[f.n_inds]
    # Rn0 = d0.Σ[f.n_inds, f.n_inds]
    # sigmapoints!(f.xn_sigma_points, xn0, Rn0, f.weight_params)
    f.t = 0
    f
end

# --- Helper functions for MUT (Marginalized Unscented Transform) ---


"""
    partition_cov(P, n_inds, l_inds)

Partition joint covariance P into blocks: Pnn, Pnl, Pln, Pll using index vectors.

# Arguments:
- `P`: Joint covariance matrix [nxn+nxl × nxn+nxl]
- `n_inds`: Index vector for nonlinear state (1:nxn), SVector if P is static
- `l_inds`: Index vector for linear state (nxn+1:nxn+nxl), SVector if P is static
"""
@views @inline function partition_cov(P, n_inds, l_inds)
    Pnn = P[n_inds, n_inds]
    Pnl = P[n_inds, l_inds]
    Pln = P[l_inds, n_inds]
    Pll = P[l_inds, l_inds]
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
    PC = cholesky(Symmetric(Pnn))
    L = Pln / PC
    # Compute conditional covariance
    Γ = Pll - Pln * (PC \ Pnl)
    return L, Γ
end

"""
    blockdiag(A, B)

Create block diagonal matrix [A 0; 0 B]
"""
function blockdiag(A::StaticMatrix{m,n}, B::StaticMatrix{p,q}) where {m,n,p,q}
    T = promote_type(eltype(A), eltype(B))
    return [[A zeros(SMatrix{m,q,T})]; [zeros(SMatrix{p,n,T}) B]]
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
    t::Real = index(f)*f.Ts;
    R1 = get_mat(f.R1, f.x[f.n_inds], u, p, t),
) where {IPD}
    nxn = length(f.n_inds)
    nxl = length(f.l_inds)
    nx = nxn + nxl

    # Get process noise covariance (full state)
    

    # Extract conditional parameters from current joint covariance
    Pnn, Pnl, Pln, Pll = partition_cov(f.R, f.n_inds, f.l_inds)
    L, Γ_curr = cond_linear_params(Pnn, Pnl, Pln, Pll)

    # Current means
    μn = f.x[f.n_inds]
    μl = f.x[f.l_inds]

    # Generate sigma points for nonlinear state only
    sp = f.sigma_point_cache.x0
    sigmapoints!(sp, μn, Pnn, f.weight_params)
    W = UKFWeights(f.weight_params, nxn)

    # Transform sigma points through dynamics
    # Yi = d(sp[i], u) + A_i*νB_i
    # where A_i = [An_i; Al_i] and νB_i = μl + L*(sp[i] - μn) is the conditional mean of xl given xn=sp[i]

    # Use cached arrays (no allocations!)
    Y = f.predict_cache.Y
    G_matrices = f.predict_cache.G_matrices
    xp = f.predict_cache.xp

    if IPD
        # In-place dynamics
        @inbounds for i in eachindex(sp)
            # Get state-dependent matrix A = [An; Al]
            A_i = get_mat(f.A, sp[i], u, p, t)
            G_matrices[i] = A_i

            # Conditional mean of xl given xn=sp[i]
            νB = μl .+ L * (sp[i] .- μn)

            # Get full dynamics [dn(xn, u); dl(xn, u)]
            xp .= 0
            f.dynamics(xp, sp[i], u, p, t)

            # Add coupling: x' = d(xn, u) + A*xl
            Y[i] = xp .+ A_i * νB
        end
    else
        # Out-of-place dynamics
        @inbounds for i in eachindex(sp)
            # Get state-dependent matrix A = [An; Al]
            A_i = get_mat(f.A, sp[i], u, p, t)
            G_matrices[i] = A_i

            νB = μl .+ L * (sp[i] .- μn)

            # Get full dynamics [dn(xn, u); dl(xn, u)]
            xp_full = f.dynamics(sp[i], u, p, t)

            # Add coupling: x' = d(xn, u) + A*xl
            Y[i] = xp_full .+ A_i * νB
        end
    end

    # Compute predicted mean
    μ_pred = zero(Y[1])
    @inbounds for i in eachindex(sp)
        w = i == 1 ? W.wm : W.wmi
        μ_pred = μ_pred .+ w .* Y[i]
    end

    # Compute spread covariance from transformed sigma points
    P_spread = zero(f.R)
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

    # Full predicted covariance (includes process noise)
    P_pred = P_spread .+ P_analytic .+ R1
    P_pred = symmetrize(P_pred)

    # Update state and covariance
    @bangbang f.x .= μ_pred
    @bangbang f.R .= P_pred

    f.t += 1
end

function correct!(
    f::MUKF{IPD,IPM},
    u,
    y,
    p = parameters(f),
    t::Real = index(f)*f.Ts;
    R2 = nothing,
) where {IPD,IPM}
    nxn = length(f.n_inds)
    nxl = length(f.l_inds)
    nx = nxn + nxl
    ny = f.ny

    # Get measurement noise covariance
    g = f.nl_measurement_model.measurement
    R2_mat = if R2 !== nothing
        get_mat(R2, f.x, u, p, t)
    else
        mm_R2 = f.nl_measurement_model.R2
        mm_R2 isa AbstractMatrix ? mm_R2 :
        (mm_R2 isa Function ? mm_R2(f.x[f.n_inds], u, p, t) : mm_R2.Σ)
    end

    # Extract conditional parameters from current joint covariance
    Pnn, Pnl, Pln, Pll = partition_cov(f.R, f.n_inds, f.l_inds)
    L, Γ_curr = cond_linear_params(Pnn, Pnl, Pln, Pll)

    # Current means
    μn = f.x[f.n_inds]
    μl = f.x[f.l_inds]
    μ_full = f.x

    # Generate sigma points for nonlinear state only
    sp = f.sigma_point_cache.x0
    sigmapoints!(sp, μn, Pnn, f.weight_params)
    W = UKFWeights(f.weight_params, nxn)

    # Transform sigma points through measurement model
    # yi = g(sp[i]) + Cl_i*νB_i
    # where νB_i = μl + L*(sp[i] - μn)

    # Use cached arrays (no allocations!)
    Y_meas = f.correct_sigma_point_cache.x1
    X_full = f.correct_cache.X_full
    Cl_matrices = f.correct_cache.Cl_matrices
    y_temp = f.correct_cache.y_temp

    if IPM
        # In-place measurement
        @inbounds for i in eachindex(sp)
            Cl_i = get_mat(f.Cl, sp[i], u, p, t)
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
            Cl_i = get_mat(f.Cl, sp[i], u, p, t)
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
        @bangbang Cl_avg .= Cl_avg .+ w .* Cl_matrices[i]
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
        @bangbang Σxy .+= w .* (δx * δy')
    end

    # Add the missing term: [0; Γ*Cl_avg'] (from equation 16 in MUT paper)
    # This is the conditional covariance contribution that was causing negative Γ!
    if Σxy isa SMatrix
        Σxy += [zero(SMatrix{length(f.n_inds), size(Σxy, 2)});Γ_curr * Cl_avg']
    else
        Σxy[f.l_inds, :] .+= Γ_curr * Cl_avg'
    end

    # Factorize S and compute Kalman gain
    Sᵪ = cholesky(Symmetric(S), check = false)
    if !issuccess(Sᵪ)
        error("Cholesky factorization of innovation covariance failed at time step $(f.t), got S = $(printarray(S))")
    end
    K = Σxy / Sᵪ

    # Innovation
    innovation = y .- yhat

    # Update full state and covariance
    μ_new = μ_full .+ K * innovation
    P_new = f.R .- K * S * K'
    P_new = symmetrize(P_new)

    # Update state and covariance
    @bangbang f.x .= μ_new
    @bangbang f.R .= P_new

    ll = extended_logpdf(SimpleMvNormal(PDMat(S, Sᵪ)), innovation)
    return (; ll, e = innovation, S, Sᵪ, K) 

end

# Make MUKF callable
(f::MUKF)(u, y, p=parameters(f), t=index(f)*f.Ts) = update!(f, u, y, p, t)

# Measurement function for MUKF
function measurement(mukf::MUKF)
    function (x, u, p, t)
        nxn = length(mukf.n_inds)
        xn = x[1:nxn]
        xl = x[nxn+1:end]

        g = mukf.nl_measurement_model.measurement
        Cl = get_mat(mukf.Cl, xn, u, p, t)

        return g(xn, u, p, t) .+ Cl * xl
    end
end

# Dynamics function for MUKF
function dynamics(mukf::MUKF)
    function (x, u, p, t)
        nxn = length(mukf.n_inds)
        xn = x[1:nxn]
        xl = x[nxn+1:end]

        # Get combined A matrix and extract An, Al by row indexing
        A_mat = get_mat(mukf.A, xn, u, p, t)
        An = A_mat[mukf.n_inds, :]  # First nxn rows
        Al = A_mat[mukf.l_inds, :]  # Remaining nxl rows

        dyn_full = mukf.dynamics(xn, u, p, t)
        xn_next = dyn_full[mukf.n_inds] .+ An * xl
        xl_next = dyn_full[mukf.l_inds] .+ Al * xl

        return [xn_next; xl_next]
    end
end

# Simulation support
function sample_state(f::MUKF, p=parameters(f); noise=true)
    return noise ? rand(f.d0) : f.d0.μ
end

function sample_state(f::MUKF{IPD}, x, u, p=parameters(f), t=index(f)*f.Ts; noise=true) where IPD
    nxn = length(f.n_inds)
    nxl = length(f.l_inds)
    xn = x[f.n_inds]
    xl = x[f.l_inds]

    # Get combined A matrix and extract An, Al by row indexing
    A_mat = get_mat(f.A, xn, u, p, t)
    An = A_mat[f.n_inds, :]  # First nxn rows
    Al = A_mat[f.l_inds, :]  # Remaining nxl rows

    # Get full dynamics [dn; dl]
    if IPD
        dyn_full = similar(x)
        f.dynamics(dyn_full, xn, u, p, t)
        xn_next = dyn_full[f.n_inds] .+ An * xl
        xl_next = dyn_full[f.l_inds] .+ Al * xl
    else
        dyn_full = f.dynamics(xn, u, p, t)
        xn_next = dyn_full[f.n_inds] .+ An * xl
        xl_next = dyn_full[f.l_inds] .+ Al * xl
    end

    if noise
        R1n_mat = get_mat(f.R1, xn, u, p, t)[f.n_inds, f.n_inds]  # Extract nonlinear block
        xn_next = xn_next .+ rand(SimpleMvNormal(R1n_mat))

        R1l_mat = get_mat(f.R1, xn, u, p, t)[f.l_inds, f.l_inds]  # Extract linear block
        xl_next = xl_next .+ rand(SimpleMvNormal(R1l_mat))
    end

    return [xn_next; xl_next]
end

function sample_measurement(f::MUKF, x, u, p=parameters(f), t=index(f)*f.Ts; noise=true)
    nxn = length(f.n_inds)
    xn = x[1:nxn]
    xl = x[nxn+1:end]

    g = f.nl_measurement_model.measurement
    Cl = get_mat(f.Cl, xn, u, p, t)

    y = g(xn, u, p, t) .+ Cl * xl
    if noise
        y = y .+ rand(f.nl_measurement_model.R2)
    end
    return y
end
