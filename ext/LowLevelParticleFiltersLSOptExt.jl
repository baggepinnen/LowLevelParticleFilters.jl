module LowLevelParticleFiltersLSOptExt

using LowLevelParticleFilters
import LowLevelParticleFilters: autotune_covariances, triangular, invtriangular, reconstruct_filter
using LowLevelParticleFilters: AbstractKalmanFilteringSolution, forward_trajectory, SimpleMvNormal, StaticCovMat
using LeastSquaresOptim
using ForwardDiff
using LinearAlgebra
using StaticArrays

"""
Helper function to compute Inverse-Wishart log-prior residuals for least-squares optimization.

For covariance Σ with Inverse-Wishart(v, Ψ) prior:
log p(Σ) ∝ -(v + n + 1)/2 * log|Σ| - 1/2 * tr(Ψ Σ⁻¹)

To use in least-squares, we express the negative log-prior as squared residuals.
"""
function inverse_wishart_residuals!(res_view, Σ, v, Ψ)
    n = size(Σ, 1)

    # Compute Σ⁻¹
    Σc = cholesky(Symmetric(Σ))

    # Term 1: log|Σ| contributes to logdet term
    # -(v + n + 1)/2 * log|Σ|
    # We add this as sqrt((v + n + 1)/2) * sqrt(-log|Σ|)
    # But since log|Σ| can be negative, we handle this via the logdet mechanism already in prediction_errors
    logdet_weight = sqrt((v + n + 1) / 2)
    res_view[1] = logdet_weight * sqrt(abs(logdet(Σc)))

    # Term 2: tr(Ψ Σ⁻¹) - trace inner product
    # -1/2 * tr(Ψ Σ⁻¹) = -1/2 * sum(Ψ .* Σ⁻¹)
    # Express as squared residuals: sqrt(1/2) * sqrt(sum(Ψ .* Σ⁻¹))
    trace_term = tr(Ψ / Σc)
    trace_weight = sqrt(0.5)
    # For numerical stability, we can also decompose this into individual elements
    # But simpler: use single residual for trace term
    res_view[2] = trace_weight * sqrt(trace_term)

    nothing
end

function autotune_covariances(
    sol::AbstractKalmanFilteringSolution;
    diagonal = true,
    optimize_x0 = false,
    offset = 0.0,
    optimizer = LevenbergMarquardt(),
    show_trace = true,
    show_every = 1,
    autodiff = :forward,
    v_R1 = nothing,
    v_R2 = nothing,
    kwargs...
)
    # Extract information from solution
    f = sol.f
    u = sol.u
    y = sol.y
    p = f.p

    # Get original covariances and initial state
    R1_orig = f.R1
    R2_orig = f.R2
    x0_orig = f.d0.μ

    nx = f.nx
    ny = f.ny
    nu = f.nu
    nw = size(R1_orig, 1)
    T = length(y)

    # Determine if we're working with static arrays
    is_static_R1 = R1_orig isa StaticCovMat
    is_static_R2 = R2_orig isa StaticCovMat

    # Validate MAP prior parameters
    use_map_R1 = v_R1 !== nothing
    use_map_R2 = v_R2 !== nothing

    if use_map_R1
        v_R1 <= nw - 1 && throw(ArgumentError("v_R1 must be > nw-1 = $(nw-1) for proper Inverse-Wishart prior, got $(v_R1)"))
    end

    if use_map_R2
        v_R2 <= ny - 1 && throw(ArgumentError("v_R2 must be > ny-1 = $(ny-1) for proper Inverse-Wishart prior, got $(v_R2)"))
    end

    # Use initial covariances as prior mean: Ψ = (v - n - 1) * R_orig
    # This makes E[Σ] = Ψ/(v - n - 1) = R_orig
    Ψ_R1_use = use_map_R1 ? (v_R1 - nw - 1) * Matrix(R1_orig) : nothing
    Ψ_R2_use = use_map_R2 ? (v_R2 - ny - 1) * Matrix(R2_orig) : nothing

    # Initialize parameter vector
    if diagonal
        # Diagonal parametrization: optimize log of diagonal elements
        R1_diag = diag(R1_orig isa UpperTriangular ? R1_orig'R1_orig : R1_orig)
        R2_diag = diag(R2_orig isa UpperTriangular ? R2_orig'R2_orig : R2_orig)
        all(>(0), R1_diag) || error("All diagonal elements of R1 must be positive for log-parametrization, got $(R1_diag)")
        all(>(0), R2_diag) || error("All diagonal elements of R2 must be positive for log-parametrization, got $(R2_diag)")

        if optimize_x0
            θ0 = vcat(log.(R1_diag), log.(R2_diag), x0_orig)
        else
            θ0 = vcat(log.(R1_diag), log.(R2_diag))
        end

        n_R1_params = length(R1_diag)
        n_R2_params = length(R2_diag)
    else
        # Full parametrization: optimize triangular representation
        # For positive definiteness, we parametrize as R = T'T where T is upper triangular
        if R1_orig isa UpperTriangular
            T1 = R1_orig
        else
            # Compute Cholesky and get upper triangular
            T1 = cholesky(Symmetric(R1_orig isa SMatrix ? Matrix(R1_orig) : R1_orig)).U
        end

        if R2_orig isa UpperTriangular
            T2 = R2_orig
        else
            T2 = cholesky(Symmetric(R2_orig isa SMatrix ? Matrix(R2_orig) : R2_orig)).U
        end

        R1_tri = invtriangular(T1)
        R2_tri = invtriangular(T2)

        if optimize_x0
            θ0 = vcat(R1_tri, R2_tri, x0_orig)
        else
            θ0 = vcat(R1_tri, R2_tri)
        end

        n_R1_params = length(R1_tri)
        n_R2_params = length(R2_tri)
    end

    # Define residuals function for optimization
    function residuals!(res, θ::AbstractVector{Ty}) where Ty
        # Extract parameters
        x0i = optimize_x0 ? θ[end-nx+1:end] : Ty.(x0_orig)

        # Reconstruct covariances
        if diagonal
            R1_diag = exp.(θ[1:n_R1_params])
            R2_diag = exp.(θ[n_R1_params+1:n_R1_params+n_R2_params])
            R1i = is_static_R1 ? SMatrix{n_R1_params,n_R1_params}(Diagonal(SVector{n_R1_params}(R1_diag))) : Diagonal(R1_diag)
            R2i = is_static_R2 ? SMatrix{n_R2_params,n_R2_params}(Diagonal(SVector{n_R2_params}(R2_diag))) : Diagonal(R2_diag)
        else
            T1i = triangular(θ[1:n_R1_params])
            T2i = triangular(θ[n_R1_params+1:n_R1_params+n_R2_params])
            R1i = is_static_R1 ? SMatrix{nw,nw}(T1i'T1i) : T1i'T1i
            R2i = is_static_R2 ? SMatrix{ny,ny}(T2i'T2i) : T2i'T2i
        end

        # Reconstruct filter with new covariances
        fi = reconstruct_filter(f, R1i, R2i, x0i)

        # Compute likelihood terms
        ol_likelihood = T * (ny + 1)  # Offset for likelihood residuals

        if !use_map_R1 && !use_map_R2
            # Pure MLE: no priors, use full residual vector for likelihood
            try
                LowLevelParticleFilters.prediction_errors!(res, fi, u, y, p, loglik=true, offset=offset)
            catch
                res .= Inf
            end
        else
            # MAP: likelihood + priors
            try
                # Likelihood residuals (first part of residual vector)
                LowLevelParticleFilters.prediction_errors!(@view(res[1:ol_likelihood]), fi, u, y, p, loglik=true, offset=offset)
            catch
                res[1:ol_likelihood] .= Inf
            end

            # Add Inverse-Wishart prior residuals
            idx = ol_likelihood + 1

            if use_map_R1
                # Convert R1i to Matrix for prior calculation
                R1i_mat = R1i isa AbstractMatrix ? Matrix(R1i) : R1i
                try
                    inverse_wishart_residuals!(@view(res[idx:idx+1]), R1i_mat, v_R1, Ψ_R1_use)
                catch
                    res[idx:idx+1] .= Inf
                end
                idx += 2
            end

            if use_map_R2
                # Convert R2i to Matrix for prior calculation
                R2i_mat = R2i isa AbstractMatrix ? Matrix(R2i) : R2i
                try
                    inverse_wishart_residuals!(@view(res[idx:idx+1]), R2i_mat, v_R2, Ψ_R2_use)
                catch
                    res[idx:idx+1] .= Inf
                end
            end
        end
        nothing
    end

    # Calculate output length
    output_length = T * (ny + 1)  # Base: likelihood terms
    if use_map_R1
        output_length += 2  # Add 2 residuals for R1 prior (logdet + trace terms)
    end
    if use_map_R2
        output_length += 2  # Add 2 residuals for R2 prior (logdet + trace terms)
    end

    res_opt = optimize!(
        LeastSquaresProblem(;
            x = Vector{Float64}(θ0),
            f! = residuals!,
            output_length,
            autodiff,
        ),
        optimizer;
        show_trace,
        show_every,
        kwargs...,
    )

    # Extract optimized parameters
    θ_opt = res_opt.minimizer
    x0_opt = optimize_x0 ? θ_opt[end-nx+1:end] : x0_orig

    if diagonal
        R1_diag_opt = exp.(θ_opt[1:n_R1_params])
        R2_diag_opt = exp.(θ_opt[n_R1_params+1:n_R1_params+n_R2_params])
        R1_opt = is_static_R1 ? SMatrix{nw,nw}(Diagonal(SVector{nw}(R1_diag_opt))) : Diagonal(R1_diag_opt)
        R2_opt = is_static_R2 ? SMatrix{ny,ny}(Diagonal(SVector{ny}(R2_diag_opt))) : Diagonal(R2_diag_opt)
    else
        T1_opt = triangular(θ_opt[1:n_R1_params])
        T2_opt = triangular(θ_opt[n_R1_params+1:n_R1_params+n_R2_params])
        R1_opt = is_static_R1 ? SMatrix{nw,nw}(T1_opt'T1_opt) : T1_opt'T1_opt
        R2_opt = is_static_R2 ? SMatrix{ny,ny}(T2_opt'T2_opt) : T2_opt'T2_opt
    end

    # Create optimized filter
    f_opt = reconstruct_filter(f, R1_opt, R2_opt, x0_opt)
    sol_opt = forward_trajectory(f_opt, u, y, p)

    return (;
        filter = f_opt,
        result = res_opt,
        R1 = R1_opt,
        R2 = R2_opt,
        x0 = x0_opt,
        sol_opt,
    )
end

end
