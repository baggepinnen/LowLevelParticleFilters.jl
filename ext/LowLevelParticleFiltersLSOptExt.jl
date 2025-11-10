module LowLevelParticleFiltersLSOptExt

using LowLevelParticleFilters
import LowLevelParticleFilters: autotune_covariances, triangular, invtriangular, reconstruct_filter
using LowLevelParticleFilters: KalmanFilteringSolution, forward_trajectory, SimpleMvNormal, StaticCovMat
using LeastSquaresOptim
using ForwardDiff
using LinearAlgebra
using StaticArrays

function autotune_covariances(
    sol::KalmanFilteringSolution;
    diagonal = true,
    optimize_x0 = false,
    offset = 0.0,
    optimizer = LevenbergMarquardt(),
    show_trace = true,
    show_every = 1,
    autodiff = :forward,
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
    function residuals!(res, θ::AbstractVector{T}) where T
        # Extract parameters
        x0i = optimize_x0 ? θ[end-nx+1:end] : T.(x0_orig)

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

        # Compute prediction errors with loglik
        try
            LowLevelParticleFilters.prediction_errors!(res, fi, u, y, p, loglik=true, offset=offset)
        catch
            res .= Inf
        end
        nothing
    end

    # Run optimization
    output_length = T * (ny + 1)

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
