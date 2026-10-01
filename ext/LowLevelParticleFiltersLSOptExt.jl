module LowLevelParticleFiltersLSOptExt

using LowLevelParticleFilters
import LowLevelParticleFilters: autotune_covariances, reconstruct_filter, cov_from_logchol, factor_from_logchol, logchol_from_cov
using LowLevelParticleFilters: AbstractKalmanFilteringSolution, forward_trajectory, SimpleMvNormal, StaticCovMat, SqKalmanFilter
using LeastSquaresOptim
using ForwardDiff
using LinearAlgebra
using StaticArrays

# Added inside the square root of the diagonal prior residuals to keep their derivative finite where the residual is zero
const IW_EPS = 1e-12

"""
    inverse_wishart_residuals!(r, Σ, v, Lmode)

Residuals of the Inverse-Wishart(v, Ψ) prior on the `n×n` covariance matrix `Σ`, such that `r'r = -logpdf(InverseWishart(v, Ψ), Σ) + const`. `Lmode` is the lower Cholesky factor of the mode `Σmode = Ψ/(v+n+1)` of the prior, and `r` has length `n(n+1)/2`.

With `Σ = LL'` and `G = L⁻¹Lmode`, the negative log density is, up to a constant,
`(v+n+1)/2 * (sum(G[i,j]^2 for i ≠ j) + sum(φ(G[i,i])))`, where `φ(g) = g^2 - 1 - 2log(g) ≥ 0`. Each term is represented by one residual, which yields a full-rank Gauss-Newton approximation of the Hessian of the prior term.
"""
function inverse_wishart_residuals!(r, Σ, v, Lmode)
    n = size(Σ, 1)
    c = v + n + 1
    L = cholesky(Symmetric(Matrix(Σ))).L
    G = L \ Lmode
    k = 0
    for j in 1:n, i in j:n
        k += 1
        if i == j
            g = G[i, i]
            φ = g^2 - 1 - 2log(g)
            r[k] = sign(g - 1)*sqrt(c/2*max(φ, zero(φ)) + IW_EPS)
        else
            r[k] = sqrt(c/2)*G[i, j]
        end
    end
    r
end

_isstatic(R) = R isa StaticCovMat
_covariance(R::UpperTriangular) = R'R # Covariance factors are stored as UpperTriangular, e.g., by SqKalmanFilter
_covariance(R) = R

"""
    default_offset(sol)

Offset for the log-determinant residuals of `prediction_errors!(...; loglik = true)`, computed from the innovation covariances stored in the solution `sol`, with a margin that permits the innovation covariance of each output to decrease by several orders of magnitude during optimization.
"""
function default_offset(sol)
    ny = sol.f.ny
    cmin = Inf
    for S in sol.S
        (S === missing || S === nothing) && continue
        cmin = min(cmin, 0.5*(logdet(S) + ny*log(2π)))
    end
    isfinite(cmin) || return 10.0*ny
    max(0.0, -cmin) + 10.0*ny
end

"""
    autotune_setup(sol; diagonal, optimize_x0, offset, v_R1, v_R2)

Collect the information required to evaluate the residuals of [`autotune_covariances`](@ref), returned as a named tuple that includes the initial parameter vector `θ0` and the residual length `output_length`.
"""
function autotune_setup(sol::AbstractKalmanFilteringSolution; diagonal = true, optimize_x0 = false, offset = nothing, v_R1 = nothing, v_R2 = nothing)
    offset = something(offset, default_offset(sol))
    f = sol.f
    R1_orig = f.R1
    R2_orig = f.R2
    (R1_orig isa AbstractMatrix && R2_orig isa AbstractMatrix) || throw(ArgumentError("autotune_covariances requires the covariance matrices R1 and R2 of the filter to be matrices, got $(typeof(R1_orig)) and $(typeof(R2_orig))"))
    # Filters that store covariance factors receive factors from the parameterization
    uses_factors = f isa SqKalmanFilter
    Σ1 = Matrix(_covariance(R1_orig))
    Σ2 = Matrix(_covariance(R2_orig))
    nw = size(Σ1, 1)
    ny = size(Σ2, 1)
    nx = f.nx
    T = length(sol.y)
    x0_orig = f.d0.μ

    use_map_R1 = v_R1 !== nothing
    use_map_R2 = v_R2 !== nothing
    use_map_R1 && v_R1 <= nw + 1 && throw(ArgumentError("v_R1 must be > nw+1 = $(nw+1) for the mean of the Inverse-Wishart prior to exist, got $(v_R1)"))
    use_map_R2 && v_R2 <= ny + 1 && throw(ArgumentError("v_R2 must be > ny+1 = $(ny+1) for the mean of the Inverse-Wishart prior to exist, got $(v_R2)"))

    # The prior mean equals the initial covariance: Ψ = (v - n - 1) R_orig, and the prior mode is Ψ/(v + n + 1)
    L1mode = use_map_R1 ? cholesky(Symmetric((v_R1 - nw - 1)/(v_R1 + nw + 1) * Σ1)).L : nothing
    L2mode = use_map_R2 ? cholesky(Symmetric((v_R2 - ny - 1)/(v_R2 + ny + 1) * Σ2)).L : nothing

    if diagonal
        R1_diag = diag(Σ1)
        R2_diag = diag(Σ2)
        all(>(0), R1_diag) || error("All diagonal elements of R1 must be positive for log-parametrization, got $(R1_diag)")
        all(>(0), R2_diag) || error("All diagonal elements of R2 must be positive for log-parametrization, got $(R2_diag)")
        θ1 = log.(R1_diag)
        θ2 = log.(R2_diag)
    else
        θ1 = try
            logchol_from_cov(Σ1)
        catch err
            err isa ArgumentError || rethrow()
            throw(ArgumentError("The full parameterization of autotune_covariances requires a positive-definite initial R1. A singular R1, e.g., R1 = Bw*Σw*Bw' with fewer noise inputs than state variables, can be tuned with `diagonal = true`, or by letting the noise enter through a noise input of the dynamics of an UnscentedKalmanFilter with augmented dynamics, in which case R1 = Σw is tuned. Original error: $(err.msg)"))
        end
        θ2 = logchol_from_cov(Σ2)
    end
    θ0 = optimize_x0 ? vcat(θ1, θ2, x0_orig) : vcat(θ1, θ2)
    n1 = length(θ1)
    n2 = length(θ2)
    output_length = T*(f.ny + 1) + use_map_R1*(nw*(nw+1) ÷ 2) + use_map_R2*(ny*(ny+1) ÷ 2)

    (; f, u = sol.u, y = sol.y, p = f.p, diagonal, optimize_x0, offset, offset_too_small = Ref(false), uses_factors,
        static1 = _isstatic(R1_orig), static2 = _isstatic(R2_orig), nw, ny, nx, T, n1, n2, x0_orig,
        use_map_R1, use_map_R2, v_R1, v_R2, L1mode, L2mode, θ0, output_length)
end

function _unpack_cov(θ, n, diagonal, static, uses_factors)
    if diagonal
        d = exp.(θ)
        if uses_factors
            U = static ? SMatrix{n,n}(Diagonal(SVector{n}(sqrt.(d)))) : Matrix(Diagonal(sqrt.(d)))
            return UpperTriangular(U)
        else
            return static ? SMatrix{n,n}(Diagonal(SVector{n}(d))) : Diagonal(d)
        end
    else
        if uses_factors
            return static ? factor_from_logchol(θ, Val(n)) : factor_from_logchol(θ)
        else
            return static ? cov_from_logchol(θ, Val(n)) : cov_from_logchol(θ)
        end
    end
end

"""
    R1, R2, x0 = autotune_unpack(θ, s)

Map the parameter vector `θ` to the covariance matrices (or covariance factors, for filters that store factors) and the initial state mean, using the setup `s` from [`autotune_setup`](@ref).
"""
function autotune_unpack(θ, s)
    x0 = s.optimize_x0 ? θ[end-s.nx+1:end] : eltype(θ).(s.x0_orig)
    R1 = _unpack_cov(θ[1:s.n1], s.nw, s.diagonal, s.static1, s.uses_factors)
    R2 = _unpack_cov(θ[s.n1+1:s.n1+s.n2], s.ny, s.diagonal, s.static2, s.uses_factors)
    R1, R2, x0
end

"""
    autotune_residuals!(res, θ, s)

Residuals of the negative log-likelihood (or negative log-posterior) used by [`autotune_covariances`](@ref), such that `res'res = -loglik - logprior + const`.
"""
function autotune_residuals!(res, θ, s)
    R1i, R2i, x0i = autotune_unpack(θ, s)
    fi = reconstruct_filter(s.f, R1i, R2i, x0i)
    ol = s.T*(s.f.ny + 1)
    try
        LowLevelParticleFilters.prediction_errors!(@view(res[1:ol]), fi, s.u, s.y, s.p, loglik=true, offset=s.offset)
    catch err
        err isa ErrorException && occursin("Increase the offset", err.msg) && (s.offset_too_small[] = true)
        res[1:ol] .= Inf
    end
    idx = ol
    for (use, v, Lmode, Ri, n) in ((s.use_map_R1, s.v_R1, s.L1mode, R1i, s.nw), (s.use_map_R2, s.v_R2, s.L2mode, R2i, s.ny))
        use || continue
        m = n*(n+1) ÷ 2
        r = @view(res[idx+1:idx+m])
        try
            inverse_wishart_residuals!(r, _covariance(Ri), v, Lmode)
        catch
            r .= Inf
        end
        idx += m
    end
    res
end

function autotune_covariances(
    sol::AbstractKalmanFilteringSolution;
    diagonal = true,
    optimize_x0 = false,
    offset = nothing,
    optimizer = LevenbergMarquardt(),
    show_trace = true,
    show_every = 1,
    autodiff = :forward,
    v_R1 = nothing,
    v_R2 = nothing,
    kwargs...
)
    s = autotune_setup(sol; diagonal, optimize_x0, offset, v_R1, v_R2)

    res_opt = optimize!(
        LeastSquaresProblem(;
            x = Vector{Float64}(s.θ0),
            f! = (res, θ) -> autotune_residuals!(res, θ, s),
            output_length = s.output_length,
            autodiff,
        ),
        optimizer;
        show_trace,
        show_every,
        kwargs...,
    )

    if s.offset_too_small[]
        @warn "The log-likelihood could not be evaluated for some parameter values during the optimization since the offset ($(s.offset)) was too small, the result may not be a maximum of the likelihood. Pass a larger value of the keyword argument `offset` to autotune_covariances."
    end
    θ_opt = res_opt.minimizer
    R1i, R2i, x0_opt = autotune_unpack(θ_opt, s)
    x0_opt = optimize_x0 ? x0_opt : s.x0_orig
    f_opt = reconstruct_filter(s.f, R1i, R2i, x0_opt)
    sol_opt = forward_trajectory(f_opt, s.u, s.y, s.p)

    return (;
        filter = f_opt,
        result = res_opt,
        R1 = _covariance(R1i),
        R2 = _covariance(R2i),
        x0 = x0_opt,
        sol_opt,
    )
end

end
