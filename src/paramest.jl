# Parameter estimation utilities for LowLevelParticleFilters
# This file provides helper functions for automatic tuning of filter parameters

using LinearAlgebra
using StaticArrays
using LowLevelParticleFilters: KalmanFilteringSolution, AbstractKalmanFilter, KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, SqKalmanFilter, StaticCovMat


# Helper functions for triangular parametrization of covariance matrices
# These allow optimizing full covariance matrices while maintaining positive definiteness

"Return `n` such that `m == n(n+1)/2`, or throw an `ArgumentError` if no such integer exists."
function _tri_dim(m::Integer)
    n = round(Int, (-1 + sqrt(1 + 8m)) / 2)
    n*(n+1) ÷ 2 == m || throw(ArgumentError("The length of the parameter vector must be n(n+1)/2 for an integer n, got length $m"))
    n
end

"Linear index of entry `(i, j)`, `i ≤ j`, in the row-wise vectorization of the upper triangle of an `n×n` matrix used by [`triangular`](@ref)."
_triind(i, j, n) = (i-1)*n - ((i-1)*(i-2)) ÷ 2 + j - i + 1

"""
    triangular(x)

Convert a vector of parameters into an upper triangular matrix.
The length of `x` should be n(n+1)/2 for an n×n matrix. The upper triangle is filled row by row.

See also [`invtriangular`](@ref) and [`LowLevelParticleFilters.cov_from_logchol`](@ref) for a parameterization of positive-definite matrices.

# Example
```julia
x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 6 parameters for 3×3 matrix
T = triangular(x)  # Returns 3×3 upper triangular matrix
```
"""
function triangular(x)
    m = length(x)
    n = _tri_dim(m)
    T = zeros(eltype(x), n, n)
    k = 1
    for i = 1:n, j = i:n
        T[i,j] = x[k]
        k += 1
    end
    T
end

"""
    invtriangular(T)

Convert an upper triangular matrix into a vector of parameters.
This is the inverse operation of `triangular`.

# Example
```julia
T = [1.0 2.0 3.0; 0.0 4.0 5.0; 0.0 0.0 6.0]
x = invtriangular(T)  # Returns [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
```
"""
invtriangular(T) = [T[i,j] for i = 1:size(T,1) for j = i:size(T,1)]

## Log-Cholesky parameterization ===============================================

_logchol_entry(θ, i, j, n, ::Type{T}) where T = i > j ? zero(T) : i == j ? exp(θ[_triind(i, j, n)]) : T(θ[_triind(i, j, n)])

"""
    U = factor_from_logchol(θ)
    U = factor_from_logchol(θ, Val(n))

Return the upper-triangular Cholesky factor `U` of the positive-definite matrix `R = U'U` parameterized by the vector `θ` of length `n(n+1)/2`. The diagonal entries of `U` are `exp.(θ)` of the corresponding parameters and the off-diagonal entries are equal to the parameters, with the upper triangle filled row by row as in [`triangular`](@ref). If `Val(n)` is given, the factor is backed by an `SMatrix`.

This map is a bijection between ``\\mathbb{R}^{n(n+1)/2}`` and the set of positive-definite matrices, which makes it suitable for unconstrained optimization of covariance matrices. The factor may be passed directly as the covariance factor to filters that store factors, such as [`SqKalmanFilter`](@ref).

See also [`LowLevelParticleFilters.cov_from_logchol`](@ref) and [`LowLevelParticleFilters.logchol_from_cov`](@ref).
"""
function factor_from_logchol(θ::AbstractVector)
    n = _tri_dim(length(θ))
    T = typeof(exp(zero(eltype(θ))))
    UpperTriangular([_logchol_entry(θ, i, j, n, T) for i in 1:n, j in 1:n])
end

function factor_from_logchol(θ::AbstractVector, ::Val{n}) where n
    length(θ) == n*(n+1) ÷ 2 || throw(ArgumentError("The length of the parameter vector must be n(n+1)/2 = $(n*(n+1) ÷ 2) for n = $n, got length $(length(θ))"))
    T = typeof(exp(zero(eltype(θ))))
    U = SMatrix{n,n,T}(ntuple(Val(n*n)) do l
        i = (l-1) % n + 1
        j = (l-1) ÷ n + 1
        _logchol_entry(θ, i, j, n, T)
    end)
    UpperTriangular(U)
end

"""
    R = cov_from_logchol(θ)
    R = cov_from_logchol(θ, Val(n))

Return the positive-definite matrix `R = U'U`, where `U = factor_from_logchol(θ)`, see [`LowLevelParticleFilters.factor_from_logchol`](@ref). If `Val(n)` is given, an `SMatrix` is returned. This function is suitable for optimization of full covariance matrices, e.g., the covariance matrices of a filter in maximum-likelihood estimation:
```julia
R1 = LowLevelParticleFilters.cov_from_logchol(θ[1:3], Val(2)) # 2×2 covariance matrix from 3 parameters
```
The inverse map is [`LowLevelParticleFilters.logchol_from_cov`](@ref).
"""
function cov_from_logchol(θ::AbstractVector)
    U = parent(factor_from_logchol(θ))
    R = U'U
    (R + R') ./ 2
end

function cov_from_logchol(θ::AbstractVector, n::Val)
    U = parent(factor_from_logchol(θ, n))
    R = U'U
    (R + R') ./ 2
end

"""
    θ = logchol_from_cov(R::AbstractMatrix)
    θ = logchol_from_cov(C::Cholesky)
    θ = logchol_from_cov(U::UpperTriangular)

Return the parameter vector `θ` such that `cov_from_logchol(θ) ≈ R`, see [`LowLevelParticleFilters.cov_from_logchol`](@ref). `R` must be positive definite. A `Cholesky` factorization, or an `UpperTriangular` matrix, is interpreted as the factor `U` of `R = U'U`, following the convention used for covariance factors in, e.g., [`SqKalmanFilter`](@ref).
"""
logchol_from_cov(R::AbstractMatrix) = logchol_from_cov(_cholesky_factor(R))
logchol_from_cov(C::Cholesky) = logchol_from_cov(C.U)

function logchol_from_cov(U::UpperTriangular)
    n = size(U, 1)
    # Rows with a negative diagonal entry are negated, which leaves U'U unchanged
    s = [sign(U[i,i]) for i in 1:n]
    any(iszero, s) && throw(ArgumentError("The covariance factor must have a nonzero diagonal, i.e., the covariance matrix must be positive definite"))
    [i == j ? log(abs(U[i,i])) : s[i]*U[i,j] for i in 1:n for j in i:n]
end

function _cholesky_factor(R::AbstractMatrix)
    C = cholesky(Symmetric(Matrix(R)); check = false)
    issuccess(C) || throw(ArgumentError("The covariance matrix must be positive definite to be represented by a log-Cholesky parameterization, got $(printarray(Matrix(R)))"))
    C.U
end


## Filter reconstruction =======================================================

_cov_eltype(R) = eltype(R)
_promote_eltype(R1, R2, x0) = float(promote_type(_cov_eltype(R1), _cov_eltype(R2), eltype(x0)))

"""
    reconstruct_filter(f, R1, R2, x0)

Reconstruct a filter with new covariance matrices `R1, R2` and initial state mean `x0`. The numeric type of the reconstructed filter is promoted to the element types of `R1`, `R2` and `x0`, which enables differentiation with respect to these quantities using, e.g., ForwardDiff. User-provided Jacobians and other functions of the filter are retained, while automatically generated Jacobians are regenerated for the new numeric type.

Methods are implemented for [`KalmanFilter`](@ref), [`SqKalmanFilter`](@ref) (where `R1, R2` may be covariance matrices or `UpperTriangular` covariance factors), [`ExtendedKalmanFilter`](@ref), [`IteratedExtendedKalmanFilter`](@ref) and [`UnscentedKalmanFilter`](@ref).
"""
function reconstruct_filter(f::KalmanFilter, R1, R2, x0)
    T = _promote_eltype(R1, R2, x0)
    d0_new = SimpleMvNormal(T.(x0), T.(f.d0.Σ))
    KalmanFilter(
        f.A, f.B, f.C, f.D,
        R1, R2, d0_new;
        Ts = f.Ts,
        p = f.p,
        α = f.α,
        check = false,
        nx = f.nx,
        ny = f.ny,
        nu = f.nu,
        names = f.names
    )
end

function reconstruct_filter(f::SqKalmanFilter, R1, R2, x0)
    T = _promote_eltype(R1, R2, x0)
    d0_new = SimpleMvNormal(T.(x0), T.(f.d0.Σ))
    SqKalmanFilter(
        f.A, f.B, f.C, f.D,
        R1, R2, d0_new;
        Ts = f.Ts,
        p = f.p,
        α = f.α,
        check = false,
        names = f.names
    )
end

function reconstruct_filter(f::ExtendedKalmanFilter, R1, R2, x0)
    T = _promote_eltype(R1, R2, x0)
    kf = reconstruct_filter(getfield(f, :kf), R1, R2, x0)
    mm = reconstruct_measurement_model(getfield(f, :measurement_model), R2, T, f.nx)
    Ajac = getfield(f, :Ajac)
    ExtendedKalmanFilter(kf, getfield(f, :dynamics), mm; Ajac = Ajac isa DefaultJacobian ? nothing : Ajac, names = f.names)
end

function reconstruct_filter(f::UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}, R1, R2, x0) where {IPD,IPM,AUGD,AUGM}
    T = _promote_eltype(R1, R2, x0)
    d0_new = SimpleMvNormal(T.(x0), T.(f.d0.Σ))
    mm = reconstruct_measurement_model(f.measurement_model, R2, T, length(f.x))
    UnscentedKalmanFilter{IPD,IPM,AUGD,AUGM}(
        f.dynamics, mm,
        R1, d0_new;
        nu = f.nu,
        ny = f.ny,
        p = f.p,
        Ts = f.Ts,
        reject = f.reject,
        state_mean = f.state_mean,
        state_cov = f.state_cov,
        cholesky! = f.cholesky!,
        weight_params = f.weight_params,
        R1x = f.R1x,
        names = f.names
    )
end

reconstruct_filter(f, args...) = throw(ArgumentError("Reconstruction of filters of type $(nameof(typeof(f))) with new covariance matrices is not supported. Supported filter types are KalmanFilter, SqKalmanFilter, ExtendedKalmanFilter, IteratedExtendedKalmanFilter and UnscentedKalmanFilter."))

_user_jacobian(J) = J isa DefaultJacobian ? nothing : J

reconstruct_measurement_model(mm::EKFMeasurementModel{IPM}, R2, T, nx) where IPM =
    EKFMeasurementModel{T,IPM}(mm.measurement, R2; nx, ny = mm.ny, Cjac = _user_jacobian(mm.Cjac), R12 = mm.R12)

reconstruct_measurement_model(mm::IEKFMeasurementModel{IPM}, R2, T, nx) where IPM =
    IEKFMeasurementModel{T,IPM}(mm.measurement, R2; nx, ny = mm.ny, Cjac = _user_jacobian(mm.Cjac), R12 = mm.R12, step = mm.step, maxiters = mm.maxiters, epsilon = mm.epsilon)

reconstruct_measurement_model(mm::LinearMeasurementModel, R2, T, nx) =
    LinearMeasurementModel(mm.C, mm.D, R2; ny = mm.ny, R12 = mm.R12)

function reconstruct_measurement_model(mm::UKFMeasurementModel{IPM,AUGM}, R2, T, nx) where {IPM,AUGM}
    UKFMeasurementModel{T,IPM,AUGM}(mm.measurement, R2;
        nx,
        ny = mm.ny,
        innovation = mm.innovation,
        mean = mm.mean,
        cov = mm.cov,
        cross_cov = mm.cross_cov,
        weight_params = mm.weight_params,
        static = mm.cache.x0[1] isa StaticArray,
    )
end

reconstruct_measurement_model(mm, args...) = throw(ArgumentError("Reconstruction of measurement models of type $(nameof(typeof(mm))) with a new covariance matrix is not supported."))


"""
    autotune_covariances(
        sol::KalmanFilteringSolution;
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

Automatically tune the covariance matrices R1 and R2 (and optionally x0) of a Kalman-style filter
by maximizing the log-likelihood (MLE) or log-posterior (MAP) using Gauss-Newton optimization.

!!! info "Requires LeastSquaresOptim.jl"
    This function is available only if LeastSquaresOptim.jl is manually installed and loaded by the user.
    Install with: `using Pkg; Pkg.add("LeastSquaresOptim")`

# Arguments
- `sol::KalmanFilteringSolution`: Solution object from `forward_trajectory`. The filter, the data and the parameters `p` of the filter stored in the solution are used. Missing measurements in the data are supported, see [`LowLevelParticleFilters.ismissing_measurement`](@ref).
- `diagonal::Bool`: If true (default), only optimize diagonal elements. If false, optimize full covariance matrices using a log-Cholesky parameterization, see [`LowLevelParticleFilters.cov_from_logchol`](@ref).
- `optimize_x0::Bool`: If true, also optimize the initial state estimate (default: false)
- `offset`: Offset added to the log-determinant terms of the log-likelihood residuals to ensure that they are the square roots of positive numbers, see [`LowLevelParticleFilters.prediction_errors!`](@ref). The offset shifts the cost function by a constant and does not affect the optimum. If `nothing` (default), the offset is computed from the innovation covariances of `sol`, with a margin that permits the determinant of the innovation covariance to decrease by several orders of magnitude during the optimization. If a warning indicates that the offset was too small, pass a larger value.
- `optimizer`: Optimization algorithm from LeastSquaresOptim (default: LevenbergMarquardt())
- `show_trace::Bool`: Show optimization progress (default: true)
- `show_every::Int`: Show progress every N iterations (default: 1)
- `autodiff`: Automatic differentiation method (default: :forward)
- `v_R1::Union{Nothing,Real}`: Degrees of freedom for Inverse-Wishart prior on R1 (default: nothing, no prior). Must be > nw+1, where nw = size(R1,1), for the mean of the prior to exist. The prior mean is automatically set to the initial R1 from the filter.
- `v_R2::Union{Nothing,Real}`: Degrees of freedom for Inverse-Wishart prior on R2 (default: nothing, no prior). Must be > ny+1. The prior mean is automatically set to the initial R2 from the filter.
- `kwargs...`: Additional keyword arguments passed to LeastSquaresOptim.optimize!

Supported filter types are [`KalmanFilter`](@ref), [`SqKalmanFilter`](@ref), [`ExtendedKalmanFilter`](@ref), [`IteratedExtendedKalmanFilter`](@ref) and [`UnscentedKalmanFilter`](@ref), see [`LowLevelParticleFilters.reconstruct_filter`](@ref). The covariance matrices `R1` and `R2` stored in the filter must be matrices (not functions). For an `UnscentedKalmanFilter` with augmented dynamics, `R1` is the covariance of the noise input `w` of the dynamics `f(x, u, p, t, w)`, and the structure implied by the noise input is thus retained during tuning.

# Returns
A named tuple containing:
- `filter`: The filter with optimized covariance matrices (and x0 if applicable)
- `result`: The optimization result from LeastSquaresOptim
- `R1`: The optimized process noise covariance
- `R2`: The optimized measurement noise covariance
- `x0`: The optimized initial state (if `optimize_x0=true`)
- `sol_opt`: The solution from running `forward_trajectory` with the optimized filter

# Maximum Likelihood Estimation (MLE)
By default (when `v_R1` and `v_R2` are `nothing`), performs maximum likelihood estimation:
```julia
using LeastSquaresOptim

sol = forward_trajectory(kf, u, y)
result = autotune_covariances(sol)  # Pure MLE
```

# Maximum A Posteriori (MAP) Estimation
Use Inverse-Wishart priors for Bayesian regularization. The Inverse-Wishart distribution is the conjugate prior
for covariance matrices. For a covariance matrix Σ with dimension n:

`p(Σ) = InverseWishart(v, Ψ)`

where:
- `v` (degrees of freedom): Controls prior strength. Larger v = stronger prior. Must be > n+1 for the prior mean to exist.
- Prior mean is automatically set to the initial covariance matrices (R1_orig and R2_orig) from the filter.
- Internally, the scale matrix is computed as: Ψ = (v - n - 1) * R_orig

The mean of the Inverse-Wishart prior is E[Σ] = Ψ/(v - n - 1) = R_orig.

Typical choices for v:
- Weak prior: `v = n + 2` (prior has low confidence, stays close to MLE)
- Moderate prior: `v = n + 5` to `n + 10`
- Strong prior: `v = n + 20` or higher (high confidence, stays close to initial guess)

```julia
# MAP with weak Inverse-Wishart prior on both R1 and R2
nx, ny = 2, 2
v1 = nx + 2  # Weak prior
v2 = ny + 2

result = autotune_covariances(sol; v_R1=v1, v_R2=v2)

# MAP with prior only on R1 (useful when measurement noise is well-known)
result = autotune_covariances(sol; v_R1=nx+5)

# Strong prior to prevent overfitting with limited data
v1_strong = nx + 20
result = autotune_covariances(sol; v_R1=v1_strong)
```

# Notes
- The function uses log-likelihood optimization via `prediction_errors!` with `loglik=true`
- For diagonal parametrization, log-diagonal elements are optimized to ensure positivity
- For full parametrization, a log-Cholesky parametrization is used, see [`LowLevelParticleFilters.cov_from_logchol`](@ref)
- MAP estimation adds the negative logarithm of the Inverse-Wishart prior density to the objective function, represented as one squared residual per prior
- The prior mean is the initial covariance matrix from the filter, regularizing toward the initial guess
- The `offset` parameter is passed to `prediction_errors!` and shifts the log-likelihood residuals
- When using MAP, the optimized covariances balance fit to data (likelihood) and prior belief (prior)
- x0 optimization uses MLE only (no prior on initial state)
"""
function autotune_covariances end
