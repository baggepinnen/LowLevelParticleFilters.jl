# Parameter estimation utilities for LowLevelParticleFilters
# This file provides helper functions for automatic tuning of filter parameters

using LinearAlgebra
using StaticArrays
using LowLevelParticleFilters: KalmanFilteringSolution, AbstractKalmanFilter, KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, SqKalmanFilter, StaticCovMat


# Helper functions for triangular parametrization of covariance matrices
# These allow optimizing full covariance matrices while maintaining positive definiteness

"""
    triangular(x)

Convert a vector of parameters into an upper triangular matrix.
The length of `x` should be n(n+1)/2 for an n×n matrix.

# Example
```julia
x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]  # 6 parameters for 3×3 matrix
T = triangular(x)  # Returns 3×3 upper triangular matrix
```
"""
function triangular(x)
    m = length(x)
    n = round(Int, (-1 + sqrt(1 + 8m)) / 2)
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


"""
    reconstruct_filter(f, R1, R2, x0)

Reconstruct a filter with new covariance matrices and initial state.
Handles different filter types (KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter, etc.)
"""
function reconstruct_filter(f::KalmanFilter, R1, R2, x0)
    T = eltype(R1)
    d0_new = SimpleMvNormal(x0, T.(f.d0.Σ))
    KalmanFilter(
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
    T = eltype(R1)
    d0_new = SimpleMvNormal(x0, T.(f.d0.Σ))
    ExtendedKalmanFilter(
        f.dynamics, f.measurement,
        R1, R2, d0_new;
        nu = f.nu,
        p = f.p,
        Ts = f.Ts,
        α = f.α,
        check = false,
        names = f.names
    )
end

function reconstruct_filter(f::UnscentedKalmanFilter{T1,T2,T3,T4}, R1, R2, x0) where {T1,T2,T3,T4}
    T = eltype(R1)
    d0_new = SimpleMvNormal(x0, T.(f.d0.Σ))
    UnscentedKalmanFilter{T1,T2,T3,T4}(
        f.dynamics, f.measurement,
        R1, R2, d0_new;
        nu = f.nu,
        ny = f.ny,
        p = f.p,
        Ts = f.Ts,
        weight_params = f.weight_params,
        names = f.names
    )
end


"""
    autotune_covariances(
        sol::KalmanFilteringSolution;
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

Automatically tune the covariance matrices R1 and R2 (and optionally x0) of a Kalman-style filter
by maximizing the log-likelihood (MLE) or log-posterior (MAP) using Gauss-Newton optimization.

!!! info "Requires LeastSquaresOptim.jl"
    This function is available only if LeastSquaresOptim.jl is manually installed and loaded by the user.
    Install with: `using Pkg; Pkg.add("LeastSquaresOptim")`

# Arguments
- `sol::KalmanFilteringSolution`: Solution object from `forward_trajectory`
- `diagonal::Bool`: If true (default), only optimize diagonal elements. If false, optimize full covariance matrices.
- `optimize_x0::Bool`: If true, also optimize the initial state estimate (default: false)
- `offset::Real`: Offset added to the log-likelihood residuals to ensure positive squared residuals (default: 0.0). If you encounter an error about negative squared residuals during optimization, try increasing this value by an amount slightly larger than what is indicated in the error message.
- `optimizer`: Optimization algorithm from LeastSquaresOptim (default: LevenbergMarquardt())
- `show_trace::Bool`: Show optimization progress (default: true)
- `show_every::Int`: Show progress every N iterations (default: 1)
- `autodiff`: Automatic differentiation method (default: :forward)
- `v_R1::Union{Nothing,Real}`: Degrees of freedom for Inverse-Wishart prior on R1 (default: nothing, no prior). Must be > nw-1 for proper prior, where nw = size(R1,1). The prior mean is automatically set to the initial R1 from the filter.
- `v_R2::Union{Nothing,Real}`: Degrees of freedom for Inverse-Wishart prior on R2 (default: nothing, no prior). Must be > ny-1 for proper prior. The prior mean is automatically set to the initial R2 from the filter.
- `kwargs...`: Additional keyword arguments passed to LeastSquaresOptim.optimize!

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
- `v` (degrees of freedom): Controls prior strength. Larger v = stronger prior. Must be > n-1.
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
- For full parametrization, a triangular (Cholesky-like) parametrization is used
- MAP estimation adds Inverse-Wishart prior terms to the objective function
- The prior mean is the initial covariance matrix from the filter, regularizing toward the initial guess
- The `offset` parameter is passed to `prediction_errors!` and shifts the log-likelihood residuals
- When using MAP, the optimized covariances balance fit to data (likelihood) and prior belief (prior)
- x0 optimization uses MLE only (no prior on initial state)
"""
function autotune_covariances end
