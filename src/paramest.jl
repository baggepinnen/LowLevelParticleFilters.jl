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
        kwargs...
    )

Automatically tune the covariance matrices R1 and R2 (and optionally x0) of a Kalman-style filter
by maximizing the log-likelihood using Gauss-Newton optimization.

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
- `kwargs...`: Additional keyword arguments passed to LeastSquaresOptim.optimize!

# Returns
A named tuple containing:
- `filter`: The filter with optimized covariance matrices (and x0 if applicable)
- `result`: The optimization result from LeastSquaresOptim
- `R1`: The optimized process noise covariance
- `R2`: The optimized measurement noise covariance
- `x0`: The optimized initial state (if `optimize_x0=true`)
- `sol_opt`: The solution from running `forward_trajectory` with the optimized filter

# Example
```julia
using LeastSquaresOptim  # Must be loaded explicitly

# After running forward_trajectory
sol = forward_trajectory(kf, u, y)

# Tune covariances automatically
result = autotune_covariances(sol)

# If you get an error about negative squared residuals, increase the offset
result = autotune_covariances(sol, offset=20.0)

# Use the optimized filter
kf_opt = result.filter
@show result.result.converged
@show sol.ll  # Original log-likelihood
@show result.sol_opt.ll  # Optimized log-likelihood
```

# Notes
- The function uses log-likelihood optimization via `prediction_errors!` with `loglik=true`
- For diagonal parametrization, log-diagonal elements are optimized to ensure positivity
- For full parametrization, a triangular (Cholesky-like) parametrization is used
- The `offset` parameter is passed to `prediction_errors!` and shifts the log-likelihood residuals. This is necessary because the Gauss-Newton optimization requires the cost function to be expressible as a sum of squared residuals, but log-likelihood terms (especially logdet terms) can be negative. The offset does not affect the location of the optimum, only the reported cost function value.
"""
function autotune_covariances end
