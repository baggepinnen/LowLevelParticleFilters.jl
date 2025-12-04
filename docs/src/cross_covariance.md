# Cross-covariance between process and measurement noise

In standard Kalman filter formulations, the process noise ``w`` and measurement noise ``v`` are assumed to be uncorrelated. However, in some applications, these noise sources may be correlated. This tutorial demonstrates how to model and account for cross-covariance between process and measurement noise using the `R12` parameter.

## Background

The standard discrete-time stochastic state-space model is
```math
\begin{aligned}
x_{k+1} &= f(x_k, u_k) + w_k \\
y_k &= h(x_k, u_k) + v_k
\end{aligned}
```
where ``w_k \sim N(0, R_1)`` is the process noise and ``v_k \sim N(0, R_2)`` is the measurement noise. The standard assumption is that ``E[w_k v_j^T] = 0`` for all ``k, j``.

When process and measurement noise are correlated, we have
```math
\text{Cov}\begin{pmatrix} w_k \\ v_k \end{pmatrix} = \begin{pmatrix} R_1 & R_{12} \\ R_{12}^T & R_2 \end{pmatrix}
```
where ``R_{12} = E[w_k v_k^T]`` is the cross-covariance matrix (following the notation in Simon's "Optimal State Estimation", Section 7.1).

This correlation can arise in several scenarios:
- When process and measurement noise share a common source
- In systems where a sensor measures a quantity that is directly affected by the process disturbance
- When discretizing continuous-time systems where process and measurement noise are correlated

## Measurement models with R12 support

The following measurement models support the `R12` cross-covariance parameter:
- [`EKFMeasurementModel`](@ref): For use with [`ExtendedKalmanFilter`](@ref)
- [`LinearMeasurementModel`](@ref): For linear measurement functions
- [`IEKFMeasurementModel`](@ref): For use with [`IteratedExtendedKalmanFilter`](@ref)

The [`UKFMeasurementModel`](@ref) does not support `R12` directly since the unscented transform uses sigma-point propagation rather than analytical covariance formulas. However, you can use an `EKFMeasurementModel` or `LinearMeasurementModel` with an [`UnscentedKalmanFilter`](@ref) to get R12 support while still using the unscented transform for the prediction step.

## Example: Scalar system with correlated noise

We demonstrate the effect of R12 using a simple scalar system:
```math
\begin{aligned}
x_{k+1} &= 0.8 x_k + w_k \\
y_k &= x_k + v_k
\end{aligned}
```

```@example R12
using DisplayAs # hide
using LowLevelParticleFilters, LinearAlgebra, Plots, Statistics, Random
using LowLevelParticleFilters: SimpleMvNormal
using StaticArrays

# System parameters
A = SA[0.8;;]
B = SA[0.0;;]
C = SA[1.0;;]
R1 = SA[1.0;;]   # Process noise covariance
R2 = SA[0.1;;]   # Measurement noise covariance
R12 = SA[0.25;;] # Cross-covariance

# Initial state distribution
d0 = SimpleMvNormal(SA[0.0], SA[1.0;;])

# Dynamics and measurement functions
dynamics(x, u, p, t) = A * x
measurement(x, u, p, t) = C * x
nothing # hide
```

## Comparing EKF with and without R12

We now create two Extended Kalman Filters: one that ignores the cross-covariance (standard approach) and one that accounts for it.

```@example R12
# EKF without R12 (ignoring correlation)
ekf_no_r12 = ExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; nu=1)

# EKF with R12
ekf_with_r12 = ExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; nu=1, R12=R12)

u = fill([], 100) # No control inputs

x, u, y = simulate(ekf_with_r12, u) # Simulate data using the filter with R12

# Run both filters
sol_no_r12 = forward_trajectory(ekf_no_r12, u, y)
sol_with_r12 = forward_trajectory(ekf_with_r12, u, y)
nothing # hide
```

## Comparing estimation performance

The filter that accounts for the cross-covariance achieves a lower steady-state estimation variance. This is expected because it correctly uses all available information about the noise correlation structure.

We can compare the actual estimation errors:

```@example R12
# Estimation errors
err_no_r12 = x .- sol_no_r12.xt |> stack
err_with_r12 = x .- sol_with_r12.xt |> stack

println("RMS error without R12: ", round(sqrt(mean(abs2, err_no_r12)), digits=4))
println("RMS error with R12: ", round(sqrt(mean(abs2, err_with_r12)), digits=4))
```

```@example R12
plot(err_no_r12', label="Without R12", xlabel="Time step", ylabel="Estimation error", alpha=0.7)
plot!(err_with_r12', label="With R12", alpha=0.7)
DisplayAs.PNG(Plots.current()) # hide
```

## Using R12 with UnscentedKalmanFilter

The [`UnscentedKalmanFilter`](@ref) uses sigma-point propagation and its native [`UKFMeasurementModel`](@ref) does not yet support R12. However, you can combine an UKF with an [`EKFMeasurementModel`](@ref) or [`LinearMeasurementModel`](@ref) to get R12 support in the correction step:

```@example R12
# Create an EKFMeasurementModel with R12
mm_ekf_r12 = EKFMeasurementModel{Float64, false}(measurement, R2; nx=1, ny=1, R12=R12)

# Create UKF with this measurement model
ukf_with_r12 = UnscentedKalmanFilter(dynamics, mm_ekf_r12, R1, d0; nu=1, ny=1)
```

Similarly, you can use a [`LinearMeasurementModel`](@ref) with R12:

```@example R12
mm_linear_r12 = LinearMeasurementModel(C, 0, R2; nx=1, ny=1, R12=R12)
ukf_linear_r12 = UnscentedKalmanFilter(dynamics, mm_linear_r12, R1, d0; nu=1, ny=1)
```


## Summary

When process and measurement noise are correlated:

1. Ignoring the correlation (setting R12=0) leads to suboptimal estimation with higher estimation-error variance.

2. The `R12` parameter can be specified in:
   - [`ExtendedKalmanFilter`](@ref)
   - [`EKFMeasurementModel`](@ref)
   - [`LinearMeasurementModel`](@ref)
   - [`IEKFMeasurementModel`](@ref)

3. To use R12 with [`UnscentedKalmanFilter`](@ref), provide an `EKFMeasurementModel`, `LinearMeasurementModel`, or `IEKFMeasurementModel` instead of the default `UKFMeasurementModel`.

4. The mathematical formulas used when R12 is present follow Simon's "Optimal State Estimation" Section 7.1:
   - Innovation covariance: ``S = C R C^T + C R_{12} + R_{12}^T C^T + R_2``
   - Kalman gain: ``K = (R C^T + R_{12}) S^{-1}``
   - Updated covariance: ``R^+ = (I - K C) R - K R_{12}^T``
