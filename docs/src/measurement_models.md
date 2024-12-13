# [Measurement models](@id measurement_models)
The Kalman-type filters
- [`KalmanFilter`](@ref)
- [`ExtendedKalmanFilter`](@ref)
- [`UnscentedKalmanFilter`](@ref)

each come with their own built-in measurement model, e.g., the standard [`KalmanFilter`](@ref) uses the linear measurement model ``y = Cx + Du + e``, while the [`ExtendedKalmanFilter`](@ref) and [`UnscentedKalmanFilter`](@ref) use the nonlinear measurement model ``y = h(x,u,p,t) + e`` or ``y = h(x,u,p,t,e)``. For covariance propagation, the [`ExtendedKalmanFilter`](@ref) uses linearization to approximate the nonlinear measurement model, while the [`UnscentedKalmanFilter`](@ref) uses the unscented transform.

It is sometimes useful to mix and match dynamics and measurement models. For example, using the unscented transform from the UKF for the dynamics update ([`predict!`](@ref)), but the linear measurement model from the standard [`KalmanFilter`](@ref) for the measurement update ([`correct!`](@ref)) if the measurement model is linear.

This is possible by constructing a filter with an explicitly created measurement model. The available measurement models are
- [`LinearMeasurementModel`](@ref) performs linear propagation of covariance (as is done in [`KalmanFilter`](@ref)).
- [`EKFMeasurementModel`](@ref) uses linearization to propagate covariance (as is done in [`ExtendedKalmanFilter`](@ref)).
- [`UKFMeasurementModel`](@ref) uses the unscented transform to propagate covariance (as is done in [`UnscentedKalmanFilter`](@ref)).
- [`CompositeMeasurementModel`](@ref) combines multiple measurement models.

## Constructing a filter with a custom measurement model

Constructing a Kalman-type filter automatically creates a measurement model of the corresponding type, given the functions/matrices passed to the filter constructor. To construct a filter with a non-standard measurement model, e.g., and UKF with a KF measurement model, manually create the desired measurement model and pass it as the second argument to the constructor. For example, to construct an UKF with a linear measurement model, we do
```@example MEASUREMENT_MODELS
using LowLevelParticleFilters, LinearAlgebra
nx = 100    # Dimension of state
nu = 2      # Dimension of input
ny = 90     # Dimension of measurements

# Define linear state-space system
const __A = 0.1*randn(nx, nx)
const __B = randn(nx, nu)
const __C = randn(ny,nx)
function dynamics_ip(dx,x,u,p,t)
    # __A*x .+ __B*u
    mul!(dx, __A, x)
    mul!(dx, __B, u, 1.0, 1.0)
    nothing
end
function measurement_ip(y,x,u,p,t)
    # __C*x
    mul!(y, __C, x)
    nothing
end

R1 = I(nx)
R2 = I(ny)

mm_kf = LinearMeasurementModel(__C, 0, R2; nx, ny)
ukf = UnscentedKalmanFilter(dynamics_ip, mm_kf, R1; ny, nu)
```

When we create the filter with the custom measurement model, we do not pass the arguments that are associated with the measurement model to the filter constructor, i.e., we do not pass any measurement function, and not the measurement covariance matrix ``R_2``.


## Sensor fusion: Using several different measurement models
Above we constructed a filter with a custom measurement model, we can also pass a custom measurement model when we call `correct!`. This may be useful when, e.g., performing sensor fusion with sensors operating at different sample rates, or when parts of the measurement model are linear, and other parts are nonlinear.

The following example instantiates three different filters and three different measurement models. Each filter is updated with each measurement model, demonstrating that any combination of filter and measurement model can be used together.

```@example MEASUREMENT_MODELS
using LowLevelParticleFilters, LinearAlgebra
nx = 10    # Dimension of state
nu = 2     # Dimension of input
ny = 9     # Dimension of measurements

# Define linear state-space system
const __A = 0.1*randn(nx, nx)
const __B = randn(nx, nu)
const __C = randn(ny,nx)
function dynamics_ip(dx,x,u,p,t)
    # __A*x .+ __B*u
    mul!(dx, __A, x)
    mul!(dx, __B, u, 1.0, 1.0)
    nothing
end
function measurement_ip(y,x,u,p,t)
    # __C*x
    mul!(y, __C, x)
    nothing
end

R1 = I(nx) # Covariance matrices
R2 = I(ny)

# Construct three different filters
kf  = KalmanFilter(__A, __B, __C, 0, R1, R2)
ukf = UnscentedKalmanFilter(dynamics_ip, measurement_ip, R1, R2; ny, nu)
ekf = ExtendedKalmanFilter(dynamics_ip, measurement_ip, R1, R2; nu)

# Simulate some data
T    = 200 # Number of time steps
U = [randn(nu) for _ in 1:T]
x,u,y = LowLevelParticleFilters.simulate(kf, U) # Simulate trajectory using the model in the filter

# Construct three different measurement models
mm_kf = LinearMeasurementModel(__C, 0, R2; nx, ny)
mm_ekf = EKFMeasurementModel{Float64, true}(measurement_ip, R2; nx, ny)
mm_ukf = UKFMeasurementModel{Float64, true, false}(measurement_ip, R2; nx, ny)


mms = [mm_kf, mm_ekf, mm_ukf]
filters = [kf, ekf, ukf]

for mm in mms, filter in filters
    @info "Updating $(nameof(typeof(filter))) with measurement model $(nameof(typeof(mm)))"
    correct!(filter, mm, u[1], y[1]) # Pass the measurement model as the second argument to the correct! function if not using the measurement model built into the filter
end
nothing # hide
```

Since the dynamics in this particular example is in fact linear, we should get identical results for all three filters.
```@example MEASUREMENT_MODELS
using Test
@test kf.x ≈ ekf.x ≈ ukf.x
@test kf.R ≈ ekf.R ≈ ukf.R
```


## Video tutorial

A video demonstrating the use of multiple measurement models in a sensor-fusion context is available on YouTube:

```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/BLsJrW5XXcg?si=bkob76-uJj27-S80" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```
