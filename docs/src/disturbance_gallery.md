```@setup DISTGALLERY
using Plots
default(lab="")
```

# Disturbance gallery
Most filters in this package assume that the disturbances acting on the system are comprised of Gaussian white noise. This may at first appear as a severe limitation, but together with a dynamical model, this is a surprisingly flexible combination. Most disturbance models we list are linear, which means that they work for any state estimator, including standard Kalman filters. In the end, we also mention some nonlinear disturbance models that require a nonlinear state estimator, such as an [`UnscentedKalmanFilter`](@ref). For each disturbance model, we provide a statespace model and show a number of samples from the model, we also list a number of example scenarios where the model is useful. In many cases, models have an interpretation also in the Laplace domain or as a temporal Gaussian process. 


## Stochastic vs. deterministic but unknown
While some sources of errors are random, such as sensor noise, other sources of errors are deterministic but unknown. For example, a miscalibrated sensor is affected by a static but unknown error. We may communicate these properties to our state estimator by
1. Providing the _initial distribution_ of the state. If this is, e.g., a wide Gaussian distribution, we indicate that we are uncertain about the initial state. If the covariance is zero, we indicate that the initial state is perfectly known. The initial state distribution is usually denoted ``d_0`` in this documentation.
2. Providing the _covariance_ of the driving disturbance noise. If this is zero, the disturbance is deterministic and the uncertainty about it comes solely from the initial state distribution. If this is positive, the disturbance is random and the uncertainty about it comes from both the initial state distribution and the disturbance noise. Where distributions are assumed to be Gaussian, we refer to the covariance matrix of the dynamics noise as ``R_1`` and the measurement noise ``R_2``. When noises can take any distribution, we refer to these distributions as `df` and `dg` instead.

## White noise
This is the simplest possible disturbance model and require no dynamical system at all, just the driving white-noise input. Most state estimators in this package assume that the noise is Gaussian, but particle filters can also be used with non-Gaussian noise.

White noise has a flat spectrum (analogous to how white light contains all colors). 

### Samples
```@example DISTGALLERY
using Plots
w = randn(100)
plot(
    plot(w, label="Gaussian white noise"),
    histogram(w, title="Histogram"),
)
```

## Integrated white noise
This simplest dynamical disturbance model is white noise integrated once. This is a non-stationary process since the variance grows over time, which means that this model is suitable for disturbances that can have any arbitrary magnitude, but no particular properties of the evolution of the disturbance over time is known. This models is sometimes called a Brownian random walk, or a Wiener process.

### Model
**Continuous time**
```math
\dot{x} = w
```
**Discrete time**
```math
x[k+1] = x[k] + T_s w[k]
```

**Frequency domain**
```math
G(s) = \frac{1}{s}
```

### Samples
```@example DISTGALLERY
using ControlSystemsBase, Plots
Ts = 0.01 # Sampling time
sys = ss([1], [Ts], [1], 0, Ts) # Discrete-time integrator
res = map(1:10) do i
    w = randn(1, 1000) # White noise input
    lsim(sys, w)
end
figsim = plot(res)
plot!(res[1].t, 2 .* sqrt.(Ts .* res[1].t) .* [1 -1], label="2σ", color=:black, linestyle=:dash, linewidth=2) # Non-stationary process, variance is growing over time.
figspec = bodeplot(sys, plotphase=false)
figimp = plot(impulse(sys, 10), title="Impulse response")
plot(figsim, figspec, figimp, plot_title="Integrated white noise")
```

Note, the samples from this process do not look random and step like, but a random step-like process can nevertheless be well modeled by such a process (this is hinted at by the transfer function ``1/s`` which is identical to the Laplace transform of the step function). This model is used in a number of examples that demonstrate this property:
- [Joint state and parameter estimation](@ref)
- [Fault detection](@ref)
- [LQG control with integral action](https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/lqg_disturbance/)


### Suitable for
- Random-step like disturbances
- Friction
- Unknown change of operating point or set point
- Static or slowly varying calibration errors (if the error is completely static but initially unknown, use nonzero initial covariance but zero covariance for ``w``).
- Gyroscope drift


 

## Double integrated white noise
This is a second-order dynamical disturbance model that is white noise integrated twice. This is a non-stationary process since the variance grows over time, which means that this model is suitable for disturbances that can have any arbitrary magnitude, and where the evolution of the disturbance is subject to inertia, that is, the disturbance is expected to evolve be smoothly. 

### Model
**Continuous time**
```math
\ddot{x} = w
```
**Discrete time**
```math
x[k+1] = x[k] + v[k] \\
v[k+1] = v[k] + w[k]
```
**Frequency domain**
```math
G(s) = \frac{1}{s^2}
```
### Samples
```@example DISTGALLERY
using ControlSystemsBase, Plots
Ts = 0.01 # Sampling time
sys = ss([1 1; 0 1], [0; 1], [1 0], 0, Ts) # Discrete-time double integrator
res = map(1:10) do i
    w = randn(1, 1000) # White noise input
    lsim(sys, w)
end
figsim = plot(res)
figspec = bodeplot(sys, plotphase=false)
plot(figsim, figspec, plot_title="Double integrated white noise")
```

### Suitable for
- Random ramp-like disturbances
- Smoothly varying disturbances

## Low-pass filtered white noise
If we pass white noise through a low-pass filter, we get a signal that is random but primarily contains low frequencies. This is a stationary process, which means that the variance does not grow over time, and we can calculate the stationary covariance of the process by solving a Lyapunov equation. We do this below in order to indicate the stationary standard deviation of the process in the plot of the samples. This model is associated with a tuning parameter that determines the cutoff frequency of the low-pass filter, ```\tau``. 

### Model
**Continuous time**
```math
\dot{x} = -\frac{1}{\tau} x + w
```
**Discrete time**
```math
x[k+1] = (1 - \frac{1}{\tau}) x[k] + w[k]
```
### Transfer function 
```math
G(s) = \frac{1}{\tau s + 1}
```
### Samples
```@example DISTGALLERY
using ControlSystemsBase, Plots
Ts = 0.01 # Sampling time
τ = 1.0
sys = ss(c2d(tf(1, [τ, 1]), Ts)) # Discrete-time first-order low-pass filter
res = map(1:10) do i
    w = randn(1, 1000) # White noise input
    lsim(sys, w)
end
figsim = plot(res)
(; B,C) = sys
hline!(2*sqrt.(C*(lyap(sys, B*B'))*C') .* [1 -1], color=:black, linestyle=:dash, linewidth=2, label="2σ") # Stationary standard deviation
figspec = bodeplot(sys, plotphase=false)
figimp = plot(impulse(sys, 10), title="Impulse response")
plot(figsim, figspec, figimp, plot_title="Low-pass filtered white noise")
```

### Suitable for
- Stationary noise dominated by low frequencies

### Alternative names
- Ornstein–Uhlenbeck process
- Gaussian process: Exponential covariance function


## Higher-order low-pass filtered white noise
If we add more poles to the low-pass filter, we can model Gaussian processes with the Matérn covariance function with half-integer smoothness. The Matérn covariance with ``ν=1/2`` is equivalent to the first-order low-pass filter above, and with ``ν=3/2`` we get the model
```math
\begin{aligned}
A &= \begin{bmatrix}
0 & 1 \\
-\lambda^2 & -2 \lambda
\end{bmatrix} \\
B &= \begin{bmatrix}
0 \\ 1
\end{bmatrix} \\
C &= \begin{bmatrix}
1 & 0
\end{bmatrix} \\
\lambda &= \sqrt{3} / l
\end{aligned}
```
where ``l`` is the length scale of the covariance function.

### Samples from Matérn 3/2 covariance function
```@example DISTGALLERY
using ControlSystemsBase, Plots
Ts = 0.01 # Sampling time
l = 1.0 # Length scale
λ = sqrt(3) / l
A = [0 1; -λ^2 -2λ]
B = [0; 1]
C = [1 0]
sys = c2d(ss(A, B, C, 0), Ts)
res = map(1:10) do i
    w = randn(1, 1000) # White noise input
    lsim(sys, w)
end
figsim = plot(res)
(; B,C) = sys
hline!(2*sqrt.(C*(lyap(sys, B*B'))*C') .* [1 -1], color=:black, linestyle=:dash, linewidth=2, label="2σ") # Stationary standard deviation
figspec = bodeplot(sys, plotphase=false)
figimp = plot(impulse(sys, 10), title="Impulse response")
plot(figsim, figspec, figimp, plot_title="Low-pass (second order) filtered white noise")
```
Note how this produces smoother signals compared to the first-order low-pass filter. The Matérn covariance function with ``ν=5/2`` can be modeled by adding a third state to the system above, and so on.

For more details on the relation between temporal Gaussian processes and linear systems, see [section 3.3 in "Stochastic Differential Equation Methods for Spatio-Temporal Gaussian Process Regression", Arno Solin](https://aaltodoc.aalto.fi/server/api/core/bitstreams/aaf7725c-7955-4d21-8d31-e27fdd23c503/content).



## Periodic disturbance
If disturbances have a dominant frequency or period, such as 50Hz from the electrical grid, or 24hr from the sun, a periodic disturbance model can be used. A second-order linear system with a complex-conjugate pair of poles close to the imaginary axis has a resonance peak in the frequency response which is suitable for modeling periodic disturbances. If the disturbance is perfectly sinusoidal but the phase is unknown, we may indicate this by setting the covariance of the driving noise to zero and placing the poles exactly on the imaginary axis. 
### Model
**Continuous time**
```math
\begin{aligned}
\dot{x} &= \begin{bmatrix}
-\zeta & -\omega_0 \\
\omega_0 & -\zeta
\end{bmatrix} x + \begin{bmatrix}
\omega_0 \\
0
\end{bmatrix} w \\
y &= \begin{bmatrix}
0 & \omega_0
\end{bmatrix} x
\end{aligned}
```
**Frequency domain**
```math
G(s) = \frac{\omega_0}{s^2 + 2\zeta \omega_0 s + \omega_0^2}
```
### Samples
```@example DISTGALLERY
using ControlSystemsBase, Plots
ω0 = 2π/3 # Resonance frequency [rad/s]
ζ = 0.1 # Damping ratio, smaller value gives higher amplitude
Ts = 0.05
t = 0:Ts:20 # Time vector
sys = c2d(ss([-ζ -ω0; ω0 -ζ], [ω0; 0], [0 ω0], 0), Ts)
res = map(1:10) do i
    w = randn(1, length(t)) # White noise input
    lsim(sys, w, t)
end
figsim = plot(res)
(; B,C) = sys
hline!(2*sqrt.(C*(lyap(sys, B*B'))*C') .* [1 -1], color=:black, linestyle=:dash, linewidth=2) # Stationary standard deviation
figspec = bodeplot(sys, plotphase=false)
figimp = plot(impulse(sys, 10), title="Impulse response")
plot(figsim, figspec, figimp, plot_title="Periodic disturbance")
```

## One sided random bumps
This is a __nonlinear__ disturbance model that is useful when the disturbance is expected to be non-negative (or non-positive). The model is a combination of low-pass filtered white noise and a nonlinear integrator that integrates the low-pass filtered white noise only when it is positive. 
### Model
**Continuous time**
```math
\begin{aligned}
\dot{x} &= \begin{bmatrix}
-a x_1 + w \\
-b x_2 + \max(0, x_1)^n
\end{bmatrix} \\
y &= x_2
\end{aligned}
```

### Samples
Since this is a nonlinear model, we cannot use the `lsim` function to simulate it. Instead, we use the package [`SeeToDee.jl`](https://github.com/baggepinnen/SeeToDee.jl/) to discretize the nonlinear dynamics model, learn more under [Discretization](@ref).
```@example DISTGALLERY
using LowLevelParticleFilters, SeeToDee, Plots, Random
Ts = 0.1 # Sampling time
a = 1 # Low-pass filter (inverse) time constant, controls how often the bumps appear (higher value ⟹ more often)
b = 2 # Bump decay (inverse) time constant
n = 2 # Nonlinearity exponent

# Define dynamics function
function dynamics(x, w, p, t) # We assume that the noise is coming in through the second argument here. When using this model with an UnscentedKalmanFilter, we may instead add w as the 5:th argument and let the second argument be the control input.
    x1, x2 = x
    dx1 = -a * x1 + w[1]
    dx2 = -b * x2 + max(0.0, x1)^n
    return [dx1, dx2]
end
discrete_dynamics = SeeToDee.Rk4(dynamics, Ts)
# Measurement function (only observes x2)
function measurement(x, w, p, t)
    return [x[2]]
end
# Simulate the model
t = 0:Ts:20 # Time vector
x0 = [0.0; 0.0] # Initial state
Random.seed!(0) # For reproducibility
res = map(1:10) do i
    if i == 1
        w = [(j==0)/Ts for j in t] # Pulse input for first sample
    else
        w = [randn(1) for j in t] # White noise input
    end
    x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, w)
    reduce(hcat, measurement.(x[1:end-1], w, nothing, t))'
end
plot(t, res, title="One sided random bumps", lw=[5 ones(1,9)])
```
Note how the samples are all nonnegative, achieved by the nonlinearity. The first sample is the impulse response of the system, and this is drawn with a greater linewidth.

### Observability
This is a nonlinear model, but it is piecewise linear and we may use linear observability tests to check if the system is observable in each mode.
```@example DISTGALLERY
Ap = [-a 0; 1 -b]
An = [-a 0; 0 -b]
B = [1.0; 0]
C = [0 1]
sysp = ss(Ap, B, C, 0)
sysn = ss(An, B, C, 0)

observability(sysp)
```
When ``x_2`` is positive, the system is observable, but when ``x_2`` is negative
```@example DISTGALLERY
observability(sysn)
```
we have lost observability of the first state variable. This may pose a problem for, e.g., an ExtendedKalmanFilter, which performs linearization around the current state estimate. To mitigate the observability issue, we may change the nonlinearity to, e.g., a softplus function:
```@example DISTGALLERY
softplus(x, hardness=10) = log(1 + exp(hardness*x))/hardness # A softer version of ReLU

function dynamics(x, w, p, t) # We assume that the noise is coming in through the second argument here. When using this model with an UnscentedKalmanFilter, we may instead add w as the 5:th argument and let the second argument be the control input.
    x1, x2 = x
    dx1 = -a * x1 + w[1]
    dx2 = -b * x2 + softplus(x1)^n
    return [dx1, dx2]
end
discrete_dynamics = SeeToDee.Rk4(dynamics, Ts)
Random.seed!(0) # For reproducibility

res = map(1:10) do i
    w = [randn(1) for i in t] # White noise input
    x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, w)
    reduce(hcat, measurement.(x[1:end-1], w, nothing, t))'
end
plot(t, res, title="One sided random bumps (softplus)")
```
This produces a very similar result to the previous model, but adds a tunable hardness parameter that can trade off observability and tendency to output values that are closer to zero.

!!! note "Tip"
    The function `ControlSystemsBase.observability(f::AbstractKalmanFilter, x, u, p, t=0.0)` is overloaded for nonlinear state estimators from this package.

## One sided periodic bumps
This is similar to the previous model, but with a periodic disturbance driving the nonlinear integrator, causing the bumps to have a dominant period.

### Model
**Continuous time**
```math
\begin{aligned}
\dot{x} &= \begin{bmatrix}
x_2 \\
-2 \zeta \omega_0 x_2 - \omega_0^2 x_1 + w \\
-b x_3 + \max(0, x_1)^n
\end{bmatrix} \\
y &= x_3
\end{aligned}
```

### Samples
```@example DISTGALLERY
using SeeToDee, LowLevelParticleFilters, Plots
period = 24.0
ω = 2π / period # Resonance frequency [rad/s]
ζ = 0.1
b = 0.9 # Bump decay (inverse) time constant
n = 2 # Nonlinearity exponent
Ts = 0.1 # Sampling time

# Define dynamics function
function dynamics(x, w, p, t)
    x1, x2, x3 = x
    dx1 = x2
    dx2 = -2 * ζ * ω * x2 - ω^2 * x1 + w[1]
    relu_x1 = max(0.0, x1)
    dx3 = -b * x3 + relu_x1^n
    return [dx1, dx2, dx3]
end

discrete_dynamics = SeeToDee.Rk4(dynamics, Ts)

# Measurement function (only observes x3)
function measurement(x, u, p, t)
    return [x[3]]
end

# Simulate the model
t = 0:Ts:120 # Time vector
x0 = [0.0; 0.0; 0.0] # Initial state

res = map(1:10) do i
    if i == 1
        w = [2*(j==0)/Ts for j in t] # Pulse input for first sample
    else
        w = [randn(1) for j in t] # White noise input
    end
    x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, w)
    reduce(hcat, measurement.(x[1:end-1], w, nothing, t))'
end
plot(t, res, title="One sided periodic bumps", lw=[5 ones(1,9)])
```
The first sample is one possible impulse response of the system (the system is nonlinear and does not have a single unique impulse response), and this is drawn with a greater linewidth.

### Useful for
- One-sided periodic disturbances
- Example: Sunlight hitting a thermometer once per day, but only if it is sunny


## Deterministic disturbances
If there is a deterministic aspect to the disturbance, we may use the fact that dynamics and measurement functions (as well as Kalman filter matrices) may depend on time. For exactly, the perfectly deterministic measurement disturbance disturbance ``\sin(t)`` is easily modeled by including it in the measurement function. 
```julia
measurement(x, u, p, t) = ... + sin(t)
```

## Heteroschedastic disturbances
If the disturbance is heteroschedastic, i.e., the variance of the disturbance depends on time or on the state, we may easily indicate this to the state estimator by either
- Let the covariance matrix depend on time or state, applicable to all estimators.
- Encode varying the gain from disturbance to state/measurement in the corresponding dynamics/measurement function, applicable to nonlinear state estimators only.

## Non-Gaussian driving noise
All Kalman-type estimators assume that the driving noise is Gaussian. Particle filters are not limited to this assumption and can generally be used with any distribution that can be sampled from, see [Smoothing the track of a moving beetle](@ref) for an example, where the mode is affected by Binomial noise.


## Dynamical models of measurement disturbance
When using any of the _dynamical_ models above to model _measurement disturbances_, the noise driving the disturbance dynamics must be sourced from the dynamics noise, e.g., for a Kalman filter for the model
```math
\begin{aligned}
x' &= Ax + Bu + w \\
y  &= Cx + Du + e
\end{aligned}
```
we must let the dynamics noise ``w`` drive the disturbance model, and design ``C`` such that the estimated disturbance has the desired effect on the measurement. This model leaves no room to let the measurement noise ``e`` to pass through a dynamical system, and this is thus only useful (and required) to model white Gaussian measurement noise. See [How to tune a Kalman filter](@ref) for more insights.

Dynamical models of measurement disturbances are useful in a lot of situations, such as
- Periodic measurement noise, such as 50Hz noise from the electrical grid.
- Slow sensor drift, such as gyroscopic drift.
- Calibration errors.
- Sensor misalignment in rotating systems.
- Complimentary filtering for accelerometers and gyroscopes.
- Sensor degradation, such as deposition of dust or algae growth.