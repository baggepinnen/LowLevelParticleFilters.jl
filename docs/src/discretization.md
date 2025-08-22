# Discretization

This package operates exclusively on discrete-time dynamics, and dynamics describing, e.g., ODE systems must thus be discretized. This page describes the details around discretization for nonlinear and linear systems, as well as how to discretize continuous-time noise processes. 

## Nonlinear ODEs

Continuous-time dynamics functions on the form `(x,u,p,t) -> ẋ` can be discretized (integrated) using the function [`SeeToDee.Rk4`](https://baggepinnen.github.io/SeeToDee.jl/dev/api/#SeeToDee.Rk4), e.g.,
```julia
using SeeToDee
discrete_dynamics = SeeToDee.Rk4(continuous_dynamics, sampletime; supersample=1)
```
where the integer `supersample` determines the number of RK4 steps that is taken internally for each change of the control signal (1 is often sufficient and is the default). The returned function `discrete_dynamics` is on the form `(x,u,p,t) -> x⁺`.

!!! note
    When solving state-estimation problems, accurate integration is often less important than during simulation. The motivations for this are several
    - The dynamics model is often inaccurate, and solving an inaccurate model to high accuracy can be a waste of effort.
    - The performance is often dictated by the disturbances acting on the system.
    - State-estimation enjoys feedback from measurements that corrects for slight errors due to integration.

## Linear systems
A linear system on the form 
```math
\begin{aligned}
\dot{x}(t) &= Ax(t) + Bu(t)\\
y(t) &= Cx(t) + Du(t)
\end{aligned}
```
can be discretized using [`ControlSystems.c2d`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/synthesis/#ControlSystemsBase.c2d), which defaults to a zero-order hold discretization. See the example below for more info.

## Covariance matrices
Covariance matrices for continuous-time noise processes can also be discretized using [`ControlSystems.c2d`](https://juliacontrol.github.io/ControlSystems.jl/stable/lib/synthesis/#ControlSystemsBase.c2d-Tuple{AbstractStateSpace{%3C:Continuous},%20AbstractMatrix,%20Real})
```julia
using ControlSystemIdentification
R1d      = c2d(sys::StateSpace{Continuous}, R1c, Ts)
R1d, R2d = c2d(sys::StateSpace{Continuous}, R1c, R2c, Ts)
R1d      = c2d(sys::StateSpace{Discrete},   R1c)
R1d, R2d = c2d(sys::StateSpace{Discrete},   R1c, R2c)
```

This samples a continuous-time covariance matrix to fit the provided system `sys`.

The method used comes from theorem 5 in the reference below.

> Ref: "Discrete-time Solutions to the Continuous-time
> Differential Lyapunov Equation With Applications to Kalman Filtering", 
> Patrik Axelsson and Fredrik Gustafsson

**On singular covariance matrices:** The traditional double integrator with covariance matrix `Q = diagm([0,σ²])` warrants special consideration since it is rank-deficient, i.e., it indicates that there is a single source of randomness only, despite the presence of two state variables. If we assume that the noise is piecewise constant, we can use the input matrix ("Cholesky factor") of `Q`, e.g., the noise of variance `σ²` enters like `N = [0, 1]` which is sampled using ZoH and becomes `Nd = [Ts^2 / 2; Ts]` which results in the covariance matrix `σ² * Nd * Nd'` (see example below). If we instead assume that the noise is a continuous-time white noise process, the discretized covariance matrix is full rank and can be computed by `c2d(sys::StateSpace{Continuous}, R1c, Ts)` or directly by the function [`double_integrator_covariance_smooth`](@ref).

## Example
The following example will discretize a linear double integrator system. Double integrators arise when the position of an object is controlled by a force, i.e., when Newtons second law ``f = ma`` governs the dynamics. The system can be written on the form
```math
\begin{aligned}
\dot x(t) &= Ax(t) + Bu(t) + Nw(t)\\
y(t) &= Cx(t) + e(t)
\end{aligned}
```
where ``N = B`` are both equal to `[0, 1]`, indicating that the noise ``w(t)`` enters like a force (this could be for instance due to air resistance or friction).

We start by defining the system that takes ``u`` as an input and discretize that with a sample time of ``T_s = 0.1``.

```@example samplecov
using ControlSystemsBase
A = [0 1; 0 0]
B = [0; 1;;]
C = [1 0]
D = 0
Ts = 0.1 # Sample time

sys = ss(A,B,C,D)
sysd = c2d(sys, Ts) # Discretize the dynamics
```

We then form another system, this time with ``w(t)`` as the input, and thus ``N`` as the input matrix instead of ``B``. We assume that the noise has a standard deviation of ``\sigma_1 = 0.5``
```@example samplecov
σ1 = 0.5
N  = σ1*[0; 1;;]
sys_w  = ss(A,N,C,D)
sys_wd = c2d(sys_w, Ts) # Discretize the noise system
Nd  = sys_wd.B # The discretized noise input matrix
R1d = Nd*Nd' # The final discrete-time covariance matrix
```

We can verify that the matrix we computed corresponds to the theoretical covariance matrix for a discrete-time double integrator where the noise is piecewise constant:
```@example samplecov
R1d ≈ σ1^2*[Ts^2 / 2; Ts]*[Ts^2 / 2; Ts]'
```

If the noise is not piecewise constant the discretized covariance matrix will be full rank, but a good rank-1 approximation in this case is `R1d ./ Ts`.

For a nonlinear system, we could adopt a similar strategy by first linearizing the system around a suitable operating point. Alternatively, we could make use of the fact that some of the state estimators in this package allows the covariance matrices to be functions of the state, and thus compute a new discretized covariance matrix using a linearization around the current state.

## Sample-interval insensitive tuning
When the dynamics covariance of a state estimator is tuned, it may be desirable to have the covariance dynamics be approximately invariant to the choice of sample interval ``T_s``. How to achieve this depends on what formulation of the dynamics is used, in particular, whether the noise inputs are included in the discretization procedure or not. Note, when a higher sample-rate implies the use of more frequent measurements, the covariance dynamics will not be rendered invariant to the sample interval using the methods below, only the covariance dynamics during prediction only will have this property.

### Noise inputs are not discretized
This case arises when using the standard [`KalmanFilter`](@ref) with dynamics equation
```math
x^+ = Ax + Bu + w
```
or a nonlinear version with `dynamics(x, u, p, t)`. To achieve sample-rate invariant tuning, construct the covariance matrix as `R1 = [...] .* Ts`, i.e., tune a matrix that is scaled by the the sample interval. If you later change `Ts`, you'll get approximately the same performance of the estimator for prediction intervals during which there are no measurements available.


### Noise inputs are discretized
This case arises when using an augmented [`UnscentedKalmanFilter`](@ref) with dynamics `dynamics(x, u, p, t, w)` which is discretized using an integrator, such as
```julia
disc_dynamics = SeeToDee.Rk4(dynamics, Ts)
```
In this case, the integrator integrates also the noise process, and we instead achieve sample-rate invariant tuning by constructing the covariance matrix as `R1 = [...] ./ Ts`, i.e., tune a matrix that is scaled by the **inverse** of the sample interval. 

This case can also arise when using a linear system with noise input, i.e., the dynamics equation
```math
\dot x = Ax + Bu + Nw
```
where `N` is the input matrix for the noise process. When this system is discretized with the input matrix `[B N]`
and the `R1` matrix is derived as $R_1^d = N_d R_1^c N_d^T$, we need to further scale the covariance matrix by `1/Ts`, i.e., use $R_1^d = \frac{1}{T_s} N_d R_1^c N_d^T$.

### When using `double_integrator_covariance_smooth`
The function [`double_integrator_covariance_smooth`](@ref) already has the desired scaling with `Ts` built in, and this is thus to be used with additive noise that is not discretized.

[`double_integrator_covariance`](@ref) is for piecewise constant noise and this does generally not lead to sample-rate invariant tuning, however `double_integrator_covariance(Ts) ./ Ts` does.

## Non-uniform sample rates
Special care is needed if the sample rate is not constant, i.e., the time interval between measurements varies. 

### Dropped samples
A common case is that the sample rate is constant, but some measurements are lost. This case is very easy to handle; the filter loop iterates between two steps
1. Prediction using `predict!(filter, x, u, p, t)`
2. Correction using
    - `correct!(f, u, y, p, t)` if using the standard measurement model of the filter
    - `correct!(f, mm, u, y, p, t, mm)` to use a custom measurement model `mm`

If a measurement `y` is lacking, one simply skips the corresponding call to `correct!` where `y` is missing. Repeated calls to `predict!` corresponds to simulating the system without any feedback from measurements, like if an ODE was solved. Internally, the filter will keep track of the covariance of the estimate, which is likely to grow if no measurements are used to inform the filter about the state of the system.

### Sensors with different sample rates
For Kalman-type filters, it is possible to construct custom measurement models, and pass an instance of a measurement model as the second argument to [`correct!`](@ref). This allows for sensor fusion with sensors operating at different rates, or when parts of the measurement model are linear, and other parts are nonlinear. See examples in [Measurement models](@ref measurement_models) for how to construct explicit measurement models.

A video demonstrating the use of multiple measurement models running at different rates is available on YouTube:

```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/BLsJrW5XXcg?si=bkob76-uJj27-S80" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```



### Stochastic sample rate
In some situations, such as in event-based systems, the sample rate is truly stochastic. There is no single correct way of handling this, and we instead outline some alternative approaches.

- If the filtering is performed offline on a batch of data, time-varying dynamics can be used, for instance by supplying matrices to a [`KalmanFilter`](@ref) on the form `A[:, :, t], R1[:, :, t]`. Each `A` and `R1` is then computed as the discretization with the sample time given as the time between measurement `t` and measurement `t+1`.
- A conceptually simple approach is to choose a very small sample interval ``T_s`` which is smaller than the smallest occurring sample interval in the data, and approximate each sample interval by rounding it to the nearest integer multiple of ``T_s``. This transforms the problem to an instance of the "dropped samples" problem described above.
- Make use of an adaptive integrator instead of the fixed-step `rk4` supplied in this package, and manually keep track of the step length that needs to be taken as well as the adjustment to the dynamics covariance.

## Example: EKF with stochastic sample rate

The following example demonstrates how to perform EKF filtering when data arrives at stochastic time intervals. We simulate a Dubin's car model (a simple kinematic vehicle model) and filter the data using an Extended Kalman Filter that adapts to varying sample rates. The control inputs are assumed to be updated at a fixed time interval `Ts`, while measurements arrive stochastically with an interval chosen uniformly at random between 0 and 2s.

```@example stochastic_ekf
using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using SeeToDee
using Random, Statistics, LinearAlgebra
using StaticArrays
using Plots
Random.seed!(42)

# Dubin's car continuous-time dynamics
# State: [x, y, θ, v] - position, heading angle, velocity
# Input: [a, ω] - acceleration, angular velocity
function dubins_car(x, u, p, t)
    θ = x[3]
    v = x[4]
    a = u[1]
    ω = u[2]
    SA[
        v * cos(θ),    # ẋ
        v * sin(θ),    # ẏ
        ω,             # θ̇
        a              # v̇
    ]
end

# Measurement function - observe position only
measurement(x, u, p, t) = SA[x[1], x[2]]

# System parameters
nx = 4  # state dimension
nu = 2  # input dimension
ny = 2  # output dimension

# Base sample time for the filter, the control input is updated at this interval
Ts = 0.1

# Create discretized dynamics
discrete_dynamics = SeeToDee.Rk4(dubins_car, Ts)

# Wrapper that uses the Ts from the parameter
function adaptive_step_dynamics(x, u, p, t)
    Ts = p # The sample step duration is passed as the parameter
    return discrete_dynamics(x, u, nothing, t; Ts)
end

# Control input function of time
u_func(t) = SA[0.5 * cos(0.5 * t) - 0.1, 0.3 * cos(0.3 * t)]

x0 = SA[0.0, 0.0, 0.0, 1.0]  # Initial state

# Setup EKF
# Base process noise covariance (will be scaled by Ts during filtering)
R1_base = Diagonal([0.001, 0.001, 0.001, 0.001])  # Process noise per unit time
R2 = Diagonal([0.3^2, 0.3^2])                     # Measurement noise
d0 = SimpleMvNormal(x0, 0.1 * I(4))               # Initial state distribution

# For the EKF constructor, use the base rate scaled according to advice above
R1 = R1_base * Ts

# Create EKF with adaptive dynamics
# The dynamics function will receive the time step through the parameter p
ekf = ExtendedKalmanFilter(
    adaptive_step_dynamics,
    measurement,
    R1,
    R2,
    d0;
    nu,
    ny,
    p = Ts  # Default time step
)
```

### Simulation
The function `simulate_stochastic_ekf!` implements both simulation of the plant by means of propagating the true state and filtering the measurements using the EKF. At each time step, a decision is made whether to update the control input or to take a measurement, depending on which event occurs first.
```@example stochastic_ekf
function simulate_stochastic_ekf!(
    ekf, adaptive_step_dynamics, u_func, x0, Tf
)
    (; measurement, Ts) = ekf
    next_control_t  = Ts
    next_sample_t   = 2rand()
    t               = 0.0
    x_true          = x0
    u               = u_func(t)
    # Arrays for storing simulation data
    U               = [u]
    X               = [x_true]
    Xf              = [x0]
    Y               = [measurement(x_true, u, nothing, t)]
    T               = [t] # Array with all time points
    Ty              = [t] # Array with only time points with new measurements
    while t < Tf
        # We choose how long step to take depending on whether the next event is a measurement arrival or control update
        if next_sample_t < next_control_t
            # Step forward to next_sample_t and sample a measurement
            dt = next_sample_t - t                              # Step length to take, pass this as the parameter
            R1t = R1_base * dt                                  # Scale process noise covariance by step length. Note the advice above regarding how to perform this scaling depending on discretization context.
            x_true = adaptive_step_dynamics(x_true, u, dt, t)
            predict!(ekf, u, dt, t, R1=R1t)                     # Step the filter forward dt time units as well
            t = next_sample_t                                   # Update the current time
            y = measurement(x_true, u, nothing, t) + 0.3*randn(ekf.ny) # Simulate a measurement
            correct!(ekf, u, y, dt, t)                          # Apply filter measurement update
            push!(Y, y)                                         # Store measurement data for plotting
            push!(Ty, t)
            next_sample_t += 2rand()
        else
            # Step forward to next_control_t, in this branch there is no new measurement
            dt      = next_control_t - t
            R1t     = R1_base * dt # Scale process noise covariance by step length. Note the advice above regarding how to perform this scaling depending on discretization context.
            x_true  = adaptive_step_dynamics(x_true, u, dt, t)
            t       = next_control_t
            predict!(ekf, u, dt, t, R1=R1t)
            u       = u_func(t)
            next_control_t += Ts
        end
        push!(X, x_true)
        push!(Xf, ekf.x)
        push!(T, t)
        push!(U, u)
    end
    return (T, X, Xf, U, Y, Ty)
end

# Run the simulation
Tf = 20 # Final time, duration of the simulation
T, X, Xf, U, Y, Ty = simulate_stochastic_ekf!(
    ekf, adaptive_step_dynamics, u_func, x0, Tf
)

# Plot true and filtered estimate
using Plots

# Xf at Ty times
Xy = reduce(hcat, [x for (t, x) in zip(T, X) if t in Ty])'
Xfy = reduce(hcat, [x for (t, x) in zip(T, Xf) if t in Ty])'

# Compute filtering errors 
Ef = (Xy .- Xfy)[:, 1:2]
Ey = (Xy[:, 1:2] .- reduce(hcat, Y)')

plot(T, reduce(hcat, X)', label="\$x\$", layout=4)
scatter!(Ty, Xfy, label="\$x(t|t)\$", markersize=3, markerstrokewidth=0, sp=1:4)
scatter!(Ty, reduce(hcat, Y)', label="\$y\$", markersize=3, markerstrokewidth=0, sp=1:2)
```

In this example, we performed filtering using an [`ExtendedKalmanFilter`](@ref) that takes nonlinear dynamics discretized with an RK4 integrator. When the dynamics are linear and we employ a standard [`KalmanFilter`](@ref), varying-length discretization is similarly handled by providing custom ``A`` and ``B`` matrices to the `predict!` function. ZoH discretization of a linear system is performed using the matrix exponential ``e^{A T_s}`` (see [implementation of `c2d`](https://github.com/JuliaControl/ControlSystems.jl/blob/f04916f6afeadacbd48b2824fb0f2d833deb4f00/lib/ControlSystemsBase/src/discrete.jl#L49C7-L52C33) for how to handle also the ``B`` matrix at the same time).


## Example: Adaptive interval velocity estimation from encoder pulses

This example demonstrates how one can use a Kalman filter when measurements arrive at irregular intervals. The application is estimation of velocity (and possibly higher-order derivatives) from encoder pulses. An encoder has a fixed number of teeth, and a pulse is generated each time a tooth crosses a light sensor. For simplicity, we do not model the specifics of the encoder here, and instead simply generate a position measurement whenever a simulated position signal mas moved a certain amount (corresponding to the distance between teeth).

The Kalman filter is setup using a model corresponding to a series of ``n = `` `filter_order` integrators with nominal values for covariance and dynamics and in the loop as measurements arrive, we discretize the continuous-time dynamical model with the interval between the current and last obtained measurement. This discretization involves not only the dynamics, we also compute the discrete-time covariance matrix ``R_1`` corresponding to passing a continuous-time white-noise process through the series of ``n`` integrators.

Below, we show the results for filter order 4 (sometimes called a constant-jerk model). A lower filter order makes the system more responsive to changes in the measurement, while a higher order provides smoother estimates at the cost of increased lag. When viewed in the frequency domain, the filter order controls the slope of the rolloff for high frequencies, while the covariance value, ``\sigma^2`` below, controls the cut-off frequency.

For fun, we use the square-root version of the Kalman filter here, [`SqKalmanFilter`](@ref) so that we have a demo that uses this as well.

```@example velocity_observer
using LowLevelParticleFilters, LinearAlgebra, StaticArrays
using LowLevelParticleFilters: SimpleMvNormal
using ControlSystemsBase, Plots
traj(t)  = t < 10 ? t   :  10cos(t-10)
trajv(t) = t < 10 ? 1.0 : -10sin(t-10)
traja(t) = t < 10 ? 0.0 : -10cos(t-10)

to_step(x, n) = floor(x*n)/n
function find_crossings(traj::F, resolution) where F
    # resolution is given in number of levels per unit position
    t_candidates = 0:1e-6:20
    crossings = Float64[]
    t0 = to_step(traj(t_candidates[1]), resolution)
    for t in 1:length(t_candidates) - 1
        t1 = to_step(traj(t_candidates[t + 1]), resolution)
        if t0 != t1
            # Interpolate to get accurate t crossing
            t_cross = t_candidates[t]
            push!(crossings, t_cross)
            t0 = t1
        end
    end
    return crossings
end

sample_times = find_crossings(traj, 1)
pos_measurements = traj.(sample_times)
vel_measurements = diff(pos_measurements) ./ diff(sample_times)
##
plot(traj, 0, 20, layout=(2,1))
scatter!(sample_times, traj.(sample_times), sp=1)
plot!(trajv, 0, 20, sp=2)
scatter!(sample_times[2:end], vel_measurements, sp=2)

##

filter_order = 4 # This controls the slope of the rolloff for high frequencies
P = ss((1/tf('s')))^filter_order
(; A, C, D) = P
B  = @SMatrix zeros(filter_order, 0)
Ts = 0.1
σ2 = 1e5 # This controls the amount of filtering. Higher value gives a smoother but laggier response
R1 = LowLevelParticleFilters.n_integrator_covariance_smooth(P.nx, Ts, σ2)
R2 = [1.0;;]
d0 = SimpleMvNormal(1e9R1)
kf = SqKalmanFilter(to_static(A), B, to_static(C), to_static(D), R1, R2, d0)
chol(x) = cholesky(x).U
function kf_velocity_estimation!(
    kf, sample_times
)
    (; Ts) = kf
    Xf = [copy(kf.x)]
    for i = 2:length(sample_times)
        t0 = sample_times[i-1]
        t1 = sample_times[i]
        dt = t1 - t0
        Ai = exp(A*dt)
        R1 = LowLevelParticleFilters.n_integrator_covariance_smooth(kf.nx, dt, σ2) |> chol # This is the solution to a fixed-horizon Lyapunov equation
        predict!(kf, nothing, nothing, t0; At=Ai, R1)
        y = pos_measurements[i]
        correct!(kf, nothing, y, nothing, t1)
        push!(Xf, copy(kf.x))
    end
    Xf
end
Xf = kf_velocity_estimation!(kf, sample_times)
XF = reduce(hcat, Xf)'
##
scatter(sample_times, XF, lab=permutedims(["\$\\frac{d^$(i)\\hat{x}}{dt^$(i)}\$" for i in 0:filter_order-1]), layout=(filter_order,1), size=(1000,1000), markerstrokewidth=1, m=:cross)
plot!(traj, 0, 20, lab="True Position")
plot!(trajv, 0, 20, sp=2, lab="True Velocity")
plot!(traja, 0, 20, sp=3, lab="True Acceleration")
```
Notice how there are no measurements for a while when the velocity changes sign (the velocity is low), while there are more frequent measurements when the velocity is high.