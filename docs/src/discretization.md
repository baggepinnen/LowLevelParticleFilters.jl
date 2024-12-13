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
R1d = c2d(sys::StateSpace{<:Discrete}, R1::Matrix)
R1d, R2d = c2d(sys::StateSpace{<:Discrete}, R1::Matrix, R2::Matrix)
```

This samples a continuous-time covariance matrix to fit the provided discrete-time system `sys`.

The method used comes from theorem 5 in the reference below.

> Ref: "Discrete-time Solutions to the Continuous-time
> Differential Lyapunov Equation With Applications to Kalman Filtering", 
> Patrik Axelsson and Fredrik Gustafsson

**On singular covariance matrices:** The traditional double integrator with covariance matrix `Q = diagm([0,σ²])` can not be sampled with this method. Instead, the input matrix ("Cholesky factor") of `Q` must be manually kept track of, e.g., the noise of variance `σ²` enters like `N = [0, 1]` which is sampled using ZoH and becomes `Nd = [Ts^2 / 2; Ts]` which results in the covariance matrix `σ² * Nd * Nd'` (see example below).

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

We can verify that the matrix we computed corresponds to the theoretical covariance matrix for a discrete-time double integrator:
```@example samplecov
R1d ≈ σ1^2*[Ts^2 / 2; Ts]*[Ts^2 / 2; Ts]'
```

For a nonlinear system, we could adopt a similar strategy by first linearizing the system around a suitable operating point. Alternatively, we could make use of the fact that some of the state estimators in this package allows the covariance matrices to be functions of the state, and thus compute a new discretized covariance matrix using a linearization around the current state.

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

- If the filtering is performed offline on a batch of data, time-varying dynamics can be used, for instance by supplying matrices to a [`KalmanFilter`](@ref) on the form `A[:, :, t]`. Each `A` is then computed as the discretization with the sample time given as the time between measurement `t` and measurement `t+1`.
- A conceptually simple approach is to choose a very small sample interval ``T_s`` which is smaller than the smallest occuring sample interval in the data, and approximate each sample interval by rounding it to the nearest integer multiple of ``T_s``. This transforms the problem to an instance of the "dropped samples" problem described above.
- Make use of an adaptive integrator instead of the fixed-step `rk4` supplied in this package, and manually keep track of the step length that needs to be taken.