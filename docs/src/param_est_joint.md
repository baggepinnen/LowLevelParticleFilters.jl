# Joint state and parameter estimation

In this example, we'll show how to perform parameter estimation by treating a parameter as a state variable. This method can not only estimate constant parameters, but also time-varying parameters. The system we will consider is a quadruple tank, where two upper tanks feed into two lower tanks. The outlet for tank 1 can vary in size, simulating, e.g., that something partially blocks the outlet. We start by defining the dynamics on a form that changes the outlet area ``a_1`` at time ``t=500``:
```@example paramest
using LowLevelParticleFilters
using SeeToDee
using Distributions
using StaticArrays
using Plots, LinearAlgebra

function quadtank(h,u,p,t)
    k1, k2, g = 1.6, 1.6, 9.81
    A1 = A3 = A2 = A4 = 4.9
    a1, a3, a2, a4 = 0.03, 0.03, 0.03, 0.03
    γ1, γ2 = 0.2, 0.2

    if t > 500
        a1 *= 2 # Change the parameter at t = 500
    end

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0

    SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    ]
end

nu = 2 # number of control inputs
nx = 4 # number of state variables
ny = 2 # number of measured outputs
Ts = 1 # sample time
nothing # hide
```
We then define a measurement function, we measure the levels of tanks 1 and 2, and discretize the continuous-time dynamics using a Runge-Kutta 4 integrator [`SeeToDee.Rk4`](https://baggepinnen.github.io/SeeToDee.jl/dev/api/#SeeToDee.Rk4):
```@example paramest
measurement(x,u,p,t) = SA[x[1], x[2]]
discrete_dynamics = SeeToDee.Rk4(quadtank, Ts)
nothing # hide
```

We simulate the system using the `rollout` function and add some noise to the measurements. The inputs in this case are just square waves.
```@example paramest
Tperiod = 200
t = 0:Ts:1000
u = vcat.(0.25 .* sign.(sin.(2pi/Tperiod .* t)) .+ 0.25)
u = SVector{nu}.(vcat.(u,u))
x0 = Float64[2,2,3,3]
x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, u)[1:end-1]
y = measurement.(x, u, 0, 0)
y = [y .+ 0.01.*randn.() for y in y]

plot(
    plot(reduce(hcat, x)', title="State"),
    plot(reduce(hcat, u)', title="Inputs")
)
```

To perform the joint state and parameter estimation, we define a version of the dynamics that contains an extra state, corresponding to the unknown or time varying parameter, in this case ``a_1``. We do not have any apriori information about how this parameter changes, so we say that its derivative is 0 and it's thus only driven by noise:
```@example paramest
function quadtank_paramest(h, u, p, t)
    k1, k2, g = 1.6, 1.6, 9.81
    A1 = A3 = A2 = A4 = 4.9
    a3, a2, a4 = 0.03, 0.03, 0.03
    γ1, γ2 = 0.2, 0.2

    a1 = h[5] # the a1 parameter is a state

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0

    SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
        0 # the state is only driven by noise
    ]
end

discrete_dynamics_params = SeeToDee.Rk4(quadtank_paramest, Ts)
nothing # hide
```

We then define a nonlinear state estimator, we will use the [`UnscentedKalmanFilter`](@ref), and solve the filtering problem. We start by an initial state estimate ``x_0`` that is slightly off for the parameter ``a_1``
```@example paramest
nx = 5
names = SignalNames(x = ["h1", "h2", "h3", "h4", "a1"], y = ["h$i" for i in 1:2], u = ["p1", "p2"], name="UKF") # For nicer plot labels

R1 = SMatrix{nx,nx}(Diagonal([0.1, 0.1, 0.1, 0.1, 0.0001])) # Use of StaticArrays is generally good for performance
R2 = SMatrix{ny,ny}(Diagonal((1e-2)^2 * ones(ny)))
x0 = SA[2, 2, 3, 3, 0.02] # The SA prefix makes the array static, which is good for performance

kf = UnscentedKalmanFilter(discrete_dynamics_params, measurement, R1, R2, MvNormal(x0, R1); ny, nu, Ts, names)

sol = forward_trajectory(kf, u, y)
plot(sol, plotx=false, plotxt=true, plotu=false, ploty=true, legend=:bottomright)
plot!([0,500,500,1000], [0.03, 0.03, 0.06, 0.06], l=(:dash, :black), sp=5, lab="True param")
```
as we can see, the correct value of the parameter is quickly found (``a_1``), and it also adapts at ``t=500`` when the parameter value changes. The speed with which the parameter adapts to changes is determined by the covariance matrix ``R_1``, a higher value results in faster adaptation, but also higher sensitivity to noise.

If adaptive parameter estimation is coupled with a model-based controller, we get an adaptive controller! Note: the state that corresponds to the estimated parameter is typically not controllable, a fact that may require some special care for some control methods.

We may ask ourselves, what's the difference between a parameter and a state variable if we can add parameters as state variables? Typically, parameters do not vary with time, and if they do, they vary significantly slower than the state variables. State variables also have dynamics associate with them, whereas we often have no idea about how the parameters vary other than that they vary slowly.

Abrupt changes to the dynamics like in the example above can happen in practice, for instance, due to equipment failure or change of operating mode. This can be treated as a scenario with time-varying parameters that are continuously estimated.
