# Prediction-Error minimization using an optimizer

The state estimators in this package are all statistically motivated and thus compute things like the likelihood of the data as a by-product of the estimation. Maximum-likelihood or prediction-error estimation is thus very straight-forward by simply calling a gradient-based optimizer with gradients provided by differentiating through the state estimator using automatic differentiation. In this example, we will use the quad-tank process and estimate all its parameters. We will first use a standard optimization algorithm from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to minimize the cost function based on the prediction error, and then use a Gauss-Newton optimizer.

!!! note "Prediction-Error Method"
    Minimizing the one-step ahead prediction errors made by a state estimator is often referred to as the "Prediction-Error Method" (PEM) in the system identification literature.

## Setup

We define the quad-tank dynamics function such that it takes its parameters from the `p` input argument. We also define a variable `p_true` that contains the true values that we will use to simulate some estimation data. For an introduction to the quad-tank system, see [Joint state and parameter estimation](@ref).

```@example paramest
using LowLevelParticleFilters
using SeeToDee
using Distributions
using StaticArrays
using Plots, LinearAlgebra

function quadtank(h, u, p, t)
    k1, k2, g = p[1], p[2], 9.81
    A1 = A3 = A2 = A4 = p[3]
    a1 = a3 = a2 = a4 = p[4]
    γ1 = γ2 = p[5]

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0

    SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    ]
end

Ts = 1 # sample time
nu = 2 # number of control inputs
nx = 4 # number of state variables
ny = 2 # number of measured outputs

measurement(x,u,p,t) = SA[x[1], x[2]]
discrete_dynamics = SeeToDee.Rk4(quadtank, Ts) # Discretize the dynamics using a 4:th order Runge-Kutta integrator
p_true = [1.6, 1.6, 4.9, 0.03, 0.2]
nothing # hide
```

We simulate the system, this time using a more exciting input in order to be able to identify several parameters:
```@example paramest
using Random; Random.seed!(1) # hide
Tperiod = 200
t = 0:Ts:1000
u1 = vcat.(0.25 .* sign.(sin.(2pi/Tperiod .* (t ./ 40).^2)) .+ 0.25)
u2 = vcat.(0.25 .* sign.(sin.(2pi/Tperiod .* (t ./ 40).^2 .+ pi/2)) .+ 0.25)
u  = SVector{nu}.(vcat.(u1,u2))
x0 = SA[2.0,2,3,3] # Initial condition, static array for performance
x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, u, p_true)[1:end-1]
y = measurement.(x, u, 0, 0)
y = [y .+ 0.01 .* randn.() for y in y]

plot(
    plot(reduce(hcat, x)', title="State"),
    plot(reduce(hcat, u)', title="Inputs")
)
```


This time, we define a cost function for the optimizer to optimize, we'll use the sum of squared errors (`sse`). It's important to define the UKF with an initial state distribution with the same element type as the parameter vector so that automatic differentiation through the state estimator works, hence the explicit casting `T.(x0)` and `T.(R1)`. We also make sure to use StaticArrays for the covariance matrices and the initial condition for performance reasons (optional).
```@example paramest
R1 = SMatrix{nx,nx}(Diagonal([0.1, 0.1, 0.1, 0.1])) # Use of StaticArrays is generally good for performance
R2 = SMatrix{ny,ny}(Diagonal((1e-2)^2 * ones(ny)))
x0 = SA[2.0, 2, 3, 3]

function cost(p::Vector{T}) where T
    kf = UnscentedKalmanFilter(discrete_dynamics, measurement, R1, R2, MvNormal(T.(x0), T.(R1)); ny, nu, Ts)
    LowLevelParticleFilters.sse(kf, u, y, p) # Sum of squared prediction errors
end
nothing # hide
```
We generate a random initial guess for the estimation problem
```@example paramest
p_guess = p_true .+  0.1*p_true .* randn(length(p_true))
```


## Solving using Optim
We first minimize the cost using the BFGS optimization algorithm from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
```@example paramest
using Optim
using ADTypes: AutoForwardDiff
res = Optim.optimize(
    cost,
    p_guess,
    BFGS(),
    Optim.Options(
        show_trace = true,
        show_every = 5,
        iterations = 100,
        time_limit = 30,
    );
    autodiff = AutoForwardDiff(), # Indicate that we want to use forward-mode AD to derive gradients
)
```

We started out with a normalized parameter error of
```@example paramest
norm(p_true - p_guess) / norm(p_true)
```
and ended with
```@example paramest
p_opt = res.minimizer
norm(p_true - p_opt) / norm(p_true)
```


## Solving using Gauss-Newton optimization
Below, we optimize the sum of squared residuals again, but this time we do it using a Gauss-Newton style algorithm (Levenberg Marquardt). These algorithms want the entire residual vector rather than the sum of squares of the residuals, so we define an alternative "cost function" called `residuals` that calls the lower-level function [`LowLevelParticleFilters.prediction_errors!`](@ref)
```@example paramest
using LeastSquaresOptim

function residuals!(res, p::Vector{T}) where T
    kf = UnscentedKalmanFilter(discrete_dynamics, measurement, R1, R2, MvNormal(T.(x0), T.(R1)); ny, nu, Ts)
    LowLevelParticleFilters.prediction_errors!(res, kf, u, y, p)
end

res_gn = optimize!(LeastSquaresProblem(x = copy(p_guess), f! = residuals!, output_length = length(y)*ny, autodiff = :forward), LevenbergMarquardt())

p_opt_gn = res_gn.minimizer
norm(p_true - p_opt_gn) / norm(p_true)
```

When performing sum-of-squares minimization like here, we can, assuming that we converge to the global optimum, estimate the covariance of the estimated parameters. The _precision matrix_ ``Λ``, which is the inverse of the covariance matrix of the parameters, is given by a scaled Hessian of the cost function. The Gauss-Newton appoximation of the Hessian is given by ``J'J``, where ``J`` is the Jacobian of the residuals.
```@example paramest
using ForwardDiff
T = length(y)
out = zeros(T * ny)
J = ForwardDiff.jacobian(residuals!, out, res_gn.minimizer)
residuals!(out, res_gn.minimizer)
Λ = (T - length(p_guess))/dot(out,out) * Symmetric(J' * J) # Precision matrix of the estimated parameters
# Σ = inv(Λ) # Covariance matrix of the estimated parameters (only compute this if precision matrix is well conditioned)
svdvals(Λ)
```
In this case, the precision matrix is singular, indicating that there is at least one direction in parameter space that yields no increase in cost, and we can thus not determine where along a line in this direction the true parameter lies.

Gauss-Newton algorithms are often more efficient at sum-of-squares minimization than the more generic BFGS optimizer. This form of Gauss-Newton optimization of prediction errors is also available through [ControlSystemIdentification.jl](https://baggepinnen.github.io/ControlSystemIdentification.jl/dev/nonlinear/#Identification-of-nonlinear-models), which uses this package underneath the hood.


## Optimizing log-likelihood using Gauss-Newton optimization
We can use a Gauss-Newton optimizer to maximize the log-likelihood as well, the only thing we need to change is to pass `loglik = true` to the `prediction_errors!` function, adjust the residual output length accordingly (notice the `(ny+1)` below, we now have an additional residual per time step corresponding to a `logdet` term in the likelihood) as well as possibly providing an `offset` argument. The reason for the offset is that the `logdet` term may be negative and cannot be the result of squaring a real number. The addition of the offset does not affect the optimization process, but adds a constant offset to the computed log liklihood value (cost function). If the offset is needed, you will get an error message indicating that when calling `prediction_errors!`. The code looks like this:
```@example paramest
using LeastSquaresOptim

function residuals!(res, p::Vector{T}) where T
    kf = UnscentedKalmanFilter(discrete_dynamics, measurement, R1, R2, MvNormal(T.(x0), T.(R1)); ny, nu, Ts)
    LowLevelParticleFilters.prediction_errors!(res, kf, u, y, p, loglik=true, offset=12)
end

res_gn = optimize!(LeastSquaresProblem(x = copy(p_guess), f! = residuals!, output_length = length(y)*(ny+1), autodiff = :forward), LevenbergMarquardt())

p_opt_gn = res_gn.minimizer
norm(p_true - p_opt_gn) / norm(p_true)
```
