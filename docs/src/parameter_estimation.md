# Parameter Estimation
State estimation is an integral part of many parameter-estimation methods. Below, we will iluustrate several different methods of performing parameter estimation. We can roughly divide the methods into two camps
1. Methods that optimize prediction error or likelihood by tweaking model parameters.
2. Methods that add the parameters to be estimated as states in the model and estimate them using standard state estimation. 

From the first camp, we provide som basic functionality for maximum likelihood estimation and MAP estimation, described below. An example 2), joint state and parameter estimation, is provided in [Joint state and parameter estimation](@ref).


## Maximum-likelihood estimation
Filters calculate the likelihood while performing filtering, we may for example plot likelihood as function of the variance of the dynamics noise

```@setup ml_map
using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Distributions, Plots
nx = 2   # Dimension of state
nu = 2   # Dimension of input
ny = 2   # Dimension of measurements
N = 800 # Number of particles

const dg = MvNormal(ny,1.0)          # Measurement noise Distribution
const df = MvNormal(nx,1.0)          # Dynamics noise Distribution
const d0 = MvNormal(randn(nx),2.0)   # Initial state Distribution

Tr = randn(nx,nx)
const A = SA[1 0.1; 0 1]
const B = @SMatrix [0.0 0.1; 1 0.1]
const C = @SMatrix [1.0 0; 0 1]

dynamics(x,u,p,t) = A*x .+ B*u 
measurement(x,u,p,t) = C*x
vecvec_to_mat(x) = copy(reduce(hcat, x)') # Helper function
pf = ParticleFilter(N, dynamics, measurement, df, dg, d0)
xs,u,y = simulate(pf,300,df)
```

```@example ml_map
p = nothing
svec = exp10.(LinRange(-1.5,1.5,60))
llspf = map(svec) do s
    df = MvNormal(nx,s)
    pfs = ParticleFilter(2000, dynamics, measurement, df, dg, d0)
    loglik(pfs, u, y, p)
end
plot( svec, llspf,
    xscale = :log10,
    title = "Log-likelihood",
    xlabel = "Dynamics noise standard deviation",
    lab = "PF",
)
vline!([svec[findmax(llspf)[2]]], l=(:dash,:blue), primary=false)
```

We can do the same with a Kalman filter

```@example ml_map
eye(n) = Matrix{Float64}(I,n,n)
llskf = map(svec) do s
    kfs = KalmanFilter(A, B, C, 0, s^2*eye(nx), eye(ny), d0)
    loglik(kfs, u, y, p)
end
plot!(svec, llskf, yscale=:identity, xscale=:log10, lab="Kalman", c=:red)
vline!([svec[findmax(llskf)[2]]], l=(:dash,:red), primary=false)
```

as we can see, the result is quite noisy due to the stochastic nature of particle filtering.

## MAP estiamtion
To solve a MAP estimation problem, we need to define a function that takes a parameter vector and returns a filter

```@example ml_map
filter_from_parameters(θ, pf = nothing) = KalmanFilter(A, B, C, 0, exp(θ[1])^2*eye(nx), exp(θ[2])^2*eye(ny), d0) # Works with particle filters as well
```

The call to `exp` on the parameters is so that we can define log-normal priors

```@example ml_map
priors = [Normal(0,2),Normal(0,2)]
```

Now we call the function `log_likelihood_fun` that returns a function to be minimized

```@example ml_map
ll = log_likelihood_fun(filter_from_parameters, priors, u, y, p)
```

Since this is a low-dimensional problem, we can plot the LL on a 2d-grid

```@example ml_map
function meshgrid(a,b)
    grid_a = [i for i in a, j in b]
    grid_b = [j for i in a, j in b]
    grid_a, grid_b
end
Nv       = 20
v        = LinRange(-0.7,1,Nv)
llxy     = (x,y) -> ll([x;y])
VGx, VGy = meshgrid(v,v)
VGz      = llxy.(VGx, VGy)
heatmap(
    VGz,
    xticks = (1:Nv, round.(v, digits = 2)),
    yticks = (1:Nv, round.(v, digits = 2)),
    xlabel = "sigma v",
    ylabel = "sigma w",
) # Yes, labels are reversed
```
For higher-dimensional problems, we may estimate the parameters using an optimizer, e.g., Optim.jl.


## Bayesian inference using PMMH
We proceed like we did for MAP above, but when calling the function `metropolis`, we will get the entire posterior distribution of the parameter vector, for the small cost of a massive increase in the amount of computations.

```@example ml_map
N = 1000
filter_from_parameters(θ, pf = nothing) = KalmanFilter(A, B, C, 0, exp(θ[1])^2*I(nx), exp(θ[2])^2*I(ny), d0) # Works with particle filters as well
```

The call to `exp` on the parameters is so that we can define log-normal priors

```@example ml_map
priors = [Normal(0,2),Normal(0,2)]
ll     = log_likelihood_fun(filter_from_parameters, priors, u, y, p)
θ₀     = log.([1.0, 1.0]) # Starting point
```

We also need to define a function that suggests a new point from the "proposal distribution". This can be pretty much anything, but it has to be symmetric since I was lazy and simplified an equation.

```@example ml_map
draw   = θ -> θ .+ 0.05 .* rand.()
burnin = 200
@info "Starting Metropolis algorithm"
@time theta, lls = metropolis(ll, 2200, θ₀, draw) # Run PMMH for 2200  iterations
thetam = reduce(hcat, theta)'[burnin+1:end,:] # Build a matrix of the output (was vecofvec)
histogram(exp.(thetam), layout=(3,1)); plot!(lls[burnin+1:end], subplot=3) # Visualize
```


If you are lucky, you can run the above threaded as well. I tried my best to make particle filters thread safe with their own rngs etc., but your milage may vary. For threading to help, the dynamics must be non-allocating, e.g., by using StaticArrays etc.

```@example ml_map
@time thetalls = LowLevelParticleFilters.metropolis_threaded(burnin, ll, 2200, θ₀, draw, nthreads=2)
histogram(exp.(thetalls[:,1:2]), layout=3)
plot!(thetalls[:,3], subplot=3)
```

## Joint state and parameter estimation
In this example, we'll show how to perform parameter estimation by treating a parameter as a state. This method can not only estimate constant parameters, but also time-varying parameters. The system we will consider is a quadruple tank, where two upper tanks feed into two lower tanks. The outlet for tank 1 can vary in size, simulating, e.g., that something partially blocks the outlet. We start by defining the dynamics on a form that changes the outlet area ``a_1`` at time ``t=500``:
```@example paramest
using LowLevelParticleFilters
using Distributions
using StaticArrays
using Plots, LinearAlgebra

function quadtank(h,u,p,t)
    kc = 0.5
    k1, k2, g = 1.6, 1.6, 9.81
    A1 = A3 = A2 = A4 = 4.9
    a1, a3, a2, a4 = 0.03, 0.03, 0.03, 0.03
    γ1, γ2 = 0.2, 0.2

    if t > 500
        a1 *= 2 # Change the parameter at t = 500
    end

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0
    
    xd = SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    ]
end

nu = 2 # number of control inputs
nx = 4 # number of states
ny = 2 # number of measured outputs
Ts = 1 # sample time
```
We then define a measurement function, we measure the levels of tanks 1 and two, and discretize the continuous-time dynamics using a Runge-Kutta 4 integrator:
```@example paramest
measurement(x,u,p,t) = SA[x[1], x[2]]
discrete_dynamics = LowLevelParticleFilters.rk4(quadtank, Ts, supersample=2)
```

We simulate the system using the `rollout` function and add some noise to the measurements. The inputs in this case are just square waves.
```@example paramest
Tperiod = 200
t = 0:Ts:1000
u = vcat.(0.25 .* sign.(sin.(2pi/Tperiod .* t)) .+ 0.25)
u = vcat.(u,u)
x0 = Float64[2,2,3,3]
x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, u)[1:end-1]
y = measurement.(x, u, 0, 0)
y = [y .+ 0.01randn(ny) for y in y]

plot(
    plot(reduce(hcat, x)', title="States"),
    plot(reduce(hcat, u)', title="Inputs")
)
```

To perform the joint state and parameter estimation, we define a version of the dynamics that contains an extra state, corresponding to the unknown or time varying parameter, in this case ``a1``. We do not have any apriori information about how this parameter changes, so we say that its derivative is 0 and it's thus only driven by noise:
```@example paramest
function quadtank_paramest(h, u, p, t)
    kc = 0.5
    k1, k2, g = 1.6, 1.6, 9.81
    A1 = A3 = A2 = A4 = 4.9
    a3, a2, a4 = 0.03, 0.03, 0.03
    γ1, γ2 = 0.2, 0.2

    a1 = h[5] # the a1 parameter is a state

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0
    
    xd = SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
        0 # the state is only driven by noise
    ]
end

discrete_dynamics_params = LowLevelParticleFilters.rk4(quadtank_paramest, Ts, supersample=2)
```

We then define a nonlinear state estimator, we will use the [`UnscentedKalmanFilter`](@ref), and solve the filtering problem. We start by an initial state estimate ``x_0`` that is slightly off for the parameter ``a_1``
```@example paramest
nx = 5
R1 = Diagonal([0.1, 0.1, 0.1, 0.1, 0.0001])
R2 = Diagonal((1e-2)^2 * ones(ny))
x0 = [2, 2, 3, 3, 0.02]

kf = UnscentedKalmanFilter(discrete_dynamics_params, measurement, R1, R2, MvNormal(x0, R1); ny, nu)

sol = forward_trajectory(kf, u, y)
plot(sol, plotx=false, plotxt=true, plotu=false, ploty=true, legend=:bottomright)
plot!([0,500,500,1000], [0.03, 0.03, 0.06, 0.06], l=(:dash, :black), sp=5, lab="True param")
```
as we can see, the correct value of the parameter is quickly found (``x_5``), and it also adapts at ``t=500`` when the parameter value changes. The speed with which the parameter adapts to changes is determined by the covariance matrix ``R_1``, a higher value results in faster adaptation, but also higher sensitivity to noise. 

If adaptive parameter estimation is coupled with a model-based controller, we get an adaptive controller! Note: the state that corresponds to the estimated parameter is typically not controllable, a fact that may require some special care for some control methods.

## Using an optimizer
Maximum-likelihood or prediction-error estimation is stright-forward by simply differentiating through the state estimator using automatic differentiation. In this example, we will continue the example from above, but now estimate all the parameters of the quad-tank process. This time, they will not vary with time.

This time, we define the dynamics function such that it takes its parameters from the `p` input argument. We also define a variable `p_true` that contains the true values that we will use to simulate some estimation data
```@example paramest
function quadtank(h, u, p, t)
    kc = p[1]
    k1, k2, g = p[2], p[3], 9.81
    A1 = A3 = A2 = A4 = p[4]
    a1 = a3 = a2 = a4 = p[5]
    γ1 = γ2 = p[6]

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0
    
    xd = SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    ]
end

discrete_dynamics = LowLevelParticleFilters.rk4(quadtank, Ts, supersample=2)
p_true = [0.5, 1.6, 1.6, 4.9, 0.03, 0.2]
```

Similar to previous example, we simulate the system, this time using a more exciting input in order to be able to identify several parameters
```@example paramest
Tperiod = 200
t = 0:Ts:1000
u1 = vcat.(0.25 .* sign.(sin.(2pi/Tperiod .* (t ./ 40).^2)) .+ 0.25)
u2 = vcat.(0.25 .* sign.(sin.(2pi/Tperiod .* (t ./ 40).^2 .+ pi/2)) .+ 0.25)
u  = vcat.(u1,u2)
x0 = Float64[2,2,3,3]
x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, u, p_true)[1:end-1]
y = measurement.(x, u, 0, 0)
y = [y .+ 0.01randn(ny) for y in y]

plot(
    plot(reduce(hcat, x)', title="States"),
    plot(reduce(hcat, u)', title="Inputs")
)
```


This time, we define a cost function for the optimizer to optimize, we'll use the sum of squared errors (`sse`). It's important to define the UKF with an initial state distribution with the same element type as the parameter vector so that automatic differentiation through the state estimator works, hence the explicit casting `T.(x0)`.
```@example paramest
nx = 4
R1 = Diagonal([0.1, 0.1, 0.1, 0.1])
R2 = Diagonal((1e-2)^2 * ones(ny))
x0 = [2, 2, 3, 3]

function cost(p::Vector{T}) where T
    kf = UnscentedKalmanFilter(discrete_dynamics, measurement, R1, R2, MvNormal(T.(x0), R1); ny, nu)
    LowLevelParticleFilters.sse(kf, u, y, p)
end
```
We generate a random initial guess for the estimation problem
```@example paramest
p_guess = p_true .+  0.1*p_true .* randn(length(p_true))
```
and solve it using Optim
```@example paramest
using Optim
res = Optim.optimize(
    cost,
    p_guess,
    BFGS(),
    Optim.Options(
        show_trace = true,
        show_every = 5,
        iterations = 100,
        time_limit = 30,
    ),
    autodiff = :forward,
)
```

We started out with a normalized parameter error of
```@example paramest
using LinearAlgebra
norm(p_true - p_guess) / norm(p_true)
```
and ended with
```@example paramest
p_opt = res.minimizer
norm(p_true - p_opt) / norm(p_true)
```
There is no guarantee that we will recover the true parameters for this system, especially not if the input excitation is poor, but we will generally find parameters that results in a good predictor for the system (this is after all what we're optimizing for). A tool like [StructuralIdentifiability.jl](https://github.com/SciML/StructuralIdentifiability.jl) may be used to determine the identifiability of parameters and states, something that for this system could look like
```julia
using StructuralIdentifiability

ode = @ODEmodel(
    h1'(t) = -a1/A1 * h1(t) + a3/A1*h3(t) +     gam*k1/A1 * u1(t),
    h2'(t) = -a2/A2 * h2(t) + a4/A2*h4(t) +     gam*k2/A2 * u2(t),
    h3'(t) = -a3/A3*h3(t)                          + (1-gam)*k2/A3 * u2(t),
    h4'(t) = -a4/A4*h4(t)                          + (1-gam)*k1/A4 * u1(t),
	y1(t) = h1(t),
    y2(t) = h2(t),
)

local_id = assess_local_identifiability(ode, 0.99)
```
where we have made the substitution ``\sqrt h \rightarrow h`` due to a limitation of the tool. The output of the above analysis is 
```julia
julia> local_id = assess_local_identifiability(ode, 0.99)
Dict{Nemo.fmpq_mpoly, Bool} with 15 entries:
  a3  => 0
  gam => 1
  k2  => 0
  A4  => 0
  h4  => 0
  h2  => 1
  A3  => 0
  a1  => 0
  A2  => 0
  k1  => 0
  a4  => 0
  h3  => 0
  h1  => 1
  A1  => 0
  a2  => 0
  ```
  indicating that we can not hope to resolve all of the parameters. However, using appropriate regularization from prior information, we might still recover a lot of information about the system. Regularization could easily be added to the function `cost` above, e.g., using a penalty like `(p-p_guess)'Γ*(p-p_guess)` for some matrix ``\Gamma``, to indicate our confidence in the initial guess.