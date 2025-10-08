# Parameter Estimation
State estimation is an integral part of many parameter-estimation methods. Below, we will illustrate several different methods of performing parameter estimation. We can roughly divide the methods into two camps
1. Methods that optimize prediction error or likelihood by tweaking model parameters.
2. Methods that add the parameters to be estimated as state variables in the model and estimate them using standard state estimation. 

From the first camp, we provide som basic functionality for maximum-likelihood estimation and MAP estimation, described below. An example of (2), joint state and parameter estimation, is provided in [Joint state and parameter estimation](@ref).


## Maximum-likelihood estimation
Filters calculate the likelihood and prediction errors while performing filtering, this can be used to perform maximum likelihood estimation or prediction-error minimization. One can estimate all kinds of parameters using this method, in the example below, we will estimate the noise covariance. We may for example plot likelihood as function of the variance of the dynamics noise like this:


### Generate data by simulation
This simulates the same linear system as on the index page of the documentation
```@example ml_map
using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Distributions, Plots
nx = 2   # Dimension of state
nu = 2   # Dimension of input
ny = 2   # Dimension of measurements
N = 2000 # Number of particles

const dg = MvNormal(ny,1.0)          # Measurement noise Distribution
const df = MvNormal(nx,1.0)          # Dynamics noise Distribution
const d0 = MvNormal(@SVector(randn(nx)),2.0)   # Initial state Distribution

const A = SA[1 0.1; 0 1]
const B = @SMatrix [0.0 0.1; 1 0.1]
const C = @SMatrix [1.0 0; 0 1]

dynamics(x,u,p,t) = A*x .+ B*u 
measurement(x,u,p,t) = C*x
vecvec_to_mat(x) = copy(reduce(hcat, x)') # Helper function
pf = ParticleFilter(N, dynamics, measurement, df, dg, d0)
xs,u,y = simulate(pf,300,df)
```

### Compute likelihood for various values of the parameters
Since this example looks for a single parameter only, we can plot the likelihood as a function of this parameter. If we had been looking for more than 2 parameters, we typically use an optimizer instead (see further below).
```@example ml_map
p = nothing
svec = exp10.(LinRange(-0.8, 1.2, 60))
llspf = map(svec) do s
    df = MvNormal(nx,s)
    pfs = ParticleFilter(N, dynamics, measurement, df, dg, d0)
    loglik(pfs, u, y, p)
end
llspfaux = map(svec) do s
    df = MvNormal(nx,s)
    pfs = AuxiliaryParticleFilter(N, dynamics, measurement, df, dg, d0)
    loglik(pfs, u, y, p)
end
plot( svec, llspf,
    xscale = :log10,
    title = "Log-likelihood",
    xlabel = "Dynamics noise standard deviation",
    lab = "PF",
)
plot!(svec, llspfaux, yscale=:identity, xscale=:log10, lab="AUX PF", c=:green)
vline!([svec[findmax(llspf)[2]]], l=(:dash,:blue), primary=false)
```
the correct value for the simulated data is 1 (the simulated system is the same as on the front page of the docs).

We can do the same with a Kalman filter, shown below. When using Kalman-type filters, one may also provide a known state sequence if one is available, such as when the data is obtained from a simulation or in an instrumented lab setting. If the state sequence is provided, state-prediction errors are used for log-likelihood estimation instead of output-prediction errors.
```@example ml_map
eye(n) = SMatrix{n,n}(1.0I(n))
llskf = map(svec) do s
    kfs = KalmanFilter(A, B, C, 0, s^2*eye(nx), eye(ny), d0)
    loglik(kfs, u, y, p)
end
llskfx = map(svec) do s # Kalman filter with known state sequence, possible when data is simulated
    kfs = KalmanFilter(A, B, C, 0, s^2*eye(nx), eye(ny), d0)
    loglik_x(kfs, u, y, xs, p)
end
plot!(svec, llskf, yscale=:identity, xscale=:log10, lab="Kalman", c=:red)
vline!([svec[findmax(llskf)[2]]], l=(:dash,:red), primary=false)
plot!(svec, llskfx, yscale=:identity, xscale=:log10, lab="Kalman with known state sequence", c=:purple)
vline!([svec[findmax(llskfx)[2]]], l=(:dash,:purple), primary=false)
```

the result can be quite noisy due to the stochastic nature of particle filtering. The particle filter likelihood agrees with the Kalman-filter estimate, which is optimal for the linear example system we are simulating here, apart for when the noise variance is small. Due to particle depletion, particle filters often struggle when dynamics-noise is too small. This problem is mitigated by using a greater number of particles, or simply by not using a too small covariance.

## MAP estimation
Maximum a posteriori estimation (MAP) is similar to maximum likelihood (ML), but includes also prior knowledge of the distribution of the parameters in a way that is similar to parameter regularization. In this example, we will estimate the variance of the noises in the dynamics and the measurement functions.

To solve a MAP estimation problem, we need to define a function that takes a parameter vector and returns a filter, the parameters are used to construct the covariance matrices:
```@example ml_map
filter_from_parameters(θ, pf = nothing) = KalmanFilter(A, B, C, 0, exp(θ[1])^2*eye(nx), exp(θ[2])^2*eye(ny), d0) # Works with particle filters as well
nothing # hide
```

The call to `exp` on the parameters is so that we can define log-normal priors

```@example ml_map
priors = [Normal(0,2),Normal(0,2)]
```

Now we call the function `log_likelihood_fun` that returns a function to be minimized

```@example ml_map
ll = log_likelihood_fun(filter_from_parameters, priors, u, y, p)
nothing # hide
```

Since this is once again a low-dimensional problem, we can plot the LL on a 2d-grid

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
We proceed like we did for MAP above, but when calling the function `metropolis`, we will get the entire posterior distribution of the parameter vector, for the small cost of a massive increase in the amount of computations. [`metropolis`](@ref) runs the [Metropolis Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm), or more precisely if a particle filter is used, the "Particle Marginal Metropolis Hastings" (PMMH) algorithm. Here we use the Kalman filter simply to have the documentation build a bit faster, it can be quite heavy to run.

```@example ml_map
filter_from_parameters(θ, pf = nothing) = KalmanFilter(A, B, C, 0, exp(θ[1])^2*I(nx), exp(θ[2])^2*I(ny), d0) # Works with particle filters as well
nothing # hide
```

The call to `exp` on the parameters is so that we can define log-normal priors

```@example ml_map
priors = [Normal(0,2),Normal(0,2)]
ll     = log_likelihood_fun(filter_from_parameters, priors, u, y, p)
θ₀     = log.([1.0, 1.0]) # Starting point
nothing # hide
```

We also need to define a function that suggests a new point from the "proposal distribution". This can be pretty much anything, but it has to be symmetric since I was lazy and simplified an equation.

```@example ml_map
draw   = θ -> θ .+ 0.05 .* randn.() # This function dictates how new proposal parameters are being generated. 
burnin = 200 # remove this many initial samples ("burn-in period")
@info "Starting Metropolis algorithm"
@time theta, lls = metropolis(ll, 2200, θ₀, draw) # Run PMMH for 2200  iterations
thetam = reduce(hcat, theta)'[burnin+1:end,:] # Build a matrix of the output
histogram(exp.(thetam), layout=(3,1), lab=["R1" "R2"]); plot!(lls[burnin+1:end], subplot=3, lab="log likelihood") # Visualize
```
In this example, we initialize the MH algorithm on the correct value `θ₀`, in general, you'd see a period in the beginning where the likelihood (bottom plot) is much lower than during the rest of the sampling, this is the reason we remove a number of samples in the beginning, typically referred to as "burn in".


If you are lucky, you can run the above threaded as well. I tried my best to make particle filters thread safe with their own rngs etc., but your milage may vary. For threading to help, the dynamics must be non-allocating, e.g., by using StaticArrays etc.

```@example ml_map
@time thetalls = LowLevelParticleFilters.metropolis_threaded(burnin, ll, 2200, θ₀, draw, nthreads=2)
histogram(exp.(thetalls[:,1:2]), layout=3)
plot!(thetalls[:,3], subplot=3)
```

## Bayesian inference using  DynamicHMC.jl
The following snippet of code performs the same estimation as above, but uses the much more sophisticated HMC sampler in [DynamicHMC.jl](https://www.tamaspapp.eu/DynamicHMC.jl/stable/worked_example/) rather than the PMMH sampler above. This package requires the log-likelihood function to be wrapped in a custom struct that implements the `LogDensityProblems` interface, which is done below. We also indicate that we want to use ForwardDiff.jl to compute the gradients for fast sampling.
```julia
using DynamicHMC, LogDensityProblemsAD, ForwardDiff, LogDensityProblems, LinearAlgebra, Random

struct LogTargetDensity{F}
    ll::F
    dim::Int
end
LogDensityProblems.logdensity(p::LogTargetDensity, θ) = p.ll(θ)
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
LogDensityProblems.capabilities(::Type{LogTargetDensity}) = LogDensityProblems.LogDensityOrder{0}()

function filter_from_parameters(θ, pf = nothing)
    # It's important that the distribution of the initial state has the same
    # element type as the parameters. DynamicHMC will use Dual numbers for differentiation,
    # hence, we make sure that d0 has `eltype(d0) = eltype(θ)`
    T = eltype(θ)
    d0 = MvNormal(T.(d0.μ), T.(d0.Σ))
    KalmanFilter(A, B, C, 0, exp(θ[1])^2*eye(nx), exp(θ[2])^2*eye(ny), d0) 
end
ll = log_likelihood_fun(filter_from_parameters, priors, u, y, p)

D = length(θ₀)
ℓπ = LogTargetDensity(ll, D)
∇P = ADgradient(:ForwardDiff, ℓπ)

results = mcmc_with_warmup(Random.default_rng(), ∇P, 3000)
DynamicHMC.Diagnostics.summarize_tree_statistics(results.tree_statistics)
lls = [ts.π for ts in results.tree_statistics]

histogram(exp.(results.posterior_matrix)', layout=(3,1), lab=["R1" "R2"])
plot!(lls, subplot=3, lab="log likelihood") # Visualize
```

## Joint state and parameter estimation
In this example, we'll show how to perform parameter estimation by treating a parameter as a state variable. This method can not only estimate constant parameters, but also **time-varying parameters**. The system we will consider is a quadruple tank, where two upper tanks feed into two lower tanks. The outlet for tank 1 can vary in size, simulating, e.g., that something partially blocks the outlet. We start by defining the dynamics on a form that changes the outlet area ``a_1`` at time ``t=500``:
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

## Parameter Estimation using MUKF
The [`MUKF`](@ref) (Marginalized Unscented Kalman Filter) provides an alternative approach to parameter estimation that can be advantageous when parameters have **linear time evolution** and enter **multiplicatively** into the system dynamics. Unlike the UKF approach above where parameters are added to the state vector, MUKF explicitly separates the nonlinear states from linearly-evolving parameters, leading to:
- **Deterministic estimation**: No particle randomness, making it suitable for gradient-based optimization of hyperparameters
- **Computational efficiency**: Uses fewer sigma points than UKF for the same total state dimension
- **Natural formulation**: Parameters with linear evolution (e.g., random walk models) fit naturally into the MUKF framework

This approach is particularly well-suited for **online estimation in control systems** where deterministic, differentiable estimators are preferred, and for **disturbance and parameter estimation** where the unknowns have approximately linear dynamics.

### Problem: Quadrotor with Unknown Mass and Drag
We consider a simplified quadrotor model where the mass and drag coefficient are unknown and time-varying. The system has 8 dimensions total:
- **Nonlinear state** (6D): position ``[x, y, z]`` and velocity ``[v_x, v_y, v_z]``
- **Linear parameters** (2D): mass ``m`` and drag coefficient ``C_d``

The dynamics are given by:
```math
\begin{aligned}
\dot{x} &= v_x \\
\dot{y} &= v_y \\
\dot{z} &= v_z \\
\dot{v}_x &= \frac{F_x - C_d \cdot v_x |v_x|}{m} \\
\dot{v}_y &= \frac{F_y - C_d \cdot v_y |v_y|}{m} \\
\dot{v}_z &= \frac{F_z - C_d \cdot v_z |v_z|}{m} - g
\end{aligned}
```

Note how the parameters ``m`` and ``C_d`` enter **multiplicatively** - they scale the effect of forces and drag. This makes them perfect candidates for the linear substate in MUKF.

```@example mukfparam
using LowLevelParticleFilters
using SeeToDee
using Distributions
using StaticArrays
using Plots, LinearAlgebra, Random
Random.seed!(0) # For reproducibility

# System dimensions
nxn = 6  # Nonlinear state: [x, y, z, vx, vy, vz]
nxl = 2  # Linear parameters: [m, Cd]
nx = nxn + nxl
nu = 3   # Control inputs: [Fx, Fy, Fz] (thrust forces)
ny = 6   # Measurements: [x, y, z, vx, vy, vz] (GPS + velocity)

# Physical constants
g = 9.81  # Gravity (m/s²)
Ts = 0.01        # Sample time

nothing # hide
```
We'll simulate a scenario where:
- Mass decreases linearly from 1.0 to 0.85 kg (battery drain)
- Drag increases abruptly at t=50s from 0.01 to 0.015 (damage/configuration change)

### MUKF Formulation
For MUKF, we need to separate the dynamics into:
1. **Nonlinear part** ``f_n(x^n, u, p, t)``: dynamics of position/velocity *without* parameters
2. **Coupling matrix** ``A_n(x^n)``: how the linear parameters affect the nonlinear state
3. **Linear evolution** ``A_l``: how parameters evolve (typically identity for random walk)

```@example mukfparam
# Nonlinear dynamics (without parameters)
# This is the "shape" of the dynamics - how states evolve
function quadrotor_dynamics_nl(xn, u, p, t)
    x, y, z, vx, vy, vz = xn
    Fx, Fy, Fz = u

    # The nonlinear part: position derivatives and gravity
    SA[
        vx,
        vy,
        vz,
        Fx,      # Force without mass scaling
        Fy,
        Fz - g   # Gravity effect
    ]
end

# Coupling matrix An: how parameters [m, Cd] affect the nonlinear state
# This encodes that forces are scaled by 1/m and drag by -Cd/m
function An_matrix(xn, u, p, t)
    x, y, z, vx, vy, vz = xn

    # Parameters affect only the velocity derivatives (rows 4-6)
    # Column 1 is 1/m effect, Column 2 is Cd/m effect
    SA[
        0.0    0.0                    # ẋ not affected by parameters
        0.0    0.0                    # ẏ not affected by parameters
        0.0    0.0                    # ż not affected by parameters
        1.0    -vx*abs(vx)           # v̇x: F/m - Cd·vx|vx|/m
        1.0    -vy*abs(vy)           # v̇y: F/m - Cd·vy|vy|/m
        1.0    -vz*abs(vz)           # v̇z: F/m - Cd·vz|vz|/m
    ]
end

# Discrete coupling matrix (scaled by sampling time for proper discretization)
An_matrix_discrete(xn, u, p, t) = An_matrix(xn, u, p, t) * Ts

# Linear parameter evolution: random walk (parameters are constant + noise)
Al = SA[1.0 0.0; 0.0 1.0]  # Identity matrix
Bl = zeros(SMatrix{nxl, nu})  # Parameters don't depend on control

# Measurement: we measure all states directly (GPS + velocity sensors)
measurement(xn, u, p, t) = xn  # Measure position and velocity
Cl = zeros(SMatrix{ny, nxl})   # Parameters not directly measured

# Discretize the nonlinear dynamics
discrete_dynamics_nl = SeeToDee.Rk4(quadrotor_dynamics_nl, Ts)

# Note on discretization: For the filter, we discretize fn using Rk4 and scale An by Ts.
# This is appropriate when the coupling matrix An represents continuous-time effects.
# For the simulation, we'll use Rk4 to integrate the full continuous dynamics.

nothing # hide
```

### Simulation
We'll simulate a hovering scenario with small perturbations, where the mass decreases (battery drain) and drag increases abruptly (damage).

```@example mukfparam
T = 10000  # 100 seconds at 0.01s sampling
t_vec = (0:T-1) .* Ts

# Control: hovering thrust with small variations
m_nominal = 1.0
F_hover = m_nominal * g
u = [SA[F_hover + 0.1*randn(), F_hover + 0.1*randn(), F_hover + 0.1*randn()] for _ in 1:T]

# True parameters (time-varying)
m_true = [t < 50 ? 1.0 - 0.003*t : 0.85 for t in t_vec]  # Linear decrease
Cd_true = [t < 50 ? 0.01 : 0.015 for t in t_vec]         # Abrupt increase

# Simulate true trajectory
function simulate_quadrotor(u, m_true, Cd_true)
    x = zeros(T, nxn)
    x[1, :] = [0, 0, 10, 0, 0, 0]  # Start at 10m altitude

    for i in 1:T-1
        xn = x[i, :]
        params_i = SA[m_true[i], Cd_true[i]]

        # Define full continuous dynamics: ẋ = fn(xn) + An(xn)*params
        function full_dynamics(xn_inner, u_inner, p_inner, t_inner)
            xdot = quadrotor_dynamics_nl(xn_inner, u_inner, nothing, 0)
            An = An_matrix(xn_inner, u_inner, nothing, 0)
            xdot + An * params_i
        end

        # Use Rk4 to integrate full continuous dynamics
        discrete_step = SeeToDee.Rk4(full_dynamics, Ts)
        x[i+1, :] = discrete_step(xn, u[i], nothing, 0)
    end
    return x
end

x_true = simulate_quadrotor(u, m_true, Cd_true)

# Generate noisy measurements
y = x_true .+ 0.01 .* randn.()
y = SVector{ny}.(eachrow(y))
# Plot true trajectory and parameters
p1 = plot(t_vec, x_true[:, 3], label="Altitude (z)", xlabel="Time (s)", ylabel="m", legend=:topright)
p2 = plot(t_vec, m_true, label="Mass", xlabel="Time (s)", ylabel="kg", legend=:topright, c=:blue)
p3 = plot(t_vec, Cd_true, label="Drag", ylabel="kg·s/m", c=:red)
plot(p1, p2, p3)
```

### MUKF Setup and Estimation
Now we set up the MUKF. The key is to provide the unified initial distribution `d0` and specify `nxn` (dimension of nonlinear substate).

```@example mukfparam
# Noise covariances
R1n = SMatrix{nxn,nxn}(Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))  # Process noise for states
R1l = SMatrix{nxl,nxl}(Diagonal([0.005, 0.00001]))  # Small process noise for parameters
R1_full = [[R1n zeros(SMatrix{nxn,nxl})]; [zeros(SMatrix{nxl,nxn}) R1l]]

R2 = SMatrix{ny,ny}(Diagonal([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]))  # Measurement noise

# Initial state estimate (slightly wrong)
x0n = SA[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]  # Position/velocity
x0l = SA[0.9, 0.008]  # Initial parameter guess: m=0.9, Cd=0.008 (both wrong)
x0_full = [x0n; x0l]

R0n = SMatrix{nxn,nxn}(Diagonal([1.0, 1.0, 1.0, 0.5, 0.5, 0.5]))
R0l = SMatrix{nxl,nxl}(Diagonal([0.01, 0.0001]))
R0_full = [[R0n zeros(SMatrix{nxn,nxl})]; [zeros(SMatrix{nxl,nxn}) R0l]]

d0 = LowLevelParticleFilters.SimpleMvNormal(x0_full, R0_full)

# Create measurement model
mm = RBMeasurementModel(measurement, R2, ny)

# Create MUKF
mukf = MUKF(;
    dynamics = discrete_dynamics_nl,
    nl_measurement_model = mm,
    An = An_matrix_discrete,  # Use discrete coupling matrix
    Al,
    Bl,
    Cl,
    R1 = R1_full,
    d0,
    nxn,
    nu,
    ny,
    Ts,
)

# Run estimation
sol_mukf = forward_trajectory(mukf, u, y)

# Extract estimates
x_est_mukf = reduce(hcat, sol_mukf.xt)'
m_est_mukf = x_est_mukf[:, 7]
Cd_est_mukf = x_est_mukf[:, 8]

nothing # hide
```

### Results and Comparison
Let's visualize the parameter estimation performance:

```@example mukfparam
# Plot parameter estimates
p1 = plot(t_vec, m_true, label="True mass", lw=2, xlabel="Time (s)", ylabel="Mass (kg)",
          legend=:topright, c=:black, ls=:dash)
plot!(p1, t_vec, m_est_mukf, label="MUKF estimate", lw=2, c=:blue)

p2 = plot(t_vec, Cd_true, label="True drag", lw=2, xlabel="Time (s)", ylabel="Drag coeff (kg·s/m)",
          legend=:topleft, c=:black, ls=:dash)
plot!(p2, t_vec, Cd_est_mukf, label="MUKF estimate", lw=2, c=:red)

plot(p1, p2, layout=(2,1), size=(800,500))
```

The MUKF successfully tracks both parameters through the gradual mass decrease and the abrupt drag increase at t=50s. The estimation converges quickly from the initial guess.

### Comparison with UKF Approach
For comparison, let's solve the same problem using a standard UKF with augmented state:

```@example mukfparam
# For UKF, we augment the state with parameters
function quadrotor_dynamics_augmented(x_aug, u, p, t)
    # Extract states and parameters
    xn = x_aug[1:6]
    m, Cd = x_aug[7:8]

    # Compute full dynamics
    xdot = quadrotor_dynamics_nl(xn, u, p, t)
    An = An_matrix(xn, u, p, t)
    params = SA[m, Cd]

    xdot_full = xdot + An * params

    # Parameters evolve as random walk
    SA[xdot_full..., 0.0, 0.0]
end

discrete_dynamics_aug = SeeToDee.Rk4(quadrotor_dynamics_augmented, Ts)
measurement_aug(x, u, p, t) = x[1:6]  # Measure only position/velocity

R1_aug = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.005, 0.00001])
R2_aug = R2

ukf = UnscentedKalmanFilter(
    discrete_dynamics_aug,
    measurement_aug,
    R1_aug,
    R2_aug,
    MvNormal(x0_full, R0_full);
    ny = ny,
    nu = nu,
    Ts = Ts
)

sol_ukf = forward_trajectory(ukf, u, y)

# Extract UKF estimates
x_est_ukf = reduce(hcat, sol_ukf.xt)'
m_est_ukf = x_est_ukf[:, 7]
Cd_est_ukf = x_est_ukf[:, 8]

# Compare the two approaches
p1 = plot(t_vec, m_true, label="True", lw=2, xlabel="Time (s)", ylabel="Mass (kg)",
          legend=:topright, c=:black, ls=:dash, title="Mass Estimation")
plot!(p1, t_vec, m_est_mukf, label="MUKF", lw=2, c=:blue, alpha=0.7)
plot!(p1, t_vec, m_est_ukf, label="UKF", lw=2, c=:green, alpha=0.7, ls=:dot)

p2 = plot(t_vec, Cd_true, label="True", lw=2, xlabel="Time (s)", ylabel="Drag coeff",
          legend=:topleft, c=:black, ls=:dash, title="Drag Estimation")
plot!(p2, t_vec, Cd_est_mukf, label="MUKF", lw=2, c=:blue, alpha=0.7)
plot!(p2, t_vec, Cd_est_ukf, label="UKF", lw=2, c=:green, alpha=0.7, ls=:dot)

plot(p1, p2, layout=(2,1), size=(800,500))
```

### Performance Analysis
Let's quantify the estimation accuracy:

```@example mukfparam
using Statistics

# Compute RMSE for parameters (excluding initial transient)
transient = 500  # Exclude first 5 seconds
rmse_m_mukf = sqrt(mean((m_true[transient:end] - m_est_mukf[transient:end]).^2))
rmse_Cd_mukf = sqrt(mean((Cd_true[transient:end] - Cd_est_mukf[transient:end]).^2))

rmse_m_ukf = sqrt(mean((m_true[transient:end] - m_est_ukf[transient:end]).^2))
rmse_Cd_ukf = sqrt(mean((Cd_true[transient:end] - Cd_est_ukf[transient:end]).^2))

println("MUKF - Mass RMSE: $(round(rmse_m_mukf, digits=4)) kg")
println("MUKF - Drag RMSE: $(round(rmse_Cd_mukf, digits=6)) kg·s/m")
println()
println("UKF  - Mass RMSE: $(round(rmse_m_ukf, digits=4)) kg")
println("UKF  - Drag RMSE: $(round(rmse_Cd_ukf, digits=6)) kg·s/m")
```

Both filters perform comparably in terms of accuracy. However, MUKF uses 13 sigma points (2×6+1 for nonlinear state) compared to UKF's 17 sigma points (2×8+1 for full state)



## Using an optimizer
The state estimators in this package are all statistically motivated and thus compute things like the likelihood of the data as a by-product of the estimation. Maximum-likelihood or prediction-error estimation is thus very straight-forward by simply calling a gradient-based optimizer with gradients provided by differentiating through the state estimator using automatic differentiation. In this example, we will continue the example from above, but now estimate all the parameters of the quad-tank process. This time, they will not vary with time. We will first use a standard optimization algorithm from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) to minimize the cost function based on the prediction error, and then use a Gauss-Newton optimizer.

We now define the dynamics function such that it takes its parameters from the `p` input argument. We also define a variable `p_true` that contains the true values that we will use to simulate some estimation data
```@example paramest
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

discrete_dynamics = SeeToDee.Rk4(quadtank, Ts) # Discretize the dynamics using a 4:th order Runge-Kutta integrator
p_true = [1.6, 1.6, 4.9, 0.03, 0.2]
nothing # hide
```

Similar to previous example, we simulate the system, this time using a more exciting input in order to be able to identify several parameters
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
nx = 4
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


### Solving using Optim
We first minimize the cost using the BFGS optimization algorithm from [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)
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
    autodiff = :forward, # Indicate that we want to use forward-mode AD to derive gradients
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


### Solving using Gauss-Newton optimization
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
J = ForwardDiff.jacobian(residuals!, zeros(T * ny), res_gn.minimizer)
Λ = (T - length(p_guess)) * Symmetric(J' * J) # Precision matrix of the estimated parameters
# Σ = inv(Λ) # Covariance matrix of the estimated parameters (only compute this if precision matrix is well conditioned)
svdvals(Λ)
```
In this case, the precision matrix is singular, indicating that there is at least one diretion in parameter space that yields no increase in cost, and we can thus not determine where along a line in this direction the true parameter lies.

Gauss-Newton algorithms are often more efficient at sum-of-squares minimization than the more generic BFGS optimizer. This form of Gauss-Newton optimization of prediction errors is also available through [ControlSystemIdentification.jl](https://baggepinnen.github.io/ControlSystemIdentification.jl/dev/nonlinear/#Identification-of-nonlinear-models), which uses this package undernath the hood.

## Which method should I use?
The methods demonstrated above have slightly different applicability, here, we try to outline which methods to consider for different problems

| Method                | Parameter Estimation | Covariance Estimation | Time Varying Parameters | Online Estimation |
|-----------------------|----------------------|-----------------------|-------------------------|-------------------|
| Maximum likelihood    | 🟢                   | 🟢                    | 🟥                      | 🟥                |
| Joint state/par estim | 🔶                   | 🟥                    | 🟢                      | 🟢                |
| Prediction-error opt. | 🟢                   | 🟥                    | 🟥                      | 🟥                |


When trying to optimize parameters of the noise distributions, most commonly the covariance matrices, maximum-likelihood (or MAP) is the only recommened method. Similarly, when parameters are time varying or you want an online estimate, the method that jointly estimates state and parameter is the only applicable method. When fitting standard parameters, all methods are applicable. In this case the joint state and parameter estimation tends to be inefficient and unneccesarily complex, and it is recommended to opt for maximum likelihood or prediction-error minimization. The prediction-error minimization (PEM) with a Gauss-Newtown optimizer is often the most efficient method for this type of problem.

Maximum likelihood estimation tends to yield an estimator with better estimates of posterior covariance since this is explicitly optimized for, while PEM tends to produce the smallest possible prediction errors.

## Identifiability

### Polynomial methods
There is no guarantee that we will recover the true parameters by perfoming parameter estimation, especially not if the input excitation is poor. For the system in this tutorial, we will generally find parameters that results in a good predictor for the system (this is after all what we're optimizing for), but these may not be the "correct" parameters. A tool like [StructuralIdentifiability.jl](https://github.com/SciML/StructuralIdentifiability.jl) may be used to determine the identifiability of parameters and state variables (for rational systems), something that for this system could look like
```julia
using StructuralIdentifiability

ode = @ODEmodel(
    h1'(t) = -a1/A1 * h1(t) + a3/A1*h3(t) +     gam*k1/A1 * u1(t),
    h2'(t) = -a2/A2 * h2(t) + a4/A2*h4(t) +     gam*k2/A2 * u2(t),
    h3'(t) = -a3/A3*h3(t)                 + (1-gam)*k2/A3 * u2(t),
    h4'(t) = -a4/A4*h4(t)                 + (1-gam)*k1/A4 * u1(t),
	y1(t) = h1(t),
    y2(t) = h2(t),
)

local_id = assess_local_identifiability(ode)
```
where we have made the substitution ``\sqrt h \rightarrow h`` due to a limitation of the tool (it currently only handles rational ODEs). The output of the above analysis is 
```julia
julia> local_id = assess_local_identifiability(ode)
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

### Linear methods
This package also contains an interface to [ControlSystemsBase](https://juliacontrol.github.io/ControlSystems.jl/stable/), which allows you to call `ControlSystemsBase.observability(f, x, u, p, t)` on a filter `f` to linearize (if needed) it in the point `x,u,p,t` and assess observability using linear methods (the PHB test). Also `ControlSystemsBase.obsv(f, x, u, p, t)` for computing the observability matrix is available.

## Videos
Examples of parameter estimation are available here

By using an optimizer to optimize the likelihood of an [`UnscentedKalmanFilter`](@ref):
```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/0RxQwepVsoM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

Estimation of time-varying parameters:
```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/zJcOPPLqv4A?si=XCvpo3WD-4U3PJ2S" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

Adaptive control by means of estimation of time-varying parameters:
```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/Ip_prmA7QTU?si=Fat_srMTQw5JtW2d" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```


