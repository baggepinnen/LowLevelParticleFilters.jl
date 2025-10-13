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

### Joint state and parameter estimation using MUKF
The [`MUKF`](@ref) (Marginalized Unscented Kalman Filter) is an estimator particularily well suited to joint state and parameter estimation.  When parameters have **linear time evolution** and enter **multiplicatively** into the system dynamics, MUKF explicitly separates the nonlinear state variables from linearly-evolving variables, leading to:
- *Deterministic estimation*: No particle randomness like for particle filters, making it suitable for gradient-based optimization of hyperparameters
- *Computational efficiency*: Uses fewer sigma points than UKF for the same state dimension


#### Problem: Quadrotor with Unknown Mass and Drag
We consider a simplified quadrotor model where the mass and drag coefficient are unknown and time-varying. By cleverly partitioning the state using reparameterizations ``\theta = 1/m`` and ``\varphi = \theta C_d``, we exploit a conditionally linear structure to achieve significant computational savings.

The system has 8 state dimensions total with the following partitioning:
- **Nonlinear substate** (3D): velocities ``[v_x, v_y, v_z]``
- **Linear substate** (5D): positions ``[x, y, z]``, inverse mass ``\theta = 1/m``, and mass-scaled drag ``\varphi = \theta C_d``

The key insight is that positions evolve linearly given the velocities (``\dot{x} = v_x``), and velocity dynamics depend linearly on both ``\theta`` and ``\varphi``. This clever parameterization reduces sigma points from 17 (for full 8D UKF) to only 7 (for 3D nonlinear MUKF).

The physical dynamics are:
```math
\begin{aligned}
\dot{x} &= v_x, \quad \dot{y} = v_y, \quad \dot{z} = v_z \\
\dot{v}_x &= \frac{F_x - C_d \cdot v_x |v_x|}{m}, \quad
\dot{v}_y = \frac{F_y - C_d \cdot v_y |v_y|}{m}, \quad
\dot{v}_z = \frac{F_z - C_d \cdot v_z |v_z|}{m} - g
\end{aligned}
```

Using the inverse mass parameterization ``\theta = 1/m`` and defining ``\varphi = \theta C_d``, we can rewrite the velocity dynamics as:
```math
\begin{aligned}
\dot{v}_x &= \theta F_x - \varphi v_x |v_x|, \quad
\dot{v}_y = \theta F_y - \varphi v_y |v_y|, \quad
\dot{v}_z = \theta F_z - \varphi v_z |v_z| - g
\end{aligned}
```

This reveals the conditionally linear structure: the velocity derivatives depend linearly on both ``\theta`` and ``\varphi``, while positions evolve as ``\dot{x} = v_x`` (linear dependence on velocities). The drag coefficient can be recovered as ``C_d = \varphi / \theta`` when needed. Since ``\theta = 1/m > 0``, this division is well defined as long as the estimate of ``\theta`` is reasonable.

```@example mukfparam
using LowLevelParticleFilters
using SeeToDee
using Distributions
using StaticArrays
using Plots, LinearAlgebra, Random
Random.seed!(0) # For reproducibility

# System dimensions
nxn = 3  # Nonlinear state: [vx, vy, vz] (velocities only)
nxl = 5  # Linear state: [x, y, z, θ, φ] where θ = 1/m, φ = θ*Cd
nx = nxn + nxl
nu = 3   # Control inputs: [Fx, Fy, Fz] (thrust forces)
ny = 6   # Measurements: [x, y, z, vx, vy, vz] (GPS + velocity)

# Physical constants
g = 9.81    # Gravity (m/s²)
Ts = 0.02   # Sample time

nothing # hide
```
We'll simulate a scenario where:
- Mass decreases linearly from 1.0 to 0.85 kg (fuel drain)
- Drag increases abruptly at t=50s from 0.01 to 0.015 (damage/configuration change)

#### MUKF Formulation with Conditionally Linear Structure

By using the parameterization ``\theta = 1/m`` and ``\varphi = \theta C_d``, we exploit the conditionally linear structure from Morelande & Moran (2007), which has the form:

$$
\dot{x} = d(x^n) + A(x^n)x^l = \begin{aligned}
\dot{x}^n &= d_n(x^n) + A_n(x^n) x^l \\
\dot{x}^l &= d_l(x^n) + A_l(x^n) x^l
\end{aligned}
$$

where ``x^n = [v_x, v_y, v_z]`` and ``x^l = [x, y, z, \theta, \varphi]``. The coupling matrix ``A_n(x^n)`` is ``3 \times 5`` and captures how ``\theta`` scales the thrust forces and ``\varphi`` scales the drag forces. The term ``d_l(x^n) = [v_x, v_y, v_z, 0, 0]`` captures how positions depend on velocities.

This clever parameterization reduces the number of sigma points from 17 (for a full 8D UKF with 2nx+1 = 2×8+1) to only 7 (for a 3D nonlinear MUKF with 2×3+1), a 59% reduction. Unscented Kalman filters internally perform a Cholesky factorization of the covariance matrix (to compute sigma points), which scales roughly cubically with state dimension, but the MUKF gets away with factorizing only the part of the covariance corresponding to the nonlinear substate, leading to further computational savings.

```@example mukfparam
# Nonlinear dynamics function returns [dn; dl] where:
# - dn: uncoupled part of nonlinear state dynamics
# - dl: part of linear state dynamics that depends on nonlinear state
function quadrotor_nonlinear_dynamics(xn, u, p, t)
    vx, vy, vz = xn
    Fx, Fy, Fz = u

    # Nonlinear state dynamics (uncoupled part)
    # v̇ = dn + An*xl where xl = [x,y,z,θ,φ]
    dn = SA[
        0.0,     # v̇x base (thrust/drag coupling through An)
        0.0,     # v̇y base
        -g       # v̇z base (gravity is independent of θ and φ)
    ]

    # Linear state dynamics (part depending on xn)
    # ẋ, ẏ, ż = velocities, θ̇ = 0, φ̇ = 0
    dl = SA[vx, vy, vz, 0.0, 0.0]

    return [dn; dl]  # Return 8D vector
end

# Coupling matrix An: how linear state [x,y,z,θ,φ] affects nonlinear state [vx,vy,vz]
# θ scales thrust forces, φ scales drag forces: v̇ = θ*F - φ*v|v|
function An_matrix(xn, u, p, t)
    vx, vy, vz = xn
    Fx, Fy, Fz = u

    # 3×5 matrix: positions don't couple, θ and φ do
    SA[
        0.0  0.0  0.0  Fx        -vx*abs(vx)    # v̇x = θ*Fx - φ*vx|vx|
        0.0  0.0  0.0  Fy        -vy*abs(vy)    # v̇y = θ*Fy - φ*vy|vy|
        0.0  0.0  0.0  Fz        -vz*abs(vz)    # v̇z = θ*Fz - φ*vz|vz| - g
    ]
end

# Discrete coupling matrix (scaled by sampling time)
An_matrix_discrete(xn, u, p, t) = An_matrix(xn, u, p, t) * Ts

# Linear state evolution for discrete-time filter
# Al = I to carry over state from previous time step: xl[k+1] = xl[k] + Ts*dl(xn[k])
Al_discrete = SMatrix{nxl, nxl}(I(nxl))

# Combined A matrix for MUKF: A = [An; Al] (nx × nxl)
A_matrix_discrete(xn, u, p, t) = [An_matrix_discrete(xn, u, p, t); Al_discrete]

# Measurement: we measure [x,y,z,vx,vy,vz]
# This comes from d(xn) + Cl*xl where xl = [x,y,z,θ,φ]
measurement(xn, u, p, t) = SA[0.0, 0.0, 0.0, xn[1], xn[2], xn[3]]  # [0,0,0,vx,vy,vz]
Cl = SA[
    1.0  0.0  0.0  0.0  0.0    # x measurement
    0.0  1.0  0.0  0.0  0.0    # y measurement
    0.0  0.0  1.0  0.0  0.0    # z measurement
    0.0  0.0  0.0  0.0  0.0    # vx measurement (from xn)
    0.0  0.0  0.0  0.0  0.0    # vy measurement (from xn)
    0.0  0.0  0.0  0.0  0.0    # vz measurement (from xn)
]

# Discretize the nonlinear dynamics for the MUKF
discrete_nonlinear_dynamics(x,u,p,t) = [x; @SVector(zeros(5))] + Ts .* quadrotor_nonlinear_dynamics(x,u,p,t)

nothing # hide
```

#### Simulation
We'll simulate a hovering scenario with small perturbations, where the mass decreases (fuel drain) and drag increases abruptly (damage).

```@example mukfparam
Tf = 50  # 50 seconds at 0.01s sampling
t_vec = range(0, stop=Tf, step=Ts)
T = length(t_vec)

# Control: hovering thrust with small variations
m_nominal = 1.0
F_hover = m_nominal * g
u = [SA[F_hover + 0.1*randn(), F_hover + 0.1*randn(), F_hover + 0.1*randn()] for _ in eachindex(t_vec)]

# True parameters (time-varying)
m_true = [t < 25 ? 1.0 - 0.006*t : 0.85 for t in t_vec]  # Linear decrease
θ_true = 1.0 ./ m_true                                    # Inverse mass
Cd_true = [t < 25 ? 0.01 : 0.015 for t in t_vec]         # Abrupt increase
φ_true = θ_true .* Cd_true                                 # Scaled drag φ = θ*Cd

# Simulate true trajectory using known true parameters
function simulate_quadrotor(u, θ_true, Cd_true)
    # Define continuous dynamics with true parameters
    function dynamics_true(x_state, u_inner, p_inner, t_inner)
        θ_i, Cd_i = p_inner
        vx_s, vy_s, vz_s, px_s, py_s, pz_s = x_state
        Fx, Fy, Fz = u_inner
        SA[
            # Velocity derivatives: v̇ = θ*(F - Cd*v|v|) - g_z
            θ_i * (Fx - Cd_i * vx_s * abs(vx_s)),
            θ_i * (Fy - Cd_i * vy_s * abs(vy_s)),
            θ_i * (Fz - Cd_i * vz_s * abs(vz_s)) - g,
            # Position derivatives: ẋ = v
            vx_s,
            vy_s,
            vz_s
        ]
    end
    discrete_step = SeeToDee.Rk4(dynamics_true, Ts)

    x = zeros(T, nx)  # Full state: [vx,vy,vz,x,y,z,θ,φ]
    φ_0 = θ_true[1] * Cd_true[1]
    x[1, :] = [0, 0, 0, 0, 0, 10, θ_true[1], φ_0]  # Start at 10m altitude, zero velocity

    for i in 1:T-1
        vx, vy, vz = x[i, 1], x[i, 2], x[i, 3]
        pos_x, pos_y, pos_z = x[i, 4], x[i, 5], x[i, 6]

        # Use true parameter values at this time step
        θ_i = θ_true[i]
        Cd_i = Cd_true[i]

        p = [θ_i, Cd_i]
        # Integrate 6D state [vx,vy,vz,x,y,z] with true parameters
        state_6d = SA[vx, vy, vz, pos_x, pos_y, pos_z]
        state_next = discrete_step(state_6d, u[i], p, 0)

        # Store next state including parameters
        φ_next = θ_true[i+1] * Cd_true[i+1]
        x[i+1, :] = [state_next[1], state_next[2], state_next[3],  # vx,vy,vz
                     state_next[4], state_next[5], state_next[6],  # x,y,z
                     θ_true[i+1], φ_next]                           # θ,φ
    end
    return x
end

x_true = simulate_quadrotor(u, θ_true, Cd_true)

# Extract measurement components: [x,y,z,vx,vy,vz] from state [vx,vy,vz,x,y,z,θ,φ]
y_true = [SA[x_true[i, 4], x_true[i, 5], x_true[i, 6],  # x,y,z
              x_true[i, 1], x_true[i, 2], x_true[i, 3]]  # vx,vy,vz
          for i in eachindex(t_vec)]

# Add measurement noise
y = [y_true[i] .+ 0.01 .* @SVector(randn(ny)) for i in eachindex(t_vec)]

# Plot true trajectory and parameters
p1 = plot(t_vec, x_true[:, 6], label="Altitude (z)", xlabel="Time (s)", ylabel="m", legend=:topright)
p2 = plot(t_vec, m_true, label="Mass", xlabel="Time (s)", ylabel="kg", legend=:topright, c=:blue)
p3 = plot(t_vec, Cd_true, label="Drag", ylabel="kg·s/m", c=:red)
plot(p1, p2, p3)
```

#### MUKF Setup and Estimation
Now we set up the MUKF, which takes mostly the same configutation options as an [`UnscentedKalmanFilter`](@ref)

```@example mukfparam
# Noise covariances
R1n = SMatrix{nxn,nxn}(Diagonal([0.01, 0.01, 0.01]))  # Process noise for [vx,vy,vz]
R1l = SMatrix{nxl,nxl}(Diagonal([0.01, 0.01, 0.01, 0.0001, 0.000001]))   # Process noise for [x,y,z,θ,φ]
R1 = [[R1n zeros(SMatrix{nxn,nxl})]; [zeros(SMatrix{nxl,nxn}) R1l]]

R2 = SMatrix{ny,ny}(Diagonal([0.1, 0.1, 0.1, 0.05, 0.05, 0.05]))  # Measurement noise

# Initial state estimate (slightly wrong)
m_guess = 0.9  # Wrong mass guess
θ_guess = 1.0 / m_guess
Cd_guess = 0.008  # Wrong Cd guess
φ_guess = θ_guess * Cd_guess  # φ = θ*Cd
x0n = SA[0.0, 0.0, 0.0]  # [vx,vy,vz]
x0l = SA[0.0, 0.0, 10.0, θ_guess, φ_guess]  # [x,y,z,θ,φ]
x0_full = [x0n; x0l]

R0n = SMatrix{nxn,nxn}(Diagonal([0.5, 0.5, 0.5]))  # Uncertainty in velocities
R0l = SMatrix{nxl,nxl}(Diagonal([1.0, 1.0, 1.0, 0.01, 0.0001]))    # Uncertainty in positions, θ, and φ
R0_full = [[R0n zeros(SMatrix{nxn,nxl})]; [zeros(SMatrix{nxl,nxn}) R0l]]

d0 = LowLevelParticleFilters.SimpleMvNormal(x0_full, R0_full)

# Create measurement model
mm = RBMeasurementModel(measurement, R2, ny)

# Create MUKF
mukf = MUKF(;
    dynamics = discrete_nonlinear_dynamics,  # Returns [dn; dl]
    nl_measurement_model = mm,
    A = A_matrix_discrete,    # Combined coupling and dynamics matrix [An; Al]
    Cl,
    R1,
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
θ_est_mukf = x_est_mukf[:, 7]  # θ is the 7th state
φ_est_mukf = x_est_mukf[:, 8]  # φ is the 8th state
m_est_mukf = 1.0 ./ θ_est_mukf  # Convert back to mass
Cd_est_mukf = φ_est_mukf ./ θ_est_mukf  # Recover Cd = φ/θ

nothing # hide
```

#### Results and Comparison
Let's visualize the parameter estimation performance:

```@example mukfparam
# Plot parameter estimates
p1 = plot(t_vec, m_true, label="True mass", lw=2, xlabel="Time (s)", ylabel="Mass (kg)",
          legend=:topright, c=:black, ls=:dash)
plot!(p1, t_vec, m_est_mukf, label="MUKF estimate", lw=2, c=:blue)

p2 = plot(t_vec, Cd_true, label="True drag", lw=2, xlabel="Time (s)", ylabel="Drag coeff (kg·s/m)",
          legend=:topleft, c=:black, ls=:dash)
plot!(p2, t_vec, Cd_est_mukf, label="MUKF estimate", lw=2, c=:blue)

plot(p1, p2, layout=(2,1), size=(800,500))
```

The MUKF successfully tracks both parameters through the gradual mass decrease and the abrupt drag increase at t=50s. The estimation converges quickly from the initial guess.

#### Comparison with UKF Approach
For comparison, let's solve the same problem using a standard UKF with the full 8D state (no exploitation of conditionally linear structure):

```@example mukfparam
# For UKF, treat the entire 8D state uniformly (no structure exploitation)
function quadrotor_dynamics_ukf(x_full, u, p, t)
    xn = x_full[1:nxn]  # [vx,vy,vz]
    xl = x_full[nxn+1:end]  # [x,y,z,θ,φ]

    # Get dynamics and coupling
    dyn = quadrotor_nonlinear_dynamics(xn, u, nothing, 0)
    An = An_matrix(xn, u, nothing, 0)

    # Full derivative
    [dyn[1:nxn] + An * xl; dyn[nxn+1:end]]
end

discrete_dynamics_ukf = SeeToDee.Rk4(quadrotor_dynamics_ukf, Ts)
measurement_ukf(x, u, p, t) = SA[x[4], x[5], x[6], x[1], x[2], x[3]]  # [x,y,z,vx,vy,vz]

R1_ukf = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0001, 0.000001])
R2_ukf = R2

ukf = UnscentedKalmanFilter(
    discrete_dynamics_ukf,
    measurement_ukf,
    R1_ukf,
    R2_ukf,
    MvNormal(x0_full, R0_full);
    ny = ny,
    nu = nu,
    Ts = Ts
)

sol_ukf = forward_trajectory(ukf, u, y)

# Extract UKF estimates
x_est_ukf = reduce(hcat, sol_ukf.xt)'
θ_est_ukf = x_est_ukf[:, 7]  # θ is the 7th state
φ_est_ukf = x_est_ukf[:, 8]  # φ is the 8th state
m_est_ukf = 1.0 ./ θ_est_ukf  # Convert back to mass
Cd_est_ukf = φ_est_ukf ./ θ_est_ukf  # Recover Cd = φ/θ

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

#### Performance Analysis
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

Both filters perform comparably in terms of accuracy. However, MUKF uses only 7 sigma points (2×3+1 for 3D nonlinear state) compared to UKF's 17 sigma points (2×8+1 for 8D full state), a 59% reduction illustrating the computational benefit of exploiting the conditionally linear structure with the φ = θ·Cd parameterization.

We should note here that we have performed slightly different discretizations of the dynamics for the UKF and the MUKF. With the standard UKF, we discretized the entire dynamics using an RK4 method, a very accurate integrator in this context. For the MUKF, we instead discretized the dynamics using a simple forward Euler discretization (by multiplying ``A_n`` and the output of `quadrotor_nonlinear_dynamics` by ``T_s``). The reason for this discrepancy is that the conditional linearity that holds for this system in continuous time no longer holds after discretization, _unless_ we use forward Euler discretization, which is the only scheme simple enough to not mess with the linearity. This primitive discretization is often sufficient for state estimation when sample intervals are short, which they tend to be when controlling quadrotors. See the note under [Discretization](@ref) for more comments regarding accuracy of integration for state estimation.

In special cases, more accurate integration is possible also for MUKF estimators. For example, when ``d_l(x^n) = 0``, the linear state evolves purely linearly as ``x^l_{k+1} = A_l x^l_k``, and we can use the matrix exponential to compute a discretized ``A_l``. When ``A_n = 0``, the nonlinear state evolves purely nonlinearly as ``x^n_{k+1} = f(x^n_k, u_k)``, and we can use any accurate integrator for this part. Even when ``A_n \neq 0``, we could treat the linear part of the nonlinear state evolution ``A_n x^l`` as an additional input to the nonlinear dynamics and use an accurate integrator for this part, this is not yet implemented due to the added complexity it would bring.



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


