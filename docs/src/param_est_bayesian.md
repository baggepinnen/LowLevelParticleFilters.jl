# Bayesian inference

This page demonstrates Bayesian inference methods for parameter estimation, building upon the maximum-likelihood estimation setup. For the basic setup and introduction to likelihood-based parameter estimation, see [Maximum-likelihood and MAP estimation](@ref).

## Setup

We first set up the same linear system used in the maximum-likelihood example:

```@example bayesian
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
eye(n) = SMatrix{n,n}(1.0I(n))

pf = ParticleFilter(N, dynamics, measurement, df, dg, d0)
xs,u,y = simulate(pf,300,df)
p = nothing
nothing # hide
```

## Bayesian inference using PMMH

We proceed like we did for MAP in [Maximum-likelihood and MAP estimation](@ref), but when calling the function `metropolis`, we will get the entire posterior distribution of the parameter vector, for the small cost of a massive increase in the amount of computations. [`metropolis`](@ref) runs the [Metropolis Hastings algorithm](https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm), or more precisely if a particle filter is used, the "Particle Marginal Metropolis Hastings" (PMMH) algorithm. Here we use the Kalman filter simply to have the documentation build a bit faster, it can be quite heavy to run.

```@example bayesian
filter_from_parameters(θ, pf = nothing) = KalmanFilter(A, B, C, 0, exp(θ[1])^2*I(nx), exp(θ[2])^2*I(ny), d0) # Works with particle filters as well
nothing # hide
```

The call to `exp` on the parameters is so that we can define log-normal priors

```@example bayesian
priors = [Normal(0,2),Normal(0,2)]
ll     = log_likelihood_fun(filter_from_parameters, priors, u, y, p)
θ₀     = log.([1.0, 1.0]) # Starting point
nothing # hide
```

We also need to define a function that suggests a new point from the "proposal distribution". This can be pretty much anything, but it has to be symmetric since I was lazy and simplified an equation.

```@example bayesian
draw   = θ -> θ .+ 0.05 .* randn.() # This function dictates how new proposal parameters are being generated.
burnin = 200 # remove this many initial samples ("burn-in period")
@info "Starting Metropolis algorithm"
@time theta, lls = metropolis(ll, 2200, θ₀, draw) # Run PMMH for 2200  iterations
thetam = reduce(hcat, theta)'[burnin+1:end,:] # Build a matrix of the output
histogram(exp.(thetam), layout=(3,1), lab=["R1" "R2"]); plot!(lls[burnin+1:end], subplot=3, lab="log likelihood") # Visualize
```
In this example, we initialize the MH algorithm on the correct value `θ₀`, in general, you'd see a period in the beginning where the likelihood (bottom plot) is much lower than during the rest of the sampling, this is the reason we remove a number of samples in the beginning, typically referred to as "burn in".


If you are lucky, you can run the above threaded as well. I tried my best to make particle filters thread safe with their own rngs etc., but your milage may vary. For threading to help, the dynamics must be non-allocating, e.g., by using StaticArrays etc.

```@example bayesian
@time thetalls = LowLevelParticleFilters.metropolis_threaded(burnin, ll, 2200, θ₀, draw, nthreads=2)
histogram(exp.(thetalls[:,1:2]), layout=3)
plot!(thetalls[:,3], subplot=3)
```

## Bayesian inference using DynamicHMC.jl
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
