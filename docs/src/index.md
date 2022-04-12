# LowLevelParticleFilters

[![CI](https://github.com/baggepinnen/LowLevelParticleFilters.jl/workflows/CI/badge.svg)](https://github.com/baggepinnen/LowLevelParticleFilters.jl/actions)
[![codecov](https://codecov.io/gh/baggepinnen/LowLevelParticleFilters.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/LowLevelParticleFilters.jl)

# Types
We provide a number of filter types
- [`ParticleFilter`](@ref): This filter is simple to use and assumes that both dynamics noise and measurement noise are additive.
- [`AuxiliaryParticleFilter`](@ref): This filter is identical to [`ParticleFilter`](@ref), but uses a slightly different proposal mechanism for new particles.
- [`AdvancedParticleFilter`](@ref): This filter gives you more flexibility, at the expense of having to define a few more functions. More instructions on this type below.
- [`KalmanFilter`](@ref). A standard Kalman filter. Has the same features as the particle filters, but is restricted to linear dynamics (possibly time varying) and Gaussian noise.
- [`ExtendedKalmanFilter`](@ref): For nonlinear systems, the EKF runs a regular Kalman filter on linearized dynamics. Uses ForwardDiff.jl for linearization. The noise model must be Gaussian.
- [`UnscentedKalmanFilter`](@ref): The Unscented kalman filter often performs slightly better than the Extended Kalman filter but may be slightly more computationally expensive. The UKF handles nonlinear dynamics and measurement model, but still requires an additive Gaussian noise model.
- [`DAEUnscentedKalmanFilter`](@ref): An Unscented Kalman filter for differential-algebraic systems (DAE).

# Functionality
- Filtering
- Smoothing ([`smooth`](@ref))
- Parameter estimation using ML or PMMH (Particle Marginal Metropolis Hastings)


# Particle filter
Defining a particle filter is straightforward, one must define the distribution of the noise `df` in the dynamics function, `dynamics(x,u,t)` and the noise distribution `dg` in the measurement function `measurement(x,u,t)`. The distribution of the initial state `d0` must also be provided. An example for a linear Gaussian system is given below.

```@example lingauss
using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Distributions, Plots
```

Define problem

```@example lingauss
n = 2   # Dimension of state
m = 2   # Dimension of input
p = 2   # Dimension of measurements
N = 500 # Number of particles

const dg = MvNormal(p,1.0)          # Measurement noise Distribution
const df = MvNormal(n,1.0)          # Dynamics noise Distribution
const d0 = MvNormal(randn(n),2.0)   # Initial state Distribution
```

Define random linear state-space system

```@example lingauss
Tr = randn(n,n)
const A = SMatrix{n,n}(Tr*diagm(0=>LinRange(0.5,0.95,n))/Tr)
const B = @SMatrix randn(n,m)
const C = @SMatrix randn(p,n)
```

Next, we define the dynamics and measurement equations, they both take the signature `(x,u,p,t) = (state, input, parameters, time)` 
```@example lingauss
dynamics(x,u,p,t) = A*x .+ B*u
measurement(x,u,p,t) = C*x
vecvec_to_mat(x) = copy(reduce(hcat, x)') # Helper function
```
the parameter `p` can be anything, and is often optional. If `p` is not provided when performing operations on filters, any `p` stored in the filter objects (if supported) is used. The default if none is provided and none is stored in the filter is `p = SciMLBase.NullParameters()`.

We are now ready to define and use a filter

```@example lingauss
pf = ParticleFilter(N, dynamics, measurement, df, dg, d0)
```

With the filter in hand, we can simulate from its dynamics and query some properties
```@example lingauss
xs,u,y = simulate(pf,200,df) # We can simulate the model that the pf represents
pf(u[1], y[1])               # Perform one filtering step using input u and measurement y
particles(pf)                # Query the filter for particles, try weights(pf) or expweights(pf) as well
x̂ = weigthed_mean(pf)        # using the current state
```

If you want to perform filtering using vectors of inputs and measurements, try any of the functions

```@example lingauss
sol = forward_trajectory(pf, u, y) # Filter whole vectors of signals
x̂,ll = mean_trajectory(pf, u, y)
plot(sol, xreal=xs, markersize=2)
```


If [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl) is loaded, you may transform the output particles to `Matrix{MonteCarloMeasurements.Particles}` with the layout `T × n_states` using `Particles(x,we)`. Internally, the particles are then resampled such that they all have unit weight. This is conventient for making use of the [plotting facilities of MonteCarloMeasurements.jl](https://baggepinnen.github.io/MonteCarloMeasurements.jl/stable/#Plotting-1).

For a full usage example, see the benchmark section below or [example_lineargaussian.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl/blob/master/src/example_lineargaussian.jl)

# Smoothing
We also provide a particle smoother, based on forward filtering, backward simulation (FFBS)

```@example lingauss
N     = 2000 # Number of particles
T     = 80   # Number of time steps
M     = 100  # Number of smoothed backwards trajectories
pf    = ParticleFilter(N, dynamics, measurement, df, dg, d0)
du    = MvNormal(2,1)     # Control input distribution
x,u,y = simulate(pf,T,du) # Simulate trajectory using the model in the filter
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
x,u,y = tosvec.((x,u,y))

xb,ll = smooth(pf, M, u, y) # Sample smooting particles
xbm   = smoothed_mean(xb)   # Calculate the mean of smoothing trajectories
xbc   = smoothed_cov(xb)    # And covariance
xbt   = smoothed_trajs(xb)  # Get smoothing trajectories
xbs   = [diag(xbc) for xbc in xbc] |> vecvec_to_mat .|> sqrt
plot(xbm', ribbon=2xbs, lab="PF smooth")
plot!(vecvec_to_mat(x), l=:dash, lab="True")
```


We can plot the particles themselves as well

```@example lingauss
downsample = 5
plot(vecvec_to_mat(x), l=(4,), layout=(2,1), show=false)
scatter!(xbt[1, 1:downsample:end, :]', subplot=1, show=false, m=(1,:black, 0.5), lab="")
scatter!(xbt[2, 1:downsample:end, :]', subplot=2, m=(1,:black, 0.5), lab="")
```




# Kalman filter
A Kalman filter is easily created using the constructor. Many of the functions defined for particle filters, are defined also for Kalman filters, e.g.:

```@example lingauss
eye(n) = Matrix{Float64}(I,n,n)
kf     = KalmanFilter(A, B, C, 0, eye(n), eye(p), MvNormal([1.,1.]))
sol = forward_trajectory(kf, u, y) # filtered, prediction, pred cov, filter cov, loglik
xT,R,lls = smooth(kf, u, y) # Smoothed state, smoothed cov, loglik
```

It can also be called in a loop like the `pf` above

```julia
p = nothing
for t = 1:T
    kf(u,y,p) # Performs both correct and predict!!
    # alternatively
    ll += correct!(kf, y, p, t) # Returns loglik
    x   = state(kf)
    R   = covariance(kf)
    predict!(kf, u, p, t)
end
```

## Unscented Kalman Filter
The UKF takes the same arguments as a regular [`KalmanFilter`](@ref), but the matrices definiting the dynamics are replaced by two functions, `dynamics` and `measurement`, working in the same way as for the `ParticleFilter` above.
```@example lingauss
ukf    = UnscentedKalmanFilter(dynamics, measurement, eye(n), eye(p), MvNormal([1.,1.]), nu=m, ny=p)
```

### UKF for DAE systems
See the docstring for [`DAEUnscentedKalmanFilter`](@ref) or the [test file](https://github.com/baggepinnen/LowLevelParticleFilters.jl/blob/master/test/test_ukf.jl).


# Troubleshooting
Tuning a particle filter can be quite the challenge. To assist with this, we provide som visualization tools

```@example lingauss
debugplot(pf,u[1:20],y[1:20], runall=true, xreal=x[1:20])
```



The plot displays all states and all measurements. The heatmap in the background represents the weighted particle distributions per time step. For the measurement sequences, the heatmap represent the distibutions of predicted measurements. The blue dots corresponds to measured values. In this case, we simulated the data and we had access to states as well, if we do not have that, just omit `xreal`.
You can also manually step through the time-series using
- `commandplot(pf,u,y; kwargs...)`
For options to the debug plots, see `?pplot`.
## Smoothing using KF

```@example lingauss
kf = KalmanFilter(A, B, C, 0, eye(n), eye(p), MvNormal(diagm(ones(2))))
xT,R,lls = smooth(kf, u, y, p) # Smoothed state, smoothed cov, loglik
```

Plot and compare PF and KF

```@example lingauss
plot(vecvec_to_mat(xT), lab="Kalman smooth", layout=2)
plot!(xbm', lab="pf smooth")
plot!(vecvec_to_mat(x), lab="true")
```




Something seems to be off with this figure as the hottest spot is not really where we would expect it

Optimization of the log likelihood can be done by, e.g., global/black box methods, see [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl). Standard tricks apply, such as performing the parameter search in log-space etc.



# AdvancedParticleFilter
The `AdvancedParticleFilter` type requires you to implement the same functions as the regular `ParticleFilter`, but in this case you also need to handle sampling from the noise distributions yourself.
The function `dynamics` must have a method signature like below. It must provide one method that accepts state vector, control vector, time and `noise::Bool` that indicates whether or not to add noise to the state. If noise should be added, this should be done inside `dynamics` An example is given below

```@example lingauss
using Random
const rng = Random.Xoshiro()
function dynamics(x, u, p, t, noise=false) # It's important that this defaults to false
    x = A*x .+ B*u # A simple dynamics model
    if noise
        x += rand(rng, df) # it's faster to supply your own rng
    end
    x
end
```

The `measurement_likelihood` function must have a method accepting state, measurement and time, and returning the log-likelihood of the measurement given the state, a simple example below:

```@example lingauss
function measurement_likelihood(x, u, y, p, t)
    logpdf(dg, C*x-y) # A simple linear measurement model with normal additive noise
end
```

This gives you very high flexibility. The noise model in either function can, for instance, be a function of the state, something that is not possible for the simple `ParticleFilter`
To be able to simulate the `AdvancedParticleFilter` like we did with the simple filter above, the `measurement` method with the signature `measurement(x,u,t,noise=false)` must be available and return a sample measurement given state (and possibly time). For our example measurement model above, this would look like this

```@example lingauss
measurement(x, u, p, t, noise=false) = C*x + noise*rand(rng, dg)
```

We now create the `AdvancedParticleFilter` and use it in the same way as the other filters:

```@example lingauss
apf = AdvancedParticleFilter(N, dynamics, measurement, measurement_likelihood, df, d0)
sol = forward_trajectory(apf, u, y, p)
```

```@example lingauss
plot(sol, xreal=xs)
```
We can even use this type as an AuxiliaryParticleFilter

```@example lingauss
apfa = AuxiliaryParticleFilter(apf)
sol = forward_trajectory(apfa, u, y, p)
plot(sol, xreal=xs)
```

```@example lingauss
plot(sol, dim=1, xreal=xs) # Same as above, but only plots a single dimension
```
