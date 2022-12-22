# LowLevelParticleFilters

[![CI](https://github.com/baggepinnen/LowLevelParticleFilters.jl/workflows/CI/badge.svg)](https://github.com/baggepinnen/LowLevelParticleFilters.jl/actions)
[![codecov](https://codecov.io/gh/baggepinnen/LowLevelParticleFilters.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/LowLevelParticleFilters.jl)

This is a library for *state estimation*, that is, given measurements ``y(t)`` from a dynamical system, estimate the state vector ``x(t)``. Throughout, we assume dynamics on the form
```math
x(t+1) = f(x(t), u(t), p, t, w(t))\\
y(t) = g(x(t), u(t), p, t, e(t))
```
where ``x`` is the state vector, ``u`` an input, ``p`` some form of parameters, ``t`` is the time and ``w,e`` are disturbances (noise). Throughout the documentation, we often call the function ``f`` `dynamics` and the function ``g`` `measurement`.

The dynamics above describe a *discrete-time* system, i.e., the function ``f`` takes the current state and produces the *next state*. This is in contrast to a *continuous-time* system, where ``f`` takes the current state but produces the *time derivative* of the state. A continuous-time system can be *discretized*, described in detail in [Discretization](@ref).

The parameters ``p`` can be anything, or left out. You may write the dynamics functions such that they depend on ``p`` and include parameters when you create a filter object. You may also override the parameters stored in the filter object when you call any function on the filter object. This behavior is modeled after the SciML ecosystem.

Depending on the nature of ``f`` and ``g``, the best method of estimating the state may vary. If ``f,g`` are linear and the disturbances are additive and Gaussian, the [`KalmanFilter`](@ref) is an optimal state estimator. If any of the above assumptions fail to hold, we may need to resort to more advanced estimators. This package provides several filter types, outlined below.

## Estimator types
We provide a number of filter types
- [`KalmanFilter`](@ref). A standard Kalman filter. Is restricted to linear dynamics (possibly time varying) and Gaussian noise.
- [`ExtendedKalmanFilter`](@ref): For nonlinear systems, the EKF runs a regular Kalman filter on linearized dynamics. Uses ForwardDiff.jl for linearization. The noise model must be Gaussian.
- [`UnscentedKalmanFilter`](@ref): The Unscented Kalman filter often performs slightly better than the Extended Kalman filter but may be slightly more computationally expensive. The UKF handles nonlinear dynamics and measurement model, but still requires an additive Gaussian noise model and assumes all posterior distributions are Gaussian, i.e., can not handle multi-modal posteriors.
- [`ParticleFilter`](@ref): The particle filter is a nonlinear estimator. This filter is simple to use and assumes that both dynamics noise and measurement noise are additive. Particle filters handle multi-modal posteriors.
- [`AuxiliaryParticleFilter`](@ref): This filter is identical to [`ParticleFilter`](@ref), but uses a slightly different proposal mechanism for new particles.
- [`AdvancedParticleFilter`](@ref): This filter gives you more flexibility, at the expense of having to define a few more functions. This filter does not require the noise to be additive and is thus the most flexible filter type.
- [`DAEUnscentedKalmanFilter`](@ref): An Unscented Kalman filter for differential-algebraic systems (DAE).

## Functionality
This package provides 
- Filtering, estimating ``x(t)`` given measurements up to and including time ``t``. We call the filtered estimate ``x(t|t)`` (read as ``x`` at ``t`` given ``t``).
- Smoothing, estimating ``x(t)`` given data up to ``T > t``, i.e., ``x(t|T)``.
- Parameter estimation.

All filters work in two distinct steps.
1. The *prediction* step ([`predict!`](@ref)). During prediction, we use the dynamics model to form ``x(t|t-1) = f(x(t-1), ...)``
2. The *correction* step ([`correct!`](@ref)). In this step, we adjust the predicted state ``x(t|t-1)`` using the measurement ``y(t)`` to form ``x(t|t)``.

In general, all filters represent not only a point estimate of ``x(t)``, but a representation of the complete posterior probability distribution over ``x`` given all the data available up to time ``t``. One major difference between different filter types is how they represent these probability distributions.



# Particle filter
A particle filter represents the probability distribution over the state as a collection of samples, each sample is propagated through the dynamics function ``f`` individually. When a measurement becomes available, the samples, called *particles*, are given a weight based on how likely the particle is given the measurement. Each particle can thus be seen as representing a hypothesis about the current state of the system. After a few time steps, most weights are inevitably going to be extremely small, a manifestation of the curse of dimensionality, and a resampling step is incorporated to refresh the particle distribution and focus the particles on areas of the state space with high posterior probability.

Defining a particle filter is straightforward, one must define the distribution of the noise `df` in the dynamics function, `dynamics(x,u,p,t)` and the noise distribution `dg` in the measurement function `measurement(x,u,p,t)`. Both of these noise sources are assumed to be additive, but can have any distribution. The distribution of the initial state `d0` must also be provided. An example for a linear Gaussian system is given below.

```@example lingauss
using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Distributions, Plots
```

Define problem

```@example lingauss
n = 2   # Dimension of state
m = 1   # Dimension of input
p = 1   # Dimension of measurements
N = 500 # Number of particles

const dg = MvNormal(p,1.0)          # Measurement noise Distribution
const df = MvNormal(n,1.0)          # Dynamics noise Distribution
const d0 = MvNormal(randn(n),2.0)   # Initial state Distribution
nothing # hide
```

Define linear state-space system (using StaticArrays for maximum performance)

```@example lingauss
const A = SA[0.97043   -0.097368
             0.09736    0.970437]
const B = SA[0.1; 0;;]
const C = SA[0 1.0]
nothing # hide
```

Next, we define the dynamics and measurement equations, they both take the signature `(x,u,p,t) = (state, input, parameters, time)` 
```@example lingauss
dynamics(x,u,p,t) = A*x .+ B*u
measurement(x,u,p,t) = C*x
vecvec_to_mat(x) = copy(reduce(hcat, x)') # Helper function
nothing # hide
```
the parameter `p` can be anything, and is often optional. If `p` is not provided when performing operations on filters, any `p` stored in the filter objects (if supported) is used. The default if none is provided and none is stored in the filter is `p = SciMLBase.NullParameters()`.

We are now ready to define and use a filter

```@example lingauss
pf = ParticleFilter(N, dynamics, measurement, df, dg, d0)
```

With the filter in hand, we can simulate from its dynamics and query some properties
```@example lingauss
du = MvNormal(m,1.0)         # Random input distribution for simulation
xs,u,y = simulate(pf,200,du) # We can simulate the model that the pf represents
pf(u[1], y[1])               # Perform one filtering step using input u and measurement y
particles(pf)                # Query the filter for particles, try weights(pf) or expweights(pf) as well
x̂ = weighted_mean(pf)        # using the current state
```

If you want to perform filtering using vectors of inputs and measurements, try any of the functions

```@example lingauss
sol = forward_trajectory(pf, u, y) # Filter whole vectors of signals
x̂,ll = mean_trajectory(pf, u, y)
plot(sol, xreal=xs, markersize=2)
```
`u` ad `y` are then assumed to be vectors of vectors. StaticArrays is recommended for maximum performance.


If [MonteCarloMeasurements.jl](https://github.com/baggepinnen/MonteCarloMeasurements.jl) is loaded, you may transform the output particles to `Matrix{MonteCarloMeasurements.Particles}` with the layout `T × n_states` using `Particles(x,we)`. Internally, the particles are then resampled such that they all have unit weight. This is conventient for making use of the [plotting facilities of MonteCarloMeasurements.jl](https://baggepinnen.github.io/MonteCarloMeasurements.jl/stable/#Plotting-1).

For a full usage example, see the benchmark section below or [example_lineargaussian.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl/blob/master/src/example_lineargaussian.jl)

### Resampling
The particle filter will perform a resampling step whenever the distribution of the weights has become degenerate. The resampling is triggered when the *effective number of samples* is smaller than `pf.resample_threshold` ``\in [0, 1]``, this value can be set when constructing the filter. How the resampling is done is governed by `pf.resampling_strategy`, we currently provide `ResampleSystematic <: ResamplingStrategy` as the only implemented strategy. See https://en.wikipedia.org/wiki/Particle_filter for more info.

# Particle Smoothing
Smoothing is the process of finding the best state estimate given both past and future data. Smoothing is thus only possible in an offline setting. This package provides a particle smoother, based on forward filtering, backward simulation (FFBS), example usage follows:
```@example lingauss
N     = 2000 # Number of particles
T     = 80   # Number of time steps
M     = 100  # Number of smoothed backwards trajectories
pf    = ParticleFilter(N, dynamics, measurement, df, dg, d0)
du    = MvNormal(m,1)     # Control input distribution
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
The [`KalmanFilter`](@ref) ([wiki](https://en.wikipedia.org/wiki/Kalman_filter)) assumes that ``f`` and ``g`` are linear functions, i.e., that they can be written on the form
```math
x(t+1) = Ax(t) + Bu(t) + w(t)\\
y(t) = Cx(t) + Du(t) + e(t)
```
for some matrices ``A,B,C,D`` where ``w \sim N(0, R_1)`` and ``e \sim N(0, R_2)`` are zero mean and Gaussian. The Kalman filter represents the posterior distributions over ``x`` by the mean and a covariance matrix. The magic behind the Kalman filter is that linear transformations of Gaussian distributions remain Gaussian, and we thus have a very efficient way of representing them.

A Kalman filter is easily created using the constructor. Many of the functions defined for particle filters, are defined also for Kalman filters, e.g.:

```@example lingauss
R1 = cov(df)
R2 = cov(dg)
kf = KalmanFilter(A, B, C, 0, R1, R2, d0)
sol = forward_trajectory(kf, u, y) # filtered, prediction, pred cov, filter cov, loglik
nothing # hide
```

It can also be called in a loop like the `pf` above

```julia
for t = 1:T
    kf(u,y) # Performs both correct and predict!!
    # alternatively
    ll, e = correct!(kf, y, nothing, t) # Returns loglikelihood and prediction error
    x     = state(kf)
    R     = covariance(kf)
    predict!(kf, u, nothing, t)
end
```

The matrices in the Kalman filter may be *time varying*, such that `A[:, :, t]` is ``A(t)``. They may also be provided as functions on the form ``A(t) = A(x, u, p, t)``. This works for both dynamics and covariance matrices.

The numeric type used in the Kalman filter is determined from the mean of the initial state distribution, so make sure that this has the correct type if you intend to use, e.g., `Float32` or `ForwardDiff.Dual` for automatic differentiation.

## Smoothing using KF
Kalman filters can also be used for smoothing 
```@example lingauss
kf = KalmanFilter(A, B, C, 0, cov(df), cov(dg), d0)
xT,R,lls = smooth(kf, u, y, p) # Smoothed state, smoothed cov, loglik
nothing # hide
```

Plot and compare PF and KF

```@example lingauss
plot(vecvec_to_mat(xT), lab="Kalman smooth", layout=2)
plot!(xbm', lab="pf smooth")
plot!(vecvec_to_mat(x), lab="true")
```


# Unscented Kalman Filter

The [`UnscentedKalmanFilter`](@ref) represents posterior distributions over ``x`` as Gaussian distributions, but propagate them through a nonlinear function ``f`` by a deterministic sampling of a small number of particles called *sigma points* (this is referred to as the [*unscented transform*](https://en.wikipedia.org/wiki/Unscented_transform)). This UKF thus handles nonlinear functions ``f,g``, but only Gaussian disturbances and unimodal posteriors.

The UKF takes the same arguments as a regular [`KalmanFilter`](@ref), but the matrices defining the dynamics are replaced by two functions, `dynamics` and `measurement`, working in the same way as for the `ParticleFilter` above.
```@example lingauss
ukf = UnscentedKalmanFilter(dynamics, measurement, cov(df), cov(dg), MvNormal([1.,1.]), nu=m, ny=p)
```

## UKF for DAE systems
See the docstring for [`DAEUnscentedKalmanFilter`](@ref) or the [test file](https://github.com/baggepinnen/LowLevelParticleFilters.jl/blob/master/test/test_ukf.jl). This filter is modeled after
> "Nonlinear State Estimation of Differential Algebraic Systems"
> Ravi K. Mandela, Raghunathan Rengaswamy, Shankar Narasimhan

!!! warning
    This filter is still considered experimental and subject to change without respecting semantic versioning. Use at your own risk.

# Extended Kalman Filter
The [`ExtendedKalmanFilter`](@ref) ([EKF](https://en.wikipedia.org/wiki/Extended_Kalman_filter)) is similar to the UKF, but propagates Gaussian distributions by linearizing the dynamics and using the formulas for linear systems similar to the standard Kalman filter. This can be slightly faster than the UKF (not always), but also less accurate for strongly nonlinear systems. The linearization is performed automatically using ForwardDiff.jl. In general, the UKF is recommended over the EKF unless the EKF is faster and computational performance is the top priority.

The EKF has the following signature
```julia
ExtendedKalmanFilter(kf, dynamics, measurement)
```
where `kf` is a standard [`KalmanFilter`](@ref) from which the covariance properties are taken.


# AdvancedParticleFilter
The [`AdvancedParticleFilter`](@ref) works very much like the [`ParticleFilter`](@ref), but admits more flexibility in its noise models.

The [`AdvancedParticleFilter`](@ref) type requires you to implement the same functions as the regular `ParticleFilter`, but in this case you also need to handle sampling from the noise distributions yourself.
The function `dynamics` must have a method signature like below. It must provide one method that accepts state vector, control vector, parameter, time *and* `noise::Bool` that indicates whether or not to add noise to the state. If noise should be added, this should be done inside `dynamics` An example is given below

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
nothing # hide
```

The `measurement_likelihood` function must have a method accepting state, input, measurement, parameter and time, and returning the log-likelihood of the measurement given the state, a simple example below:

```@example lingauss
function measurement_likelihood(x, u, y, p, t)
    logpdf(dg, C*x-y) # A simple linear measurement model with normal additive noise
end
nothing # hide
```

This gives you very high flexibility. The noise model in either function can, for instance, be a function of the state, something that is not possible for the simple [`ParticleFilter`](@ref).
To be able to simulate the [`AdvancedParticleFilter`](@ref) like we did with the simple filter above, the `measurement` method with the signature `measurement(x,u,p,t,noise=false)` must be available and return a sample measurement given state (and possibly time). For our example measurement model above, this would look like this

```@example lingauss
measurement(x, u, p, t, noise=false) = C*x + noise*rand(rng, dg)
nothing # hide
```

We now create the [`AdvancedParticleFilter`](@ref) and use it in the same way as the other filters:

```@example lingauss
apf = AdvancedParticleFilter(N, dynamics, measurement, measurement_likelihood, df, d0)
sol = forward_trajectory(apf, u, y, p)
```

```@example lingauss
plot(sol, xreal=x)
```
We can even use this type as an AuxiliaryParticleFilter

```@example lingauss
apfa = AuxiliaryParticleFilter(apf)
sol = forward_trajectory(apfa, u, y, p)
plot(sol, dim=1, xreal=x) # Same as above, but only plots a single dimension
```


# Troubleshooting and tuning
Tuning a particle filter can be quite the challenge. To assist with this, we provide som visualization tools

```@example lingauss
debugplot(pf,u[1:20],y[1:20], runall=true, xreal=x[1:20])
```



The plot displays all states and all measurements. The heatmap in the background represents the weighted particle distributions per time step. For the measurement sequences, the heatmap represent the distributions of predicted measurements. The blue dots corresponds to measured values. In this case, we simulated the data and we had access to states as well, if we do not have that, just omit `xreal`.
You can also manually step through the time-series using
- `commandplot(pf,u,y; kwargs...)`
For options to the debug plots, see `?pplot`.




Something seems to be off with this figure as the hottest spot is not really where we would expect it

Optimization of the log likelihood can be done by, e.g., global/black box methods, see [BlackBoxOptim.jl](https://github.com/robertfeldt/BlackBoxOptim.jl), see examples in [Parameter Estimation](@ref). Standard tricks apply, such as performing the parameter search in log-space etc.

