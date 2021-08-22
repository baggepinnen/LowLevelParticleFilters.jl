# LowLevelParticleFilters
[![CI](https://github.com/baggepinnen/LowLevelParticleFilters.jl/workflows/CI/badge.svg)](https://github.com/baggepinnen/LowLevelParticleFilters.jl/actions)
[![codecov](https://codecov.io/gh/baggepinnen/LowLevelParticleFilters.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/LowLevelParticleFilters.jl)
[![Documentation, stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable)
[![Documentation, latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://baggepinnen.github.io/LowLevelParticleFilters.jl/latest)

This readme is auto generated from the file [src/example_lineargaussian.jl](https://github.com/baggepinnen/LowLevelParticleFilters.jl/blob/master/src/example_lineargaussian.jl) using [Literate.jl](https://github.com/fredrikekre/Literate.jl)

# Types
We provide a number of filter types
- `ParticleFilter`: This filter is simple to use and assumes that both dynamics noise and measurement noise are additive.
- `AuxiliaryParticleFilter`: This filter is identical to `ParticleFilter`, but uses a slightly different proposal mechanism for new particles.
- `AdvancedParticleFilter`: This filter gives you more flexibility, at the expense of having to define a few more functions. More instructions on this type below.
- `KalmanFilter`. Is what you would expect. Has the same features as the particle filters, but is restricted to linear dynamics and gaussian noise.
- `UnscentedKalmanFilter`. Is also what you would expect. Has almost the same features as the Kalman filters, but handle nonlinear dynamics and measurement model, still requires an additive Gaussian noise model.

# Functionality
- Filtering
- Smoothing
- Parameter estimation using ML or PMMH (Particle Marginal Metropolis Hastings)

# Documentation
[![Documentation, stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable)
[![Documentation, latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://baggepinnen.github.io/LowLevelParticleFilters.jl/latest)