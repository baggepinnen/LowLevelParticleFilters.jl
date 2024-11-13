# LowLevelParticleFilters
[![CI](https://github.com/baggepinnen/LowLevelParticleFilters.jl/workflows/CI/badge.svg)](https://github.com/baggepinnen/LowLevelParticleFilters.jl/actions)
[![codecov](https://codecov.io/gh/baggepinnen/LowLevelParticleFilters.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/baggepinnen/LowLevelParticleFilters.jl)
[![Documentation, stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable)
[![Documentation, latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev)

This is a library for state estimation, smoothing and parameter estimation.

# Estimator Types
We provide a number of filter types
- `ParticleFilter`: This filter is simple to use and assumes that both dynamics noise and measurement noise are additive.
- `AuxiliaryParticleFilter`: This filter is identical to `ParticleFilter`, but uses a slightly different proposal mechanism for new particles.
- `AdvancedParticleFilter`: This filter gives you more flexibility, at the expense of having to define a few more functions.
- `KalmanFilter`. A standard Kalman filter. Has the same features as the particle filters, but is restricted to linear dynamics (possibly time varying) and Gaussian noise.
- `SqKalmanFilter`. A standard Kalman filter on square-root form (slightly slower but more numerically stable with ill-conditioned covariance).
- `ExtendedKalmanFilter`: For nonlinear systems, the EKF runs a regular Kalman filter on linearized dynamics. Uses ForwardDiff.jl for linearization. The noise model must be Gaussian.
- `UnscentedKalmanFilter`: The Unscented kalman filter often performs slightly better than the Extended Kalman filter but may be slightly more computationally expensive. The UKF handles nonlinear dynamics and measurement models, but still requires an Gaussian noise model (may be non additive). 



# Documentation
[![Documentation, stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://baggepinnen.github.io/LowLevelParticleFilters.jl/stable)
[![Documentation, latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev)
