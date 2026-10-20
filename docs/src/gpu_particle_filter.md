# Particle filters on a GPU

The computational cost of a particle filter is proportional to the number of particles ``N``, and problems with a high state dimension, a small measurement noise or a large deviation between the proposal and the posterior density may require ``10^5`` particles or more. [`ReactantParticleFilter`](@ref) stores the particles and the weights on an accelerator (GPU) and performs the propagation, the weighting and the resampling there, using [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl). It wraps an ordinary [`ParticleFilter`](@ref) or [`AdvancedParticleFilter`](@ref), defined by the same per-particle functions as on the CPU, and supports the same interface: [`update!`](@ref), [`predict!`](@ref), [`correct!`](@ref), [`loglik`](@ref), [`forward_trajectory`](@ref), [`smooth`](@ref), [`simulate`](@ref), [`AuxiliaryParticleFilter`](@ref) etc.

The functionality is provided by a package extension and requires
```julia
using Reactant
```

## Division of work between the device and the host

In [`update!`](@ref), [`forward_trajectory`](@ref) etc., a time step consists of two calls of compiled device functions:

- [`correct!`](@ref): the measurement log-likelihood of each particle, the normalization of the weights and the log-likelihood of the measurement, which is the only quantity transferred to the host.
- [`predict!`](@ref): the effective sample size, systematic resampling if the effective sample size is below `resample_threshold * N`, and the propagation of the particles through the dynamics including the noise. For an [`AuxiliaryParticleFilter`](@ref), also the first-stage weights of the noise-free predictions.

The time loop, the handling of missing measurements and the accumulation of the log-likelihood are the generic implementations of the package. [`loglik`](@ref) instead performs all time steps in one device program, with the same step functions and the same result as the generic implementation, but without a transfer between the host and the device in each time step; the inputs and measurements of the whole trajectory are transferred once, and the program is compiled for each length of the trajectory. Functions that operate on the particles or the weights on the host, such as [`particles`](@ref), [`weights`](@ref), [`expweights`](@ref), [`weighted_mean`](@ref) and [`forward_trajectory`](@ref), copy them from the device. The particle smoother [`smooth`](@ref) runs on the device, see [Particle smoothing](@ref gpu_smoothing).

## Example: `ParticleFilter`

The model below is a nonlinear system with state dimension 4. The dynamics and the measurement function are written for a single particle, exactly as for the [`ParticleFilter`](@ref).
```julia
using LowLevelParticleFilters, Reactant, StaticArrays, LinearAlgebra
const LLPF = LowLevelParticleFilters

function dynamics(x, u, p, t)
    Ts = 0.05
    SA[x[1] + Ts * x[3],
       x[2] + Ts * x[4],
       x[3] + Ts * (-p[1] * x[3] + p[2] * sin(x[2]) + u[1]),
       x[4] + Ts * (-p[1] * x[4] - p[2] * sin(x[1]))]
end
measurement(x, u, p, t) = SA[x[1], x[2] + 0.1 * x[1]^2]

nx, ny = 4, 2
p = [0.5, 1.0]
df = LLPF.SimpleMvNormal(zeros(nx), Matrix(0.01I(nx))) # Dynamics noise
dm = LLPF.SimpleMvNormal(zeros(ny), Matrix(0.1I(ny)))  # Measurement noise
d0 = LLPF.SimpleMvNormal(zeros(nx), Matrix(1.0I(nx)))  # Initial state

N = 100_000
pf = ParticleFilter(N, dynamics, measurement, df, dm, d0; p, resample_threshold = 0.5)
rpf = ReactantParticleFilter(pf; T = Float32)
```
The filter is used as any other particle filter:
```julia
u = [randn(1) for _ in 1:200]
x, u, y = simulate(pf, u)        # Simulation on the host
ll = loglik(rpf, u, y, p)        # Log-likelihood
sol = forward_trajectory(rpf, u, y, p)
apf = AuxiliaryParticleFilter(rpf)
ll_apf = loglik(apf, u, y, p)
```

## Example: `AdvancedParticleFilter`

An [`AdvancedParticleFilter`](@ref) admits noise that depends on the parameters, enters through a matrix or enters the dynamics nonlinearly, and an arbitrary measurement likelihood. On the device, the noise of the dynamics must be drawn with [`randn_noise`](@ref)`(noise, ...)` instead of `randn`: on the CPU, the filter passes `noise::Bool` and `randn_noise` returns a random sample or zero, on the device, the filter passes a source of random numbers that is evaluated on the device. The model below has the noise intensity `p.σ` as a parameter, process noise that enters through the vector `Bw` (a singular covariance matrix) and measurements quantized to integers.
```julia
const A  = SA[1.0 0.1; 0.0 1.0]
const Bw = SA[0.005, 0.1]
dynamics(x, u, p, t, noise = false) = A*x + Bw * (p.σ * randn_noise(noise))
measurement(x, u, p, t, noise = false) = SA[round(x[1] + noise * 0.1 * randn())] # Used by simulate on the host
function measurement_likelihood(x, u, y, p, t)
    # Probability that x[1] + e, e ~ N(0, 0.1²), rounds to y[1]
    s = 0.1 * sqrt(2)
    log(max(0.5 * (erf((y[1] + 0.5 - x[1]) / s) - erf((y[1] - 0.5 - x[1]) / s)), 1e-30))
end

d0 = LLPF.SimpleMvNormal(zeros(2), Matrix(1.0I(2)))
p = (; σ = 1.0)
apf = AdvancedParticleFilter(100_000, dynamics, measurement, measurement_likelihood, nothing, d0; p)
x, u, y = simulate(apf, [zeros(0) for _ in 1:200])
rpf = ReactantParticleFilter(apf; T = Float32)
ll = [loglik(rpf, u, y, (; σ)) for σ in (0.5, 1.0, 2.0)]
```
(`erf` requires `using SpecialFunctions`.) The same `apf` runs on the CPU unchanged.

## [Particle smoothing](@id gpu_smoothing)

[`smooth`](@ref)`(rpf, M, u, y, p)` performs particle smoothing by forward filtering and backward simulation of `M` trajectories, also for an [`AuxiliaryParticleFilter`](@ref) of a `ReactantParticleFilter`. The cost of the backward simulation is proportional to ``M N T``: in each time step, the transition density between each of the ``M`` smoothed particles and each of the ``N`` filter particles is evaluated, which is well suited for a GPU. The particles and weights of the forward pass remain on the device, which requires memory for ``n_x N T`` numbers; only the smoothed particles are transferred to the host. The result has the same format as for the particle filters on the CPU:
```julia
xb, ll = smooth(rpf, 10_000, u, y, p) # xb: M × T matrix of SVectors
xm = smoothed_mean(xb)               # nx × T
```
The backward simulation uses the transition density defined by the Gaussian `dynamics_density` of a [`ParticleFilter`](@ref). The other smoothing methods of the package, `smooth(rpf, M, u, y, p; method = GenealogySmoother(...))` and `method = FFBSi(; transition_logpdf)`, run on the host after [`forward_trajectory`](@ref) with the filter on the device, also for an [`AdvancedParticleFilter`](@ref). The [`ParticleGibbsSmoother`](@ref) is not supported on the device.

For the first example above with ``T = 100`` time steps, the smoother took 9.5 s with ``N = M = 1000`` and 85 s with ``N = M = 3000`` on a single CPU thread, and 0.10 s and 0.15 s on an NVIDIA H100 GPU. With ``N = M = 3 \cdot 10^4``, it took 2.8 s in `Float32` and 4.7 s in `Float64` on the GPU.

## Requirements on the model

Reactant traces the per-particle functions once with symbolic numbers and evaluates the resulting program for all particles at once. This imposes the following restrictions:

- The functions must not branch on the values of `x`, `u`, `p` or `t`. Use `ifelse(c, a, b)` instead of `if c a else b end`, and `max`/`min` instead of comparisons followed by branches. Functions that branch internally, such as `norm`, are not supported either.
- The functions must not draw random numbers with `rand` or `randn`, since these would be drawn once when the function is traced and compiled as constants. The filter detects such calls and throws an error. Use [`randn_noise`](@ref) in the dynamics of an [`AdvancedParticleFilter`](@ref).
- For a [`ParticleFilter`](@ref), the dynamics noise is additive and Gaussian, given by a density with `mean` and `cov`, such as `SimpleMvNormal` or `Distributions.MvNormal`, or absent (`nothing`). The measurement noise is additive and Gaussian, given in the same way by `measurement_density`, unless the keyword argument `measurement_likelihood = (x, u, y, p, t) -> log p(y | x)` is given.
- Values captured by the functions (closures, global constants) are compiled as constants. Values that change between calls, such as the parameters during an optimization, must be passed in `p`. The parameters may be a vector, a named tuple of numbers and arrays, or a tuple.
- Missing measurements (`missing`) are supported as for the other particle filters.

The device functions are compiled at their first call for each combination of the model functions and the argument types and sizes. The compilation takes several seconds. The compiled functions are shared between filters with the same model functions, so that filters created for new parameter values, for example inside an objective function of an optimizer, do not trigger a new compilation.

## Precision

The keyword argument `T` selects the precision of the particles and the weights on the device, `Float64` by default. `Float32` is faster on most GPUs, in particular on consumer GPUs whose `Float64` throughput is a small fraction of the `Float32` throughput. The parameters, inputs and measurements are converted to `T` on the device. Dynamics with time constants that are long compared with the sample interval may require `Float64`, since the increment of the state in one time step may then be smaller than the resolution of `Float32`.

## Random numbers

The noise and the resampling use a counter-based random-number generator on the device, whose key is drawn from the random-number generator `rng` of the filter in [`reset!`](@ref), which [`loglik`](@ref) and [`forward_trajectory`](@ref) call. The initial particles are drawn from `rng` as well. The log-likelihood is thus a deterministic function of the parameters for a filter created with a seeded `rng`, for example `rng = Xoshiro(1)`, which may be used to obtain common random numbers for a gradient-free optimization of the parameters.

## Performance

The following table lists the time per time step of [`loglik`](@ref) for the first example above (state dimension 4, ``200`` time steps, `resample_threshold = 0.5`) on an NVIDIA H100 GPU, and for the [`ParticleFilter`](@ref) on a single CPU thread with the densities defined by static arrays (`SVector`, `SMatrix`). The [`AdvancedParticleFilter`](@ref) and the [`AuxiliaryParticleFilter`](@ref) take approximately the same time on the device.

| ``N`` | `Float64`, device | `Float32`, device | `ParticleFilter`, CPU |
|------:|------------------:|------------------:|----------------------:|
| ``10^3`` | 0.04 ms | 0.03 ms | 0.33 ms |
| ``10^4`` | 0.06 ms | 0.06 ms | 1.6 ms |
| ``10^5`` | 0.22 ms | 0.21 ms | 16 ms |
| ``10^6`` | 2.5 ms | 2.5 ms | |

On the device, the cost of a time step of `loglik` consists of a fixed part of approximately 0.03–0.04 ms and a part proportional to ``N``. The device is therefore advantageous already for ``N \approx 10^3`` particles. [`update!`](@ref) and [`forward_trajectory`](@ref), which call the device functions in each time step, have a fixed cost of approximately 0.4 ms per time step (0.44 ms for ``N = 10^3``, 0.69 ms for ``N = 10^5``).
