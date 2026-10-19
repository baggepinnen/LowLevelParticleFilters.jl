# Particle filters on a GPU

The computational cost of a particle filter is proportional to the number of particles ``N``, and problems with a high state dimension, a small measurement noise or a large deviation between the proposal and the posterior density may require ``10^5`` particles or more. [`ReactantParticleFilter`](@ref) stores and propagates the particles on an accelerator (GPU) using [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl). It is a wrapper of an ordinary [`ParticleFilter`](@ref), defined by the same per-particle dynamics and measurement functions, and supports the same interface: [`update!`](@ref), [`predict!`](@ref), [`correct!`](@ref), [`loglik`](@ref), [`forward_trajectory`](@ref), [`smooth`](@ref), [`AuxiliaryParticleFilter`](@ref) etc.

The functionality is provided by a package extension and requires
```julia
using Reactant
```

## Division of work between the device and the host

Only the operations whose cost is dominated by the evaluation of the model are performed on the device:

- the propagation of the particles through the dynamics, including the generation of the dynamics noise ([`LowLevelParticleFilters.propagate_particles!`](@ref)),
- the evaluation of the measurement log-likelihood of each particle ([`LowLevelParticleFilters.measurement_equation!`](@ref)),
- the first-stage weights of an [`AuxiliaryParticleFilter`](@ref).

The weights, the normalization, the decision to resample and the resampling itself are performed on the host by the same functions as for the other particle filters of the package. In each time step, the measurement log-likelihoods (``N`` numbers) are therefore transferred from the device to the host, and after a resampling, the ancestor indices (``N`` integers) are transferred to the device. Functions that operate on the particles on the host, such as [`particles`](@ref) and [`weighted_mean`](@ref), copy them from the device.

## Example

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
The device functions are compiled at their first call for each new combination of argument types and sizes. The compilation takes several seconds. Subsequent calls with parameters `p` of the same type and size, for example during an optimization of the parameters, do not trigger a new compilation.

## Requirements on the model

Reactant traces the per-particle functions once with symbolic values and evaluates the resulting program for all particles at once. This imposes the following restrictions:

- The functions must not branch on the values of `x`, `u`, `p` or `t`. Use `ifelse(c, a, b)` instead of `if c a else b end` and `max`/`min` instead of comparisons followed by branches.
- The functions must not draw random numbers. The dynamics noise is added by the filter from `dynamics_density`.
- The dynamics noise is additive and Gaussian, given by a density with `mean` and `cov`, such as `SimpleMvNormal` or `Distributions.MvNormal`, or absent (`nothing`).
- The measurement noise is additive and Gaussian, given in the same way by `measurement_density`. Other measurement models are supported through the keyword argument `measurement_likelihood = (x, u, y, p, t) -> log p(y | x)`, which replaces the density of the additive noise. Note that a [`ParticleFilter`](@ref) without `measurement_density` requires the keyword argument `ny`.
- Missing measurements (`missing`) are supported as for the other particle filters.

## Precision

The keyword argument `T` selects the precision of the particles on the device, `Float64` by default. `Float32` is faster on most GPUs, in particular on consumer GPUs whose `Float64` throughput is a small fraction of the `Float32` throughput. The parameters, inputs and measurements are converted to `T` before they are used on the device. Dynamics with time constants that are long compared with the sample interval may require `Float64`, since the increment of the state in one time step may then be smaller than the resolution of `Float32`. The weights are always computed in `Float64` on the host.

## Random numbers

The device noise is generated by a counter-based random-number generator whose key is drawn from the random-number generator `rng` of the filter in each time step, and the initial particles are drawn from `rng` as well. The resampling uses the global random-number generator, as for the other particle filters of the package. The log-likelihood is thus a deterministic function of the parameters if both the filter's `rng` and the global random-number generator are seeded before each evaluation, which may be used to obtain common random numbers for a gradient-free optimization of the parameters.

## Performance

The following table lists the time per time step of [`loglik`](@ref) for the example above, with ``200`` time steps and `resample_threshold = 0.5`, on an NVIDIA H100 GPU, and for the [`ParticleFilter`](@ref) on a single CPU thread, with the densities defined by static arrays (`SVector`, `SMatrix`). (With `threads = true`, the CPU filter was not faster for this model.)

| ``N`` | `Float64`, device | `Float32`, device | `ParticleFilter`, CPU |
|------:|------------------:|------------------:|----------------------:|
| ``10^3`` | 0.70 ms | 0.72 ms | 0.33 ms |
| ``10^4`` | 0.95 ms | 0.90 ms | 1.6 ms |
| ``10^5`` | 4.4 ms | 2.8 ms | 16 ms |
| ``10^6`` | 25 ms | 22 ms | |

The cost of a time step consists of a fixed part of approximately 0.7 ms, determined by the calls of the device functions and the transfers of the inputs, measurements and parameters, and a part proportional to ``N``, determined mainly by the operations on the weights and the resampling on the host and by the transfers of the weights and the ancestor indices. The device is therefore advantageous for ``N \gtrsim 10^4`` particles.
