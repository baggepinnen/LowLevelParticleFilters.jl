# Performance tips

## StaticArrays
Use of [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl) is recommended for optimal performance when the state dimension is small, e.g., less than about 10-15 for Kalman filters and less than about 100 for particle filters. In the section [Parameter optimization](https://baggepinnen.github.io/LowLevelParticleFilters.jl/dev/parameter_estimation/#Using-an-optimizer) we demonstrate one workflow that makes use of StaticArrays everywhere it is needed for an [`UnscentedKalmanFilter`](@ref) in order to get a completely allocation free filter. The following arrays must be static for this to hold

- The initial state distribution (the vector and matrix passed to `d0 = MvNormal(μ, Σ)` for Kalman filters). If you are performing parameter optimization with gradients derived using ForwardDiff.jl, these must further have the correct element type. How to achieve this is demonstrated in the liked example above.
- Inputs `u` measured outputs `y`.
- In case of Kalman filters, the dynamic model matrices `A`, `B`, `C`, `D` and the covariance matrices `R1`, `R2`.
- The dynamics functions for [`UnscentedKalmanFilter`](@ref) and particle filters must further return static arrays when passed static arrays as inputs.

## Simplified measurement model
While using, e.g., an [`UnscentedKalmanFilter`](@ref) for a system with a linear or almost linear measurement model, consider using a linear or EKF measurement model. See [Measurement models](@ref) for more details.

## Analysis using JET
All flavors of Kalman filters are analyzed for potential runtime dispatch using [JET.jl](https://github.com/aviatesk/JET.jl). This analysis is performed [in the tests](https://github.com/baggepinnen/LowLevelParticleFilters.jl/blob/master/test/test_jet.jl) and generally requires a completely static filter using static arrays internally. See the tests for an example of how to set a filter up this way.

# High performance Distributions
When `using LowLevelParticleFilters`, a number of methods related to distributions are defined for static arrays, making `logpdf` etc. faster. We also provide a new kind of distribution: `TupleProduct <: MultivariateDistribution` that behaves similarly to the `Product` distribution. The `TupleProduct` however stores the individual distributions in a tuple, has compile-time known length and supports `Mixed <: ValueSupport`, meaning that it can be a product of both `Continuous` and `Discrete` dimensions, something not supported by the standard `Product`. Example

```julia
using BenchmarkTools, LowLevelParticleFilters, Distributions, StaticArrays
dt = TupleProduct((Normal(0,2), Normal(0,2), Binomial())) # Mixed value support
```

A small benchmark

```julia
sv = @SVector randn(2)
d = Distributions.Product([Normal(0,2), Normal(0,2)])
dt = TupleProduct((Normal(0,2), Normal(0,2)))
dm = MvNormal(2, 2)
@btime logpdf($d,$(Vector(sv)))  # 19.536 ns (0 allocations: 0 bytes)
@btime logpdf($dt,$(Vector(sv))) # 13.742 ns (0 allocations: 0 bytes)
@btime logpdf($dm,$(Vector(sv))) # 11.392 ns (0 allocations: 0 bytes)
```

```julia
@btime logpdf($d,$sv)  # 13.964 ns (0 allocations: 0 bytes)
@btime logpdf($dt,$sv) # 12.817 ns (0 allocations: 0 bytes)
@btime logpdf($dm,$sv) # 8.383  ns (0 allocations: 0 bytes)
```

Without loading `LowLevelParticleFilters`, the timing for the native distributions are the following

```julia
@btime logpdf($d,$sv)  # 18.040 ns (0 allocations: 0 bytes)
@btime logpdf($dm,$sv) # 9.938  ns (0 allocations: 0 bytes)
```

