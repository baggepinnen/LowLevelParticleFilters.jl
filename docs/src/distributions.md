
# High performance Distributions
When `using LowLevelParticleFilters`, a number of methods related to distributions are defined for static arrays, making `logpdf` etc. faster. We also provide a new kind of distribution: `TupleProduct <: MultivariateDistribution` that behaves similarly to the `Product` distribution. The `TupleProduct` however stores the individual distributions in a tuple, has compile-time known length and supports `Mixed <: ValueSupport`, meaning that it can be a product of both `Continuous` and `Discrete` dimensions, something not supported by the standard `Product`. Example

```julia
using BenchmarkTools, LowLevelParticleFilters, Distributions
dt = TupleProduct((Normal(0,2), Normal(0,2), Binomial())) # Mixed value support
```

A small benchmark

```julia
sv = @SVector randn(2)
d = Distributions.Product([Normal(0,2), Normal(0,2)])
dt = TupleProduct((Normal(0,2), Normal(0,2)))
dm = MvNormal(2, 2)
@btime logpdf($d,$(Vector(sv))) # 32.449 ns (1 allocation: 32 bytes)
@btime logpdf($dt,$(Vector(sv))) # 21.141 ns (0 allocations: 0 bytes)
@btime logpdf($dm,$(Vector(sv))) # 48.745 ns (1 allocation: 96 bytes)
```

```julia
@btime logpdf($d,$sv) # 22.651 ns (0 allocations: 0 bytes)
@btime logpdf($dt,$sv) # 0.021 ns (0 allocations: 0 bytes)
@btime logpdf($dm,$sv) # 0.021 ns (0 allocations: 0 bytes)
```

Without loading `LowLevelParticleFilters`, the timing for the native distributions are the following

```julia
@btime logpdf($d,$sv) # 32.621 ns (1 allocation: 32 bytes)
@btime logpdf($dm,$sv) # 46.415 ns (1 allocation: 96 bytes)
```

