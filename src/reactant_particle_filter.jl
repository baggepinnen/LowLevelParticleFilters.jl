"""
    randn_noise(noise, T::Type{<:AbstractFloat} = Float64)
    randn_noise(noise, ::Type{<:SVector{n}})

Standard normal noise in the dynamics `dynamics(x, u, p, t, noise)` of an [`AdvancedParticleFilter`](@ref): a
random sample if `noise == true`, and zero if `noise == false`. Drawing the noise of the dynamics with this function
instead of `randn` allows a [`ReactantParticleFilter`](@ref) to generate it on the device, where the filter passes a
source of random numbers as `noise`. Example:
```julia
dynamics(x, u, p, t, noise = false) = A*x + B*u + p.σ * Bw * randn_noise(noise, SVector{1})
```
"""
randn_noise(noise::Bool) = noise ? randn() : 0.0
randn_noise(noise::Bool, ::Type{T}) where {T <: AbstractFloat} = noise ? randn(T) : zero(T)
randn_noise(noise::Bool, ::Type{S}) where {S <: StaticVector} = noise ? randn(S) : zero(S)
randn_noise(::Nothing, args...) = randn_noise(false, args...)

"""
State of a [`ReactantParticleFilter`](@ref): the particles `x` and `xprev` (nx × N) and the normalized
log-weights `w` (1 × N) on the device, the key `seed` of the random numbers, the time index `t`, and a buffer `bins`
(N) on the host for the functions that operate on the host, such as the backward simulation of [`smooth`](@ref)
"""
mutable struct DevicePFstate{XT, WT}
    x::XT
    xprev::XT
    w::WT
    seed::UInt64
    t::Base.RefValue{Int}
    bins::Vector{Float64}
end

"""
    ReactantParticleFilter(pf::ParticleFilter; T = Float64, measurement_likelihood = nothing)
    ReactantParticleFilter(pf::AdvancedParticleFilter; T = Float64)
    ReactantParticleFilter(N, dynamics, measurement, dynamics_density, measurement_density, initial_density; T = Float64, measurement_likelihood = nothing, kwargs...)

A particle filter whose particles, weights and resampling are on an accelerator (GPU), using Reactant.jl. Requires
`using Reactant` (package extension).

The model is the one of `pf`, defined by the same per-particle functions as for the filter on the CPU. Reactant
traces these functions for a single particle and evaluates them for all particles at once on the device. They must
therefore be free of branches on the values of `x`, `u`, `p` and `t` (use `ifelse` instead of `if`), and they must not
call `rand` or `randn`.

- For a [`ParticleFilter`](@ref), the noise of the dynamics is additive with the Gaussian density `dynamics_density`
  (or `nothing`), and the measurement noise is additive with the Gaussian density `measurement_density`, unless
  `measurement_likelihood(x, u, y, p, t) -> log p(y | x)` is given.
- For an [`AdvancedParticleFilter`](@ref), the dynamics `dynamics(x, u, p, t, noise)` must draw their noise with
  [`randn_noise`](@ref)`(noise, ...)`, and the measurements enter through `measurement_likelihood(x, u, y, p, t)`.
  Noise that depends on the parameters, enters through a matrix or enters nonlinearly is thus supported.

The interface is the one of the other particle filters: [`update!`](@ref), [`predict!`](@ref), [`correct!`](@ref),
[`loglik`](@ref), [`forward_trajectory`](@ref), [`smooth`](@ref), [`simulate`](@ref),
`AuxiliaryParticleFilter(ReactantParticleFilter(pf); first_stage)`, etc. A time step requires one call of a compiled
device function for `correct!` and one for `predict!`; only the inputs, the measurement and the log-likelihood are
transferred between the host and the device. [`loglik`](@ref) performs all time steps in one device program. Functions that operate on the particles or the weights on the host,
such as [`particles`](@ref), [`weights`](@ref) and [`weighted_mean`](@ref), copy them from the device.

- `T`: precision of the particles and weights on the device. `Float32` is faster on most GPUs, but the accuracy may
  be insufficient for slow dynamics.

The random numbers of the noise and of the resampling are generated on the device by a counter-based generator,
whose key is drawn from the random-number generator `rng` of `pf` in [`reset!`](@ref). A log-likelihood is thus a
deterministic function of the parameters for a seeded `rng`. The device functions are compiled at their first call
for each combination of the model functions and the argument types and sizes, which may take several seconds; filters
created later with the same functions, for example for other parameter values, use the compiled functions. Values
captured by the model functions (closures) are compiled as constants, values that change between calls must be passed
in the parameters `p`.
"""
struct ReactantParticleFilter{PF<:Union{ParticleFilter, AdvancedParticleFilter}, ST<:DevicePFstate, T, ML} <: AbstractParticleFilter
    "The filter that defines the model and the settings"
    pf::PF
    state::ST
    "Log-likelihood `(x, u, y, p, t) -> log p(y | x)` of a measurement, or `nothing` for `measurement_density`"
    measurement_likelihood::ML
    "Constant device arrays of the filter (particle indices, noise factors)"
    device_data::Dict{Symbol, Any}
end

"""
    make_device_state(pf, T)

Device state of a [`ReactantParticleFilter`](@ref), defined in the Reactant extension
"""
function make_device_state end

default_measurement_likelihood(::ParticleFilter) = nothing
default_measurement_likelihood(pf::AdvancedParticleFilter) = pf.measurement_likelihood

function ReactantParticleFilter(pf::Union{ParticleFilter, AdvancedParticleFilter}; T::Type{<:AbstractFloat} = Float64,
        measurement_likelihood = default_measurement_likelihood(pf))
    isempty(methods(make_device_state)) &&
        error("ReactantParticleFilter requires Reactant.jl, load it with `using Reactant`")
    measurement_likelihood === nothing && pf isa ParticleFilter && pf.measurement_density === nothing &&
        throw(ArgumentError("either a measurement_density or a measurement_likelihood is required"))
    s = make_device_state(pf, T)
    ReactantParticleFilter{typeof(pf), typeof(s), T, typeof(measurement_likelihood)}(pf, s, measurement_likelihood, Dict{Symbol, Any}())
end

ReactantParticleFilter(N::Integer, args...; T::Type{<:AbstractFloat} = Float64, measurement_likelihood = nothing, kwargs...) =
    ReactantParticleFilter(ParticleFilter(N, args...; kwargs...); T, measurement_likelihood)

function Base.getproperty(pf::ReactantParticleFilter, s::Symbol)
    s ∈ fieldnames(ReactantParticleFilter) && return getfield(pf, s)
    s === :nx && return size(getfield(pf, :state).x, 1)
    getproperty(getfield(pf, :pf), s)
end

parameters(pf::ReactantParticleFilter) = parameters(pf.pf)

@forward ReactantParticleFilter.pf sample_state, sample_measurement, dynamics, measurement, dynamics_density, measurement_density, initial_density, resample_threshold, resampling_strategy, rng

"Precision of the particles on the device"
device_eltype(::ReactantParticleFilter{<:Any, <:Any, T}) where {T} = T

num_particles(s::DevicePFstate) = size(s.x, 2)
num_particles(pf::ReactantParticleFilter) = num_particles(state(pf))
particletype(pf::ReactantParticleFilter) = SVector{pf.nx, device_eltype(pf)}

"""
    particles(pf::ReactantParticleFilter)
    weights(pf::ReactantParticleFilter)
    expweights(pf::ReactantParticleFilter)

The particles (a vector of `SVector`s), the normalized log-weights and the normalized weights, copied from the
device to the host
"""
particles(pf::ReactantParticleFilter) = particles(state(pf))
particles(s::DevicePFstate) = collect(reinterpret(SVector{size(s.x, 1), eltype(s.x)}, vec(Array(s.x))))
weights(pf::ReactantParticleFilter) = weights(state(pf))
weights(s::DevicePFstate) = Float64.(vec(Array(s.w)))
function expweights(pf::ReactantParticleFilter)
    we = exp.(weights(pf))
    we ./= sum(we) # Normalized in Float64, the weights on the device may be Float32
end
weighted_mean(pf::ReactantParticleFilter) = weighted_mean(particles(pf), expweights(pf))

(pf::ReactantParticleFilter)(u, y, p = parameters(pf), t = index(pf)*pf.Ts) = update!(pf, u, y, p, t)
