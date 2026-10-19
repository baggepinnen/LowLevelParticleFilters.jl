"""
    BatchedParticleFilter(N; nx, nw, dynamics, measurement_loglik, initial_state, first_stage = nothing,
                          ess_threshold = 0.5, unroll = 2)

Particle filter for the evaluation of the log-likelihood on an accelerator (GPU) with Reactant.jl, for a batch
of `B` parameter sets or seeds at once. The user functions operate on all particles and all batch members at
once. A state is a tuple of `nx` arrays of size (B, N), one per state variable; the parameters are a named
tuple whose arrays have the batch as their first dimension (scalars as (B, 1) columns). The functions must
be written with broadcasting only (no branches on values: `ifelse` instead of `if`), so that the same code
runs on `Array`s and, traced by Reactant, on the device.

- `dynamics(X, u, p, t, W) -> X⁺`: `W` is a tuple of `nw` arrays (B, N) of standard normal numbers.
- `measurement_loglik(X, u, y, p, t) -> (B, N)`: log-density of the measurement `y` ((ny, B)).
- `initial_state(p, Z) -> X`: initial particles from the tuple `Z` of `nx` standard normal arrays (B, N).
- `first_stage(X, u, y_next, p, t) -> (B, N)`: first-stage log-weights of the auxiliary particle filter
  (for example the noise-free prediction against `y_next` with an inflated measurement variance);
  `nothing` gives the bootstrap filter.

`ess_threshold`: resampling when the effective sample size is below `ess_threshold N` (the auxiliary filter
resamples at every measurement). `unroll`: time steps per iteration of the traced loop.

The random numbers are counter-based: the normal number `s` of particle `n` in step `k` is a function of
(seed, k, n, s) only, so that the log-likelihood is a deterministic function of the parameters for fixed seeds
(common random numbers), which suits gradient-free optimization.

Evaluate with [`compile_loglik`](@ref), which requires `using Reactant`.
"""
struct BatchedParticleFilter{F, G, H, I}
    N::Int
    nx::Int
    nw::Int
    dynamics::F
    measurement_loglik::G
    initial_state::H
    first_stage::I
    ess_threshold::Float64
    unroll::Int
end

function BatchedParticleFilter(N::Integer; nx, nw, dynamics, measurement_loglik, initial_state,
        first_stage = nothing, ess_threshold = 0.5, unroll = 2)
    BatchedParticleFilter(Int(N), Int(nx), Int(nw), dynamics, measurement_loglik, initial_state, first_stage,
        Float64(ess_threshold), Int(unroll))
end

"""
    ll = compile_loglik(pf::BatchedParticleFilter, u, y, p; T = Float32, burnin = 0)
    ll(p, seeds) -> Vector{Float64}

Compile the log-likelihood of the measurements `y` (ny × T; `NaN` or `missing` in a column for a time step
without a measurement) with the inputs `u` (nu × T) for the batch of parameters `p` (see
[`BatchedParticleFilter`](@ref)). The measurements of the first `burnin` time steps are not in the
log-likelihood. Returns a callable that evaluates the log-likelihoods of the batch members (one per member,
with the seeds `seeds`, a vector of length B) for new parameter values of the same shapes without a new
compilation. `T` is the precision of the particles. Requires `using Reactant` (package extension).
"""
function compile_loglik end
