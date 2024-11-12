module LowLevelParticleFilters

export KalmanFilter, SqKalmanFilter, UnscentedKalmanFilter, DAEUnscentedKalmanFilter, ExtendedKalmanFilter, ParticleFilter, AuxiliaryParticleFilter, AdvancedParticleFilter, PFstate, index, state, covariance, num_particles, effective_particles, weights, expweights, particles, particletype, smooth, sample_measurement, simulate, loglik, log_likelihood_fun, forward_trajectory, mean_trajectory, mode_trajectory, weighted_mean, weighted_cov, update!, predict!, correct!, reset!, metropolis, shouldresample, TupleProduct
@deprecate weigthed_mean weighted_mean
@deprecate weigthed_cov weighted_cov

export densityplot, debugplot, commandplot

using StatsAPI
import StatsAPI: weights, predict!
using StatsBase, Parameters, Lazy, Random, LinearAlgebra, Printf, LoopVectorization
import PDMats # To extend some methods on static arrays
import PDMats: PDMat
using StaticArrays
using Statistics
using RecipesBase
using ForwardDiff
using Polyester

# using SciMLBase
struct NullParameters end

abstract type ResamplingStrategy end
struct ResampleSystematic <: ResamplingStrategy end

abstract type AbstractFilter end

include("PFtypes.jl")
include("solutions.jl")
include("kalman.jl")
include("ukf.jl")
include("filtering.jl")
include("resample.jl")
include("utils.jl")
include("smoothing.jl")
include("plotting.jl")
include("ekf.jl")
include("sq_kalman.jl")

index(f::AbstractFilter) = f.t

if !isdefined(Base, :get_extension) # Backwards compat
    include("../ext/LowLevelParticleFiltersControlSystemsBaseExt.jl")
end

"""
    pdata = pplot(x, w, y, yhat, a, t, pdata; kwargs...)
    pdata = pplot(pf, y, pdata; kwargs...)

To be called inside a particle filter, plots either particle density (`density=true`) or individual particles (`density=false`) \n
Will plot all the real state variables in `xindices` as well as the expected vs real measurements of `yindices`.
# Arguments:
- `x`: `Vector{Vector}(N)`. The states of each particle where `N` number of Particles
- `w`: `Vector(N)`. weight of each particle
- `y`: `Vector{Vector}(T)`. All true outputs. `T` is total number of time steps (will only use index `t`)
- `yhat`: `Vector{Vector}(N)` The expected output per particle. `N` number of Particles
- `a`, `Vector(N)`, reorderng of particles (e.g. `1:N`)
- `t`, Current time step
- `xreal`: `Vector{Vector}(T)`. All true states if available. `T` is total number of time steps (will only use index `t`)
- `xprev`: Same as `x`, but for previous time step, only used when `!density` to show states origins
- `pdata`: Persistant data for plotting. Set to `nothing` in first call and pdata on remaining \n
- `density = true` To only plot the particle trajectories, set (`leftonly=false`)\n
- `leftonly = true`: only plot the left column\n
- `xindices = 1:n_state`\n
- `yindices = 1:n_measurements`\n
Returns: `pdata`

!!! note
    This function requires `using Plots` to be called before it is used.
"""
function pplot end

"""
    commandplot(pf, u, y, p=parameters(pf); kwargs...)

Produce a helpful plot. For customization options (`kwargs...`), see `?pplot`.
After each time step, a command from the user is requested.
- q: quit
- s n: step `n` steps

!!! note
    This function requires `using Plots` to be called before it is used.
"""
function commandplot end

"""
    debugplot(pf, u, y, p=parameters(pf); runall=false, kwargs...)

Produce a helpful plot. For customization options (`kwargs...`), see `?pplot`.
- ` runall=false:` if true, runs all time steps befor displaying (faster), if false, displays the plot after each time step.

The generated plot becomes quite heavy. Initially, try limiting your input to 100 time steps to verify that it doesn't crash.

!!! note
    This function requires `using Plots` to be called before it is used.
"""
function debugplot end

end # module
