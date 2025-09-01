module LowLevelParticleFilters

export KalmanFilter, SqKalmanFilter, UnscentedKalmanFilter, DAEUnscentedKalmanFilter, ExtendedKalmanFilter, SqExtendedKalmanFilter, IteratedExtendedKalmanFilter, ParticleFilter, AuxiliaryParticleFilter, AdvancedParticleFilter, SignalNames, PFstate, index, state, covariance, num_particles, effective_particles, weights, expweights, particles, particletype, smooth, sample_measurement, simulate, loglik, loglik_x, log_likelihood_fun, forward_trajectory, mean_trajectory, mode_trajectory, weighted_mean, weighted_cov, weighted_quantile, update!, predict!, correct!, reset!, metropolis, shouldresample, TupleProduct
export double_integrator_covariance, double_integrator_covariance_smooth, n_integrator_covariance, n_integrator_covariance_smooth
export UKFWeights, TrivialParams, MerweParams, WikiParams
export IMM, interact!, combine!
export RBPF, RBParticle, RBMeasurementModel
export LinearMeasurementModel, EKFMeasurementModel, IEKFMeasurementModel, UKFMeasurementModel, CompositeMeasurementModel
export KalmanFilteringSolution, KalmanSmoothingSolution, ParticleFilteringSolution
@deprecate weigthed_mean weighted_mean
@deprecate weigthed_cov weighted_cov

export densityplot, debugplot, commandplot
export unscentedplot, unscentedplot!, covplot, covplot!

using StatsAPI
import StatsAPI: weights, predict!
using StatsBase, Lazy, Random, LinearAlgebra, Printf, SLEEFPirates
import PDMats # To extend some methods on static arrays
import PDMats: PDMat
using StaticArrays
using Statistics
using RecipesBase
using ForwardDiff
using MaybeInplace
import SpecialFunctions # Normpdf and friends

# using SciMLBase
struct NullParameters end

abstract type ResamplingStrategy end
struct ResampleSystematic <: ResamplingStrategy end

abstract type AbstractFilter end
abstract type AbstractKalmanFilter <: AbstractFilter end

macro maybe_threads(flag, expr)
    quote
        if $(flag)
            Threads.@threads $expr
        else
            $expr
        end
    end |> esc
end

include("indexing_matrix.jl")
include("signalnames.jl")
include("PFtypes.jl")
include("solutions.jl")
include("measurement_model.jl")
include("kalman.jl")
include("ukf.jl")
include("filtering.jl")
include("resample.jl")
include("utils.jl")
include("smoothing.jl")
include("plotting.jl")
include("ekf.jl")
include("iekf.jl")
include("sq_kalman.jl")
include("sq_ekf.jl")
include("imm.jl")
include("rbpf.jl")

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

"""
    unscentedplot(ukf;          n_std = 2, N = 100, dims=1:2)
    unscentedplot(sigmapoints;  n_std = 2, N = 100, dims=1:2)

Plot the sigma points and their corresponding covariance ellipse. `dims` indicate the two dimensions to plot, and defaults to the first two dimensions.

If an UKF is passed, the sigma points after the last dynamics update are extracted from the filter. To plot the sigma points of the output, pass those in manually, they are available as `ukf.measurement_model.cache.x0` and `ukf.measurement_model.cache.x1`, denoting the input and output points of the measurement model.

Note: The covariance of the sigma points does not in general equal the predicted covariance of the state, since the state covariance is updated as `cov(sigmapoints) + R1`. Only when `AUGD = true` (augmented dynamics), the covariance of the state is given by the first `nx` sigmapoints.

See also `covplot`.

!!! note
    This function requires `using Plots` to be called before it is used.
"""
function unscentedplot end
function unscentedplot! end

"""
    covplot(μ, Σ; n_std = 2, dims=1:2)
    covplot(kf; n_std = 2, dims=1:2)

Plot the covariance ellipse of the state `μ` and covariance `Σ`. `dims` indicate the two dimensions to plot, and defaults to the first two dimensions.

If a Kalman-type filter is passed, the state and covariance are extracted from the filter.

See also `unscentedplot`.

!!! note
    This function requires `using Plots` to be called before it is used.
"""
function covplot end
function covplot! end
end # module
