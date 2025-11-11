module LowLevelParticleFilters

export KalmanFilter, SqKalmanFilter, UnscentedKalmanFilter, DAEUnscentedKalmanFilter, ExtendedKalmanFilter, SqExtendedKalmanFilter, IteratedExtendedKalmanFilter, ParticleFilter, AuxiliaryParticleFilter, AdvancedParticleFilter, SignalNames, PFstate, index, state, covariance, num_particles, effective_particles, weights, expweights, particles, particletype, smooth, sample_measurement, simulate, loglik, loglik_x, log_likelihood_fun, forward_trajectory, mean_trajectory, mode_trajectory, weighted_mean, weighted_cov, weighted_quantile, update!, predict!, correct!, reset!, metropolis, shouldresample, TupleProduct, autotune_covariances
export double_integrator_covariance, double_integrator_covariance_smooth, n_integrator_covariance, n_integrator_covariance_smooth
export UKFWeights, TrivialParams, MerweParams, WikiParams
export IMM, interact!, combine!
export RBPF, RBParticle, RBMeasurementModel
export MUKF
export UIKalmanFilter
export LinearMeasurementModel, EKFMeasurementModel, IEKFMeasurementModel, UKFMeasurementModel, CompositeMeasurementModel
export KalmanFilteringSolution, KalmanSmoothingSolution, ParticleFilteringSolution
@deprecate weigthed_mean weighted_mean
@deprecate weigthed_cov weighted_cov

export densityplot, debugplot, commandplot
export unscentedplot, unscentedplot!, covplot, covplot!, validationplot, validationplot!, sampleplot, sampleplot!

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

"""
    printarray(A)

Simple utility to convert arrays to strings for error messages.
Required for Julia 1.12+ where arrays cannot be directly interpolated into error messages without JET complaining.
"""
function printarray(A::AbstractVector)
    io = IOBuffer()
    print(io, "[")
    for (i, x) in enumerate(A)
        print(io, x)
        i < length(A) && print(io, ", ")
    end
    print(io, "]")
    return String(take!(io))
end

function printarray(A::AbstractMatrix)
    io = IOBuffer()
    m, n = size(A)
    print(io, "[")
    for i in 1:m
        for j in 1:n
            print(io, A[i,j])
            j < n && print(io, " ")
        end
        i < m && print(io, "; ")
    end
    print(io, "]")
    return String(take!(io))
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
include("mukf.jl")
include("uikf.jl")
include("paramest.jl")

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

"""
    validationplot(sol::KalmanFilteringSolution)

Perform statistical validation of Kalman filter performance by analyzing the innovation sequence.

Creates a 4-subplot figure with the following diagnostics:

1. **Root Mean Square (RMS) of Innovation**: Shows the RMS value for each output dimension.
   - Lower values indicate better filter performance

2. **Normalized Innovation Squared (NIS)**: Plots NIS over time with 95% confidence bounds
   - NIS = ``e(t)' * S(t)⁻¹ * e(t)`` where ``e`` is innovation and ``S`` is innovation covariance
   - Should follow a chi-squared distribution with ``n_y`` degrees of freedom
   - Points consistently outside bounds indicate filter mistuning (wrong R1 or R2)

3. **Autocorrelation of Innovation**: Shows autocorrelation vs lag with white noise bounds
   - Innovations should be white (uncorrelated over time)
   - Autocorrelation outside ±1.96/√T bounds indicates filter issues
   - High autocorrelation suggests model mismatch or underestimated noise

4. **Cross-correlation between Innovation and Past Inputs**: Shows correlation vs lag
   - Should be near zero at all lags (innovations independent of past inputs)
   - Correlation outside ±1.96/√T bounds indicates model errors
   - Non-zero cross-correlation suggests incorrect system model

# Usage
```julia
using Plots, Distributions
sol = forward_trajectory(kf, u, y)
validationplot(sol)
```

!!! note "Requires Distributions.jl"
    This function requires Distributions.jl to be manually installed and loaded.
"""
function validationplot end
function validationplot! end
end # module
