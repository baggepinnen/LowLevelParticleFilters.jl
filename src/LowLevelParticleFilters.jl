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
using Distributions
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

using Requires
function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plots.jl")
end

end # module
