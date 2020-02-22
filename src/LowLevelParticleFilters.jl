module LowLevelParticleFilters

export KalmanFilter, UnscentedKalmanFilter, ParticleFilter, AuxiliaryParticleFilter, AdvancedParticleFilter, SigmaFilter, PFstate, index, state, covariance, num_particles, effective_particles, weights, expweights, particles, particletype, smooth, sample_measurement, simulate, loglik, log_likelihood_fun, forward_trajectory, mean_trajectory, mode_trajectory, update!, predict!, correct!, reset!, metropolis, shouldresample, TupleProduct

export densityplot, debugplot, commandplot, trajectorydensity, dimensiondensity

using StatsBase, Parameters, Lazy, Random, LinearAlgebra, Printf, LoopVectorization
import PDMats # To extend some methods on static arrays
using StaticArrays
using Distributions
using RecipesBase

abstract type ResamplingStrategy end
struct ResampleSystematic <: ResamplingStrategy end

abstract type AbstractFilter end

include("PFtypes.jl")
include("kalman.jl")
include("filtering.jl")
include("resample.jl")
include("utils.jl")
include("smoothing.jl")
include("plotting.jl")

index(f::AbstractFilter) = f.t[]


using Requires
function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plots.jl")
end

end # module
