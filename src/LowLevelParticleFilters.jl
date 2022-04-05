module LowLevelParticleFilters

export KalmanFilter, UnscentedKalmanFilter, ExtendedKalmanFilter, ParticleFilter, AuxiliaryParticleFilter, AdvancedParticleFilter, SigmaFilter, PFstate, index, state, covariance, num_particles, effective_particles, weights, expweights, particles, particletype, smooth, sample_measurement, simulate, loglik, log_likelihood_fun, forward_trajectory, mean_trajectory, mode_trajectory, weigthed_mean, weigthed_cov, update!, predict!, correct!, reset!, metropolis, shouldresample, TupleProduct

export densityplot, debugplot, commandplot, trajectorydensity, dimensiondensity

using StatsBase, Parameters, Lazy, Random, LinearAlgebra, Printf, LoopVectorization
import PDMats # To extend some methods on static arrays
import PDMats: PDMat
using StaticArrays
using Distributions
using RecipesBase
using ForwardDiff
# using NonlinearSolve

abstract type ResamplingStrategy end
struct ResampleSystematic <: ResamplingStrategy end

abstract type AbstractFilter end

include("PFtypes.jl")
include("kalman.jl")
include("ukf.jl")
include("filtering.jl")
include("resample.jl")
include("utils.jl")
include("smoothing.jl")
include("plotting.jl")
include("ekf.jl")

index(f::AbstractFilter) = f.t[]


using Requires
function __init__()
    @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plots.jl")
    @require MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca" include("mcm.jl")
    # @require NonlinearSolve = "8913a72c-1f9b-4ce2-8d82-65094dcecaec" include("dae_ukf.jl")
end

end # module
