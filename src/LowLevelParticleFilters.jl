module LowLevelParticleFilters

export KalmanFilter, ParticleFilter, AuxiliaryParticleFilter, AdvancedParticleFilter, PFstate, index, state, covariance, num_particles, effective_particles, weights, expweights, particles, particletype, smooth, sample_measurement, simulate, loglik, log_likelihood_fun, forward_trajectory, mean_trajectory, mode_trajectory, update!, predict!, correct!, reset!, metropolis, shouldresample, TupleProduct

export densityplot, debugplot, commandplot, trajectorydensity, dimensiondensity

using StatsBase, Parameters, Lazy, Yeppp, Random, LinearAlgebra, Printf
import PDMats # To extend some methods on static arrays
using StaticArrays
using Distributions
using Plots

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

index(f::AbstractFilter)               = f.t[]

end # module
