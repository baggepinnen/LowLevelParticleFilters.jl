module LowLevelParticleFilters

export KalmanFilter, ParticleFilter, AuxiliaryParticleFilter, AdvancedParticleFilter, PFstate, index, state, covariance, num_particles, effective_particles, weights, expweights, particles, particletype, smooth, sample_measurement, simulate, loglik, log_likelihood_fun, forward_trajectory, mean_trajectory, update!, predict!, correct!, reset!, metropolis, shouldresample

export densityplot, debugplot, commandplot

using StatsBase, Parameters, Lazy, Yeppp, Random, LinearAlgebra, Printf
import PDMats # To extend some methods on static arrays
using StaticArrays
using Distributions
using StatsPlots

abstract type ResamplingStrategy end
struct ResampleSystematic <: ResamplingStrategy end
struct ResampleSystematicExp <: ResamplingStrategy end

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
