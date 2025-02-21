module LowLevelParticleFiltersDistributionsExt
import LowLevelParticleFilters
using Distributions
using Distributions: MultivariateDistribution, Distribution, UnivariateDistribution, Continuous, Discrete, ValueSupport
using StaticArrays
using Random
using Random: AbstractRNG
using LinearAlgebra

@inline Base.:(-)(x::StaticArray, ::Distributions.Zeros) = x
@inline Base.:(-)(::Distributions.Zeros, x::StaticArray) = -x
@inline Distributions.logpdf(d::Distribution,x,xp,t) = logpdf(d,x-xp)
@inline Distributions.sqmahal(d::MvNormal, x::StaticArray) = Distributions.invquad(d.Σ, x - d.μ)

"""
Mixed value support indicates that the distribution is a mix of continuous and discrete dimensions.
"""
struct Mixed <: ValueSupport end

"""
    TupleProduct(v::NTuple{N,UnivariateDistribution})

Create a product distribution where the individual distributions are stored in a tuple. Supports mixed/hybrid Continuous and Discrete distributions
"""
struct TupleProductType{N,S,V<:NTuple{N,UnivariateDistribution}} <: MultivariateDistribution{S}
    v::V
    function TupleProductType(v::V) where {N,V<:NTuple{N,UnivariateDistribution}}
        all(Distributions.value_support(typeof(d)) == Discrete for d in v) &&
            return new{N,Discrete,V}(v)
        all(Distributions.value_support(typeof(d)) == Continuous for d in v) &&
            return new{N,Continuous,V}(v)
        return new{N,Mixed,V}(v)
    end
end

LowLevelParticleFilters.TupleProduct(d::Distribution...) = TupleProductType(d)
LowLevelParticleFilters.TupleProduct(t) = TupleProductType(t)

Base.length(d::TupleProductType{N}) where N = N
# Distributions._rand!(rng::AbstractRNG, d::TupleProductType, x::AbstractVector{<:Real}) =     broadcast!(dn->rand(rng, dn), x, d.v)

@generated function Distributions._rand!(rng::AbstractRNG, d::TupleProductType{N}, x::AbstractVector{<:Real}) where N
    quote
        Base.Cartesian.@nexprs $N i->(x[i] = rand(rng, d.v[i]))
        x
    end
end

@generated function Distributions._logpdf(d::TupleProductType{N}, x::AbstractVector{<:Real}) where N
    :(Base.Cartesian.@ncall $N Base.:+ i->logpdf(d.v[i], x[i]))
end

# To make it a bit faster also for the regular Product
@generated function Distributions._logpdf(d::Product, x::StaticVector{N}{<:Real}) where N
    :(Base.Cartesian.@ncall $N Base.:+ i->logpdf(d.v[i], x[i]))
end

Distributions.mean(d::TupleProductType) = vcat(mean.(d.v)...)
Distributions.var(d::TupleProductType) = vcat(var.(d.v)...)
Distributions.cov(d::TupleProductType) = Diagonal(var(d))
Distributions.entropy(d::TupleProductType) = sum(entropy, d.v)
Base.extrema(d::TupleProductType) = minimum.(d.v), maximum.(d.v)

@generated function Random.rand(rng::AbstractRNG, d::TupleProductType{N}) where N
    quote
        SVector(Base.Cartesian.@ntuple $N i->(rand(rng, d.v[i])))
    end
end

LowLevelParticleFilters.extended_logpdf(d::Distribution, args...) = Distributions.logpdf(d, args...)

# resolve ambiguity
Base.@propagate_inbounds function LowLevelParticleFilters.propagate_particles!(pf::LowLevelParticleFilters.ParticleFilter, u, p, t::Real, d::Distributions.Sampleable=pf.dynamics_density)
    f = pf.dynamics
    x,xp = pf.state.x, pf.state.xprev
    VecT = eltype(pf.state.x)
    D = length(VecT)
    noise = zeros(D)
    for i = eachindex(x)
        x[i] =  f(xp[i], u, p, t) + VecT(rand!(pf.rng, d, noise))
    end
    x
end

end