module LowLevelParticleFiltersDistributionsExt
import LowLevelParticleFilters
import LowLevelParticleFilters: validationplot, validationplot!
using Distributions
using Distributions: MultivariateDistribution, Distribution, UnivariateDistribution, Continuous, Discrete, ValueSupport, Chisq
using StaticArrays
using Random
using Random: AbstractRNG
using LinearAlgebra
using StatsBase
using RecipesBase

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

@generated function Random.rand!(rng::AbstractRNG, d::TupleProductType{N}, out::AbstractMatrix{<:Real}) where N
    quote
        Base.Cartesian.@nexprs $N i->(rand!(rng, d.v[i], view(out, i, :)))
        out
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

function compute_nis(sol)
    map(eachindex(sol.e, sol.S)) do i
        e = sol.e[i]
        S = sol.S[i]
        if isnothing(S)
            return NaN
        end
        e'*(S \ e)
    end
end

@userplot Validationplot

@recipe function validationplot(p::Validationplot; σ=0.95)
    sol = p.args[1]
    # Extract data
    e = sol.e  # Innovation sequence
    S = sol.S  # Innovation covariances
    u = sol.u  # Inputs
    timevec = sol.t
    ny = length(e[1])
    nu = length(u[1])
    T = length(e)
    names = sol.f.names
    name = names.name
    ynames = names.y
    unames = names.u

    # Setup layout
    layout --> (2, 2)
    size --> (1000, 800)

    # Convert to matrices for easier manipulation
    e_mat = reduce(hcat, e)'  # T×ny matrix
    u_mat = reduce(hcat, u)'  # T×nu matrix

    # Subplot 1: RMS of Innovation
    @series begin
        subplot := 1
        seriestype := :bar
        title --> "RMS of Innovation"
        xlabel --> "Output"
        ylabel --> "RMS"
        label --> ""
        legend --> false
        xticks --> (1:ny, ynames)
        rms = [sqrt(mean(e_mat[:, i].^2)) for i in 1:ny]
        1:ny, rms
    end

    # Subplot 2: NIS with chi-squared bounds
    if !isempty(S) && !isnothing(S[1])
        nis = compute_nis(sol)
        chi2_lower = quantile(Chisq(ny), (1-σ)/2)
        chi2_upper = quantile(Chisq(ny), 1-(1-σ)/2)

        @series begin
            subplot := 2
            title --> "Normalized Innovation Squared (NIS)"
            xlabel --> "Time"
            ylabel --> "NIS"
            label --> "NIS $name"
            seriestype := :scatter
            markersize := 2
            timevec, nis
        end

        @series begin
            subplot := 2
            label --> "$(100*σ)% bounds"
            seriestype := :hline
            linestyle := :dash
            linecolor := :black
            primary := [true false]
            linewidth := 2
            [chi2_upper chi2_lower]
        end

    end

    # Subplot 3: Autocorrelation of Innovation
    maxlag = min(50, T÷4)
    white_noise_bound = 1.96 / sqrt(T)

    for i in 1:ny
        acf = autocor(e_mat[:, i], 0:maxlag)
        @series begin
            framestyle --> :zerolines
            subplot := 3
            title --> "Innovation Autocorrelation"
            xlabel --> "Lag"
            ylabel --> "Autocorrelation"
            label --> "$(ynames[i])"
            seriestype := :stem
            markershape := :circle
            0:maxlag, acf
        end
    end

    @series begin
        subplot := 3
        label --> "95% bounds"
        seriestype := :hline
        linestyle := :dash
        linecolor := :black
        primary := [true false]
        linewidth := 2
        [white_noise_bound -white_noise_bound]
    end


    # Subplot 4: Cross-correlation between Innovation and Past Inputs
    zero_corr_bound = 1.96 / sqrt(T)

    for i in 1:ny
        for j in 1:nu
            ccf = crosscor(e_mat[:, i], u_mat[:, j], 1:maxlag)
            label_str = "e$(i)-u$(j)"
            @series begin
                framestyle --> :zerolines
                subplot := 4
                title --> "Innovation-Input Cross-correlation"
                xlabel --> "Lag"
                ylabel --> "Cross-correlation"
                label --> label_str
                seriestype := :stem
                markershape := :circle
                1:maxlag, ccf
            end
        end
    end

    @series begin
        subplot := 4
        label --> "95% bounds"
        seriestype := :hline
        linestyle := :dash
        linecolor := :black
        primary := [true false]
        linewidth := 2
        [zero_corr_bound -zero_corr_bound]
    end

end

end