export weigthed_mean, weigthed_cov, plot_trajectories, scatter_particles, logsumexp!, smoothed_mean, smoothed_cov, smoothed_trajs, plot_priors

"""
ll = logsumexp!(w, we [, maxw])
Normalizes the weight vector `w` and returns the weighted log-likelihood

https://arxiv.org/pdf/1412.8695.pdf eq 3.8 for p(y)
https://discourse.julialang.org/t/fast-logsumexp/22827/7?u=baggepinnen for stable logsumexp
"""
function logsumexp!(w,we,maxw=Ref(zero(eltype(w))))::eltype(w)
    offset,maxind = findmax(w)
    w  .-= offset
    Yeppp.exp!(we,w)
    s    = sum_all_but(we,maxind) # s = ∑wₑ-1
    we .*= 1/(s+1)
    w  .-= log1p(s)
    maxw[] += offset
    log1p(s) + maxw[] - log(length(w))
end

function logsumexp!(w,we)::eltype(w)
    offset,maxind = findmax(w)
    w  .-= offset
    Yeppp.exp!(we,w)
    s    = sum_all_but(we,maxind) # s = ∑wₑ-1
    we .*= 1/(s+1)
    w  .-= log1p(s)
    log1p(s) + offset - log(length(w))
end

@inline logsumexp!(s) = logsumexp!(s.w,s.we,s.maxw)
@inline logsumexp!(pf::AbstractParticleFilter) = logsumexp!(pf.state)
@inline logsumexp!(pf::SigmaFilter) = logsumexp!(pf.w, pf.we)

"""
    expnormalize!(out,w)
    expnormalize!(w)
- `out .= exp.(w)/sum(exp,w)`. Does not modify `w`
- If called with only one argument, `w` is modified in place
"""
function expnormalize!(we,w)
    offset,maxind = findmax(w)
    w .-= offset
    Yeppp.exp!(we,w)
    w .+= offset
    s    = sum_all_but(we,maxind) # s = ∑wₑ-1
    we .*= 1/(s+1)
end

function expnormalize!(w)
    offset,maxind = findmax(w)
    w .-= offset
    Yeppp.exp!(w,w)
    s    = sum_all_but(w,maxind) # s = ∑wₑ-1
    w .*= 1/(s+1)
end


function sum_all_but(w,i)
    w[i] -= 1
    s = sum(w)
    w[i] += 1
    s
end

function reset_weights!(s)
    N = num_particles(s)
    fill!(s.w, log(1/N))
    fill!(s.we, 1/N)
    s.maxw[] = 0
end
reset_weights!(pf::AbstractParticleFilter) = reset_weights!(state(pf))

function permute_with_buffer!(x, buf, j)
    for i in eachindex(x)
        buf[i] = x[j[i]]
    end
    copyto!(x,buf)
end


"""
    numparameters(f)
Returns the number of parameters of `f` for the method which has the most parameters. This function is shamefully borrowed from [DiffEqBase.jl](https://github.com/JuliaDiffEq/DiffEqBase.jl/blob/master/src/utils.jl#L6)
"""
function numargs(f)
    numparam = [num_types_in_tuple(m.sig) for m in methods(f)]
    return ((numparam .- 1)...,) #-1 in v0.5 since it adds f as the first parameter
end

function num_types_in_tuple(sig)
    length(sig.parameters)
end

function num_types_in_tuple(sig::UnionAll)
    length(Base.unwrap_unionall(sig).parameters)
end


# Make distributions faster for static arrays

@inline Base.:(-)(x::StaticArray, ::Distributions.ZeroVector) = x
@inline Base.:(-)(::Distributions.ZeroVector, x::StaticArray) = x
Distributions.logpdf(d::Distribution,x,xp,t) = logpdf(d,x-xp)
Distributions.sqmahal(d::MvNormal, x::StaticArray) = Distributions.invquad(d.Σ, x - d.μ)
@inline PDMats.invquad(a::PDMats.ScalMat, x::StaticVector) = dot(x,x) * a.inv_value
PDMats.invquad(a::PDMats.PDMat, x::StaticVector) = dot(x, a \ x) # \ not implemented
Base.:(\)(a::PDMats.PDMat, x::StaticVector) = a.chol \ x
PDMats.invquad(a::PDMats.PDiagMat, x::StaticVector) = PDMats.wsumsq(a.inv_diag, x)



"""
Mixed value support indicates that the distribution is a mix of continuous and discrete dimensions.
"""
struct Mixed <: ValueSupport end

"""
    TupleProduct(v::NTuple{N,UnivariateDistribution})

Create a product distribution where the individual distributions are stored in a tuple. Supports mixed/hybrid Continuous and Discrete distributions
"""
struct TupleProduct{N,S,V<:NTuple{N,UnivariateDistribution}} <: MultivariateDistribution{S}
    v::V
    function TupleProduct(v::V) where {N,V<:NTuple{N,UnivariateDistribution}}
        all(Distributions.value_support(typeof(d)) == Discrete for d in v) &&
            return new{N,Discrete,V}(v)
        all(Distributions.value_support(typeof(d)) == Continuous for d in v) &&
            return new{N,Continuous,V}(v)
        return new{N,Mixed,V}(v)
    end
end
TupleProduct(d::Distribution...) = TupleProduct(d)
Base.length(d::TupleProduct{N}) where N = N
# Distributions._rand!(rng::AbstractRNG, d::TupleProduct, x::AbstractVector{<:Real}) =     broadcast!(dn->rand(rng, dn), x, d.v)

@generated function Distributions._rand!(rng::AbstractRNG, d::TupleProduct{N}, x::AbstractVector{<:Real}) where N
    quote
        Base.Cartesian.@nexprs $N i->(x[i] = rand(rng, d.v[i]))
        x
    end
end

@generated function Distributions._logpdf(d::TupleProduct{N}, x::AbstractVector{<:Real}) where N
    :(Base.Cartesian.@ncall $N Base.:+ i->logpdf(d.v[i], x[i]))
end

# To make it a bit faster also for the regular Product
@generated function Distributions._logpdf(d::Product, x::StaticVector{N}{<:Real}) where N
    :(Base.Cartesian.@ncall $N Base.:+ i->logpdf(d.v[i], x[i]))
end

Distributions.mean(d::TupleProduct) = vcat(mean.(d.v)...)
Distributions.var(d::TupleProduct) = vcat(var.(d.v)...)
Distributions.cov(d::TupleProduct) = Diagonal(var(d))
Distributions.entropy(d::TupleProduct) = sum(entropy, d.v)
