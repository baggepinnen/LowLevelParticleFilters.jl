export logsumexp!, smoothed_mean, smoothed_cov, smoothed_trajs

"""
    ll = logsumexp!(w, we [, maxw])
Normalizes the weight vector `w` and returns the weighted log-likelihood

https://arxiv.org/pdf/1412.8695.pdf eq 3.8 for p(y)
https://discourse.julialang.org/t/fast-logsumexp/22827/7?u=baggepinnen for stable logsumexp
"""
function logsumexp!(w,we,maxw=Ref(zero(eltype(w))))::eltype(w)
    offset,maxind = findmax(w)
    w  .-= offset
    LoopVectorization.vmap!(exp,we,w)
    s    = sum_all_but(we,maxind) # s = ∑wₑ-1
    we .*= 1/(s+1)
    w  .-= log1p(s)
    maxw[] = offset
    log1p(s) + maxw[] #- log(length(w))
end

# function logsumexp!(w,we)::eltype(w)
#     offset,maxind = findmax(w)
#     w  .-= offset
#     LoopVectorization.vmap!(exp,we,w)
#     s    = sum_all_but(we,maxind) # s = ∑wₑ-1
#     we .*= 1/(s+1)
#     w  .-= log1p(s)
#     log1p(s) + offset - log(length(w))
# end

@inline logsumexp!(s) = logsumexp!(s.w,s.we,s.maxw)
@inline logsumexp!(pf::AbstractParticleFilter) = logsumexp!(pf.state)

"""
    expnormalize!(out,w)
    expnormalize!(w)
- `out .= exp.(w)/sum(exp,w)`. Does not modify `w`
- If called with only one argument, `w` is modified in place
"""
function expnormalize!(we,w)
    offset,maxind = findmax(w)
    w .-= offset
    LoopVectorization.vmap!(exp,we,w)
    w .+= offset
    s    = sum_all_but(we,maxind) # s = ∑wₑ-1
    we .*= 1/(s+1)
end

function expnormalize!(w)
    offset,maxind = findmax(w)
    w .-= offset
    LoopVectorization.vmap!(exp,w,w)
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

Returns the number of parameters of `f` for the method which has the most parameters. This function is shamelessly borrowed from [DiffEqBase.jl](https://github.com/JuliaDiffEq/DiffEqBase.jl/blob/master/src/utils.jl#L6)
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

@inline PDMats.invquad(a::PDMats.ScalMat, x::StaticVector) = dot(x,x) / a.value
@inline PDMats.invquad(a::PDMats.PDMat, x::StaticVector) = dot(x, a \ x) # \ not implemented
@inline Base.:(\)(a::PDMats.PDMat, x::StaticVector) = a.chol \ x
@inline PDMats.invquad(a::PDMats.PDiagMat, x::StaticVector) = PDMats.wsumsq(1 ./ a.diag, x)

function TupleProduct end

"""
    C = double_integrator_covariance(h, σ2=1)

Returns the covariance matrix of a discrete-time integrator with piecewise constant force as input.
Assumes the state [x; ẋ]. `h` is the sample time. `σ2` scales the covariance matrix with the variance of the noise.

This matrix is rank deficient and some applications might require a small increase in the diagonal to make it positive definite.

See also `double_integrator_covariance_smooth`](@ref) for the version that does not assume piecewise constant noise.
"""
function double_integrator_covariance(h, σ2=1)
    σ2*SA[h^4/4 h^3/2
    h^3/2  h^2]
end

function double_integrator_covariance_smooth(h, σ2=1)
    σ2*SA[h^3/3 h^2/2
    h^2/2  h]
end

function rk4(f::F, Ts0; supersample::Integer = 1) where {F}
    supersample ≥ 1 || throw(ArgumentError("supersample must be positive."))
    # Runge-Kutta 4 method
    Ts = Ts0 / supersample # to preserve type stability in case Ts0 is an integer
    let Ts = Ts
        function (x, u, p, t)
            for _ in 1:supersample
                f1 = f(x, u, p, t)
                f2 = f(x + Ts / 2 * f1, u, p, t + Ts / 2)
                f3 = f(x + Ts / 2 * f2, u, p, t + Ts / 2)
                f4 = f(x + Ts * f3, u, p, t + Ts)
                x += Ts / 6 * (f1 + 2 * f2 + 2 * f3 + f4)
                t += Ts
            end
            return x
        end
    end
end

## 

struct SimpleMvNormal{M,S}
    μ::M
    Σ::S
end

SimpleMvNormal(Σ::Union{SMatrix, PDMats.PDMat{<:Any, <:SMatrix}, Diagonal{<:Any, <:SVector}}) = SimpleMvNormal(@SVector(zeros(size(Σ,1))), Σ)
SimpleMvNormal(Σ::AbstractMatrix) = SimpleMvNormal(zeros(size(Σ,1)), Σ)
SimpleMvNormal(Σ::Function) = error("A SimpleMvNormal distribution must be initialized with a covariance matrix, not a function. If this error is a result of calling a Kalman-filter type constructor where the dynamics-noise covariance is provided as a non-matrix type, you must also explicitly provide the distribution `d0` of the initial state, since the default choice of `d0` cannot be created from a function.")


# We define this new function extended_logpdf and overload that for Distributions.jl in the extension
extended_logpdf(d::SimpleMvNormal, x) = mvnormal_c0(d) - PDMats.invquad(d.Σ, x .- d.μ)/2
const log2π = log(2π)
function mvnormal_c0(d::SimpleMvNormal)
    ldcd = logdet(d.Σ)
    return - (length(d) * oftype(ldcd, log2π) + ldcd) / 2
end


Base.rand(rng::AbstractRNG, d::SimpleMvNormal) = d.μ + cholesky(d.Σ).L*randn(rng, length(d.μ))

Base.rand(rng::AbstractRNG, d::SimpleMvNormal{<:SVector{N}}) where N = d.μ + cholesky(d.Σ).L*@SVector(randn(rng, N))

function Random.rand!(rng::AbstractRNG, d::SimpleMvNormal, out)
    randn!(rng, out)
    mul!(out, cholesky(d.Σ).L, out) # This might work due to L being lower triangular
    out .+= d.μ
end
Base.length(d::SimpleMvNormal) = length(d.μ)
Base.eltype(d::SimpleMvNormal) = eltype(d.μ)

Statistics.mean(d::SimpleMvNormal) = d.μ
Statistics.cov(d::SimpleMvNormal) = d.Σ