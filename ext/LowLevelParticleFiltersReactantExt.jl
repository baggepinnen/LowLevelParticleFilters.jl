"""
Device functions of `ReactantParticleFilter`: the propagation of the particles, the log-likelihood of a measurement
and the first-stage weights of the auxiliary particle filter. The per-particle functions of the model are traced for
a single particle and batched over the columns of the particle matrix (nx × N) with `mapslices`. Every device
function is compiled at its first call for each combination of argument types and sizes and stored in the cache of
the filter.
"""
module LowLevelParticleFiltersReactantExt

using LowLevelParticleFilters
using LowLevelParticleFilters: ReactantParticleFilter, DevicePFstate, ParticleFilter, state, rng, num_particles,
    dynamics, measurement, initial_density, dynamics_density, measurement_density, device_eltype,
    ismissing_measurement
import LowLevelParticleFilters: make_device_state, propagate_particles!, measurement_equation!, first_stage_weights!,
    aux_propagate!, reset!
using Reactant: Reactant, Ops, TracedRNumber, TracedRArray, ConcreteRNumber
using StaticArrays, LinearAlgebra, Statistics

# ------------------------------------------------------------------------------------------------------------
# Conversions for traced numbers that Reactant does not provide
# ------------------------------------------------------------------------------------------------------------
"Logical right shift (Base.>> has no method for traced unsigned integers)"
@inline shr(x::TracedRNumber{UInt64}, n) = Ops.shift_right_logical(x, Reactant.promote_to(TracedRNumber{UInt64}, UInt64(n)))
@inline shr(x::UInt64, n) = x >> n
"Conversion of a traced number to the floating-point type `T`"
@inline tofloat(::Type{T}, x::TracedRNumber{S}) where {T, S} = S === T ? x : Ops.convert(TracedRNumber{T}, x)
@inline tofloat(::Type{T}, x::Real) where {T} = T(x)
"""
Conversion of the element type of an array (Reactant promotes Float32 to Float64 in operations with `Irrational`s
such as `2π`)
"""
convert_array(::Type{T}, x::TracedRArray{S, N}) where {T, S, N} = S === T ? x : Ops.convert(TracedRArray{T, N}, x)

# ------------------------------------------------------------------------------------------------------------
# Counter-based random numbers (SplitMix64 output function over a Weyl sequence, Box-Muller)
# ------------------------------------------------------------------------------------------------------------
@inline function mix64(z)
    z = (z ⊻ shr(z, 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ shr(z, 27)) * 0x94d049bb133111eb
    z ⊻ shr(z, 31)
end
"Random 64-bit word number `ctr` of the stream `key`"
@inline draw64(key, ctr) = mix64(key + (ctr + one(ctr)) * 0x9e3779b97f4a7c15)
"Uniform number in (0, 1] (`open0 = true`) or [0, 1) of the precision `T` from the word `x`"
@inline u01(::Type{Float64}, x, open0) = tofloat(Float64, shr(x, 11) + ifelse(open0, one(x), zero(x))) * 1.1102230246251565e-16
@inline u01(::Type{Float32}, x, open0) = tofloat(Float32, shr(x, 40) + ifelse(open0, one(x), zero(x))) * 5.9604645f-8
"Pair of standard normal numbers (Box-Muller) from the words `ctr2` and `ctr2 + 1` of the stream `key`"
@inline function normal_pair(::Type{T}, key, ctr2) where {T}
    u1 = u01(T, draw64(key, ctr2), true)
    u2 = u01(T, draw64(key, ctr2 + one(ctr2)), false)
    r = sqrt(-2 * log(u1))
    θ = T(2π) * u2
    (r * cos(θ), r * sin(θ))
end
@inline normal_cos(::Type{T}, key, ctr2) where {T} = normal_pair(T, key, ctr2)[1]
@inline normal_sin(::Type{T}, key, ctr2) where {T} = normal_pair(T, key, ctr2)[2]

"""
Standard normal matrix (n × N) of the stream `key`: the entry (s, i) uses the words 2 (np (i − 1) + ⌈s/2⌉ − 1) and
the following one, with np = ⌈n/2⌉ pairs per particle. `iv` is the row (1, N) of 0:N−1.
"""
function normals(::Type{T}, key, iv, n) where {T}
    np = cld(n, 2)
    rows = ntuple(n) do s
        ctr = 2 .* (iv .* UInt64(np) .+ UInt64((s + 1) ÷ 2 - 1))
        isodd(s) ? normal_cos.(T, key, ctr) : normal_sin.(T, key, ctr)
    end
    n == 1 ? rows[1] : vcat(rows...)
end

# ------------------------------------------------------------------------------------------------------------
# Batched evaluation of per-particle functions
# ------------------------------------------------------------------------------------------------------------
"Traced number of the type `T`"
tracedscalar(::Type{T}, s::TracedRNumber) where {T} = tofloat(T, s)
tracedscalar(::Type{T}, s::Number) where {T} = Reactant.promote_to(TracedRNumber{T}, T(s))
"Traced vector of the type `T` of the output of a per-particle function (an `SVector` of traced numbers or a number)"
tovec(::Type{T}, v::TracedRArray) where {T} = convert_array(T, vec(v))
tovec(::Type{T}, v::Number) where {T} = Reactant.TracedUtils.broadcast_to_size(tracedscalar(T, v), (1,))
tovec(::Type{T}, v::AbstractVector) where {T} = length(v) == 1 ? tovec(T, v[1]) : vcat(map(s -> tovec(T, s), Tuple(v))...)

"""
    batched(f, T, X)

The per-particle function `f(x::SVector) -> vector or number` evaluated for every column of the traced matrix
`X` (nx × N), as a matrix (n × N) of the type `T`
"""
function batched(f, ::Type{T}, X) where {T}
    nx = Val(size(X, 1))
    mapslices(X; dims = 1) do x
        Reactant.@allowscalar tovec(T, f(SVector(ntuple(i -> x[i], nx))))
    end
end

# ------------------------------------------------------------------------------------------------------------
# Device functions
# ------------------------------------------------------------------------------------------------------------
"""
    device_propagate(f, Val(gather), Val(noise), X, j, u, p, t, key, iv, μ, L)

Propagated particles f(X[:, j]) + μ + L Z with the standard normal matrix Z of the stream `key`; `f = nothing` for
X[:, j] + μ + L Z (the noise-free predictions are already in `X`). `gather = false` for the particles in their own
order, `noise = false` for f(X).
"""
function device_propagate(f, ::Val{gather}, ::Val{noise}, X, j, u, p, t, key, iv, μ, L) where {gather, noise}
    T = Reactant.unwrapped_eltype(X)
    Xg = gather ? X[:, j] : X
    Xn = f === nothing ? Xg : batched(x -> f(x, u, p, t), T, Xg)
    noise || return Xn
    Z = normals(T, key, iv, size(L, 2))
    Xn .+ μ .+ L * Z
end

"""
Log-likelihood c − ½‖L⁻¹ (y − g(x) − μ)‖² of a measurement with additive Gaussian noise, for every particle (1 × N).
(A `vec` of the result would be returned as a lazy wrapper, whose transfer to the host is slow.)
"""
function device_gaussian_loglik(g, X, u, y, p, t, μ, Linv, c)
    T = Reactant.unwrapped_eltype(X)
    E = y .- batched(x -> g(x, u, p, t), T, X) .- μ
    Z = Linv * E
    c .- sum(abs2, Z; dims = 1) ./ 2
end

"Per-particle log-likelihood `g(x, u, y, p, t)` (a measurement likelihood or first-stage weights) for every particle (1 × N)"
function device_loglik(g, X, u, y, p, t)
    T = Reactant.unwrapped_eltype(X)
    batched(x -> g(x, u, y, p, t), T, X)
end

# ------------------------------------------------------------------------------------------------------------
# Host side
# ------------------------------------------------------------------------------------------------------------
"Key of the argument types and sizes of a compiled device function"
argkey(a::AbstractArray) = (typeof(a), size(a))
argkey(a::Union{Tuple, NamedTuple}) = map(argkey, a)
argkey(a) = typeof(a)

"""
    device_call(pf, name, f, args...)

`f(args...)` with the compiled version of `f` from the cache of `pf`, compiled at the first call for the name `name`
and the argument types and sizes of `args`
"""
function device_call(pf::ReactantParticleFilter, name, f, args...)
    thunk = get!(pf.cache, (name, map(argkey, args))) do
        Reactant.compile(f, args)
    end
    thunk(args...)
end

"""
Parameters with floating-point numbers and arrays in the precision `T` of the particles, so that the particles keep
their element type through the user functions
"""
convert_params(::Type{T}, p::AbstractArray{<:AbstractFloat}) where {T} = T.(p)
convert_params(::Type{T}, p::AbstractFloat) where {T} = T(p)
convert_params(::Type{T}, p::Union{Tuple, NamedTuple}) where {T} = map(x -> convert_params(T, x), p)
convert_params(::Type{T}, p) where {T} = p

"Input, measurement or time as a device value of the type `T`"
todevice(::Type{T}, x::AbstractArray{<:Real}) where {T} = Reactant.to_rarray(T.(collect(x)))
todevice(::Type{T}, x::Real) where {T} = ConcreteRNumber(T(x))
todevice(::Type{T}, ::Nothing) where {T} = nothing

"The parameters `p` on the device, uploaded again only when they differ from those of the previous call"
function device_params(pf::ReactantParticleFilter, p)
    cached = get(pf.cache, :params, nothing)
    cached !== nothing && isequal(cached[1], p) && return cached[2]
    pd = Reactant.to_rarray(convert_params(device_eltype(pf), p))
    pf.cache[:params] = (deepcopy(p), pd)
    pd
end

"Row (1, N) of 0:N−1 on the device, the particle counters of the random numbers"
device_iv(pf::ReactantParticleFilter) =
    get!(() -> Reactant.to_rarray(reshape(UInt64.(0:num_particles(pf)-1), 1, :)), pf.cache, :iv)

"Mean and Cholesky factor of a Gaussian density (`nothing` for no noise)"
function gaussian_factors(::Type{T}, d) where {T}
    d === nothing && return nothing
    μ = T.(collect(mean(d)))
    L = T.(Matrix(cholesky(Symmetric(Matrix(cov(d)))).L))
    (; μ, L)
end

"Mean and Cholesky factor of the dynamics noise on the device"
function device_dynamics_noise(pf::ReactantParticleFilter)
    get!(pf.cache, :dynamics_noise) do
        fac = gaussian_factors(device_eltype(pf), dynamics_density(pf))
        fac === nothing ? nothing : (; μ = Reactant.to_rarray(fac.μ), L = Reactant.to_rarray(fac.L))
    end
end

"Mean, inverse Cholesky factor and normalization constant of a Gaussian measurement density on the device"
function device_measurement_noise(pf::ReactantParticleFilter)
    get!(pf.cache, :measurement_noise) do
        fac = gaussian_factors(Float64, measurement_density(pf))
        T = device_eltype(pf)
        ny = length(fac.μ)
        c = -ny / 2 * log(2π) - sum(log, diag(fac.L))
        (; μ = Reactant.to_rarray(T.(fac.μ)), Linv = Reactant.to_rarray(T.(Matrix(inv(LowerTriangular(fac.L))))),
           c = ConcreteRNumber(T(c)))
    end
end

function make_device_state(pf::ParticleFilter, ::Type{T}) where {T}
    s = pf.state
    X = T.(reduce(hcat, s.x))
    DevicePFstate(Reactant.to_rarray(X), Reactant.to_rarray(copy(X)), copy(s.w), copy(s.we), Ref(s.maxw[]),
        copy(s.j), copy(s.bins), Ref(s.t[]))
end

function reset!(pf::ReactantParticleFilter)
    s = state(pf)
    N = num_particles(pf)
    X = device_eltype(pf).(reduce(hcat, [rand(rng(pf), initial_density(pf)) for _ in 1:N]))
    s.xprev = Reactant.to_rarray(X)
    s.x = Reactant.to_rarray(copy(X))
    fill!(s.w, -log(N))
    fill!(s.we, 1/N)
    s.maxw[] = 0
    s.t[] = 1
    pf
end

"""
    propagate(pf, f, gather, noise, X, j, u, p, t)

Propagation on the device, see `device_propagate`. The result replaces the particles `x` of the state.
"""
function propagate(pf::ReactantParticleFilter, f, gather::Bool, noise::Bool, X, j, u, p, t)
    T = device_eltype(pf)
    dn = device_dynamics_noise(pf)
    noise &= dn !== nothing
    jd = gather ? Reactant.to_rarray(j) : nothing
    key = ConcreteRNumber(rand(rng(pf), UInt64))
    μ, L = dn === nothing ? (nothing, nothing) : (dn.μ, dn.L)
    name = (:propagate, f === nothing, gather, noise)
    fd = (X, j, u, p, t, key, iv, μ, L) -> device_propagate(f, Val(gather), Val(noise), X, j, u, p, t, key, iv, μ, L)
    state(pf).x = device_call(pf, name, fd, X, jd, todevice(T, u), device_params(pf, p), todevice(T, t), key,
        device_iv(pf), μ, L)
end

propagate_particles!(pf::ReactantParticleFilter, u, j::Vector{Int}, p, t::Real) =
    propagate(pf, dynamics(pf), true, true, state(pf).xprev, j, u, p, t)
propagate_particles!(pf::ReactantParticleFilter, u, p, t::Real) =
    propagate(pf, dynamics(pf), false, true, state(pf).xprev, nothing, u, p, t)
propagate_particles!(pf::ReactantParticleFilter, u, p, t::Real, ::Nothing) =
    propagate(pf, dynamics(pf), false, false, state(pf).xprev, nothing, u, p, t)
# The noise-free predictions are in x: the new particles are the predictions of the ancestors plus the noise
aux_propagate!(pf::ReactantParticleFilter, u, j, p, t) =
    propagate(pf, nothing, true, true, state(pf).x, j, u, p, t)

function measurement_equation!(pf::ReactantParticleFilter, u, y, p, t, w = weights(pf); g = pf.measurement_likelihood)
    (y === missing || ismissing_measurement(y)) && return w
    T = device_eltype(pf)
    X = state(pf).x
    args = (todevice(T, u), todevice(T, y), device_params(pf, p), todevice(T, t))
    lw = if g === nothing
        mn = device_measurement_noise(pf)
        gm = measurement(pf)
        device_call(pf, :gaussian_loglik, (X, u, y, p, t, μ, Linv, c) -> device_gaussian_loglik(gm, X, u, y, p, t, μ, Linv, c),
            X, args..., mn.μ, mn.Linv, mn.c)
    else
        device_call(pf, (:loglik, g), (X, u, y, p, t) -> device_loglik(g, X, u, y, p, t), X, args...)
    end
    w .+= vec(Array(lw))
end

function first_stage_weights!(pf::ReactantParticleFilter, λ, g1, u, y1, p, t)
    T = device_eltype(pf)
    X = state(pf).x
    lw = device_call(pf, (:loglik, g1), (X, u, y, p, t) -> device_loglik(g1, X, u, y, p, t),
        X, todevice(T, u), todevice(T, y1), device_params(pf, p), todevice(T, t))
    λ .= vec(Array(lw))
end

end # module
