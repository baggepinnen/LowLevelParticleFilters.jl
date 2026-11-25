"""
Device functions of `ReactantParticleFilter`. The per-particle functions of the model are traced for a single particle
and batched over the columns of the particle matrix (nx × N). A time step consists of two compiled device functions:
the correction (measurement log-likelihood, normalization of the weights, log-likelihood of the measurement), and the
prediction (effective sample size, systematic resampling, propagation), for the bootstrap filter or the auxiliary
particle filter. The compiled functions are stored in a global cache, by the model functions and the argument types
and sizes.
"""
module LowLevelParticleFiltersReactantExt

using LowLevelParticleFilters
using LowLevelParticleFilters: ReactantParticleFilter, DevicePFstate, ParticleFilter, AdvancedParticleFilter,
    ParticleSmoothingMethod, FFBSi, GenealogySmoother, ParticleGibbsSmoother,
    AuxiliaryParticleFilter, state, rng, num_particles, dynamics, measurement, initial_density, dynamics_density,
    measurement_density, resample_threshold, device_eltype, ismissing_measurement, parameters, index
import LowLevelParticleFilters: make_device_state, propagate_particles!, measurement_equation!, reset!, predict!,
    correct!, logsumexp!, reset_weights!, randn_noise, smooth, loglik, batched_loglik
using Reactant: Reactant, Ops, TracedRNumber, TracedRArray, ConcreteRNumber, @trace
using StaticArrays, LinearAlgebra, Statistics

"""
Compile options: the default loop-raising passes may hoist quantities that depend only on the counters of the random
numbers out of loops as large arrays
"""
const COMPILE_OPTIONS = Reactant.CompileOptions(; disable_loop_raising_passes = true)

# ------------------------------------------------------------------------------------------------------------
# Conversions for traced numbers that Reactant does not provide
# ------------------------------------------------------------------------------------------------------------
"Materialization of a wrapped traced array (reshape, view), which `scatter` and indexing reject"
mat(x) = x isa TracedRArray ? x : Reactant.TracedUtils.materialize_traced_array(x)
"Logical right shift (Base.>> has no method for traced unsigned integers)"
@inline shr(x::TracedRNumber{UInt64}, n) = Ops.shift_right_logical(x, Reactant.promote_to(TracedRNumber{UInt64}, UInt64(n)))
@inline shr(x::UInt64, n) = x >> n
"Conversion of a traced number to the type `T`"
@inline tofloat(::Type{T}, x::TracedRNumber{S}) where {T, S} = S === T ? x : Ops.convert(TracedRNumber{T}, x)
@inline tofloat(::Type{T}, x::Real) where {T} = T(x)
"""
Conversion of the element type of an array (Reactant promotes Float32 to Float64 in operations with `Irrational`s
such as `2π`)
"""
convert_array(::Type{T}, x::TracedRArray{S, N}) where {T, S, N} = S === T ? x : Ops.convert(TracedRArray{T, N}, x)
convert_array(::Type{T}, x) where {T} = convert_array(T, mat(x))

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

const SALT_PROP = 0x0000000000000000
const SALT_RESAMPLE = 0x5851f42d4c957f2d
const SALT_SMOOTH = 0x14057b7ef767814f
"Key of the stream `salt` of the time step `k` for the seed `seed`"
step_key(seed, k, salt) = mix64(seed ⊻ mix64(k * 0x9e3779b97f4a7c15 + salt))

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

"""
Source of standard normal numbers of one particle in the traced dynamics of an `AdvancedParticleFilter`, passed as
the argument `noise`: the draw number `c` of the particle uses the words 2c and 2c + 1 of the stream `key`. The counter
advances while the function is traced, so that every call of `randn_noise` has its own numbers.
"""
struct DeviceNoise{T, K}
    key::K
    counter::Base.RefValue{Int}
end
DeviceNoise{T}(key) where {T} = DeviceNoise{T, typeof(key)}(key, Ref(0))

function next_normal(noise::DeviceNoise{T}) where {T}
    c = noise.counter[]
    noise.counter[] += 1
    normal_cos(T, noise.key, UInt64(2c))
end
randn_noise(noise::DeviceNoise) = next_normal(noise)
randn_noise(noise::DeviceNoise, ::Type{<:AbstractFloat}) = next_normal(noise)
randn_noise(noise::DeviceNoise, ::Type{S}) where {S <: StaticVector} = SVector(ntuple(_ -> next_normal(noise), Val(length(S))))

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

"`SVector` of the entries of a traced vector of static length `n`"
svec(v, ::Val{n}) where {n} = Reactant.@allowscalar SVector(ntuple(i -> v[i], Val(n)))

"""
    batched(f, T, X, keys = nothing)

The per-particle function `f(x::SVector) -> vector or number` evaluated for every column of the traced matrix `X`
(nx × N), as a matrix (n × N) of the type `T`. With the per-particle keys `keys` (N), `f(x, key)`.

`f` is traced by native execution with traced numbers rather than by Reactant's interpreter: the interpreter replaces
the arithmetic of `SVector`s of numbers with traced arrays, so that expressions that mix constant `SVector`s or
matrices with `SVector`s of traced numbers fail (`A*x + B*(σ*w)`), whereas the methods of StaticArrays operate on the
traced numbers directly.
"""
function batched(f, ::Type{T}, X, keys = nothing) where {T}
    nx = Val(size(X, 1))
    Xt = permutedims(X, (2, 1))
    res = if keys === nothing
        Ops.batch([Xt], [size(X, 2)]) do x
            Reactant.@allowscalar tovec(T, Reactant.call_with_native(f, svec(x, nx)))
        end
    else
        Ops.batch([Xt, keys], [size(X, 2)]) do x, k
            Reactant.@allowscalar tovec(T, Reactant.call_with_native(f, svec(x, nx), k[]))
        end
    end
    permutedims(only(res), (2, 1))
end

# ------------------------------------------------------------------------------------------------------------
# Log-sum-exp and systematic resampling
# ------------------------------------------------------------------------------------------------------------
"Log of the sum of the exponentials over the particles: (1, N) → (1, 1)"
function lse_rows(a)
    mx = maximum(a; dims = 2)
    mx .+ log.(sum(exp.(a .- mx); dims = 2))
end

"""
Cumulative sum of the weights p (B, N) along the particles: Float32 within blocks of `MIXED_BLOCK` particles and
Float64 for the offsets of the blocks. A Float64 cumulative sum is slow on consumer GPUs (1/64 of the Float32 rate);
in Float32 alone the position N c_i / c_N has an error of the order of 1e-7 N slots.
"""
const MIXED_BLOCK = 1024
function cumsum_weights(p)
    B, N = size(p)
    N <= MIXED_BLOCK && return cumsum(convert_array(Float64, p); dims = 2)
    nb = cld(N, MIXED_BLOCK)
    Np = nb * MIXED_BLOCK
    pp = Np == N ? p : hcat(p, Ops.fill(zero(Reactant.unwrapped_eltype(p)), (B, Np - N)))
    loc = cumsum(reshape(pp, B, MIXED_BLOCK, nb); dims = 2)
    tot = convert_array(Float64, loc[:, MIXED_BLOCK:MIXED_BLOCK, :])
    off = cumsum(tot; dims = 3) .- tot
    c = reshape(off .+ convert_array(Float64, loc), B, Np)
    Np == N ? c : c[:, 1:N]
end

"""
    systematic(a, u, consts)

Ancestor indices (1, N) of systematic resampling of the normalized log-weights `a` (1, N) with the uniform number `u`.
The ancestor of slot n (0-based) is j_n = 1 + #{i < N : N c_i/c_N < u + n}. With the first slot f_i = ⌊N c_i/c_N − u⌋ + 1
after particle i, nondecreasing in i, the count is max{i < N : f_i ≤ n}: the last particle of each value of f_i is
written to the slot f_i (scatter with the combiner `max`), followed by a cumulative maximum over the slots. A histogram
by scatter-add suffered from atomic contention, and a plain store with unique indices raced (the cumulative sum of XLA
is not exactly monotone in floating point).
"""
function systematic(a, u, consts)
    B, N = size(a)
    c = cumsum_weights(exp.(a))
    s = c .* (Float64(N) ./ c[:, N:N])
    f = convert_array(Int64, clamp.(floor.(s .- u) .+ 1, 0.0, Float64(N)))
    f = ifelse.(consts.lastslot, N, f)
    fnext = hcat(f[:, 2:N], Ops.fill(Int64(N + 1), (B, 1)))
    target = ifelse.(f .!= fnext, f, N) # slot N is out of bounds
    lin = mat(reshape(1 .+ target, B * N, 1))
    hist = Ops.fill(Int32(0), (B * N,))
    hist = Ops.scatter(max, [hist], lin, [mat(reshape(consts.iota, B * N))];
        update_window_dims = Int64[], inserted_window_dims = [1], input_batching_dims = Int64[],
        scatter_indices_batching_dims = Int64[], scatter_dims_to_operand_dims = [1], index_vector_dim = 2)[1]
    Int32(1) .+ accumulate(max, reshape(hist, B, N); dims = 2)
end

"Columns of `X` at the ancestors `j` (1, N)"
gather(X, j) = X[:, mat(vec(convert_array(Int64, j)))]

# ------------------------------------------------------------------------------------------------------------
# Device functions
# ------------------------------------------------------------------------------------------------------------
"""
Static settings of a compiled device function as a type parameter (numbers captured by a traced function would be
traced): `nu`, `ny` lengths of the input and the measurement in the packed host data, `kind` (:pf or :advanced),
`gaussian` (Gaussian measurement density), `th` resampling threshold
"""
struct Static{C} end
cval(::Static{C}) where {C} = C

"""
    unpack(T, v, cfg)

Seed, time-step counter, time, input and measurement from the packed host data `v` (Float64):
[seed ÷ 2³², seed mod 2³², k, t, u..., y...]
"""
function unpack(::Type{T}, v, cfg) where {T}
    nu, ny = cval(cfg).nu, cval(cfg).ny
    Reactant.@allowscalar begin
        seed = Ops.convert(TracedRNumber{UInt64}, v[1]) * 0x0000000100000000 + Ops.convert(TracedRNumber{UInt64}, v[2])
        k = Ops.convert(TracedRNumber{UInt64}, v[3])
        t = tofloat(T, v[4])
        u = SVector(ntuple(i -> tofloat(T, v[4+i]), Val(nu)))
        y = SVector(ntuple(i -> tofloat(T, v[4+nu+i]), Val(ny)))
    end
    (; seed, k, t, u, y)
end

"Log-likelihood (1 × N) of the measurement `y` for the particles `X`, with `fns.loglik` or the Gaussian density"
function meas_loglik(fns, cfg, X, inp, y, p, consts)
    T = Reactant.unwrapped_eltype(X)
    u, t = inp.u, inp.t
    if cval(cfg).gaussian
        g = fns.measurement
        mn = consts.meas
        E = reshape(tovec(T, y), :, 1) .- batched(x -> g(x, u, p, t), T, X) .- mn.μ
        Z = mn.Linv * E
        mn.c .- sum(abs2, Z; dims = 1) ./ 2
    else
        g = fns.loglik
        batched(x -> g(x, u, y, p, t), T, X)
    end
end

"Noise-free prediction of the particles `X`"
function predict_mean(fns, cfg, X, inp, p)
    T = Reactant.unwrapped_eltype(X)
    f = fns.dynamics
    u, t = inp.u, inp.t
    cval(cfg).kind === :pf ? batched(x -> f(x, u, p, t), T, X) : batched(x -> f(x, u, p, t, false), T, X)
end

"""
Propagation of the particles `X` with noise from the stream `key`: f(x) + μ + L z for a `ParticleFilter`, and
f(x, u, p, t, noise) with a `DeviceNoise` per particle for an `AdvancedParticleFilter`. `Xmean`: the noise-free
predictions f(x) for a `ParticleFilter` if already computed.
"""
function propagate_noisy(fns, cfg, X, inp, p, consts, key; Xmean = nothing)
    T = Reactant.unwrapped_eltype(Xmean === nothing ? X : Xmean)
    u, t = inp.u, inp.t
    if cval(cfg).kind === :pf
        Xm = Xmean === nothing ? predict_mean(fns, cfg, X, inp, p) : Xmean
        dn = consts.dyn
        dn === nothing && return Xm
        Xm .+ dn.μ .+ dn.L * normals(T, key, consts.iv, size(dn.L, 2))
    else
        f = fns.dynamics
        keys = mix64.(vec(consts.iv) .⊻ key)
        batched((x, k) -> f(x, u, p, t, DeviceNoise{T}(k)), T, X, keys)
    end
end

"Correction: log-weights plus the measurement log-likelihood, normalized; returns the new weights and log Σ"
function device_correct(fns, cfg, X, w, v, p, consts)
    T = Reactant.unwrapped_eltype(X)
    inp = unpack(T, v, cfg)
    a = w .+ convert_array(T, meas_loglik(fns, cfg, X, inp, inp.y, p, consts))
    l = lse_rows(a)
    a .- l, l
end

"Normalization of the log-weights; returns the new weights and log Σ"
function device_normalize(w)
    l = lse_rows(w)
    w .- l, l
end

"Measurement log-likelihood added to the log-weights, without normalization"
function device_add_loglik(fns, cfg, X, w, v, p, consts)
    T = Reactant.unwrapped_eltype(X)
    inp = unpack(T, v, cfg)
    w .+ convert_array(T, meas_loglik(fns, cfg, X, inp, inp.y, p, consts))
end

"""
Prediction of the bootstrap filter: resampling when the effective sample size is below `th N`, propagation with noise.
Returns the new particles, log-weights and ancestor indices (1 × N).
"""
function device_predict(fns, cfg, X, w, v, p, consts)
    T = Reactant.unwrapped_eltype(X)
    N = size(X, 2)
    inp = unpack(T, v, cfg)
    th = cval(cfg).th
    uu = u01(Float64, draw64(step_key(inp.seed, inp.k, SALT_RESAMPLE), zero(inp.k)), false)
    if th >= 1
        j = systematic(w, uu, consts)
        wn = Ops.fill(T(-log(N)), (1, N))
    else
        ess = 1 ./ sum(exp.(2 .* w); dims = 2)
        res = ess .< T(th * N)
        j = ifelse.(res, systematic(w, uu, consts), consts.iota)
        wn = ifelse.(res, T(-log(N)), w)
    end
    Xn = propagate_noisy(fns, cfg, gather(X, j), inp, p, consts, step_key(inp.seed, inp.k, SALT_PROP))
    Xn, wn, j
end

"Propagation without resampling (`j = nothing`) or from the ancestors `j` (N), with noise"
function device_propagate(fns, cfg, X, j, v, p, consts)
    T = Reactant.unwrapped_eltype(X)
    inp = unpack(T, v, cfg)
    Xg = j === nothing ? X : X[:, j]
    propagate_noisy(fns, cfg, Xg, inp, p, consts, step_key(inp.seed, inp.k, SALT_PROP))
end

"""
Prediction of the auxiliary particle filter with the next measurement y1: first-stage weights λ of the noise-free
predictions, resampling with the weights w λ, propagation from the ancestors with noise, log-weights
log Σ wλ − λ_j − log N (see `predict!(::AuxiliaryParticleFilter, ...)`). Returns the new particles, log-weights and
ancestor indices (1 × N).
"""
function device_apf_predict(fns, cfg, X, w, v, p, consts)
    T = Reactant.unwrapped_eltype(X)
    N = size(X, 2)
    inp = unpack(T, v, cfg)
    Xm = predict_mean(fns, cfg, X, inp, p)
    λ = if fns.first_stage === nothing
        convert_array(T, meas_loglik(fns, cfg, Xm, inp, inp.y, p, consts))
    else
        g1 = fns.first_stage
        u, y, t = inp.u, inp.y, inp.t
        batched(x -> g1(x, u, y, p, t), T, Xm)
    end
    a = w .+ λ
    l1 = lse_rows(a)
    uu = u01(Float64, draw64(step_key(inp.seed, inp.k, SALT_RESAMPLE), zero(inp.k)), false)
    j = systematic(a .- l1, uu, consts)
    key = step_key(inp.seed, inp.k, SALT_PROP)
    Xn = if cval(cfg).kind === :pf
        propagate_noisy(fns, cfg, nothing, inp, p, consts, key; Xmean = gather(Xm, j))
    else
        propagate_noisy(fns, cfg, gather(X, j), inp, p, consts, key)
    end
    Xn, l1 .- gather(λ, j) .- T(log(N)), j
end

# ------------------------------------------------------------------------------------------------------------
# Log-likelihood of a whole trajectory in one device program
# ------------------------------------------------------------------------------------------------------------
"""
    device_loglik_loop(fns, cfg, Val(apf), X, w, Vc, Vp, hasy, hasy1, p, consts)

The time steps of `loglik` in one traced loop: correction with the packed data `Vc[:, k]` if step `k` has a measurement
(`hasy[k]`), followed by the prediction with `Vp[:, k]`; for the auxiliary particle filter (`apf`), the prediction with
the next measurement in `Vp[:, k]` if it is present (`hasy1[k]`). Returns the final particles, log-weights and the
log-likelihood (1 × 1).
"""
function device_loglik_loop(fns, cfg, apf::Val, X, w, Vc, Vp, hasy, hasy1, p, consts)
    ll = Ops.fill(0.0, (1, 1))
    # Numbers captured by the body, such as the values captured by closures among the model functions, are compiled as
    # constants as in the device functions of single time steps (`track_numbers = false`); with the default, they are
    # traced outside of the batched per-particle functions, which XLA rejects
    @trace track_numbers = false for k in 1:size(Vc, 2)
        h = Reactant.@allowscalar hasy[k]
        wc, l = device_correct(fns, cfg, X, w, Vc[:, k], p, consts)
        w = ifelse.(h, wc, w)
        ll = ll .+ ifelse.(h, convert_array(Float64, l), 0.0)
        X, w = loop_predict(apf, fns, cfg, X, w, Vp[:, k], Reactant.@allowscalar(hasy1[k]), p, consts)
    end
    X, w, ll
end

"Prediction in the traced loop of `loglik`: of the bootstrap filter, or of the auxiliary filter if the next measurement is present (`h1`)"
loop_predict(::Val{false}, fns, cfg, X, w, vp, h1, p, consts) = device_predict(fns, cfg, X, w, vp, p, consts)[1:2]
function loop_predict(::Val{true}, fns, cfg, X, w, vp, h1, p, consts)
    @trace track_numbers = false if h1
        Xn, wn = device_apf_predict(fns, cfg, X, w, vp, p, consts)[1:2]
    else
        Xn, wn = device_predict(fns, cfg, X, w, vp, p, consts)[1:2]
    end
    Xn, wn
end

# ------------------------------------------------------------------------------------------------------------
# Particle smoother (forward filtering, backward simulation)
# ------------------------------------------------------------------------------------------------------------
"""
    draw_rows(lw, uu)

Categorical draws (M × 1), one per row of the unnormalized log-weights `lw` (M × N), by inversion of the cumulative
weights with the uniform numbers `uu` (M × 1)
"""
function draw_rows(lw, uu)
    N = size(lw, 2)
    c = cumsum_weights(exp.(lw .- maximum(lw; dims = 2)))
    count = sum(convert_array(Int64, c .< uu .* c[:, N:N]); dims = 2)
    min.(count .+ 1, N)
end

"Uniform numbers (M × 1) of the stream `key`, `im` is 0:M−1"
uniform_rows(key, im) = reshape(u01.(Float64, draw64.(key, im), false), :, 1)

"Initial smoothed particles (nx × M): draws from the filter particles `Xf` with the log-weights `wf` (1 × N)"
function device_smooth_init(Xf, wf, im, v, cfg)
    T = Reactant.unwrapped_eltype(Xf)
    inp = unpack(T, v, cfg)
    lw = wf .+ Ops.fill(zero(T), (length(im), 1))
    gather(Xf, draw_rows(lw, uniform_rows(step_key(inp.seed, inp.k, SALT_SMOOTH), im)))
end

"""
Step t + 1 → t of the backward simulation: the smoothed particle m at t is the filter particle n at t drawn with the
probabilities ∝ w_n p(xb_m | x_n), with the Gaussian transition density p(x′ | x) = N(x′; f(x) + μ, L Lᵀ). The quadratic
forms ‖L⁻¹(xb_m − f(x_n) − μ)‖² are computed from the differences (M × N) rather than by the expansion
‖a‖² + ‖b‖² − 2 aᵀb, which loses accuracy in Float32 when the states are large compared with the noise.
"""
function device_smooth_step(fns, cfg, Xf, wf, xb, im, v, p, consts)
    T = Reactant.unwrapped_eltype(Xf)
    inp = unpack(T, v, cfg)
    dn = consts.dyn
    G = dn.Linv * (predict_mean(fns, cfg, Xf, inp, p) .+ dn.μ) # (nx, N)
    Z = dn.Linv * xb                                           # (nx, M)
    nx, N = size(G)
    M = size(Z, 2)
    q = reshape(sum(abs2, reshape(Z, nx, M, 1) .- reshape(G, nx, 1, N); dims = 1), M, N)
    lw = wf .- q ./ 2
    gather(Xf, draw_rows(lw, uniform_rows(step_key(inp.seed, inp.k, SALT_SMOOTH), im)))
end

# ------------------------------------------------------------------------------------------------------------
# Compilation and the global cache of compiled functions
# ------------------------------------------------------------------------------------------------------------
"Key of the argument types and sizes of a compiled device function"
argkey(a::AbstractArray) = (typeof(a), size(a))
argkey(a::Union{Tuple, NamedTuple}) = map(argkey, a)
argkey(a) = typeof(a)

const CACHE = Dict{Any, Any}()
const CACHE_LOCK = ReentrantLock()

"""
    device_call(name, fns, cfg, f, args...)

`f(args...)` with the compiled version of `f`, compiled at the first call for the name `name`, the model functions
`fns`, the settings `cfg` and the argument types and sizes of `args`
"""
function device_call(name, fns, cfg, f, args...)
    key = (name, fns, cfg, map(argkey, args))
    thunk = lock(CACHE_LOCK) do
        get!(CACHE, key) do
            check_no_random(f, args)
            Reactant.compile(f, args; compile_options = COMPILE_OPTIONS)
        end
    end
    thunk(args...)
end

"""
Error if the model functions draw random numbers (`rand`, `randn`): such numbers are drawn once, when the function is
traced, and compiled as constants, so that the same numbers would be used in every time step and for every particle.
They are detected as a difference between two traces of the function, or as a random-number operation of Reactant.
"""
function check_no_random(f, args)
    hlo1 = string(Reactant.@code_hlo optimize = false f(args...))
    hlo2 = string(Reactant.@code_hlo optimize = false f(args...))
    (hlo1 != hlo2 || occursin("rng_bit_generator", hlo1)) && error("The model functions of a ReactantParticleFilter " *
        "must not draw random numbers (`rand`, `randn`), these would be compiled as constants. Draw the noise of the " *
        "dynamics of an AdvancedParticleFilter with `randn_noise(noise, ...)`.")
    nothing
end

# ------------------------------------------------------------------------------------------------------------
# Host side
# ------------------------------------------------------------------------------------------------------------
"""
Parameters with floating-point numbers and arrays in the precision `T` of the particles, so that the particles keep
their element type through the user functions
"""
convert_params(::Type{T}, p::AbstractArray{<:AbstractFloat}) where {T} = T.(p)
convert_params(::Type{T}, p::AbstractFloat) where {T} = T(p)
convert_params(::Type{T}, p::Union{Tuple, NamedTuple}) where {T} = map(x -> convert_params(T, x), p)
convert_params(::Type{T}, p) where {T} = p

"""
The parameters `p` on the device, uploaded again only when they differ from those of the previous call. Floating-point
numbers are traced (without `track_numbers`, Reactant compiles numbers as constants), integers are compiled as
constants.
"""
function device_params(pf::ReactantParticleFilter, p)
    cached = get(pf.device_data, :params, nothing)
    cached !== nothing && isequal(cached[1], p) && return cached[2]
    pd = Reactant.to_rarray(convert_params(device_eltype(pf), p); track_numbers = AbstractFloat)
    pf.device_data[:params] = (deepcopy(p), pd)
    pd
end

veclength(::Nothing) = 0
veclength(x) = length(x)

"""
Packed host data of a device function: [seed ÷ 2³², seed mod 2³², k, t, u..., y...], uploaded as a single array.
By default the seed and the time-step counter `k` of the state of the filter.
"""
function pack(pf::ReactantParticleFilter, t, u, y = nothing; seed = state(pf).seed, k = state(pf).t[])
    v = Float64[seed >> 32, seed & 0xffffffff, k, t]
    u === nothing || append!(v, u)
    y === nothing || append!(v, y)
    Reactant.to_rarray(v)
end

"Mean and Cholesky factor of a Gaussian density (`nothing` for no noise)"
function gaussian_factors(::Type{T}, d) where {T}
    d === nothing && return nothing
    μ = T.(collect(mean(d)))
    L = T.(Matrix(cholesky(Symmetric(Matrix(cov(d)))).L))
    (; μ, L)
end

"Constant device data of the filter: particle indices and the factors of the Gaussian noise densities"
function device_consts(pf::ReactantParticleFilter)
    get!(pf.device_data, :consts) do
        T = device_eltype(pf)
        N = num_particles(pf)
        inner = pf.pf
        dyn = nothing
        if inner isa ParticleFilter
            fac = gaussian_factors(T, dynamics_density(pf))
            dyn = fac === nothing ? nothing : (; μ = Reactant.to_rarray(fac.μ), L = Reactant.to_rarray(fac.L),
                Linv = Reactant.to_rarray(T.(Matrix(inv(LowerTriangular(Float64.(fac.L)))))))
        end
        meas = nothing
        if pf.measurement_likelihood === nothing
            fac = gaussian_factors(Float64, measurement_density(pf))
            ny = length(fac.μ)
            c = -ny / 2 * log(2π) - sum(log, diag(fac.L))
            meas = (; μ = Reactant.to_rarray(T.(fac.μ)), Linv = Reactant.to_rarray(T.(Matrix(inv(LowerTriangular(fac.L))))),
                c = ConcreteRNumber(T(c)))
        end
        (; iv = Reactant.to_rarray(reshape(UInt64.(0:N-1), 1, N)), iota = Reactant.to_rarray(reshape(Int32.(1:N), 1, N)),
           lastslot = Reactant.to_rarray(reshape((1:N) .== N, 1, N)), dyn, meas)
    end
end

"Model functions of the filter, the part of the key of the compiled functions that identifies the model"
function model_functions(pf::ReactantParticleFilter; loglik = pf.measurement_likelihood, first_stage = nothing)
    (; dynamics = dynamics(pf), measurement = pf.pf isa ParticleFilter ? measurement(pf) : nothing, loglik, first_stage)
end

function config(pf::ReactantParticleFilter, u, y; loglik = pf.measurement_likelihood)
    Static{(; nu = veclength(u), ny = veclength(y), kind = pf.pf isa ParticleFilter ? :pf : :advanced,
        gaussian = loglik === nothing, th = resample_threshold(pf))}()
end

function make_device_state(pf::Union{ParticleFilter, AdvancedParticleFilter}, ::Type{T}) where {T}
    X = T.(reduce(hcat, pf.state.x))
    N = size(X, 2)
    DevicePFstate(Reactant.to_rarray(X), Reactant.to_rarray(X), Reactant.to_rarray(fill(T(-log(N)), 1, N)),
        rand(rng(pf), UInt64), Ref(0), zeros(N), nothing)
end

function reset!(pf::ReactantParticleFilter)
    s = state(pf)
    N = num_particles(pf)
    T = device_eltype(pf)
    X = T.(reduce(hcat, [rand(rng(pf), initial_density(pf)) for _ in 1:N]))
    s.x = Reactant.to_rarray(X)
    s.xprev = s.x
    s.w = Reactant.to_rarray(fill(T(-log(N)), 1, N))
    s.seed = rand(rng(pf), UInt64)
    s.t[] = 1
    s.ancestors = nothing
    pf
end

function reset_weights!(s::DevicePFstate)
    N = num_particles(s)
    s.w = Reactant.to_rarray(fill(eltype(s.w)(-log(N)), 1, N))
    s
end

"""
    ll, e = correct!(pf::ReactantParticleFilter, u, y, p = parameters(pf), t = index(pf)*pf.Ts; g = pf.measurement_likelihood)

Correction with the measurement `y` on the device: the log-weights are incremented by the measurement log-likelihood
and normalized. Returns the log-likelihood of `y` and `0`. `g`: a measurement likelihood `(x, u, y, p, t) -> log p(y | x)`
that replaces the one of the filter.
"""
function correct!(pf::ReactantParticleFilter, u, y, p = parameters(pf), t = index(pf)*pf.Ts; g = pf.measurement_likelihood)
    (y === missing || ismissing_measurement(y)) && return 0.0, 0
    s = state(pf)
    fns = model_functions(pf; loglik = g)
    cfg = config(pf, u, y; loglik = g)
    s.w, l = device_call(:correct, fns, cfg, (X, w, v, p, c) -> device_correct(fns, cfg, X, w, v, p, c),
        s.x, s.w, pack(pf, t, u, y), device_params(pf, p), device_consts(pf))
    Float64(only(Array(l))), 0
end

"""
    predict!(pf::ReactantParticleFilter, u, p = parameters(pf), t = index(pf)*pf.Ts)

Prediction on the device: systematic resampling if the effective sample size is below `resample_threshold N`,
propagation of the particles with noise
"""
function predict!(pf::ReactantParticleFilter, u, p = parameters(pf), t = index(pf)*pf.Ts)
    s = state(pf)
    fns = model_functions(pf)
    cfg = config(pf, u, nothing)
    s.x, s.w, s.ancestors = device_call(:predict, fns, cfg, (X, w, v, p, c) -> device_predict(fns, cfg, X, w, v, p, c),
        s.x, s.w, pack(pf, t, u), device_params(pf, p), device_consts(pf))
    s.xprev = s.x
    s.t[] += 1
    pf
end

correct!(apf::AuxiliaryParticleFilter{<:ReactantParticleFilter}, u, y, p = parameters(apf), t = index(apf)*apf.Ts; kwargs...) =
    correct!(apf.pf, u, y, p, t; kwargs...)

function predict!(apf::AuxiliaryParticleFilter{<:ReactantParticleFilter}, u, y1, p = parameters(apf), t = index(apf)*apf.Ts)
    (y1 === missing || ismissing_measurement(y1)) && return predict!(apf.pf, u, p, t)
    pf = apf.pf
    s = state(pf)
    fns = model_functions(pf; first_stage = apf.first_stage)
    cfg = config(pf, u, y1)
    s.x, s.w, s.ancestors = device_call(:apf_predict, fns, cfg, (X, w, v, p, c) -> device_apf_predict(fns, cfg, X, w, v, p, c),
        s.x, s.w, pack(pf, t, u, y1), device_params(pf, p), device_consts(pf))
    s.xprev = s.x
    s.t[] += 1
    apf
end

"""
    measurement_equation!(pf::ReactantParticleFilter, u, y, p, t; g = pf.measurement_likelihood)

Increment of the log-weights on the device by the measurement log-likelihood, without normalization (see
[`logsumexp!`](@ref))
"""
function measurement_equation!(pf::ReactantParticleFilter, u, y, p, t; g = pf.measurement_likelihood)
    (y === missing || ismissing_measurement(y)) && return pf
    s = state(pf)
    fns = model_functions(pf; loglik = g)
    cfg = config(pf, u, y; loglik = g)
    s.w = device_call(:add_loglik, fns, cfg, (X, w, v, p, c) -> device_add_loglik(fns, cfg, X, w, v, p, c),
        s.x, s.w, pack(pf, t, u, y), device_params(pf, p), device_consts(pf))
    pf
end

function logsumexp!(s::DevicePFstate)
    s.w, l = device_call(:normalize, nothing, nothing, device_normalize, s.w)
    Float64(only(Array(l)))
end

"""
    propagate_particles!(pf::ReactantParticleFilter, u, p, t)
    propagate_particles!(pf::ReactantParticleFilter, u, j::AbstractVector{<:Integer}, p, t)

Propagation of the particles with noise on the device, without resampling or from the ancestors `j`
"""
function propagate_particles!(pf::ReactantParticleFilter, u, j::Union{Nothing, AbstractVector{<:Integer}}, p, t::Real)
    s = state(pf)
    fns = model_functions(pf)
    cfg = config(pf, u, nothing)
    jd = j === nothing ? nothing : Reactant.to_rarray(collect(Int64, j))
    s.x = device_call(:propagate, fns, cfg, (X, j, v, p, c) -> device_propagate(fns, cfg, X, j, v, p, c),
        s.xprev, jd, pack(pf, t, u), device_params(pf, p), device_consts(pf))
end
propagate_particles!(pf::ReactantParticleFilter, u, p, t::Real) = propagate_particles!(pf, u, nothing, p, t)

"""
    ll = loglik(pf::ReactantParticleFilter, u, y, p = parameters(pf))
    ll = loglik(apf::AuxiliaryParticleFilter{<:ReactantParticleFilter}, u, y, p = parameters(apf))

Log-likelihood of the trajectory `u, y`, with all time steps in one device program. The result is the one of the generic
`loglik`, which calls `update!` in every time step, but without a transfer between the host and the device in every
time step. The program is compiled for each length of the trajectory.
"""
loglik(pf::ReactantParticleFilter, u, y, p = parameters(pf)) = traced_loglik(pf, pf, u, y, p)
loglik(apf::AuxiliaryParticleFilter{<:ReactantParticleFilter}, u, y, p = parameters(apf)) = traced_loglik(apf, apf.pf, u, y, p)

function traced_loglik(f, pf::ReactantParticleFilter, u, y, p)
    length(u) == length(y) || throw(ArgumentError("u and y must have the same length"))
    apf = f isa AuxiliaryParticleFilter
    reset!(f)
    s = state(pf)
    Tn = length(y)
    hasy = [!(yk === missing || ismissing_measurement(yk)) for yk in y]
    ny = any(hasy) ? length(y[findfirst(hasy)]) : 0
    nu = veclength(u[1])
    # The times of the generic loglik: k Ts for the bootstrap filter, (k - 1) Ts for the auxiliary filter
    time(k) = (apf ? k - 1 : k) * pf.Ts
    function column(k, yk)
        v = Float64[s.seed >> 32, s.seed & 0xffffffff, k, time(k)]
        u[k] === nothing || append!(v, u[k])
        yk === nothing ? append!(v, zeros(ny)) : append!(v, yk)
    end
    Vc = reduce(hcat, [column(k, hasy[k] ? y[k] : nothing) for k in 1:Tn])
    hasy1 = [k < Tn && hasy[k+1] for k in 1:Tn]
    Vp = reduce(hcat, [column(k, hasy1[k] ? y[k+1] : nothing) for k in 1:Tn])
    fns = model_functions(pf; first_stage = apf ? f.first_stage : nothing)
    cfg = Static{(; nu, ny, kind = pf.pf isa ParticleFilter ? :pf : :advanced,
        gaussian = pf.measurement_likelihood === nothing, th = resample_threshold(pf))}()
    s.x, s.w, ll = device_call(:loglik_loop, fns, (cfg, apf),
        (X, w, Vc, Vp, hasy, hasy1, p, c) -> device_loglik_loop(fns, cfg, Val(apf), X, w, Vc, Vp, hasy, hasy1, p, c),
        s.x, s.w, Reactant.to_rarray(Vc), Reactant.to_rarray(Vp), Reactant.to_rarray(hasy), Reactant.to_rarray(hasy1),
        device_params(pf, p), device_consts(pf))
    s.xprev = s.x
    s.t[] += Tn
    Float64(only(Array(ll)))
end

"""
    xb, ll = smooth(pf::ReactantParticleFilter, M, u, y, p = parameters(pf); method = FFBSi())
    xb, ll = smooth(apf::AuxiliaryParticleFilter{<:ReactantParticleFilter}, M, u, y, p = parameters(apf); method = FFBSi())

Particle smoothing of `M` trajectories. Returns the smoothed particles `xb` (M × T, as for the other particle filters)
and the log-likelihood of the forward pass.

- `FFBSi()`: forward filtering and backward simulation on the device. The particles and weights of the forward pass
  remain on the device, which requires memory for `nx N T` numbers. Requires a `ParticleFilter` with a Gaussian
  `dynamics_density`, which defines the transition density.
- `GenealogySmoother(...)` and `FFBSi(; transition_logpdf)`: [`forward_trajectory`](@ref) with the filter on the
  device, and the smoother of the package on the host.
- `ParticleGibbsSmoother` is not supported, since the conditional particle filter is not implemented on the device.
"""
smooth(pf::ReactantParticleFilter, M::Integer, u, y, p = parameters(pf); method::ParticleSmoothingMethod = FFBSi()) =
    device_smooth(method, pf, pf, M, u, y, p)
smooth(apf::AuxiliaryParticleFilter{<:ReactantParticleFilter}, M::Integer, u, y, p = parameters(apf); method::ParticleSmoothingMethod = FFBSi()) =
    device_smooth(method, apf, apf.pf, M, u, y, p)

"Smoothing methods of the package on the host, with the forward pass of the filter on the device"
device_smooth(method::ParticleSmoothingMethod, f, pf, M, u, y, p) = smooth(forward_trajectory(f, u, y, p), M, p; method)
device_smooth(::ParticleGibbsSmoother, f, pf, M, u, y, p) = throw(ArgumentError("ParticleGibbsSmoother is not " *
    "supported by the ReactantParticleFilter, use GenealogySmoother() or FFBSi(), or the filter on the CPU"))

function device_smooth(method::FFBSi, f, pf::ReactantParticleFilter, M, u, y, p)
    method.transition_logpdf === nothing || return smooth(forward_trajectory(f, u, y, p), M, p; method)
    pf.pf isa ParticleFilter && dynamics_density(pf) !== nothing || throw(ArgumentError("FFBSi on the device " *
        "requires a ParticleFilter with a Gaussian dynamics_density (the transition density); use " *
        "GenealogySmoother() or FFBSi(; transition_logpdf) for an AdvancedParticleFilter"))
    length(u) == length(y) || throw(ArgumentError("u and y must have the same length"))
    Tn = length(y)
    # Forward pass: the particles and weights after the correction of every time step
    reset!(f)
    s = state(pf)
    X = Vector{typeof(s.x)}(undef, Tn)
    W = Vector{typeof(s.w)}(undef, Tn)
    ll = 0.0
    for t in 1:Tn
        ti = (t - 1) * pf.Ts
        ll += correct!(f, u[t], y[t], p, ti)[1]
        X[t], W[t] = s.x, s.w
        t < Tn || break
        f isa AuxiliaryParticleFilter ? predict!(f, u[t], y[t+1], p, ti) : predict!(f, u[t], p, ti)
    end
    # Backward simulation
    seed = rand(rng(pf), UInt64)
    im = Reactant.to_rarray(UInt64.(0:M-1))
    fns = model_functions(pf)
    cfg = config(pf, u[1], nothing)
    cfg0 = config(pf, nothing, nothing)
    xd = device_call(:smooth_init, nothing, cfg0, (Xf, wf, im, v) -> device_smooth_init(Xf, wf, im, v, cfg0),
        X[Tn], W[Tn], im, pack(pf, 0.0, nothing; seed, k = Tn))
    nx = size(s.x, 1)
    xb = Matrix{SVector{nx, device_eltype(pf)}}(undef, M, Tn)
    tohost(x) = collect(reinterpret(SVector{nx, eltype(x)}, vec(Array(x))))
    xb[:, Tn] = tohost(xd)
    consts = device_consts(pf)
    pd = device_params(pf, p)
    for t in Tn-1:-1:1
        xd = device_call(:smooth_step, fns, cfg, (Xf, wf, xb, im, v, p, c) -> device_smooth_step(fns, cfg, Xf, wf, xb, im, v, p, c),
            X[t], W[t], xd, im, pack(pf, (t - 1) * pf.Ts, u[t]; seed, k = t), pd, consts)
        xb[:, t] = tohost(xd)
    end
    xb, ll
end

include("reactant_kalman.jl")

end # module
