"""
Log-likelihood of a `BatchedParticleFilter` on an accelerator with Reactant.jl: the whole time loop is one
traced loop (`@trace for`); every decision (measurement or not, resampling or not) is a selection
(`ifelse`), so that each batch member takes its own decisions.

Port of the implementation for the model of an energy well (HomeAssistantBot.jl,
`dyadmodel/HouseComponents/scripts/borehole_gpu/gpu_pf.jl`) with the model-specific parts replaced by the
user functions of `BatchedParticleFilter`. Work in progress: see the pull request for the state and the
findings of the model-specific version.
"""
module LowLevelParticleFiltersReactantExt

using LowLevelParticleFilters: LowLevelParticleFilters, BatchedParticleFilter
using Reactant: Reactant, Ops, @trace, TracedRNumber, TracedRArray

"""
Compile options: the default loop-raising passes hoist the counters of the random numbers out of the loop
as an array (T − 1) × N, which takes 2 GB at N = 1000
"""
const COMPILE_OPTIONS = Reactant.CompileOptions(; disable_loop_raising_passes = true)

"Materialization of a wrapped traced array (reshape, view), which `dot_general`, `scatter` and indexing reject"
mat(x) = x isa TracedRArray ? x : Reactant.TracedUtils.materialize_traced_array(x)
"""
Optimization barrier: without it, XLA fuses the elementwise results that feed several consumers (the
log-weights of the measurements, the propagated particles) into the reductions that consume them, which ran
about four times slower than a separate elementwise kernel
"""
barrier(x...) = Ops.optimization_barrier(map(mat, x)...)

# ------------------------------------------------------------------------------------------------------------
# Conversions for traced numbers that Reactant does not provide
# ------------------------------------------------------------------------------------------------------------
"Logical right shift (Base.>> has no method for traced unsigned integers)"
@inline shr(x::TracedRNumber{UInt64}, n) = Ops.shift_right_logical(x, Reactant.promote_to(TracedRNumber{UInt64}, UInt64(n)))
@inline shr(x::UInt64, n) = x >> n
"Conversion of an unsigned integer to the floating-point type `T`"
@inline tofloat(::Type{T}, x::TracedRNumber{S}) where {T, S} = S === T ? x : Ops.convert(TracedRNumber{T}, x)
@inline tofloat(::Type{T}, x::Union{UInt64, Float32}) where {T} = T(x)
"Conversion of the element type of an array"
convert_array(::Type{T}, x::TracedRArray{S, N}) where {T, S, N} = S === T ? x : Ops.convert(TracedRArray{T, N}, x)
convert_array(::Type{T}, x::AbstractArray) where {T} = T.(x)
"Conversion of a (traced) integer to UInt64"
touint(k) = k isa TracedRNumber ? Ops.convert(TracedRNumber{UInt64}, k) : UInt64(k)

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
const SALT_PROP = 0x0000000000000000
const SALT_RESAMPLE = 0x5851f42d4c957f2d
const SALT_INIT = 0x14057b7ef767814f
"Key of the stream `salt` of the seed `seed`"
stream_key(seed, salt) = mix64(mix64(UInt64(seed) + 0x632be59bd9b4e019) ⊻ salt)
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
Standard normal arrays (B, N), `nw` of them, of step `k`: the counters of particle n are
2 np (k N + n) + 2 (pair − 1) with np = ⌈nw/2⌉ pairs. `iv` is 2 (0:N−1) as a (1, N) row.
"""
function normals(::Type{T}, key, k, iv, nw) where {T}
    N = size(iv, 2) # (an Int captured by the traced loop body would be traced)
    np = cld(nw, 2)
    base = touint(k) * UInt64(2 * np * N)
    ntuple(nw) do s
        ctr = iv .* UInt64(np) .+ UInt64(2 * ((s + 1) ÷ 2 - 1)) .+ base
        isodd(s) ? normal_cos.(T, key, ctr) : normal_sin.(T, key, ctr)
    end
end

# ------------------------------------------------------------------------------------------------------------
# Log-sum-exp and systematic resampling
# ------------------------------------------------------------------------------------------------------------
"Log of the sum of the exponentials over the particles: a (B, N) → (B, 1)"
function lse_rows(a)
    mx = maximum(a; dims = 2)
    mx .+ log.(sum(exp.(a .- mx); dims = 2))
end

"""
Cumulative sum of the weights p (B, N) along the particles: Float32 within blocks of `MIXED_BLOCK` particles
and Float64 for the offsets of the blocks. A Float64 cumulative sum is slow on GeForce GPUs (1/64 of the
Float32 rate); in Float32 alone the position N c_i / c_N has an error of the order of 1e-7 N slots.
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

Ancestor indices (B, N) of systematic resampling of the normalized log-weights `a` (B, N) with the uniform
numbers `u` (B, 1). The ancestor of slot n (0-based) is j_n = 1 + #{i < N : N c_i/c_N < u + n}. With the
first slot f_i = ⌊N c_i/c_N − u⌋ + 1 after particle i, nondecreasing in i, the count is max{i < N : f_i ≤ n}:
the last particle of each value of f_i is written to the slot f_i (scatter with the combiner `max`), followed
by a cumulative maximum over the slots. A histogram by scatter-add suffered from atomic contention, and a plain
store with unique indices raced (the cumulative sum of XLA is not exactly monotone in floating point).
"""
function systematic(a, u, consts)
    B, N = size(a)
    c = cumsum_weights(exp.(a))
    s = c .* (Float64(N) ./ c[:, N:N])
    f = convert_array(Int64, clamp.(floor.(s .- u) .+ 1, 0.0, Float64(N)))
    f = ifelse.(consts.lastslot, N, f)
    fnext = hcat(f[:, 2:N], Ops.fill(Int64(N + 1), (B, 1)))
    target = ifelse.(f .!= fnext, f, N) # slot N is out of bounds
    lin = mat(reshape(consts.bidx .+ B .* target, B * N, 1))
    hist = Ops.fill(Int32(0), (B * N,))
    hist = Ops.scatter(max, [hist], lin, [mat(reshape(consts.iota, B * N))];
        update_window_dims = Int64[], inserted_window_dims = [1], input_batching_dims = Int64[],
        scatter_indices_batching_dims = Int64[], scatter_dims_to_operand_dims = [1], index_vector_dim = 2)[1]
    Int32(1) .+ accumulate(max, reshape(hist, B, N); dims = 2)
end

"Array (B, N) at the linear indices `lin`"
gather_rows(v, lin) = reshape(mat(reshape(v, length(v)))[lin], size(v))
"Linear indices of the ancestors j (B, N) in an array (B, N)"
ancestor_index(j, bidx) = mat(reshape(bidx .+ size(j, 1) .* (convert_array(Int64, j) .- 1), length(j)))

# ------------------------------------------------------------------------------------------------------------
# Filter
# ------------------------------------------------------------------------------------------------------------
"""
Static settings of the filter as a type parameter: every number captured by the traced function (also a
field of a captured struct) is traced, and a traced size or loop bound fails. The user functions are passed
separately; a closure that captures numbers has the same problem.
"""
struct Static{C} end
cval(::Static{C}) where {C} = C
"""
Step k → k + 1 of the auxiliary particle filter (with `first_stage`) or of the bootstrap filter. The particles
are gathered at their ancestors and then propagated by the user dynamics, so that XLA can fuse the gather into
the kernel of the dynamics. (For linear dynamics, the model-specific version predicts before the resampling,
which saves one pass over the particles.)
"""
function step(fns, cfg, X, w, ll, k, data, p, consts)
    T = Reactant.unwrapped_eltype(w)
    B, N = size(w)
    u = data.u[:, k]
    ynext = data.y[:, :, k+1]
    h = Reactant.@allowscalar data.hasy[k+1]
    λ = fns.first_stage === nothing ? zero(T) .* w : ifelse.(h, fns.first_stage(X, u, ynext, p, k), zero(T))
    λ, = barrier(λ)
    a = w .+ λ
    la = lse_rows(a)
    ess = 1 ./ sum(exp.(2 .* w); dims = 2)
    res = (fns.first_stage === nothing ? false : h) .| (ess .< T(cval(cfg).ess_threshold * N))
    uu = u01.(Float64, draw64.(consts.ku, touint(k)), false)
    j = ifelse.(res, systematic(a .- la, uu, consts), consts.iota)
    lin = ancestor_index(j, consts.bidx)
    Xg = map(x -> gather_rows(x, lin), X)
    W = normals(T, consts.kp, k, consts.iv, cval(cfg).nw)
    X = fns.dynamics(Xg, u, p, k, W)
    lw = ifelse.(h, fns.measurement_loglik(X, u, ynext, p, k + 1), zero(T)) .- gather_rows(λ, lin)
    out = barrier(X..., lw)
    X, lw = out[1:end-1], out[end]
    w0 = ifelse.(res, T(-log(N)), w)
    # Increment of the log-likelihood: log Σ w λ + log Σ w0 exp(lw) (bootstrap filter: λ = 0, la = 0)
    l2 = lse_rows(w0 .+ lw)
    inc = convert_array(Float64, vec(ifelse.(res, la, zero(T)) .+ l2))
    ll = ll .+ ifelse.(h & (k + 1 > consts.burnin), inc, 0.0)
    X, ifelse.(h, w0 .+ lw .- l2, w0), ll
end

function loglik_device(fns, cfg, data, p, consts)
    T = Reactant.unwrapped_eltype(consts.w0)
    X = fns.initial_state(p, normals(T, consts.k0, 0, consts.iv, cval(cfg).nx))
    w = consts.w0
    # The measurement of the first time step, by a plain correction (in the likelihood if burnin = 0)
    h1 = Reactant.@allowscalar data.hasy[1]
    a = w .+ ifelse.(h1, fns.measurement_loglik(X, data.u[:, 1], data.y[:, :, 1], p, 1), zero(T))
    l1 = lse_rows(a)
    w = a .- l1
    ll = ifelse.(h1 & (consts.burnin < 1), convert_array(Float64, vec(l1)), 0.0)
    K = size(data.u, 2) - 1
    U = cval(cfg).unroll
    @trace for kk in 1:(K ÷ U)
        for r in 1:cval(cfg).unroll # (U itself would be traced inside the loop body)
            X, w, ll = step(fns, cfg, X, w, ll, cval(cfg).unroll * (kk - 1) + r, data, p, consts)
        end
    end
    for k in (U * (K ÷ U) + 1):K
        X, w, ll = step(fns, cfg, X, w, ll, k, data, p, consts)
    end
    ll
end

"Batch size of the parameters: the first dimension of their arrays"
batchsize(p) = maximum(x -> x isa AbstractArray ? size(x, 1) : 1, values(p))

"Constant arrays of the filter for the seeds `seeds` (one per batch member)"
function make_consts(::Type{T}, N, B, seeds; burnin) where {T}
    length(seeds) == B || throw(ArgumentError("one seed per batch member is required ($B), got $(length(seeds))"))
    (; k0 = reshape(UInt64[stream_key(s, SALT_INIT) for s in seeds], B, 1),
       kp = reshape(UInt64[stream_key(s, SALT_PROP) for s in seeds], B, 1),
       ku = reshape(UInt64[stream_key(s, SALT_RESAMPLE) for s in seeds], B, 1),
       iv = reshape(UInt64.(2 .* (0:N-1)), 1, N),
       iota = repeat(Int32.(1:N)', B, 1),
       bidx = reshape(collect(1:B), B, 1),
       lastslot = repeat(reshape((1:N) .== N, 1, N), B, 1),
       w0 = fill(T(-log(N)), B, N),
       burnin)
end

function LowLevelParticleFilters.compile_loglik(pf::BatchedParticleFilter, u, y, p; T = Float32, burnin = 0)
    B = batchsize(p)
    hasy = [!any(v -> ismissing(v) || isnan(v), y[:, k]) for k in axes(y, 2)]
    ydev = repeat(reshape(T.(coalesce.(y, zero(T))), size(y, 1), 1, :), 1, B, 1)
    data = Reactant.to_rarray((; u = T.(u), y = ifelse.(isnan.(ydev), zero(T), ydev), hasy))
    c0 = make_consts(T, pf.N, B, 1:B; burnin)
    fns = (; pf.dynamics, pf.measurement_loglik, pf.initial_state, pf.first_stage)
    cfg = Static{(; pf.nx, pf.nw, pf.unroll, pf.ess_threshold)}()
    f = Reactant.compile((data, p, consts) -> loglik_device(fns, cfg, data, p, consts),
        (data, Reactant.to_rarray(p), Reactant.to_rarray(c0)); compile_options = COMPILE_OPTIONS)
    function (p, seeds)
        Array(f(data, Reactant.to_rarray(p), Reactant.to_rarray(make_consts(T, pf.N, B, seeds; burnin))))
    end
end

end # module
