# ------------------------------------------------------------------------------------------------------------
# Log-likelihood of a KalmanFilter for a batch of parameter candidates, in one device program
# ------------------------------------------------------------------------------------------------------------
# The parameters of a candidate are an object (number, array, tuple or named tuple). Its floating-point leaves are
# stacked along a leading batch dimension and traced, all other leaves are compiled as constants and must be equal for
# all candidates. The time loop is traced once; in every time step, the correction and the prediction of a single
# candidate are traced and batched over the candidates with `Ops.batch`.

"Floating-point leaves of a parameter object, in the order of `rebuild`"
float_leaves(p::AbstractFloat) = (p,)
float_leaves(p::AbstractArray{<:AbstractFloat}) = (p,)
float_leaves(p::Union{Tuple, NamedTuple}) = (Iterators.flatten(map(float_leaves, Tuple(p)))...,)
float_leaves(p) = ()

"Leaves of a parameter object that are compiled as constants"
const_leaves(p::AbstractFloat) = ()
const_leaves(p::AbstractArray{<:AbstractFloat}) = ()
const_leaves(p::Union{Tuple, NamedTuple}) = (Iterators.flatten(map(const_leaves, Tuple(p)))...,)
const_leaves(p) = (p,)

"Leaf `i` of all candidates, stacked along a leading batch dimension, in the precision `T`"
stack_leaf(::Type{T}, ls::Vector{<:Number}) where {T} = T.(ls)
function stack_leaf(::Type{T}, ls::Vector{<:AbstractArray}) where {T}
    S = zeros(T, length(ls), size(ls[1])...)
    for (b, l) in enumerate(ls)
        selectdim(S, 1, b) .= l
    end
    S
end

"""
The parameter object `template` with its floating-point leaves replaced by the traced leaves `ls` of a single
candidate (static arrays of traced numbers)
"""
function rebuild(template, ls)
    i = Ref(0)
    rb(t::AbstractFloat) = Reactant.@allowscalar ls[i[] += 1][]
    function rb(t::AbstractArray{<:AbstractFloat})
        l = ls[i[] += 1]
        Reactant.@allowscalar SArray{Tuple{size(t)...}}(ntuple(j -> l[j], Val(length(t))))
    end
    rb(t::Union{Tuple, NamedTuple}) = map(rb, t)
    rb(t) = t
    rb(template)
end

"Lower Cholesky factor of a symmetric `SMatrix`, without branches (NaN entries if the matrix is not positive definite)"
@generated function chol_lower(S::SMatrix{n, n}) where {n}
    L = [Symbol(:L, i, :_, j) for i in 1:n, j in 1:n]
    ex = quote end
    for j in 1:n
        s = :(S[$j, $j])
        for k in 1:j-1
            s = :($s - $(L[j, k])^2)
        end
        push!(ex.args, :($(L[j, j]) = sqrt($s)))
        for i in j+1:n
            s = :(S[$i, $j])
            for k in 1:j-1
                s = :($s - $(L[i, k]) * $(L[j, k]))
            end
            push!(ex.args, :($(L[i, j]) = $s / $(L[j, j])))
        end
    end
    entries = [i >= j ? L[i, j] : :(zero($(L[1, 1]))) for i in 1:n, j in 1:n]
    push!(ex.args, :(SMatrix{$n, $n}($(entries...))))
    ex
end

"Forward substitution `L \\ b` with a lower-triangular `L`"
@generated function fwdsub(L::SMatrix{n, n}, b::SVector{n}) where {n}
    z = [Symbol(:z, i) for i in 1:n]
    ex = quote end
    for i in 1:n
        s = :(b[$i])
        for k in 1:i-1
            s = :($s - L[$i, $k] * $(z[k]))
        end
        push!(ex.args, :($(z[i]) = $s / L[$i, $i]))
    end
    push!(ex.args, :(SVector($(z...))))
    ex
end
fwdsub(L::SMatrix{n, n}, B::SMatrix{n, m}) where {n, m} = hcat(ntuple(j -> fwdsub(L, B[:, j]), Val(m))...)

valtype(::Val{T}) where {T} = T

"Traced array of the type `T` from a static array of traced numbers or a constant array"
tarr(::Type{T}, A::StaticArray{<:Any, <:TracedRNumber}) where {T} = reshape(tovec(T, vec(A)), size(A))
tarr(::Type{T}, A::AbstractArray{<:Number}) where {T} = Ops.constant(T.(Array(A)))
tarr(::Type{T}, A::TracedRArray) where {T} = convert_array(T, A)
tarr(::Type{T}, A::Number) where {T} = tracedscalar(T, A)
"`SMatrix` (n × m) of the entries of a traced matrix"
smat(A, ::Val{n}, ::Val{m}) where {n, m} = Reactant.@allowscalar SMatrix{n, m}(ntuple(i -> A[i], Val(n * m)))
symm(P) = (P + transpose(P)) ./ 2

"""
Correction (if `h`, the measurement is available) and prediction of the state `x` (nx) and the covariance `P` (nx × nx)
of a single candidate with the parameters `p`, in the same order and with the same time `t` as `update!`. Returns the
new state, covariance and the log-likelihood of the measurement. The matrix products are traced-array operations; the
user functions, the Cholesky factorization of the innovation covariance (ny × ny) and the triangular solves with it
are traced with StaticArrays arithmetic on traced numbers.
"""
function kalman_step(m, cfg, x, P, p, u, y, t, h)
    (; nx, nu, ny) = cval(cfg)
    T = valtype(cval(cfg).T)
    gm(M, xs) = tarr(T, Reactant.call_with_native(LowLevelParticleFilters.get_mat, M, xs, u, p, t))
    xs = svec(x, Val(nx))
    C, D, R2 = gm(m.C, xs), gm(m.D, xs), gm(m.R2, xs)
    uv = nu == 0 ? nothing : tarr(T, u)
    e = nu == 0 ? tarr(T, y) - C * x : tarr(T, y) - C * x - D * uv
    CP = C * P
    S = symm(CP * transpose(C)) + R2
    L = Reactant.call_with_native(chol_lower, smat(S, Val(ny), Val(ny)))
    W = tarr(T, Reactant.call_with_native(fwdsub, L, smat(CP, Val(ny), Val(nx)))) # L⁻¹CP
    z = tarr(T, Reactant.call_with_native(fwdsub, L, svec(e, Val(ny))))          # L⁻¹e
    ll = -ny * log(2π) / 2 - sum(log, tovec(T, diag(L))) - sum(abs2, z) / 2
    Wt = transpose(W)
    x = ifelse.(h, x + Wt * z, x)
    P = ifelse.(h, symm(P - Wt * W), P)
    ll = ifelse.(h, ll, zero(ll))
    xs = svec(x, Val(nx))
    A, B, R1 = gm(m.A, xs), gm(m.B, xs), gm(m.R1, xs)
    x = nu == 0 ? A * x : A * x + B * uv
    P = m.α .* symm(A * P * transpose(A)) + R1
    x, P, ll
end

"One time step of all candidates: states `X` (B × nx), covariances `P` (B × nx × nx) and log-likelihoods (B)"
function batched_kalman_step(m, cfg, template, X, P, v, h, pl)
    (; nu, ny) = cval(cfg)
    T = valtype(cval(cfg).T)
    r = Ops.batch([X, P, pl...], [size(X, 1)]) do x, Pm, ls...
        t, u, y = Reactant.@allowscalar begin
            (tofloat(T, v[1]), SVector(ntuple(i -> tofloat(T, v[1 + i]), Val(nu))),
                SVector(ntuple(i -> tofloat(T, v[1 + nu + i]), Val(ny))))
        end
        xn, Pn, l = kalman_step(m, cfg, mat(x), mat(Pm), rebuild(template, ls), u, y, t, h)
        (xn, Pn, tovec(Float64, l))
    end
    r[1], r[2], Ops.reshape(r[3], [size(X, 1)])
end

"Log-likelihoods of all candidates for the packed data `V` (columns [t, u..., y...]) and the measurement mask `hasy`"
function device_kalman_loglik(m, cfg, template, X, P, V, hasy, pl...)
    ll = Ops.fill(0.0, (size(X, 1),))
    @trace track_numbers = false for k in 1:size(V, 2)
        X, P, l = batched_kalman_step(m, cfg, template, X, P, V[:, k], Reactant.@allowscalar(hasy[k]), pl)
        ll = ll .+ l
    end
    ll
end

"Packed data on the device, kept for the last `(u, y)` (by identity and content)"
const KALMAN_DATA = Ref{Any}(nothing)

function kalman_device_data(kf, u, y)
    key = (objectid(u), objectid(y), hash(u), hash(y), kf.Ts)
    cached = KALMAN_DATA[]
    cached !== nothing && cached[1] == key && return cached[2]
    length(u) == length(y) || throw(ArgumentError("u and y must have the same length"))
    hasy = [!(yk === missing || ismissing_measurement(yk)) for yk in y]
    any(hasy) || throw(ArgumentError("y contains no measurements"))
    ny = length(y[findfirst(hasy)])
    V = reduce(hcat, [Float64[(k - 1) * kf.Ts; u[k]; hasy[k] ? y[k] : zeros(ny)] for k in eachindex(u)])
    data = (; V = Reactant.to_rarray(V), hasy = Reactant.to_rarray(hasy), nu = length(u[1]), ny)
    KALMAN_DATA[] = (key, data)
    data
end

function batched_loglik(kf::KalmanFilter, u, y, ps::AbstractVector; T::Type{<:AbstractFloat} = Float64)
    isempty(ps) && throw(ArgumentError("ps must contain at least one parameter object"))
    for f in (:A, :B, :C, :D, :R1, :R2)
        getfield(kf, f) isa AbstractArray{<:Any, 3} && throw(ArgumentError("batched_loglik does not support " *
            "time-varying matrices given as 3-D arrays, give $f as a function $f(x, u, p, t) instead"))
    end
    template = ps[1]
    consts = const_leaves(template)
    for p in ps
        typeof(p) == typeof(template) && map(size, float_leaves(p)) == map(size, float_leaves(template)) &&
            isequal(const_leaves(p), consts) || throw(ArgumentError("All parameter objects in ps must have the same " *
            "type, the same sizes of the floating-point arrays and equal entries that are not floating-point numbers"))
    end
    data = lock(() -> kalman_device_data(kf, u, y), CACHE_LOCK)
    B = length(ps)
    lvs = map(float_leaves, ps)
    pl = [Reactant.to_rarray(stack_leaf(T, [l[i] for l in lvs])) for i in eachindex(lvs[1])]
    X = Reactant.to_rarray(repeat(T.(collect(kf.d0.μ))', B))
    P = Reactant.to_rarray(permutedims(repeat(T.(Matrix(kf.d0.Σ)), 1, 1, B), (3, 1, 2)))
    sm(M::AbstractMatrix) = SMatrix{size(M)...}(T.(M))
    sm(M) = M
    m = (; A = sm(kf.A), B = sm(kf.B), C = sm(kf.C), D = sm(kf.D), R1 = sm(kf.R1), R2 = sm(kf.R2), α = T(kf.α))
    cfg = Static{(; nx = length(kf.d0.μ), data.nu, data.ny, T = Val(T))}()
    f = (X, P, V, hasy, pl...) -> device_kalman_loglik(m, cfg, template, X, P, V, hasy, pl...)
    ll = device_call(:kalman_loglik, (m, typeof(template), consts), cfg, f, X, P, data.V,
        data.hasy, pl...)
    map(l -> isnan(l) ? -Inf : l, Array(ll))
end
