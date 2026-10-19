# Smoke test of BatchedParticleFilter with Reactant: x⁺ = a x + σw w, y = x + σe e, against the Kalman filter
using LowLevelParticleFilters, Reactant, Random, Statistics, LinearAlgebra, Printf
const LLPF = LowLevelParticleFilters

a, σw, σe, Tn = 0.9, 0.5, 1.0, 200
rng = Xoshiro(1)
x = zeros(Tn); y = zeros(Tn)
x[1] = randn(rng)
for k in 1:Tn
    k > 1 && (x[k] = a * x[k-1] + σw * randn(rng))
    y[k] = x[k] + σe * randn(rng)
end
kf = KalmanFilter([a;;], zeros(1, 0), [1.0;;], zeros(1, 0), [σw^2;;], [σe^2;;], LLPF.SimpleMvNormal([0.0], [1.0;;]); check = false)
llkf = loglik(kf, [zeros(0) for _ in 1:Tn], [[v] for v in y])

B = 4
p = (; a = fill(a, B, 1), σw = fill(σw, B, 1), σe = fill(σe, B, 1))
dyn(X, u, p, t, W) = (p.a .* X[1] .+ p.σw .* W[1],)
mll(X, u, y, p, t) = -0.5 .* log.(2π .* p.σe .^ 2) .- 0.5 .* (reshape(y[1, :], :, 1) .- X[1]) .^ 2 ./ p.σe .^ 2
init(p, Z) = (Z[1],)
fs(X, u, y, p, t) = -0.5 .* (reshape(y[1, :], :, 1) .- p.a .* X[1]) .^ 2 ./ (p.σe .^ 2 .+ p.σw .^ 2)
U = zeros(1, Tn)
Y = reshape(y, 1, :)
for (name, first_stage) in (("bootstrap", nothing), ("auxiliary", fs)), T in (Float64, Float32)
    pf = BatchedParticleFilter(4000; nx = 1, nw = 1, dynamics = dyn, measurement_loglik = mll, initial_state = init, first_stage)
    t = @elapsed ll = compile_loglik(pf, U, Y, p; T)
    v = ll(p, 1:B)
    v2 = ll(p, 1:B)
    @printf("%-9s %s: Kalman %.2f, particle filter %s (repeat identical: %s), compile %.1f s\n", name, T, llkf,
        round.(v; digits = 2), v == v2, t)
end
