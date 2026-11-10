# Population-based optimization on a GPU

Gradient-free global optimizers such as particle swarm optimization, differential evolution and evolution strategies evaluate a population of candidate parameter vectors in every iteration. When the cost function is the negative log-likelihood of a Kalman filter, the candidates are independent filters run over the same data, so the whole population can be evaluated in one program on an accelerator (GPU). [`batched_loglik`](@ref) computes
```julia
[loglik(kf, u, y, p) for p in ps]
```
for a vector of parameter objects `ps`, with the time loop traced once into a single device program by [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl), and the correction and prediction of each time step batched over the candidates. The functionality is provided by a package extension and requires
```julia
using Reactant
```

The time recursion of a Kalman filter is sequential, and the duration of a time step on the device is determined mainly by the latency of the device rather than by the number of candidates ``B``. The method is therefore advantageous for large populations; see [Performance](@ref batched_kf_performance) below.

## Parameters on the host and on the device

The matrices `A, B, C, D, R1, R2` of the [`KalmanFilter`](@ref) are given as functions `(x, u, p, t) -> matrix` of the parameter object `p`, or as constant matrices. These functions are evaluated on the device for each candidate. A parameter object is a number, an array, or a (named) tuple of these; its floating-point numbers and arrays differ between the candidates, other entries are compiled as constants and must be equal for all candidates.

Computations that cannot be traced by Reactant, or that are performed once per candidate rather than once per time step, belong in the construction of the parameter objects on the host. A typical example is the discretization of a continuous-time model with a matrix exponential: the parameter object contains the discrete-time matrices, and the functions of the filter select among them.

## Example: thermal model of a borehole heat exchanger

The model below describes the heat transfer from the fluid in a borehole of a ground-source heat pump to the surrounding rock, a radial thermal network with a node for the borehole and ``n_r`` rings of rock with geometrically increasing radii, the outer ring coupled to the undisturbed ground temperature ``T_g``. The heat pump runs in cycles. While it runs, heat is extracted at the rate ``Q`` and the fluid circulates, which increases the conductance between the borehole node and the first ring. The model therefore has two regimes, selected by the input `u[1]`. The measurement is the fluid temperature ``T_{bh} - R_{fb} Q / L``, available only while the heat pump has been running for some time. The unknown parameters are the thermal conductivity of the rock ``λ``, the borehole thermal resistance ``R_{fb}``, and the standard deviations of the process noise ``σ_w`` and of the measurement noise ``σ_e``.

```julia
using LowLevelParticleFilters, Reactant, StaticArrays, LinearAlgebra, Random
using Metaheuristics: optimize, PSO, Options, minimizer # Metaheuristics also exports SA (simulated annealing)
const LLPF = LowLevelParticleFilters

const L, rb, rmax, ρc, cbh, Tg, Ts = 200.0, 0.055, 10.0, 2.2e6, 4.6e4, 10.0, 60.0
nr = 11      # number of rings
nx = nr + 1  # state dimension

"Continuous-time model x' = A x + Bq Q + b of the regime `run`"
function ct_matrices(λ, run)
    q = (rmax / rb)^(1 / nr)
    G = 2π * λ * L / log(q)
    A = zeros(nx, nx); Bq = zeros(nx); b = zeros(nx)
    Gb = run ? 2G : 0.5G # conductance between the borehole node and the first ring
    C = [cbh * L; [ρc * π * rb^2 * (q^(2i) - q^(2i - 2)) * L for i in 1:nr]] # heat capacities
    A[1, 1] = -Gb; A[1, 2] = Gb; Bq[1] = -1.0
    for i in 1:nr
        r = i + 1
        gin = i == 1 ? Gb : G
        A[r, r-1] += gin; A[r, r] -= gin
        if i < nr
            A[r, r+1] += G; A[r, r] -= G
        else
            A[r, r] -= 2G; b[r] += 2G * Tg
        end
    end
    A ./ C, Bq ./ C, b ./ C
end

"Exact discretization with the input u = [run, Q, 1]"
function discretize(λ, run)
    A, Bq, b = ct_matrices(λ, run)
    M = zeros(nx + 2, nx + 2)
    M[1:nx, 1:nx] .= A .* Ts; M[1:nx, nx+1] .= Bq .* Ts; M[1:nx, nx+2] .= b .* Ts
    E = exp(M)
    SMatrix{nx, nx}(E[1:nx, 1:nx]), SMatrix{nx, 3}([zeros(nx) E[1:nx, nx+1] E[1:nx, nx+2]])
end

"Parameter object of the decision vector θ = [λ, R_fb, σ_w, σ_e], constructed on the host"
function θ2p(θ)
    λ, Rfb, σw, σe = θ
    Φ1, Γ1 = discretize(λ, false)
    Φ2, Γ2 = discretize(λ, true)
    (; Φ1, Φ2, Γ1, Γ2, Rfb, Q = SMatrix{nx, nx}(Diagonal([σw^2; fill(1e-6, nr)])), σe)
end

# Functions of the filter, evaluated on the device. The regime is selected with ifelse, elementwise with map
A(x, u, p, t) = map((a, b) -> ifelse(u[1] > 0.5, a, b), p.Φ2, p.Φ1)
B(x, u, p, t) = map((a, b) -> ifelse(u[1] > 0.5, a, b), p.Γ2, p.Γ1)
D(x, u, p, t) = SA[0.0 (-p.Rfb / L) 0.0]
R1(x, u, p, t) = p.Q
R2(x, u, p, t) = SMatrix{1, 1}(p.σe^2)
C = SMatrix{1, nx}([1.0; zeros(nr)]')
d0 = LLPF.SimpleMvNormal(SVector{nx}(fill(Tg, nx)), SMatrix{nx, nx}(1.0I))
kf = KalmanFilter(A, B, C, D, R1, R2, d0; nx, nu = 3, ny = 1, check = false)
```

We simulate data with on/off cycles of random length, and remove the measurements while the heat pump is stopped and during the first 15 minutes of each cycle.
```julia
function cycles(Tn; rng = Xoshiro(1))
    run = falses(Tn); k = 1; on = false
    while k <= Tn
        len = on ? rand(rng, 30:60) : rand(rng, 40:120)
        run[k:min(Tn, k + len - 1)] .= on
        k += len; on = !on
    end
    run
end
Tn = 50_000
run = cycles(Tn)
u = [SA[Float64(run[k]), run[k] ? 4000.0 + 300randn() : 0.0, 1.0] for k in 1:Tn]
θ_true = [3.0, 0.05, 0.01, 0.1]
_, _, ytrue = simulate(kf, u, θ2p(θ_true))
y = Union{Missing, SVector{1, Float64}}[k > 15 && all(run[k-15:k]) ? ytrue[k] : missing for k in 1:Tn]
```

The optimizers of [Metaheuristics.jl](https://github.com/jmejia8/Metaheuristics.jl) evaluate the cost of the whole population in one call when `parallel_evaluation = true`; the cost function then receives a matrix with one candidate per row and returns a vector. The optimization is performed over the logarithms of the parameters.
```julia
lb = log.([1.0, 0.005, 1e-4, 0.01])
ub = log.([10.0, 0.3, 0.5, 1.0])
cost(X) = -batched_loglik(kf, u, y, [θ2p(exp.(X[i, :])) for i in axes(X, 1)])
options = Options(; iterations = 40, parallel_evaluation = true, seed = 1)
res = optimize(cost, [lb ub], PSO(; N = 256, options))
θ_est = exp.(minimizer(res))
```

With 40 iterations of a population of 256 candidates (10240 evaluations of the log-likelihood over 50000 time steps), the estimate is ``θ = [3.004, 0.0490, 0.00893, 0.1002]``, close to the true parameters ``[3, 0.05, 0.01, 0.1]``. The negative log-likelihood of the estimate is slightly lower than that of the true parameters, as expected for a maximum-likelihood estimate. On an NVIDIA H100 GPU, the optimization takes 35 s, of which one evaluation of a population takes 0.86 s; the same optimization with the cost evaluated by [`loglik`](@ref) with `Threads.@threads` over the candidates on 64 threads of an AMD EPYC 9354 takes 358 s, and arrives at the same estimate. The first call of `batched_loglik` in a Julia session took 82 s, mainly for the compilation of the tracing code.

!!! note "Several threads on the CPU"
    [`loglik`](@ref) mutates the state of the filter, so a filter object must not be shared between threads. Use one copy of the filter per candidate or per task, e.g., `loglik(deepcopy(kf), u, y, p)`.

## Requirements on the model

- The filter is a [`KalmanFilter`](@ref). Time-varying matrices given as 3-D arrays are not supported; give them as functions of `u`, `p` or `t` instead.
- Inside the functions of the model, the state, input, parameters and time are static arrays and numbers of traced values. Comparisons of traced numbers return traced Booleans; select between values with `ifelse` rather than with `if` or `? :`, applied elementwise with `map` to arrays, as in the example. Broadcasting over static arrays with a traced number as an argument (`ifelse.(c, a, b)`, `a .* s`) is not supported, while `s * a` is.
- The initial state distribution `kf.d0` is the same for all candidates.
- A candidate for which the innovation covariance is not positive definite obtains the log-likelihood `-Inf`, whereas [`loglik`](@ref) throws an error.

The program is compiled at the first call for a given filter, type of the parameter objects, number of candidates and length of the data; the first call in a Julia session also includes the compilation of the tracing code, which took 40–80 s in the measurements below. The inputs and measurements are transferred to the device when they differ from those of the previous call.

## [Performance](@id batched_kf_performance)

The table below lists the duration of one time step for all ``B`` candidates in milliseconds, for the model above with ``T = 5000`` and the state dimensions 6 and 12, measured on an NVIDIA H100 GPU (`Float64`) and with [`loglik`](@ref) on an AMD EPYC 9354 (32 cores), on one thread and with `Threads.@threads` over the candidates on 64 threads (one copy of the filter per candidate). The CPU was shared with other jobs during the measurements. The log-likelihoods computed on the device agree with those of [`loglik`](@ref) to a relative error of about ``10^{-14}``.

| ``B`` | device, ``n_x = 6`` | 1 thread | 64 threads | device, ``n_x = 12`` | 1 thread | 64 threads |
|------:|------:|------:|------:|------:|------:|------:|
| 16    | 0.013 | 0.013 | 0.007 | 0.012 | 0.026 | 0.032 |
| 256   | 0.016 | 0.22  | 0.077 | 0.015 | 0.34  | 0.18  |
| 4096  | 0.036 | 5.5   | 2.1   | 0.077 | 4.9   | 2.0   |
| 16384 | 0.11  | 15    | 8.2   | 0.27  | 22    | 7.8   |
| 65536 | 0.44  | 52    | 33    | 1.3   | 75    | 30    |

Up to about a thousand candidates, the duration of a time step on the device is nearly independent of ``B``, about 0.01–0.02 ms, determined by the latency of the device. The device is faster than 64 CPU threads from a few hundred candidates, and 25–75 times faster for 4096 or more candidates. With this model, [`loglik`](@ref) allocates memory in every time step (about 4 kB), which limits the speedup of the threaded evaluation on the CPU to a factor of about 2–3. Since the time recursion is sequential, the duration of a call is proportional to the length of the data: for ``10^6`` time steps, a population of 256 candidates is evaluated in about 15 s.
