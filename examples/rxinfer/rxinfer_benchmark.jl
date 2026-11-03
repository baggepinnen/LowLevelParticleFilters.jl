# Benchmark: LowLevelParticleFilters (with StaticArrays) vs RxInfer.jl
# on the linear-Gaussian state-space model from RxInfer's own benchmark suite
# (benchmarks/Linear Multivariate Gaussian State Space Model Benchmark.ipynb):
#
#   x[t] ~ N(A x[t-1], P),  y[t] ~ N(B x[t], Q)
#
# with 2D rotation dynamics. Both methods compute the exact Bayesian filtering
# distribution, so the filtered means must agree to machine precision.

using Pkg
Pkg.activate(@__DIR__)

using RxInfer, LowLevelParticleFilters, StaticArrays, LinearAlgebra, BenchmarkTools
using Random: MersenneTwister

# RxInfer is also benchmarked with static arrays below, but does not support them out
# of the box: its cholesky path dispatches to StaticArrays' cholesky, which demands
# exact symmetry, while the covariance propagated through A*Σ*A' + P is only Hermitian
# up to machine epsilon (the dense-Matrix path is robust to this). Patch the static
# path to be equally robust so the comparison is fair.
import FastCholesky
FastCholesky.fastcholesky(x::SMatrix) = cholesky(Hermitian((x + x') / 2))

θ = π / 15
A = [cos(θ) -sin(θ); sin(θ) cos(θ)] # state transition
B = diagm([1.3, 0.7])               # measurement matrix (C in LLPF notation)
P = Matrix(0.05I(2))                # process noise covariance  (R1 in LLPF notation)
Q = Matrix(10.0I(2))                # measurement noise covariance (R2 in LLPF notation)
x0_mean = zeros(2)
x0_cov = Matrix(100.0I(2))

function generate_data(rng, n, A, B, P, Q)
    x = Vector{Vector{Float64}}(undef, n)
    y = Vector{Vector{Float64}}(undef, n)
    x_prev = x0_mean
    for t in 1:n
        x[t] = rand(rng, MvNormalMeanCovariance(A * x_prev, P))
        y[t] = rand(rng, MvNormalMeanCovariance(B * x[t], Q))
        x_prev = x[t]
    end
    x, y
end

## RxInfer recursive filtering (from their official benchmark notebook)

@model function linear_gaussian_ssm_filtering(x_min_t_mean, x_min_t_cov, y_t, A, B, P, Q)
    x_min_t ~ MvNormal(μ = x_min_t_mean, Σ = x_min_t_cov)
    x_t     ~ MvNormal(μ = A * x_min_t, Σ = P)
    y_t     ~ MvNormal(μ = B * x_t, Σ = Q)
end

function rxinfer_filtering(observations, A, B, P, Q, x0_mean = x0_mean, x0_cov = x0_cov)
    autoupdates = @autoupdates begin
        x_min_t_mean, x_min_t_cov = mean_cov(q(x_t))
    end
    result = infer(
        model = linear_gaussian_ssm_filtering(A = A, B = B, P = P, Q = Q),
        data = (y_t = observations,),
        autoupdates = autoupdates,
        initialization = @initialization(q(x_t) = MvNormalMeanCovariance(x0_mean, x0_cov)),
        historyvars = (x_t = KeepLast(),),
        keephistory = length(observations),
    )
    result.history[:x_t]
end

## LowLevelParticleFilters with static arrays

A_s = SMatrix{2,2}(A)
B_s = SMatrix{2,0,Float64,0}() # no input
C_s = SMatrix{2,2}(B)
R1_s = SMatrix{2,2}(P)
R2_s = SMatrix{2,2}(Q)
d0 = LowLevelParticleFilters.SimpleMvNormal(SVector{2}(x0_mean), SMatrix{2,2}(x0_cov))

# Note: RxInfer's filtering model predicts before correcting (the prior for x_t is
# propagated through the dynamics), so we match it with predict! followed by correct!.
function llpf_filtering(observations)
    kf = KalmanFilter(A_s, B_s, C_s, 0, R1_s, R2_s, d0)
    u = SA_F64[]
    xt = Vector{SVector{2,Float64}}(undef, length(observations))
    Rt = Vector{SMatrix{2,2,Float64,4}}(undef, length(observations))
    for (t, y) in enumerate(observations)
        predict!(kf, u)
        correct!(kf, u, y)
        xt[t] = kf.x
        Rt[t] = kf.R
    end
    xt, Rt
end

## Static-array inputs for RxInfer (same matrices as the LLPF version above)

x0_mean_s = SVector{2}(x0_mean)
x0_cov_s = SMatrix{2,2}(x0_cov)

## Verify that all methods compute the same filtering distribution

rng = MersenneTwister(123)
states, observations = generate_data(rng, 1000, A, B, P, Q)
obs_static = [SVector{2}(y) for y in observations]

posts_rx = rxinfer_filtering(observations, A, B, P, Q)
posts_rx_s = rxinfer_filtering(obs_static, A_s, C_s, R1_s, R2_s, x0_mean_s, x0_cov_s)
xt_llpf, Rt_llpf = llpf_filtering(observations)

err_mean = maximum(norm(mean(posts_rx[t]) - xt_llpf[t]) for t in eachindex(observations))
err_cov  = maximum(norm(cov(posts_rx[t]) - Rt_llpf[t]) for t in eachindex(observations))
err_mean_s = maximum(norm(mean(posts_rx_s[t]) - xt_llpf[t]) for t in eachindex(observations))
err_cov_s  = maximum(norm(cov(posts_rx_s[t]) - Rt_llpf[t]) for t in eachindex(observations))
println("Max deviation of filtered means:       $err_mean (static RxInfer: $err_mean_s)")
println("Max deviation of filtered covariances: $err_cov (static RxInfer: $err_cov_s)")
@assert err_mean < 1e-8 && err_cov < 1e-8
@assert err_mean_s < 1e-8 && err_cov_s < 1e-8

## Benchmark

results = map([100, 1_000, 10_000]) do n
    states, observations = generate_data(rng, n, A, B, P, Q)
    obs_s = [SVector{2}(y) for y in observations]
    t_rx   = @belapsed rxinfer_filtering($observations, $A, $B, $P, $Q)
    t_rx_s = @belapsed rxinfer_filtering($obs_s, $A_s, $C_s, $R1_s, $R2_s, $x0_mean_s, $x0_cov_s)
    t_llpf = @belapsed llpf_filtering($observations)
    println("n = $(lpad(n, 6)):  RxInfer $(round(1000t_rx, sigdigits=4)) ms,  RxInfer static $(round(1000t_rx_s, sigdigits=4)) ms,  LLPF $(round(1000t_llpf, sigdigits=4)) ms,  speedup $(round(t_rx / t_llpf, sigdigits=4))x / $(round(t_rx_s / t_llpf, sigdigits=4))x")
    (; n, t_rx, t_rx_s, t_llpf)
end
