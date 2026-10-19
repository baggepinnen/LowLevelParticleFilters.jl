# ReactantParticleFilter against the Kalman filter for the scalar model x⁺ = a x + w, y = x + e
using LowLevelParticleFilters, Reactant, Test, Random, Statistics, LinearAlgebra, StaticArrays
const LLPF = LowLevelParticleFilters

a, σw, σe, Tn = 0.9, 0.5, 1.0, 100
rng = Xoshiro(1)
x = zeros(Tn); y = zeros(Tn)
x[1] = randn(rng)
for k in 1:Tn
    k > 1 && (x[k] = a * x[k-1] + σw * randn(rng))
    y[k] = x[k] + σe * randn(rng)
end
kf = KalmanFilter([a;;], zeros(1, 0), [1.0;;], zeros(1, 0), [σw^2;;], [σe^2;;], LLPF.SimpleMvNormal([0.0], [1.0;;]); check = false)
u = [zeros(0) for _ in 1:Tn]
yv = [[v] for v in y]
llkf = loglik(kf, u, yv)

dyn(x, u, p, t) = SA[p[1] * x[1]]
meas(x, u, p, t) = SA[x[1]]
p = [a]
N = 2000
df = LLPF.SimpleMvNormal([0.0], [σw^2;;])
dm = LLPF.SimpleMvNormal([0.0], [σe^2;;])
d0 = LLPF.SimpleMvNormal([0.0], [1.0;;])
ml(x, u, y, p, t) = -0.5 * log(2π * σe^2) - 0.5 * (y[1] - x[1])^2 / σe^2
fs(x, u, y, p, t) = -0.5 * (y[1] - x[1])^2 / (σe^2 + σw^2)
# The Monte Carlo standard deviation of the log-likelihood estimates is about 0.2 for these settings
tol = 1.5

@testset "ReactantParticleFilter $T" for T in (Float64, Float32)
    rpf = ReactantParticleFilter(N, dyn, meas, df, dm, d0; p, T, resample_threshold = 0.5)
    @test particletype(rpf) == SVector{1, T}
    @test num_particles(rpf) == N
    @test mean(loglik(rpf, u, yv, p) for _ in 1:3) ≈ llkf atol = tol

    @testset "measurement_likelihood" begin
        rml = ReactantParticleFilter(ParticleFilter(N, dyn, meas, df, dm, d0; p); T, measurement_likelihood = ml)
        @test mean(loglik(rml, u, yv, p) for _ in 1:3) ≈ llkf atol = tol
    end

    @testset "AuxiliaryParticleFilter" begin
        for first_stage in (nothing, fs)
            apf = AuxiliaryParticleFilter(ReactantParticleFilter(ParticleFilter(N, dyn, meas, df, dm, d0; p); T); first_stage)
            @test mean(loglik(apf, u, yv, p) for _ in 1:3) ≈ llkf atol = tol
        end
    end

    @testset "forward_trajectory and smoothing" begin
        sol = forward_trajectory(rpf, u, yv, p)
        @test sol.ll ≈ llkf atol = 2tol
        xm = only.(mean_trajectory(sol))
        solk = forward_trajectory(kf, u, yv)
        @test sqrt(mean(abs2, xm .- only.(solk.xt))) < 0.1
        @test weighted_mean(rpf) isa AbstractVector
        rs = ReactantParticleFilter(200, dyn, meas, df, dm, d0; p, T)
        xb, ll = smooth(rs, 50, u, yv, p)
        ssol = smooth(KalmanFilter([a;;], zeros(1, 0), [1.0;;], zeros(1, 0), [σw^2;;], [σe^2;;], LLPF.SimpleMvNormal([0.0], [1.0;;]); check = false), u, yv)
        @test sqrt(mean(abs2, vec(mean(only.(xb), dims = 1)) .- only.(ssol.xT))) < 0.2
    end

    @testset "missing measurements" begin
        ym = Union{Vector{Float64}, Vector{Missing}}[yv...]
        ym[10:20] .= Ref([missing])
        @test loglik(rpf, u, ym, p) ≈ loglik(kf, u, ym) atol = tol
    end

    @testset "reproducibility" begin
        # The initial particles and the device noise are drawn from the filter's rng, the resampling from the global rng
        l = map(1:2) do _
            Random.seed!(0)
            loglik(ReactantParticleFilter(N, dyn, meas, df, dm, d0; p, T, rng = Xoshiro(5)), u, yv, p)
        end
        @test l[1] == l[2]
    end
end
