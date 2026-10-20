# ReactantParticleFilter against the Kalman filter for linear Gaussian models
using LowLevelParticleFilters, Reactant, Test, Random, Statistics, LinearAlgebra, StaticArrays
const LLPF = LowLevelParticleFilters

# Scalar model x⁺ = a x + w, y = x + e
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

# The same model as an AdvancedParticleFilter, with the noise intensity as a parameter
adyn(x, u, p, t, noise = false) = SA[p[1] * x[1]] + p[2] * randn_noise(noise, SVector{1})
ameas(x, u, p, t, noise = false) = SA[x[1]] + noise * σe * randn(SVector{1})
pa = [a, σw]

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

    @testset "AdvancedParticleFilter" begin
        adv = AdvancedParticleFilter(N, adyn, ameas, ml, nothing, d0; p = pa)
        radv = ReactantParticleFilter(adv; T)
        @test mean(loglik(radv, u, yv, pa) for _ in 1:3) ≈ llkf atol = tol
        aapf = AuxiliaryParticleFilter(ReactantParticleFilter(adv; T); first_stage = fs)
        @test mean(loglik(aapf, u, yv, pa) for _ in 1:3) ≈ llkf atol = tol
        # A different noise intensity in the parameters, with the compiled functions of the first filter
        pa2 = [a, 2σw]
        kf2 = KalmanFilter([a;;], zeros(1, 0), [1.0;;], zeros(1, 0), [(2σw)^2;;], [σe^2;;], LLPF.SimpleMvNormal([0.0], [1.0;;]); check = false)
        @test mean(loglik(radv, u, yv, pa2) for _ in 1:3) ≈ loglik(kf2, u, yv) atol = tol
        # Parameters as a named tuple of numbers, which must not be compiled as constants
        ndyn(x, u, p, t, noise = false) = SA[p.a * x[1]] + p.σ * randn_noise(noise, SVector{1})
        rn = ReactantParticleFilter(AdvancedParticleFilter(N, ndyn, ameas, ml, nothing, d0; p = (; a, σ = σw)); T)
        @test mean(loglik(rn, u, yv, (; a, σ = σw)) for _ in 1:3) ≈ llkf atol = tol
        @test mean(loglik(rn, u, yv, (; a, σ = 2σw)) for _ in 1:3) ≈ loglik(kf2, u, yv) atol = tol
    end

    @testset "forward_trajectory" begin
        sol = forward_trajectory(rpf, u, yv, p)
        @test sol.ll ≈ llkf atol = 2tol
        xm = only.(mean_trajectory(sol))
        solk = forward_trajectory(kf, u, yv)
        @test sqrt(mean(abs2, xm .- only.(solk.xt))) < 0.1
        @test weighted_mean(rpf) isa AbstractVector
        @test sum(expweights(rpf)) ≈ 1 atol = 1e-4
    end

    @testset "smoothing" begin
        ssol = smooth(kf, u, yv)
        xT, RT = only.(ssol.xT), only.(ssol.RT)
        for rs in (ReactantParticleFilter(N, dyn, meas, df, dm, d0; p, T),
                   AuxiliaryParticleFilter(ReactantParticleFilter(N, dyn, meas, df, dm, d0; p, T); first_stage = fs))
            xb, ll = smooth(rs, 500, u, yv, p)
            @test size(xb) == (500, Tn)
            @test eltype(xb) == SVector{1, T}
            @test ll ≈ llkf atol = 2tol
            xs = only.(xb)
            # The smoothed mean and variance against the Rauch-Tung-Striebel smoother (posterior standard deviation ≈ 0.6)
            @test sqrt(mean(abs2, vec(mean(xs, dims = 1)) .- xT)) < 0.1
            @test mean(vec(var(xs, dims = 1)) ./ RT) ≈ 1 atol = 0.15
        end
        radv = ReactantParticleFilter(AdvancedParticleFilter(N, adyn, ameas, ml, nothing, d0; p = pa); T)
        @test_throws ArgumentError smooth(radv, 10, u, yv, pa)
        @test_throws ArgumentError smooth(radv, 10, u, yv, pa; method = ParticleGibbsSmoother())
        # Smoothers of the package on the host, with the forward pass on the device
        rs = ReactantParticleFilter(N, dyn, meas, df, dm, d0; p, T)
        sol = forward_trajectory(rs, u, yv, p)
        @test all(a -> 1 <= a <= N, sol.a)
        @test any(!=(0), diff(sol.a; dims = 1)) # resampling has occurred
        for (f, pp) in ((rs, p), (radv, pa))
            xb, ll = smooth(f, 500, u, yv, pp; method = GenealogySmoother(lag = 10))
            @test size(xb) == (500, Tn)
            @test sqrt(mean(abs2, vec(mean(only.(xb), dims = 1)) .- xT)) < 0.2
        end
        tl(x⁺, x, u, p, t) = -0.5 * (x⁺[1] - a * x[1])^2 / σw^2
        xb, _ = smooth(rs, 200, u, yv, p; method = FFBSi(; transition_logpdf = tl))
        @test sqrt(mean(abs2, vec(mean(only.(xb), dims = 1)) .- xT)) < 0.15
    end

    @testset "missing measurements" begin
        ym = Union{Vector{Float64}, Vector{Missing}}[yv...]
        ym[10:20] .= Ref([missing])
        @test loglik(rpf, u, ym, p) ≈ loglik(kf, u, ym) atol = tol
    end

    @testset "reproducibility" begin
        # The initial particles and the keys of the device random numbers are drawn from the filter's rng
        l = [loglik(ReactantParticleFilter(N, dyn, meas, df, dm, d0; p, T, rng = Xoshiro(5)), u, yv, p) for _ in 1:2]
        @test l[1] == l[2]
        l = [loglik(ReactantParticleFilter(AdvancedParticleFilter(N, adyn, ameas, ml, nothing, d0; p = pa, rng = Xoshiro(5)); T), u, yv, pa) for _ in 1:2]
        @test l[1] == l[2]
    end

    @testset "loglik in one device program" begin
        # The generic loglik, which calls update! in every time step
        function stepwise(f, u, y, p)
            reset!(f)
            if f isa AuxiliaryParticleFilter
                sum(t -> f(u[t], y[t], y[t+1], p, (t - 1) * f.Ts)[1], 1:length(u)-1) + f.pf(u[end], y[end], p, (length(u) - 1) * f.Ts)[1]
            else
                sum(t -> f(u[t], y[t], p)[1], 1:length(u))
            end
        end
        ym = Union{Vector{Float64}, Vector{Missing}}[yv...]
        ym[10:20] .= Ref([missing])
        makers = [
            () -> ReactantParticleFilter(N, dyn, meas, df, dm, d0; p, T, resample_threshold = 0.5, rng = Xoshiro(7)),
            () -> AuxiliaryParticleFilter(ReactantParticleFilter(N, dyn, meas, df, dm, d0; p, T, rng = Xoshiro(7)); first_stage = fs),
            () -> ReactantParticleFilter(AdvancedParticleFilter(N, adyn, ameas, ml, nothing, d0; p = pa, rng = Xoshiro(7)); T),
        ]
        for mk in makers, yy in (yv, ym)
            f = mk()
            pp = f isa ReactantParticleFilter && f.pf isa AdvancedParticleFilter ? pa : p
            l1 = loglik(f, u, yy, pp)
            l2 = stepwise(mk(), u, yy, pp)
            # Float32: XLA may round differently in the fused loop, which can change a resampling decision
            T == Float64 ? (@test l1 ≈ l2 rtol = 1e-10) : (@test l1 ≈ l2 atol = tol)
        end
    end

    @testset "lower-level functions" begin
        reset!(rpf)
        LLPF.measurement_equation!(rpf, u[1], yv[1], p, 0.0)
        @test logsumexp!(rpf) isa Float64
        @test sum(expweights(rpf)) ≈ 1 atol = 1e-4
        LLPF.propagate_particles!(rpf, u[1], p, 0.0)
        LLPF.propagate_particles!(rpf, u[1], collect(N:-1:1), p, 0.0)
        @test length(particles(rpf)) == N
    end
end

@testset "Noise through a matrix" begin
    # Double integrator with noise on the velocity only (singular covariance of the dynamics noise)
    Ts = 0.1
    A = [1 Ts; 0 1]
    Bw = [Ts^2 / 2, Ts]
    σa, σy = 1.0, 0.1
    Tn2 = 100
    rng2 = Xoshiro(3)
    xs = [randn(rng2, 2)]
    for k in 2:Tn2
        push!(xs, A * xs[end] + Bw * σa * randn(rng2))
    end
    ys = [[x[1] + σy * randn(rng2)] for x in xs]
    us = [zeros(0) for _ in 1:Tn2]
    kf2 = KalmanFilter(A, zeros(2, 0), [1.0 0.0], zeros(1, 0), σa^2 * Bw * Bw', [σy^2;;], LLPF.SimpleMvNormal(zeros(2), Matrix(1.0I(2))); check = false)
    ll2 = loglik(kf2, us, ys)
    As, Bs = SMatrix{2, 2}(A), SVector{2}(Bw)
    dyn2(x, u, p, t, noise = false) = As * x + Bs * (p[1] * randn_noise(noise))
    meas2(x, u, p, t, noise = false) = SA[x[1]]
    ml2(x, u, y, p, t) = -0.5 * log(2π * σy^2) - 0.5 * (y[1] - x[1])^2 / σy^2
    fs2(x, u, y, p, t) = -0.5 * (y[1] - x[1])^2 / (σy^2 + 0.01)
    adv = AdvancedParticleFilter(10_000, dyn2, meas2, ml2, nothing, LLPF.SimpleMvNormal(zeros(2), Matrix(1.0I(2))); p = [σa])
    apf = AuxiliaryParticleFilter(ReactantParticleFilter(adv); first_stage = fs2)
    @test mean(loglik(apf, us, ys, [σa]) for _ in 1:3) ≈ ll2 atol = 3
end

@testset "rand in the model functions" begin
    bad(x, u, p, t, noise = false) = SA[p[1] * x[1] + randn()]
    radv = ReactantParticleFilter(AdvancedParticleFilter(100, bad, ameas, ml, nothing, d0; p = pa))
    @test_throws ErrorException loglik(radv, u, yv, pa)
end

@testset "Smoothing, state dimension 2" begin
    Ts = 0.1
    A = [1 Ts; 0 1]
    R1 = 0.5^2 * [Ts^2 / 2, Ts] * [Ts^2 / 2, Ts]' + 1e-3 * I(2)
    σy = 0.2
    Tn2 = 100
    rng2 = Xoshiro(4)
    xs = [randn(rng2, 2)]
    for k in 2:Tn2
        push!(xs, A * xs[end] + cholesky(R1).L * randn(rng2, 2))
    end
    ys = [[x[1] + σy * randn(rng2)] for x in xs]
    us = [zeros(0) for _ in 1:Tn2]
    d02 = LLPF.SimpleMvNormal(zeros(2), Matrix(1.0I(2)))
    kf2 = KalmanFilter(A, zeros(2, 0), [1.0 0.0], zeros(1, 0), Matrix(R1), [σy^2;;], d02; check = false)
    ssol = smooth(kf2, us, ys)
    As = SMatrix{2, 2}(A)
    dyn2(x, u, p, t) = As * x
    meas2(x, u, p, t) = SA[x[1]]
    rpf = ReactantParticleFilter(5000, dyn2, meas2, LLPF.SimpleMvNormal(zeros(2), Matrix(R1)), LLPF.SimpleMvNormal([0.0], [σy^2;;]), d02; rng = Xoshiro(1))
    xb, ll = smooth(rpf, 500, us, ys)
    xm = smoothed_mean(xb)
    xT = reduce(hcat, ssol.xT)
    sdT = sqrt.(reduce(hcat, diag.(ssol.RT)))
    # Errors of the smoothed means relative to the posterior standard deviations (Monte Carlo error with M = 500 and
    # the degeneracy of the paths about 0.1–0.2)
    @test sqrt(mean(abs2, (xm .- xT) ./ sdT)) < 0.3
end
