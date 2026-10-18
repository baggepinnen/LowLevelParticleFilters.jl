using LowLevelParticleFilters, LinearAlgebra, Random, Statistics, Test
using LowLevelParticleFilters: SimpleMvNormal

# Log-likelihoods of the particle filters against the exact log-likelihood of the Kalman filter on a linear
# Gaussian model, x⁺ = a x + w, w ~ N(0, σw²), y = x + e, e ~ N(0, σe²), data simulated with a = 0.9
function apf_test_problem(σw, σe; T = 200, N = 2000)
    rng = Xoshiro(1)
    x = 0.0
    y = Vector{Vector{Float64}}(undef, T)
    for t in 1:T
        y[t] = [x + σe * randn(rng)]
        x = 0.9 * x + σw * randn(rng)
    end
    u = [Float64[] for _ in 1:T]
    d0 = SimpleMvNormal([0.0], [1.0;;])
    dw = SimpleMvNormal([0.0], [σw^2;;])
    de = SimpleMvNormal([0.0], [σe^2;;])
    kf(a) = KalmanFilter([a;;], zeros(1, 0), [1.0;;], 0, [σw^2;;], [σe^2;;], d0)
    # Dynamics with the four arguments of a ParticleFilter (no method with a fifth argument)
    pf(a; seed, kwargs...) = ParticleFilter(N, (x, u, p, t) -> a .* x, (x, u, p, t) -> x, dw, de, d0; rng = Xoshiro(seed), kwargs...)
    function adv(a; seed)
        nrng = Xoshiro(seed + 1000)
        dyn(x, u, p, t, noise = false) = a .* x .+ (noise ? σw .* randn(nrng, 1) : zeros(1))
        meas(x, u, p, t, noise = false) = x
        lik(x, u, y, p, t) = -0.5 * ((y[1] - x[1]) / σe)^2 - log(σe * sqrt(2π))
        AdvancedParticleFilter(N, dyn, meas, lik, dw, d0; rng = Xoshiro(seed))
    end
    # First-stage weights with the predictive variance of the measurement
    σp = sqrt(σe^2 + σw^2)
    predictive(x, u, y, p, t) = -0.5 * ((y[1] - x[1]) / σp)^2 - log(σp * sqrt(2π))
    (; u, y, kf, pf, adv, predictive)
end
mean_ll(f, u, y) = mean(loglik(f(seed), u, y) for seed in 1:3)

@testset "Particle-filter log-likelihoods against the Kalman filter" begin
    # Measurement noise larger than the process noise: the default first stage of the auxiliary filter
    (; u, y, kf, pf, adv) = apf_test_problem(0.5, 1.0)
    for a in (0.7, 0.9)
        lkf = loglik(kf(a), u, y)
        # The default resample_threshold leaves steps without resampling, which propagate with the noise of
        # dynamics_density
        @test mean_ll(seed -> pf(a; seed), u, y) ≈ lkf atol = 3
        @test mean_ll(seed -> pf(a; seed, resample_threshold = 1.0), u, y) ≈ lkf atol = 3
        @test mean_ll(seed -> adv(a; seed), u, y) ≈ lkf atol = 3
        @test mean_ll(seed -> AuxiliaryParticleFilter(pf(a; seed)), u, y) ≈ lkf atol = 3
        @test mean_ll(seed -> AuxiliaryParticleFilter(adv(a; seed)), u, y) ≈ lkf atol = 3
    end

    # Measurement noise smaller than the process noise: the first stage with the predictive variance
    (; u, y, kf, pf, adv, predictive) = apf_test_problem(1.0, 0.5)
    for a in (0.7, 0.9)
        lkf = loglik(kf(a), u, y)
        @test mean_ll(seed -> pf(a; seed), u, y) ≈ lkf atol = 3
        @test mean_ll(seed -> AuxiliaryParticleFilter(pf(a; seed); first_stage = predictive), u, y) ≈ lkf atol = 3
        @test mean_ll(seed -> AuxiliaryParticleFilter(adv(a; seed); first_stage = predictive), u, y) ≈ lkf atol = 3
    end
    # The difference of the log-likelihoods between a = 0.7 and a = 0.9 is that of the Kalman filter
    dkf = loglik(kf(0.9), u, y) - loglik(kf(0.7), u, y)
    for f in (a -> (seed -> pf(a; seed)), a -> (seed -> AuxiliaryParticleFilter(pf(a; seed); first_stage = predictive)),
            a -> (seed -> AuxiliaryParticleFilter(adv(a; seed); first_stage = predictive)))
        @test mean_ll(f(0.9), u, y) - mean_ll(f(0.7), u, y) ≈ dkf atol = 5
    end

    # Missing measurements: the auxiliary filter predicts without the look-ahead and skips the correction
    ym = Vector{Union{Missing, Vector{Float64}}}(y)
    ym[50:60] .= missing
    lkf = loglik(kf(0.9), u, ym)
    @test mean(loglik(AuxiliaryParticleFilter(pf(0.9; seed); first_stage = predictive), u, ym) for seed in 1:3) ≈ lkf atol = 3
    @test mean(loglik(AuxiliaryParticleFilter(adv(0.9; seed); first_stage = predictive), u, ym) for seed in 1:3) ≈ lkf atol = 3
end
