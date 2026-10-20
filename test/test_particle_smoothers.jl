# Wrapped in a module to avoid name clashes with the other test files included in the same session
module TestParticleSmoothers
using LowLevelParticleFilters, LinearAlgebra, Statistics, Test, Random, StaticArrays
const LLPF = LowLevelParticleFilters

# Linear Gaussian model, the RTS smoother of the corresponding Kalman filter is the reference
Random.seed!(0)
A, B, C = SA[0.9;;], SA[1.0;;], SA[1.0;;]
R1, R2 = SA[0.1^2;;], SA[0.2^2;;]
d0 = LLPF.SimpleMvNormal(SA[0.0], SA[1.0;;])
kf = KalmanFilter(A, B, C, 0, R1, R2, d0)
T = 100
u = [SA[sin(k/10)] for k in 1:T]
x, u, y = LLPF.simulate(kf, u)
ksol = smooth(kf, u, y)
xT = getindex.(ksol.xT, 1)
σT = sqrt.(getindex.(ksol.RT, 1))

dyn(x, u, p, t, noise = false) = A*x + B*u + noise*0.1*randn(SVector{1})
meas(x, u, p, t, noise = false) = C*x + noise*0.2*randn(SVector{1})
meas_ll(x, u, y, p, t) = LLPF.extended_logpdf(LLPF.SimpleMvNormal(R2), y - C*x)
# No dynamics density is provided to the advanced filter
apf = AdvancedParticleFilter(500, dyn, meas, meas_ll, nothing, d0)
pf = ParticleFilter(500, (x, u, p, t) -> A*x + B*u, (x, u, p, t) -> C*x, LLPF.SimpleMvNormal(R1), LLPF.SimpleMvNormal(R2), d0)
filters = (; apf, aux = AuxiliaryParticleFilter(apf), pf, auxpf = AuxiliaryParticleFilter(pf))

smean(xb) = vec(smoothed_mean(xb))
sstd(xb) = sqrt.(getindex.(smoothed_cov(xb), 1))
rmse(a, b) = sqrt(mean(abs2, a .- b))

@testset "Ancestor indices" begin
    for (name, f) in pairs(filters)
        sol = forward_trajectory(f, u, y)
        N = num_particles(f)
        @test size(sol.a) == (N, T)
        @test sol.a[:, 1] == 1:N
        @test all(in(1:N), sol.a)
    end
    # Without resampling, every particle is its own ancestor
    apf0 = AdvancedParticleFilter(100, dyn, meas, meas_ll, nothing, d0; resample_threshold = 0.0)
    sol = forward_trajectory(apf0, u, y)
    @test all(sol.a[:, t] == 1:100 for t in 1:T)
    # A solution constructed without ancestor indices
    sol_noa = ParticleFilteringSolution(sol.f, sol.u, sol.y, sol.x, sol.w, sol.we, sol.ll)
    @test sol_noa.a === nothing
    @test_throws ArgumentError smooth(sol_noa, 10; method = GenealogySmoother())
end

@testset "Conditional resampling" begin
    Random.seed!(1)
    N = 5
    we = [0.1, 0.2, 0.3, 0.4, 0.0]
    j = zeros(Int, N)
    bins = zeros(N)
    counts = zeros(Int, N)
    reference_kept = true
    for _ in 1:20000
        LLPF.conditional_resample!(we, j, bins)
        reference_kept &= j[N] == N
        for i in 1:N-1
            counts[j[i]] += 1
        end
    end
    @test reference_kept
    @test counts[N] == 0 # A particle with zero weight is never selected
    @test counts ./ sum(counts) ≈ we atol = 0.01
end

@testset "Conditional particle filter keeps the reference trajectory" begin
    for (name, f) in pairs(filters)
        N = num_particles(f)
        xref = reshape([SA[0.1*k] for k in 1:T], 1, T)
        xs = Array{particletype(f)}(undef, N, T)
        we = zeros(N, T)
        a = Matrix{Int}(undef, N, T)
        a[:, 1] .= 1:N
        LLPF.conditional_forward_trajectory!(xs, we, a, f, u, y, LLPF.parameters(f), xref)
        @test xs[N, :] == vec(xref)
        @test all(a[N, 2:end] .== N)
        @test all(sum(we, dims = 1) .≈ 1)
    end
end

@testset "FFBSi requires a transition density" begin
    @test_throws ArgumentError smooth(apf, 10, u, y)
    @test_throws ArgumentError smooth(AuxiliaryParticleFilter(apf), 10, u, y; method = FFBSi())
    # The default method for a filter with an additive dynamics density is unchanged
    xb, ll = smooth(pf, 100, u, y)
    @test size(xb) == (100, T)
    @test rmse(smean(xb), xT) < 0.03
end

@testset "ParticleGibbsSmoother" begin
    Random.seed!(2)
    for (name, f) in pairs(filters)
        xb, ll = smooth(f, 200, u, y; method = ParticleGibbsSmoother(burnin = 20))
        @test size(xb) == (200, T)
        @test isfinite(ll)
        @test rmse(smean(xb), xT) < 0.03
        @test mean(sstd(xb) ./ σT) ≈ 1 atol = 0.15
        @info "ParticleGibbsSmoother $name" rmse(smean(xb), xT) mean(sstd(xb) ./ σT)
    end
    # Reuse of a forward solution and thinning
    sol = forward_trajectory(apf, u, y)
    xb, ll = smooth(sol, 20; method = ParticleGibbsSmoother(burnin = 5, thin = 2))
    @test size(xb) == (20, T)
    @test ll == sol.ll
    @test_throws ArgumentError smooth(sol, 20; method = ParticleGibbsSmoother(thin = 0))
end

@testset "GenealogySmoother" begin
    Random.seed!(3)
    for (name, f) in pairs(filters)
        sol = forward_trajectory(f, u, y)
        # Full ancestral paths: few distinct ancestors far from the final time step
        xb, ll = smooth(sol, 200; method = GenealogySmoother())
        @test size(xb) == (200, T)
        @test ll == sol.ll
        @test xb[:, T] ⊆ sol.x[:, T]
        @test rmse(smean(xb), xT) < 0.15
        # Fixed lag
        xbl, _ = smooth(sol, 200; method = GenealogySmoother(lag = 10))
        @test rmse(smean(xbl), xT) < 0.04
        @test mean(sstd(xbl) ./ σT) ≈ 1 atol = 0.25
        # Lag zero gives samples of the filtering distribution
        xb0, _ = smooth(sol, 2000; method = GenealogySmoother(lag = 0))
        @test rmse(smean(xb0), vec(mean_trajectory(sol))) < 0.03
        @info "GenealogySmoother $name" rmse(smean(xb), xT) rmse(smean(xbl), xT)
    end
    @test_throws ArgumentError smooth(apf, 10, u, y; method = GenealogySmoother(lag = -1))
end

@testset "FFBSi with a transition density for non-additive noise" begin
    # The standard deviation of the dynamics noise depends on the state
    Random.seed!(4)
    σ(x) = 0.05 + 0.1*abs(x[1])
    dyn_sd(x, u, p, t, noise = false) = A*x + B*u + noise*σ(x)*randn(SVector{1})
    function transition_logpdf(x⁺, x, u, p, t)
        LLPF.extended_logpdf(LLPF.SimpleMvNormal(SA[σ(x)^2;;]), x⁺ - A*x - B*u)
    end
    f = AdvancedParticleFilter(1000, dyn_sd, meas, meas_ll, nothing, d0)
    xs, _, ys = LLPF.simulate(f, u)
    sol = forward_trajectory(f, u, ys)
    xb_ffbsi, _ = smooth(sol, 300; method = FFBSi(; transition_logpdf))
    xb_pg, _ = smooth(sol, 300; method = ParticleGibbsSmoother(burnin = 20))
    # The process noise is larger than the measurement noise for large |x|, the first-stage weights of the
    # auxiliary filter therefore include the process noise (evaluated approximately at the noise-free prediction)
    first_stage(x, u, y, p, t) = LLPF.extended_logpdf(LLPF.SimpleMvNormal(SA[0.2^2 + σ((x - B*u)/0.9)^2;;]), y - C*x)
    aux = AuxiliaryParticleFilter(f; first_stage)
    xb_aux, _ = smooth(aux, 300, u, ys; method = FFBSi(; transition_logpdf))
    xb_auxpg, _ = smooth(aux, 300, u, ys; method = ParticleGibbsSmoother(burnin = 20))
    @test rmse(smean(xb_ffbsi), smean(xb_pg)) < 0.04
    @test rmse(smean(xb_aux), smean(xb_pg)) < 0.04
    @test rmse(smean(xb_auxpg), smean(xb_pg)) < 0.04
    @test rmse(smean(xb_ffbsi), getindex.(xs, 1)) < rmse(vec(mean_trajectory(sol)), getindex.(xs, 1))
    @info "FFBSi with transition_logpdf" rmse(smean(xb_ffbsi), smean(xb_pg)) rmse(smean(xb_aux), smean(xb_pg)) rmse(smean(xb_auxpg), smean(xb_pg))
end

end # module
