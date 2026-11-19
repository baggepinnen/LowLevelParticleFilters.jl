using LowLevelParticleFilters, Distributions, LinearAlgebra, Test
using LowLevelParticleFilters: log_likelihood_fun, metropolis, metropolis_threaded

A, B, C = [0.9;;], [1.0;;], [1.0;;]
d0 = LowLevelParticleFilters.SimpleMvNormal([0.0], [1.0;;])
kf = KalmanFilter(A, B, C, 0, [0.1;;], [0.1;;], d0)
_, u, y = simulate(kf, [randn(1) for _ in 1:50])
priors = [Normal(0, 2), Normal(0, 2)]

@testset "constructor with one argument" begin
    f1(θ) = KalmanFilter(A, B, C, 0, [exp(2θ[1]);;], [exp(2θ[2]);;], d0)
    ll = log_likelihood_fun(f1, priors, u, y)
    @test isfinite(ll([0.0, 0.0]))
    @test ll([0.0, 0.0]) == ll([0.0, 0.0]) # second call, the method f1(θ, previous) does not exist
end

@testset "exception during construction gives -Inf" begin
    function f2(θ, prev = nothing)
        θ[1] < 0 && throw(DomainError(θ[1], "the variance must be nonnegative"))
        KalmanFilter(A, B, C, 0, [θ[1];;], [exp(2θ[2]);;], d0)
    end
    ll = log_likelihood_fun(f2, priors, u, y)
    @test isfinite(ll([0.1, 0.0]))
    @test ll([-0.1, 0.0]) == -Inf
    f3(θ, prev = nothing) = undefined_function(θ)
    @test_throws UndefVarError log_likelihood_fun(f3, priors, u, y)([0.0, 0.0])
end

@testset "metropolis_threaded does not share the previous filter between chains" begin
    owners = Dict{UInt, Task}()
    lk = ReentrantLock()
    violations = Threads.Atomic{Int}(0)
    function f4(θ, prev = nothing)
        if prev !== nothing
            owner = lock(() -> get(owners, objectid(prev), nothing), lk)
            owner !== nothing && owner !== current_task() && Threads.atomic_add!(violations, 1)
        end
        kf = KalmanFilter(A, B, C, 0, [exp(2θ[1]);;], [exp(2θ[2]);;], d0)
        lock(() -> (owners[objectid(kf)] = current_task()), lk)
        kf
    end
    ll = log_likelihood_fun(f4, priors, u, y)
    res = metropolis_threaded(10, ll, 200, [0.0, 0.0], θ -> θ .+ 0.1 .* randn(2); nthreads = 4)
    @test size(res) == (4*190, 3)
    @test violations[] == 0
end
