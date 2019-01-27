using LowLevelParticleFilters
import LowLevelParticleFilters.resample
using Test, Random
Random.seed!(0)


@testset "weigthed_mean" begin
    x = [randn(3) for i = 1:10000]
    w = ones(10000)
    logsumexp!(w)
    @test sum(abs, weigthed_mean(x,w)) < 0.05
end

# @inline logsumexp!(w) = w .-= log(sum(exp, w))

@testset "logsumexp" begin
    w = randn(10)
    logsumexp!(w)
    wc = copy(w)
    I = ones(10)
    logsumexp!(I)
    @test w ≈ wc.-log(sum(exp, wc))
    @test I ≈ fill(log(1/10),10)
end



@testset "resample" begin
    s = PFstate([zeros(10)],[zeros(10)],ones(10),zeros(Int,10),zeros(10), Ref(1))
    logsumexp!(s.w)
    @test resample(s.w) ≈ 1:10

    w = [1.,1,1,2,2,2,3,3,3]
    logsumexp!(w)
    @test w |> resample |> sum >= 56
    @test length(resample(w)) == length(w)
    for i = 1:10000
        w = randn(100)
        logsumexp!(w)
        j = resample(w)
        @test maximum(j) <= 100
        @test minimum(j) >= 1
    end
end


# testpf = ParticleFilter(100, p0, linear_gaussian_f, linear_gaussian_g)
# @btime resample(testpf.state)
# w = randn(1000)
# @btime LowLevelParticleFilters.logsumexp!($w) # ≈ 15 μs
# @code_warntype LowLevelParticleFilters.logsumexp!(w)
# @code_warntype LowLevelParticleFilters.resample(w, zeros(Int,length(w)), zeros(length(w)))



@testset "readme" begin

    using LowLevelParticleFilters, StaticArrays, Distributions, StatPlots, Random, LinearAlgebra

    n = 2   # Dinemsion of state
    m = 2   # Dinemsion of input
    p = 2   # Dinemsion of measurements
    N = 500 # Number of particles

    Random.seed!(1)
    df = MvNormal(n,1.0)          # Dynamics noise Distribution
    dg = MvNormal(p,1.0)          # Measurement noise Distribution
    d0 = MvNormal(randn(n),2.0)   # Initial state Distribution

    # Define random linenar state-space system x' = Ax + Bu; y = Cx
    Tr = randn(n,n)
    A = SMatrix{n,n}(Tr*diagm(0=>range(0.5, stop=0.99, length=n))/Tr)
    B = @SMatrix randn(n,m)
    C = @SMatrix randn(p,n)

    dynamics(x,u)  = A*x .+ B*u
    measurement(x) = C*x
    pf = ParticleFilter(N, dynamics, measurement, df, dg, d0)

    function run_test()
        particle_count = Int[20, 500]
        time_steps = Int[20, 200]
        RMSE = zeros(length(particle_count),length(time_steps)) # Store the RMS errors
        propagated_particles = 0
        t = @elapsed for (Ti,T) = enumerate(time_steps)
            for (Ni,N) = enumerate(particle_count)
                montecarlo_runs = 4*maximum(particle_count)*maximum(time_steps) ÷ T ÷ N
                Random.seed!(0)
                E = sum(1:montecarlo_runs) do mc_run
                    pf = ParticleFilter(N, dynamics, measurement, df, dg, d0)
                    u = randn(m)
                    x = rand(d0)
                    y = sample_measurement(pf,x,1)
                    error = 0.0
                    @inbounds for t = 1:T-1
                        pf(u, y) # Update the particle filter
                        x .= dynamics(x,u)
                        y .= sample_measurement(pf,x,t)
                        randn!(u)
                        error += sum(abs2,x-weigthed_mean(pf))
                    end # t
                    √(error/T)
                end # MC
                RMSE[Ni,Ti] = E/montecarlo_runs
                propagated_particles += montecarlo_runs*N*T
                @show N
            end # N
            @show T
        end # T
        println("Propagated $propagated_particles particles in $t seconds for an average of $(propagated_particles/t) particles per second")
        #
        return RMSE
    end
    RMSE = run_test()
    @test sum(RMSE) < 3.5*length(RMSE)
    # Plot results
    time_steps     = [20, 200]
    particle_count = [20, 500]
    nT             = length(time_steps)
    leg            = reshape(["$(time_steps[i]) time steps" for i = 1:nT], 1,:)
    plot(particle_count,RMSE,xscale=:log10, ylabel="RMS errors", xlabel=" Number of particles", lab=leg)


    eye(n) = Matrix{Float64}(I,n,n)
    kf = KalmanFilter(A, B, I, 0I, eye(n), eye(p))
    T     = 200 # Number of time steps
    du    = MvNormal(2,1) # Control input distribution
    x,u,y = LowLevelParticleFilters.simulate(kf,T,du)


    xf,xt,R,Rt,ll = forward_trajectory(kf, u, y) # filtered, prediction, pred cov, filter cov, loglik
    @test sum(x->abs.(x), x.-xf) < sum(x->abs.(x), x)
    xT,R,lls = smooth(kf, u, xf) # Smoothed state, smoothed cov, loglik
    @test sum(x->abs.(x), x.-xT) < sum(x->abs.(x), x)

    N     = 500 # Number of particles
    T     = 200 # Number of time steps
    M     = 100 # Number of smoothed backwards trajectories
    pf    = ParticleFilter(N, dynamics, measurement, df, dg, d0)
    du    = MvNormal(2,1) # Control input distribution
    x,u,y = LowLevelParticleFilters.simulate(pf,T,du)

    Random.seed!(1)
    xb  = smooth(pf, M, u, y)[1]
    xbm = smoothed_mean(xb)
    @test sum(x->abs.(x), reduce(hcat,x).-xbm) < sum(sum(x->abs.(x), x))
    xbc = smoothed_cov(xb)
    xbt = smoothed_trajs(xb)
    xbs = hcat([diag(xbc) for xbc in xbc]...)' .|> sqrt
    svec = exp10.(range(-2, stop=2, length=50))
    lls = map(svec) do s
        pfs = ParticleFilter(N, dynamics, measurement, MvNormal(n,s), dg, d0)
        loglik(pfs,u,y)
    end
    filter_from_parameters(θ) = ParticleFilter(N, dynamics, measurement, MvNormal(n,θ[1]), MvNormal(p,θ[2]), d0)
    priors = [Distributions.Gamma(1,10),Distributions.Gamma(1,10)]
    # plot_priors(priors, xscale=:log10, yscale=:log10)
    averaging = 3
    ll = log_likelihood_fun(filter_from_parameters,priors,u,y,averaging)


end
