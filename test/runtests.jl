using LowLevelParticleFilters
import LowLevelParticleFilters.resample
using Test, Random, LinearAlgebra, Statistics, StaticArrays, Distributions
Random.seed!(0)

@testset "LowLevelParticleFilters" begin
    @testset "logsumexp" begin
        w = randn(10); we = similar(w)
        ll = logsumexp!(w,we)
        @test sum(we) ≈ 1
        @test sum(exp.(w)) ≈ 1
        wc = copy(w)
        I = ones(10)
        logsumexp!(I,we)
        @test w ≈ wc.-log(sum(exp, wc))
        @test I ≈ fill(log(1/10),10)
        w = randn(10); we = similar(w)
        wc = copy(w)
        LowLevelParticleFilters.expnormalize!(we,w)
        @test sum(we) ≈ 1
        @test w ≈ wc atol=1e-15
        LowLevelParticleFilters.expnormalize!(w)
        @test sum(w) ≈ 1
    end

    @testset "weigthed_mean" begin
        x = [randn(3) for i = 1:10000]
        w = ones(10000); we = similar(w)
        logsumexp!(w,we)
        @test sum(abs, weigthed_mean(x,we)) < 0.06
    end


    @testset "resample" begin
        s = PFstate(10)
        @test effective_particles(s.we) ≈ 10
        logsumexp!(s.w, s.we)
        @test resample(s.we) ≈ 1:10

        w = [1.,1,1,2,2,2,3,3,3]; we=similar(w)
        logsumexp!(w,we)
        @test we |> resample |> sum >= 56
        @test length(resample(we)) == length(we)
        for i = 1:10000
            w = randn(100); we = randn(100)
            logsumexp!(w,we)
            j = resample(we)
            @test maximum(j) <= 100
            @test minimum(j) >= 1
        end
    end

    @testset "static distributions" begin
        x = @SVector ones(2)
        d = MvNormal(2,2)
        @test logpdf(d,x) == logpdf(d,Vector(x))
        d = Product([Normal(0,2), Normal(0,2)])
        @test logpdf(d,x) == logpdf(d,Vector(x))

        dt = LowLevelParticleFilters.TupleProduct((Normal(0,2), Normal(0,2)))
        @test logpdf(dt,x) == logpdf(dt,Vector(x)) == logpdf(d,x)
        @test_nowarn rand(dt)
        @test var(dt) == var(d)
        @test cov(dt) == cov(d)
        @test entropy(dt) == entropy(d)
        @test rand(dt) isa Vector{Float64}

    end


    @testset "End to end" begin
        eye(n) = Matrix{Float64}(I,n,n)
        n = 2 # Dinemsion of state
        m = 2 # Dinemsion of input
        p = 2 # Dinemsion of measurements

        dg = MvNormal(p,1.0)          # Dynamics noise Distribution
        df = MvNormal(n,0.1)          # Measurement noise Distribution
        d0 = MvNormal(randn(n),2.0)   # Initial state Distribution

        # Define random linenar state-space system
        Tr = randn(n,n)
        A = SMatrix{n,n}([0.99 0.1; 0 0.2])
        B = @SMatrix randn(n,m)
        C = SMatrix{p,p}(eye(p))
        # C = SMatrix{p,n}([1 1])

        dynamics(x,u) = A*x .+ B*u
        measurement(x) = C*x


        N     = 1000 # Number of particles
        T     = 200 # Number of time steps
        M     = 100 # Number of smoothed backwards trajectories
        pf    = ParticleFilter(N, dynamics, measurement, df, dg, d0)
        pfa   = AuxiliaryParticleFilter(N, dynamics, measurement, df, dg, d0)
        @test !shouldresample(pf)
        @test !shouldresample(pfa)
        du    = MvNormal(2,1) # Control input distribution
        x,u,y = LowLevelParticleFilters.simulate(pf,T,du) # Simuate trajectory using the model in the filter
        xm = reduce(hcat,x)
        tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
        x,u,y = tosvec.((x,u,y))

        xb,ll = smooth(pf, M, u, y) # Sample smooting trajectories
        @test size(xb) == (M,T)
        xbm = smoothed_mean(xb)     # Calculate the mean of smoothing trajectories
        @test mean(abs2, xm - xbm) < 5

        xb,ll = smooth(pfa, M, u, y)
        xbma = smoothed_mean(xb)
        @test mean(abs2, xm - xbma) < 5

        @show mean(abs2, xm - xbm)
        @show mean(abs2, xm - xbma)


        xbc = smoothed_cov(xb)      # And covariance
        @test all(tr(C) < 2 for C in xbc)
        maximum(tr(C) for C in xbc)
        xbt = smoothed_trajs(xb)    # Can't remember what this does

        kf     = KalmanFilter(A, B, C, 0, eye(n), eye(p), MvNormal(x[1], [1.,1.]))
        # x,u,y = simulate(kf,T,du)
        xf,xt,R,Rt,ll = forward_trajectory(kf, u, y)
        xT,R,lls = smooth(kf, u, y)

        @test 50 > mean(abs2, xm - reduce(hcat,xf)) > mean(abs2, xm - reduce(hcat,xt)) > mean(abs2, xm - reduce(hcat,xT))
        # plot(xm', layout=2)
        # plot!(reduce(hcat,xf)')
        # plot!(reduce(hcat,xt)')
        # plot!(reduce(hcat,xT)')



        svec = exp10.(LinRange(-1,2,22))
        llspf = map(svec) do s
            df = MvNormal(n,s)
            pfs = ParticleFilter(N, dynamics, measurement, df, dg, d0)
            loglik(pfs,u,y)
        end
        @test all(s < 0 for s in llspf)

        llspfa = map(svec) do s
            df = MvNormal(n,s)
            pfs = AuxiliaryParticleFilter(N, dynamics, measurement, df, dg, d0)
            loglik(pfs,u,y)
        end

        llskf = map(svec) do s
            kfs = KalmanFilter(A, B, C, 0, s^2*eye(n), eye(p), d0)
            loglik(kfs,u,y)
        end
        @test all(s < 0 for s in llskf)
        # plot(svec, [llspf llspfa llskf], xscale=:log10, lab=["PF" "APF" "KF"])

        @testset "Metropolis" begin
            N = 1000
            function filter_from_parameters(θ,pf=nothing)
                pf === nothing && (return ParticleFilter(N, dynamics, measurement, MvNormal(n,exp(θ[1])), MvNormal(p,exp(θ[2])), d0))
                ParticleFilter(pf.state, dynamics, measurement, df,dg, d0)
            end
            # The call to `exp` on the parameters is so that we can define log-normal priors
            priors = [Normal(1,2),Normal(1,2)]
            ll     = log_likelihood_fun(filter_from_parameters,priors,u,y)
            θ₀ = log.([1.,1.]) # Starting point
            # We also need to define a function that suggests a new point from the "proposal distribution". This can be pretty much anything, but it has to be symmetric since I was lazy and simplified an equation.
            draw = θ -> θ .+ rand(MvNormal(0.1ones(2)))
            burnin = 200
            theta, lls = metropolis(ll, 20, θ₀, draw)

        end

        @testset "example_lineargaussian" begin
            include("../src/example_lineargaussian.jl")
        end

    end



    @testset "debugplot" begin
        eye(n) = Matrix{Float64}(I,n,n)
        n = 2 # Dinemsion of state
        m = 1 # Dinemsion of input
        p = 1 # Dinemsion of measurements

        dg = MvNormal(p,1.0)          # Dynamics noise Distribution
        df = MvNormal(n,0.1)          # Measurement noise Distribution
        d0 = MvNormal(randn(n),2.0)   # Initial state Distribution

        # Define random linenar state-space system
        Tr = randn(n,n)
        A = SMatrix{n,n}([0.99 0.1; 0 0.2])
        B = @SMatrix [0;1]
        C = @SMatrix [1 0]
        # C = SMatrix{p,n}([1 1])

        dynamics(x,u) = A*x .+ B*u
        measurement(x) = C*x


        N     = 100 # Number of particles
        T     = 5   # Number of time steps
        pf    = ParticleFilter(N, dynamics, measurement, df, dg, d0)
        pfa   = AuxiliaryParticleFilter(N, dynamics, measurement, df, dg, d0)
        du    = MvNormal(1,1) # Control input distribution
        x,u,y = LowLevelParticleFilters.simulate(pf,T,du)
        ##

        debugplot(pf,u,y, runall=true, xreal=x, density=true)
        debugplot(pf,u,y, runall=true, xreal=x, density=false)
        debugplot(pf,u,y, runall=true, xreal=x, leftonly=false)
        debugplot(pf,u,y, runall=true, xreal=x, leftonly=false, density=false)
        # debugplot(pfa,x,u,y, runall=true)

        # commandplot(pf,x,u,y)
        # commandplot(pfa,x,u,y)
    end



@testset "UKF" begin

    eye(n) = Matrix{Float64}(I,n,n)
    n = 2 # Dinemsion of state
    m = 2 # Dinemsion of input
    p = 2 # Dinemsion of measurements

    dg = MvNormal(p,1.0)          # Dynamics noise Distribution
    df = MvNormal(n,0.1)          # Measurement noise Distribution
    d0 = MvNormal(randn(n),2.0)   # Initial state Distribution
    du    = MvNormal(2,1) # Control input distribution

    # Define random linenar state-space system
    Tr = randn(n,n)
    A = SMatrix{n,n}([0.99 0.1; 0 0.2])
    B = @SMatrix randn(n,m)
    C = SMatrix{p,p}(eye(p))
    # C = SMatrix{p,n}([1 1])

    dynamics(x,u) = A*x .+ B*u
    measurement(x) = C*x

    T     = 200 # Number of time steps
    kf   = KalmanFilter(A, B, C, 0, eye(n), eye(p), d0)
    ukf  = UnscentedKalmanFilter(dynamics, measurement, eye(n), eye(p), d0)
    x,u,y = LowLevelParticleFilters.simulate(kf,T,du) # Simuate trajectory using the model in the filter
    @test_nowarn LowLevelParticleFilters.simulate(ukf,T,du)
    tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
    x,u,y = tosvec.((x,u,y))


    reskf = forward_trajectory(kf, u, y) # filtered, prediction, pred
    resukf = forward_trajectory(ukf, u, y)

    norm(mean(x .- reskf[1]))
    norm(mean(x .- resukf[1]))

    norm(mean(x .- reskf[2]))
    norm(mean(x .- resukf[2]))
    @test norm(mean(x .- reskf[2])) < norm(mean(x .- reskf[1])) # Filtered should be better than prediction
    @test norm(mean(x .- resukf[2])) < norm(mean(x .- resukf[1]))
    @test norm(mean(x .- reskf[2])) ≈ norm(mean(x .- resukf[2])) atol=5e-2
    # @test norm(mean(x .- reskf[2])) < norm(mean(x .- resukf[2]))  # KF should be better than UKF
    # @test norm(mean(x .- reskf[1])) < norm(mean(x .- resukf[1]))  # KF should be better than UKF
    @test norm(mean(x .- reskf[2])) < 0.2
end

end
