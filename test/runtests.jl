ENV["GKSwstype"] = 100 # workaround for gr segfault on GH actions
using LowLevelParticleFilters
import LowLevelParticleFilters.resample
using Test, Random, LinearAlgebra, Statistics, StaticArrays, Distributions, Plots
using MonteCarloMeasurements
gr(show=false)
Random.seed!(0)

mvnormal(d::Int, σ::Real) = MvNormal(LinearAlgebra.Diagonal(fill(float(σ) ^ 2, d)))
mvnormal(μ::AbstractVector{<:Real}, σ::Real) = MvNormal(μ, float(σ) ^ 2 * I)

@testset "LowLevelParticleFilters" begin
    @info "testing LowLevelParticleFilters"
    @testset "logsumexp" begin
        @info "testing logsumexp"
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
        @info "testing weigthed_mean"
        x = [randn(3) for i = 1:10000]
        w = ones(10000); we = similar(w)
        logsumexp!(w,we)
        @test sum(abs, weigthed_mean(x,we)) < 0.06
    end


    @testset "resample" begin
        @info "testing resample"
        s = PFstate(10)
        @test effective_particles(s.we) ≈ 10
        logsumexp!(s.w, s.we)
        @test resample(s.we) ≈ 1:10

        w = [1.,1,1,2,2,2,3,3,3]; we=similar(w)
        logsumexp!(w,we)
        @test we |> resample |> sum >= 56
        @test length(resample(we)) == length(we)
        for i = 1:10
            w = randn(100); we = randn(100)
            logsumexp!(w,we)
            j = resample(we)
            @test maximum(j) <= 100
            @test minimum(j) >= 1
        end
    end

    @testset "static distributions" begin
        @info "testing static distributions"
        x = @SVector ones(2)
        d = mvnormal(2,2)
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
        @info "testing End to end"
        eye(n) = Matrix{Float64}(I,n,n)

        # Define random linenar state-space system
        n = 2   # Dimension of state
        m = 1   # Dimension of input
        p = 1   # Dimension of measurements

        dg = mvnormal(p,1.0)          # Dynamics noise Distribution
        df = mvnormal(n,0.1)          # Measurement noise Distribution
        d0 = mvnormal(randn(n),2.0)   # Initial state Distribution
        
        # Define random linenar state-space system
        A_test = SA[0.97043   -0.097368
                     0.09736    0.970437]
        B_test = SA[0.1; 0;;]
        C_test = SA[0 1.0]

        dynamics(x,u,p,t) = A_test*x .+ B_test*u
        measurement(x,u,p,t) = C_test*x


        N     = 1000 # Number of particles
        T     = 200 # Number of time steps
        M     = 100 # Number of smoothed backwards trajectories
        pf    = ParticleFilter(N, dynamics, measurement, df, dg, d0)
        pfa   = AuxiliaryParticleFilter(N, dynamics, measurement, df, dg, d0)
        @test !shouldresample(pf)
        @test !shouldresample(pfa)
        du    = mvnormal(m,1) # Control input distribution
        xp,up,yp = LowLevelParticleFilters.simulate(pf,T,du,0,100)
        @test xp[1] isa MonteCarloMeasurements.Particles{Float64,100}
        @test size(xp) == (T,n)
        x,u,y = LowLevelParticleFilters.simulate(pf,T,du)

        sol = forward_trajectory(pf, u, y)
        parts = Particles(sol.x,sol.we)
        @test size(parts) == (T,n)
        @test length(parts[1].particles) == N

        xm = reduce(hcat,x)
        tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
        x,u,y = tosvec.((x,u,y))

        xpf = mean_trajectory(pf,u,y)[1]

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
        xbt = smoothed_trajs(xb)

        kf     = KalmanFilter(A_test, B_test, C_test, 0, 0.01eye(n), eye(p), d0)
        # x,u,y = simulate(kf,T,du)
        ksol = forward_trajectory(kf, u, y)
        plot(ksol)
        xT,R,lls = smooth(kf, u, y)

        @test mean(abs2, xm) > mean(abs2, xm - reduce(hcat,ksol.x)) > mean(abs2, xm - reduce(hcat,ksol.xt)) #> mean(abs2, xm - reduce(hcat,xT)) # Kalman: prediction > filtering > smoothing
        @test mean(abs2, xm) > mean(abs2, xm - reduce(hcat,xpf)) #> mean(abs2, xm - reduce(hcat,xT)) # particle filtering improves but not as good as kalman smoothing
        @test_skip mean(abs2, xm - reduce(hcat,xpf)) > mean(abs2, xm - xbm) # particle smoothing improves over filtering
        # NOTE: smoothing sometimes fails to improve so some tests are deactivated
        # plot(xm', layout=2)
        # plot!(reduce(hcat,xf)')
        # plot!(reduce(hcat,xt)')
        # plot!(reduce(hcat,xT)')



        svec = exp10.(LinRange(-1,2,22))
        llspf = map(svec) do s
            df = mvnormal(n,s)
            pfs = ParticleFilter(N, dynamics, measurement, df, dg, d0)
            loglik(pfs,u,y)
        end

        llspfa = map(svec) do s
            df = mvnormal(n,s)
            pfs = AuxiliaryParticleFilter(N, dynamics, measurement, df, dg, d0)
            loglik(pfs,u,y)
        end

        llskf = map(svec) do s
            kfs = KalmanFilter(A_test, B_test, C_test, 0, s^2*eye(n), eye(p), d0)
            loglik(kfs,u,y)
        end
        # plot(svec, [llspf llspfa llskf], xscale=:log10, lab=["PF" "APF" "KF"])

        @testset "Metropolis" begin
            @info "testing Metropolis"
            N = 1000
            function filter_from_parameters(θ,pf=nothing)
                pf === nothing && (return ParticleFilter(N, dynamics, measurement, mvnormal(n,exp(θ[1])), mvnormal(p,exp(θ[2])), d0))
                ParticleFilter(pf.state, dynamics, measurement, df,dg, d0)
            end
            # The call to `exp` on the parameters is so that we can define log-normal priors
            priors = [Normal(1,2),Normal(1,2)]
            ll     = log_likelihood_fun(filter_from_parameters,priors,u,y,0)
            θ₀ = log.([1.,1.]) # Starting point
            # We also need to define a function that suggests a new point from the "proposal distribution". This can be pretty much anything, but it has to be symmetric since I was lazy and simplified an equation.
            draw = θ -> θ .+ rand(MvNormal(Diagonal(0.1^2*ones(2))))
            burnin = 200
            theta, lls = metropolis(ll, 20, θ₀, draw)

        end

    end



    @testset "debugplot" begin
        @info "testing debugplot"
        eye(n) = Matrix{Float64}(I,n,n)
        n = 2 # Dinemsion of state
        m = 1 # Dinemsion of input
        p = 1 # Dinemsion of measurements

        dg = mvnormal(p,1.0)          # Dynamics noise Distribution
        df = mvnormal(n,0.1)          # Measurement noise Distribution
        d0 = mvnormal(randn(n),2.0)   # Initial state Distribution

        # Define random linenar state-space system
        Tr = randn(n,n)
        A = SMatrix{n,n}([0.99 0.1; 0 0.2])
        B = @SMatrix [0;1]
        C = @SMatrix [1 0]
        # C = SMatrix{p,n}([1 1])

        dynamics(x,u,p,t) = A*x .+ B*u
        measurement(x,u,p,t) = C*x


        N     = 100 # Number of particles
        T     = 3   # Number of time steps
        pf    = ParticleFilter(N, dynamics, measurement, df, dg, d0)
        pfa   = AuxiliaryParticleFilter(N, dynamics, measurement, df, dg, d0)
        du    = mvnormal(1,1) # Control input distribution
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
    @info "testing UKF"
    include("test_ukf.jl")
end




@testset "Advanced filters" begin
    @info "testing Advanced filters"

    eye(n) = Matrix{Float64}(I,n,n)
    n = 2 # Dinemsion of state
    m = 2 # Dinemsion of input
    p = 2 # Dinemsion of measurements

    dg = mvnormal(p,1.0)          # Dynamics noise Distribution
    df = mvnormal(n,0.1)          # Measurement noise Distribution
    d0 = mvnormal(randn(n),2.0)   # Initial state Distribution
    du = mvnormal(2,1)            # Control input distribution

    # Define random linenar state-space system
    A = SMatrix{n,n}([0.99 0.1; 0 0.2])
    B = @SMatrix randn(n,m)
    C = SMatrix{p,p}(eye(p))
    # C = SMatrix{p,n}([1 1])

    dynamics(x,u,p,t,noise=false) = A*x .+ B*u + noise*rand(df)
    measurement(x,u,p,t,noise=false) = C*x .+ noise*rand(dg)
    measurement_likelihood(x,u,y,p,t) = logpdf(dg, measurement(x,u,p,t)-y)
    tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy

    T     = 200 # Number of time steps
    N     = 500
    Random.seed!(0)
    apf = AdvancedParticleFilter(N, dynamics, measurement, measurement_likelihood, df, d0)
    x,u,y = LowLevelParticleFilters.simulate(apf,T,du) # Simuate trajectory using the model in the filter
    x,u,y = tosvec.((x,u,y))
    @time resapf,ll = mean_trajectory(apf, u, y)
    
    norm(mean(x))
    norm(mean(x .- resapf))
    
end

@testset "ekf" begin
    @info "testing ekf"
    include("test_ekf.jl")
end

@testset "parameters" begin
    @info "Testing parameters"
    include("test_parameters.jl")
end



end

@testset "example_quadtank" begin
    @info "Testing example_quadtank"
    include("../src/example_quadtank.jl")
end

@testset "example_lineargaussian" begin
    @info "testing example_lineargaussian"
    include("../src/example_lineargaussian.jl")
end