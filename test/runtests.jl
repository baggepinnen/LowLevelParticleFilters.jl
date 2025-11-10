ENV["GKSwstype"] = 100 # workaround for gr segfault on GH actions
using LowLevelParticleFilters
import LowLevelParticleFilters.resample
using Test, Random, LinearAlgebra, Statistics, StaticArrays, Distributions, Plots
using MonteCarloMeasurements
const LLPF = LowLevelParticleFilters
gr(show=false)
Random.seed!(0)

mvnormal(d::Int, σ::Real) = MvNormal(LinearAlgebra.Diagonal(fill(float(σ) ^ 2, d)))
mvnormal(μ::AbstractVector{<:Real}, σ::Real) = MvNormal(μ, float(σ) ^ 2 * I)

S = [1 0.2; 0.2 2]
d = LowLevelParticleFilters.SimpleMvNormal(randn(2), S)
@test cov(d) == S
r = [rand(d) for i = 1:10000]
r = reduce(hcat, r)
@test cov(r, dims=2) ≈ S atol = 0.1

out = zeros(2, 10000)
@test_nowarn rand!(Random.default_rng(), d, out)
@test cov(out, dims=2) ≈ S atol = 0.1

@test LowLevelParticleFilters.printarray([1,2]) == "[1, 2]"
@test LowLevelParticleFilters.printarray([1 2; 3 4]) == "[1 2; 3 4]"

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

    @testset "weighted_mean" begin
        @info "testing weighted_mean"
        x = [randn(3) for i = 1:10000]
        w = ones(10000); we = similar(w)
        logsumexp!(w,we)
        @test sum(abs, weighted_mean(x,we)) < 0.06
    end


    @testset "weighted_quantile" begin
        @info "testing weighted_quantile"

        # Test case 1: Simple example with known quantiles
        x = [randn(3) for i = 1:100]
        w = ones(100); we = similar(w)
        logsumexp!(w, we)
        q = 0.5
        result = weighted_quantile(x, we, q)[]
        expected = [quantile(getindex.(x, i), LLPF.ProbabilityWeights(we), q) for i = 1:length(x[1])]
        @test result ≈ expected
        @test result ≈ [median(getindex.(x, i)) for i = 1:length(x[1])]

        # Test case 2: Different weights
        w = rand(100); we = similar(w)
        logsumexp!(w, we)
        q = 0.75
        result = weighted_quantile(x, we, q)[]
        expected = [quantile(getindex.(x, i), LLPF.ProbabilityWeights(we), q) for i = 1:length(x[1])]
        @test result ≈ expected

        # Test case 3: Edge case with all weights zero except one
        we = zeros(100); we[1] = 1
        q = 0.5
        result = weighted_quantile(x, we, q)[]
        expected = [quantile(getindex.(x, i), LLPF.ProbabilityWeights(we), q) for i = 1:length(x[1])]
        @test result ≈ expected
        @test result ≈ x[1]
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
        d = product_distribution([Normal(0,2), Normal(0,2)])
        @test logpdf(d,x) == logpdf(d,Vector(x))

        dt = LowLevelParticleFilters.TupleProduct((Normal(0,2), Normal(0,2)))
        @test dt == LowLevelParticleFilters.TupleProduct(Normal(0,2), Normal(0,2))
        @test logpdf(dt,x) == logpdf(dt,Vector(x)) == logpdf(d,x)
        @test_nowarn rand(dt)
        @test rand(dt) isa SVector{2, Float64}
        @test var(dt) == var(d)
        @test cov(dt) == cov(d)
        @test extrema(dt) == ((-Inf, -Inf), (Inf, Inf))
        @test entropy(dt) == entropy(d)

        out = zeros(2, 10000)
        rand!(Random.default_rng(), dt, out)
        @test cov(out, dims=2) ≈ cov(dt) atol = 0.5


    end


    @testset "rk4" begin
        @info "Testing rk4"
        fun = (x,u,p,t)->[-1]
        dfun = LowLevelParticleFilters.rk4(fun, 1)
        @test dfun([1],1,1,1) ≈ [0]
        @test dfun([0],1,1,1) ≈ [-1]
    end

    @testset "n_integrator_covariance" begin
        @info "Testing n_integrator_covariance"
        using ControlSystemsBase
        
        Ts = 0.1
        σ2 = 1.0
        
        # Test n=2 
        P = c2d(ss(1/tf('s'))^2 * sqrt(σ2), Ts)
        R_expected = P.B * P.B'
        R_n2 = LowLevelParticleFilters.n_integrator_covariance(2, Ts, σ2)
        @test R_n2 ≈ R_expected
        
        # Test n=3
        P3 = c2d(ss(1/tf('s'))^3 * sqrt(σ2), Ts)
        R_expected3 = P3.B * P3.B'
        R_n3 = LowLevelParticleFilters.n_integrator_covariance(3, Ts, σ2)
        @test R_n3 ≈ R_expected3
        
        # Test n=4 with different σ2
        σ2_2 = 2.0
        P4 = c2d(ss(1/tf('s'))^4 * sqrt(σ2_2), Ts)
        R_expected4 = P4.B * P4.B'
        R_n4 = LowLevelParticleFilters.n_integrator_covariance(4, Ts, σ2_2)
        @test R_n4 ≈ R_expected4
    end

    @testset "n_integrator_covariance_smooth" begin
        @info "Testing n_integrator_covariance_smooth"
        using ControlSystemsBase
        
        Ts = 0.1
        σ2 = 1.0
        
        # Test n=2
        sys2 = ss(1/tf('s'))^2
        R_expected2 = c2d(sys2, diagm([0, σ2]), Ts)
        R_n2 = LowLevelParticleFilters.n_integrator_covariance_smooth(2, Ts, σ2)
        @test R_n2 ≈ R_expected2
        
        # Test n=3 with Val
        sys3 = ss(1/tf('s'))^3
        R_expected3 = c2d(sys3, diagm([0, 0, σ2]), Ts)
        R_n3 = LowLevelParticleFilters.n_integrator_covariance_smooth(Val(3), Ts, σ2)
        @test R_n3 ≈ R_expected3
        
        # Test n=4 with different σ2
        σ2_2 = 2.0
        sys4 = ss(1/tf('s'))^4
        R_expected4 = c2d(sys4, diagm([0, 0, 0, σ2_2]), Ts)
        R_n4 = LowLevelParticleFilters.n_integrator_covariance_smooth(4, Ts, σ2_2)
        @test R_n4 ≈ R_expected4
    end


    @testset "End to end" begin
        @info "testing End to end"
        eye(n) = Matrix{Float64}(I,n,n)
        Random.seed!(0)

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
        plot(sol)
        plot(sol, q=[0.5])
        plot(sol, q=0.5)
        parts = Particles(sol.x,sol.we)
        @test size(parts) == (T,n)
        @test length(parts[1].particles) == N
        WM = weighted_mean(sol.x, sol.we)
        @test WM == weighted_mean(sol)
        @test length(WM) == T
        @test WM[1] ≈ weighted_mean(sol.x[:, 1], sol.we[:, 1])

        WQ1 = weighted_quantile(sol, 0.1)
        WQ9 = weighted_quantile(sol, 0.9)

        @test all(all(WM[i] .< WQ9[i]) for i in eachindex(WM))
        @test all(all(WM[i] .> WQ1[i]) for i in eachindex(WM))

        C = weighted_cov(sol)
        C2 = zero(C[2])
        for (i, w) in enumerate(sol.we[:, 2])
            d = sol.x[i, 2] .- WM[2]
            C2 .+= w * d*d'
        end
        @test C2*N/(N-1) ≈ C[2] # C is normalized by N-1 but the weights are normalized by N

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
        @test :t ∈ propertynames(ksol)
        @test length(ksol.t) == T
        xT,R,lls = smooth(kf, u, y)
        ssol,_,lt,lh,r = LowLevelParticleFilters.smooth_mbf(ksol, kf)
        xTmbf,Rmbf = ssol.xT, ssol.RT
        @test sum(abs, sum(xTmbf .- xT)) < 1e-10
        @test sum(abs, sum(Rmbf .- R)) < 1e-10

        @test_skip mean(abs2, xm) > mean(abs2, xm - reduce(hcat,ksol.x)) > mean(abs2, xm - reduce(hcat,ksol.xt)) #> mean(abs2, xm - reduce(hcat,xT)) # Kalman: prediction > filtering > smoothing
        @test_skip mean(abs2, xm) > mean(abs2, xm - reduce(hcat,xpf)) #> mean(abs2, xm - reduce(hcat,xT)) # particle filtering improves but not as good as kalman smoothing
        @test_skip mean(abs2, xm - reduce(hcat,xpf)) > mean(abs2, xm - xbm) # particle smoothing improves over filtering
        # NOTE: smoothing sometimes fails to improve so some tests are deactivated
        # plot(xm', layout=2)

        w = [x[i+1] - (A_test*x[i] + B_test*u[i]) for i = 1:T-1]
        w_pred = [ksol.x[i+1] - (A_test*ksol.x[i] + B_test*u[i]) for i = 1:T-1]
        w_filt = [ksol.xt[i+1] - (A_test*ksol.xt[i] + B_test*u[i]) for i = 1:T-1]
        w_pf = [ksol.xt[i] - ksol.x[i] for i = 1:T-1] # This is almost the same as w_filt
        w_smooth = [xT[i+1] - (A_test*xT[i] + B_test*u[i]) for i = 1:T-1]

        w_smooth_mbf = [xTmbf[i+1] - (A_test*xTmbf[i] + B_test*u[i]) for i = 1:T-1]
        e = [y[i] - C_test*x[i] for i = 1:T]
        e_smooth = [y[i] - C_test*xT[i] for i = 1:T]

        @test cov(w) ≈ 0.01*eye(n) atol=0.004
        @test cov(e) ≈ eye(1) rtol=0.3
        @test cov(e_smooth) ≈ eye(1) rtol=0.3


        # plot(reduce(hcat, w)', layout=2, lab="w")
        # plot!(reduce(hcat, w_pred)', lab="wp")
        # plot!(reduce(hcat, w_filt)', lab="wf")
        # plot!((kf.R1*reduce(hcat, r))', lab="r")
        # # plot!(reduce(hcat, w_pf)', lab="wpf")
        # plot!(reduce(hcat, w_smooth)', lab="ws")

        # plot(reduce(hcat, e)', lab="e")
        # plot!(reduce(hcat, e_smooth)', lab="ê")


        sqkf   = SqKalmanFilter(A_test, B_test, C_test, 0, cholesky(0.01eye(n)).U, eye(p), d0)
        sqksol = forward_trajectory(sqkf, u, y)
        @test ksol.x ≈ sqksol.x
        @test ksol.xt ≈ sqksol.xt
        @test ksol.R ≈ sqksol.R
        @test ksol.Rt ≈ sqksol.Rt

        @test_nowarn simulate(sqkf, T, du)

        @testset "Diagonal static covariance" begin
            R1_diag = Diagonal(SVector(0.01, 0.01))
            kf_diag = KalmanFilter(A_test, B_test, C_test, 0, R1_diag, eye(p), d0)
            x_d, u_d, y_d = LowLevelParticleFilters.simulate(kf_diag, 50, du)
            sol_diag = forward_trajectory(kf_diag, u_d, y_d)
            @test length(sol_diag.x) == 50
            @test sol_diag.R[1] isa SMatrix{2, 2, Float64, 4}
        end

        kf   = KalmanFilter(A_test, B_test, C_test, 0, 0.01eye(n), eye(p), d0)
        x,u,y = LowLevelParticleFilters.simulate(kf,2000,du)

        svec = exp10.(LinRange(-2,0,11))
        llspf = map(svec) do s
            df = MvNormal(@SVector(zeros(n)),s^2*eye(n))
            pfs = ParticleFilter(N, dynamics, measurement, df, dg, d0)
            loglik(pfs,u,y)
        end

        llspfa = map(svec) do s
            df = MvNormal(@SVector(zeros(n)),s^2*eye(n))
            pfs = AuxiliaryParticleFilter(N, dynamics, measurement, df, dg, d0)
            loglik(pfs,u,y)
        end

        llskf = map(svec) do s
            kfs = KalmanFilter(A_test, B_test, C_test, 0, s^2*eye(n), eye(p), d0)
            loglik(kfs,u,y)
        end
        llskfx = map(svec) do s # Kalman filter with known state sequence, possible when data is simulated
            kfs = KalmanFilter(A_test, B_test, C_test, 0, s^2*eye(n), eye(p), d0)
            loglik_x(kfs, u, y, x)
        end
        plot(svec, [llspf llspfa llskf llskfx], xscale=:log10, lab=["PF" "APF" "KF" "KF known x"])
        vline!([0.1])

        m,mi = findmax(llspf)
        @test 5 ≤ mi ≤ 7

        m,mi = findmax(llspfa)
        @test 5 ≤ mi ≤ 7

        m,mi = findmax(llskf)
        @test 5 ≤ mi ≤ 7

        m,mi = findmax(llskfx)
        @test 5 ≤ mi ≤ 7

        @test maximum(abs, llskf .- llspf) < 20
        @test maximum(abs, llskf .- llspfa) < 20
        @test maximum(llskfx) > maximum(llskf)

        @testset "Metropolis" begin
            @info "testing Metropolis"
            N = 200 # A small number of particles for testing
            function filter_from_parameters(θ,pf=nothing)
                pf === nothing && (return ParticleFilter(N, dynamics, measurement, mvnormal(n,exp(θ[1])), mvnormal(p,exp(θ[2])), d0))
                ParticleFilter(pf.state, dynamics, measurement, df,dg, d0)
            end
            # The call to `exp` on the parameters is so that we can define log-normal priors
            priors = [Normal(1,2),Normal(1,2)]
            ll     = log_likelihood_fun(filter_from_parameters,priors,u,y,0)
            θ₀ = log.([1.1,1.1]) # Starting point
            # We also need to define a function that suggests a new point from the "proposal distribution". This can be pretty much anything, but it has to be symmetric since I was lazy and simplified an equation.
            draw = θ -> θ .+ rand(MvNormal(Diagonal(0.1^2*ones(2))))
            theta, lls = metropolis(ll, 20, θ₀, draw)
            theta, lls = metropolis(ll, 20, θ₀) # Default draw
            thetalls = LowLevelParticleFilters.metropolis_threaded(2, ll, 22, θ₀, nthreads=2) # Default draw
            @test size(thetalls) == (2*20,3) # 2 parameters + ll
        end

        @testset "prediction_errors" begin
            @info "Testing prediction_errors"
            res1 = zeros(length(y[1])*length(y))
            LowLevelParticleFilters.prediction_errors!(res1, sqkf, u, y)

            res2 = zeros(length(y[1])*length(y))
            LowLevelParticleFilters.prediction_errors!(res2, kf, u, y)

            @test res1 ≈ res2

            res3 = zeros((length(y[1])+1)*length(y))
            LowLevelParticleFilters.prediction_errors!(res3, kf, u, y, loglik=true)
            @test loglik(kf, u, y) ≈ -res3'res3 rtol=1e-6
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


@testset "jet" begin
    @info "Testing jet"
    include("test_jet.jl")
end

@testset "large" begin
    @info "Testing large"
    include("test_large.jl")
end

@testset "diff" begin
    @info "Testing diff"
    include("test_diff.jl")
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
    Random.seed!(0)
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
    apf = AdvancedParticleFilter(N, dynamics, measurement, measurement_likelihood, df, d0, threads=false)
    x,u,y = LowLevelParticleFilters.simulate(apf,T,du) # Simuate trajectory using the model in the filter
    x,u,y = tosvec.((x,u,y))
    @time resapf,ll = mean_trajectory(apf, u, y)
    
    norm(mean(x))
    @test norm(mean(x .- resapf)) < 5

    # With threads
    Random.seed!(0)
    apf = AdvancedParticleFilter(N, dynamics, measurement, measurement_likelihood, df, d0, threads=true)
    x,u,y = LowLevelParticleFilters.simulate(apf,T,du) # Simuate trajectory using the model in the filter
    x,u,y = tosvec.((x,u,y))
    @time resapf,ll = mean_trajectory(apf, u, y)
    
    norm(mean(x))
    @test norm(mean(x .- resapf)) < 5
    
end

@testset "ekf" begin
    @info "testing ekf"
    include("test_ekf.jl")
end

@testset "iekf" begin
    @info "testing iekf"
    include("test_iekf.jl")
end

@testset "imm" begin
    @info "Testing imm"
    include("test_imm.jl")
end

@testset "parameters" begin
    @info "Testing parameters"
    include("test_parameters.jl")
end

@testset "measurement_models" begin
    @info "Testing measurement_models"
    include("test_measurement_models.jl")
end

@testset "rbpf" begin
    @info "Testing rbpf"
    include("test_rbpf.jl")
end

@testset "mukf" begin
    @info "Testing mukf"
    include("test_mukf.jl")
end

@testset "uikf" begin
    @info "Testing mukf"
    include("test_uikf.jl")
end

@testset "function_versions" begin
    @info "Testing function_versions"
    include("test_function_versions.jl")
end

@testset "constraint_handling" begin
    @info "Testing constraint handling"
    include("test_constraint_handling.jl")
end

@testset "indexing_matrix" begin
    @info "Testing indexing_matrix"
    include("test_indexing_matrix.jl")
end

@testset "autotune_covariances" begin
    @info "Testing autotune_covariances"
    include("test_autotune_covariances.jl")
end


end

@testset "example_quadtank" begin
    @info "Testing example_quadtank"
    include("../examples/example_quadtank.jl")
end

@testset "example_lineargaussian" begin
    @info "testing example_lineargaussian"
    include("../examples/example_lineargaussian.jl")
end

@testset "example_beetle" begin
    @info "testing example_beetle"
    include("../examples/example_beetle.jl")
end

@testset "IEKF benchmarks" begin
    @info "testing example_beetle"
    include("../examples/example_nonlinear_kf.jl")
end