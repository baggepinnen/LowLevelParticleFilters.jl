using LowLevelParticleFilters, StaticArrays, Distributions, Optim, RecursiveArrayTools, BenchmarkTools, StatPlots

# Define problem

n = 2 # Dinemsion of state
m = 2 # Dinemsion of input
p = 2 # Dinemsion of measurements

const dg = MvNormal(p,1.0)          # Dynamics noise Distribution
const df = MvNormal(n,1.0)          # Measurement noise Distribution
const d0 = MvNormal(randn(n),2.0)   # Initial state Distribution

# Define random lienar state-space system
Tr = randn(n,n)
const A = SMatrix{n,n}(Tr*diagm(linspace(0.5,0.95,n))/Tr)
const B = @SMatrix randn(n,m)
const C = @SMatrix randn(p,n)

dynamics(x,u) = A*x .+ B*u
measurement(x) = C*x

function run_test()
    particle_count = Int[20, 50, 100, 200, 500, 1000, 10_000]
    time_steps = Int[20, 100, 200]
    RMSE = zeros(length(particle_count),length(time_steps)) # Store the RMS errors
    propagated_particles = 0
    t = @elapsed for (Ti,T) = enumerate(time_steps)
        for (Ni,N) = enumerate(particle_count)
            montecarlo_runs = 2*maximum(particle_count)*maximum(time_steps) ÷ T ÷ N
            E = sum(1:montecarlo_runs) do mc_run
                pf = ParticleFilter(N, dynamics, measurement, df, dg, d0)
                u = randn(m)
                x = rand(d0)
                y = sample_measurement(pf,x)
                error = 0.0
                @inbounds for t = 1:T-1
                    pf(u, y) # Update the particle filter
                    x .= dynamics(x,u)
                    y .= sample_measurement(pf,x)
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


@time RMSE = run_test()


time_steps     = [20, 100, 200]
particle_count = [20, 50, 100, 200, 500, 1000, 10_000]
nT             = length(time_steps)
leg            = reshape(["$(time_steps[i]) time steps" for i = 1:nT], 1,:)
plot(particle_count,RMSE,xscale=:log10, ylabel="RMS errors", xlabel=" Number of particles", lab=leg)
gui()



N     = 1000 # Number of particles
T     = 200 # Number of time steps
M     = 100 # Number of smoothed backwards trajectories
pf    = ParticleFilter(N, dynamics, measurement, df, dg, d0)
du    = MvNormal(2,1) # Control input distribution
x,u,y = simulate(pf,T,du)

xb  = particle_smooth(pf, M, u, y)
xbm = smoothed_mean(xb)
xbc = smoothed_cov(xb)
xbt = smoothed_trajs(xb)
xbs = [diag(xbc) for xbc in xbc] |> vecvec_to_mat .|> sqrt
plot(xbm', ribbon=2xbs)
plot!(vecvec_to_mat(x), l=:dash)

plot(vecvec_to_mat(x), l=(4,), layout=(2,1), show=false)
scatter!(xbt[1,:,:]', subplot=1, show=false, m=(1,:black, 0.5))
scatter!(xbt[2,:,:]', subplot=2, m=(1,:black, 0.5))


svec = logspace(-2,2,50)
lls = map(svec) do s
    pfs = ParticleFilter(N, dynamics, measurement, MvNormal(n,s), dg, d0)
    loglik(pfs,u,y)
end
plot(svec, -lls, yscale=:log10, xscale=:log10)




filter_from_parameters(θ) = ParticleFilter(N, dynamics, measurement, MvNormal(n,exp(θ[1])), MvNormal(p,exp(θ[2])), d0)
priors = [Distributions.Gamma(1,1),Distributions.Gamma(1,1)]
averaging = 3
nll       = negative_log_likelihood_fun(filter_from_parameters,priors,u,y,averaging)
# plot_priors(priors, xscale=:log10, yscale=:log10)

# plot(logspace(-2,2,100), nll, xscale=:log10, yscale=:log10)

res = optimize(nll, log.([2., 2.]), show_trace=true, iterations=50)
@show res
θ   = Optim.minimizer(res) .|> exp
pfθ = filter_from_parameters(log.(θ))
