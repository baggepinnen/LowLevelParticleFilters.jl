using LowLevelParticleFilters, TimerOutputs, StaticArrays, Distributions, Optim, RecursiveArrayTools, BenchmarkTools

n = 2
m = 2
p = 2

const dg = Distributions.MvNormal(p,1.0)
const df = Distributions.MvNormal(n,1.0)
const d0 = Distributions.MvNormal(randn(n),2.0)

Tr = randn(n,n)
const A = SMatrix{n,n}(Tr*diagm(linspace(0.5,0.99,n))/Tr)
const B = @SMatrix randn(n,m)
const C = @SMatrix randn(p,n)


dynamics(x,u) = A*x .+ B*u

measurement(x) = C*x

function run_test()
    particle_count = Int[5, 10, 20, 50, 100, 200, 500, 1000, 10_000]
    time_steps = Int[20, 100, 200]
    RMSE = zeros(length(particle_count),length(time_steps)) # Store the RMS errors
    propagated_particles = 0
    Ti = 0
    for T = time_steps
        Ti += 1
        Ni = 0
        for N = particle_count
            Ni += 1
            montecarlo_runs = 2*maximum(particle_count)*maximum(time_steps) ÷ T ÷ N
            E = sum(1:montecarlo_runs) do mc_run
                pf = ParticleFilter(N, linear_gaussian_dynamics, linear_gaussian_measurement, df, dg, d0)
                u = randn(m)
                x = rand(d0)
                y = sample_measurement(pf,x)

                error = 0.0
                @timeit "pf" @inbounds for t = 1:T-1
                    pf(u, y)
                    x .= dynamics(x,u)
                    y .= sample_measurement(pf,x)
                    randn!(u)
                    error += sum(abs2,x-weigthed_mean(pf))
                end # t
                √(error/T)
            end # MC
            RMSE[Ni,Ti] = E/montecarlo_runs
            propagated_particles += montecarlo_runs*N*T
            # @show N
        end # N
        # @show T
    end # T
    println("Propagated $propagated_particles particles")
    #
    return RMSE
end

reset_timer!()
@time RMSE = run_test()

# Profile.print()
function plotting(RMSE)
    time_steps     = [20, 100, 200]
    particle_count = [5, 10, 20, 50, 100, 200, 500, 1000, 10_000]
    nT             = length(time_steps)
    leg            = reshape(["$(time_steps[i]) time steps" for i = 1:nT], 1,:)
    plot(particle_count,RMSE,xscale=:log10, ylabel="RMS errors", xlabel=" Number of particles", lab=leg)
    gui()
end

plotting(RMSE)

N     = 500
T     = 200
M     = 100
pf    = ParticleFilter(N, linear_gaussian_dynamics, linear_gaussian_measurement, df, dg, d0)
pf2   = ParticleFilter(N, linear_gaussian_dynamics, linear_gaussian_measurement, MvNormal(n,2), dg, d0)
x,u,y = simulate(pf,T,MvNormal(2,1))


xb  = particle_smooth(pf, M, u, y)
xbm = smoothed_mean(xb)
xbc = smoothed_cov(xb)
xbt = smoothed_trajs(xb)
xbs = [diag(xbc) for xbc in xbc] |> vecvec_to_mat .|> sqrt
plot(xbm', ribbon=2xbs)
plot!(vecvec_to_mat(x), l=:dash)

plot(vecvec_to_mat(x), l=(4,), layout=(2,1), show=false)
scatter!(xbt[1,:,:]', subplot=1, show=false)
scatter!(xbt[2,:,:]', subplot=2)



mc = 1

svec = logspace(-2,2,50)
lls = map(svec) do s
    pfs = ParticleFilter(N, linear_gaussian_dynamics, linear_gaussian_measurement, MvNormal(n,s), dg, d0)
    loglik(pfs,u,y)
end
plot(svec, -lls, yscale=:log10, xscale=:log10)




filter_from_parameters(θ) = ParticleFilter(N, linear_gaussian_dynamics, linear_gaussian_measurement, MvNormal(n,θ[1]), MvNormal(p,θ[2]), d0)
priors = [Distributions.Gamma(1,10),Distributions.Gamma(1,10)]
nll    = negative_log_likelihood_fun(filter_from_parameters,priors,u,y,1)
plot_priors(priors, xscale=:log10, yscale=:log10)

# plot(logspace(-2,2,100), nll, xscale=:log10, yscale=:log10)

res = optimize(nll, [2.,2], show_trace=true, iterations=50)
θ   = Optim.minimizer(res)
pfθ = filter_from_parameters(θ)

plot_trajectories(pf,y,x)

nll(θ)
