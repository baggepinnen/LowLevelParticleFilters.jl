using LowLevelParticleFilters, TimerOutputs, StaticArrays, Distributions

n = 2
m = 2
p = 2

const pg = Distributions.MvNormal(p,1.0)
const pf = Distributions.MvNormal(n,1.0)
const p0 = Distributions.MvNormal(randn(n),2.0)

T = randn(n,n)
const A = SMatrix{n,n}(T*diagm(linspace(0.5,0.99,n))/T)
const B = @SMatrix randn(n,m)
const C = @SMatrix randn(p,n)

function linear_gaussian_f(x,xp,u,j)
    Bu = B*u
    @inbounds for i = eachindex(x)
        x[i] =  A*xp[j[i]] + Bu + rand(pf)
    end
    x
end
function linear_gaussian_f(x,u)
    Bu = B*u
    @inbounds for i = eachindex(x)
        x[i] =  A*x[i] .+ Bu .+ rand(pf)
    end
    x
end

function linear_gaussian_g(w,x,y)
    @inbounds for i = 1:length(w)
        w[i] += logpdf(pg, Vector(y-C*x[i]))
        w[i] = ifelse(w[i] < -1000, -1000, w[i])
    end
    w
end

function run_test()
    particle_count = [5, 10, 20, 50, 100, 200, 500, 1000, 10_000]
    time_steps = [20, 100, 200]
    RMSE = zeros(length(particle_count),length(time_steps)) # Store the RMS errors
    propagated_particles = 0
    for (Ti,T) in enumerate(time_steps)
        for (Ni, N) in enumerate(particle_count)
            montecarlo_runs = 2*maximum(particle_count)*maximum(time_steps) / T / N
            #             montecarlo_runs = 1

            E = sum(1:montecarlo_runs) do mc_run
                u = randn(m)
                x = rand(p0)
                y = C*x + rand(pg)

                pf = ParticleFilter(N, p0, linear_gaussian_f, linear_gaussian_g)
                error = 0.0
                @timeit "pf" @inbounds for t = 1:T-1
                    # plot_particles2(xh,w,y,x,t)
                    pf(u, y)
                    x .= linear_gaussian_f([x],u)[]
                    y = C*x .+ rand(pg)
                    randn!(u)
                    error += sum(abs2,x-weigthed_mean(pf))
                end # t
                √(error/T)
            end # MC
            RMSE[Ni,Ti] = E/montecarlo_runs
            propagated_particles += montecarlo_runs*N*T
            #     figure();plot([x xh])

            @show N
        end # N
        @show T
    end # T
    println("Propagated $propagated_particles particles")
    #
    return RMSE
end

# @enter pf!(zeros(4),zeros(4), ones(4), ones(4), ones(4), g, f)
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
