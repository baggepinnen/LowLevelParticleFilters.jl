using LowLevelParticleFilters, StaticArrays, Distributions, RecursiveArrayTools, BenchmarkTools, StatsPlots

# Define problem

n = 2 # Dinemsion of state
m = 2 # Dinemsion of input
p = 2 # Dinemsion of measurements

const dg = MvNormal(p,1.0)          # Dynamics noise Distribution
const df = MvNormal(n,1.0)          # Measurement noise Distribution
const d0 = MvNormal(randn(n),2.0)   # Initial state Distribution

# Define random linenar state-space system
Tr = randn(n,n)
const A = SMatrix{n,n}(Tr*diagm(0=>LinRange(0.5,0.95,n))/Tr)
const B = @SMatrix randn(n,m)
const C = @SMatrix randn(p,n)

dynamics(x,u) = A*x .+ B*u
measurement(x) = C*x

function run_test()
    particle_count = [10, 20, 50, 100, 200, 500, 1000]
    time_steps = [20, 100, 200]
    RMSE = zeros(length(particle_count),length(time_steps)) # Store the RMS errors
    propagated_particles = 0
    t = @elapsed for (Ti,T) = enumerate(time_steps)
        for (Ni,N) = enumerate(particle_count)
            montecarlo_runs = 1*maximum(particle_count)*maximum(time_steps) ÷ T ÷ N
            E = sum(1:montecarlo_runs) do mc_run
                pf = ParticleFilter(N, dynamics, measurement, df, dg, d0)
                u = SVector{2,Float64}(randn(2))
                x = SVector{2,Float64}(rand(d0))
                y = SVector{2,Float64}(sample_measurement(pf,x,1))
                error = 0.0
                @inbounds for t = 1:T-1
                    pf(u, y) # Update the particle filter
                    x = dynamics(x,u)
                    y = SVector{2,Float64}(sample_measurement(pf,x,t))
                    u = @SVector randn(2)
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
    println("Propagated $propagated_particles particles in $t seconds for an average of $(propagated_particles/t/1000) particles per millisecond")
    #
    return RMSE
end

##
@time RMSE = run_test()


time_steps     = [20, 100, 200]
particle_count = [10, 20, 50, 100, 200, 500, 1000]
nT             = length(time_steps)
leg            = reshape(["$(time_steps[i]) time steps" for i = 1:nT], 1,:)
plot(particle_count,RMSE,xscale=:log10, ylabel="RMS errors", xlabel=" Number of particles", lab=leg)
gui()


##
N     = 2000 # Number of particles
T     = 200 # Number of time steps
M     = 100 # Number of smoothed backwards trajectories
pf    = ParticleFilter(N, dynamics, measurement, df, dg, d0)
du    = MvNormal(2,1) # Control input distribution
x,u,y = simulate(pf,T,du)
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
x,u,y = tosvec.((x,u,y))
##
xb,ll = smooth(pf, M, u, y)
xbm = smoothed_mean(xb)
xbc = smoothed_cov(xb)
xbt = smoothed_trajs(xb)
xbs = [diag(xbc) for xbc in xbc] |> vecvec_to_mat .|> sqrt
plot(xbm', ribbon=2xbs, lab="PF smooth")
plot!(vecvec_to_mat(x), l=:dash, lab="True")

plot(vecvec_to_mat(x), l=(4,), layout=(2,1), show=false)
scatter!(xbt[1,:,:]', subplot=1, show=false, m=(1,:black, 0.5))
scatter!(xbt[2,:,:]', subplot=2, m=(1,:black, 0.5))

##
svec = exp10.(LinRange(-0.7,1,40))
llspf = map(svec) do s
    df = MvNormal(n,s)
    pfs = ParticleFilter(N, dynamics, measurement, df, dg, d0)
    loglik(pfs,u,y)
end
plot(svec, llspf, yscale=:identity, xscale=:log10, ylabel="PF (blue)", title="Loglik")
##
eye(n) = Matrix{Float64}(I,n,n)
llskf = map(svec) do s
    kfs = KalmanFilter(A, B, I, 0, s^2*eye(n), eye(p), d0)
    loglik(kfs,u,y)
end
plot!(twinx(),svec, llskf, yscale=:identity, xscale=:log10, ylabel="Kalman (red)", c=:red)
##
# Same thing with KF
kf = KalmanFilter(A, B, I, 0, eye(n), eye(p), MvNormal(2,1))
xf,xh,R,Rt,ll = forward_trajectory(kf, u, y) # filtered, prediction, pred cov, filter cov, loglik
xT,R,lls = smooth(kf, u, x) # Smoothed state, smoothed cov, loglik

# Compare to kf
xpf,wpf,wepf,llpf = forward_trajectory(pf, u, y)
plot(vecvec_to_mat(xT), lab="Kalman smooth", layout=2)
plot!(xbm', lab="pf smooth")
plot!(vecvec_to_mat(x), lab="true")
##


filter_from_parameters(θ) = ParticleFilter(N, dynamics, measurement, MvNormal(n,θ[1]), MvNormal(p,θ[2]), d0)
priors = [Distributions.Gamma(1.5,1),Distributions.Gamma(1.5,1)]
ll       = log_likelihood_fun(filter_from_parameters,priors,u,y)
plot_priors(priors)
v = exp10.(LinRange(-1,1,8))
llxy = (x,y) -> ll([x;y])
VGx, VGy = meshgrid(v,v)
VGz = llxy.(VGx, VGy)
heatmap(VGz, xticks=(1:8,round.(v,digits=1)),yticks=(1:8,round.(v,digits=1)), xlabel="sigma v", ylabel="sigma w") # Yes, labels are reversed
##


#
# VGx, VGy = meshgrid(1:10,1:5)
# VGz = ((x,y) -> x*y).(VGx,VGy)
# heatmap(VGz, yticks=(1:10,1:10), xticks=(1:5,1:5))


##

N = 1000
filter_from_parameters(θ) = ParticleFilter(N, dynamics, measurement, MvNormal(n,exp(θ[1])), MvNormal(p,exp(θ[2])), d0)
priors = [Normal(1,2),Normal(1,2)]
ll       = log_likelihood_fun(filter_from_parameters,priors,u,y,1)
θ₀ = log.([1.,1.])
draw = θ -> θ .+ rand(MvNormal(0.1ones(2)))
burnin = 200
@time theta, lls = metropolis(ll, 1000, θ₀, draw)
thetam = reduce(hcat, theta)'
histogram(exp.(thetam), layout=(3,1)); plot!(lls, subplot=3)

# @time thetalls = LowLevelParticleFilters.metropolis_threaded(burnin, ll, 500, θ₀, draw)
# histogram(exp.(thetalls[:,1:2]), layout=3)
# plot!(thetalls[:,3], subplot=3)
