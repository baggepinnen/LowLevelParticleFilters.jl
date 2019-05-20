# # Smoothing the track of a moving beetle
# This is an example of smoothing the 2-dimensional trajectory of a moving beetle (or anything really). It spurred off of [this Discourse topic](https://discourse.julialang.org/t/smoothing-tracks-with-a-kalman-filter/24209?u=yakir12).
# In this system we will describe the coordinates, `x` and `y`, of the beetle as a function of its velocity, `vt`, and direction, `θt`:
# ```
# xt+1=xt+cos(θt)vt
# yt+1=yt+sin(θt)vt
# vt+1=vt+et
# θt+1=θt+wt
# where
# et∼N(0,σe),wt∼N(0,σw)
# ```
# # Define the problem
# In this example we will try to smooth a mock trajectory. In order to do that we'll first generate some data.
#
using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Distributions,  StatsPlots

N = 1000 # Number of particles in the particle filter
n = 4 # Dinemsion of state: we have speed and angle, so two
p = 2 # Dinemsion of measurements, we can measure the x and the y, so also two
@inline pos(s) = s[SVector(1,2)]
@inline vel(s) = s[3]
@inline ϕ(s) = s[4]

@inline function dynamics(s,u) # stepping forward
    v = exp(vel(s))
    SVector{4,Float64}((pos(s) + SVector(sincos(ϕ(s)))*v)..., 0.999vel(s), identity(ϕ(s)))
end

@inline measurement(s) = s[SVector(1,2)] # We observer the position coordinates with the measurement
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
vecvec_to_mat(x) = reduce(hcat, x)'
##

# pf = ParticleFilter(N, dynamics, measurement, df, dg, d0)
# s,u,y = LowLevelParticleFilters.simulate(pf, nsteps, du)
# s,u,y = tosvec.((s,u,y))
#
# plot(last.(pos.(s)), first.(pos.(s)))
#
# M = 500 # Number of smoothing trajectories, NOTE: if this is set higher, the result will be better at the expense of linear scaling of the computational cost.
# sb,ll = smooth(pf, M, u, y) # Sample smooting particles (b for backward-trajectory)
# sbm = smoothed_mean(sb)     # Calculate the mean of smoothing trajectories
# sbc = smoothed_cov(sb)      # And covariance
# sbt = smoothed_trajs(sb)    # Get smoothing trajectories
# sbs = [diag(sbc) for sbc in sbc] |> vecvec_to_mat .|> sqrt
# plot(sbm', ribbon=2sbs, lab="PF smooth")
# plot(sbm[1,:],sbm[2,:], lab="Smoothed trjectories")
#


##

using DataDeps, DelimitedFiles
register(DataDep("track", "one example track of a dung beetle", "https://s3.eu-central-1.amazonaws.com/vision-group-file-sharing/Data%20backup%20and%20storage/Yakir/track.csv", "60158980eaf665c2757c032e906fd8b4fcdacc9396d8f72257f5c78823bfd3ee"))
xyt = readdlm(datadep"track/track.csv")

##
# using Makie, AbstractPlotting
# get_track(I) = [(xyt[i,1], xyt[i,2]) for i in 1:I]
# m, M = extrema(xyt[:,1:2])
# m -= 10
# M += 10
# limits = FRect(m, m, M - m, M - m)
# sc = Scene(limits = limits, scale_plot = false)
# i = Node(2)
# lines!(sc, lift(get_track, i))
# T = size(xyt,1)
# for j in 2:T
#     sleep((xyt[j,3] - xyt[j-1,3])/100) # speed up the walk by 100
#     push!(i, j)
# end

##
dgσ = 1 # the deviation of the measurement noise distribution
dvσ = 0.3#0.8 # the deviation of the dynamics noise distribution
ϕσ = 1

N = 1000 # Number of particles in the particle filter
n = 4 # Dinemsion of state: we have speed and angle, so two
p = 2 # Dinemsion of measurements, we can measure the x and the y, so also two

dg = MvNormal(@SVector(zeros(p)), dgσ^2) # Measurement noise Distribution
df = MvNormal(@SVector(zeros(n)), [1e-1, 1e-1, dvσ^2, ϕσ^2]) # Dynamics noise Distribution NOTE: MvNormal wants a variance, not std
y = tosvec(collect(eachrow(xyt[:,1:2])))
d0 = MvNormal(SVector(y[1]..., log(0.5), atan((y[2]-y[1])...)), [3.,3,2,2])

u = zeros(length(y))
pf = AuxiliaryParticleFilter(N, dynamics, measurement, df, dg, d0)
T = length(y)
x,w,we,ll=forward_trajectory(pf,u[1:T],y[1:T])
@show ll
trajectorydensity(pf,x,we,y[1:T], markerstrokecolor=:auto, m=(2,0.5))
##
# xh,ll = mean_trajectory(pf,u,y)
# xh = vecvec_to_mat(xh)
xh = mean_trajectory(x,we)
#
plot(xh[:,1],xh[:,2], layout=4, c=:blue, lab="xh", legend=:bottomleft)
scatter!(exp.(vel.(x))'[:,1:20:end], m=(:black, 0.02, 1), subplot=3, lab="", ylims=(0,5))
scatter!(ϕ.(x)'[:,1:20:end], m=(:black, 0.02, 1), subplot=4, lab="")
Plots.plot!(xyt[:,1],xyt[:,2], subplot=1, c=:red, lab="y")
Plots.plot!(xh[:,1:2], subplot=2, lab=["xh" ""], c=:blue, legend=:bottomleft)
Plots.plot!(xyt[:,1:2], subplot=2, lab=["y" ""], c=:red)
Plots.plot!(exp.(xh[:,3]), subplot=3, lab="vel", legend=:topleft)
Plots.plot!(identity.(xh[:,4]), subplot=4, lab="angle", legend=:topleft)
##
M = 5 # Number of smoothing trajectories, NOTE: if this is set higher, the result will be better at the expense of linear scaling of the computational cost.
sb,ll = smooth(pf, M, u, y) # Sample smooting particles (b for backward-trajectory)
sbm = smoothed_mean(sb)     # Calculate the mean of smoothing trajectories
sbc = smoothed_cov(sb)      # And covariance
sbt = smoothed_trajs(sb)    # Get smoothing trajectories
sbs = [diag(sbc) for sbc in sbc] |> vecvec_to_mat .|> sqrt
##
Plots.plot!(sbm[1,:],sbm[2,:], subplot=1, lab="xs")
Plots.plot!(sbm'[:,1:2], subplot=2, lab=["xs" "ys"])
Plots.plot!(exp.(sbm'[:,3]), subplot=3, lab="vs")
Plots.plot!(identity.(sbm'[:,4]), subplot=4, lab="as")
##
Plots.plot!(sbt[1,:,:]', subplot=2, lab="", l=(:black,0.2))
Plots.plot!(sbt[2,:,:]', subplot=2, lab="", l=(:black,0.2))
Plots.plot!(exp.(sbt[3,:,:]'), subplot=3, lab="", l=(:blue,0.2))
Plots.plot!(identity.(sbt[4,:,:]'), subplot=4, lab="", l=(:orange,0.2))

## Parameter estimation
svec = exp10.(LinRange(-0.7,1.5,20))
llspf = map(svec) do s
    df = MvNormal(@SVector(zeros(n)), [1e-4, 1e-4, s, s])
    pfs = ParticleFilter(1000, dynamics, measurement, df, dg, d0)
    loglik(pfs,u,y) # Assuming we enter the measured data y here, u can be set to zero then
end
plot(svec, llspf, xscale=:log10, title="Log-likelihood", xlabel="Dynamics noise standard deviation", lab="")
vline!([svec[findmax(llspf)[2]]], l=(:dash,:blue), lab="maximum")




## PMMH

N = 500
d0 = MvNormal(SVector(y[1]..., log(0.5), atan((y[2]-y[1])...)), [3.,3,2,2])
function filter_from_parameters(θ, pf=nothing)
    dvσ, ϕσ, dgσ = exp.(θ)
    dg = MvNormal(@SVector(zeros(p)), dgσ^2)
    df = MvNormal(@SVector(zeros(n)), [1e-6, 1e-6, dvσ^2, ϕσ^2])
    pf === nothing && (return ParticleFilter(N, dynamics, measurement, df,dg, d0))
    ParticleFilter(pf.state, dynamics, measurement, df,dg, d0)
end

priors = [Normal(1,2),Normal(1,2),Normal(1,2)]
ll     = log_likelihood_fun(filter_from_parameters,priors,u,y)
θ₀ = log.([0.7,0.8,1.]) # Starting point

draw = θ -> θ .+ 0.01randn(3)
burnin = 40
@time theta, lls = metropolis(ll, 300, θ₀, draw) # Run PMMH for 2000  iterations, takes about half a minute on my laptop
thetam = reduce(hcat, theta)'#[burnin+1:end,:] # Build a matrix of the output (was vecofvec)
histogram(exp.(thetam), layout=4); plot!(lls, subplot=4) # Visualize



##
# debugplot(pf,u,y,runall=false)
commandplot(pf,u,y)
