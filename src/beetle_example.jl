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

dgσ = 0.2 # the deviation of the measurement noise distribution
dfσ = 0.3 # the deviation of the dynamics noise distribution

N = 1000 # Number of particles in the particle filter
n = 4 # Dinemsion of state: we have speed and angle, so two
p = 2 # Dinemsion of measurements, we can measure the x and the y, so also two

dg = MvNormal(@SVector(zeros(p)), dgσ^2) # Measurement noise Distribution
df = MvNormal(@SVector(zeros(n)), [1e-4, 1e-4, dfσ, dfσ]) # Dynamics noise Distribution NOTE: MvNormal wants a variance, not std
d0 = MvNormal(SVector(0.0, 0.0, 0.6, 0.7), 0.2^2)   # Initial state Distribution, represents your uncretainty regarding where the beetle starts in any given video
du = df # We set the driving noise to equal the state-drift noise for now, it will be unused since there is not control input to the system

nsteps = 300 # how many steps will we have to smooth

pos(s) = s[1:2]
vel(s) = s[3]
ϕ(s) = s[4]

function dynamics(s,u) # stepping forward
    [pos(s) + SVector(sincos(ϕ(s)))*vel(s); 0.999vel(s); ϕ(s)]
end

measurement(s) = s[1:2] # We observer the position coordinates with the measurement

pf = ParticleFilter(N, dynamics, measurement, df, dg, d0)
s,u,y = LowLevelParticleFilters.simulate(pf, nsteps, du)
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
s,u,y = tosvec.((s,u,y))

plot(last.(pos.(s)), first.(pos.(s)))

vecvec_to_mat(x) = reduce(hcat, x)'
M = 500 # Number of smoothing trajectories, NOTE: if this is set higher, the result will be better at the expense of linear scaling of the computational cost.
sb,ll = smooth(pf, M, u, y) # Sample smooting particles (b for backward-trajectory)
sbm = smoothed_mean(sb)     # Calculate the mean of smoothing trajectories
sbc = smoothed_cov(sb)      # And covariance
sbt = smoothed_trajs(sb)    # Get smoothing trajectories
sbs = [diag(sbc) for sbc in sbc] |> vecvec_to_mat .|> sqrt
plot(sbm', ribbon=2sbs, lab="PF smooth")
plot(sbm[1,:],sbm[2,:], lab="Smoothed trjectories")



# Parameter estimation
svec = exp10.(LinRange(-1.5,1,60))
llspf = map(svec) do s
    df = MvNormal(@SVector(zeros(n)), [1e-4, 1e-4, s, s])
    pfs = ParticleFilter(2000, dynamics, measurement, df, dg, d0)
    loglik(pfs,u,y) # Assuming we enter the measured data y here, u can be set to zero then
end
plot(svec, llspf, xscale=:log10, title="Log-likelihood", xlabel="Dynamics noise standard deviation", lab="")
vline!([svec[findmax(llspf)[2]]], l=(:dash,:blue), lab="maximum")



using DataDeps, DelimitedFiles
register(DataDep("track", "one example track of a dung beetle", "https://s3.eu-central-1.amazonaws.com/vision-group-file-sharing/Data%20backup%20and%20storage/Yakir/track.csv", "60158980eaf665c2757c032e906fd8b4fcdacc9396d8f72257f5c78823bfd3ee"))
xyt = readdlm(datadep"track/track.csv")

using Makie, AbstractPlotting

get_track(I) = [(xyt[i,1], xyt[i,2]) for i in 1:I]
m, M = extrema(xyt[:,1:2])
m -= 10
M += 10
limits = FRect(m, m, M - m, M - m)
sc = Scene(limits = limits, scale_plot = false)
i = Node(2)
lines!(sc, lift(get_track, i))
n = size(xyt,1)
for j in 2:n
    sleep((xyt[j,3] - xyt[j-1,3])/100) # speed up the walk by 100
    push!(i, j)
end


