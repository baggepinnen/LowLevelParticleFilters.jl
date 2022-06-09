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

##

using DelimitedFiles
# path = download("https://vision-group-file-sharing.s3.amazonaws.com/Data%20backup%20and%20storage/Yakir/track.csv?X[…]460738cad19bd24e7fa6f10375c5b65b234857fa5b99ea2")
path = "/home/fredrikb/Downloads/track.csv"
xyt = readdlm(path)
##

using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Distributions, Plots
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
vecvec_to_mat(x) = reduce(hcat, x)'

N = 1000 # Number of particles in the particle filter
n = 4 # Dinemsion of state: we have speed and angle, so two
p = 2 # Dinemsion of measurements, we can measure the x and the y, so also two
@inline pos(s) = s[SVector(1,2)]
@inline vel(s) = s[3]
@inline ϕ(s) = s[4]
@inline mode(s) = s[5]

dgσ = 1 # the deviation of the measurement noise distribution
dvσ = 0.3#0.8 # the deviation of the dynamics noise distribution
ϕσ = 0.5
const switch_prob = 0.03
const dg = MvNormal(@SVector(zeros(p)), dgσ^2) # Measurement noise Distribution
# const df = MvNormal(@SVector(zeros(n)), [1e-1, 1e-1, dvσ^2, ϕσ^2]) # Dynamics noise Distribution NOTE: MvNormal wants a variance, not std
const df = LowLevelParticleFilters.TupleProduct((Normal.(0,[1e-1, 1e-1, dvσ, ϕσ])...,Binomial(1,switch_prob)))
const df2 = LowLevelParticleFilters.TupleProduct((Normal.(0,[1e-1, 1e-1, dvσ, ϕσ])...,))#Binomial(1,switch_prob)))
# const df = Product(Normal.(0,[1e-1, 1e-1, dvσ, ϕσ]))#;Binomial(1,switch_prob)])
y = tosvec(collect(eachrow(xyt[:,1:2])))
const d0 = MvNormal(SVector(y[1]..., 0.5, atan((y[2]-y[1])...), 0), [3.,3,2,2,0])

const noisevec = zeros(5)

@inline function dynamics(s,u,p,t,noise=false)
    # current states
    m = mode(s)
    v = vel(s)
    a = ϕ(s)
    p = pos(s)
    # get noise
    if noise
        y_noise, x_noise, v_noise, ϕ_noise,_ = rand!(df, noisevec)
    else
        y_noise, x_noise, v_noise, ϕ_noise = 0.,0.,0.,0.
    end
    # next states
    v⁺ = max(0.999v + v_noise, 0.0)
    m⁺ = Float64(m == 0 ? rand() < switch_prob : true)
    a⁺ = a + (ϕ_noise*(1 + m*10))/(1 + v) # next state velocity is used here
    p⁺ = p + SVector(y_noise, x_noise) + SVector(sincos(a))*v # current angle but next velocity
    SVector{5,Float64}(p⁺[1], p⁺[2], v⁺, a⁺, m⁺) # all next states
end
function measurement_likelihood(s,u,y,p,t)
    logpdf(dg, pos(s)-y) # A simple linear measurement model with normal additive noise
end
@inline measurement(s,u,p,t,noise=false) = s[SVector(1,2)] + noise*rand(dg) # We observer the position coordinates with the measurement



N = 1000 # Number of particles in the particle filter
n = 4 # Dinemsion of state: we have speed and angle, so two
p = 2 # Dinemsion of measurements, we can measure the x and the y, so also two


u = zeros(length(y))
pf = AuxiliaryParticleFilter(AdvancedParticleFilter(N, dynamics, measurement, measurement_likelihood, df, d0))
T = length(y)
sol=forward_trajectory(pf,u[1:T],y[1:T])
(x,w,we,ll) = sol
@show ll
plot(sol, markerstrokecolor=:auto, m=(2,0.5))
##
# xh,ll = mean_trajectory(pf,u,y)
# xh = vecvec_to_mat(xh)
xh = mean_trajectory(x,we)
#

"plotting helper function"
function to1series(x::AbstractVector, y)
    r,c = size(y)
    y2 = vec([y; fill(Inf, 1, c)])
    x2 = repeat([x; Inf], c)
    x2,y2
end
to1series(y) = to1series(1:size(y,1),y)


plot(xh[:,1],xh[:,2], layout=4, c=:blue, lab="xh", legend=:bottomleft)
scatter!(to1series(vel.(x)'[:,1:40:end])..., m=(:black, 0.02, 1), subplot=3, lab="", ylims=(0,5))
scatter!(to1series(ϕ.(x)'[:,1:40:end])..., m=(:black, 0.02, 1), subplot=4, lab="")
plot!(xyt[:,1],xyt[:,2], subplot=1, c=:red, lab="y")
plot!(xh[:,1:2], subplot=2, lab=["xh" ""], c=:blue, legend=:bottomleft)
plot!(xyt[:,1:2], subplot=2, lab=["y" ""], c=:red)
plot!(xh[:,3], subplot=3, lab="vel", legend=:topleft)
plot!(identity.(xh[:,4]), subplot=4, lab="angle", legend=:topleft)
vline!([findfirst(x->x>0.9, xh[:,5])], l=(:green, :dash, 2), subplot=4, lab="Switch", ylims=(-30, 70))
##

M = 10 # Number of smoothing trajectories, NOTE: if this is set higher, the result will be better at the expense of linear scaling of the computational cost.
sb,ll = smooth(pf, M, u, y) # Sample smooting particles (b for backward-trajectory)
sbm = smoothed_mean(sb)     # Calculate the mean of smoothing trajectories
sbc = smoothed_cov(sb)      # And covariance
sbt = smoothed_trajs(sb)    # Get smoothing trajectories
sbs = [diag(sbc) for sbc in sbc] |> vecvec_to_mat .|> sqrt
##
plot!(sbm[1,:],sbm[2,:], subplot=1, lab="xs")
plot!(sbm'[:,1:2], subplot=2, lab=["xs" "ys"])
plot!(sbm'[:,3], subplot=3, lab="vs")
plot!(identity.(sbm'[:,4]), subplot=4, lab="as")
##
plot!(sbt[1,:,:]', subplot=2, lab="", l=(:black,0.2))
plot!(sbt[2,:,:]', subplot=2, lab="", l=(:black,0.2))
plot!(sbt[3,:,:]', subplot=3, lab="", l=(:blue,0.2))
plot!(identity.(sbt[4,:,:]'), subplot=4, lab="", l=(:orange,0.2))
##
plot(xh[:,5], lab="Filtering")
plot!(to1series(sbt[5,:,:]')..., lab="Smoothing", title="Mode trajectories", l=(:black,0.2))


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


# function LowLevelParticleFilters.smooth(pf::Union{AdvancedParticleFilter, AuxiliaryParticleFilter{<:AdvancedParticleFilter}}, M, u, y)
#     T = length(y)
#     N = num_particles(pf)
#     f = LowLevelParticleFilters.dynamics(pf)
#     xf,wf,wef,ll = forward_trajectory(pf, u, y)
#     @assert M <= N "Must extend cache size of bins and j to allow this"
#     xb = Array{particletype(pf)}(undef,M,T)
#     j = LowLevelParticleFilters.resample(LowLevelParticleFilters.ResampleSystematic, wef[:,T], M)
#     for i = 1:M
#         xb[i,T] = xf[j[i], T]
#     end
#     wb = Vector{Float64}(undef,N)
#     df = dynamics_density(pf)
#     bin = Binomial(1,switch_prob)
#     inds = SVector(1,2,3,4)
#     @inbounds for t = T-1:-1:1
#         for m = 1:M
#             for n = 1:N
#                 x1 = f(xf[n,t],u[t],t)
#                 switch = mode(x1) != mode(xb[m,t+1])
#                 wb[n] = wf[n,t] + logpdf(df, x1[inds], xb[m,t+1][inds],t) + logpdf(bin, switch)*(1-mode(xb[m,t+1])) # only add switch pdf if previous mode was 0
#             end
#             i = LowLevelParticleFilters.draw_one_categorical(pf,wb)
#             xb[m,t] = xf[i, t]
#         end
#         # @show tset
#     end
#     return xb,ll
# end