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

n = 2 # Dinemsion of state: we have speed and angle, so two
p = 2 # Dinemsion of measurements, we can measure the x and the y, so also two

const dg = MvNormal(p, dgσ)          # Measurement noise Distribution
const df = MvNormal(n, dfσ)          # Dynamics noise Distribution
# const d0 = MvNormal(randn(n), 2.0)   # Initial state Distribution ??? not sure what this is

mutable struct State # a custom type to save the previous states (and avoid globals)
    yx::SVector{2, Float64} # yx and not xy because of the sincos function (and lack of a cossin function)
    vθ::SVector{2, Float64} # the speed and direction
end

s = State(SVector(0.0, 0.0), SVector(0.6, 0.7)) # just some initial states, coordinate at origo, with some speed and an initial direction

nsteps = 300 # how many steps will we have to smooth
yxs = [SVector{2, Float64}(0, 0) for i in 1:nsteps] # saving each coordinate

function step!(s::State) # stepping forward
    s.vθ += rand(df) # add the dynamics noise
    s.yx += SVector(sincos(s.vθ[2]))*s.vθ[1] + rand(dg) # step forward with the new speed and direction, while adding some measurement noise
end

i = 1
yxs[i] = s.yx # store initial coordinate
for i in 2:nsteps
    step!(s)
    yxs[i] = s.yx
end
    
plot(last.(yxs), first.(yxs))

