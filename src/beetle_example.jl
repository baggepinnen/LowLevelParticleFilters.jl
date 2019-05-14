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
#
using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Distributions,  StatsPlots
