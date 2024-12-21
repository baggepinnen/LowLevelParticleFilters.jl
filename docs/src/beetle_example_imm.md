# Filtering the track of a moving beetle using IMM
This tutorial is very similar to [Smoothing the track of a moving beetle](@ref), but uses an Interacting Multiple Models (IMM) filter to model the mode switching of the beetle. The IMM filter is a mixture model, in this case with internal Unscented Kalman filters, where each Kalman filter represents a different mode of the system. The IMM filter is able to switch between these modes based on the likelihood of the mode given the data.

This is an example of smoothing the 2-dimensional trajectory of a moving dung beetle. The example spurred off of [this Discourse topic](https://discourse.julialang.org/t/smoothing-tracks-with-a-kalman-filter/24209?u=yakir12). For more information about the research behind this example, see [Artificial light disrupts dung beetles’ sense of direction](https://www.lunduniversity.lu.se/article/artificial-light-disrupts-dung-beetles-sense-direction) and [A dung beetle that path integrates without the use of landmarks](https://pubmed.ncbi.nlm.nih.gov/32902692/). Special thanks to Yakir Gagnon for providing this example.

In this example we will describe the position coordinates, ``x`` and ``y``, of the beetle as functions of its velocity, ``v_t``, and direction, ``θ_t``:
```math
\begin{aligned}
x_{t+1} &= x_t + \cos(θ_t)v_t \\
y_{t+1} &= y_t + \sin(θ_t)v_t \\
v_{t+1} &= v_t + e_t \\
θ_{t+1} &= θ_t + w_t
\end{aligned}
```
where
``
e_t ∼ N(0,σ_e), w_t ∼ N(0,σ_w)
``
The beetle further has two "modes", one where it's moving towards a goal, and one where it's searching in a more erratic manner. Figuring out when this mode switch occurs is the goal of the filtering. The mode will be encoded as two different models, where the difference between the models lies in the amount of dynamic noise affecting the angle of the beetle, i.e., in the searching mode, the beetle has more angle noise. The mode switching is modeled as a stochastic process with a binomial distribution (coin flip) describing the likelihood of a switch from mode 0 (moving to goal) and mode 1 (searching). Once the beetle has started searching, it stays in that mode, i.e., the searching mode is "sticky" or "terminal".

The image below shows an example video from which the data is obtained
![Bettle](https://global.discourse-cdn.com/julialang/original/3X/7/9/79a9255c4fc79677249fce7321c8c70f3df46431.gif)

We load a single experiment from file for the purpose of this example (in practice, there may be hundreds of experiments)
```@example beetle_imm
using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Distributions, Plots, Random
using DisplayAs # hide
using DelimitedFiles
cd(@__DIR__)
path = "../track.csv"
xyt = readdlm(path)
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy # helper function
y = tosvec(collect(eachrow(xyt[:,1:2])))
nothing # hide
```
We then define some properties of the dynamics and the filter. We will use an [`AdvancedParticleFilter`](@ref) since we want to have fine-grained control over the noise sampling for the mode switch.
```@example beetle_imm
nx = 4 # Dimension of state: we have position (2d), speed and angle
ny = 2 # Dimension of measurements, we can measure the x and the y
@inline pos(s) = s[SVector(1,2)]
@inline vel(s) = s[3]
@inline ϕ(s) = s[4]
nothing # hide
```

We then define the probability distributions we need. The IMM filter takes a transition-probability matrix, ``P``, and an initial mixing probability, ``μ``. ``P`` is a Markov (stochastic) matrix, where each row sums to one, and `P[i, j]` is the probability of switching from mode `i` to mode `j`. `μ` is a vector of probabilities, where `μ[i]` is the probability of starting in mode `i`. We also define the noise distributions for the dynamics and the measurements. The dynamics noise is modeled as a Gaussian distribution with a standard deviation of `dvσ` for the velocity and `ϕσ` for the angle. The measurement noise is modeled as a Gaussian distribution with a standard deviation of `dgσ`. The initial state is modeled as a Gaussian distribution with a mean at the first measurement and a standard deviation of `d0`.
```@example beetle_imm
dgσ = 1.0 # the deviation of the measurement noise distribution
dvσ = 0.3 # the deviation of the dynamics noise distribution
ϕσ  = 0.5
P = [0.995 0.005; 0.0 1] # Transition probability matrix, we model the search mode as "terminal"
μ = [1.0, 0.0] # Initial mixing probabilities
R1 = Diagonal([1e-1, 1e-1, dvσ, ϕσ].^2)
R2 = dgσ^2*I(ny) # Measurement noise covariance matrix
d0 = MvNormal(SVector(y[1]..., 0.5, atan((y[2]-y[1])...)), [3.,3,2,2])
nothing # hide
```

We now define the dynamics, which is directly defined in discrete time. The third argument is a parameter we call `modegain`, which is used to scale the amount of noise in the angle of the beetle depending on the mode in which it is in. The last argument is a boolean that tells the dynamics function which mode it is in, we will close over this argument when defining the dynamics for the individual Kalman filters that are part of the IMM, one will use `m = false` and one will use `m = true`.
```@example beetle_imm
@inline function dynamics(s,_,modegain,t,w,m)
    # current state
    v = vel(s)
    a = ϕ(s)
    p = pos(s)

    y_noise, x_noise, v_noise, ϕ_noise = w

    # next state
    v⁺ = max(0.999v + v_noise, 0.0)
    a⁺ = a + (ϕ_noise*(1 + m*modegain))/(1 + v⁺) # next state velocity is used here
    p⁺ = p + SVector(y_noise, x_noise) + SVector(sincos(a))*v⁺ # current angle but next velocity
    SVector(p⁺[1], p⁺[2], v⁺, a⁺) # all next state
end
@inline measurement(s,u,p,t) = s[SVector(1,2)] # We observe the position coordinates with the measurement
nothing # hide
```

In this example, we have no control inputs, we thus define a vector of only zeros. We then solve the forward filtering problem and plot the results.
```@example beetle_imm
u = zeros(length(y)) # no control inputs
kffalse = UnscentedKalmanFilter{false,false,true,false}((x,u,p,t,w)->dynamics(x,u,p,t,w,false), measurement, R1, R2, d0; ny, nu=0, p=10)
kftrue = UnscentedKalmanFilter{false,false,true,false}((x,u,p,t,w)->dynamics(x,u,p,t,w,true), measurement, R1, R2, d0; ny, nu=0, p=10)

imm = IMM([kffalse, kftrue], P, μ; p = 10)

T = length(y)
sol = forward_trajectory(imm, u, y, interact=true)
figx = plot(sol, plotu=false, plotRt=true)
figmode = plot(sol.extra', title="Mode")
plot(figx, figmode)
DisplayAs.PNG(Plots.current()) # hide
```

If you have followed the particle filter tutorial [Smoothing the track of a moving beetle](@ref), you will notice that the result here is much worse. We used noise parameters similar to in the particle-gilter example, but those were tuned fo the particle filter. Below, we will attempt to optimize the performance of the IMM filter.


## Tuning by optimization
We will attempt to optimize the dynamics and measurement noise covariance matrices and the `modegain` parameter. We code this up in two functions, one that takes the parameter vector and returns an [`IMM`](@ref) filter, and one that calculates the loss given the filter. We will optimize the log-likelihood of the data given the filter.


```@example beetle_imm
params = [log10.(diag(R1)); log10(1); log10(10)]

function get_opt_kf(p)
    T = eltype(p)
    R1i = Diagonal(SVector{4}(exp10.(p[1:4])))
    R2i = SMatrix{2,2}(exp10(p[5])*R2)
    d0i = MvNormal(SVector{4, T}(T.(d0.μ)), SMatrix{4,4}(T.(d0.Σ)))
    modegain = 2+exp10(p[6])
    Pi = SMatrix{2,2, Float64,4}(P)
    # sigmoid(x) = 1/(1+exp(-x))
    # switch_prob = sigmoid(p[7])
    # Pi = [1-switch_prob switch_prob; 0 1]
    kffalse = UnscentedKalmanFilter{false,false,true,false}((x,u,p,t,w)->dynamics(x,u,p,t,w,false), measurement, R1i, R2i, d0i; ny, nu=0)
    kftrue = UnscentedKalmanFilter{false,false,true,false}((x,u,p,t,w)->dynamics(x,u,p,t,w,true), measurement, R1i, R2i, d0i; ny, nu=0)

    IMM([kffalse, kftrue], Pi, T.(μ), p=modegain)
end
function cost(pars)
    try
        imm = get_opt_kf(pars)
        T = length(y)
        ll = loglik(imm, u[1:T], y[1:T], interact=true) - 1/2*logdet(imm.models[1].R1)
        return -ll
    catch e
        # rethrow() # If you only get Inf, you can uncomment this line to see the error message
        return eltype(pars)(Inf)
	end
end

using Optim
Random.seed!(0)
res = Optim.optimize(
    cost,
    params,
    ParticleSwarm(), # Use a gradient-free optimizer. ForwardDiff works, but the algorithm is numerically difficult to compute gradients through and may suffer from overflows in the gradient computation
    Optim.Options(
        show_trace        = true,
        show_every        = 5,
        iterations        = 100,
        time_limit        = 30,
    ),
)

imm = get_opt_kf(res.minimizer)
imm = get_opt_kf([-0.1981314138910982, -0.18626406669394405, -2.7342979645906547, 0.17994244691004058, -11.706419070755908, -54.16703441089562]) #make sure it goes well # hide

sol = forward_trajectory(imm, u, y)
plot(sol.extra', title="Mode (optimized filter)")
```

If it went well, the filter should be in mode 1 (the `false` mode) from the start until around 200 time steps, at which point it should switch to model 2 (`true`). This method of detecting the mode switch of the beetle appears to be somewhat less robust than the particle filter, but is significantly cheaper computationally. 

The IMM filter does not stick in mode 2 perpetually after having reached it since it never actually becomes fully confident that mode 2 has been reached, but detecting the first switch is sufficient to know that the switch has occurred. 

The log-likelihood of the solution
```@example beetle_imm
sol.ll
```

should be similar to that of the particle-filter in the tutorial [Smoothing the track of a moving beetle](@ref), which was around -1660.