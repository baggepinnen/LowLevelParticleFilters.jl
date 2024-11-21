# Adaptive Neural-Network training
In this example, we will demonstrate how we can take the estimation of time-varying parameters to the extreme, and use a nonlinear state estimator to estimate the weights in a neural-network model of a dynamical system. 

In the tutorial [Joint state and parameter estimation](@ref), we demonstrated how we can add a parameter as a state variable and let the state estimator estimate this alongside the state. In this example, we will try to learn an entire black-box model of the system dynamics using a neural network, and treat the network weights as time-varying parameters by adding them to the state.

We start by generating some data from a simple dynamical system, we will continue to use the quadruple-tank system from [Joint state and parameter estimation](@ref).

```@example ADAPTIVE_NN
using LowLevelParticleFilters, Lux, Random, SeeToDee, StaticArrays, Plots, LinearAlgebra, ComponentArrays, DifferentiationInterface, SparseMatrixColorings
using SparseConnectivityTracer: TracerSparsityDetector
using DisplayAs # hide

using LowLevelParticleFilters: SimpleMvNormal

function quadtank(h,u,p,t)
    kc = 0.5
    k1, k2, g = 1.6, 1.6, 9.81
    A1 = A3 = A2 = A4 = 4.9
    a1, a3, a2, a4 = 0.03, 0.03, 0.03, 0.03
    γ1, γ2 = 0.2, 0.2

    if t > 2000
        a1 *= 1.5 # Change the parameter at t = 2000
    end

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0

    SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    ]
end

Ts = 30 # sample time
discrete_dynamics = SeeToDee.Rk4(quadtank, Ts) # Discretize dynamics
nu = 2 # number of control inputs
nx = 4 # number of state variables
ny = 4 # number of measured outputs

function generate_data()   
    measurement(x,u,p,t) = x#SA[x[1], x[2]]
    Tperiod = 200
    t = 0:Ts:4000
    u = vcat.((0.25 .* sign.(sin.(2pi/Tperiod .* t)) .+ 0.25) .* sqrt.(rand.()))
    u = SVector{nu, Float32}.(vcat.(u,u))
    x0 = Float32[2,2,3,3]
    x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, u)[1:end-1]
    y = measurement.(x, u, 0, 0)
    y = [Float32.(y .+ 0.01.*randn.()) for y in y] # Add some noise to the measurement

    (; x, u, y, nx, nu, ny, Ts)
end

rng = Random.default_rng()
Random.seed!(rng, 1)
data = generate_data()
nothing # hide
```


## Neural network dynamics
Our neural network will be a small feedforward network built using the package [Lux.jl](https://lux.csail.mit.edu/stable/tutorials/beginner/5_OptimizationIntegration). 

```@example ADAPTIVE_NN
ni = ny + nu
nhidden = 8
const model_ = Chain(Dense(ni, nhidden, tanh), Dense(nhidden, nhidden, tanh), Dense(nhidden, ny))
```

Since the network is rather small, we will train on the CPU only, this will be fast enough for this use case. We may extract the parameters of the network using the function `Lux.setup`, and convert them to a ComponentArray to make it easier to refer to different parts of the combined state vector.
```@example ADAPTIVE_NN
dev = cpu_device()
ps, st = Lux.setup(rng, model_) |> dev
parr = ComponentArray(ps)
nothing # hide
```

The dynamics of our black-box model will call the neural network to predict the next state given the current state and input. We bias the dynamics towards low frequencies by adding a multiple of the current state to the prediction of the next state, `0.95*x`. We also add a small amount of weight decay to the parameters of the neural network for regularization, `0.995*p`.
```@example ADAPTIVE_NN
function dynamics(out0, xp0, u, _, t)
    xp = ComponentArray(xp0, getaxes(s0))
    out = ComponentArray(out0, getaxes(s0))
    x = xp.x
    p = xp.p
    xp, _ = Lux.apply(model_, [x; u], p, st)
    @. out.x = 0.95f0*x+xp
    @. out.p = 0.995f0*p
    nothing
end

@views measurement(out, x, _, _, _) = out .= x[1:nx] # Assume measurement of the full state vector
nothing # hide
```

For simplicity, we have assumed here that we have access to measurements of the entire state vector of the original process. This is many times unrealistic, and if we do not have such access, we may instead augment the measured signals with delayed versions of themselves (sometimes called a delay embedding). This is a common technique in discrete-time system identification, used in e.g., `ControlSystemIdentification.arx` and `subspaceid`.

The initial state of the process `x0` and the initial parameters of the neural network `parr` can now be concatenated to form the initial augmented state `s0`.
```@example ADAPTIVE_NN
x0 = Float32[2; 2; 3; 3]
s0 = ComponentVector(; x=x0, p=parr)
nothing # hide
```

## Kalman filter setup
We will estimate the parameters using two different nonlinear Kalman filters, the [`ExtendedKalmanFilter`](@ref) and the [`UnscentedKalmanFilter`](@ref). The covariance matrices for the filters, `R1, R2`, may be tuned such that we get the desired learning speed of the weights, where larger covariance for the network weights will allow for faster learning, but also more noise in the estimates. 
```@example ADAPTIVE_NN
R1 = Diagonal([0.1ones(nx); 0.01ones(length(parr))]) .|> Float32
R2 = Diagonal((1e-2)^2 * ones(ny)) .|> Float32
nothing # hide
```

The [`ExtendedKalmanFilter`](@ref) uses Jacobians of the dynamics and measurement model, and if we do not provide those functions they will be automatically computed using ForwardDiff.jl. Since our Jacobians will be relatively large but sparse in this example, we will make use of the sparsity-aware features of DifferentiationInterface.jl in order to get efficient Jacobian computations. 
```@example ADAPTIVE_NN
function Ajacfun(x,u,p,t) # Function that returns a function for the Jacobian of the dynamics
    # For large neural networks, it might be faster to use an OOP formulation with Zygote instead of ForwardDiff. Zygote does not handle the in-place version
    backend = AutoSparse(
        AutoForwardDiff(),
        # AutoZygote(),
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    )
    out = similar(getdata(x))
    inner = (out,x)->dynamics(out,x,u,p,t)
    prep = prepare_jacobian(inner, out, backend, getdata(x))
    jac = one(eltype(x)) .* sparsity_pattern(prep)
    function (x,u,p,t)
        inner2 = (out,x)->dynamics(out,x,u,p,t)
        DifferentiationInterface.jacobian!(inner2, out, jac, prep, backend, x)
    end
end

Ajac = Ajacfun(s0, data.u[1], nothing, 0)

const CJ_ = [I(nx) zeros(Float32, nx, length(parr))] # The jacobian of the measurement model is constant
Cjac(x,u,p,t) = CJ_
nothing # hide
```

## Estimation
We may now initialize our filters and perform the estimation. Here, we use the function [`forward_trajectory`](@ref) to perform filtering along the entire data trajectory at once, but we may use this in a streaming fashion as well, as more data becomes available in real time.

We plot the one-step ahead prediction of the outputs and compare to the "measured" data.
```@example ADAPTIVE_NN
ekf = ExtendedKalmanFilter(dynamics, measurement, R1, R2, SimpleMvNormal(s0, 100R1); nu, check=false, Ajac, Cjac, Ts)
ukf = UnscentedKalmanFilter(dynamics, measurement, R1, R2, SimpleMvNormal(s0, 100R1); nu, ny, Ts)

@time sole = forward_trajectory(ekf, data.u, data.x)
@time solu = forward_trajectory(ukf, data.u, data.x)

plot(sole, plotx=false, plotxt=false, plotyh=true, plotyht=false, plotu=false, plote=true, name="EKF", layout=(nx, 1))
plot!(solu, plotx=false, plotxt=false, plotyh=true, plotyht=false, plotu=false, plote=true, name="UKF", ploty=false, size=(1200, 1500))
DisplayAs.PNG(Plots.current()) # hide
```

We see that prediction errors, $e$, are large in the beginning when the network weights are randomly initialized, but after about half the trajectory the errors are significantly reduced. Just like in the tutorial [Joint state and parameter estimation](@ref), we modified the true dynamics after some time, at $t=2000$, and we see that the filters are able to adapt to this change after a transient increase in prediction error variance.

We may also plot the evolution of the neural-network weights over time, and see how the filters adapt to the changing dynamics of the system.
```@example ADAPTIVE_NN
plot(
    plot(0:Ts:4000, reduce(hcat, sole.xt)'[:, nx+1:end], title="EKF parameters"),
    plot(0:Ts:4000, reduce(hcat, solu.xt)'[:, nx+1:end], title="UKF parameters"),
    legend = false,
)
DisplayAs.PNG(Plots.current()) # hide
```

## Benchmarking
The neural network used in this example has
```@example ADAPTIVE_NN
length(parr)
```
parameters, and the length of the data is
```@example ADAPTIVE_NN
length(data.u)
```

Performing the estimation using the Extended Kalman Filter took
```julia
using BenchmarkTools
@btime forward_trajectory(ekf, data.u, data.x);
  46.034 ms (77872 allocations: 123.45 MiB)
```
and with the Unscented Kalman Filter
```julia
@btime forward_trajectory(ukf, data.u, data.x);
  142.608 ms (2134370 allocations: 224.82 MiB)
```

The EKF is a bit faster, which is to be expected. Both methods are very fast from a neural-network training perspective, but the performance will not scale favorably to very large network sizes.

## Closing remarks

We have seen how to estimate train a black-box neural network dynamics model by treating the parameter estimation as a state-estimation problem. This example is very simple and leaves a lot of room for improvement, such as
- We assumed very little prior knowledge of the dynamics. In practice, we may want to model as much as possible from first principles and add a neural network to capture only the residuals that out first-principles model cannot capture.
- We used forward-mode AD to compute the Jacobian. The Jacobian of the dynamics has dense rows, which means that it's theoretically favorable to use reverse-mode AD to compute it. This is possible using Zygote.jl, but Zygote does not handle array mutation, and one must thus avoid the in-place version of the dynamics. Since the number of parameters in this example is small, sparse forward mode AD ended up being slightly faster.