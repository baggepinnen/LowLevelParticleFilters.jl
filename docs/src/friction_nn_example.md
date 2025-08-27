# Learning Friction Dynamics with Neural Networks

In this example, we demonstrate how to learn an unknown friction model using a neural network within the Extended Kalman Filter framework. We consider an actuated mass subject to friction with Stribeck effect - a nonlinear phenomenon where friction decreases from static friction to Coulomb friction as velocity increases.

The key point is that while we know the basic physics (Newton's second law), we might not have an accurate friction model. We'll use a neural network to learn this friction component while leveraging our knowledge of the system structure.

Combining first-principles models and black-box neural networks, today popularized under the name "Universal Differential Equations" (UDEs), has been explored for a long time in the process-control community, early work includes [^UDE1] and [^UDE2], where neural networks were used to augment continuous-time models of chemical processes. Contrary to those works, we will let the state estimator itself find the parameters of the neural network alongside the state, rather than relying on a separate training phase using gradient descent.

[^UDE1]: Psichogios, Dimitris C., and Lyle H. Ungar. "A hybrid neural network‐first principles approach to process modeling." AIChE Journal 38.10 (1992): 1499-1511.
[^UDE2]: Psichogios, Dimitris C., and Lyle H. Ungar. "Process modeling using structured neural networks." 1992 American Control Conference. IEEE, 1992.

## System Description

We consider a simple mass moving in one dimension:
- State variables: position `x₁` and velocity `x₂`  
- Control input: applied force `u`
- Unknown friction force: `f_friction(x₂)`

The continuous-time dynamics are:
```
ẋ₁ = x₂
ẋ₂ = (u - f_friction(x₂)) / m
```

The true friction model includes Stribeck effect:
```
f_friction(v) = (f_c + (f_s - f_c) * exp(-|v|/v_s)) * sign(v) + f_v * v
```
where:
- `f_s`: static friction coefficient
- `f_c`: Coulomb friction coefficient  
- `v_s`: Stribeck velocity
- `f_v`: viscous friction coefficient

## Setup and Data Generation

```@example FRICTION_NN
using LowLevelParticleFilters, Lux, Random, SeeToDee, StaticArrays, Plots, LinearAlgebra
using ComponentArrays, DifferentiationInterface, SparseMatrixColorings
using SparseConnectivityTracer: TracerSparsityDetector
using DisplayAs # hide

using LowLevelParticleFilters: SimpleMvNormal

# True friction model with Stribeck effect
function true_friction(v; f_s=2.0, f_c=1.0, v_s=0.1, f_v=0.5)
    if abs(v) < 1e-6
        return 0.0f0  # Avoid numerical issues at zero velocity
    else
        return Float32((f_c + (f_s - f_c) * exp(-abs(v)/v_s)) * sign(v) + f_v * v)
    end
end

plot(true_friction, -2, 2, title="True Friction Model", xlabel="Velocity", ylabel="Friction Force")
```

```@example FRICTION_NN
# True system dynamics
function mass_dynamics(x, u, p, t)
    m = 1.0f0  # Mass
    x₁, x₂ = x
    force = u[1]
    friction = true_friction(x₂)
    
    # Add time-varying behavior: friction increases after t=200
    if t > 200
        friction *= 1.3f0
    end
    
    SA[
        x₂,  # ẋ₁ = velocity
        (force - friction) / m  # ẋ₂ = acceleration
    ]
end

# Discretize the system
Ts = 0.1f0  # Sample time
discrete_dynamics = SeeToDee.Rk4(mass_dynamics, Ts)

# System dimensions
nx = 2  # State dimension [position, velocity]
nu = 1  # Input dimension [force]
ny = 2  # Output dimension [position, velocity]

# Generate training data
function generate_data(rng)
    measurement(x, u, p, t) = x  # Measure full state
    
    # Time vector
    t = 0:Ts:200
    N = length(t)
    
    # Generate varied control inputs to excite different velocities
    u = Float32[]
    for i in 1:N
        if i < N÷4
            push!(u, 3.0f0 * cos(0.1f0 * t[i]))  # Slow oscillation
        elseif i < N÷2  
            push!(u, 5.0f0 * sign(sin(0.5f0 * t[i])))  # Square wave
        elseif i < 3N÷4
            push!(u, 2.0f0 * randn(rng))  # Random excitation
        else
            freq = 0.05f0 + 0.2f0 * (i - 3N÷4) / (N÷4)
            push!(u, 4.0f0 * sin(2π * freq * t[i]))  # Chirp signal
        end
    end
    u = [SA[u_i] for u_i in u]
    u = [u; u]
    
    # Initial state
    x0 = Float32[0.0, 0.0]
    
    # Simulate system
    x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, u)[1:end-1]
    
    # Add measurement noise
    y = [Float32.(x_i + 0.01f0 * randn(2)) for x_i in x]
    
    (; x, u, y, nx, nu, ny, Ts)
end

# Generate data
rng = Random.default_rng()
Random.seed!(rng, 42)
data = generate_data(rng)
nothing # hide
```

## Neural Network Friction Model

We'll use a small feedforward network to learn the friction as a function of velocity only:

```@example FRICTION_NN
# Neural network for friction model
# Input: velocity (1D)
# Output: friction force (1D)
ni = 1  # Network input dimension (velocity only)
no = 1  # Network output dimension (friction force)
nhidden = 6  # Hidden layer size

const friction_model2 = Chain(
    Dense(ni, nhidden, tanh),
    Dense(nhidden, nhidden, tanh),
    Dense(nhidden, nhidden, tanh),
    Dense(nhidden, no)
)

# Setup network parameters
dev = cpu_device()
ps, st = Lux.setup(rng, friction_model2) |> dev
parr = ComponentArray(ps)
nothing # hide
```

## Hybrid Dynamics Model

We combine our knowledge of the physics with the neural network friction model, the only part of the friction we assume known is that it is anti-symmetric around zero velocity:

```@example FRICTION_NN
# Initial state combining physical states and NN parameters
x0 = Float32[0.0, 0.0]
s0 = ComponentVector(; x=x0, p=parr)

function friction_function(v, params, st)
    # Neural network predicts friction based on velocity
    # We assume that we know that friction is anti-symmetric around zero velocity
    friction_nn, _ = Lux.apply(friction_model2, SA[abs(v)], params, st)
    return friction_nn[1]*sign(v)
end

# Continuous-time hybrid dynamics: known physics + learned friction
function hybrid_dynamics_continuous(xp, u, p, t)
    xp_comp = ComponentArray(xp, getaxes(s0))
    
    x₁, x₂ = xp_comp.x
    params = xp_comp.p
    m = 1.0f0
    
    friction = friction_function(x₂, params, st)
    
    # Known physics: Newton's second law
    force = u[1]
    acceleration = (force - friction) / m
    
    # Combine state and parameter dynamics
    ComponentVector(
        x = SA[x₂, acceleration],  # State derivatives
        p = -0.0001f0 * params  # Parameter dynamics (slow decay equivalent to 0.999 in discrete time)
    )
end

# Discretize the hybrid dynamics
discrete_hybrid_dynamics = SeeToDee.Rk4(hybrid_dynamics_continuous, Ts)

# Wrapper for in-place version needed by EKF
function hybrid_dynamics(out0, xp0, u, p, t)
    xp_next = discrete_hybrid_dynamics(xp0, u, p, t)
    out0 .= xp_next
    nothing
end

# Measurement model (observe full state)
@views measurement(out, xp, _, _, _) = out .= xp[1:nx]
nothing # hide
```

## Extended Kalman Filter Setup

```@example FRICTION_NN
# Process and measurement noise covariances
R1 = Diagonal([
    0.001f0 * ones(nx);             # Process noise for physical state
    0.0001f0 * ones(length(parr))   # Noise for NN parameters (allows learning)
])
R2 = Diagonal(0.05f0^2 * ones(ny))  # Measurement noise

# Jacobian computation with sparsity detection
function Ajacfun(x, u, p, t)
    backend = AutoSparse(
        AutoForwardDiff(),
        sparsity_detector=TracerSparsityDetector(),
        coloring_algorithm=GreedyColoringAlgorithm(),
    )
    out = similar(getdata(x))
    inner = (out, x) -> hybrid_dynamics(out, x, u, p, t)
    prep = prepare_jacobian(inner, out, backend, getdata(x))
    jac = one(eltype(x)) .* sparsity_pattern(prep)
    
    function (x, u, p, t)
        inner2 = (out, x) -> hybrid_dynamics(out, x, u, p, t)
        DifferentiationInterface.jacobian!(inner2, out, jac, prep, backend, x)
    end
end

Ajac = Ajacfun(s0, data.u[1], nothing, 0)

# Constant measurement Jacobian
const CJ_ = [I(nx) zeros(Float32, nx, length(parr))]
Cjac(x, u, p, t) = CJ_

# Initialize Extended Kalman Filter
ynames = ["position", "velocity"]
xnames = [ynames; ["nn_$i" for i in 1:length(parr)]]
unames = ["force"]
snames = SignalNames(x = xnames, y = ynames, u = unames, name="EKF")
ekf = ExtendedKalmanFilter(
    hybrid_dynamics, 
    measurement, 
    R1, 
    R2, 
    SimpleMvNormal(s0, 10000R1);
    nu, 
    ny,
    check=false, 
    Ajac, 
    Cjac, 
    Ts,
    names = snames,
)
nothing # hide
```

## State Estimation and Friction Learning

```@example FRICTION_NN
# Perform filtering
@time sol = forward_trajectory(ekf, data.u, data.y)

# Plot state estimation results
kwargs = (plotx=false, plotxt=false, plotyh=true, plotyht=false, plotu=true, plote=true)
p1 = plot(sol; name="EKF", layout=(nx+nu, 1), size=(900, 600), kwargs...)
DisplayAs.PNG(p1) # hide
```

## Learned Friction vs True Friction

Let's compare the learned friction model with the true friction:

```@example FRICTION_NN
# Extract final parameters
final_params = ComponentArray(sol.xt[end][nx+1:end], getaxes(parr))

# Generate velocity range for comparison
v_test = LinRange(-3.0f0, 3.0f0, 100)

# Compute true friction
friction_true = [true_friction(v) for v in v_test]

# Compute learned friction (at t=200, before change)
friction_learned_mid = Float32[]
params_mid = ComponentArray(sol.xt[2000-1][nx+1:end], getaxes(parr))  # At t=200
for v in v_test
    friction = friction_function(Float32(v), params_mid, st)
    push!(friction_learned_mid, friction)
end

# Compute learned friction (final, after adaptation)
friction_learned_final = Float32[]
for v in v_test
    friction = friction_function(Float32(v), final_params, st)
    push!(friction_learned_final, friction)
end

# Compute modified true friction (after t=200)
friction_true_modified = [1.3f0 * true_friction(v) for v in v_test]

# Plot comparison
p2 = plot(v_test, friction_true, label="True friction (initial)", lw=2, ls=:dash, c=1)
plot!(v_test, friction_true_modified, label="True friction (after t=200)", lw=2, ls=:dash, c=2)
plot!(v_test, friction_learned_mid, label="Learned (at t=200)", lw=2, alpha=0.7, c=1)
plot!(v_test, friction_learned_final, label="Learned (final)", lw=2, c=2)
plot!(xlabel="Velocity", ylabel="Friction Force", title="Friction Model Comparison")
plot!(legend=:bottomright, size=(800, 500))
DisplayAs.PNG(p2) # hide
```


## Closing Remarks

We combined first-principles knowledge (Newton's laws) with a neural network to learn only the unknown component (friction). This is more sample-efficient and interpretable than learning the entire dynamics.

The Extended Kalman Filter continuously updates the neural network parameters, allowing the model to adapt to changes in the system (e.g., the friction increase at t=200).
