# Learning a disturbance model using SciML

In this example we will attempt to learn how an unknown and stochastic input, sun shining in through a window, influences a dynamical system, the temperature in a house. How the sun is shining on a house on a cloud-free day, absent any surrounding trees or buildings can be readily simulated. However, the real world offers a number of challenges that influence the effect this has on the inside temperature
- Surrounding trees and buildings may cast shadows on the house at certain parts of the day.
- The sun shining in through a window has much greater effect than if it's shining on a wall.
- Cloud cover modulates the effect of the sun.
- As a vendor of, e.g., HVAC equipment with interesting control systems, you may not want to model each individual site in detail, including the location and size of windows and surrounding shading elements. Even if these are static, they are thus to be considered unknown.

We can model this as some deterministic parts and one stochastic parts, some known and some unknown. The path of the sun across the sky is deterministic and periodic, with one daily and one yearly component. The surroundings, like trees and buildings is for the most part static, but the influence this has on the insolation is unknown, and so is the exact location of windows on the house. However, the cloud cover is stochastic. We can thus model insolation by
- Treating the current cloud cover as a stochastic variable ``C_{cloud} \in [0, 1]`` to be estimated continuously. We achieve this by including the cloud cover as a state variable in our system.
- Treating the insolation when there is _no cloud cover_ as a deterministic function of the time of day (we ignore the yearly component here for simplicity). This function will be modeled as a basis-function expansion that will be learned from data.
- The effective insolation at any point in time is thus ``I_{solar} = (1 - C_{cloud}) I_{solar, clear}``, that is, the cloud-free insolation is modulated by the current cloud cover.

## System Description

We consider a simplified thermal model of a single-room house:
- State variable: room temperature `T`
- Control input: heater power `P_heater`
- Disturbances: external temperature `T_ext` and solar insolation through windows

The heat transfer dynamics follow Newton's law of cooling with additional terms for heating and solar gains:
```math
C_{thermal} Ṫ = -k_{loss}  (T - T_{ext}) + η  P_{heater} + A_{window} I_{solar}
```
where:
- `C_thermal`: thermal capacity of the room
- `k_loss`: heat loss coefficient
- `η`: heater efficiency
- `A_window`: effective window area
- `I_solar`: solar insolation (W/m²)


## Data Generation

First, let's generate realistic thermal data with time-varying external conditions:

```@example THERMAL_NN
using LowLevelParticleFilters, Random, SeeToDee, StaticArrays, Plots, LinearAlgebra, Statistics
using LowLevelParticleFilters: SimpleMvNormal
using Optim
using DisplayAs # hide

# System parameters
const C_thermal = 10.0f0      # Thermal capacity (kWh/°C)
const k_loss = 0.5f0          # Heat loss coefficient (kW/°C)
const η = 0.95f0              # Heater efficiency
const A_window = 20.0f0       # Effective window area factor

# Time constants
const hours_per_day = 24.0f0
const Ts = 0.25f0            # Sample time (15 minutes)

# Helper function for time of day (0-24 hours)
time_of_day(t) = mod(t, hours_per_day)

# External temperature (sinusoidal daily variation)
function external_temp(t)
    tod = time_of_day(t)
    T_mean = 10.0f0  # Mean temperature (°C)
    T_amplitude = 5.0f0  # Daily variation amplitude
    T_mean + T_amplitude * sin(2π * (tod - 6) / hours_per_day)  # Peak at 12:00
end

# True solar insolation pattern (W/m²)
function true_insolation(t, cloud_cover)
    tod = time_of_day(t)
    # Cropped sinusoid
    base_insolation = max(500.0f0 * (0.2 + sin(π * (tod - 6) / 12)), 0)
    return base_insolation * (1 - cloud_cover)
end

# Plot the daily patterns
t_plot = 0:0.1:48  # Two days for visualization
plot(t_plot, external_temp.(t_plot), label="External Temperature (°C)", lw=2)
plot!(t_plot, true_insolation.(t_plot, 0) ./ 100, label="Clear Sky Insolation (×100 W/m²)", lw=2)
plot!(t_plot, true_insolation.(t_plot, 0.5) ./ 100, label="50% Cloud Cover (×100 W/m²)", lw=2, ls=:dash)
xlabel!("Time (hours)")
title!("Daily Environmental Patterns")
```

Now let's define the true system dynamics and generate training data:

```@example THERMAL_NN
# True system dynamics with time-varying cloud cover
function thermal_dynamics_true(x, u, p, t)
    T_room, cloud_cover = x
    P_heater = u[1]
    
    # External conditions
    T_ext = external_temp(t)
    I_solar = true_insolation(t, cloud_cover)
    
    # Heat balance
    dT_dt = (-k_loss * (T_room - T_ext) + η * P_heater + A_window * I_solar / 1000) / C_thermal
    
    # Cloud cover changes slowly (random walk)
    dcloud_dt = 0.0f0  # Driven by process noise, zero deterministic dynamics
    
    SA[dT_dt, dcloud_dt]
end

# Discretize the dynamics
discrete_dynamics_true = SeeToDee.Rk4(thermal_dynamics_true, Ts)

# Generate training data
function generate_thermal_data(; days=7, measure_cloud_cover=true)
    rng = Random.default_rng()
    Random.seed!(rng, 123)
    
    t = 0:Ts:(days * hours_per_day)
    N = length(t)
    
    # Generate control inputs (varied heating patterns)
    u = Vector{SVector{1, Float32}}(undef, N)
    for (i, t) in enumerate(t)
        tod = time_of_day(t)
        # Different heating strategies throughout the day
        if 6 <= tod < 8 || 17 <= tod < 22  # Morning and evening comfort
            u[i] = SA[3.0f0 + 0.5f0 * randn(rng)]  # Higher heating
        elseif 22 <= tod || tod < 6  # Night setback
            u[i] = SA[1.0f0 + 0.2f0 * randn(rng)]  # Lower heating
        else  # Day time
            u[i] = SA[2.0f0 + 0.3f0 * randn(rng)]  # Moderate heating
        end
        u[i] = SA[clamp(u[i][1], 0.0f0, 5.0f0)]  # Heater power limits
    end
    
    # Initial conditions
    x0 = SA[20.0f0, 0.3f0]  # Initial room temp and cloud cover
    
    # Simulate with slowly varying cloud cover
    x = Vector{SVector{2, Float32}}(undef, N)
    x[1] = x0
    for i in 2:N
        # Process noise (small for temperature, larger for cloud cover)
        w = SA[0.01f0 * randn(rng), 0.06f0 * randn(rng)]
        x_next = discrete_dynamics_true(x[i-1], u[i-1], nothing, t[i-1])
        x[i] = x_next + w
        # Keep cloud cover in [0, 1]
        x[i] = SA[x[i][1], clamp(x[i][2]*0.999, 0.0f0, 1.0f0)]
    end
    
    # Measurements - optionally include cloud cover
    if measure_cloud_cover
        y = [SA[x_i[1] + 0.1f0 * randn(rng), x_i[2] + 0.01f0 * randn(rng)] for x_i in x]
        ny = 2
    else
        y = [SA[x_i[1] + 0.1f0 * randn(rng)] for x_i in x]
        ny = 1
    end
    
    (; x, u, y, t, N, Ts, ny, measure_cloud_cover)
end

# Generate data (with cloud cover measurements by default)
measure_cloud_cover = false  # Set to false to exclude cloud cover measurements
data = generate_thermal_data(; days=14, measure_cloud_cover)

# Visualize the generated data
p1 = plot(data.t, [x[1] for x in data.x], label="Room Temperature", ylabel="Temperature (°C)")
plot!(data.t, [external_temp(t) for t in data.t], label="External Temperature", ls=:dash, alpha=0.7)

p2 = plot(data.t, [x[2] for x in data.x], label="Cloud Cover", ylabel="Cloud Cover (0-1)", color=:orange)
p3 = plot(data.t, [u[1] for u in data.u], label="Heater Power", ylabel="Power (kW)", color=:red)

plot(p1, p2, p3, layout=(3,1), size=(900, 600), xlabel="Time (hours)")
```

## Radial Basis Function Model

We'll use a radial basis function expansion to learn the solar insolation pattern. This provides a more interpretable model than a neural network, with basis functions centered during daylight hours:

```@example THERMAL_NN
# Initialize RBF weights (parameters to be learned)
rng = Random.default_rng()
Random.seed!(rng, 456)
const n_basis = 8  # Number of basis functions
# Initialize with positive weights since insolation is always positive ("negative insolation" could model things like someone always opening a window in the morning letting cold air in)
rbf_weights = 100.0f0 * rand(Float32, n_basis)  # Random positive initialization

function basis_functions(t)
    tod = time_of_day(t)
    centers = LinRange(7.0f0, 17.0f0, n_basis) # Centers spread from 9 AM to 5 PM
    width = 1.5f0  # Width of each Gaussian basis function (in hours)
    @. exp(-((tod - centers) / width)^2)
end

# RBF evaluation function
function compute_nn_insolation(t, weights)
    return weights'basis_functions(t) # Linear combination of basis functions
end

nothing # hide
```

## Hybrid Dynamics for Estimation

Define the dynamics model that uses the neural network for insolation estimation:

```@example THERMAL_NN
# Measurement model, we measure temperature and optionally also cloud cover (this makes the problem much easier)
if data.measure_cloud_cover
    measurement_fun = (x, u, p, t) -> SA[x[1], x[2]]  # Temperature and cloud cover
else
    measurement_fun = (x, u, p, t) -> SA[x[1]]  # Only temperature
end

# Hybrid dynamics with neural network and cloud cover state
function thermal_dynamics_hybrid(x, u, p, t)
    T_room, cloud_cover = x
    P_heater = u[1]
    
    # External temperature (known)
    T_ext = external_temp(t)
    
    # Solar insolation from neural network
    I_base = compute_nn_insolation(t, p)
    I_solar = I_base * (1 - cloud_cover)
    
    # Heat balance
    dT_dt = (-k_loss * (T_room - T_ext) + η * P_heater + A_window * I_solar / 1000) / C_thermal
    
    # Cloud cover changes slowly
    dcloud_dt = 0.0001f0*(0.3f0 - cloud_cover)  # Driven by process noise, assume we know mean cloud cover over time
    
    SA[dT_dt, dcloud_dt]
end

# Discretize hybrid dynamics
const discrete_dynamics_hybrid5 = SeeToDee.ForwardEuler(thermal_dynamics_hybrid, Ts)

function clamped_dynamics(x,u,p,t)
    xp = discrete_dynamics_hybrid5(x,u,p,t)
    SA[xp[1], clamp(xp[2], 0.0f0, 1.0f0)]
end

# System dimensions for the filter
nx = 2  # State: [temperature, cloud_cover]
nu = 1  # Input: heater power
ny = data.ny  # Output dimension depends on whether cloud cover is measured
nothing # hide
```


## Parameter Estimation

Now we'll set up the state estimator and the optimization problem using a quasi-Newton method:

```@example THERMAL_NN
# Process and measurement noise for the filter
R1 = SMatrix{nx, nx}(Diagonal([0.01f0, 0.06f0]))  # Process noise
if data.measure_cloud_cover
    R2 = SMatrix{ny, ny}(Diagonal([0.1f0^2, 0.01f0^2]))  # Temperature and cloud cover noise
else
    R2 = SMatrix{ny, ny}(Diagonal([0.1f0^2]))  # Only temperature noise
end

# Initial state estimate
x0 = SA[20.0f0, 0.5f0]  # Initial temperature and cloud cover guess

# Cost function for optimization (sum of squared errors)
function cost(θ)
    T = eltype(θ)
    
    # Create filter with current parameters
    kf = UnscentedKalmanFilter(
        clamped_dynamics, 
        measurement_fun, 
        R1, 
        R2, 
        SimpleMvNormal(T.(x0), T.(2*R1));
        ny, nu, Ts,
    )
    
    # Compute sum of squared prediction errors
    LowLevelParticleFilters.sse(kf, data.u, data.y, θ)
end

# Initial parameters from the neural network initialization
θ_init = copy(rbf_weights)

result = Optim.optimize(
    cost,
    θ_init,
    BFGS(),
    Optim.Options(
        show_trace = false,
        store_trace = true,
        iterations = 200,
        g_tol = 1e-12,
    );
    autodiff = :forward,  # Use forward-mode AD for gradients
)

params_opt = result.minimizer

@info "Optimization complete. Converged: $(Optim.converged(result)), Iterations: $(Optim.iterations(result))"
@info "Final cost: $(Optim.minimum(result))"

# Plot convergence history
# plot(getfield.(result.trace, :value), #yscale=:log10, 
#      xlabel="Iteration", ylabel="Cost (SSE)",
#      title="Convergence", lw=2)
```

## Results Analysis

Let's analyze the results by running the filter with optimized parameters:

```@example THERMAL_NN
# Run filter with optimized parameters
kf_final = UnscentedKalmanFilter(
    clamped_dynamics,
    measurement_fun,
    R1,
    R2,
    SimpleMvNormal(x0, R1);
    p = params_opt,
    ny, nu, Ts
)

sol = forward_trajectory(kf_final, data.u, data.y)

# Extract estimated states
T_est = [sol.xt[i][1] for i in 1:length(sol.xt)]
cloud_est = [sol.xt[i][2] for i in 1:length(sol.xt)]
T_true = [data.x[i][1] for i in 1:length(data.x)]
cloud_true = [data.x[i][2] for i in 1:length(data.x)]

cloud_error = sqrt(mean(abs2, cloud_true .- cloud_est))

# Plot temperature estimation
p1 = plot(data.t, T_true, label="True Temperature", lw=2, color=:blue)
plot!(data.t, T_est, label="Estimated Temperature", lw=2, ls=:dash, color=:red)
plot!(data.t, [y[1] for y in data.y], label="Measurements", alpha=0.3, seriestype=:scatter, ms=1, color=:gray)
ylabel!("Temperature (°C)")
title!("Temperature Estimation")

# Plot cloud cover estimation
p2 = plot(data.t, cloud_true, label="True Cloud Cover", lw=2, color=:blue)
plot!(data.t, cloud_est, label="Estimated Cloud Cover", lw=2, ls=:dash, color=:red)
ylabel!("Cloud Cover")
xlabel!("Time (hours)")
title!("Cloud Cover Estimation")

plot(p1, p2, layout=(2,1), size=(1200, 800))
```

As we can see, it's easy to estimate the internal temperature, after all, we measure this directly. Estimating the cloud cover is significantly harder, notice in particular how the estimation drifts to 0.5 each night when there is no sun. This is expected since it is impossible to observe (in the estimation-theoretical sense) the cloud cover when there is no sun, since when there is no sun there is no effect of the cloud cover on the variable we do measure, the temperature. The fact that it drifts to 0.5 in particular can be explained by the growing estimated covariance during night combined with the clamping of the estimated cloud cover variable between 0 and 1.

## Learned vs True Insolation Pattern

We now have a look at the function we learned for the effect of insolation on the internal temperature, absent of clouds. Since this is a simulated example, we have access to the true function to compare with:

```@example THERMAL_NN
# Generate time points for one day
tod_test = LinRange(0.0f0, 24.0f0, 100)

# Compute true insolation (without clouds)
I_true = [true_insolation(t, 0.0f0) for t in tod_test]

# Compute learned insolation
I_learned = [compute_nn_insolation(t, params_opt) for t in tod_test]

# Plot comparison
plot(tod_test, I_true, label="True Insolation", lw=3, color=:blue)
plot!(tod_test, I_learned, label="Learned Insolation", lw=2, ls=:dash, color=:red)
xlabel!("Time of Day (hours)")
ylabel!("Insolation (W/m²)")
title!("Learned Solar Insolation Pattern")
vline!([6, 18], ls=:dot, color=:gray, alpha=0.5, label="Sunrise/Sunset")
```
Hopefully, we see that the estimation has captured the general shape of the true insolation pattern, but perhaps not perfectly, since this function is "hidden" behind an unknown and noisy estimate of the cloud cover.


## Discussion

This example demonstrates a classical SciML workflow, the combination of physics-based thermal dynamics with a data-driven model to capture unknown solar patterns. During the day, we were able to roughly estimate the cloud cover despite not being directly measured, by leveraging its effect on the temperature dynamics, but during night our estimator has no fighting chance of doing a good job here, a limitation inherent to the unobservability of the cloud cover in the absence of sunlight.



## Diving deeper: How to handle constraints
The variable cloud cover is constrained to be between 0 and 1. The Kalman-filtering framework does not handle such a constraint natively, but there are several different more or less heuristic methods available to handle it. Above, we simply clamped the estimated value to be between 0 and 1, simple but effective. Can we do any better? This section compares a number of different methods
- The clamping method
- Reformulating the dynamics to use an unconstrained variable that is projected onto the constraint set using a sigmoid function
- Projection implemented as a "perfect measurement": We may treat the projection as a fictitious measurement update, imagining that we have obtained a zero-variance measurement that the constrained variable is at the constraint boundary. This is similar to the naive clamping above, but also updates the covariance. We perform this projection using the function `LowLevelParticleFilters.project_bound` and make use of a callback in order to apply it during the estimation.

```@example THERMAL_NN
# Evaluation function to compare methods
function evaluate_solution(sol, data, params)
    # Extract estimated states
    T_est = [sol.xt[i][1] for i in 1:length(sol.xt)]
    cloud_est = [sol.xt[i][2] for i in 1:length(sol.xt)]
    T_true = [data.x[i][1] for i in 1:length(data.x)]
    cloud_true = [data.x[i][2] for i in 1:length(data.x)]
    
    # Compute errors
    temp_rmse = sqrt(mean(abs2, T_true .- T_est))
    cloud_rmse = sqrt(mean(abs2, cloud_true .- cloud_est))
    
    # Compute learned insolation vs true
    tod_test = LinRange(0.0f0, 24.0f0, 100)
    I_true = [true_insolation(t, 0.0f0) for t in tod_test]
    I_learned = [compute_nn_insolation(t, params) for t in tod_test]
    insolation_rmse = sqrt(mean(abs2, I_true .- I_learned))
    
    return (;
        temp_rmse,
        cloud_rmse,
        insolation_rmse,
        T_est,
        cloud_est,
        I_learned
    )
end

# Evaluate the clamping method (already optimized)
eval_clamping = evaluate_solution(sol, data, params_opt)
@info "Clamping method - Temperature RMSE: $(round(eval_clamping.temp_rmse, digits=3))°C, Cloud RMSE: $(round(eval_clamping.cloud_rmse, digits=3))"
```

## Comparison of Constraint Handling Methods

The cloud cover state must be constrained to lie in the interval [0, 1]. We have explored three different methods to handle this constraint:

1. **Clamping**: Directly clamp the cloud cover after each dynamics update
2. **Sigmoid transformation**: Transform the state through a sigmoid function  
3. **Projection**: Project the state back to the constraint set after filter updates

Let's compare these three approaches:

### Method 1: Clamping (Already Implemented)

The clamping method was used in the main tutorial above. It simply clips the cloud cover to [0, 1] after each dynamics step:

```julia
function clamped_dynamics(x,u,p,t)
    xp = discrete_dynamics_hybrid(x,u,p,t)
    SA[xp[1], clamp(xp[2], 0.0f0, 1.0f0)]
end
```

### Method 2: Sigmoid Transformation

```@example THERMAL_NN
# Sigmoid transformation method
sigmoid(x) = 1 / (1 + exp(-x))
sigmoid_inv(y) = log(y / (1 - y))

function thermal_dynamics_sigmoid(x, u, p, t)
    T_room, log_cloud_cover = x
    cloud_cover = sigmoid(log_cloud_cover)
    P_heater = u[1]
    
    # External temperature (known)
    T_ext = external_temp(t)
    
    # Solar insolation from RBF model
    I_base = compute_nn_insolation(t, p)
    I_solar = I_base * (1 - cloud_cover)
    
    # Heat balance
    dT_dt = (-k_loss * (T_room - T_ext) + η * P_heater + A_window * I_solar / 1000) / C_thermal
    
    # Cloud cover changes slowly (in transformed space)
    dlogcloud_dt = 0.0001f0*(sigmoid_inv(0.3f0) - log_cloud_cover)
    
    SA[dT_dt, dlogcloud_dt]
end

# Discretize sigmoid dynamics
discrete_dynamics_sigmoid = SeeToDee.ForwardEuler(thermal_dynamics_sigmoid, Ts)

# Measurement model for sigmoid method
measurement_sigmoid = if data.measure_cloud_cover
    (x, u, p, t) -> SA[x[1], sigmoid(x[2])]
else
    (x, u, p, t) -> SA[x[1]]
end

# Optimize sigmoid method
function cost_sigmoid(θ)
    T = eltype(θ)
    x0_sigmoid = SA[20.0f0, sigmoid_inv(0.5f0)]
    
    kf = UnscentedKalmanFilter(
        discrete_dynamics_sigmoid,
        measurement_sigmoid,
        R1,
        R2,
        SimpleMvNormal(T.(x0_sigmoid), T.(2*R1));
        ny, nu, Ts,
    )
    
    LowLevelParticleFilters.sse(kf, data.u, data.y, θ)
end

# Run optimization for sigmoid method
@info "Optimizing sigmoid method..."
result_sigmoid = Optim.optimize(
    cost_sigmoid,
    θ_init,
    BFGS(),
    Optim.Options(
        show_trace = false,
        iterations = 200,
        g_tol = 1e-12,
    );
    autodiff = :forward,
)

params_sigmoid = result_sigmoid.minimizer

# Evaluate sigmoid method
x0_sigmoid = SA[20.0f0, sigmoid_inv(0.5f0)]
kf_sigmoid = UnscentedKalmanFilter(
    discrete_dynamics_sigmoid,
    measurement_sigmoid,
    R1,
    R2,
    SimpleMvNormal(x0_sigmoid, R1);
    p = params_sigmoid,
    ny, nu, Ts
)

sol_sigmoid = forward_trajectory(kf_sigmoid, data.u, data.y)

# Transform cloud estimates back from log space
sol_sigmoid_transformed = deepcopy(sol_sigmoid)
for i in 1:length(sol_sigmoid_transformed.xt)
    x = sol_sigmoid_transformed.xt[i]
    sol_sigmoid_transformed.xt[i] = SA[x[1], sigmoid(x[2])]
end

eval_sigmoid = evaluate_solution(sol_sigmoid_transformed, data, params_sigmoid)
@info "Sigmoid method - Temperature RMSE: $(round(eval_sigmoid.temp_rmse, digits=3))°C, Cloud RMSE: $(round(eval_sigmoid.cloud_rmse, digits=3))"
```

### Method 3: Projection

```@example THERMAL_NN
# Projection method - uses standard dynamics but projects after updates
using ForwardDiff

function post_update_cb(kf, u, y, p, ll, e)
    if !(0 <= kf.x[2] <= 1)
        xn, Rn = LowLevelParticleFilters.project_bound(kf.x, kf.R, 2; lower=0, upper=1, tol=1e-9)
        kf.x = xn
        kf.R = Rn
    end
    nothing
end

# Use original unclamped dynamics for projection method
discrete_dynamics_unclamped = discrete_dynamics_hybrid5

# Optimize projection method
function cost_projection(θ)
    T = eltype(θ)
    
    kf = UnscentedKalmanFilter(
        discrete_dynamics_unclamped,
        measurement_fun,
        R1,
        R2,
        SimpleMvNormal(T.(x0), T.(2*R1));
        ny, nu, Ts,
    )
    
    LowLevelParticleFilters.sse(kf, data.u, data.y, θ; post_update_cb)
end

# Run optimization for projection method
@info "Optimizing projection method..."
result_projection = Optim.optimize(
    cost_projection,
    θ_init,
    BFGS(),
    Optim.Options(
        show_trace = false,
        iterations = 200,
        g_tol = 1e-12,
    );
    autodiff = :forward,
)

params_projection = result_projection.minimizer

# Evaluate projection method
kf_projection = UnscentedKalmanFilter(
    discrete_dynamics_unclamped,
    measurement_fun,
    R1,
    R2,
    SimpleMvNormal(x0, R1);
    p = params_projection,
    ny, nu, Ts
)

post_predict_cb(kf, p) = post_update_cb(kf, 0, 0, p, 0, 0)
sol_projection = forward_trajectory(kf_projection, data.u, data.y; post_predict_cb, post_correct_cb=post_predict_cb)

eval_projection = evaluate_solution(sol_projection, data, params_projection)
@info "Projection method - Temperature RMSE: $(round(eval_projection.temp_rmse, digits=3))°C, Cloud RMSE: $(round(eval_projection.cloud_rmse, digits=3))"
```

### Comparison Results

```@example THERMAL_NN
# Plot comparison of all three methods
p1 = plot(data.t, [data.x[i][2] for i in 1:length(data.x)], 
    label="True Cloud Cover", lw=3, color=:black, alpha=0.7)
plot!(data.t, eval_clamping.cloud_est, label="Clamping", lw=2, color=:blue)
plot!(data.t, eval_sigmoid.cloud_est, label="Sigmoid", lw=2, color=:red, ls=:dash)
plot!(data.t, eval_projection.cloud_est, label="Projection", lw=2, color=:green, ls=:dot)
ylabel!("Cloud Cover")
xlabel!("Time (hours)")
title!("Cloud Cover Estimation - Method Comparison")

# Compare learned insolation patterns
tod_test = LinRange(0.0f0, 24.0f0, 100)
I_true_plot = [true_insolation(t, 0.0f0) for t in tod_test]

p2 = plot(tod_test, I_true_plot, label="True", lw=3, color=:black, alpha=0.7)
plot!(tod_test, eval_clamping.I_learned, label="Clamping", lw=2, color=:blue)
plot!(tod_test, eval_sigmoid.I_learned, label="Sigmoid", lw=2, color=:red, ls=:dash)
plot!(tod_test, eval_projection.I_learned, label="Projection", lw=2, color=:green, ls=:dot)
xlabel!("Time of Day (hours)")
ylabel!("Insolation (W/m²)")
title!("Learned Insolation Patterns")

plot(p1, p2, layout=(2,1), size=(1200, 800))
```


#### Summary table
```@example THERMAL_NN
println("\n=== Method Comparison Summary ===")
println("Method      | Temp RMSE | Cloud RMSE | Insolation RMSE")
println("----------- | --------- | ---------- | ---------------")
println("Clamping    | $(round(eval_clamping.temp_rmse, digits=3))     | $(round(eval_clamping.cloud_rmse, digits=3))      | $(round(eval_clamping.insolation_rmse, digits=1))")
println("Sigmoid     | $(round(eval_sigmoid.temp_rmse, digits=3))     | $(round(eval_sigmoid.cloud_rmse, digits=3))      | $(round(eval_sigmoid.insolation_rmse, digits=1))")
println("Projection  | $(round(eval_projection.temp_rmse, digits=3))     | $(round(eval_projection.cloud_rmse, digits=3))      | $(round(eval_projection.insolation_rmse, digits=1))")
```

### Discussion of Constraint Methods

Each method has different trade-offs:

1. **Clamping**: Simple and computationally efficient, but creates discontinuities in the dynamics that can affect filter consistency.

2. **Sigmoid**: Smooth transformation that naturally keeps states bounded, but changes the noise characteristics and can make optimization more difficult due to the nonlinear transformation.

3. **Projection**: Maintains filter consistency by properly updating both mean and covariance, but requires additional computation and can be numerically sensitive.

The results show that all three methods achieve similar performance for this problem, with the choice depending on the specific requirements of your application regarding computational efficiency, theoretical guarantees, and implementation complexity.