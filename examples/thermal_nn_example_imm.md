# Learning a disturbance model using SciML and IMM

In this example we will attempt to learn how an unknown and stochastic input, sun shining in through a window, influences a dynamical system, the temperature in a house. This example builds on the [Learning a disturbance model using SciML](@ref) tutorial, but uses an **Interacting Multiple Models (IMM) filter** to handle cloud cover uncertainty in a fundamentally different way.

## IMM Approach vs State-Variable Approach

In the original thermal example, cloud cover was modeled as a continuous state variable that evolved over time. Here, we take a different approach:

- **Two discrete modes**: One mode assumes clear sky (0% cloud cover), another assumes fully overcast conditions (100% cloud cover)
- **Mixing probabilities as belief**: The IMM filter maintains mixing probabilities `μ[1]` and `μ[2]` representing the belief that the current weather is in mode 1 (clear) or mode 2 (overcast)
- **Effective cloud cover estimate**: We can interpret `μ[2]` as an estimate of cloud cover, where values near 0 indicate clear sky and values near 1 indicate overcast conditions
- **No constraint handling needed**: Unlike the continuous state-variable approach, we don't need to worry about keeping cloud cover between 0 and 1—the mixing probabilities automatically satisfy this constraint

The IMM approach offers:
- **Advantages**: Natural probability constraints, robust mode switching, interpretable as discrete weather states
- **Limitations**: Discrete modes may not capture intermediate cloud conditions as smoothly as continuous states

## System Description

We consider the same simplified thermal model of a single-room house:
- State variable: room temperature `T` (note: no cloud cover state!)
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
- `I_solar`: solar insolation (W/m²), which depends on cloud cover

The key difference from the original example is that we don't explicitly model cloud cover as a state. Instead, the IMM filter will maintain a probabilistic belief about whether conditions are clear or overcast, and this belief will be updated based on how well each mode explains the observed temperature measurements.

## Data Generation

First, let's generate realistic thermal data with time-varying external conditions. We reuse the data generation from the original example to have ground truth for comparison:

```@example THERMAL_IMM
using LowLevelParticleFilters, Random, SeeToDee, StaticArrays, Plots, LinearAlgebra, Statistics
using LowLevelParticleFilters: SimpleMvNormal
using Optim
using ADTypes: AutoForwardDiff
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
plot!(t_plot, true_insolation.(t_plot, 1.0) ./ 100, label="100% Cloud Cover (×100 W/m²)", lw=2, ls=:dash)
xlabel!("Time (hours)")
title!("Daily Environmental Patterns")
```

Now let's define the true system dynamics and generate training data:

```@example THERMAL_IMM
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
function generate_thermal_data(; days=7)
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

    # Measurements - only temperature (no cloud cover measurements)
    y = [SA[x_i[1] + 0.1f0 * randn(rng)] for x_i in x]

    (; x, u, y, t, N, Ts)
end

# Generate data
data = generate_thermal_data(; days=14)

# Visualize the generated data
p1 = plot(data.t, [x[1] for x in data.x], label="Room Temperature", ylabel="Temperature (°C)")
plot!(data.t, [external_temp(t) for t in data.t], label="External Temperature", ls=:dash, alpha=0.7)

p2 = plot(data.t, [x[2] for x in data.x], label="Cloud Cover (true)", ylabel="Cloud Cover (0-1)", color=:orange)
p3 = plot(data.t, [u[1] for u in data.u], label="Heater Power", ylabel="Power (kW)", color=:red)

plot(p1, p2, p3, layout=(3,1), size=(900, 600), xlabel="Time (hours)")
```

## Radial Basis Function Model

We'll use a radial basis function expansion to learn the clear-sky solar insolation pattern. This function will be shared across both IMM modes:

```@example THERMAL_IMM
# Initialize RBF weights (parameters to be learned)
rng = Random.default_rng()
Random.seed!(rng, 456)
const n_basis = 8  # Number of basis functions
# Initialize with positive weights since insolation is always positive
rbf_weights = 100.0f0 * rand(Float32, n_basis)  # Random positive initialization

function basis_functions(t)
    tod = time_of_day(t)
    centers = LinRange(7.0f0, 17.0f0, n_basis) # Centers spread from 7 AM to 5 PM
    width = 1.5f0  # Width of each Gaussian basis function (in hours)
    @. exp(-((tod - centers) / width)^2)
end

# RBF evaluation function - computes clear-sky insolation
function compute_nn_insolation(t, weights)
    return weights'basis_functions(t) # Linear combination of basis functions
end

nothing # hide
```

## IMM Filter Setup

Now we define the two modes for our IMM filter. Each mode represents a different assumption about cloud cover:

**Mode 1**: Clear sky (0% cloud cover) - assumes full insolation effect
**Mode 2**: Overcast (100% cloud cover) - assumes zero insolation effect

Both modes share the same RBF parameters for learning the clear-sky insolation pattern:

```@example THERMAL_IMM
# Thermal dynamics for IMM - simplified state (temperature only)
# The 'cloud_mode' parameter determines which cloud assumption to use:
# cloud_mode = 0.0 → clear sky (0% cloud)
# cloud_mode = 1.0 → overcast (100% cloud)
function thermal_dynamics_imm(x, u, p, t, cloud_mode)
    T_room = x[1]  # Only temperature in state now
    P_heater = u[1]
    rbf_params = p  # RBF weights passed as parameters

    # External temperature (known)
    T_ext = external_temp(t)

    # Solar insolation from neural network, modulated by cloud assumption
    I_base = compute_nn_insolation(t, rbf_params)
    I_solar = I_base * (1 - cloud_mode)  # Mode determines cloud level

    # Heat balance
    dT_dt = (-k_loss * (T_room - T_ext) + η * P_heater + A_window * I_solar / 1000) / C_thermal

    SA[dT_dt]
end

# Discretize for each mode
discrete_dynamics_clear = SeeToDee.ForwardEuler((x,u,p,t)->thermal_dynamics_imm(x,u,p,t,0.0f0), Ts)
discrete_dynamics_overcast = SeeToDee.ForwardEuler((x,u,p,t)->thermal_dynamics_imm(x,u,p,t,1.0f0), Ts)

# Measurement model - we only measure temperature
C = SA[1.0f0;;]  # Measure temperature
ny = 1
nu = 1
nx = 1  # State dimension: just temperature

# Transition probability matrix - slow weather transitions (sticky modes)
P_transition = [0.9 0.1;   # 90% chance to stay clear, 10% to become overcast
                0.1 0.9]    # 10% chance to become clear, 90% to stay overcast

# Initial mixing probabilities - assume we start in clear conditions
μ_initial = [0.7, 0.3]  # 70% belief in clear, 30% in overcast

nothing # hide
```

## Parameter Estimation

Now we set up parameter estimation. We'll optimize the process noise, measurement noise, transition probability matrix, and RBF weights. The transition matrix has 4 elements but only 2 degrees of freedom since each row must sum to 1:

```@example THERMAL_IMM
# Initial state estimate
x0 = SA[20.0f0]  # Initial temperature guess

# Helper function to construct stochastic matrix from 2 parameters
# We use sigmoid to ensure probabilities are in (0, 1)
sigmoid(x) = 1 / (1 + exp(-x))

function build_transition_matrix(p_stay_clear_logit, p_stay_overcast_logit)
    p_stay_clear = sigmoid(p_stay_clear_logit)
    p_stay_overcast = sigmoid(p_stay_overcast_logit)
    [p_stay_clear (1-p_stay_clear);
     (1-p_stay_overcast) p_stay_overcast]
end

# Function to create IMM filter with given parameters
function get_imm_filter(θ)
    T = eltype(θ)

    # Extract parameters
    process_noise = exp10(θ[1])           # Process noise std for temperature
    meas_noise = exp10(θ[2])              # Measurement noise std
    p_stay_clear_logit = θ[3]             # Logit of P[1,1] (stay in clear)
    p_stay_overcast_logit = θ[4]          # Logit of P[2,2] (stay in overcast)
    rbf_params = θ[5:end]                 # RBF weights

    # Build transition probability matrix
    P_mat = SMatrix{2,2}(build_transition_matrix(p_stay_clear_logit, p_stay_overcast_logit))

    # Covariance matrices
    R1 = SMatrix{1, 1}(Diagonal([process_noise^2]))
    R2 = SMatrix{1, 1}(Diagonal([meas_noise^2]))

    # Measurement model
    measurement_model = LinearMeasurementModel(C, 0, R2; ny=ny)

    # Initial distribution
    d0 = SimpleMvNormal(T.(x0), T(2.0) .* R1)

    # Create two UKF models - one for each cloud assumption
    kf_clear = UnscentedKalmanFilter(
        discrete_dynamics_clear,
        measurement_model,
        R1,
        d0;
        ny, nu, Ts,
    )

    kf_overcast = UnscentedKalmanFilter(
        discrete_dynamics_overcast,
        measurement_model,
        R1,
        d0;
        ny, nu, Ts,
    )

    # Create IMM filter
    IMM([kf_clear, kf_overcast], P_mat, T.(μ_initial); p=rbf_params)
end

# Cost function for optimization
function cost(θ)
    try
        imm = get_imm_filter(θ)
        # Use log-likelihood for optimization
        ll = LowLevelParticleFilters.loglik(imm, data.u, data.y, interact=true)
        return -ll + 0.5*length(data.u)*(imm.P[2,1]^2+imm.P[1,2]^2)  # Minimize negative log-likelihood
    catch e
        return eltype(θ)(Inf)
    end
end

# Initial parameters: [log10(process_noise), log10(meas_noise), P_logits..., rbf_weights...]
# Start with initial guess from P_transition defined earlier
logit(x) = log(x / (1 - x))  # For inverse sigmoid
θ_init = vcat(
    log10(0.01),              # Process noise std
    log10(0.1),               # Measurement noise std
    logit(P_transition[1,1]), # Logit of stay-in-clear probability
    logit(P_transition[2,2]), # Logit of stay-in-overcast probability
    copy(rbf_weights)         # RBF weights
)

# Define optimization options
opt_options = Optim.Options(
    show_trace = true,
    store_trace = true,
    iterations = 2000,
    g_tol = 1e-12,
)

using Optim.LineSearches
@info "Starting optimization..."
result = Optim.optimize(
    cost,
    θ_init,
    # BFGS(alphaguess = LineSearches.InitialStatic(alpha=0.1), linesearch = LineSearches.BackTracking()),
    Newton(), # NOTE: Newton makes it converge to save solution as tutorial in docs, but BFGS gets stuck in slightly suboptimal minimum.
    # ParticleSwarm(),
    opt_options;
    autodiff = AutoForwardDiff(),  # Use forward-mode AD for gradients
)

params_opt = result.minimizer

@info "Optimization complete. Converged: $(Optim.converged(result)), Iterations: $(Optim.iterations(result))"
@info "Final cost: $(Optim.minimum(result))"

# Display optimized transition matrix
P_opt = build_transition_matrix(params_opt[3], params_opt[4])
@info "Optimized transition probability matrix:" P_opt
```

## Results Analysis

Let's analyze the results by running the IMM filter with optimized parameters:

```@example THERMAL_IMM
# Run IMM filter with optimized parameters
imm_final = get_imm_filter(params_opt)

sol = forward_trajectory(imm_final, data.u, data.y)

# Extract estimated states and mixing probabilities
T_est = [sol.xt[i][1] for i in 1:length(sol.xt)]
T_true = [data.x[i][1] for i in 1:length(data.x)]
cloud_true = [data.x[i][2] for i in 1:length(data.x)]

# Mixing probabilities: μ[1] = clear, μ[2] = overcast
μ_clear = sol.extra[1, :]
μ_overcast = sol.extra[2, :]

# Interpret μ_overcast as estimated cloud cover
cloud_est = μ_overcast

# Compute errors
temp_rmse = sqrt(mean(abs2, T_true .- T_est))
# Only compute cloud RMSE when sun is above horizon
sun_up_mask = [true_insolation(data.t[i], 0.0f0) > 0 for i in 1:length(data.t)]
cloud_rmse = sqrt(mean(abs2, cloud_true[sun_up_mask] .- cloud_est[sun_up_mask]))

@info "Temperature RMSE: $(round(temp_rmse, digits=3))°C"
@info "Cloud cover RMSE (daytime only): $(round(cloud_rmse, digits=3))"

# Plot temperature estimation
p1 = plot(data.t, T_true, label="True Temperature", lw=2, color=:blue)
plot!(data.t, T_est, label="Estimated Temperature", lw=2, ls=:dash, color=:red)
plot!(data.t, [y[1] for y in data.y], label="Measurements", alpha=0.3, seriestype=:scatter, ms=1, color=:gray)
ylabel!("Temperature (°C)")
title!("Temperature Estimation with IMM")

# Plot cloud cover: true vs IMM mixing probabilities
p2 = plot(data.t, cloud_true, label="True Cloud Cover", lw=2, color=:blue)
plot!(data.t, cloud_est, label="Estimated (μ_overcast)", lw=2, ls=:dash, color=:red)
plot!(data.t, μ_clear, label="μ_clear", lw=1, ls=:dot, color=:green, alpha=0.5)
ylabel!("Cloud Cover / Mixing Probability")
xlabel!("Time (hours)")
title!("Cloud Cover Estimation via IMM Mixing Probabilities")

plot(p1, p2, layout=(2,1), size=(1200, 800))
```

As we can see, the IMM filter is able to estimate the temperature accurately, and the mixing probability `μ_overcast` provides a reasonable estimate of cloud cover during daytime. Notice how the mixing probabilities drift during night when there is no sun—this is expected since it is impossible to observe cloud cover when there is no insolation effect. The IMM filter naturally expresses this uncertainty through mixing probabilities that become less decisive.

## Learned vs True Insolation Pattern

Let's examine the learned clear-sky insolation function:

```@example THERMAL_IMM
# Generate time points for one day
tod_test = LinRange(0.0f0, 24.0f0, 100)

# Compute true insolation (without clouds)
I_true = [true_insolation(t, 0.0f0) for t in tod_test]

# Compute learned insolation using optimized RBF weights
rbf_params_opt = params_opt[(1:length(rbf_weights)) .+ 4]  # Extract RBF weights from optimized params
I_learned = [compute_nn_insolation(t, rbf_params_opt) for t in tod_test]

# Plot comparison
plot(tod_test, I_true, label="True Clear-Sky Insolation", lw=3, color=:blue)
plot!(tod_test, I_learned, label="Learned Insolation (IMM)", lw=2, ls=:dash, color=:red)
xlabel!("Time of Day (hours)")
ylabel!("Insolation (W/m²)")
title!("Learned Solar Insolation Pattern")
```

The IMM filter has successfully learned the general shape of the clear-sky insolation pattern, which is shared across both modes and modulated by the mixing probabilities.


## Discussion

This example demonstrates using an **Interacting Multiple Models (IMM) filter** for state estimation with discrete mode switching, applied to learning an unknown disturbance model in a thermal system.

### Comparison: IMM vs State-Variable Approach

| Aspect | State-Variable Approach | IMM Approach |
|--------|------------------------|--------------|
| **Cloud representation** | Continuous state variable | Discrete modes + mixing probabilities |
| **Constraints** | Requires clamping/projection/sigmoid | Automatic (probabilities sum to 1) |
| **Intermediate values** | Naturally represents any value 0-1 | Represents via weighted average of extremes |
| **Interpretability** | Direct state estimate | Probabilistic belief over discrete scenarios |
| **Computational cost** | Single filter | Multiple filters (higher cost) |
| **Mode switching** | Smooth transitions | Explicit transition probabilities |

### Key Insights

1. **Observability**: Just like in the state-variable approach, cloud cover is only observable during daytime when solar insolation affects temperature. At night, mixing probabilities drift since both modes predict similar temperatures.

2. **Discrete modes work well**: Even though weather is truly continuous, the two-mode approximation (clear vs overcast) captures the essential dynamics reasonably well. The mixing probabilities provide a soft estimate between the extremes.

3. **No constraint handling needed**: Unlike the continuous state approach where we explored clamping, sigmoid transformations, projection, and truncated moment matching, the IMM approach naturally satisfies `0 ≤ μ[2] ≤ 1` through the probabilistic framework.

4. **Shared parameter learning**: The RBF model for clear-sky insolation is shared across both modes, making parameter estimation more efficient and ensuring consistency.

5. **Transition probabilities matter**: The sticky transition matrix (high diagonal values) reflects the physical reality that weather doesn't change instantaneously. This could be further optimized as a parameter if desired.

### When to Use IMM vs State-Variable Approach

**Use IMM when**:
- The system has distinct operating modes or regimes
- You want explicit probabilistic beliefs about which mode is active
- Constraint handling is challenging
- You need to model discrete switching phenomena

**Use state-variable approach when**:
- The hidden variable is truly continuous
- Smooth transitions are essential
- Computational efficiency is critical (single filter)
- You have good methods for constraint handling

Both approaches successfully learn the solar insolation pattern and estimate cloud cover, demonstrating the flexibility of Kalman-type filtering frameworks for scientific machine learning applications.
