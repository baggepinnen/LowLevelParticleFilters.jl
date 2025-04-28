# Noise-adaptive Kalman filter
In this tutorial we will consider filtering of a 1D position track, similar in spirit to what one could have obtained from a GPS device, but limited to 1D for easier visualization. We will use a constant-velocity model, i.e., use a double integrator,
```math
\begin{aligned}
x_{k+1} &= \begin{bmatrix} 1 & T_s \\ 0 & 1 \end{bmatrix} x_k + \begin{bmatrix} T_s^2/2 \\ T_s \end{bmatrix} w_k \\
y_k &= \begin{bmatrix} 1 & 0 \end{bmatrix} x_k + v_k
\end{aligned}
```
where $w_k \sim \mathcal{N}(0, σ_w)$ is the process noise, and $v_k \sim \mathcal{N}(0, R_2)$ is the measurement noise, and illustrate how we can make use of an adaptive noise covariance to improve the filter performance.

## Data generation
We start by generating some position data that we want to perform filtering on. The "object" we want to track is initially stationary, and transitions to moving with a constant velocity after a while. 

```@example ADAPTIVE_KALMAN
using LowLevelParticleFilters, Plots, Random
Random.seed!(1)

# Create a time series for filtering
x = [zeros(50); 0:100]
T = length(x)
Y = x + randn(T)
plot([Y x], lab=["Measurement" "True state to be tracked"], c=[1 :purple])
```

## Simple Kalman filtering

We will use a Kalman filter to perform the filtering. The model is a double integrator, i.e., a constant-acceleration model. The state vector is thus $x = [p, v]^T$, where $p$ is the position and $v$ is the velocity. When designing a Kalman filter, we need to specify the noise covariances $R_1$ and $R_2$. While it's often easy to measure the covariance of the measurement noise, ``R_2``, it can be quite difficult to know ahead of time what the dynamics noise covariance, ``R_1``, should be. In this example, we will use an adaptive filter, where we will increase the dynamics noise covariance if the filter prediction error is too large. However, we first run the filter twice, once with a large ``R_1`` and once with a small ``R_1`` to illustrate the difference.

```@example ADAPTIVE_KALMAN
y = [[y] for y in Y] # create a vector of vectors for the KF
u = fill([], T) # No inputs in this example :(

# Define the model
Ts = 1
A = [1 Ts; 0 1]
B = zeros(2, 0)
C = [1 0]
D = zeros(0, 0)
R2 = [1;;]

σws = [1e-2, 1e-5] # Dynamics noise standard deviations

fig = plot(Y, lab="Measurement")
for σw in σws
    R1 = σw*[Ts^3/3 Ts^2/2; Ts^2/2 Ts] # The dynamics noise covariance matrix is σw*Bw*Bw' where Bw = [Ts^2/2; Ts]
    kf = KalmanFilter(A, B, C, D, R1, R2)
    measure = LowLevelParticleFilters.measurement(kf)
    yh = [measure(state(kf), u[1], nothing, 1)] 
    for t = 1:T # Main filter loop
        kf(u[t], y[t]) # Performs both prediction and correction
        xh = state(kf)
        yht = measure(xh, u[t], nothing, t)
        push!(yh, yht)
    end

    Yh = reduce(hcat, yh)
    plot!(Yh', lab="Estimate \$σ_w\$ = $σw")
end
fig
```
When ``R_1`` is small (controlled by ``σ_w``), we get a nice and smooth filter estimate, but this estimate clearly lags behind the true state. When ``R_1`` is large, the filter estimate is much more responsive, but it also has a lot of noise.

## Adaptive noise covariance

Below, we will implement an adaptive filter, where we keep the dynamics noise covariance low by default, but increase it if the filter prediction error is too large. We will use a Z-score to determine if the prediction error is too large. The Z-score is defined as the number of standard deviations the prediction error is away from the estimated mean. This time around we use separate [`correct!`](@ref) and [`predict!`](@ref) calls, so that we can access the prediction error as well as the prior covariance of the prediction error, ``S``. ``S`` (or the Cholesky factor ``Sᵪ``) will be used to compute the Z-score.

When implementing behavior such as time varying covariance, we may either implement the filtering loop manually, like we do below, or make use of the callback functionality available in [`forward_trajectory`](@ref), which we do in the next code snippet.

```@example ADAPTIVE_KALMAN
σw = 1e-5 # Set the covariance to a low value by default
R1 = σw*[Ts^3/3 Ts^2/2; Ts^2/2 Ts]
kf = KalmanFilter(A, B, C, D, R1, R2)
measure = LowLevelParticleFilters.measurement(kf)

# Some arrays to store simulation data
yh = []
es = Float64[]
σs = Float64[]
for t = 1:T # Main filter loop
    ll, e, S, Sᵪ = correct!(kf, u[t], y[t], nothing, t) # Manually call the prediction step
    xh = state(kf)
    yht = measure(xh, u[t], nothing, t)

    σ = √(e'*(Sᵪ\e)) # Compute the Z-score
    push!(es, e[]) # Save for plotting
    push!(σs, σ)
    if σ > 3 # If the Z-score is too high
        # we temporarily increase the dynamics noise covariance by 1000x to adapt faster
        predict!(kf, u[t], nothing, t; R1 = 1000kf.R1) 
    else
        predict!(kf, u[t], nothing, t)
    end

    push!(yh, yht)
end

Yh = reduce(hcat, yh)
plot([Y Yh'], lab=["Measurement" "Adaptive estimate"])
```
Not too bad! This time the filter estimate is much more responsive during the transition, but exhibits favorable noise properties during the stationary phases. We can also plot the prediction error and the Z-score to see how the filter adapts to the dynamics noise covariance.

```@example ADAPTIVE_KALMAN
plot([es σs], lab=["Prediction error" "Z-score"])
```

Notice how the prediction errors, that should ideally be centered around zero, remain predominantly negative for a long time interval after the transition. This can be attributed to an overshoot in the velocity state of the estimator, but the rapid decrease of the covariance after the transition makes the filter slow at correcting its overshoot. If we want, we could mitigate this and make the adaptation even more sophisticated by letting the covariance remain large for a while after a transition in operating mode has been detected. Below, we implement a simple version of this, where we use a multiplier ``σ_{wt}`` that defaults to 1, but is increase to a very large value of 1000 if a transition is detected. When no transition is detected, ``σ_{wt}`` is decreased exponentially back down to 1.

As mentioned above, in this code snippet we make use of the callback functionality of [`forward_trajectory`](@ref) rather than implementing the filtering loop manually, we thus add the logic for modifying the covariance in the `pre_predict_cb` callback function. 

```@example ADAPTIVE_KALMAN
σw  = 1e-5 # Set the covariance to a low value by default
σwt = 1.0
R1  = σw*[Ts^3/3 Ts^2/2; Ts^2/2 Ts]
kf  = KalmanFilter(A, B, C, D, R1, R2)
measure = LowLevelParticleFilters.measurement(kf)

function pre_predict_cb(kf, u, y, p, t, ll, e, S, Sᵪ)
    σ = √(e'*(Sᵪ\e)) # Compute the Z-score
    global σwt
    if σ > 3 # If the Z-score is too high
        σwt = 1000.0 # Set the R1 multiplier to a very large value
    else
        σwt = max(0.9σwt, 1.0) # Decrease exponentially back to 1
    end
    push!(σs, σ)
    push!(σwts, σwt)
    σwt*kf.R1 # The pre_predict_cb may return either nothing (operate through side effects) or a modified R1 matrix to use for this particular time step. Here, we make use of both approaches.
end

# Some arrays to store simulation data
σs = Float64[]
σwts = Float64[]

sol = forward_trajectory(kf, u, y; pre_predict_cb)
es = reduce(vcat, sol.e) # Extract prediciton errors
Yh = reduce(hcat, measure.(sol.xt, sol.u, nothing, nothing)) # Extract predicted outputs
plot([Y Yh'], lab=["Measurement" "Adaptive estimate"])
```

```@example ADAPTIVE_KALMAN
plot([es σs σwts], lab=["Prediction error" "Z-score" "\$σ_{wt}\$ multiplier"], layout=2, sp=[1 1 2])
```
This time, the prediction errors look more like white noise centered around zero after the initial transient caused by the transition.

## Summary
This tutorial demonstrated simple Kalman filtering for a double integrator without control inputs. We saw how the filtering estimate could be improved by playing around with the covariance matrices of the estimator, helping it catch up to fast changes in the behavior of the system without sacrificing steady-state noise properties.

In this case, we handled the modification of ``R_1`` outside of the filter, implementing our own filtering loop. Some applications get away with instead providing time-varying matrices in the form of a 3-dimension array, where the third dimension corresponds to time, or instead of providing a matrix, providing a function ``R_1(x, u, p, t)`` allows the matrix to be a function of state, input, parameters and time. These options apply to all matrices in the filter, including the dynamics matrices, ``A,B,C,D``.

Lastly, we mention the ability of the [`KalmanFilter`](@ref) to act like a recursive least-squares estimator, by setting the "forgetting factor ``α>1`` when creating the [`KalmanFilter`](@ref). ``α>1`` will cause the filter will exhibit exponential forgetting similar to an RLS estimator, in addition to the covariance inflation due to R1. It is thus possible to get a RLS-like algorithm by setting ``R_1 = 0, R_2 = 1/α`` and ``α > 1``.

## Disturbance modeling and noise tuning
See [this notebook](https://juliahub.com/pluto/editor.html?id=ad9ecbf9-bf83-45e7-bbe8-d2e5194f2240) for a blog post about disturbance modeling and noise tuning using LowLevelParticleFilter.jl