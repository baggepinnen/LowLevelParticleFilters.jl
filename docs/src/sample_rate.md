# Influence of sample rate on performance

Naturally, if we sample more often, we obtain more information about the system and can thus expect better performance. Frequent sampling allows for an averaging effect that can mitigate the influence of measurement noise. Sampling "frequently enough" is also important in order to rely on theoretical concepts such as observability when analyzing what modes of the system can be accurately recovered. To understand this, consider an extreme case: a particle subject to a random force where we can measure the position of the particle (this is a double integrator system). If we measure the position of the particle often, we can infer both the position and the velocity of the particle, the system is _observable_. However, if we measure the position rather infrequently, we can't say much about the velocity of the particle since the noisy force driving the particle will have had a large influence on the velocity, which isn't directly measured, since the last measurement of the position. When analyzing observability traditionally, we may compute
- The observability matrix. This can tell us the theoretical observable and unobservable subspaces of the system. One may take measurement noise into account by scaling the outputs to have equal variance, but one can not take driving noise properties into account.
- The observability Gramian. This gives us a measure of how well we can estimate modes of the system in a balanced realization from the available measurements, but there is once again no way to take the driving noise into account.

To estimate the _practical observability_ of the system we may instead consider an analysis of the _stationary error covariance_ of a state estimator. For a linear-Gaussian system observed with a Kalman filter, the stationary Kalman gain and error covariance is obtained by solving an algebraic Riccati equation:
```math
\begin{align}
x^+ &= Ax + w\\
y &= Cx + e \\
\\
x̂^+ &= Ax̂ + K(y - Cx̂)  \qquad &\text{estimator}\\
\\
ε &= x - x̂ \qquad &\text{prediction error}\\
ε^+ &= Ax + w - \big(Ax̂ + K(y - Cx̂) \big) \qquad &\text{prediction error dynamics}\\
ε^+ &= Ax + w - \big(Ax̂ + K(Cx + e - Cx̂) \big)\\
ε^+ &= (A - KC)ε + w - Ke \\
[E(we^T) &= 0] \Longrightarrow \\
E\{(w - Ke)(w - Ke)^T\} &= R_1 + K R_2 K^T
\end{align}
```
The stationary covariance of the prediction error ``ε(t+1|t)``, ``R_∞(t+1|t) = E_∞(ε(t+1|t)ε(t+1|t)^T)``, is automatically computed by the solver of the algebraic Riccati equation that computes the stationary Kalman gain ``K``.

By incorporating the measurement, we form a _filtering estimate_ ``ε(t|t)`` and in doing so, reduce the covariance of the prediction error according to ``R_∞(t|t) = (I - KC)R_∞(t|t-1)``. 


!!! note
    Due to the exact formulation of ``K`` returned by the Riccati solver in MatrixEquations.jl, we must either use ``A^{-1}K`` or compute ``K = RC^T (R_2 + C R C^T)^{-1}`` ourselves. `MatrixEquations.ared` solves the Riccati equation corresponding to the filter form, but returns the ``K`` matrix for the prediction form. 

```@example SAMPLERATE
using ControlSystemsBase
import ControlSystemsBase.MatrixEquations

function kalman_are(sys::AbstractStateSpace{<:Discrete}, R1, R2)
    A,B,C,D = ssdata(sys)
    R∞, p, K, args... = MatrixEquations.ared(A', C', R2, R1)
    K', R∞
end
```


To perform an analysis of the performance as a function of the sample rate, we will assume that we have a continuous-time LTI system with a continuous-time Gaussian noise process driving the system. This will allow us to discretize both the system dynamics and noise process at varying sample rates and compute the stationary Kalman filter and associated stationary covariance matrix (see [Discretization](@ref) for more details). We assume that the measurement noise is a _discrete-time_ noise process with a fixed covariance, representing a scenario where the sensor equipment is predetermined but the sample rate is not. 


## Example 1: Double integrator
A double integrator can be thought of as a particle subject to a force, the continuous-time dynamics are ``\ddot x = w``.
```@example SAMPLERATE
using ControlSystemsBase, Plots, Test
sysc = ss([0 1; 0 0], [0; 1], [1 0], 0) # Continuous-time double integrator
R1c = [0 0; 0 1]                        # Continuous-time process noise covariance
R2  = [1;;]                             # Measurement noise covariance

Ts    = 1                               # Sample interval
sysd  = c2d(sysc, Ts)                   # Discretize the system
R1d   = c2d(sysc, R1c, Ts)              # Discretize the process noise covariance
K, R∞ = kalman_are(sysd, R1d, R2)       # Compute the stationary Kalman gain and covariance
R∞
```

Does the computed stationary covariance matrix match the expected covariance matrix we derived above? 
```@example SAMPLERATE
A,B,C,D = ssdata(sysd)
@test lyap(Discrete, A-K*C, R1d + K*R2*K') ≈ R∞
```



For this system, we expect the stationary _filtering covariance_ ``R_∞(t|t)`` to go to zero for small ``T_s`` since the system is observable. For large ``T_s``, we expect the variance of the position estimate to approach the variance of the measurement noise, i.e., we don't expect the model to be of any use if it's forced to predict for too long. The variance of the velocity estimate is expected to go to infinity since the disturbance force has increasingly more time to affect the velocity and we cannot measure this. Let's investigate:
```@example SAMPLERATE
Tss = exp10.(LinRange(-3, 3, 30))
R∞s = map(Tss) do Ts
    sysd    = c2d(sysc, Ts)
    A,B,C,D = ssdata(sysd)
    R1d     = c2d(sysc, R1c, Ts)
    AK, R∞  = kalman_are(sysd, R1d, R2)

    # diag((I-A\AK*C)*R∞) # This also works

    K = (R∞*C')/(R2 + C*R∞*C')
    diag((I-K*C)*R∞)
end
plot(Tss, reduce(hcat, R∞s)', label=["\$σ^2 p\$" "\$σ^2 v\$"], xlabel="Sample interval [s]", ylabel="Stationary filtering variance", title="Double integrator", xscale=:log10, yscale=:log10)
hline!(R2, label="\$R_2\$", linestyle=:dash, legend=:bottomright)
```
The plot confirms our expectations. Note: this plot shows the _filtering covariance_, the prediction-error covariance ``R_∞(t+1|t)`` would in this case go to infinity for both state variables.


## Example 2: Double integrator with friction
If we take the same system as above, but introduce some friction in the system, we expect similar behavior for small sample intervals, but for large sample intervals we expect the stationary variance of the velocity to converge to a finite value due to the dissipation of energy in the system:
```@example SAMPLERATE
sysc = ss([0 1; 0 -0.02], [0; 1], [1 0], 0) # Continuous-time double integrator with friction

R∞s = map(Tss) do Ts
    sysd    = c2d(sysc, Ts)
    A,B,C,D = ssdata(sysd)
    R1d     = c2d(sysc, R1c, Ts)
    AK, R∞  = kalman_are(sysd, R1d, R2)
    K       = (R∞*C')/(R2 + C*R∞*C')
    diag((I-K*C)*R∞)
end
plot(Tss, reduce(hcat, R∞s)', label=["\$σ^2 p\$" "\$σ^2 v\$"], xlabel="Sample interval [s]", ylabel="Stationary filtering variance", title="Double integrator with friction", xscale=:log10, yscale=:log10)
hline!(R2, label="\$R_2\$", linestyle=:dash, legend=:bottomright)
```
Nice, it did.

If we instead look at the prediction-error covariance, we see the opposite behavior
```@example SAMPLERATE
R∞s = map(Tss) do Ts
    sysd   = c2d(sysc, Ts)
    R1d    = c2d(sysc, R1c, Ts)
    AK, R∞ = kalman_are(sysd, R1d, R2)
    diag(R∞)
end
plot(Tss, reduce(hcat, R∞s)', label=["\$σ^2 p\$" "\$σ^2 v\$"], xlabel="Sample interval [s]", ylabel="Stationary filtering variance", title="Double integrator with friction", xscale=:log10, yscale=:log10)
hline!(R2, label="\$R_2\$", linestyle=:dash, legend=:bottomright)
```
in this case, the variance of the position prediction goes to infinity for large ``T_s`` since the position dynamics still contain a pure integrator. The velocity prediction variance still converges to a finite value, the same finite value as the filtering variance, which also happens to be the same value as we get by computing the stationary covariance of the noise filtered through the system without any Kalman filter. This indicates that the estimator is completely useless for large sample intervals and we can't predict the velocity any better than its long-term average:
```@example SAMPLERATE
velocity_dynamics = ss(-0.02, 1, 1, 0)
R∞s[end][end] ≈ lyap(velocity_dynamics, [1;;])[]
```