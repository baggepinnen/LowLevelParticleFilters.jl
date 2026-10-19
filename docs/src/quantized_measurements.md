# Quantized measurements

Many sensors deliver their measurements with a limited resolution: a thermometer that reports whole degrees, an encoder that counts pulses, or a signal that is stored with few bits. In general, the quantizer acts on the output of a measurement function ``h(x, u, p, t)`` that may be nonlinear, as everywhere else in this package. To keep the exposition simple, this tutorial considers a linear measurement function ``h(x) = C x``, for which the posterior of a Gaussian prior and one quantized measurement is available in closed form; the extension to nonlinear measurement functions is discussed at the end. A sensor with resolution ``Δ`` then reports
```math
y_k = Δ \operatorname{round}\big((C x_k + v_k)/Δ\big), \quad v_k \sim N(0, σ_v^2),
```
i.e., the measurement does not determine the output ``C x_k + v_k`` but only the band ``[y_k - Δ/2, y_k + Δ/2]`` in which it lies. The likelihood of a measurement is thus the probability of this band, which is not Gaussian.

If the output varies over several quantization levels between samples, or if the noise ``v_k`` is large compared to ``Δ``, the quantization error is approximately uniformly distributed, independent between samples, and has variance ``Δ^2/12``. A Kalman filter with this variance added to the measurement noise then works well, see the section on quantization in [Missing data and outliers](@ref). This tutorial treats the case where this approximation fails: the uncertainty of the estimated output is smaller than ``Δ``, and the quantization error is strongly correlated in time. We compare the following estimators on a simple system:
- a Kalman filter with the quantization error modeled as white noise of variance ``Δ^2/12``,
- an unscented Kalman filter with the quantizer in the measurement function,
- an assumed-density filter that computes the exact posterior of a Gaussian prior and one quantized measurement and approximates it by a Gaussian (moment matching),
- a particle filter with the exact likelihood of the quantized measurement.

## The system
The state ``x_k`` follows a first-order system with a known input ``u_k`` that moves the state across several quantization levels, and the measurement is rounded to whole numbers. The process noise is small compared to the quantization step.
```@example quantized
using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Plots, Random, Statistics, Printf
using LowLevelParticleFilters: SimpleMvNormal
using Distributions: Normal, cdf, logcdf, logpdf
using DisplayAs # hide
Random.seed!(0)

a  = 0.98   # Pole of the system
Δ  = 1.0    # Quantization step
σw = 0.03   # Standard deviation of the process noise
σv = 0.01   # Standard deviation of the measurement noise before quantization
T  = 1000
A  = SA[a;;]
B  = SA[1-a;;]
C  = SA[1.0;;]
R1 = SA[σw^2;;]
d0 = SimpleMvNormal(SA[3.2], SA[σw^2/(1-a^2);;]) # Stationary distribution about the initial mean
quantize(z) = Δ .* round.(z ./ Δ)

u = [SA[3.2 + 1.5sin(2π*k/400)] for k in 1:T]
x = [d0.μ + sqrt(d0.Σ[1])*randn(SVector{1})]
for k in 1:T-1
    push!(x, A*x[k] + B*u[k] + σw*randn(SVector{1}))
end
y = [quantize(C*xk + σv*randn(SVector{1})) for xk in x]
xtrue = getindex.(x, 1)

plot(xtrue, label="State", xlabel="Time step")
plot!(getindex.(y, 1), label="Quantized measurement", seriestype=:steppost)
DisplayAs.PNG(Plots.current()) # hide
```
To compare the estimators, we compute the root-mean-square error of the filtered estimate and the fraction of time steps at which the true state lies in the interval of ±1.96 standard deviations about the estimate, which should be close to 0.95 for a consistent estimator.
```@example quantized
function evaluate(xh, σ)
    e = xtrue .- xh
    (; rmse = sqrt(mean(abs2, e)), coverage = mean(abs.(e) .<= 1.96 .* σ))
end
nothing # hide
```

## Kalman filter with the quantization error as white noise
The Kalman filter treats the quantization error as additional white measurement noise with variance ``Δ^2/12``:
```@example quantized
kf = KalmanFilter(A, B, C, 0, R1, SA[σv^2 + Δ^2/12;;], d0)
sol = forward_trajectory(kf, u, y)
xh_kf = getindex.(sol.xt, 1)
σ_kf  = sqrt.(getindex.(sol.Rt, 1))
evaluate(xh_kf, σ_kf)
```
Between two changes of the quantized value, the filter receives the same measurement at every time step and treats the errors of these measurements as independent. Its estimate is drawn towards the quantized value, and its covariance decreases as if each measurement carried new information, which makes the filter overconfident.

## Unscented Kalman filter with the quantizer in the measurement function
An [`UnscentedKalmanFilter`](@ref) with augmented measurement noise (the fourth type parameter `AUGM = true`) accepts a measurement function `h(x, u, p, t, e)` in which the measurement noise `e` enters before a nonlinearity, here the quantizer:
```@example quantized
h(x, u, p, t, e) = quantize(C*x + e)
ukf = UnscentedKalmanFilter{false, false, false, true}((x, u, p, t) -> A*x + B*u, h, R1, SA[σv^2;;], d0; ny=1, nu=1)
nothing # hide
```
The sigma points sample the quantizer, a piecewise constant function, at a few points. If all sigma points fall into the same band, the predicted measurement has zero variance, and the correction step either fails because the innovation covariance is singular, or leaves the estimate unchanged. If one sigma point falls into a neighboring band, the predicted measurement covariance is determined by the weight of that single point, and the correction may produce a covariance that is not positive definite. In the loop below, a failed correction is undone and counted, and so is a correction with a vanishing innovation variance:
```@example quantized
function ukf_trajectory(ukf, u, y)
    xh = zeros(length(y)); σ = zeros(length(y))
    nfailed = 0; nvanishing = 0; ll = 0.0
    for k in eachindex(y)
        x0, R0 = ukf.x, ukf.R
        try
            res = correct!(ukf, u[k], y[k])
            res.S[1] < 1e-10 && (nvanishing += 1)
            ukf.R[1] > 0 || error("Covariance not positive definite")
            ll += res.ll
        catch
            ukf.x, ukf.R = x0, R0
            nfailed += 1
        end
        xh[k], σ[k] = ukf.x[1], sqrt(ukf.R[1])
        predict!(ukf, u[k])
    end
    xh, σ, nfailed, nvanishing, ll
end
xh_ukf, σ_ukf, nfailed, nvanishing, ll_ukf = ukf_trajectory(ukf, u, y)
(; nfailed, nvanishing, ll_ukf, evaluate(xh_ukf, σ_ukf)...)
```
Most corrections fail or carry no information. The filter therefore uses the measurements essentially only when the quantized value changes, and ignores the information that the output remains in the band of the measured value at the other time steps. The log-likelihood `ll_ukf` accumulated over the successful corrections is dominated by the corrections with a nearly vanishing innovation variance and is meaningless; it cannot be used for parameter estimation.

## Assumed-density filter (moment matching)
For a linear measurement, the exact posterior of a Gaussian prior and one quantized measurement is available in closed form. With the prior ``x \sim N(\hat x, P)``, the output ``z = c^T x + v`` before quantization is Gaussian, ``z \sim N(μ, S)`` with ``μ = c^T \hat x`` and ``S = c^T P c + σ_v^2``. The measurement ``y`` states that ``z`` lies in the band ``[y - Δ/2, y + Δ/2]``, so the posterior of ``z`` is a normal distribution truncated to the band. With ``α = (y - Δ/2 - μ)/\sqrt{S}``, ``β = (y + Δ/2 - μ)/\sqrt{S}``, the standard normal density ``φ`` and distribution function ``Φ``, the probability of the band and the mean and variance of the truncated distribution are
```math
\begin{aligned}
Z &= Φ(β) - Φ(α), \\
E[z \mid y] &= μ + \sqrt{S}\, \frac{φ(α) - φ(β)}{Z}, \\
\operatorname{Var}[z \mid y] &= S \left(1 + \frac{α φ(α) - β φ(β)}{Z} - \left(\frac{φ(α) - φ(β)}{Z}\right)^2\right).
\end{aligned}
```
Since ``x`` and ``z`` are jointly Gaussian, the conditional distribution of ``x`` given ``z`` is Gaussian with a mean linear in ``z``, and the posterior mean and covariance of ``x`` follow from those of ``z``:
```math
\begin{aligned}
K &= P c / S, \\
\hat x^+ &= \hat x + K\big(E[z \mid y] - μ\big), \\
P^+ &= P - K \big(S - \operatorname{Var}[z \mid y]\big) K^T.
\end{aligned}
```
The only approximation is that the posterior is replaced by the Gaussian with the same mean and covariance. The contribution of the measurement to the log-likelihood is ``\log Z``. Several quantized outputs with independent noise are processed one at a time, since their likelihood factorizes. The probability of a band in the tails of the distribution is computed in logarithmic form to avoid cancellation:
```@example quantized
const N01 = Normal()
"log P(α < s < β) for s ~ N(0, 1)"
function logband(α, β)
    α > 0 && return logband(-β, -α) # Mirror to the lower tail
    if β < 0
        lb = logcdf(N01, β)
        lb + log(-expm1(logcdf(N01, α) - lb))
    else
        log(cdf(N01, β) - cdf(N01, α))
    end
end

"Log-probability, mean and variance of s ~ N(0, 1) truncated to (α, β)"
function truncated_moments(α, β)
    α > 0 && return ((lz, m, v) = truncated_moments(-β, -α); (lz, -m, v))
    lz = logband(α, β)
    pa = exp(logpdf(N01, α) - lz)
    pb = exp(logpdf(N01, β) - lz)
    m = pa - pb
    v = 1 + α*pa - β*pb - m^2
    lz, m, clamp(v, 1e-8, 1.0)
end

"""
    quantized_correct!(kf, u, y; Δ, σv)

Correction of a Kalman-type filter `kf` with linear measurement matrix `kf.C` by the measurement `y`,
quantized with step `Δ` after the addition of Gaussian noise with standard deviations `σv`.
Returns the log-likelihood of `y`.
"""
function quantized_correct!(kf, u, y; Δ, σv)
    x, P, Cm = kf.x, kf.R, kf.C
    ll = 0.0
    for i in eachindex(y)
        c = Cm[i, :]
        g = P*c
        μ = dot(c, x)
        S = dot(c, g) + σv[i]^2
        lz, m, v = truncated_moments((y[i] - Δ/2 - μ)/sqrt(S), (y[i] + Δ/2 - μ)/sqrt(S))
        K = g/S
        x = x + K*(sqrt(S)*m)
        P = P - (1 - v)*K*g'
        ll += lz
    end
    kf.x, kf.R = x, (P + P')/2
    ll
end
nothing # hide
```
The function updates the mean and covariance of a [`KalmanFilter`](@ref), whose prediction step is used unchanged:
```@example quantized
function adf_trajectory(kf, u, y; Δ, σv)
    reset!(kf)
    xh = zeros(length(y)); σ = zeros(length(y)); ll = 0.0
    for k in eachindex(y)
        ll += quantized_correct!(kf, u[k], y[k]; Δ, σv)
        xh[k], σ[k] = kf.x[1], sqrt(kf.R[1])
        predict!(kf, u[k])
    end
    xh, σ, ll
end
xh_adf, σ_adf, ll_adf = adf_trajectory(kf, u, y; Δ, σv = [σv])
evaluate(xh_adf, σ_adf)
```

## Particle filter
A particle filter represents the posterior by samples, and the likelihood of the quantized measurement, the probability of its band, enters through the measurement likelihood of an [`AdvancedParticleFilter`](@ref):
```@example quantized
dynamics(x, u, p, t, noise=false) = A*x + B*u + noise*σw*randn(SVector{1})
measurement(x, u, p, t, noise=false) = quantize(C*x + noise*σv*randn(SVector{1}))
function measurement_likelihood(x, u, y, p, t)
    z = C*x
    sum(logband((y[i] - Δ/2 - z[i])/σv, (y[i] + Δ/2 - z[i])/σv) for i in eachindex(y))
end
pf = AdvancedParticleFilter(2000, dynamics, measurement, measurement_likelihood, SimpleMvNormal(R1), d0)
solpf = forward_trajectory(pf, u, y)
xs = getindex.(solpf.x, 1)                  # Particles × time steps
xh_pf = vec(sum(solpf.we .* xs, dims=1))
σ_pf  = vec(sqrt.(sum(solpf.we .* (xs .- xh_pf') .^ 2, dims=1)))
evaluate(xh_pf, σ_pf)
```

## Comparison
```@example quantized
for (name, xh, σ) in [("Kalman filter, Δ²/12", xh_kf, σ_kf), ("UKF, quantizer in h", xh_ukf, σ_ukf),
                      ("Moment matching", xh_adf, σ_adf), ("Particle filter", xh_pf, σ_pf)]
    r = evaluate(xh, σ)
    @printf("%-22s RMSE %.4f   coverage of the 95 %% interval %.2f\n", name, r.rmse, r.coverage)
end
```
The moment-matching filter and the particle filter attain nearly the same error, with a coverage close to the nominal 0.95. The Kalman filter with the quantization noise has a larger error and is strongly overconfident: it is drawn towards the quantized value between the changes of the measurement. In this example, with small process noise, most of the information is contained in the changes of the quantized value, and the error of the unscented Kalman filter is only slightly larger than that of the moment-matching filter; it is, however, overconfident, and its likelihood cannot be used. The difference grows with the process noise, when the information that the output remains in its band becomes more important. The plot below shows the estimates in an interval in which the state crosses quantization thresholds.
```@example quantized
inds = 250:450
plot(inds, xtrue[inds], label="True state", l=(:black, 2), xlabel="Time step", legend=:bottomleft)
for (name, xh, σ) in [("Kalman filter, Δ²/12", xh_kf, σ_kf), ("UKF, quantizer in h", xh_ukf, σ_ukf),
                      ("Moment matching", xh_adf, σ_adf), ("Particle filter", xh_pf, σ_pf)]
    plot!(inds, xh[inds], ribbon=1.96σ[inds], fillalpha=0.15, label=name)
end
hline!([2.5, 3.5], l=(:dash, :gray), label="Quantization thresholds")
DisplayAs.PNG(Plots.current()) # hide
```

## Parameter estimation
The log-likelihood of the moment-matching filter, the sum of ``\log Z`` over the measurements, approximates the exact log-likelihood of the quantized data and can be used for maximum-likelihood estimation. Below, we compute the log-likelihood as a function of the standard deviation of the process noise ``σ_w`` for the moment-matching filter, the particle filter and the Kalman filter with the quantization noise:
```@example quantized
σws = σw .* exp2.(-2:0.25:2)
ll_adf_prof = map(σws) do σ
    kfσ = KalmanFilter(A, B, C, 0, SA[σ^2;;], SA[σv^2 + Δ^2/12;;], d0)
    adf_trajectory(kfσ, u, y; Δ, σv = [σv])[3]
end
ll_kf_prof = map(σws) do σ
    loglik(KalmanFilter(A, B, C, 0, SA[σ^2;;], SA[σv^2 + Δ^2/12;;], d0), u, y)
end
ll_pf_prof = map(σws) do σ
    dyn(x, u, p, t, noise=false) = A*x + B*u + noise*σ*randn(SVector{1})
    pfσ = AdvancedParticleFilter(2000, dyn, measurement, measurement_likelihood, SimpleMvNormal(SA[σ^2;;]), d0)
    Random.seed!(1) # The same random numbers for every value of σ
    loglik(pfσ, u, y)
end
plot(σws, [ll_adf_prof .- maximum(ll_adf_prof), ll_pf_prof .- maximum(ll_pf_prof), ll_kf_prof .- maximum(ll_kf_prof)],
    label=["Moment matching" "Particle filter" "Kalman filter, Δ²/12"], xscale=:log10, m=:o,
    xlabel="σw", ylabel="Log-likelihood − maximum", ylims=(-60, 5), legend=:bottomleft)
vline!([σw], l=(:dash, :black), label="True σw")
DisplayAs.PNG(Plots.current()) # hide
```
The maximum of the likelihood of the moment-matching filter is close to that of the particle filter and to the true value, while the Kalman filter with the quantization noise overestimates the process noise, since it explains the jumps of the quantized measurement at the thresholds by large state changes.

## Discussion
- The moment-matching filter has the cost of a Kalman filter. Its approximation is accurate when each measurement is weakly informative compared to the prior, such that the posterior is close to Gaussian, which is the case for quantized measurements of a system whose state is observed over many time steps.
- In Monte Carlo simulations of the system above over a range of process- and measurement-noise levels, the root-mean-square error of the moment-matching filter was within 1 % of that of a particle filter with ``10^5`` particles in every case, while that of the unscented Kalman filter with the quantizer in the measurement function was up to 74 % larger, and that of the Kalman filter with the quantization noise up to 96 % larger.
- In a study on a model of a geothermal borehole with a state dimension of 12 and two outputs quantized to whole degrees, the moment-matching filter attained the root-mean-square error of a particle filter with ``10^5`` particles to within 1 %, and its maximum-likelihood estimates were unbiased. The Gaussian-sum filter of [Cedeño et al. (2025)](https://arxiv.org/abs/2509.07837), which approximates the indicator function of the band by a Gaussian mixture, was marginally closer to the posterior of the particle filter, at a computational cost about four orders of magnitude larger.
- For a nonlinear measurement function, the mean, the covariance ``S`` of the output and the cross-covariance ``P c`` may be computed by linearization or by sigma points of the unquantized output; the update above then applies unchanged.
- If ``σ_v`` is large compared to ``Δ``, or the output changes by several quantization levels between samples, the Kalman filter with the variance ``Δ^2/12`` added to the measurement noise is adequate.
