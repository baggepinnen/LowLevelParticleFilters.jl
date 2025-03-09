# Unscented transform

In this section, we demonstrate how the unscented transform, used in the [`UnscentedKalmanFilter`](@ref), propagates a normal distribution through a nonlinear function. 




## Covariance propagation through nonlinear functions
The propagation of a Gaussian distribution through an affine (or linear) function ``f(x) = Ax+b`` is trivial, the distribution ``N(μ, Σ)`` is transformed into ``N(Aμ + b, AΣA^T)``, i.e., it remains Gaussian. This fact is what makes the standard [`KalmanFilter`](@ref) so computationally efficient. However, when the function is nonlinear, the transformation is not as straightforward and the posterior is generally not Gaussian. The unscented transform (UT) is a method to _approximate_ the transformation of a Gaussian distribution through a nonlinear function. The UT is based on the idea of propagating a set of _sigma points_ through the function and then computing the mean and covariance of the resulting distribution. Below, we demonstrate how a normal distribution is transformed through a number of nonlinear functions.

For comparison, we also show how the [`ExtendedKalmanFilter`](@ref) and [`ParticleFilter`](@ref) propagate the covariance. EKF uses linearization while particle filters propagate a large number of samples. We load the `ForwardDiff` package to compute the Jacobian of the function.

```@example UT
using DisplayAs # hide
using ForwardDiff, Distributions
function ekf_propagate_plot(f, μ, Σ; kwargs...)
    x = μ
    A = ForwardDiff.jacobian(f, x)
    μ = f(x)
    Σ = A * Σ * A'
    covplot!(μ, Σ; kwargs...)
end

function sample_propagate_plot(f, μ, Σ; kwargs...)
    xpart = rand(MvNormal(μ, Σ), 10000)
    ypart = f.(eachcol(xpart))
    scatter!(first.(ypart), last.(ypart); markerstrokealpha=0, markerstrokewidth=0, markeralpha=0.15, markersize=1, kwargs..., lab="")
    ym = mean(ypart)
    yS = cov(ypart)
    covplot!(ym, yS; kwargs...)
    scatter!([ym[1]], [ym[2]]; markersize=4, markershape=:x, kwargs..., lab="")
end
```

```@example UT
μ = [1.0, 2.0]
Σ = [1.0 0.5; 0.5 1.0]
x = sigmapoints(μ, Σ, TrivialParams())
n = length(x)
f1(x) = [x[1]^2+1, sin(x[2])]
y = f1.(x)
unscentedplot(x; lab="Input", c=:blue, fillalpha=0.1, kwargs...)
unscentedplot!(y; lab="Output UKF", c=:red, kwargs...)
ekf_propagate_plot(f1, μ, Σ; lab="Output EKF", c=:orange)
sample_propagate_plot(f1, μ, Σ; lab="Output particles", c=:green)
# Plot lines from each input point to each output point
plot!([first.(x)'; first.(y)'; fill(Inf, 1, n)][:], [last.(x)'; last.(y)'; fill(Inf, 1, n)][:], c=:black, alpha=0.5, primary=false)
DisplayAs.PNG(Plots.current()) # hide
```
For this first function, ``f_1(x) = [x_1^2+1, sin(x_2)]``, the UT and linearization-based propagation produce somewhat similar results, but the posterior distribution of the UT is much closer to the particle distribution than the EKF.

```@example UT
f2(x) = [x[1]*x[2], x[1]+x[2]]
y = f2.(x)
unscentedplot(x; lab="Input", c=:blue, fillalpha=0.1, kwargs...)
unscentedplot!(y; lab="Output UKF", c=:red, kwargs...)
ekf_propagate_plot(f2, μ, Σ; lab="Output EKF", c=:orange)
sample_propagate_plot(f2, μ, Σ; lab="Output particles", c=:green)
plot!([first.(x)'; first.(y)'; fill(Inf, 1, n)][:], [last.(x)'; last.(y)'; fill(Inf, 1, n)][:], c=:black, alpha=0.5, primary=false, xlims=(-5, 12))
DisplayAs.PNG(Plots.current()) # hide
```
For the second function, ``f_2(x) = [x_1 x_2, x_1+x_2]``, the posterior distribution is highly non-Gaussian. Both the UT and EKF style propagation do reasonable jobs capturing the posterior mean, but the UT does a better, although far from perfect, job at capturing the posterior covariance.


```@example UT
f3((x,y)) = [sqrt((1 - x)^2 + (0.1 - y)^2), atan(0.9 - y, 1.0 - x)] # Robot localization measurement model
y = f3.(x)
unscentedplot(x; lab="Input", c=:blue, fillalpha=0.1, kwargs...)
unscentedplot!(y; lab="Output UKF", c=:red, kwargs...)
ekf_propagate_plot(f3, μ, Σ; lab="Output EKF", c=:orange)
sample_propagate_plot(f3, μ, Σ; lab="Output particles", c=:green)
plot!([first.(x)'; first.(y)'; fill(Inf, 1, n)][:], [last.(x)'; last.(y)'; fill(Inf, 1, n)][:], c=:black, alpha=0.5, primary=false)
DisplayAs.PNG(Plots.current()) # hide
```

For the function ``f_3(x) = [\sqrt{(1 - x)^2 + (0.1 - y)^2}, \atan(0.9 - y, 1.0 - x)]``, the posterior distribution is once again highly non-Gaussian. The EKF misses to place any significant output probability mass in the region around the input, which the UT does by placing one sigma point in this region. When the particle distribution is approximated by a Gaussian, neither the UT or EKF does very well approximating this Gaussian.




## Tuning parameters

The unscented transform that underpins the [`UnscentedKalmanFilter`](@ref) may be tuned to adjust the spread of the points. By default, [`TrivialParams`](@ref) are used, but one may also opt for the [`WikiParams`](@ref) or [`MerweParams`](@ref) which are more commonly used in the literature.

The code snippets below demonstrate how to create different sets of parameters and visualizes the sigma points generated by each set of parameters for a trivial normal 2D distribution, as well has how the points propagate through a simple function.


We start by visualizing the sigma points generated by the different parameters sets using their default parameters.
```@example UT
using LowLevelParticleFilters, Plots
using LowLevelParticleFilters: sigmapoints
Plots.default(fillalpha=0.3) # This makes the covariance ellipse more transparent
kwargs = (; markersize=4, markeralpha=0.7, markerstrokewidth=0)
μ = [0.0, 0.0]
Σ = [1.0 0.0; 0.0 1.0]
wpars = WikiParams(α = 1, β = 0.0, κ = 1)
wxs = sigmapoints(μ, Σ, wpars)

mpars = MerweParams(α = 1e-3, β = 2.0, κ = 0.0)
mxs = sigmapoints(μ, Σ, mpars)

tpars = TrivialParams()
txs = sigmapoints(μ, Σ, tpars)

unscentedplot(wxs, wpars; lab="Wiki", c=:green, kwargs...)
unscentedplot!(mxs, mpars; lab="Merwe", c=:red, kwargs...)
unscentedplot!(txs, tpars; lab="Trivial", c=:blue, kwargs...)
```
In this plot, we hardly see the Merwe points because they are all very close to the origin due to the small default value of `α`. The Wiki points are more spread out, and the Trivial points are the most spread out. They all represent exactly the same distribution though, all their covariance ellipses overlap. Different sets of points can represent the same probability distribution by means of different weights that are assigned to each point.

Below, we demonstrate how the points propagate through a simple function. We use the function $f(x) = [\max(0, x[1]), x[2]]$ which is a simple function that forces the first state component to be positive.


```@example UT
f(x) = [max(zero(x[1]), x[1]), x[2]]
wxs2 = f.(wxs) # Propagate the points through the function
mxs2 = f.(mxs)
txs2 = f.(txs)
unscentedplot(wxs2, wpars; lab="Wiki", c=:green ,kwargs...)
unscentedplot!(mxs2, mpars; lab="Merwe", c=:red ,kwargs...)
unscentedplot!(txs2, tpars; lab="Trivial", c=:blue ,kwargs...)
```

We now see that the Merwe points resulted in an enormous covariance of the output, so large that the covariance ellipse for the other parameter sets aren't even visible in the plot. If we constrain the x-limits of the plot, we can have a better view of the other parameter sets
```@example UT
plot!(xlims=(-0.5, 2.5)) 
```
Here, we see that the posterior mean is skewed positively due to the clamping of ``f``, but the mean is skewed less for the trivial parameters than the Wiki parameters. 

By tweaking the parameters, we can obtain different behavior, below we show the spread of the points for different values of ``α, β, κ``

```@example UT
wpars = WikiParams(α = 5, β = -3.0, κ = 1)
wxs = sigmapoints(μ, Σ, wpars)

mpars = MerweParams(α = 2, β = -3.0, κ = 4.0)
mxs = sigmapoints(μ, Σ, mpars)

unscentedplot(wxs, wpars; lab="Wiki", c=:green, kwargs...)
unscentedplot!(mxs, mpars; lab="Merwe", c=:red, kwargs...)
unscentedplot!(txs, tpars; lab="Trivial", c=:blue, kwargs...)
```
This time, the Wiki and Merwe parameters are much more spread out than the Trivial parameters. When we propagate these points through the function:

```@example UT
wxs2 = f.(wxs)
mxs2 = f.(mxs)
txs2 = f.(txs)
unscentedplot(wxs2, wpars; lab="Wiki", c=:green, kwargs...)
unscentedplot!(mxs2, mpars; lab="Merwe", c=:red, kwargs...)
unscentedplot!(txs2, tpars; lab="Trivial", c=:blue, kwargs...)
```
we see that the Merwe parameters produced in a posterior mean closest to zero, but the Wiki parameters resulted in the smallest covariance estimate.