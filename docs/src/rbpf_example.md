# Rao-Blackwellized particle filter

This example will demonstrate use of the Rao-Blackwellized particle filter ([`RBPF`](@ref)), also called "Marginalized particle filter".

This filter is effectively a particle filter where each particle is a Kalman filter that is responsible for the estimation of a linear sub structure.

The filter assumes that the dynamics follow "model 2" in the article ["Marginalized Particle Filters for Mixed Linear/Nonlinear State-space Models" by Thomas Schön, Fredrik Gustafsson, and Per-Johan Nordlund](https://people.isy.liu.se/rt/schon/Publications/SchonGN2004.pdf), i.e., the dynamics is described by
```math
\begin{align}
    x_{t+1}^n &= f_n(x_t^n, u, p, t) + A_n(x_t^n, u, p, t) x_t^l + w_t^n, \quad &w_t^n \sim \mathcal{N}(0, R_1^n) \\
    x_{t+1}^l &= A(...) x_t^l + Bu + w_t^l, \quad &w_t^l \sim \mathcal{N}(0, R_1^l) \\
    y_t &= g(x_t^n, u, p, t) + C(...) x_t^l + e_t, \quad &e_t \sim \mathcal{N}(0, R_2)
\end{align}
```
where ``x^n`` is a subset of the state that has nonlinear dynamics or measurement function, and ``x^l`` is a subset of the state where both dynamics and measurement function are linear and Gaussian. The entire state vector is represented by a special type [`RBParticle`](@ref) that behaves like the vector `[xn; xl]`, but stores `xn, xl` and the covariance `R` of `xl` separately.

!!! warning "Experimental"
    This filter is currently considered experimental and the user interface may change in the future without respecting semantic versioning.


Below, we define all functions and matrices that are needed to perform marginalized particle filtering for the dynamical system
```math
\begin{align}
x_{t+1}^n &= \arctan x_t^n + \begin{pmatrix} 1 & 0 & 0 \end{pmatrix} x_t^1 + w_t^n, \tag{1a} \\
x_{t+1}^1 &= \begin{pmatrix}
1 & 0.3 & 0 \\
0 & 0.92 & -0.3 \\
0 & 0.3 & 0.92
\end{pmatrix} x_t^1 + w_t^1, \tag{1b} \\
y_t &= \begin{pmatrix}
0.1(x_t^n)^2 \operatorname{sgn}(x_t^n) \\
0
\end{pmatrix} + \begin{pmatrix}
0 & 0 & 0 \\
1 & -1 & 1
\end{pmatrix} x_t^1 + e_t, \tag{1c} \\
\text{where}\\
w_t &= \begin{pmatrix}
w_t^n \\
w_t^1
\end{pmatrix} \sim \mathcal{N}(0, 0.01I_{4\times 4}), \tag{1d} \\
e_t &\sim \mathcal{N}(0, 0.1I_{2\times 2}), \tag{1e} \\
x_0^n &\sim \mathcal{N}(0, 1), \tag{1f} \\
x_0^1 &\sim \mathcal{N}(0_{3\times 1}, 0_{3\times 3}). \tag{1g}
\end{align}
```

Since this is a tracking problem without control inputs, and there are no parameters and time dependence, we define functions with the signature `fn(xn, args...)` to handle the fact that the filter will pass empty arguments for inputs, parameters and time.

Below, we define functions that return the matrix ``A_n`` despite that it is constant, we do this to illustrate that this matrix may in general be a function of the nonlinear state, parameter and time. If the matrix is constant, it's okay to let `An` be a `Matrix` or `SMatrix` instead of a function.

```@example RBPF
using LowLevelParticleFilters, LinearAlgebra
using LowLevelParticleFilters: SimpleMvNormal
using DisplayAs # hide
nxn = 1         # Dimension of nonlinear state
nxl = 3         # Dimension of linear state
nx  = nxn + nxl # Total dimension of state
nu  = 0         # Dimension of control input
ny  = 2         # Dimension of measurement
N   = 200       # Number of particles
fn(xn, args...) = atan.(xn)         # Nonlinear part of nonlinear state dynamics
An  = [1.0 0.0 0.0]     # Linear part of nonlinear state dynamics
Al  = [1.0  0.3   0.0;  # Linear part of linear state dynamics (the standard Kalman-filter A matrix). It's defined as a matrix here, but it can also be a function of (x, u, p, t)
                   0.0  0.92 -0.3; 
                   0.0  0.3   0.92] # 3x3 matrix
Cl = [0.0  0.0 0.0; 
      1.0 -1.0 1.0]    # 2x3 measurement matrix
g(xn, args...) = [0.1 * xn[]^2 * sign(xn[]), 0.0] # 2x1 vector

Bl = zeros(nxl, nu)

# Noise parameters
R1n = [0.01;;]          # Scalar variance for w^n
R1l = 0.01 * I(3)       # 3x3 covariance for w^l
R2  = 0.1 * I(2)         # 2x2 measurement noise (shared between linear and nonlinear parts)

# Initial states (xn ~ N(0,1), xl ~ N(0, 0.01I))
x0n = zeros(nxn)
R0n = [1.0;;]
x0l = zeros(nxl)
R0l = 0.01 * I(nxl)

d0l = SimpleMvNormal(x0l, R0l)
d0n = SimpleMvNormal(x0n, R0n)

kf    = KalmanFilter(Al, Bl, Cl, 0, R1l, R2, d0l; ny, nu) # Since we are providing a function instead of a matrix for C, we also provide the number of outputs ny
mm    = RBMeasurementModel(g, R2, ny)
names = SignalNames(x=["\$x^n_1\$", "\$x^l_2\$", "\$x^l_3\$", "\$x^l_4\$"], u=[], y=["\$y_1\$", "\$y_2\$"], name="RBPF") # For nicer labels in the plot
pf    = RBPF(N, kf, fn, mm, R1n, d0n; nu, An, Ts=1.0, names)

# Simulate some data from the filter dynamics
u     = [zeros(nu) for _ in 1:100]
x,u,y = simulate(pf, u)

# Perform the filtering
sol = forward_trajectory(pf, u, y)

using Plots
plot(sol, size=(800,600), xreal=x, markersize=1, nbinsy=50, colorbar=false)
for i = 1:nx
    plot!(ylims = extrema(getindex.(x, i)) .+ (-1, 1), sp=i)
end
current()
DisplayAs.PNG(Plots.current()) # hide
```
The cyan markers represent the true state in the state plots, and the measurements in the measurement plots. The heatmap represents the particle distribution. Note, since each particle has an additional covariance estimate for the linear sub state, the heatmaps for the linear sub state are constructed by drawing a small number of samples from this marginal distribution. Formally, the marginal distribution over the linear sub state is a gaussian-mixture model where the weight of each gaussian is the weight of the particle. This fact is not taken into account when the heat map for the predicted measurement is constructed, so interpret this heatmap with caution.


In this example, we made use of standard julia arrays for the dynamics and covariances etc., for optimum performance (the difference may be dramatic), make use of static arrays from [StaticArrays.jl](https://github.com/JuliaArrays/StaticArrays.jl). 

The paper referenced above mention a lot of special cases in which the filter can be simplified, it's worth a read if you are considering using this filter.

```julia
using Distributions
we = sol.we[:, end]
sum(we)
x = getindex.(sol.x[:, end], 2)
Rv = [sol.x[i, end].R[1,1] for i = 1:num_particles(pf)]

components = [Normal(x[i], sqrt(Rv[i])) for i = 1:num_particles(pf)]
prior = we
D = Distributions.MixtureModel(components, prior)

plot(D)
```