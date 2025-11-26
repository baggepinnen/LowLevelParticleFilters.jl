# Identifiability

There is no guarantee that we will recover the true parameters by perfoming parameter estimation, especially not if the input excitation is poor. For the quad-tank system used in [Using an optimizer](@ref), we will generally find parameters that results in a good predictor for the system (this is after all what we're optimizing for), but these may not be the "correct" parameters.

## Polynomial methods
A tool like [StructuralIdentifiability.jl](https://github.com/SciML/StructuralIdentifiability.jl) may be used to determine the identifiability of parameters and state variables (for rational systems), something that for the quad-tank system could look like
```julia
using StructuralIdentifiability

ode = @ODEmodel(
    h1'(t) = -a1/A1 * h1(t) + a3/A1*h3(t) +     gam*k1/A1 * u1(t),
    h2'(t) = -a2/A2 * h2(t) + a4/A2*h4(t) +     gam*k2/A2 * u2(t),
    h3'(t) = -a3/A3*h3(t)                 + (1-gam)*k2/A3 * u2(t),
    h4'(t) = -a4/A4*h4(t)                 + (1-gam)*k1/A4 * u1(t),
	y1(t) = h1(t),
    y2(t) = h2(t),
)

local_id = assess_local_identifiability(ode)
```
where we have made the substitution ``\sqrt h \rightarrow h`` due to a limitation of the tool (it currently only handles rational ODEs). The output of the above analysis is
```julia
julia> local_id = assess_local_identifiability(ode)
Dict{Nemo.fmpq_mpoly, Bool} with 15 entries:
  a3  => 0
  gam => 1
  k2  => 0
  A4  => 0
  h4  => 0
  h2  => 1
  A3  => 0
  a1  => 0
  A2  => 0
  k1  => 0
  a4  => 0
  h3  => 0
  h1  => 1
  A1  => 0
  a2  => 0
```

indicating that we can not hope to resolve all of the parameters. However, using appropriate regularization from prior information, we might still recover a lot of information about the system. Regularization could easily be added to the function `cost` in [Using an optimizer](@ref), e.g., using a penalty like `(p-p_guess)'Γ*(p-p_guess)` for some matrix ``\Gamma``, to indicate our confidence in the initial guess.

## Linear methods
This package also contains an interface to [ControlSystemsBase](https://juliacontrol.github.io/ControlSystems.jl/stable/), which allows you to call `ControlSystemsBase.observability(f, x, u, p, t)` on a filter `f` to linearize (if needed) it in the point `x,u,p,t` and assess observability using linear methods (the PHB test). Also `ControlSystemsBase.obsv(f, x, u, p, t)` for computing the observability matrix is available.

## Fisher Information and Augmented State Covariance

When using augmented-state methods for joint state and parameter estimation (see [Joint state and parameter estimation](@ref)), we embed the parameters as additional state variables, often with zero process noise, i.e.

```math
z_k =
\begin{bmatrix}
x_k \\
p
\end{bmatrix},
\qquad
z_{k+1} =
\underbrace{\begin{bmatrix}
A_k & A^{(p)}_k \\
0   & I
\end{bmatrix}}_{A_k^{\text{aug}}}
z_k +
\begin{bmatrix}
w_k \\ 0
\end{bmatrix},
\qquad
y_k = C_k x_k + e_k,
```

where

* $x_k$ = original system state
* $p$ = constant parameters (no process noise)
* $w_k \sim \mathcal{N}(0,R_1)$ = process noise driving the state
* $e_k \sim \mathcal{N}(0,R_2)$ = measurement noise
* $A^{(p)}_k = \frac{\partial f}{\partial p}\big|_{(x_k,u_k,p)}$ encodes the parameter influence
* Parameters evolve as $p_{k+1} = p_k$, modeled here as a random walk with zero covariance

---

### Connection to Fisher Information

Running an Extended or Unscented Kalman Filter on this augmented system produces a covariance matrix

```math
R_k =
\begin{bmatrix}
R_{xx,k} & R_{xp,k} \\
R_{xp,k}^\top & R_{pp,k}
\end{bmatrix}.
```

The block ($R_{pp,k}$) represents the *covariance of the parameter estimates* at time index ``k``.
For *constant parameters* ($R_1^p = 0$), this parameter covariance evolves by accumulating information from measurements:

```math
R_{pp,k+1}^{-1}
=
R_{pp,k}^{-1} +
J_k^\top S_k^{-1} J_k,
```

where

* $J_k = C_k S^{(p)}_k$ is the *output sensitivity* to parameters,
* $S_k = C_k R_{xx,k} C_k^\top + R_2$ is the *innovation covariance*,
* $S^{(p)}_k = \frac{\partial x_k}{\partial p}$ is the *state sensitivity* satisfying the recursion (from the chain rule applied to the nonlinear dynamics)

  ```math
  S^{(p)}_{k+1} = A_k S^{(p)}_k + A^{(p)}_k, \qquad S^{(p)}_0 = 0.
  ```


If we define the *(trajectory) Fisher Information Matrix (FIM)* as

```math
\mathcal{I}(p) =
\sum_{k=1}^N J_k^\top S_k^{-1} J_k,
```

then it follows that

```math
R_{pp,N}^{-1}
=
R_{pp,0}^{-1} + \mathcal{I}(p).
```

This shows a connection between the Fisher Information Matrix and the parameter covariance arising from augmented-state filtering:

- The FIM measures how much _information_ the data contains about the parameters
- When parameters are constant ($R_1^p = 0$), the *augmented Kalman filter accumulates FIM over time*
- The *Cramér–Rao lower bound* becomes

```math
\mathrm{cov}(\hat{p}) \succeq \mathcal{I}(p)^{-1},
```


### Summary

In joint state–parameter estimation with constant parameters, the parameter covariance block $R_{pp,k}$ of the augmented Kalman filter (or EKF/UKF approximation) decreases as information is accumulated according to the *Fisher information matrix.
This link provides a principled way to analyze *parameter identifiability* and *experiment excitation design* using information theory.


This relationship is useful for understanding how well parameters can be estimated from data, and it explains why *insufficient excitation* or *poor observability* leads to *slow decay of $R_{pp,k}$* and unreliable parameter estimates in practice. The FIM is also equal to the Hessian of the negative log-likelihood function at the optimum found when performing maximum-likelihood parameter estimation.

