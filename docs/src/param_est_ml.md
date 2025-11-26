# Maximum-likelihood and MAP estimation

Filters calculate the likelihood and prediction errors while performing filtering, this can be used to perform maximum likelihood estimation or prediction-error minimization. One can estimate all kinds of parameters using this method, in the example below, we will estimate the noise covariance. We may for example plot likelihood as function of the variance of the dynamics noise like this:


## Generate data by simulation
This simulates the same linear system as on the index page of the documentation
```@example ml_map
using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Distributions, Plots
nx = 2   # Dimension of state
nu = 2   # Dimension of input
ny = 2   # Dimension of measurements
N = 2000 # Number of particles

const dg = MvNormal(ny,1.0)          # Measurement noise Distribution
const df = MvNormal(nx,1.0)          # Dynamics noise Distribution
const d0 = MvNormal(@SVector(randn(nx)),2.0)   # Initial state Distribution

const A = SA[1 0.1; 0 1]
const B = @SMatrix [0.0 0.1; 1 0.1]
const C = @SMatrix [1.0 0; 0 1]

dynamics(x,u,p,t) = A*x .+ B*u
measurement(x,u,p,t) = C*x
vecvec_to_mat(x) = copy(reduce(hcat, x)') # Helper function
pf = ParticleFilter(N, dynamics, measurement, df, dg, d0)
xs,u,y = simulate(pf,300,df)
```

## Compute likelihood for various values of the parameters
Since this example looks for a single parameter only, we can plot the likelihood as a function of this parameter. If we had been looking for more than 2 parameters, we typically use an optimizer instead (see [Using an optimizer](@ref)).
```@example ml_map
p = nothing
svec = exp10.(LinRange(-0.8, 1.2, 60))
llspf = map(svec) do s
    df = MvNormal(nx,s)
    pfs = ParticleFilter(N, dynamics, measurement, df, dg, d0)
    loglik(pfs, u, y, p)
end
llspfaux = map(svec) do s
    df = MvNormal(nx,s)
    pfs = AuxiliaryParticleFilter(N, dynamics, measurement, df, dg, d0)
    loglik(pfs, u, y, p)
end
plot( svec, llspf,
    xscale = :log10,
    title = "Log-likelihood",
    xlabel = "Dynamics noise standard deviation",
    lab = "PF",
)
plot!(svec, llspfaux, yscale=:identity, xscale=:log10, lab="AUX PF", c=:green)
vline!([svec[findmax(llspf)[2]]], l=(:dash,:blue), primary=false)
```
the correct value for the simulated data is 1 (the simulated system is the same as on the front page of the docs).

We can do the same with a Kalman filter, shown below. When using Kalman-type filters, one may also provide a known state sequence if one is available, such as when the data is obtained from a simulation or in an instrumented lab setting. If the state sequence is provided, state-prediction errors are used for log-likelihood estimation instead of output-prediction errors.
```@example ml_map
eye(n) = SMatrix{n,n}(1.0I(n))
llskf = map(svec) do s
    kfs = KalmanFilter(A, B, C, 0, s^2*eye(nx), eye(ny), d0)
    # loglik(kfs, u, y, p)


    res3 = zeros((length(y[1])+1)*length(y))
    LowLevelParticleFilters.prediction_errors!(res3, kfs, u, y, loglik=true)
    -res3'res3

end
llskfx = map(svec) do s # Kalman filter with known state sequence, possible when data is simulated
    kfs = KalmanFilter(A, B, C, 0, s^2*eye(nx), eye(ny), d0)
    loglik_x(kfs, u, y, xs, p)
end
plot!(svec, llskf, yscale=:identity, xscale=:log10, lab="Kalman", c=:red)
vline!([svec[findmax(llskf)[2]]], l=(:dash,:red), primary=false)
plot!(svec, llskfx, yscale=:identity, xscale=:log10, lab="Kalman with known state sequence", c=:purple)
vline!([svec[findmax(llskfx)[2]]], l=(:dash,:purple), primary=false)
```

the result can be quite noisy due to the stochastic nature of particle filtering. The particle filter likelihood agrees with the Kalman-filter estimate, which is optimal for the linear example system we are simulating here, apart for when the noise variance is small. Due to particle depletion, particle filters often struggle when dynamics-noise is too small. This problem is mitigated by using a greater number of particles, or simply by not using a too small covariance.

## MAP estimation
Maximum a posteriori estimation (MAP) is similar to maximum likelihood (ML), but includes also prior knowledge of the distribution of the parameters in a way that is similar to parameter regularization. In this example, we will estimate the variance of the noises in the dynamics and the measurement functions.

To solve a MAP estimation problem, we need to define a function that takes a parameter vector and returns a filter, the parameters are used to construct the covariance matrices:
```@example ml_map
filter_from_parameters(θ, pf = nothing) = KalmanFilter(A, B, C, 0, exp(θ[1])^2*eye(nx), exp(θ[2])^2*eye(ny), d0) # Works with particle filters as well
nothing # hide
```

The call to `exp` on the parameters is so that we can define log-normal priors

```@example ml_map
priors = [Normal(0,2),Normal(0,2)]
```

Now we call the function `log_likelihood_fun` that returns a function to be minimized

```@example ml_map
ll = log_likelihood_fun(filter_from_parameters, priors, u, y, p)
nothing # hide
```

Since this is once again a low-dimensional problem, we can plot the LL on a 2d-grid

```@example ml_map
function meshgrid(a,b)
    grid_a = [i for i in a, j in b]
    grid_b = [j for i in a, j in b]
    grid_a, grid_b
end
Nv       = 20
v        = LinRange(-0.7,1,Nv)
llxy     = (x,y) -> ll([x;y])
VGx, VGy = meshgrid(v,v)
VGz      = llxy.(VGx, VGy)
heatmap(
    VGz,
    xticks = (1:Nv, round.(v, digits = 2)),
    yticks = (1:Nv, round.(v, digits = 2)),
    xlabel = "sigma v",
    ylabel = "sigma w",
) # Yes, labels are reversed
```
For higher-dimensional problems, we may estimate the parameters using an optimizer, e.g., Optim.jl. See [Using an optimizer](@ref) for examples, including how to maximize the log-likelihood using Gauss-Newton optimization.
