# Fault detection
This is also a video tutorial, available below:
```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/NgDcMuewPbI?si=6_bgIDiz9PFIE_gQ" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

# Fault detection using state estimation
This tutorial explores the use of a Kalman filter for fault detection in a thermal system
- Modeling
- Filtering
- Maximum-likelihood estimation of covariance and model parameters
- Monitor prediction-error Z-score to detect faults
    - A fault may be faulty sensor or unexpected temperature fluctuations


```@example FAULT_DETECTION
using DelimitedFiles, Plots, Dates
using LowLevelParticleFilters, LinearAlgebra, StaticArrays
using LowLevelParticleFilters: AbstractKalmanFilter, particletype, covtype,state,  covariance, parameters, KalmanFilteringSolution
using Optim
using ADTypes: AutoForwardDiff
using DisplayAs # hide
```

## Load data
From [kaggle.com/datasets/arashnic/sensor-fault-detection-data](https://www.kaggle.com/datasets/arashnic/sensor-fault-detection-data)

A time series of temperature measurements
```@example FAULT_DETECTION
using Downloads
url = "https://drive.google.com/uc?export=download&id=1zuIBaOhhrCxnifbvY7qJQTOyKWBDeBRh"
filename = "sensor-fault-detection.csv"
Downloads.download(url, filename)
raw_data = readdlm(filename, ';')
header = raw_data[1,:]
df = dateformat"yyyy-mm-ddTHH:MM:SS"
nothing # hide
```

The data is not stored in order

```@example FAULT_DETECTION
time_unsorted = DateTime.(getindex.(raw_data[2:end, 1], Ref(1:19)), df)
```

so we compute a sorting permutation that brings it into chronological order

```@example FAULT_DETECTION
perm = sortperm(time_unsorted)
time = time_unsorted[perm]
y = raw_data[2:end, 3][perm] .|> float
nothing # hide
```

`y` is the recorded temperature data.

## Look at the data

```@example FAULT_DETECTION
plot(time, y, ylabel="Temperature", legend=false)
DisplayAs.PNG(Plots.current()) # hide
```

```@example FAULT_DETECTION
timev = Dates.value.(time)  ./ 1000 # A numerical time vector, time was in milliseconds
plot(diff(timev), yscale=:log10, title="Time interval between measurement points", legend=false)
DisplayAs.PNG(Plots.current()) # hide
```
Samples are not evenly spaced (lots of missing data), but the interval is always a multiple of $(Ts)
```@example FAULT_DETECTION
intervals = sort(unique(diff(timev)))
intervals ./ intervals[1]
Ts = intervals[1]
nothing # hide
```

```@example FAULT_DETECTION
Tf = intervals[end] - intervals[1]
nothing # hide
```

We expand the data arrays such that we can treat them as having a constant sample interval, time points where there is no data available are indicated as `missing`

```@example FAULT_DETECTION
time_full = range(timev[1], timev[end], step=Ts)

available_inds = [findfirst(==(t), time_full) for t in timev]

y_full = fill(NaN, length(time_full))
y_full[available_inds] .= y
y_full = replace(y_full, NaN=>missing)
y_full = SVector{1}.(y_full)
nothing # hide
```

## Design Kalman filter
### Modeling

A simple model of temperature change is
```math
\dot T(t) = \alpha \big(T(t) - T_{env}(t)\big) + w(t)
```
Where ``T`` is the temperature of the system, ``T_{env}`` the temperature of the environment and ``w`` represents thermal energy added or removed by unmodeled sources.

Since we have no knowledge of ``T_{env}`` and ``w``, but we observe that they vary slowly, we add yet another state variable to the model corresponding to an integrating disturbance model:
```math
\begin{aligned}
\dot T(t) &= z(t) + b_T w_T(t) \\
\dot z(t) &=  b_z w_z(t)
\end{aligned}
```
This model is linear, and can be written on the form
```math
\begin{aligned}
\dot x &= Ax + Bw \\
y &= Cx + e
\end{aligned}
```
with ``A`` matrix 
```math
A = \begin{bmatrix}
0 & 1 \\
0 & 0
\end{bmatrix}
```
which, when discretized (assuming unit sample interval), becomes
```math
A = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix}
```


```@example FAULT_DETECTION
A,B,C,D = SA[1.0 1; 0 1], @SMatrix(zeros(2,0)), SA[1.0 0], 0;
nothing # hide
```
### Picking covariance matrices

```@example FAULT_DETECTION
R1 = 1e-4LowLevelParticleFilters.double_integrator_covariance(1) |> SMatrix{2,2}
R2 = SA[0.1^2;;]
d0 = LowLevelParticleFilters.SimpleMvNormal(SA[y[1], 0], SA[100.0 0; 0 0.1])
kf = KalmanFilter(A,B,C,D,R1,R2,d0; Ts)
```

### Perform filtering
When data is missing, we omit the call to `correct!`. We still perform the prediction step though.

```@example FAULT_DETECTION
function special_forward_trajectory(kf::AbstractKalmanFilter, u::AbstractVector, y::AbstractVector, p=parameters(kf))
    reset!(kf)
    T    = length(y)
    x    = Array{particletype(kf)}(undef,T)
    xt   = Array{particletype(kf)}(undef,T)
    R    = Array{covtype(kf)}(undef,T)
    Rt   = Array{covtype(kf)}(undef,T)
    e    = zeros(eltype(particletype(kf)), length(y))
	σs   = zeros(eltype(particletype(kf)), length(y))
    ll   = zero(eltype(particletype(kf)))
    S    = Vector{Any}(undef, T)
    K    = Vector{Any}(undef, T)
    for t = 1:T
        ti = (t-1)*kf.Ts
        x[t]  = state(kf)      |> copy
        R[t]  = covariance(kf) |> copy
		if !any(ismissing, y[t])
        	lli, ei, Si, Sᵪi, Ki = correct!(kf, u[t], y[t], p, ti)
			σs[t] = √(ei'*(Sᵪi\ei)) # Compute the Z-score
			e[t] = ei[]
			ll += lli
            S[t] = Sᵪi
            K[t] = Ki
		end
        xt[t] = state(kf)      |> copy
        Rt[t] = covariance(kf) |> copy
        predict!(kf, u[t], p, ti)
    end
    KalmanFilteringSolution(kf,u,y,x,xt,R,Rt,ll,vcat.(e),K,S), σs
end

u_full = [@SVector(zeros(0)) for y in y_full];

start = 1 # Change this value to display different parts of the data set
N = 1000  # Number of data points to include (to limit plot size in the docs, plot with Plots.plotly() and N = length(y_full) to see the full data set with the ability to zoom interactively in the plot)

sol, σs = special_forward_trajectory(kf, u_full[(1:N) .+ (start-1)], y_full[(1:N) .+ (start-1)])

sol.ll
```

#### Smoothing
For good measure, we also perform smoothing, computing
```math
x(k \,|\, T_f)
```
as opposed to filtering which is computing
```math
x(k \,|\, k)
```
or prediction
```math
x(k \,|\, k-1)
```

```@example FAULT_DETECTION
smoothsol = smooth(sol)
nothing # hide
```

### Visualize the filtered and smoothed trajectories

```@example FAULT_DETECTION
timevec = range(0, step=Ts, length=length(sol.y))

plot(smoothsol,
    plotx   = false, # prediction
    plotxt  = true,  # filtered
    plotxT  = true,  # smoothed
    plotRt  = true,
    plotRT  = true,
    plotyh  = false,
    plotyht = true,
    size = (650,600), seriestype = [:line :line :scatter :line], link = :x,
)
plot!(timevec, reduce(hcat, smoothsol.xT)[1,:], sp=3, label="Smoothed")
DisplayAs.PNG(Plots.current()) # hide
```

## Estimate the dynamics covariance using maximum-likelihood estimation (MLE)
Since we have a single parameter only, we may plot the loss landscape.

```@example FAULT_DETECTION
svec = exp10.(range(-5, -2, length=30)) # Covariance values to try

# Compute the log-likelihood for all covariance values
lls = map(svec) do s # 
	R1 = s*LowLevelParticleFilters.double_integrator_covariance(1) |> SMatrix{2,2}
	kf = KalmanFilter(A,B,C,D,R1,R2,d0; Ts)
	sol, σs = special_forward_trajectory(kf, u_full, y_full)
	sol.ll
end

plot(svec, lls, xscale=:log10, title="Log-likelihood estimation")
```

Get the covariance parameter associated with the maximum likelihood:

```@example FAULT_DETECTION
svec[argmax(lls)]
```

## Optimize "friction" and covariance jointly
We can add some damping to the velocity state in the double-integrator model. When doing so, we should also estimate the full covariance matrix of the dynamics noise. This gives us an estimation problem with 1 + 3 parameters, 3 for the triangular part of the covariance matrix Cholesky factor. Estimating the Cholesky factor instead of the full covariance matrix yields fewer optimizaiton variables and ensures that the result is a valid, positive definite and symmetric covariance matrix. To ensure that the "friction parameter" is positive, we optimize the ``\log`` of the parameter.

A double integrator has the dynamics matrix
```math
\begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix}
```
By modifying this to
```math
\begin{bmatrix}
1 & 1 \\
0 & \alpha
\end{bmatrix}
```
where ``0 \leq \alpha \leq 1``, we can add some damping to the velocity, i.e., if no force is acting on it it will eventually slow down to velocity zero. It's not quite correct to call the parameter ``\alpha`` a "damping term", the formulation ``\beta = 1 - \alpha`` would be closer to an actual discrete-time damping factor.


```@example FAULT_DETECTION
function triangular(x)
    m = length(x)
    n = round(Int, sqrt(2m-1))
    T = zeros(eltype(x), n, n)
    k = 1
    for i = 1:n, j = i:n
        T[i,j] = x[k]
        k += 1
    end
    T
end

invtriangular(T) = [T[i,j] for i = 1:size(T,1) for j = i:size(T,1)]

params = log.([invtriangular(cholesky(R1).U); 1])

function get_opt_kf(logp)
	T = eltype(logp)
	p = exp.(logp)
	R1c = triangular(p[1:3]) |> SMatrix{2,2}
	R1 = R1c'R1c + 1e-8I
	vel = p[4]
	vel > 1 && (return T(Inf))
	A = SA[1 1; 0 vel]
	d0T = LowLevelParticleFilters.SimpleMvNormal(T.(d0.μ), T.(d0.Σ + 0.01I))
	kf = KalmanFilter(A,B,C,D,R1,R2,d0T; Ts, check=false)
end

function cost(logp)
	try
		kf = get_opt_kf(logp)
		soli, σs = special_forward_trajectory(kf, u_full, y_full)
		return -soli.ll
	catch e
		return eltype(logp)(Inf)
	end
end

cost(params)
```

### Optimize

```@example FAULT_DETECTION
res = Optim.optimize(
    cost,
    params,
    LBFGS(),
    Optim.Options(
        show_trace        = true,
        show_every        = 5,
        iterations        = 1000,
		x_tol 			  = 1e-7,
    ),
	autodiff = AutoForwardDiff(),
)
get_opt_kf(res.minimizer).R1
```

The initial guess was 
```@example FAULT_DETECTION
R1
```

Compare optimized parameter vector with initial guess:
```@example FAULT_DETECTION
exp.([params res.minimizer])
```

### Visualize optimized filtering trajectory



```@example FAULT_DETECTION
kf2 = get_opt_kf(res.minimizer)
sol2, σs2 = special_forward_trajectory(kf2, u_full[(1:N) .+ (start-1)], y_full[(1:N) .+ (start-1)])

smoothsol2 = smooth(sol2, kf2, sol2.u, sol2.y)

plot(smoothsol2, plotx=false, plotxt=true, plotRt=true, plotyh=false, plotyht=true, size=(650,600), seriestype=[:line :line :scatter :line], link=:x)
plot!(timevec, reduce(hcat, smoothsol2.xT)[1,:], sp=3, label="Smoothed")

outliers = findall(σs2 .> 5)
vline!([timevec[outliers]], sp=3)
DisplayAs.PNG(Plots.current()) # hide
```

## Fault detection
We implement a simple fault detector using Z-scores. When the Z-score is higher than 4, we consider it a fault.

```@example FAULT_DETECTION
plot(timevec, σs2); hline!([1 2 3 4], label=false)
DisplayAs.PNG(Plots.current()) # hide
```
(change the value of the variable `start` to see different parts of the data set, e.g., set `start = 30_000`)

Z-scores may not capture large outliers if they occur when the estimator is very uncertain
Does Z-score correlate with "velocity", i.e., are faults correlated with large continuous slopes in the data?
```@example FAULT_DETECTION
sol_full, σs_full = special_forward_trajectory(kf2, u_full, y_full)
scatter(abs.(getindex.(sol_full.xt, 2)), σs_full, ylabel="Z-score", xlabel="velocity")
DisplayAs.PNG(Plots.current()) # hide
```
not really, it looks like large Z-scores can appear even when the estimated velocity is small.

### Alternative fault-detection strategies

In this tutorial, we used the Z-score of the prediction error to detect faults. A Kalman filter, being a statistical estimator, maintains a _belief_ about the state of the system, whenever this belief is inconsistent with fault-free operation, we may experiencing a fault. Below are some alternative ways in which we can detect faults using a Kalman filter:

- A single measurement has a Z-score larger than a threshold. The benefit of this approach is that it can isolate issues to a single sensor.
- The entire measurement vector has a large Z-score. This can detect issues that cause unexpected correlation in the output, but where each individual output looks as expected on its own.
- The filter may be augmented with a _disturbance model_. If the estimated disturbance is larger than expected, e.g., significantly different from zero, it may indicate a fault. See [How to tune a Kalman filter](@ref) and [Disturbance gallery](@ref) for more information on how to do this.
- Parameters of the system may be modeled as time-varying and estimated online. If, e.g., an estimated gain parameter decreases significantly, it may indicate a fault. This is similar in spirit to adding a disturbance model, but instead of estimating an input disturbance, we estimate a property of the system. See [Joint state and parameter estimation](@ref) for an example of this.
- An article suggesting several consistency checks similar to the Z-score check used here is "New Kalman filter and smoother consistency tests" by Gibbs, all of which can be readily computed from the quantities saved in the `KalmanFilteringSolution` object and the result of `smooth`. One suggestion is to use the filter error and associated filter-error covariance instead of the prediction error, another one is similar but using a smoothed error instead. The last suggestion is to use the smoothed stat error in a similar check.

## Summary
- A state estimator can indicate faults when the error is larger than _expected_
- What is _expected_ is determined by the model


The notebook used in the tutorial is available here:
- [`identification_12_fault_detection.jl` on GitHub](https://github.com/baggepinnen/notebooks/blob/main/system_identification/identification_12_fault_detection.jl)