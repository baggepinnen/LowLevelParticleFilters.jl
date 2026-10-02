# Parameter Estimation

State estimation is an integral part of many parameter-estimation methods. Below, we will illustrate several different methods of performing parameter estimation. We can roughly divide the methods into three families:

1. Methods that optimize the likelihood or the prediction errors of a state estimator with respect to the model parameters.
2. Methods that add the parameters to be estimated as state variables in the model and estimate them using standard state estimation.
3. Methods that draw samples from the posterior distribution of the parameters, using the likelihood computed by a state estimator.

From the first family, we provide functionality for maximum-likelihood and MAP estimation in [Maximum-likelihood and MAP estimation](@ref), for estimation of the noise covariance matrices in [Estimating noise covariances](@ref), and for prediction-error minimization in [Using an optimizer](@ref "Prediction-Error minimization using an optimizer"), [Joint estimation of plant parameters and noise covariances](@ref) and [Multi-step prediction-error estimation](@ref). An example of the second family, joint state and parameter estimation, is provided in [Joint state and parameter estimation](@ref), and the third family is covered in [Bayesian inference](@ref).

## Which method should I use?

The following questions, considered in order, indicate which method to use for a particular problem.

1. **Is a linear black-box model sufficient, or are the data given in the frequency domain?** In this case, the methods of [ControlSystemIdentification.jl](https://baggepinnen.github.io/ControlSystemIdentification.jl/dev/), such as `subspaceid` and `newpem`, are often the most efficient. An identified model can be converted to a [`KalmanFilter`](@ref) by `KalmanFilter(model, x0)`, the noise covariance matrices of which may subsequently be refined using [`autotune_covariances`](@ref).
2. **Is the model a ModelingToolkit model?** [LowLevelParticleFiltersMTK.jl](https://baggepinnen.github.io/LowLevelParticleFiltersMTK.jl/dev/) constructs state estimators from ModelingToolkit models, and provides functionality for setting a subset of the model parameters from a parameter vector, which is required for the methods on these pages.
3. **What is unknown?**
    - Only the noise covariance matrices: maximum-likelihood or MAP estimation, see [Estimating noise covariances](@ref).
    - Only the parameters of the dynamics and measurement models (plant parameters), and the noise covariances are known with reasonable accuracy: prediction-error minimization, see [Using an optimizer](@ref "Prediction-Error minimization using an optimizer").
    - Both plant parameters and noise covariances: maximum-likelihood estimation of all parameters, see [Joint estimation of plant parameters and noise covariances](@ref).
    - Parameters that vary in time, or an estimate that is updated online as data arrives: joint state and parameter estimation, see [Joint state and parameter estimation](@ref) and [Joint state and parameter estimation using MUKF](@ref).
    - The full posterior distribution of the parameters: [Bayesian inference](@ref).
4. **What is the model used for?** The criterion should reflect the intended use of the model. For filtering and anomaly detection, the likelihood is the appropriate criterion. For short-horizon prediction, one-step prediction errors are appropriate. For models used for prediction over a longer horizon, such as the prediction model of an MPC controller, or for systems with slow dynamics relative to the sample interval, multi-step prediction errors are appropriate, see [Multi-step prediction-error estimation](@ref). For open-loop simulation of a stable system, a simulation-error criterion is appropriate, which is obtained as a limiting case of the prediction-error criteria, see the section on filtering, prediction and simulation below.
5. **Are there missing samples, outliers or multiple experiments?** See [Missing data and outliers](@ref) and the section on multiple experiments in [Joint estimation of plant parameters and noise covariances](@ref).
6. **Are the parameters identifiable from the data?** See [Identifiability](@ref). This should be investigated before the estimates are interpreted.

The methods demonstrated in this section have the following properties:

| Method | Plant parameters | Noise covariances | Time-varying parameters | Online estimation | Uncertainty estimate |
|:-------|:-----------------|:------------------|:------------------------|:------------------|:---------------------|
| Maximum likelihood and MAP | 🟢 | 🟢 | 🟥 | 🟥 | Hessian of the negative log-likelihood |
| One-step prediction-error minimization | 🟢 | 🟥 | 🟥 | 🟥 | Gauss-Newton approximation |
| Multi-step prediction-error minimization | 🟢 | 🟥 | 🟥 | 🟥 | not provided |
| Joint state and parameter estimation | 🔶 | 🟥 | 🟢 | 🟢 | Covariance of the augmented state |
| Bayesian inference | 🟢 | 🟢 | 🟥 | 🟥 | Samples from the posterior |

When trying to optimize parameters of the noise distributions, most commonly the covariance matrices, maximum likelihood (or MAP) is the recommended method, since prediction-error criteria do not penalize overconfident or underconfident covariance estimates. When parameters are time varying or an online estimate is required, joint state and parameter estimation is the applicable method. When fitting time-invariant plant parameters, all methods are applicable. In this case joint state and parameter estimation tends to be inefficient and unnecessarily complex, and it is recommended to opt for maximum likelihood or prediction-error minimization. Prediction-error minimization (PEM) with a Gauss-Newton optimizer is often the most efficient method for this type of problem.

Maximum-likelihood estimation tends to yield an estimator with better estimates of the posterior covariance since this is explicitly optimized for, while PEM tends to produce the smallest possible prediction errors.

## Filtering, prediction and simulation

A model estimated by minimizing one-step prediction errors of a state estimator is optimized for filtering and short-horizon prediction. The measurements correct the state estimate in every time step, and model errors may thus be compensated for by the state estimator. Two quantities determine where between one-step prediction and open-loop simulation the criterion is located:

- The dynamics noise covariance ``R_1``: as ``R_1 \rightarrow 0``, the Kalman gain tends to zero after an initial transient, and the one-step prediction errors tend to the errors of an open-loop simulation from the estimated initial state.
- The prediction horizon ``h`` of a multi-step prediction-error criterion: as ``h`` increases, the predictions are made over a longer horizon without correction, and the criterion tends to a simulation-error criterion.

A simulation-error criterion is appropriate only if the system is stable and not subject to significant unmeasured disturbances. For unstable systems, and for systems with unmeasured disturbances, prediction-error criteria remain applicable since the state estimator stabilizes the predictor. See [Multi-step prediction-error estimation](@ref) for a demonstration.

## Related packages
- [ControlSystemIdentification.jl](https://baggepinnen.github.io/ControlSystemIdentification.jl/dev/) provides identification of linear models in the time and frequency domains, as well as `nonlinear_pem`, a prediction-error method for nonlinear models that uses the [`UnscentedKalmanFilter`](@ref) from this package.
- [LowLevelParticleFiltersMTK.jl](https://baggepinnen.github.io/LowLevelParticleFiltersMTK.jl/dev/) constructs state estimators for ModelingToolkit models, and documents estimation of parameters and noise covariances of such models.

## Pages in this section

- [Maximum-likelihood and MAP estimation](@ref): Learn how to use particle filters and Kalman filters to compute likelihoods for parameter estimation.
- [Estimating noise covariances](@ref): Tune the noise covariance matrices of a Kalman filter by maximum-likelihood or MAP estimation, and validate the result.
- [Bayesian inference](@ref): Full Bayesian inference using PMMH and DynamicHMC.
- [Joint state and parameter estimation](@ref): Estimate time-varying parameters by augmenting the state.
- [Joint state and parameter estimation using MUKF](@ref): Use the Marginalized Unscented Kalman Filter for efficient joint estimation.
- [Using an optimizer](@ref "Prediction-Error minimization using an optimizer"): Use gradient-based optimization with automatic differentiation for parameter estimation.
- [Joint estimation of plant parameters and noise covariances](@ref): Estimate plant parameters and noise covariance matrices simultaneously, quantify the uncertainty, and combine multiple experiments.
- [Multi-step prediction-error estimation](@ref): Estimate models intended for prediction over a horizon.
- [Identifiability](@ref): Analyze structural identifiability and Fisher information for parameter estimation problems.

## Videos

Examples of parameter estimation are available here.

By using an optimizer to optimize the likelihood of an [`UnscentedKalmanFilter`](@ref):
```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/0RxQwepVsoM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

Estimation of time-varying parameters:
```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/zJcOPPLqv4A?si=XCvpo3WD-4U3PJ2S" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

Adaptive control by means of estimation of time-varying parameters:
```@raw html
<iframe style="height: 315px; width: 560px" src="https://www.youtube.com/embed/Ip_prmA7QTU?si=Fat_srMTQw5JtW2d" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
```

```@raw html
<script>
(function() {
    var hash = window.location.hash;
    if (!hash) return;

    var redirects = {
        '#Maximum-likelihood-estimation': '../param_est_ml/#Maximum-likelihood-estimation',
        '#Generate-data-by-simulation': '../param_est_ml/#Generate-data-by-simulation',
        '#Compute-likelihood-for-various-values-of-the-parameters': '../param_est_ml/#Compute-likelihood-for-various-values-of-the-parameters',
        '#MAP-estimation': '../param_est_ml/#MAP-estimation',
        '#Bayesian-inference-using-PMMH': '../param_est_bayesian/#Bayesian-inference-using-PMMH',
        '#Bayesian-inference-using-DynamicHMC.jl': '../param_est_bayesian/#Bayesian-inference-using-DynamicHMC.jl',
        '#Joint-state-and-parameter-estimation': '../param_est_joint/#Joint-state-and-parameter-estimation',
        '#Joint-state-and-parameter-estimation-using-MUKF': '../param_est_mukf/#Joint-state-and-parameter-estimation-using-MUKF',
        '#Using-an-optimizer': '../param_est_optimizer/#Using-an-optimizer',
        '#Solving-using-Optim': '../param_est_optimizer/#Solving-using-Optim',
        '#Solving-using-Gauss-Newton-optimization': '../param_est_optimizer/#Solving-using-Gauss-Newton-optimization',
        '#Identifiability': '../param_est_identifiability/#Identifiability',
        '#Polynomial-methods': '../param_est_identifiability/#Polynomial-methods',
        '#Linear-methods': '../param_est_identifiability/#Linear-methods',
        '#Fisher-Information-and-Augmented-State-Covariance': '../param_est_identifiability/#Fisher-Information-and-Augmented-State-Covariance',
    };

    if (redirects[hash]) {
        window.location.replace(redirects[hash]);
    }
})();
</script>
```
