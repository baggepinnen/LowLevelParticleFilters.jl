# Parameter Estimation

State estimation is an integral part of many parameter-estimation methods. Below, we will illustrate several different methods of performing parameter estimation. We can roughly divide the methods into two camps:

1. Methods that optimize prediction error or likelihood by tweaking model parameters.
2. Methods that add the parameters to be estimated as state variables in the model and estimate them using standard state estimation.

From the first camp, we provide some basic functionality for maximum-likelihood estimation and MAP estimation in [Maximum-likelihood and MAP estimation](@ref). An example of (2), joint state and parameter estimation, is provided in [Joint state and parameter estimation](@ref).

## Which method should I use?

The methods demonstrated in this section have slightly different applicability, here, we try to outline which methods to consider for different problems:

| Method                | Parameter Estimation | Covariance Estimation | Time Varying Parameters | Online Estimation |
|-----------------------|----------------------|-----------------------|-------------------------|-------------------|
| Maximum likelihood    | 🟢                   | 🟢                    | 🟥                      | 🟥                |
| Joint state/par estim | 🔶                   | 🟥                    | 🟢                      | 🟢                |
| Prediction-error opt. | 🟢                   | 🟥                    | 🟥                      | 🟥                |


When trying to optimize parameters of the noise distributions, most commonly the covariance matrices, maximum-likelihood (or MAP) is the only recommened method. Similarly, when parameters are time varying or you want an online estimate, the method that jointly estimates state and parameter is the only applicable method. When fitting standard parameters, all methods are applicable. In this case the joint state and parameter estimation tends to be inefficient and unneccesarily complex, and it is recommended to opt for maximum likelihood or prediction-error minimization. The prediction-error minimization (PEM) with a Gauss-Newtown optimizer is often the most efficient method for this type of problem.

Maximum likelihood estimation tends to yield an estimator with better estimates of posterior covariance since this is explicitly optimized for, while PEM tends to produce the smallest possible prediction errors.

## Pages in this section

- [Maximum-likelihood and MAP estimation](@ref): Learn how to use particle filters and Kalman filters to compute likelihoods for parameter estimation.
- [Bayesian inference](@ref): Full Bayesian inference using PMMH and DynamicHMC.
- [Joint state and parameter estimation](@ref): Estimate time-varying parameters by augmenting the state space.
- [Joint state and parameter estimation using MUKF](@ref): Use the Marginalized Unscented Kalman Filter for efficient joint estimation.
- [Using an optimizer](@ref): Use gradient-based optimization with automatic differentiation for parameter estimation.
- [Identifiability](@ref): Analyze structural identifiability and Fisher information for parameter estimation problems.

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
        '#Videos': '../param_est_identifiability/#Videos'
    };

    if (redirects[hash]) {
        window.location.replace(redirects[hash]);
    }
})();
</script>
```
