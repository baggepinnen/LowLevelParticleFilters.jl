ENV["GKSwstype"] = 100 # workaround for gr segfault on GH actions
# ENV["GKS_WSTYPE"] = 100 # workaround for gr segfault on GH actions
using Documenter, LowLevelParticleFilters

ENV["JULIA_DEBUG"]=Documenter # Enable this for debugging

using Plots, DisplayAs
gr(format=:png)


makedocs(
      sitename = "LowLevelParticleFilters Documentation",
      doctest = false,
      modules = [LowLevelParticleFilters],
      # pagesonly = true,
      pages = [
            "Home" => "index.md",
            "Discretization" => "discretization.md",
            "Multiple measurement models" => "measurement_models.md",
            "Parameter estimation" => [
                  "Overview" => "parameter_estimation.md",
                  "Maximum-likelihood and MAP" => "param_est_ml.md",
                  "Bayesian inference" => "param_est_bayesian.md",
                  "Joint state and parameter estimation" => "param_est_joint.md",
                  "MUKF for parameter estimation" => "param_est_mukf.md",
                  "Using an optimizer" => "param_est_optimizer.md",
                  "Identifiability" => "param_est_identifiability.md",
            ],
            "Benchmark" => "benchmark.md",
            "Performance tips" => "distributions.md",
            "Tutorials" => [
                  "Kalman-filter tutorial with LowLevelParticleFilters" => "adaptive_kalmanfilter.md",
                  "Noise tuning and disturbance modeling for Kalman filtering" => "noisetuning.md",
                  "Particle-filter tutorial" => "beetle_example.md",
                  "IMM-filter tutorial" => "beetle_example_imm.md",
                  "Rao-Blackwellized filter tutorial" => "rbpf_example.md",
                  "State estimation for DAE systems" => "dae.md",
                  "Adaptive estimation and control" => "adaptive_control.md",
                  "Adaptive Neural-Network training" => "neural_network.md",
                  "SciML: Adaptive Universal Differential Equation" => "friction_nn_example.md",
                  "SciML: Learning a sunshine disturbance model" => "thermal_nn_example.md",
                  "Fault detection" => "fault_detection.md",
                  "Unscented transform" => "ut.md",
                  "Disturbance gallery" => "disturbance_gallery.md",
                  "Influence of sample rate on performance" => "sample_rate.md",
            ],
            "API" => "api.md",
      ],
      format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
      warnonly = [:docs_block, :missing_docs, :cross_references],
) # Due to lots of plots, this will just have to be run on my local machine

deploydocs(
      # deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-cinder"),
      repo = "github.com/baggepinnen/LowLevelParticleFilters.jl.git",
      push_preview = true,
)
