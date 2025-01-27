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
            "Parameter estimation" => "parameter_estimation.md",
            "Benchmark" => "benchmark.md",
            "Performance tips" => "distributions.md",
            "Tutorials" => [
                  "Kalman-filter tutorial with LowLevelParticleFilters" => "adaptive_kalmanfilter.md",
                  "Noise tuning and disturbance modeling for Kalman filtering" => "noisetuning.md",
                  "Particle-filter tutorial" => "beetle_example.md",
                  "IMM-filter tutorial" => "beetle_example_imm.md",
                  "State estimation for DAE systems" => "dae.md",
                  "Adaptive estimation and control" => "adaptive_control.md",
                  "Adaptive Neural-Network training" => "neural_network.md",
                  "Fault detection" => "fault_detection.md",
            ],
            "API" => "api.md",
      ],
      format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
      warnonly = [:docs_block, :missing_docs, :cross_references],
) # Due to lots of plots, this will just have to be run on my local machine

deploydocs(
      # deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-cinder"),
      repo = "github.com/baggepinnen/LowLevelParticleFilters.jl.git",
)
