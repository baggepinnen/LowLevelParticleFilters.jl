ENV["GKSwstype"] = 322 # workaround for gr segfault on GH actions
using Documenter, LowLevelParticleFilters

using Plots
gr()


makedocs(
      sitename = "LowLevelParticleFilters Documentation",
      doctest = false,
      modules = [LowLevelParticleFilters],
      pages = [
            "Home" => "index.md",
            "Parameter estimation" => "parameter_estimation.md",
            "Benchmark" => "benchmark.md",
            "High-performance distributions" => "distributions.md",
            "API" => "api.md",
      ],
      format = Documenter.HTML(prettyurls = haskey(ENV, "CI")),
) # Due to lots of plots, this will just have to be run on my local machine

deploydocs(
      deps = Deps.pip("pygments", "mkdocs", "python-markdown-math", "mkdocs-cinder"),
      repo = "github.com/baggepinnen/LowLevelParticleFilters.jl.git",
)
