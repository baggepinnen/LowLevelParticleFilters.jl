module LowLevelParticleFiltersSciMLBaseExt

using LowLevelParticleFilters
using SciMLBase: NonlinearProblem, solve

function LowLevelParticleFilters.scimlbase_solver(alg; kwargs...)
    (f, z0) -> solve(NonlinearProblem{false}((z, _) -> f(z), z0), alg; kwargs...).u
end

end # module
