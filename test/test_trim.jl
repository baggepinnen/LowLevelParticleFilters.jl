module StateEstimator

using LowLevelParticleFilters
using Random, LinearAlgebra, StaticArrays
println(Core.stdout, 0)
const R1 = SA[1.0 0.0; 0.0 1.0]
const R2 = SA[1.0 0.0; 0.0 1.0]
const nx = 2 # Dimension of state
const nu = 2 # Dimension of input
const ny = 2 # Dimension of measurements

# const d0 = MvNormal(SA[0.0, 0.0],R1)   # Initial state Distribution

# # Define linenar state-space system, to be replaced by MTK dynamics eventually
const _A = SA[0.99 0.1; 0 0.2]
const _B = SA[-0.7400216956683083 1.6097265310456392; -1.4384539113366408 1.7467811974822818]
const _C = R1

dynamics(x,u,p,t) = _A*x .+ _B*u
measurement(x,u,p,t) = _C*x

println(Core.stdout, 1)
# const kf   = KalmanFilter(_A, _B, _C, 0, R1, R2, check=false)
const kf = UnscentedKalmanFilter(dynamics, measurement, R1, R2; ny, nu, p=nothing)
println(Core.stdout, 2)

# using Serialization # This does not go well
# serialize("kf",kf)
# const kf = deserialize("/home/fredrikb/.julia/dev/LowLevelParticleFilters/kf")


Base.@ccallable function main()::Cint
    println(Core.stdout, 3)

    T     = 200 # Number of time steps
    u = [SA[0.1, 0.1] for _ in 1:T]
    y = [SA[0.1, 0.1] for _ in 1:T]

    println(Core.stdout, 4)
    sol = forward_trajectory(kf, u, y)
    println(Core.stdout, "I got loglik = ", sol.ll)
    return zero(Cint)
end

end



