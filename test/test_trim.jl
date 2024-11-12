module StateEstimator

using LowLevelParticleFilters
using Random, LinearAlgebra, StaticArrays, Distributions
println(Core.stdout, 0)
const R1 = SA[1.0 0.0; 0.0 1.0]
const R2 = SA[1.0 0.0; 0.0 1.0]
const nx = 2 # Dimension of state
const nu = 2 # Dimension of input
const ny = 2 # Dimension of measurements

const d0 = MvNormal(SA[0.0, 0.0],R1)   # Initial state Distribution

# # Define linenar state-space system, to be replaced by MTK dynamics eventually
const _A = SA[0.99 0.1; 0 0.2]
const _B = SA[-0.7400216956683083 1.6097265310456392; -1.4384539113366408 1.7467811974822818]
const _C = R1

dynamics(x,u,p,t) = _A*x .+ _B*u
measurement(x,u,p,t) = _C*x

println(Core.stdout, 1)
const kf   = KalmanFilter(_A, _B, _C, 0, R1, R2, d0, check=false)
# const kf = UnscentedKalmanFilter(dynamics, measurement, R1, R2, d0; ny, nu, p=nothing)
println(Core.stdout, 2)

# using Serialization # This does not go well
# serialize("kf",kf)
# const kf = deserialize("/home/fredrikb/.julia/dev/LowLevelParticleFilters/kf")


Base.@ccallable function main()::Cint
    println(Core.stdout, 3)

    T    = 200 # Number of time steps
    u = [SA[0.1, 0.1] for _ in 1:T]
    y = [SA[0.1, 0.1] for _ in 1:T]

    println(Core.stdout, 4)
    sol = forward_trajectory(kf, u, y)
    println(Core.stdout, "I got loglik = ", sol.ll)
    return zero(Cint)
end

end















# module StateEstimator

# using LowLevelParticleFilters
# using StaticArrays, Distributions

# Base.@ccallable function main()::Cint
    
#     println(Core.stdout, 0)
#     R1 = SA[1.0 0.0; 0.0 1.0]
#     R2 = SA[1.0 0.0; 0.0 1.0]
#     nx = 2 # Dimension of state
#     nu = 2 # Dimension of input
#     ny = 2 # Dimension of measurements
    
#     println(Core.stdout, 1)
#     d0 = MvNormal(SA[0.0, 0.0], R1)   # Initial state Distribution
#     println(Core.stdout, 2)
#     # # Define linenar state-space system
#     A = SA[0.99 0.1; 0 0.2]
#     B = SA[-0.7400216956683083 1.6097265310456392; -1.4384539113366408 1.7467811974822818]
#     C = R1
    
#     dynamics(x,u,p,t) = A*x .+ B*u
#     measurement(x,u,p,t) = C*x
#     println(Core.stdout, 3)
#     kf   = KalmanFilter(A, B, C, 0, R1, R2, d0)
#     println(Core.stdout, 4)
#     # kf = UnscentedKalmanFilter(dynamics, measurement, R1, R2, d0; ny, nu)

#     T    = 200 # Number of time steps
#     u = [SA[0.1, 0.1] for _ in 1:T]
#     y = [SA[0.1, 0.1] for _ in 1:T]

#     println(Core.stdout, 5)
#     sol = forward_trajectory(kf, u, y)
#     println(Core.stdout, "I got loglik = ", sol.ll)
#     return 0
# end

# end



# module StateEstimator

# println(Core.stdout, 1)
# using LowLevelParticleFilters
# println(Core.stdout, 2)
# using StaticArrays
# println(Core.stdout, 3)
# using Distributions
# println(Core.stdout, 4)

# Base.@ccallable function main()::Cint
#     println(Core.stdout, 5)
#     R1 = [1.0 0.0; 0.0 1.0]
#     R2 = [1.0 0.0; 0.0 1.0]
#     nx = 2 # Dimension of state
#     nu = 2 # Dimension of input
#     ny = 2 # Dimension of measurements
    
#     d0 = MvNormal([0.0, 0.0], R1)   # Initial state Distribution
    
#     # # Define linenar state-space system
#     A = [0.99 0.1; 0 0.2]
#     B = [-0.7400216956683083 1.6097265310456392; -1.4384539113366408 1.7467811974822818]
#     C = R1
    
#     dynamics(x,u,p,t) = A*x .+ B*u
#     measurement(x,u,p,t) = C*x
#     println(Core.stdout, 6)
#     kf   = KalmanFilter(A, B, C, 0, R1, R2, d0)
#     println(Core.stdout, 7)
#     # kf = UnscentedKalmanFilter(dynamics, measurement, R1, R2, d0; ny, nu)

#     T    = 200 # Number of time steps
#     u = [[0.1, 0.1] for _ in 1:T]
#     y = [[0.1, 0.1] for _ in 1:T]

#     sol = forward_trajectory(kf, u, y)
#     println(Core.stdout, "I got loglik = ", sol.ll)
#     return 0
# end

# end