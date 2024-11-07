module StateEstimator
using LowLevelParticleFilters
using Random, LinearAlgebra, StaticArrays, Distributions


## KF
Base.@ccallable function main()::Cint

    eye(n) = SMatrix{n,n}(Matrix{Float64}(I,n,n))
    nx = 2 # Dinemsion of state
    nu = 2 # Dinemsion of input
    ny = 2 # Dinemsion of measurements

    d0 = MvNormal(@SVector(randn(nx)),2.0)   # Initial state Distribution
    du = MvNormal(2,1) # Control input distribution

    # Define linenar state-space system
    _A = SA[0.99 0.1; 0 0.2]
    _B = @SMatrix [-0.7400216956683083 1.6097265310456392; -1.4384539113366408 1.7467811974822818]
    _C = SMatrix{ny,ny}(eye(ny))
    # C = SMatrix{p,n}([1 1])

    dynamics(x,u,p,t) = _A*x .+ _B*u
    measurement(x,u,p,t) = _C*x

    T    = 200 # Number of time steps
    kf   = KalmanFilter(_A, _B, _C, 0, eye(nx), eye(ny), d0)
    # kf = UnscentedKalmanFilter(dynamics, measurement, eye(nx), eye(ny), d0; ny, nu)
    # kf = SqKalmanFilter(_A, _B, _C, 0, eye(nx), eye(ny), d0)
    # kf = ExtendedKalmanFilter(dynamics, measurement, eye(nx), eye(ny), d0; nu)


    u = [SA[0.1, 0.1] for _ in 1:T]
    y = [SA[0.1, 0.1] for _ in 1:T]



## Test allocations ============================================================
    sol = forward_trajectory(kf, u, y)
    println(Core.stdout, "I got loglik = ", sol.ll)
    return 0
end

end