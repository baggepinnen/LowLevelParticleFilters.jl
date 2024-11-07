using LowLevelParticleFilters
using Test, Random, LinearAlgebra, Statistics, StaticArrays, Distributions
using ForwardDiff
Random.seed!(0)
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy


## KF

eye(n) = SMatrix{n,n}(Matrix{Float64}(I,n,n))
nx = 2 # Dinemsion of state
nu = 2 # Dinemsion of input
ny = 2 # Dinemsion of measurements

d0 = MvNormal(@SVector(randn(nx)),2.0)   # Initial state Distribution
du = MvNormal(2,1) # Control input distribution

# Define linenar state-space system
const _A = SA[0.99 0.1; 0 0.2]
const _B = @SMatrix [-0.7400216956683083 1.6097265310456392; -1.4384539113366408 1.7467811974822818]
const _C = SMatrix{ny,ny}(eye(ny))
# C = SMatrix{p,n}([1 1])

dynamics(x,u,p,t) = _A*x .+ _B*u
measurement(x,u,p,t) = _C*x

T    = 200 # Number of time steps
kf   = KalmanFilter(_A, _B, _C, 0, eye(nx), eye(ny), d0)
x,u,y = LowLevelParticleFilters.simulate(kf,T,du) # Simuate trajectory using the model in the filter
x,u,y = tosvec.((x,u,y))


# Test differentiability w.r.t. R1
function costfun1(p::AbstractArray{T}) where T
    R1 = p[1]*eye(nx)

    d0 = MvNormal(@SVector(zeros(T, nx)),T(1.0))   # Initial state Distribution

    kf  = KalmanFilter(_A, _B, _C, 0, R1, eye(ny), d0)
    ukf = UnscentedKalmanFilter(dynamics, measurement, R1, eye(ny), d0; ny, nu)
    skf = SqKalmanFilter(_A, _B, _C, 0, R1, eye(ny), d0)
    ekf = ExtendedKalmanFilter(dynamics, measurement, R1, eye(ny), d0; nu, check=false)

    filters = [kf, ukf, skf, ekf]
    out = zero(T)
    for filter in filters
        predict!(filter, u[1])
        (; ll, e) = correct!(filter, u[1], y[1])
        out -= ll
        out += sum(e)
    end
    out
end

@test_nowarn g1 = ForwardDiff.gradient(costfun1, [1.0])

# Test differentiability w.r.t. R2
function costfun2(p::AbstractArray{T}) where T
    R1 = eye(nx)
    R2 = p[1]*eye(ny)

    d0 = MvNormal(@SVector(zeros(T, nx)),T(1.0))   # Initial state Distribution

    kf  = KalmanFilter(_A, _B, _C, 0, R1, R2, d0)
    ukf = UnscentedKalmanFilter(dynamics, measurement, R1, R2, d0; ny, nu)
    skf = SqKalmanFilter(_A, _B, _C, 0, R1, R2, d0)
    ekf = ExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; nu, check=false)

    filters = [kf, ukf, skf, ekf]
    out = zero(T)
    for filter in filters
        predict!(filter, u[1])
        (; ll, e) = correct!(filter, u[1], y[1])
        out -= ll
        out += sum(e)
    end
    out
end

@test_nowarn g2 = ForwardDiff.gradient(costfun2, [1.0])

## Test differentiability w.r.t. p in dynamics

dynamics3(x,u,p,t) = _A*x .+ _B*u .+ p
measurement3(x,u,p,t) = _C*x .+ p

function costfun3(p::AbstractArray{T}) where T
    R1 = eye(nx)
    R2 = eye(ny)

    d0 = MvNormal(@SVector(zeros(T, nx)),T(1.0))   # Initial state Distribution

    ukf = UnscentedKalmanFilter(dynamics3, measurement3, R1, R2, d0; ny, nu, p)
    ekf = ExtendedKalmanFilter(dynamics3, measurement3, R1, R2, d0; nu, check=false, p)

    filters = [ukf, ekf]
    out = zero(T)
    for filter in filters
        predict!(filter, u[1])
        (; ll, e) = correct!(filter, u[1], y[1])
        out -= ll
        out += sum(e)
    end
    out
end

@test_nowarn g3 = ForwardDiff.gradient(costfun3, [1.0])


## 
# using DifferentiationInterface, Enzyme
# backend = AutoEnzyme()
# NOTE: these all segfault
# g1_e = DifferentiationInterface.gradient(costfun1, backend, [1.0]) 
# g2_e = DifferentiationInterface.gradient(costfun2, backend, [1.0])
# g3_e = DifferentiationInterface.gradient(costfun3, backend, [1.0])

# @test g1 ≈ g1_e
# @test g2 ≈ g2_e
# @test g3 ≈ g3_e