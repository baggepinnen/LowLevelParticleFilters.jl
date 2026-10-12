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

g1 = ForwardDiff.gradient(costfun1, [1.0])

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

g2 = ForwardDiff.gradient(costfun2, [1.0])

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

g3 = ForwardDiff.gradient(costfun3, [1.0])


## 
# using DifferentiationInterface

# using Enzyme # NOTE: these all segfault
# backend = AutoEnzyme()

# using Zygote # Cannot handle ukf due to mutation, cannot handle logpdf due to who knows what
# backend = AutoZygote()

# using ReverseDiff # Works for 1,2 and 3 for UKF only
# backend = AutoReverseDiff()

# g1_e = DifferentiationInterface.gradient(costfun1, backend, [1.0]) 
# g2_e = DifferentiationInterface.gradient(costfun2, backend, [1.0])
# g3_e = DifferentiationInterface.gradient(costfun3, backend, [1.0])

# @test g1 ≈ g1_e
# @test g2 ≈ g2_e
# @test g3 ≈ g3_e
## Test differentiability of forward_trajectory, which stores the prediction errors along the trajectory

@testset "forward_trajectory with dual numbers" begin
    y_missing = Union{Missing, eltype(y)}[k % 5 == 0 ? missing : y[k] for k in eachindex(y)]
    for yd in (y, [Vector(yi) for yi in y], y_missing)
        function costfun4(p::AbstractArray{T}) where T
            d0 = MvNormal(@SVector(zeros(T, nx)), T(1.0))
            ukf = UnscentedKalmanFilter(dynamics3, measurement3, eye(nx), eye(ny), d0; ny, nu, p)
            ekf = ExtendedKalmanFilter(dynamics3, measurement3, eye(nx), eye(ny), d0; nu, check=false, p)
            -forward_trajectory(ukf, u, yd).ll - forward_trajectory(ekf, u, yd).ll
        end
        g4 = ForwardDiff.gradient(costfun4, [1.0])
        h = 1e-6
        @test g4[1] ≈ (costfun4([1.0 + h]) - costfun4([1.0 - h])) / 2h rtol=1e-5
    end

    # The prediction errors are stored with the number type of the state, and the type is unchanged for Float64 states
    D = ForwardDiff.Dual{Nothing, Float64, 1}
    @test LowLevelParticleFilters.innovation_type(SVector{2, Float64}, Float64) == SVector{2, Float64}
    @test LowLevelParticleFilters.innovation_type(SVector{2, Float64}, D) == SVector{2, D}
    @test LowLevelParticleFilters.innovation_type(Vector{Float64}, D) == Vector{D}
    @test LowLevelParticleFilters.innovation_type(Union{Missing, SVector{2, Float64}}, D) == Union{Missing, SVector{2, D}}
    @test LowLevelParticleFilters.innovation_type(Vector{Int}, Float64) == Vector{Float64}
    @test LowLevelParticleFilters.innovation_type(Missing, D) == Missing
    # Measurements constructed as SVector{1}.(y), where y contains missing entries, have an abstract element type that is kept
    YA = eltype(SVector{1}.(Union{Missing, Float64}[1.0, missing]))
    @test LowLevelParticleFilters.innovation_type(YA, Float64) == YA
    @test LowLevelParticleFilters.innovation_type(YA, D) == YA
    YU = Union{SVector{1, Float64}, SVector{1, Missing}}
    @test LowLevelParticleFilters.innovation_type(YU, Float64) == YU
    @test LowLevelParticleFilters.innovation_type(YU, D) == Union{SVector{1, D}, SVector{1, Missing}}
    @test LowLevelParticleFilters.innovation_type(Vector{Union{Missing, Float64}}, Float64) == Vector{Union{Missing, Float64}}
    @test LowLevelParticleFilters.innovation_type(Vector{Union{Missing, Float64}}, D) == Vector{Union{Missing, D}}
end
