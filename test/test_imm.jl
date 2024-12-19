using LowLevelParticleFilters
import LowLevelParticleFilters: SimpleMvNormal
using Test, Random, LinearAlgebra, Statistics, StaticArrays, Distributions, Plots
Random.seed!(0)


eye(n) = SMatrix{n,n}(1.0I(n))
nx = 2 # Dinemsion of state
nu = 2 # Dinemsion of input
ny = 2 # Dinemsion of measurements

d0 = SimpleMvNormal(@SVector(randn(nx)), I(nx))   # Initial state Distribution
du = SimpleMvNormal(zeros(nu), I(nu)) # Control input distribution

# Define random linenar state-space system
Tr = randn(nx,nx)
const _A = SA[0.99 0.1; 0 0.2]
const _B = @SMatrix [-0.7400216956683083 1.6097265310456392; -1.4384539113366408 1.7467811974822818]
const _C = SMatrix{ny,ny}(eye(ny))
# C = SMatrix{p,n}([1 1])

R2 = eye(ny)

T    = 40 # Number of time steps

## Filters are identical =======================================================
kf1   = KalmanFilter(_A, _B, _C, 0, eye(nx), R2, d0)
kf2   = KalmanFilter(_A, _B, _C, 0, eye(nx), R2, d0)
x,u,y = LowLevelParticleFilters.simulate(kf1,T,du) # Simuate trajectory using the model in the filter

μ = [0.5,0.5]
P = [0.5 0.5; 0.5 0.5]
imm = IMM([kf1,kf2], P, μ)

reset!(imm)
for i = 1:T
    update!(imm, u[i], y[i])
    @test kf1.x ≈ kf2.x
    @test kf1.R ≈ kf2.R
    @test imm.μ ≈ [0.5,0.5]
end
@test imm.μ ≈ (P^100)[1,:]


## One filter is garbage =======================================================
kf1   = KalmanFilter(_A, _B, _C, 0, eye(nx), R2, d0)
kf2   = KalmanFilter(10000*_A, _B, _C, 0, eye(nx), 100R2, d0)

μ = [0.5,0.5]
P = [0.5 0.5; 0.5 0.5]
imm = IMM([kf1,kf2], P, μ)

reset!(imm)
for i = 1:T
    update!(imm, u[i], y[i])
    # @show imm.x, kf1.x, kf2.x
    # @show imm.μ
    @test imm.μ[1] > 0.95
    @test sum(imm.μ) ≈ 1.0
end


## One Mode is always left immedeately =========================================
kf1   = KalmanFilter(_A, _B, _C, 0, eye(nx), R2, d0)
kf2   = KalmanFilter(_A, _B, _C, 0, eye(nx), R2, d0)

μ = [0.5,0.5]
P = [0.5 0.5; 1 0]
imm = IMM([kf1,kf2], P, μ)

reset!(imm)
for i = 1:T
    update!(imm, u[i], y[i])
    # @show imm.x, kf1.x, kf2.x
    # @show imm.μ
    # @test imm.μ[1] > 0.95
    @test sum(imm.μ) ≈ 1.0
end
s1(x) = x ./ sum(x)
@test imm.μ ≈ (P^100)[1,:] # Two ways of testing the same thing
@test imm.μ ≈ s1(eigvecs(P')[:,2])

## One Mode is sticky ==========================================================
kf1   = KalmanFilter(_A, _B, _C, 0, eye(nx), R2, d0)
kf2   = KalmanFilter(_A, _B, _C, 0, eye(nx), R2, d0)

μ = [0.5,0.5]
P = [0.5 0.5; 0 1]
imm = IMM([kf1,kf2], P, μ)

reset!(imm)
for i = 1:T
    update!(imm, u[i], y[i])
    # @show imm.x, kf1.x, kf2.x
    # @show imm.μ
    @test kf1.x ≈ kf2.x
    @test kf1.R ≈ kf2.R
    @test imm.μ[1] ≈ 0.5^(i+1) # The probability of the first mode should decrease exponentially
    @test sum(imm.μ) ≈ 1.0
end

## One Mode is sticky but the sticky-mode filter is bad ========================
kf1   = KalmanFilter(_A, _B, _C, 0, eye(nx), R2, d0)
kf2   = KalmanFilter(10000*_A, _B, _C, 0, eye(nx), 100R2, d0)

μ = [0.5,0.5]
P = [0.5 0.5; 0 1]
imm = IMM([kf1,kf2], P, μ)

reset!(imm)
for i = 1:T
    update!(imm, u[i], y[i])
    # @show imm.x, kf1.x, kf2.x
    # @show imm.μ
    @test imm.μ[1] > (i == 1 ? 0.8 : 0.99)
    @test sum(imm.μ) ≈ 1.0
end


## Various functions
x,u,y = simulate(imm, T, du) # Simulate the IMM

sol = forward_trajectory(imm, u, y) # Forward trajectory

plot(sol)