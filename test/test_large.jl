using LowLevelParticleFilters
using Test, Random, LinearAlgebra, Statistics, Test
Random.seed!(0)


## KF

nx = 100 # Dimension of state
nu = 2 # Dimension of input
ny = 90 # Dimension of measurements


# Define linenar state-space system
const __A = 0.1*randn(nx, nx)
const __B = randn(nx, nu)
const __C = randn(ny,nx)

dynamics_large(x,u,p,t) = __A*x .+ __B*u
measurement_large(x,u,p,t) = __C*x

T    = 200 # Number of time steps
kf   = KalmanFilter(__A, __B, __C, 0, I(nx), I(ny))
skf = SqKalmanFilter(__A, __B, __C, 0, I(nx), I(ny))
ukf = UnscentedKalmanFilter(dynamics_large, measurement_large, I(nx), I(ny); ny, nu)
ekf = ExtendedKalmanFilter(dynamics_large, measurement_large, I(nx), I(ny); nu)

U = [randn(nu) for _ in 1:T]
x,u,y = LowLevelParticleFilters.simulate(kf, U) # Simuate trajectory using the model in the filter



## Test allocations ============================================================
sol_kf = forward_trajectory(kf, u, y) 
a = @allocations forward_trajectory(kf, u, y) 
@test a <= 6810*2*1.1  # the x2 is for julia v1.11 vs. 1.10, the .1 is for 10% tolerance

a = @allocated forward_trajectory(kf, u, y) 
@test a <= 167477104*1.1 

sol_ukf = forward_trajectory(ukf, u, y) 
a = @allocations forward_trajectory(ukf, u, y) 
@test a <= 294609*2*1.1 # the x2 is for julia v1.11 vs. 1.10, the .1 is for 10% tolerance

a = @allocated forward_trajectory(ukf, u, y)
@test a <=  527_397_952*1.1

sol_ekf = forward_trajectory(ekf, u, y)
a = @allocations forward_trajectory(ekf, u, y)
@test a <=  20811*2*1.1 # the x2 is for julia v1.11 vs. 1.10, the .1 is for 10% tolerance

a = @allocated forward_trajectory(ekf, u, y)
@test a <=  304_488_320*1.1

sol_sqkf = forward_trajectory(skf, u, y)
a = @allocations forward_trajectory(skf, u, y)
@test a <=  14810*2*1.1 # the x2 is for julia v1.11 vs. 1.10, the .1 is for 10% tolerance

a = @allocated forward_trajectory(skf, u, y)
@test a <=  425_413_104*1.1


@test sol_kf.ll ≈ sol_ukf.ll ≈ sol_ekf.ll ≈ sol_sqkf.ll


## In place ====================================================================

function dynamics_large_ip(dx,x,u,p,t)
    # __A*x .+ __B*u
    mul!(dx, __A, x)
    mul!(dx, __B, u, 1.0, 1.0)
    nothing
end
function measurement_large_ip(y,x,u,p,t)
    # __C*x
    mul!(y, __C, x)
    nothing
end

ukf = UnscentedKalmanFilter(dynamics_large_ip, measurement_large_ip, I(nx), I(ny); ny, nu)
ekf = ExtendedKalmanFilter(dynamics_large_ip, measurement_large_ip, I(nx), I(ny); nu)

sol_ukf = forward_trajectory(ukf, u, y)
a = @allocations forward_trajectory(ukf, u, y)
@test a <=  259416*1.1 # measured on julia v1.11, the .1 is for 10% tolerance

a = @allocated forward_trajectory(ukf, u, y)
@test a <=  390_719_072*1.1

sol_ekf = forward_trajectory(ekf, u, y)
a = @allocations forward_trajectory(ekf, u, y)
@test a <=  15616*1.1

a = @allocated forward_trajectory(ekf, u, y)
@test a <=  220_814_208*1.1