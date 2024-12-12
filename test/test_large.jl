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

R1 = I(nx)
R2 = I(ny)

T    = 200 # Number of time steps
kf   = KalmanFilter(__A, __B, __C, 0, R1, R2)
skf = SqKalmanFilter(__A, __B, __C, 0, R1, R2)
ukf = UnscentedKalmanFilter(dynamics_large, measurement_large, R1, R2; ny, nu)
ekf = ExtendedKalmanFilter(dynamics_large, measurement_large, R1, R2; nu)

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

ukf = UnscentedKalmanFilter(dynamics_large_ip, measurement_large_ip, R1, R2; ny, nu)
ekf = ExtendedKalmanFilter(dynamics_large_ip, measurement_large_ip, R1, R2; nu)

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



## Plotting ====================================================================
using Plots
plot(sol_kf, plothy = true, plote = true)
plot(sol_ukf, plothy = true, plote = true, plotR=true)
plot(sol_ekf, plothy = true, plote = true, plotRt=true)
plot(sol_sqkf, plothy = true, plote = true)

## Test mixing of measurement models ===========================================

mm_ukf = UKFMeasurementModel{Float64, true, false}(measurement_large_ip, R2; nx, ny)
mm_ekf = EKFMeasurementModel{Float64, true}(measurement_large_ip, R2; nx, ny)
mm_kf = LinearMeasurementModel(__C, 0, R2; nx, ny)
mm = CompositeMeasurementModel(mm_ukf, mm_ekf, mm_kf)

mms = [mm_ukf, mm_ekf, mm_kf, mm]

for mm in mms
    @show nameof(typeof(mm))
    
    correct!(kf, mm, u[1], y[1])
    correct!(ekf, mm, u[1], y[1])
    correct!(ukf, mm, u[1], y[1])

    @test kf.x ≈ ekf.x ≈ ukf.x
    @test kf.R ≈ ekf.R ≈ ukf.R
end