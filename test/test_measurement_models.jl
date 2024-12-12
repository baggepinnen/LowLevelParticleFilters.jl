using LowLevelParticleFilters
using Test, Random, LinearAlgebra, Statistics, Test
Random.seed!(0)


## KF

nx = 5 # Dimension of state
nu = 2 # Dimension of input
ny = 3 # Dimension of measurements


# Define linenar state-space system
const __A_ = 0.1*randn(nx, nx)
const __B_ = randn(nx, nu)
const __C_ = randn(ny,nx)
const __D_ = randn(ny,nu)

dynamics_l(x,u,p,t) = __A_*x .+ __B_*u
measurement_l(x,u,p,t) = __C_*x .+ __D_*u

R1 = I(nx)
R2 = I(ny)

T    = 200 # Number of time steps
kf   = KalmanFilter(__A_, __B_, __C_, 0, R1, R2)
skf = SqKalmanFilter(__A_, __B_, __C_, 0, R1, R2)
ukf = UnscentedKalmanFilter(dynamics_l, measurement_l, R1, R2; ny, nu)
ekf = ExtendedKalmanFilter(dynamics_l, measurement_l, R1, R2; nu)

U = [randn(nu) for _ in 1:T]
x,u,y = LowLevelParticleFilters.simulate(kf, U) # Simuate trajectory using the model in the filter

## Test mixing of measurement models ===========================================

mm_ukf = UKFMeasurementModel{Float64, false, false}(measurement_l, R2; nx, ny)
mm_ekf = EKFMeasurementModel{Float64, false}(measurement_l, R2; nx, ny)
mm_kf = LinearMeasurementModel(__C_, 0, R2; nx, ny)
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


## Filters with measurement models in them
using Plots
for mm in mms
    @show nameof(typeof(mm))

    ukf = UnscentedKalmanFilter(dynamics_l, mm, R1; ny, nu)
    ekf = ExtendedKalmanFilter(dynamics_l, mm, R1; nu)
    
    correct!(ekf, mm, u[1], y[1])
    correct!(ukf, mm, u[1], y[1])

    @test ekf.x ≈ ukf.x
    @test ekf.R ≈ ukf.R

    sol_ukf = forward_trajectory(ukf, u, y)
    sol_ekf = forward_trajectory(ekf, u, y)
    plot(sol_ukf, plotyh = true, plotyht=true, plote=true) # |> display
    plot(sol_ekf, plotyh = true, plotyht=true, plote=true) # |> display

    @test sol_ukf.x ≈ sol_ekf.x
    @test sol_ukf.xt ≈ sol_ekf.xt
    @test sol_ukf.R ≈ sol_ekf.R
    @test sol_ukf.Rt ≈ sol_ekf.Rt
    @test sol_ukf.e ≈ sol_ekf.e

    if mm isa CompositeMeasurementModel
        @test length(sol_ukf.e[1]) == 3ny
    end
end