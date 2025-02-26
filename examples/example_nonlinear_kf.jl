#=
This file contains two benchmarks of nonlinear filtering, we compare the performance of the Unscented Kalman Filter, Extended Kalman Filter, and Iterated Extended Kalman Filter.

The first is the system
f(x,u,p,t) = x
h(x,u,p,t) = [atan((x[2]-1.5)/(x[1]-0)), atan((x[2]-0)/(x[1]-0))]

The second is the system
f(x,u,p,t) = 0.5x + 25x / (1 + x^2) + 8cos(1.2*(t-1))
h(x,u,p,t) = x^2 / 20
=#
cd(@__DIR__)
using StaticArrays
using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using LinearAlgebra
using Random
using Plots

function run(;dynamics, measurement, R1, R2, d0, nu, ny, x0, N=N, Tmax=Tmax, steplength = 0.5)

    Random.seed!(0) # Setting the seed

    args = (dynamics, measurement, R1, R2, d0)
    kwargs = (; nu, ny)

    UKFx = zeros(Tmax,N)
    EKFx = zeros(Tmax,N)
    IEKFx = zeros(Tmax,N)

    for i in 1:N
        ukf = UnscentedKalmanFilter(args...; kwargs...)
        ekf = ExtendedKalmanFilter(args...; kwargs...)
        iekf = IteratedExtendedKalmanFilter(args...; step=steplength, kwargs...)

        x = x0

        for t in 1:Tmax
            x = dynamics(x, nothing, nothing, t) + rand(procnoise)
            y = measurement(x, nothing, nothing, t) + rand(measnoise)

            predict!(ukf, nothing, nothing,t)
            correct!(ukf, nothing, y, nothing,t)

            predict!(ekf, nothing, nothing, t)
            correct!(ekf, nothing, y, nothing,t)

            predict!(iekf, nothing, nothing,t)
            correct!(iekf, nothing, y, nothing,t)

            # save the error
            UKFx[t,i] = sum(abs2, ukf.x - x)
            EKFx[t,i] = sum(abs2, ekf.x - x)
            IEKFx[t,i] = sum(abs2, iekf.x - x)
        end
    end
    UKFx, EKFx, IEKFx
    ukfrms = [sqrt(mean(UKFx[t,:])) for t in 1:Tmax]
    ekfrms = [sqrt(mean(EKFx[t,:])) for t in 1:Tmax]
    iekfrms = [sqrt(mean(IEKFx[t,:])) for t in 1:Tmax]
    ukfrms, ekfrms, iekfrms
end

## System 1
dynamics(x,u,p,t) = x
measurement(x,u,p,t) = SA[atan((x[2]-1.5)/(x[1]-0)), atan((x[2]-0)/(x[1]-0))]

N = 10000 # Number of simulations
Tmax = 25 # Number of time steps

R1 = 0.1 * SA[1.0 0.0; 0.0 1.0] # Process noise
R2 = 1e-4 * SA[1.0 0.0; 0.0 1.0]# Measurement noise

procnoise = SimpleMvNormal(SA[0.0, 0.0], R1)
measnoise = SimpleMvNormal(SA[0.0, 0.0], R2)

xhat = SA[1.5, 1.5]
R0 = 0.1 * SA[1.0 0.0; 0.0 1.0]
d0 = SimpleMvNormal(xhat,R0)

ukfrms, ekfrms, iekfrms = run(;dynamics, measurement, R1, R2, d0, nu=0, ny=2, x0=xhat)

plot(1:Tmax, ukfrms, label="UKF", xlabel="Time", ylabel="RMSE", title="RMSE vs Time", lw=2)
plot!(1:Tmax, ekfrms, label="EKF", lw=2)
plot!(1:Tmax, iekfrms, label="IEKF", lw=2)



## System 2

dynamics_ugmd(x,u,p,t) = @. 0.5x + 25x / (1 + x^2) + 8cos(1.2*(t-1))
measurement_ugmd(x,u,p,t) = x.^2 ./20

N = 10000 # Number of simulations
Tmax = 50 # Number of time steps

R1 = 0.1 * @SMatrix ones(1,1) # Process noise
R2 = 0.1 * @SMatrix ones(1,1) # Measurement noise

procnoise = SimpleMvNormal([0.0], R1)
measnoise = SimpleMvNormal([0.0], R2)

xhat = SA[0.0]
R0 = SA[1.0;;]
d0 = SimpleMvNormal(xhat,R0)

ukfrms, ekfrms, iekfrms = run(;dynamics=dynamics_ugmd, measurement=measurement_ugmd, R1, R2, d0, nu=0, ny=1, x0=SA[0.1], Tmax)

plot(1:Tmax, ukfrms, label="UKF", xlabel="Time", ylabel="RMSE", title="RMSE vs Time", lw=2)
plot!(1:Tmax, ekfrms, label="EKF", lw=2)
plot!(1:Tmax, iekfrms, label="IEKF", lw=2)