using LowLevelParticleFilters
import LowLevelParticleFilters.resample
using Test, Random, LinearAlgebra, Statistics, StaticArrays, Distributions, Plots
Random.seed!(0)

mvnormal(d::Int, σ::Real) = MvNormal(LinearAlgebra.Diagonal(fill(float(σ) ^ 2, d)))
mvnormal(μ::AbstractVector{<:Real}, σ::Real) = MvNormal(μ, float(σ) ^ 2 * I)


## Standard UKF

eye(n) = Matrix{Float64}(I,n,n)
n = 2 # Dinemsion of state
m = 2 # Dinemsion of input
p = 2 # Dinemsion of measurements

dg = mvnormal(p,1.0)          # Dynamics noise Distribution
df = mvnormal(n,0.1)          # Measurement noise Distribution
d0 = mvnormal(randn(n),2.0)   # Initial state Distribution
du = mvnormal(2,1) # Control input distribution

# Define random linenar state-space system
Tr = randn(n,n)
A = SA[0.99 0.1; 0 0.2]
B = @SMatrix randn(n,m)
C = SMatrix{p,p}(eye(p))
# C = SMatrix{p,n}([1 1])

dynamics(x,u,t) = A*x .+ B*u
measurement(x,u,t) = C*x

T     = 200 # Number of time steps
kf   = KalmanFilter(A, B, C, 0, eye(n), eye(p), d0)
ukf  = UnscentedKalmanFilter(dynamics, measurement, eye(n), eye(p), d0)
x,u,y = LowLevelParticleFilters.simulate(kf,T,du) # Simuate trajectory using the model in the filter
@test_nowarn LowLevelParticleFilters.simulate(ukf,T,du)
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
x,u,y = tosvec.((x,u,y))


reskf = forward_trajectory(kf, u, y) # filtered, prediction, pred
resukf = forward_trajectory(ukf, u, y)

norm(mean(x .- reskf[1]))
norm(mean(x .- resukf[1]))

norm(mean(x .- reskf[2]))
norm(mean(x .- resukf[2]))
@test norm(mean(x .- reskf[2])) < norm(mean(x .- reskf[1])) # Filtered should be better than prediction
@test norm(mean(x .- resukf[2])) < norm(mean(x .- resukf[1]))
@test norm(mean(x .- reskf[2])) ≈ norm(mean(x .- resukf[2])) atol=5e-2
# @test norm(mean(x .- reskf[2])) < norm(mean(x .- resukf[2]))  # KF should be better than UKF
# @test norm(mean(x .- reskf[1])) < norm(mean(x .- resukf[1]))  # KF should be better than UKF
@test norm(mean(x .- reskf[2])) < 0.2


## DAE UKF =====================================================================
function pend(state, f, t=0)
    x,y,u,v,λ = state
    g = 9.82
    SA[
        u
        v
        -λ*x + f[1]
        -λ*y - g + f[2]
        # x^2 + y^2 - 1
        # x*u + y*v
        u^2 + v^2 − λ*(x^2 + y^2) − g*y + x*f[1] + y*f[2]
    ]
end

n = 4 # Dinemsion of differential state
m = 2 # Dinemsion of input
p = 2 # Dinemsion of measurements
const Ts = 0.01

dg = mvnormal(p,1.0)          # Dynamics noise Distribution
df = mvnormal(n,0.1)          # Measurement noise Distribution
d0 = mvnormal([1,0,0,0],0.0001)   # Initial state Distribution
du = mvnormal(2,0.1) # Control input distribution
xz0 = [mean(d0); 0]

get_x(xz) = SA[xz[1],xz[2],xz[3],xz[4]]
get_z(xz) = SA[xz[5]]
get_x_z(xz) = get_x(xz), get_z(xz)
build_xz(x, z) = [x; z]
g((x,y,u,v), (λ,), f, t) = SA[u^2 + v^2 − λ*(x^2 + y^2) − 9.82*y + x*f[1] + y*f[2]]


function dynamics(xz,u,t)
    der = pend(xz,u,t)
    xp = get_x(xz) + Ts*get_x(der) # Euler step
    xzp = build_xz(xp, get_z(xz))
    LowLevelParticleFilters.get_xz(xzp, get_x_z, build_xz, g, u, t) # Adjust z
end
measurement(x,u,t) = x[1:2]

dynamics(xz0,zeros(m),0)


T     = 200 # Number of time steps
ukf0 = UnscentedKalmanFilter(dynamics, measurement, eye(n), eye(p), d0)
ukf  = LowLevelParticleFilters.DAEUnscentedKalmanFilter(ukf0; g, get_x_z, build_xz, xz0, nu=m)
x,u,y = LowLevelParticleFilters.simulate(ukf,T,du)



tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
x,u,y = tosvec.((x,u,y))

xf,xft,R,Rt,ll = forward_trajectory(ukf, u, y)


plot(reduce(hcat, x)', layout=length(x[1]))
plot!(reduce(hcat, xf)')
plot!(reduce(hcat, xft)')