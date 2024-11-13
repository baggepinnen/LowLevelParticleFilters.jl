using LowLevelParticleFilters
import LowLevelParticleFilters.resample
using Test, Random, LinearAlgebra, Statistics, StaticArrays, Distributions, Plots
Random.seed!(0)

mvnormal(d::Int, σ::Real) = MvNormal(LinearAlgebra.Diagonal(fill(float(σ) ^ 2, d)))
mvnormal(μ::AbstractVector{<:Real}, σ::Real) = MvNormal(μ, float(σ) ^ 2 * I)

## Test sigmapoints
m = randn(3)
S = randn(3,3)
S = S'S
xs = LowLevelParticleFilters.sigmapoints(m, S)
X = reduce(hcat, xs)
@test vec(mean(X, dims=2)) ≈ m
@test cov(X, dims=2) ≈ S

m = [1,2]
S = [3. 1; 1 4]
xs = LowLevelParticleFilters.sigmapoints(m, S)
X = reduce(hcat, xs)
@test vec(mean(X, dims=2)) ≈ m
@test cov(X, dims=2) ≈ S


## Standard UKF

eye(n) = SMatrix{n,n}(1.0I(n))
nx = 2 # Dinemsion of state
nu = 2 # Dinemsion of input
ny = 2 # Dinemsion of measurements

d0 = mvnormal(@SVector(randn(nx)),2.0)   # Initial state Distribution
du = mvnormal(2,1) # Control input distribution

# Define random linenar state-space system
Tr = randn(nx,nx)
const _A = SA[0.99 0.1; 0 0.2]
const _B = @SMatrix [-0.7400216956683083 1.6097265310456392; -1.4384539113366408 1.7467811974822818]
const _C = SMatrix{ny,ny}(eye(ny))
# C = SMatrix{p,n}([1 1])

dynamics(x,u,p,t) = _A*x .+ _B*u
measurement(x,u,p,t) = _C*x

R2 = eye(ny)

T    = 200 # Number of time steps
kf   = KalmanFilter(_A, _B, _C, 0, eye(nx), R2, d0)
ukf  = UnscentedKalmanFilter(dynamics, measurement, eye(nx), R2, d0; ny, nu)
x,u,y = LowLevelParticleFilters.simulate(kf,T,du) # Simuate trajectory using the model in the filter
@test_nowarn LowLevelParticleFilters.simulate(ukf,T,du)
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
x,u,y = tosvec.((x,u,y))


reskf = forward_trajectory(kf, u, y) # filtered, prediction, pred
resukf = forward_trajectory(ukf, u, y)

sse(x) = sum(sum.(abs2, x))
sse(x .- reskf.x)
sse(x .- resukf.x)

sse(x .- reskf.xt)
sse(x .- resukf.xt)
@test sse(x .- reskf.xt) < sse(x .- reskf.x) # Filtered should be better than prediction
@test sse(x .- resukf.xt) < sse(x .- resukf.x)*1.00001
@test sse(x .- reskf.xt) ≈ sse(x .- resukf.xt) rtol=0.1
@test sse(x .- reskf.xt) < sse(x .- resukf.xt)*1.01  # KF should be better than UKF
@test sse(x .- reskf.x) < sse(x .- resukf.x)*1.01  # KF should be better than UKF
@test sse(x .- reskf.xt) < 250

@test reskf.ll ≈ resukf.ll rtol=1e-2


xT,RT,ll = smooth(reskf, kf, u, y)
xT2,RT2,ll2 = smooth(resukf, ukf, u, y)
@test sse(x .- xT) < sse(x .- reskf.xt)*1.01 # Test ukf smoothing better than ukf filtering
@test sse(x .- xT2) < sse(x .- resukf.xt)*1.01

# plot(reduce(hcat, vec.(reskf.Rt))', lab="Filter", layout=4)
# plot!(reduce(hcat, vec.(RT))', lab="Smoothed")


# plot(reduce(hcat, x)', lab="true", layout=2)
# plot!(reduce(hcat, resukf.xt)', lab="Filter")
# plot!(reduce(hcat, xT)', lab="Smoothed")

## Custom type for u
dynamics(x,u::NamedTuple,p,t) = _A*x .+ _B*[u.a; u.b]
unt = reinterpret(@NamedTuple{a::Float64, b::Float64}, u)
resukfnt = forward_trajectory(ukf, unt, y)
@test resukf.xt ≈ resukfnt.xt
xTnt,RTnt,llnt = smooth(resukfnt, ukf, unt, y)
@test xT ≈ xTnt
@test RT ≈ RTnt
@test ll ≈ llnt

## Non-static arrays ===========================================================

function dynamics_ip(xp,x,u,p,t)
    mul!(xp, _A, x) 
    mul!(xp, _B, u, 1.0, 1.0)
    xp
end
function measurement_ip(y,x,u,p,t)
    mul!(y, _C, x)
    y
end

ukf2  = UnscentedKalmanFilter(dynamics_ip, measurement_ip, eye(nx), R2, d0; ny, nu)
vu,vy = Vector.(u), Vector.(y)
resukf2 = forward_trajectory(ukf2, vu, vy)
# 0.001769 seconds (32.01 k allocations: 2.241 MiB)

@test sse(resukf2.xt .- resukf.xt) < 1e-10



## Augmented dynamics
dynamics_w(x,u,p,t,w) = _A*x .+ _B*u .+ w
ukfw  = UnscentedKalmanFilter{false,false,true,false}(dynamics_w, measurement, eye(nx), R2, d0; ny, nu)
resukfw = forward_trajectory(ukfw, u, y)
@test reduce(hcat, resukfw.xt) ≈ reduce(hcat, resukf.xt) atol=1e-6
@test reduce(hcat, resukfw.x) ≈ reduce(hcat, resukf.x) atol=1e-6
@test reduce(hcat, resukfw.R) ≈ reduce(hcat, resukf.R) atol=1e-6
@test reduce(hcat, resukfw.Rt) ≈ reduce(hcat, resukf.Rt) atol=1e-6
@test resukfw.ll ≈ resukf.ll rtol=1e-6


measurement_v(x,u,p,t,v) = _C*x .+ v
ukfv  = UnscentedKalmanFilter{false,false,false,true}(dynamics, measurement_v, eye(nx), R2, d0; ny, nu)
resukfv = forward_trajectory(ukfv, u, y)
@test reduce(hcat, resukfv.xt) ≈ reduce(hcat, resukf.xt) atol=1e-6
@test reduce(hcat, resukfv.x) ≈ reduce(hcat, resukf.x) atol=1e-6
@test reduce(hcat, resukfv.R) ≈ reduce(hcat, resukf.R) atol=1e-6
@test reduce(hcat, resukfv.Rt) ≈ reduce(hcat, resukf.Rt) atol=1e-6
@test resukfv.ll ≈ resukf.ll rtol=1e-6

## DAE UKF =====================================================================
# "A pendulum in DAE form"
# function pend(state, f, p, t=0)
#     x,y,u,v,λ = state
#     g = 9.82
#     SA[
#         u
#         v
#         -λ*x + f[1]
#         -λ*y - g + f[2]
#         # x^2 + y^2 - 1 # Index 3, position constraint
#         # x*u + y*v # index 2, tangential velocity
#         u^2 + v^2 - λ*(x^2 + y^2) - g*y + x*f[1] + y*f[2] # index 1, centripetal acceleration
#     ]
# end

# nx = 4 # Dinemsion of differential state
# nu = 2 # Dinemsion of input
# ny = 2 # Dinemsion of measurements
# const Ts_ekf = 0.001
# @inbounds measurement(x,u,p,t) = SA[x[1], x[end]] # measure one position and the algebraic state 

# d0 = mvnormal([1,0,0,0],0.1)   # Initial state Distribution
# du = mvnormal(2,0.1) # Control input distribution
# xz0 = [mean(d0); 0]
# u0 = zeros(nu)

# get_x(xz) = SA[xz[1],xz[2],xz[3],xz[4]]
# get_z(xz) = SA[xz[5]]
# get_x_z(xz) = get_x(xz), get_z(xz)
# build_xz(x, z) = [x; z]
# g((x,y,u,v), (λ,), f, p, t) = SA[u^2 + v^2 - λ*(x^2 + y^2) - 9.82*y + x*f[1] + y*f[2]]
# # g((x,y,u,v), (λ,), f, t) = SA[x*u + y*v]
# g(xz, u, p, t) = g(get_x(xz), get_z(xz), u, p, t)

# # Discretization of the continuous-time dynamics, we use a naive Euler approximation, real-world use should use a proper DAE solver, for example using the integrator interface in OrdinaryDiffEq.jl
# function dynamics(xz,u,p,t)
#     Tsi = Ts_ekf/100
#     for i = 1:100
#         der = pend(xz,u,p,t)
#         xp = get_x(xz) + Tsi*get_x(der) # Euler step
#         xzp = build_xz(xp, get_z(xz))
#         xz = LowLevelParticleFilters.calc_xz(get_x_z, build_xz, g, xzp, u, p, t) # Adjust z
#     end
#     xz
# end


# u0 = randn(nu)
# xzp = dynamics(xz0,u0,0,0)
# @test g(xzp, u0, 0, 0)[] ≈ 0 atol=0.01

# ukf0 = UnscentedKalmanFilter(dynamics, measurement, 0.0001eye(nx), 0.01eye(ny), d0; ny, nu)
# threads = false
# for threads = (false, true)
#     local ukf  = LowLevelParticleFilters.DAEUnscentedKalmanFilter(ukf0; g, get_x_z, build_xz, xz0, nu=nu, threads)

#     let u0 = zeros(nu)
#         xzp = LowLevelParticleFilters.calc_xz(ukf, xz0, u0, 0, 0)
#         @test xzp[end] ≈ 0 atol=0.01 # zero centripetal acceleration at the point (x,y) = (1,0)
#         @test g(xzp, u0, 0, 0)[] ≈ 0 atol=0.01

#         xzp = LowLevelParticleFilters.calc_xz(ukf, randn(nx+1), u0, 0, 0)
#         @test g(xzp, u0, 0, 0)[] ≈ 0 atol=0.01
#     end

#     t = 0:Ts_ekf:3
#     U = [sin.(t.^2) sin.(reverse(t).^2)]
#     u = U |> eachrow .|> vcat
#     local x, u, y
#     while true
#         # global x, u, y
#         x,u,y = LowLevelParticleFilters.simulate(ukf, 1 .* u, dynamics_noise=false)
#         norm(x) < 3000 && break
#     end

#     tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
#     x,u,y = tosvec.((x,u,y))

#     state(ukf) .= [2,1,1,1,0]

#     sol = forward_trajectory(ukf, u, y)

#     @test all(zip(sol.R, sol.Rt)) do (R,Rt)
#         det(Rt) < det(R)
#     end # test that the covariance decreases by the measurement update
        

#     @test norm(x[10:end] .- sol.x[10:end]) / norm(x) < 0.2
#     @test norm(x[10:end] .- sol.xt[10:end]) / norm(x) < 0.2

#     if isinteractive()
#         plot(reduce(hcat, x)', layout=length(x[1]))
#         plot!(reduce(hcat, sol.x)')
#         plot!(reduce(hcat, sol.xt)') |> display
#     end
# end
