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

eye(n) = Matrix{Float64}(I,n,n)
nx = 2 # Dinemsion of state
nu = 2 # Dinemsion of input
ny = 2 # Dinemsion of measurements

d0 = mvnormal(randn(nx),2.0)   # Initial state Distribution
du = mvnormal(2,1) # Control input distribution

# Define random linenar state-space system
Tr = randn(nx,nx)
A = SA[0.99 0.1; 0 0.2]
B = @SMatrix randn(nx,nu)
C = SMatrix{ny,ny}(eye(ny))
# C = SMatrix{p,n}([1 1])

dynamics(x,u,p,t) = A*x .+ B*u
measurement(x,u,p,t) = C*x

T    = 200 # Number of time steps
kf   = KalmanFilter(A, B, C, 0, eye(nx), eye(ny), d0)
ukf  = UnscentedKalmanFilter(dynamics, measurement, eye(nx), eye(ny), d0)
x,u,y = LowLevelParticleFilters.simulate(kf,T,du) # Simuate trajectory using the model in the filter
@test_nowarn LowLevelParticleFilters.simulate(ukf,T,du)
tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
x,u,y = tosvec.((x,u,y))


reskf = forward_trajectory(kf, u, y) # filtered, prediction, pred
resukf = forward_trajectory(ukf, u, y)

norm(mean(x .- reskf.x))
norm(mean(x .- resukf.x))

norm(mean(x .- reskf.xt))
norm(mean(x .- resukf.xt))
@test norm(mean(x .- reskf.xt)) < norm(mean(x .- reskf.x)) # Filtered should be better than prediction
@test norm(mean(x .- resukf.xt)) < norm(mean(x .- resukf.x))
@test norm(mean(x .- reskf.xt)) ≈ norm(mean(x .- resukf.xt)) atol=5e-2
# @test norm(mean(x .- reskf.xt)) < norm(mean(x .- resukf.xt))  # KF should be better than UKF
# @test norm(mean(x .- reskf.x)) < norm(mean(x .- resukf.x))  # KF should be better than UKF
@test norm(mean(x .- reskf.xt)) < 0.2


## DAE UKF =====================================================================
"A pendulum in DAE form"
function pend(state, f, p, t=0)
    x,y,u,v,λ = state
    g = 9.82
    SA[
        u
        v
        -λ*x + f[1]
        -λ*y - g + f[2]
        # x^2 + y^2 - 1 # Index 3, position constraint
        # x*u + y*v # index 2, tangential velocity
        u^2 + v^2 - λ*(x^2 + y^2) - g*y + x*f[1] + y*f[2] # index 1, centripetal acceleration
    ]
end

nx = 4 # Dinemsion of differential state
nu = 2 # Dinemsion of input
ny = 2 # Dinemsion of measurements
const Ts = 0.001
@inbounds measurement(x,u,p,t) = SA[x[1], x[end]] # measure one position and the algebraic state 

d0 = mvnormal([1,0,0,0],0.1)   # Initial state Distribution
du = mvnormal(2,0.1) # Control input distribution
xz0 = [mean(d0); 0]
u0 = zeros(nu)

get_x(xz) = SA[xz[1],xz[2],xz[3],xz[4]]
get_z(xz) = SA[xz[5]]
get_x_z(xz) = get_x(xz), get_z(xz)
build_xz(x, z) = [x; z]
g((x,y,u,v), (λ,), f, p, t) = SA[u^2 + v^2 - λ*(x^2 + y^2) - 9.82*y + x*f[1] + y*f[2]]
# g((x,y,u,v), (λ,), f, t) = SA[x*u + y*v]
g(xz, u, p, t) = g(get_x(xz), get_z(xz), u, p, t)

# Discretization of the continuous-time dynamics, we use a naive Euler approximation, real-world use should use a proper DAE solver, for example using the integrator interface in OrdinaryDiffEq.jl
function dynamics(xz,u,p,t)
    Tsi = Ts/100
    for i = 1:100
        der = pend(xz,u,p,t)
        xp = get_x(xz) + Tsi*get_x(der) # Euler step
        xzp = build_xz(xp, get_z(xz))
        xz = LowLevelParticleFilters.calc_xz(get_x_z, build_xz, g, xzp, u, p, t) # Adjust z
    end
    xz
end


u0 = randn(nu)
xzp = dynamics(xz0,u0,0,0)
@test g(xzp, u0, 0, 0)[] ≈ 0 atol=0.01

ukf0 = UnscentedKalmanFilter(dynamics, measurement, 0.0001eye(nx), 0.01eye(ny), d0)
threads = false
for threads = (false, true)
    ukf  = LowLevelParticleFilters.DAEUnscentedKalmanFilter(ukf0; g, get_x_z, build_xz, xz0, nu=nu, threads)

    let u0 = zeros(nu)
        xzp = LowLevelParticleFilters.calc_xz(ukf, xz0, u0, 0, 0)
        @test xzp[end] ≈ 0 atol=0.01 # zero centripetal acceleration at the point (x,y) = (1,0)
        @test g(xzp, u0, 0, 0)[] ≈ 0 atol=0.01

        xzp = LowLevelParticleFilters.calc_xz(ukf, randn(nx+1), u0, 0, 0)
        @test g(xzp, u0, 0, 0)[] ≈ 0 atol=0.01
    end

    t = 0:Ts:3
    U = [sin.(t.^2) sin.(reverse(t).^2)]
    u = U |> eachrow .|> vcat
    local x, u, y
    while true
        # global x, u, y
        x,u,y = LowLevelParticleFilters.simulate(ukf, 1 .* u, dynamics_noise=false)
        norm(x) < 3000 && break
    end

    tosvec(y) = reinterpret(SVector{length(y[1]),Float64}, reduce(hcat,y))[:] |> copy
    x,u,y = tosvec.((x,u,y))

    state(ukf) .= [2,1,1,1,0]

    sol = forward_trajectory(ukf, u, y)

    @test all(zip(sol.R, sol.Rt)) do (R,Rt)
        det(Rt) < det(R)
    end # test that the covariance decreases by the measurement update
        

    @test norm(x[10:end] .- sol.x[10:end]) / norm(x) < 0.2
    @test norm(x[10:end] .- sol.xt[10:end]) / norm(x) < 0.2

    if isinteractive()
        plot(reduce(hcat, x)', layout=length(x[1]))
        plot!(reduce(hcat, sol.x)')
        plot!(reduce(hcat, sol.xt)') |> display
    end
end
