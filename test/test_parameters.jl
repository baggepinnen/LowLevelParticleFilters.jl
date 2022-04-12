using LowLevelParticleFilters, StaticArrays
eye(n) = Matrix{Float64}(I,n,n)
nx = 2 # Dinemsion of state
nu = 2 # Dinemsion of input
ny = 2 # Dinemsion of measurements


# Define random linenar state-space system
Tr = randn(nx,nx)
A = SMatrix{nx,nx}([0.99 0.1; 0 0.2])
B = @SMatrix randn(nx,nu)
C = SMatrix{ny,ny}(eye(ny))

dynamics(x,u,p,t) = A*x .+ B*u .+ p
measurement(x,u,p,t) = C*x

kf = UnscentedKalmanFilter(dynamics, measurement, 0.01eye(nx), eye(ny); p=1, nu, ny)

T = 20
u = vcat.(eachcol(randn(nu, T)))
x,u,y = simulate(kf,u)
x2,u2,y2 = simulate(kf,u, -1)
# plot(reduce(hcat, x)')
# plot!(reduce(hcat, x2)')

ksol = forward_trajectory(kf, u, y)
ksol2 = forward_trajectory(kf, u2, y2, -1)

# @test ksol.p == 1
# @test ksol2.p == -1
@test ksol.ll > -100
@test ksol2.ll > -100 # This fails if p is not adjusted

# plot(ksol)
# plot!(ksol2)
