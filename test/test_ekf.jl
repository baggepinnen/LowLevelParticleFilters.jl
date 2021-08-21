using LowLevelParticleFilters, ForwardDiff, Distributions
const LLPF = LowLevelParticleFilters
eye(n) = Matrix{Float64}(I,n,n)
n = 2 # Dinemsion of state
m = 2 # Dinemsion of input
p = 2 # Dinemsion of measurements

# Define random linenar state-space system
A = SMatrix{n,n}([0.99 0.1; 0 0.2])
B = @SMatrix randn(n,m)
C = SMatrix{p,p}(eye(p))
# C = SMatrix{p,n}([1 1])

dynamics(x,u,t) = A*x .+ B*u
measurement(x,u,t) = C*x

T = 800 # Number of time steps

d0 = MvNormal(randn(n),2.0)   # Initial state Distribution
du = MvNormal(2,1) # Control input distribution
kf = KalmanFilter(A, B, C, 0, 0.001eye(n), eye(p), d0)
ekf = LLPF.ExtendedKalmanFilter(kf, dynamics, measurement)
x,u,y = LLPF.simulate(kf,T,du)


xf,xt,R,Rt,ll = forward_trajectory(kf, u, y)
xf2,xt2,R2,Rt2,ll2 = forward_trajectory(ekf, u, y)


# When the dynamics is linear, these should be the same
@test reduce(hcat, xf) ≈ reduce(hcat, xf2)
@test reduce(hcat, xt) ≈ reduce(hcat, xt2)
@test reduce(hcat, R)  ≈ reduce(hcat, R2)
@test reduce(hcat, Rt) ≈ reduce(hcat, Rt2)
@test ll ≈ ll2


## add nonlinear dynamics
dynamics2(x,u,t) = A*x - 0.01*abs.(x) .+ B*u
ekf = LLPF.ExtendedKalmanFilter(kf, dynamics2, measurement)
x,u,y = LLPF.simulate(ekf,T,du)

xf,xt,R,Rt,ll = forward_trajectory(kf, u, y)
xf2,xt2,R2,Rt2,ll2 = forward_trajectory(ekf, u, y)

@test norm(reduce(hcat, x .- xf)) > norm(reduce(hcat, x .- xf2))
@test norm(reduce(hcat, x .- xt)) > norm(reduce(hcat, x .- xt2))
@test ll < ll2

# plot(reduce(hcat, x)', layout=2, lab="True")
# plot!(reduce(hcat, xf)', lab="Kf")
# plot!(reduce(hcat, xf2)', lab="EKF")