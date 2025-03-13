using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using Random, StaticArrays, Test

f_n(xn, args...) = xn   # Identity function for demonstration
A_n(xn, args...) = SA[0.5;;]  # Example matrix (1x1)
A = SA[0.95;;]  # Example matrix (1x1)
C2 = SA[1.0;;]    # Example matrix (1x1)
h(xn, args...) = xn       # Identity measurement function
nu = 0
ny = 1
nx = 2
B = @SMatrix zeros(1,0)
D = 0

# Noise covariances (1x1 matrices for 1D case)
R1n = SA[0.01;;]  # Nonlinear state noise
R1l = SA[0.01;;]  # Linear state noise
R2 = SA[0.1;;]    # Measurement noise

# Initial states (1D)
x0n = SA[1.0]
x0l = SA[1.0]
R0 = SA[1.0;;]
d0n = SimpleMvNormal(x0n, R1n)
d0l = SimpleMvNormal(x0l, R0)

# Generate dummy measurements (replace with real data)
function fsim(x)
    X = [x]
    for i = 1:10000
        xn = f_n(x.xn) + A_n(x.xn) * x.xl + rand(SimpleMvNormal(R1n))
        xl = A * x.xl + rand(SimpleMvNormal(R1l))
        x = RBParticle(xn, xl, R1l)
        push!(X, x)
    end
    return X
end


ps = fsim(RBParticle(x0n, x0l, R0))
y = [SVector{ny}(rand(SimpleMvNormal(h(xn) + C2 * xl, R2))) for (; xn, xl) in ps]

u = [SA_F64[] for y in y]

kf = KalmanFilter(A, B, C2, D, R1l, R2, d0l)

mm = RBMeasurementModel{false, typeof(h), typeof(R2)}(h, R2, 1)
pf = RBPF{false, false}(500, kf, f_n, mm, R1n, d0n; nu, An=A_n, Ts=1.0, names=SignalNames(x=["x1", "x2"], u=[], y=["y1"], name="RBPF"))

sol = forward_trajectory(pf, u, y)
a = @allocated forward_trajectory(pf, u, y)
@test a < 200092592*1.1

a = @allocations forward_trajectory(pf, u, y)
@test a < 2010*1.1


using Plots
plot(sol, size=(1000,800), xreal=ps)


# 0.257749 seconds (5.86 M allocations: 204.700 MiB, 6.52% gc time)
# 0.006624 seconds (305.01 k allocations: 11.242 MiB) static arrays in dynamics and covs
# 0.009420 seconds (103.01 k allocations: 5.078 MiB) new method for rand on SimpleMvNormal with static length
# 0.005039 seconds (2.01 k allocations: 1.996 MiB)B also static