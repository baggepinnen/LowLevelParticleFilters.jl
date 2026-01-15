using LowLevelParticleFilters
using Distributions
using StaticArrays
using Plots, LinearAlgebra, SeeToDee, Test
using Optim

## Nonlinear quadtank
function quadtank(h,u,p,t)
    kc = 0.5
    k1, k2, g = 1.6, 1.6, 9.81
    A1 = A3 = A2 = A4 = 4.9
    a1, a3, a2, a4 = 0.03, 0.03, 0.03, 0.03
    γ1, γ2 = 0.2, 0.2

    if t > 500
        a1 *= 2
    end

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0
    
    xd = SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    ]
end

nu = 2 # number of control inputs
nx = 4 # number of state variables
ny = 2 # number of measured outputs
Ts = 1 # sample time
measurement(x,u,p,t) = SA[x[1], x[2]]

discrete_dynamics = SeeToDee.Rk4(quadtank, Ts, supersample=2)

Tperiod = 200
t = 0:Ts:1000
u = vcat.(0.25 .* sign.(sin.(2pi/Tperiod .* t)) .+ 0.25)
u = vcat.(u,u)
x0 = Float64[2,2,3,3]
x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, u)[1:end-1]
y = measurement.(x, u, 0, 0)
y = [y .+ 0.01randn(ny) for y in y]

plot(
    plot(reduce(hcat, x)', title="State"),
    plot(reduce(hcat, u)', title="Inputs")
)



function quadtank_params(h,u,p,t)
    kc = 0.5
    k1, k2, g = 1.6, 1.6, 9.81
    A1 = A3 = A2 = A4 = 4.9
    a3, a2, a4 = 0.03, 0.03, 0.03
    γ1, γ2 = 0.2, 0.2

    a1 = h[5]

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0
    
    SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
        0
    ]
end

discrete_dynamics_params = SeeToDee.Rk4(quadtank_params, Ts, supersample=2)

nx = 5
R1 = Diagonal([0.1, 0.1, 0.1, 0.1, 0.0001])
R2 = Diagonal((1e-2)^2 * ones(ny))
x0 = [2, 2, 3, 3, 0.02]

kf = UnscentedKalmanFilter(discrete_dynamics_params, measurement, R1, R2, MvNormal(x0, R1); ny, nu)

sol = forward_trajectory(kf, u, y)
plot(sol)

@test sol.x[499][5] ≈ 0.03 atol=0.01
@test sol.x[end][5] ≈ 0.06 atol=0.01


##

function quadtank(h, u, p, t)
    kc = p[1]
    k1, k2, g = p[2], p[3], 9.81
    A1 = A3 = A2 = A4 = p[4]
    a1 = a3 = a2 = a4 = p[5]
    γ1 = γ2 = p[6]

    ssqrt(x) = √(max(x, zero(x)) + 1e-3) # For numerical robustness at x = 0
    
    xd = SA[
        -a1/A1 * ssqrt(2g*h[1]) + a3/A1*ssqrt(2g*h[3]) +     γ1*k1/A1 * u[1]
        -a2/A2 * ssqrt(2g*h[2]) + a4/A2*ssqrt(2g*h[4]) +     γ2*k2/A2 * u[2]
        -a3/A3*ssqrt(2g*h[3])                          + (1-γ2)*k2/A3 * u[2]
        -a4/A4*ssqrt(2g*h[4])                          + (1-γ1)*k1/A4 * u[1]
    ]
end

discrete_dynamics = SeeToDee.Rk4(quadtank, Ts, supersample=2)
p_true = [0.5, 1.6, 1.6, 4.9, 0.03, 0.2]

Tperiod = 200
t = 0:Ts:1000
u1 = vcat.(0.25 .* sign.(sin.(2pi/Tperiod .* (t ./ 40).^2)) .+ 0.25)
u2 = vcat.(0.25 .* sign.(sin.(2pi/Tperiod .* (t ./ 40).^2 .+ pi/2)) .+ 0.25)
u = vcat.(u1,u2)
x0 = Float64[2,2,3,3]
x = LowLevelParticleFilters.rollout(discrete_dynamics, x0, u, p_true)[1:end-1]
y = measurement.(x, u, 0, 0)
y = [y .+ 0.01randn(ny) for y in y]

plot(
    plot(reduce(hcat, x)', title="States"),
    plot(reduce(hcat, u)', title="Inputs")
)


nx = 4
R1 = Diagonal([0.1, 0.1, 0.1, 0.1])
R2 = Diagonal((1e-2)^2 * ones(ny))
x0 = [2, 2, 3, 3]

function cost(p::Vector{T}) where T
    kf = UnscentedKalmanFilter(discrete_dynamics, measurement, R1, R2, MvNormal(T.(x0), R1); ny, nu)
    LowLevelParticleFilters.sse(kf, u, y, p)
end

p_guess = p_true .+  0.1*p_true .* randn(length(p_true))

using Optim, Optim.LineSearches
using ADTypes: AutoForwardDiff
res = Optim.optimize(
    cost,
    p_guess,
    BFGS(),
    Optim.Options(
        store_trace       = true,
        show_trace        = true,
        show_every        = 8,
        iterations        = 100,
        allow_f_increases = false,
        time_limit        = 100,
        x_tol             = 0,
        f_abstol          = 0,
        g_tol             = 1e-8,
        f_calls_limit     = 0,
        g_calls_limit     = 0,
    ),
    autodiff = AutoForwardDiff(),
)

