using LowLevelParticleFilters
using LowLevelParticleFilters: resample, mean_with_weights, cov_with_weights, weighted_mean, weighted_cov, UKFWeights, SimpleMvNormal
using NonlinearSolve, StaticArrays
using Test, Random, LinearAlgebra, Statistics, StaticArrays, Distributions, Plots, PositiveFactorizations, ControlSystemsBase

Random.seed!(0)


#=
Begin system definition
=#

function differential_dynamics(x::T, z, u, p, t)::T where {T}
    return (-x + z)
end
c = SA[randn()]
residual(x, z, u, p, t) = x + z - c
measurement(x, z, u, p, t) = z
Q = SA[0.1] # TODO think about these vals
R = SA[0.2]
dt = 0.1

A = SA[-1 1]
B = SA[0] #TODO add control inputs
C = SA[0 1]
D = SA[0] #TODO add control inputs
"""
    takes
    - augmented xaug = [x, z]
    - previous step covariance estimate (for diff vars only)
    - control
    - measurement
    - parameters
    - time
    """
function kf_step(xaug, P, u, y, p, t)
    xₐ = A*xaug + B*u
    nx = length(xₐ)
    xaugₐ = [xₐ; c-xₐ]
    Pₐ = A*[P 0; 0 0]*transpose(A) + Q
    innovation = y - (C*xaugₐ + D*u)
    S = C[:,1:nx]*Pₐ*transpose(C[:,1:nx]) + R
    K = Pₐ*transpose(C[:,1:nx])*inv(S)
    xₚ = xₐ + K*innovation
    Pₚ = (I(nx) - K*C[1:nx])*Pₐ
    return (xₚ, Pₚ)
end

#=
Now for the LLPF version
=#
get_x_z(xz) = (SA[xz[1]], SA[xz[2]])
build_xz(x, z) = vcat(x, z)

function get_z_from_x(x, z)
    prob = NonlinearProblem((z, p)->residual(x, z, nothing, nothing, nothing), z)
    return solve(prob, NewtonRaphson()).u
end

function dynamics(xz, u, p, t)
    x, z = get_x_z(xz)
    new_x = x + dt*differential_dynamics(x, z, u, p, t)

    z0 = SA[z...]
    prob = NonlinearProblem((z,p)->residual(new_x,z,u,p,t), z0)
    new_z = solve(prob, NewtonRaphson()).u
    #new_z = get_z_from_x(new_x, z)
    return build_xz(new_x, new_z)
end

x0 = SA[1.0]
z0 = get_z_from_x(x0, SA[rand()])
xz0 = build_xz(x0, z0)
results = zeros(100, 2)
results[1,:] .= xz0
for i in 2:100
    results[i,:] = dynamics(results[i-1,:], 0, 0, 0)
end
plot(results, labels=["x" "z"])
plot!((c./2).*ones(100), label="c/2")
#=
End system definition
=#

@testset "DAE UKF Constructor" begin
    R1 = SA[0.1;;]
    R2 = SA[0.2;;]
    P0 = SA[0.5;;]
    d0 = SimpleMvNormal(x0, P0)
    nu = 1
    ny = 1
    Ts = 0.1

    alg = NewtonRaphson()
    daeukf = DAEUnscentedKalmanFilter(dynamics, measurement,
                                      residual, get_x_z, build_xz,
                                      R1, R2, d0;
                                      xz0, nu, ny, Ts,
                                      constraint_solve_alg = alg)

    @test daeukf isa DAEUnscentedKalmanFilter
    @test daeukf isa LowLevelParticleFilters.AbstractUnscentedKalmanFilter

    # bookkeeping
    @test daeukf.t == 0
    @test daeukf.Ts == Ts
    @test daeukf.nu == nu
    @test daeukf.ny == ny

    # differential state initialized from d0
    @test daeukf.x ≈ x0
    @test daeukf.R ≈ P0
    @test daeukf.R1 === R1
    @test daeukf.d0 === d0

    # DAE-specific buffers
    @test daeukf.xz ≈ xz0
    @test length(daeukf.xz_sigma_points) == 2*length(x0) + 1
    @test eltype(daeukf.xz_sigma_points) === typeof(xz0)

    # sigma-point cache sized for differential state
    @test length(daeukf.predict_sigma_point_cache.x0) == 2*length(x0) + 1

    # callbacks wired through
    @test daeukf.dynamics === dynamics
    @test daeukf.residual === residual
    @test daeukf.get_x_z === get_x_z
    @test daeukf.build_xz === build_xz

    # measurement model knows about the full xz, not just x
    @test daeukf.measurement_model.measurement === measurement
    @test daeukf.measurement_model.R2 === R2

    # defaults
    @test daeukf.regenerate === true
    @test daeukf.constraint_solve_alg === alg
    @test daeukf.constraint_solve_kwargs.reltol == 1e-10
    @test daeukf.weight_params isa LowLevelParticleFilters.TrivialParams
    @test daeukf.p isa LowLevelParticleFilters.NullParameters
    @test daeukf.names isa SignalNames

    # overrides take effect
    alg2 = TrustRegion()
    daeukf2 = DAEUnscentedKalmanFilter(dynamics, measurement,
                                       residual, get_x_z, build_xz,
                                       R1, R2, d0;
                                       xz0, nu, ny, Ts,
                                       regenerate = false,
                                       constraint_solve_alg = alg2,
                                       constraint_solve_kwargs = (; reltol = 1e-8))
    @test daeukf2.regenerate === false
    @test daeukf2.constraint_solve_alg === alg2
    @test daeukf2.constraint_solve_kwargs.reltol == 1e-8

    # required keyword arguments are actually required
    @test_throws UndefKeywordError DAEUnscentedKalmanFilter(
        dynamics, measurement, residual, get_x_z, build_xz,
        R1, R2, d0; nu, ny)                            # missing xz0
    @test_throws UndefKeywordError DAEUnscentedKalmanFilter(
        dynamics, measurement, residual, get_x_z, build_xz,
        R1, R2, d0; xz0, ny)                           # missing nu
    @test_throws UndefKeywordError DAEUnscentedKalmanFilter(
        dynamics, measurement, residual, get_x_z, build_xz,
        R1, R2, d0; xz0, nu)                           # missing ny
end

@testset "Linear Scalar DAE" begin
end
