using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal, UKFWeights,
                               mean_with_weights, cov_with_weights,
                               weighted_mean, weighted_cov,
                               calc_xz
using SimpleNonlinearSolve, StaticArrays
using Test, Random, LinearAlgebra, Statistics, Distributions

Random.seed!(0)


# ============================================================================
# Shared helpers
# ============================================================================

# Default decompose / recompose for scalar x, scalar z.
get_x_z_scalar(xz)        = (SA[xz[1]], SA[xz[2]])
build_xz_scalar(x, z)     = vcat(x, z)

# Time- and ensemble-averaged NEES helper. NEES samples are i.i.d. χ²_{nx},
# so the 95% consistency band on N samples follows the χ²_{N·nx} distribution.
function nees_band(N::Int, nx::Int; α=0.05)
    d = Chisq(N * nx)
    lo = quantile(d, α/2)   / N
    hi = quantile(d, 1-α/2) / N
    return (lo, hi)
end

function lag1_autocorr(e)
    em = mean(e)
    num = sum((e[k]-em)*(e[k-1]-em) for k in 2:length(e))
    den = sum((e[k]-em)^2 for k in eachindex(e))
    return num/den
end


# ============================================================================
# Constructor test (re-uses Test 1's system definitions below)
# ============================================================================

# Test 1 system (also reused by the constructor test):
#   ẋ = -x + z + w,  0 = x + z - c,  y = z + v
C1   = 0.7
DT1  = 0.1

t1_residual(x, z, u, p, t)    = x .+ z .- SA[C1]
t1_measurement(xz, u, p, t)   = SA[xz[2]]
t1_diff_dynamics(x, z, u, p, t) = (-x .+ z)

function t1_dynamics(xz, u, p, t)
    x, z = get_x_z_scalar(xz)
    new_x = x .+ DT1 .* t1_diff_dynamics(x, z, u, p, t)
    prob  = NonlinearProblem((zv, _) -> t1_residual(new_x, zv, u, p, t), z)
    new_z = solve(prob, SimpleNewtonRaphson(), reltol=1e-12).u
    build_xz_scalar(new_x, new_z)
end

@testset "DAE UKF Constructor" begin
    Q  = SA[0.1;;]
    R  = SA[0.2;;]
    P0 = SA[0.5;;]
    x0 = SA[0.3]
    z0 = SA[C1] - x0
    xz0 = build_xz_scalar(x0, z0)
    d0  = SimpleMvNormal(x0, P0)
    nu, ny, Ts = 1, 1, DT1

    solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-10)
    daeukf = DAEUnscentedKalmanFilter(t1_dynamics, t1_measurement,
                                      t1_residual, get_x_z_scalar, build_xz_scalar,
                                      Q, R, d0;
                                      xz0, nu, ny, Ts,
                                      constraint_solver = solver)

    @test daeukf isa DAEUnscentedKalmanFilter
    @test daeukf isa LowLevelParticleFilters.AbstractUnscentedKalmanFilter

    # bookkeeping
    @test daeukf.t  == 0
    @test daeukf.Ts == Ts
    @test daeukf.nu == nu
    @test daeukf.ny == ny

    # differential state initialized from d0
    @test daeukf.x  ≈ x0
    @test daeukf.R  ≈ P0
    @test daeukf.R1 === Q
    @test daeukf.d0 === d0

    # DAE-specific buffers
    @test daeukf.xz ≈ xz0
    @test length(daeukf.xz_sigma_points) == 2*length(x0) + 1
    @test eltype(daeukf.xz_sigma_points) === typeof(xz0)

    # sigma-point cache sized for differential state
    @test length(daeukf.predict_sigma_point_cache.x0) == 2*length(x0) + 1

    # callbacks wired through
    @test daeukf.dynamics   === t1_dynamics
    @test daeukf.residual   === t1_residual
    @test daeukf.get_x_z    === get_x_z_scalar
    @test daeukf.build_xz   === build_xz_scalar

    # measurement model knows about the full xz, not just x
    @test daeukf.measurement_model.measurement === t1_measurement
    @test daeukf.measurement_model.R2 === R

    # defaults
    @test daeukf.regenerate === true
    @test daeukf.constraint_solver === solver
    @test daeukf.weight_params isa LowLevelParticleFilters.TrivialParams
    @test daeukf.p isa LowLevelParticleFilters.NullParameters
    @test daeukf.names isa SignalNames

    # overrides take effect
    solver2 = LowLevelParticleFilters.scimlbase_solver(SimpleTrustRegion(); reltol = 1e-8)
    daeukf2 = DAEUnscentedKalmanFilter(t1_dynamics, t1_measurement,
                                       t1_residual, get_x_z_scalar, build_xz_scalar,
                                       Q, R, d0;
                                       xz0, nu, ny, Ts,
                                       regenerate = false,
                                       constraint_solver = solver2)
    @test daeukf2.regenerate === false
    @test daeukf2.constraint_solver === solver2

    # required keyword arguments are actually required
    @test_throws UndefKeywordError DAEUnscentedKalmanFilter(
        t1_dynamics, t1_measurement, t1_residual, get_x_z_scalar, build_xz_scalar,
        Q, R, d0; nu, ny)                            # missing xz0
    @test_throws UndefKeywordError DAEUnscentedKalmanFilter(
        t1_dynamics, t1_measurement, t1_residual, get_x_z_scalar, build_xz_scalar,
        Q, R, d0; xz0, ny)                           # missing nu
    @test_throws UndefKeywordError DAEUnscentedKalmanFilter(
        t1_dynamics, t1_measurement, t1_residual, get_x_z_scalar, build_xz_scalar,
        Q, R, d0; xz0, nu)                           # missing ny
    # missing constraint_solver triggers our helpful error (via default expression)
    @test_throws ErrorException DAEUnscentedKalmanFilter(
        t1_dynamics, t1_measurement, t1_residual, get_x_z_scalar, build_xz_scalar,
        Q, R, d0; xz0, nu, ny)                       # missing constraint_solver
end


# ============================================================================
# Test 1: Linear scalar DAE, algebraic measurement
# ----------------------------------------------------------------------------
# System:
#   ẋ = -x + z + w,    w ~ N(0, Q)
#   0 = x + z - c
#   y = z + v,         v ~ N(0, R)
# Substituting z = c - x gives the reduced ODE ẋ = -2x + c.
# Forward-Euler with step DT1:  x_{k+1} = (1 - 2·DT1) x_k + DT1·c + w
# The measurement is purely algebraic: y = c - x.
# The symmetric UT reproduces the first two moments of a linear-Gaussian
# system exactly, so the UKF posterior must agree with a 1-D KF on the
# reduced system to floating-point precision.
# ============================================================================

@testset "Test 1: Linear scalar DAE (vs analytical KF)" begin
    Q  = 0.05
    R  = 0.02
    P0 = 0.5
    x0_val = 0.3
    x0 = SA[x0_val]
    z0 = SA[C1] - x0
    xz0 = build_xz_scalar(x0, z0)
    d0 = SimpleMvNormal(x0, SA[P0;;])

    # discretized reduced system: x_{k+1} = α x_k + β c + w
    α = 1 - 2*DT1
    β = DT1
    # measurement on reduced state: y_k = c - x_k + v   →  H = -1, b = c
    H = -1.0
    b = C1

    daeukf = DAEUnscentedKalmanFilter(t1_dynamics, t1_measurement,
                                      t1_residual, get_x_z_scalar, build_xz_scalar,
                                      SA[Q;;], SA[R;;], d0;
                                      xz0, nu=1, ny=1, Ts=DT1,
                                      constraint_solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12))

    T = 1000
    # Simulate a truth trajectory under the discretized reduced system.
    x_true = zeros(T+1); x_true[1] = x0_val + 0.1
    y_seq  = zeros(T)
    for k in 1:T
        w = sqrt(Q)*randn()
        v = sqrt(R)*randn()
        x_true[k+1] = α*x_true[k] + β*C1 + w
        y_seq[k]    = (C1 - x_true[k+1]) + v
    end

    # Drive both filters with the same measurement sequence.
    x_kf, P_kf = x0_val, P0
    u = SA[0.0]
    diffs_x = zeros(T)
    diffs_P = zeros(T)
    cons    = zeros(T)
    for k in 1:T
        # predict (both)
        predict!(daeukf, u)
        x_kf = α*x_kf + β*C1
        P_kf = α^2*P_kf + Q
        # correct (both)
        correct!(daeukf, u, SA[y_seq[k]])
        S    = H^2*P_kf + R
        K    = P_kf*H/S
        innov = y_seq[k] - (H*x_kf + b)
        x_kf  = x_kf + K*innov
        P_kf  = (1 - K*H)*P_kf

        diffs_x[k] = abs(daeukf.x[1] - x_kf)
        diffs_P[k] = abs(daeukf.R[1,1] - P_kf)
        xh, zh = get_x_z_scalar(daeukf.xz)
        cons[k] = abs((xh[1] + zh[1]) - C1)
    end

    @test maximum(diffs_x) < 1e-8
    @test maximum(diffs_P) < 1e-8
    @test maximum(cons)    < 1e-10
end

@testset "Test 1: zero-noise variant tracks DAE solver" begin
    Q  = 0.0
    R  = 1e-6     # tiny but nonzero so the gain is well defined
    P0 = 1e-6
    x0_val = 0.3
    x0 = SA[x0_val]
    z0 = SA[C1] - x0
    xz0 = build_xz_scalar(x0, z0)
    d0  = SimpleMvNormal(x0, SA[P0;;])

    daeukf = DAEUnscentedKalmanFilter(t1_dynamics, t1_measurement,
                                      t1_residual, get_x_z_scalar, build_xz_scalar,
                                      SA[Q;;], SA[R;;], d0;
                                      xz0, nu=1, ny=1, Ts=DT1,
                                      constraint_solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12))

    T = 200
    α, β = 1 - 2*DT1, DT1
    x_true = x0_val
    u = SA[0.0]
    max_err = 0.0
    max_cons = 0.0
    for k in 1:T
        predict!(daeukf, u)
        x_true = α*x_true + β*C1
        z_true = C1 - x_true
        correct!(daeukf, u, SA[z_true])  # noise-free measurement
        max_err = max(max_err, abs(daeukf.x[1] - x_true))
        xh, zh = get_x_z_scalar(daeukf.xz)
        max_cons = max(max_cons, abs((xh[1] + zh[1]) - C1))
    end
    @test max_err < 1e-8
    @test max_cons < 1e-10
end

@testset "Test 1: NEES ~ χ²₁ over Monte Carlo" begin
    Q  = 0.05
    R  = 0.02
    P0 = 0.5
    x0_val = 0.3
    α, β = 1 - 2*DT1, DT1

    N_MC = 100
    T    = 50
    nees_samples = Float64[]
    for r in 1:N_MC
        x0 = SA[x0_val + sqrt(P0)*randn()]      # sample initial truth from prior
        z0 = SA[C1] - x0
        xz0 = build_xz_scalar(x0, z0)
        d0  = SimpleMvNormal(SA[x0_val], SA[P0;;])

        daeukf = DAEUnscentedKalmanFilter(t1_dynamics, t1_measurement,
                                          t1_residual, get_x_z_scalar, build_xz_scalar,
                                          SA[Q;;], SA[R;;], d0;
                                          xz0=build_xz_scalar(SA[x0_val], SA[C1]-SA[x0_val]),
                                          nu=1, ny=1, Ts=DT1,
                                          constraint_solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-10))
        x_true = x0[1]
        u = SA[0.0]
        for k in 1:T
            predict!(daeukf, u)
            x_true = α*x_true + β*C1 + sqrt(Q)*randn()
            y_k    = (C1 - x_true) + sqrt(R)*randn()
            correct!(daeukf, u, SA[y_k])
            err = daeukf.x[1] - x_true
            push!(nees_samples, err^2 / daeukf.R[1,1])
        end
    end
    lo, hi = nees_band(length(nees_samples), 1)
    @test lo < mean(nees_samples) < hi
end


# ============================================================================
# Test 2: Nonlinear scalar DAE, closed-form trajectory
# ----------------------------------------------------------------------------
# System:
#   ẋ = -z + w
#   0 = z - e^x
#   y = z + v
# Substituting:  ẋ = -e^x,  closed form  x(t) = -ln(t + e^{-x0}),
#                                        z(t) = 1/(t + e^{-x0}).
# Exercises the regeneration step: E[e^x] depends sharply on Var(x), so
# adding Q to P^{xx} without redrawing sigma points biases ŷ and shrinks P^{yy}.
# ============================================================================

DT2 = 0.05

t2_residual(x, z, u, p, t)    = z .- exp.(x)
t2_measurement(xz, u, p, t)   = SA[xz[2]]
t2_diff_dynamics(x, z, u, p, t) = -z

function t2_dynamics(xz, u, p, t)
    x, z = get_x_z_scalar(xz)
    new_x = x .+ DT2 .* t2_diff_dynamics(x, z, u, p, t)
    prob  = NonlinearProblem((zv, _) -> t2_residual(new_x, zv, u, p, t), z)
    new_z = solve(prob, SimpleNewtonRaphson(), reltol=1e-12).u
    build_xz_scalar(new_x, new_z)
end

t2_x_exact(t, x0) = -log(t + exp(-x0))
t2_z_exact(t, x0) = 1 / (t + exp(-x0))

@testset "Test 2: nonlinear scalar DAE, noise-free tracking" begin
    Q  = 0.0
    R  = 1e-8
    P0 = 1e-8
    x0_val = -0.5
    x0 = SA[x0_val]
    z0 = SA[exp(x0_val)]
    xz0 = build_xz_scalar(x0, z0)
    d0  = SimpleMvNormal(x0, SA[P0;;])

    daeukf = DAEUnscentedKalmanFilter(t2_dynamics, t2_measurement,
                                      t2_residual, get_x_z_scalar, build_xz_scalar,
                                      SA[Q;;], SA[R;;], d0;
                                      xz0, nu=1, ny=1, Ts=DT2,
                                      constraint_solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12))

    T = 1000
    u = SA[0.0]
    max_err  = 0.0
    max_cons = 0.0
    for k in 1:T
        predict!(daeukf, u)
        t_k    = k*DT2
        z_meas = t2_z_exact(t_k, x0_val)
        correct!(daeukf, u, SA[z_meas])
        max_err  = max(max_err, abs(daeukf.x[1] - t2_x_exact(t_k, x0_val)))
        xh, zh   = get_x_z_scalar(daeukf.xz)
        max_cons = max(max_cons, abs(zh[1] - exp(xh[1])))
    end
    # Forward Euler at DT2=0.05 has truncation error O(DT2²) ≈ 2.5e-3 per step;
    # over the bounded trajectory the integrated error is small but not at
    # DAE-solver tolerance. Use a generous bound; the strict tolerance check
    # belongs to the algebraic constraint.
    @test max_err  < 5e-2
    @test max_cons < 1e-10
end

@testset "Test 2: process-noise innovation whiteness" begin
    Q  = 0.01
    R  = 0.01
    P0 = 0.01
    x0_val = -0.5
    x0 = SA[x0_val]
    z0 = SA[exp(x0_val)]
    xz0 = build_xz_scalar(x0, z0)
    d0  = SimpleMvNormal(x0, SA[P0;;])

    daeukf = DAEUnscentedKalmanFilter(t2_dynamics, t2_measurement,
                                      t2_residual, get_x_z_scalar, build_xz_scalar,
                                      SA[Q;;], SA[R;;], d0;
                                      xz0, nu=1, ny=1, Ts=DT2,
                                      constraint_solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12))

    T = 500
    u = SA[0.0]
    x_true = x0_val + sqrt(P0)*randn()
    # Use the true nonlinear forward-Euler dynamics for the truth.
    innovations = Float64[]
    for k in 1:T
        # Truth step
        z_true = exp(x_true)
        x_true = x_true + DT2*(-z_true) + sqrt(Q)*randn()
        z_true = exp(x_true)
        y_k    = z_true + sqrt(R)*randn()
        # Filter
        predict!(daeukf, u)
        # Predicted measurement before update — needed for innovation
        xh, zh = get_x_z_scalar(daeukf.xz)
        push!(innovations, y_k - zh[1])
        correct!(daeukf, u, SA[y_k])
    end
    # Discard burn-in.
    e = innovations[50:end]
    r1 = lag1_autocorr(e)
    # 95% two-sided bound for white-noise lag-1 autocorrelation.
    @test abs(r1) < 2/sqrt(length(e))
end

@testset "Test 2: regeneration negative control" begin
    # Verify the regenerate code path is loaded by asserting that the two
    # filters (regenerate on vs off) diverge meaningfully along the trajectory.
    # SSE-based checks are unreliable here because the measurement (z = e^x)
    # is highly informative — both variants end up tracking the truth closely
    # regardless. The textbook symptom is that the posterior covariance and
    # state estimates differ between the two; if regenerate were a no-op, the
    # filters would be identical bit-for-bit.
    Q  = 0.05
    R  = 0.02
    P0 = 0.05
    x0_val = -0.5
    x0 = SA[x0_val]
    z0 = SA[exp(x0_val)]
    xz0 = build_xz_scalar(x0, z0)
    d0  = SimpleMvNormal(x0, SA[P0;;])

    make_ukf(regen) = DAEUnscentedKalmanFilter(
        t2_dynamics, t2_measurement, t2_residual,
        get_x_z_scalar, build_xz_scalar,
        SA[Q;;], SA[R;;], d0;
        xz0, nu=1, ny=1, Ts=DT2,
        regenerate = regen,
        constraint_solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12),
    )

    ukf_on  = make_ukf(true)
    ukf_off = make_ukf(false)

    T = 200
    u = SA[0.0]
    x_true = x0_val
    state_gap = 0.0
    cov_gap   = 0.0
    for k in 1:T
        z_true = exp(x_true)
        x_true = x_true + DT2*(-z_true) + sqrt(Q)*randn()
        z_true = exp(x_true)
        y_k    = z_true + sqrt(R)*randn()
        for f in (ukf_on, ukf_off)
            predict!(f, u)
            correct!(f, u, SA[y_k])
        end
        state_gap = max(state_gap, abs(ukf_on.x[1]  - ukf_off.x[1]))
        cov_gap   = max(cov_gap,   abs(ukf_on.R[1,1] - ukf_off.R[1,1]))
    end
    # Both diagnostics: state estimates and posterior covariances must
    # diverge by more than round-off. A bit-identical pair would prove the
    # regenerate branch is dead code.
    @test state_gap > 1e-4
    @test cov_gap   > 1e-6
end


# ============================================================================
# Test 3: 2D linear reactive cascade with mass conservation
# ----------------------------------------------------------------------------
# A → B ⇌ C (third species fast-equilibrium, modeled algebraically)
#   ȧ = -k₁ a + w_a
#   ḃ = k₁ a - k₂ b + k₋₂ c + w_b
#   0 = a + b + c - M
# Substituting c = M - a - b gives a 2×2 linear ODE with closed form.
# Measurement y = (a, c) + v — one differential, one algebraic. The
# algebraic information about c only flows back into (a,b) corrections via
# P^{xy} assembled across the augmented state.
# ============================================================================

K1   = 0.5
K2   = 0.8
KM2  = 0.2
MASS = 2.0
DT3  = 0.05

get_x_z_3(xz) = (SA[xz[1], xz[2]], SA[xz[3]])
build_xz_3(x, z) = vcat(x, z)

t3_residual(x, z, u, p, t)  = SA[x[1] + x[2] + z[1] - MASS]
t3_measurement(xz, u, p, t) = SA[xz[1], xz[3]]

function t3_diff_dynamics(x, z, u, p, t)
    a, b = x[1], x[2]
    c    = z[1]
    SA[-K1*a, K1*a - K2*b + KM2*c]
end

function t3_dynamics(xz, u, p, t)
    x, z = get_x_z_3(xz)
    new_x = x .+ DT3 .* t3_diff_dynamics(x, z, u, p, t)
    prob  = NonlinearProblem((zv, _) -> t3_residual(new_x, zv, u, p, t), z)
    new_z = solve(prob, SimpleNewtonRaphson(), reltol=1e-12).u
    build_xz_3(new_x, new_z)
end

# Closed-form integration of the reduced 2-state linear ODE; numerical
# reference produced via the same Euler scheme used by the filter so that
# discretization error cancels in the comparison.
function t3_truth_step(a, b)
    c = MASS - a - b
    da = -K1*a
    db =  K1*a - K2*b + KM2*c
    return (a + DT3*da, b + DT3*db)
end

@testset "Test 3: 2D linear cascade, noise-free tracking" begin
    Q   = SA[1e-10 0.0; 0.0 1e-10]
    R   = SA[1e-10 0.0; 0.0 1e-10]
    P0  = SA[1e-10 0.0; 0.0 1e-10]
    a0, b0 = 1.2, 0.5
    c0     = MASS - a0 - b0
    x0  = SA[a0, b0]
    z0  = SA[c0]
    xz0 = build_xz_3(x0, z0)
    d0  = SimpleMvNormal(x0, P0)

    daeukf = DAEUnscentedKalmanFilter(t3_dynamics, t3_measurement,
                                      t3_residual, get_x_z_3, build_xz_3,
                                      Q, R, d0;
                                      xz0, nu=1, ny=2, Ts=DT3,
                                      constraint_solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12))

    T = 500
    u = SA[0.0]
    a_t, b_t = a0, b0
    max_err  = 0.0
    max_cons = 0.0
    for k in 1:T
        predict!(daeukf, u)
        a_t, b_t = t3_truth_step(a_t, b_t)
        c_t       = MASS - a_t - b_t
        correct!(daeukf, u, SA[a_t, c_t])
        err = norm(daeukf.x .- SA[a_t, b_t])
        max_err = max(max_err, err)
        xh, zh = get_x_z_3(daeukf.xz)
        max_cons = max(max_cons, abs(xh[1] + xh[2] + zh[1] - MASS))
    end
    @test max_err  < 1e-6
    @test max_cons < 1e-10
end

@testset "Test 3: mass conservation under noise" begin
    Q   = SA[0.01 0.0; 0.0 0.01]
    R   = SA[0.01 0.0; 0.0 0.01]
    P0  = SA[0.1  0.0; 0.0 0.1]
    a0, b0 = 1.2, 0.5
    c0     = MASS - a0 - b0
    x0  = SA[a0, b0]
    z0  = SA[c0]
    xz0 = build_xz_3(x0, z0)
    d0  = SimpleMvNormal(x0, P0)

    daeukf = DAEUnscentedKalmanFilter(t3_dynamics, t3_measurement,
                                      t3_residual, get_x_z_3, build_xz_3,
                                      Q, R, d0;
                                      xz0, nu=1, ny=2, Ts=DT3,
                                      constraint_solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12))

    T = 300
    u = SA[0.0]
    a_t, b_t = a0, b0
    max_cons = 0.0
    for k in 1:T
        predict!(daeukf, u)
        a_next = a_t - DT3*K1*a_t                                 + sqrt(Q[1,1])*randn()
        b_next = b_t + DT3*(K1*a_t - K2*b_t + KM2*(MASS-a_t-b_t)) + sqrt(Q[2,2])*randn()
        a_t, b_t = a_next, b_next
        c_t = MASS - a_t - b_t
        y_k = SA[a_t + sqrt(R[1,1])*randn(), c_t + sqrt(R[2,2])*randn()]
        correct!(daeukf, u, y_k)
        xh, zh = get_x_z_3(daeukf.xz)
        max_cons = max(max_cons, abs(xh[1] + xh[2] + zh[1] - MASS))
    end
    @test max_cons < 1e-10
end

@testset "Test 3: NEES ~ χ²₂ over Monte Carlo" begin
    Q   = SA[0.01 0.0; 0.0 0.01]
    R   = SA[0.01 0.0; 0.0 0.01]
    P0  = SA[0.1  0.0; 0.0 0.1]
    a0_mean, b0_mean = 1.2, 0.5

    N_MC = 100
    T    = 50
    nees_samples = Float64[]
    for r in 1:N_MC
        a0 = a0_mean + sqrt(P0[1,1])*randn()
        b0 = b0_mean + sqrt(P0[2,2])*randn()
        c0 = MASS - a0 - b0
        x0 = SA[a0, b0]
        z0 = SA[c0]
        d0 = SimpleMvNormal(SA[a0_mean, b0_mean], P0)
        daeukf = DAEUnscentedKalmanFilter(t3_dynamics, t3_measurement,
                                          t3_residual, get_x_z_3, build_xz_3,
                                          Q, R, d0;
                                          xz0 = build_xz_3(SA[a0_mean, b0_mean],
                                                           SA[MASS - a0_mean - b0_mean]),
                                          nu=1, ny=2, Ts=DT3,
                                          constraint_solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12))
        a_t, b_t = a0, b0
        u = SA[0.0]
        for k in 1:T
            predict!(daeukf, u)
            a_next = a_t - DT3*K1*a_t                                          + sqrt(Q[1,1])*randn()
            b_next = b_t + DT3*(K1*a_t - K2*b_t + KM2*(MASS-a_t-b_t))          + sqrt(Q[2,2])*randn()
            a_t, b_t = a_next, b_next
            c_t = MASS - a_t - b_t
            y_k = SA[a_t + sqrt(R[1,1])*randn(), c_t + sqrt(R[2,2])*randn()]
            correct!(daeukf, u, y_k)
            err = SVector{2}(daeukf.x) - SA[a_t, b_t]
            push!(nees_samples, dot(err, daeukf.R \ err))
        end
    end
    lo, hi = nees_band(length(nees_samples), 2)
    @test lo < mean(nees_samples) < hi
end

@testset "simulate + forward_trajectory roundtrip (Test 1 system)" begin
    Q  = 0.05
    R  = 0.02
    P0 = 0.5
    x0_val = 0.3
    x0 = SA[x0_val]
    z0 = SA[C1] - x0
    xz0 = build_xz_scalar(x0, z0)
    d0  = SimpleMvNormal(x0, SA[P0;;])
    daeukf = DAEUnscentedKalmanFilter(t1_dynamics, t1_measurement,
                                      t1_residual, get_x_z_scalar, build_xz_scalar,
                                      SA[Q;;], SA[R;;], d0;
                                      xz0, nu=1, ny=1, Ts=DT1,
                                      constraint_solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12))
    T  = 200
    du = SimpleMvNormal(SA[0.0], SA[1e-10;;])
    xz_true, u, y = simulate(daeukf, T, du)
    @test length(xz_true) == T
    @test length(y)       == T
    @test all(z -> abs((z[1] + z[2]) - C1) < 1e-8, xz_true)

    sol = forward_trajectory(daeukf, u, y)
    err = sum(abs2(sol.xt[k][1] - xz_true[k][1]) for k in 1:T) / T
    @test err < 5 * R
end

@testset "in-place dynamics and measurement (Test 1 system)" begin
    # In-place versions of Test 1's dynamics and measurement. The constructor
    # auto-detects IPD/IPM via `has_oop`, so passing 5-arg signatures forces
    # IPD=IPM=true and exercises the `if IPD ...` / `if IPM ...` branches
    # inside predict!/correct!.
    function t1_dynamics_ip!(xz_next, xz, u, p, t)
        x, z = get_x_z_scalar(xz)
        new_x = x .+ DT1 .* t1_diff_dynamics(x, z, u, p, t)
        prob  = NonlinearProblem((zv, _) -> t1_residual(new_x, zv, u, p, t), z)
        new_z = solve(prob, SimpleNewtonRaphson(), reltol=1e-12).u
        xz_next .= build_xz_scalar(new_x, new_z)
        return nothing
    end
    function t1_measurement_ip!(y, xz, u, p, t)
        y[1] = xz[2]
        return nothing
    end

    Q  = 0.05; R = 0.02; P0 = 0.5
    x0 = SA[0.3]; z0 = SA[C1] - x0
    # Vector-typed xz0 so the in-place buffers and assignments work without
    # SVector→MVector→SVector conversions on every sigma point.
    xz0_v = Vector(build_xz_scalar(x0, z0))
    d0    = SimpleMvNormal(x0, SA[P0;;])

    ukf_oop = DAEUnscentedKalmanFilter(t1_dynamics, t1_measurement,
                                       t1_residual, get_x_z_scalar, build_xz_scalar,
                                       SA[Q;;], SA[R;;], d0;
                                       xz0 = xz0_v, nu=1, ny=1, Ts=DT1,
                                       constraint_solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12))
    ukf_ip  = DAEUnscentedKalmanFilter(t1_dynamics_ip!, t1_measurement_ip!,
                                       t1_residual, get_x_z_scalar, build_xz_scalar,
                                       SA[Q;;], SA[R;;], d0;
                                       xz0 = xz0_v, nu=1, ny=1, Ts=DT1,
                                       constraint_solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12))

    # Constructor auto-detection of {IPD,IPM}.
    @test ukf_oop isa DAEUnscentedKalmanFilter{false, false}
    @test ukf_ip  isa DAEUnscentedKalmanFilter{true,  true}

    # Drive both with the same input/measurement sequence; the differential
    # state estimate, covariance, and on-manifold descriptor must agree.
    T = 100
    α, β = 1 - 2*DT1, DT1
    x_truth = 0.4
    u = SA[0.0]
    for k in 1:T
        x_truth = α*x_truth + β*C1 + sqrt(Q)*randn()
        y_k     = SA[(C1 - x_truth) + sqrt(R)*randn()]
        predict!(ukf_oop, u); predict!(ukf_ip, u)
        @test ukf_oop.x ≈ ukf_ip.x
        @test ukf_oop.R ≈ ukf_ip.R
        correct!(ukf_oop, u, y_k); correct!(ukf_ip, u, y_k)
        @test ukf_oop.x  ≈ ukf_ip.x
        @test ukf_oop.R  ≈ ukf_ip.R
        @test ukf_oop.xz ≈ ukf_ip.xz
    end

    # simulate must work with in-place dynamics/measurement as well; this
    # exercises the IPD/IPM branches of sample_state and sample_measurement.
    du = SimpleMvNormal(SA[0.0], SA[1e-10;;])
    xz_sim, u_sim, y_sim = simulate(ukf_ip, 20, du)
    @test length(xz_sim) == 20
    @test length(y_sim)  == 20
    @test all(z -> abs((z[1] + z[2]) - C1) < 1e-8, xz_sim)
end


# ============================================================================
# Pendulum DAE: 4 differential states (x, y, ẋ, ẏ) + 1 algebraic Lagrange
# multiplier λ. Index-1 reduction of the position constraint x² + y² = 1 via
# two differentiations and substitution. Ported (sans threading) from the
# commented block in test_ukf.jl.
#
# Dynamics:
#   ẋ = u
#   ẏ = v
#   u̇ = -λ x + f₁
#   v̇ = -λ y - g + f₂
#   0  = u² + v² - λ(x² + y²) - g·y + x·f₁ + y·f₂      (centripetal acceleration)
#
# Measurement: position x[1] and the Lagrange multiplier λ.
# ============================================================================

const G_PEND = 9.82

function pend_full(xz, f, p, t)
    x, y, u, v, λ = xz
    SA[u,
       v,
       -λ*x + f[1],
       -λ*y - G_PEND + f[2],
       u^2 + v^2 - λ*(x^2 + y^2) - G_PEND*y + x*f[1] + y*f[2]]
end

pend_get_x_z(xz)  = (SA[xz[1], xz[2], xz[3], xz[4]], SA[xz[5]])
pend_build_xz(x, z) = vcat(x, z)

function pend_residual(x, z, f, p, t)
    xp, yp, up, vp = x
    λ = z[1]
    SA[up^2 + vp^2 - λ*(xp^2 + yp^2) - G_PEND*yp + xp*f[1] + yp*f[2]]
end

pend_measurement(xz, f, p, t) = SA[xz[1], xz[5]]

# Forward-Euler discretization with internal supersampling and a constraint
# reprojection at every sub-step.
function pend_dynamics(xz, f, p, t)
    PEND_TS = 0.01
    supersample = 10
    dt = PEND_TS / supersample
    for _ in 1:supersample
        der = pend_full(xz, f, p, t)
        x_part = SA[xz[1] + dt*der[1],
                    xz[2] + dt*der[2],
                    xz[3] + dt*der[3],
                    xz[4] + dt*der[4]]
        prob = NonlinearProblem((z, _) -> pend_residual(x_part, z, f, p, t), SA[xz[5]])
        z_new = solve(prob, SimpleNewtonRaphson(), reltol=1e-12).u
        xz = pend_build_xz(x_part, z_new)
    end
    xz
end

@testset "Pendulum DAE (constrained mechanical system)" begin
    PEND_TS = 0.01
    nx, nu, ny = 4, 2, 2

    d0 = SimpleMvNormal(SA[1.0, 0.0, 0.0, 0.0], SMatrix{4,4}(0.1*I))
    xz0 = SA[1.0, 0.0, 0.0, 0.0, 0.0]   # at rest at (1, 0), zero λ satisfies the constraint
    u0  = SA[0.0, 0.0]

    daeukf = DAEUnscentedKalmanFilter(
        pend_dynamics, pend_measurement, pend_residual,
        pend_get_x_z, pend_build_xz,
        SMatrix{4,4}(1e-4*I), SMatrix{2,2}(1e-2*I), d0;
        xz0, nu, ny, Ts = PEND_TS,
        constraint_solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12),
    )

    # ---- one-step dynamics keeps the descriptor on the constraint manifold ----
    xz_one = pend_dynamics(xz0, u0, daeukf.p, 0.0)
    x_one, z_one = pend_get_x_z(xz_one)
    @test pend_residual(x_one, z_one, u0, daeukf.p, 0.0)[1] ≈ 0 atol = 1e-10

    # ---- calc_xz from rest at (1, 0): λ should be ≈ 0 ----
    xz_rest = calc_xz(daeukf, xz0, u0, daeukf.p, 0.0)
    @test xz_rest[5] ≈ 0 atol = 1e-10
    x_r, z_r = pend_get_x_z(xz_rest)
    @test pend_residual(x_r, z_r, u0, daeukf.p, 0.0)[1] ≈ 0 atol = 1e-10

    # ---- calc_xz on a random off-manifold input still lands on the manifold ----
    Random.seed!(7)
    rand_xz = SVector{5}(randn(5))
    xz_proj = calc_xz(daeukf, rand_xz, u0, daeukf.p, 0.0)
    x_p, z_p = pend_get_x_z(xz_proj)
    @test pend_residual(x_p, z_p, u0, daeukf.p, 0.0)[1] ≈ 0 atol = 1e-10
    # `xi` defaults to the differential slice of the input; the differential
    # part should pass through, only `z` should change.
    @test x_p == SVector{4}(rand_xz[1], rand_xz[2], rand_xz[3], rand_xz[4])

    # ---- forward_trajectory: posterior covariance decreases on every correct! ----
    Random.seed!(123)
    T = 60
    t_vec = (0:T-1) .* PEND_TS
    u_drive = [SA[sin(tt^2), sin(tt^2 + 1.0)] for tt in t_vec]
    xz_true, u, y = simulate(daeukf, u_drive)

    sol = forward_trajectory(daeukf, u, y)

    @test all(zip(sol.R, sol.Rt)) do (R, Rt)
        det(Rt) < det(R)
    end

    # ---- tracking error after burn-in (relative RMSE on differential state) ----
    x_true_diff = [pend_get_x_z(xz)[1] for xz in xz_true]
    burn = 5
    err_pred = norm(reduce(hcat, x_true_diff[burn:end] .- sol.x[burn:end])) /
               norm(reduce(hcat, x_true_diff))
    err_filt = norm(reduce(hcat, x_true_diff[burn:end] .- sol.xt[burn:end])) /
               norm(reduce(hcat, x_true_diff))
    @test err_pred < 0.2
    @test err_filt < 0.2
end


# ============================================================================
# Off-nominal / defensive-branch coverage
# ----------------------------------------------------------------------------
# The tests above exercise the happy paths. The ones below close the remaining
# coverage on the guards in the new DAE-UKF code and verify each guard fires
# for the reason it exists, not merely that the line executes. All reuse the
# Test-1 scalar system (t1_dynamics / t1_measurement / t1_residual /
# get_x_z_scalar / build_xz_scalar, C1, DT1) defined above.
# ============================================================================

# Capture a thrown exception (robust across Julia versions; mirrors the style
# in test_getmat_3d_time.jl).
function _catch_err(f)
    try
        f()
        nothing
    catch e
        e
    end
end

# Build a Test-1 DAE-UKF, allowing per-test keyword overrides.
function _mk_t1(; Q = 0.05, R = 0.02, P0 = 0.5, x0 = SA[0.3],
                measurement = t1_measurement,
                solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12),
                kw...)
    z0  = SA[C1] - x0
    xz0 = build_xz_scalar(x0, z0)
    d0  = SimpleMvNormal(x0, SA[P0;;])
    DAEUnscentedKalmanFilter(t1_dynamics, measurement, t1_residual,
                             get_x_z_scalar, build_xz_scalar,
                             SA[Q;;], SA[R;;], d0;
                             xz0, nu = 1, ny = 1, Ts = DT1,
                             constraint_solver = solver, kw...)
end


@testset "custom unweighted state_mean/state_cov reproduce the defaults" begin
    # A user may supply mean/cov functions that take only the sigma points.
    # The constructor's `if !hasmethod(state_mean, (AbstractVector, UKFWeights))`
    # shims (src/ukf.jl:1087-1095) wrap them as (xs, w) -> f(xs). Under
    # TrivialParams the equal-weight UKF mean/cov ARE the sample mean and the
    # (n-1)-normalized sample covariance, so a filter using these 1-arg
    # functions must reproduce the default-function filter step-for-step.
    custom_mean = xs -> mean(xs)
    custom_cov  = xs -> (m = mean(xs); sum((x .- m) * (x .- m)' for x in xs) ./ (length(xs) - 1))

    kf_def = _mk_t1()
    kf_cus = _mk_t1(state_mean = custom_mean, state_cov = custom_cov)

    T = 100
    α, β = 1 - 2*DT1, DT1
    x_true = 0.35
    u = SA[0.0]
    Random.seed!(20)
    for k in 1:T
        x_true = α*x_true + β*C1 + sqrt(0.05)*randn()
        y_k    = SA[(C1 - x_true) + sqrt(0.02)*randn()]
        predict!(kf_def, u); predict!(kf_cus, u)
        correct!(kf_def, u, y_k); correct!(kf_cus, u, y_k)
    end
    @test kf_cus.x ≈ kf_def.x rtol = 1e-10
    @test kf_cus.R ≈ kf_def.R rtol = 1e-10
end


@testset "unweighted mean/cov combined with custom weights is rejected" begin
    # The `weight_params isa TrivialParams || error(...)` guards (src/ukf.jl:1088,
    # 1093) catch the unsupported combination of a sigma-points-only mean/cov
    # with non-trivial sigma-point weights, where the user weights would be
    # silently ignored.
    custom_mean = xs -> mean(xs)
    custom_cov  = xs -> (m = mean(xs); sum((x .- m) * (x .- m)' for x in xs) ./ (length(xs) - 1))

    e_mean = _catch_err(() -> _mk_t1(state_mean = custom_mean, weight_params = MerweParams()))
    @test e_mean isa ErrorException
    @test occursin("Unweighted state mean may not be used with custom weights",
                   sprint(showerror, e_mean))

    e_cov = _catch_err(() -> _mk_t1(state_cov = custom_cov, weight_params = MerweParams()))
    @test e_cov isa ErrorException
    @test occursin("Unweighted state covariance may not be used with custom weights",
                   sprint(showerror, e_cov))
end


@testset "getproperty: derived properties and invalid access" begin
    # Property forwarding (src/ukf.jl:1192-1205): nx/nw are derived from the
    # differential state, `measurement` is fetched from the measurement model,
    # and anything else is a clear ArgumentError rather than a silent failure.
    kf = _mk_t1()
    @test kf.nx == 1                      # :nx branch
    @test kf.nw == kf.nx                  # :nw branch (AUGD=false ⇒ nx)
    @test kf.measurement === t1_measurement   # forwarded to the measurement model's field
    @test kf.R2 === SA[0.02;;]                 # forwarded to the measurement model's field

    e = _catch_err(() -> getproperty(kf, :definitely_not_a_field))
    @test e isa ArgumentError
    @test occursin("has no property named", sprint(showerror, e))
end


@testset "calc_xz warns on a large constraint-solve residual" begin
    # calc_xz warns (src/ukf.jl:1180) when the algebraic solve leaves a large
    # residual — a sign the constraint was not satisfied. Use a no-op solver
    # that returns the initial guess unchanged so the residual stays large.
    bad_solver = (f, z0) -> z0
    kf = _mk_t1(solver = bad_solver)
    xi = SA[5.0]   # far off the manifold x + z = C1, so the residual ≈ 4.7 ≫ 1e-3
    @test_logs (:warn, r"constraint solve residual was large") match_mode = :any begin
        calc_xz(kf, kf.xz, SA[0.0], kf.p, 0.0, xi)
    end
end


@testset "AUGM=true is rejected by correct! and sample_measurement" begin
    # AUGM=true (augmented measurement noise) is plumbed but not implemented;
    # the guards (src/ukf.jl:1344, 1469) must reject it with a clear error.
    Q, R, P0 = 0.05, 0.02, 0.5
    x0  = SA[0.3]
    xz0 = build_xz_scalar(x0, SA[C1] - x0)
    d0  = SimpleMvNormal(x0, SA[P0;;])
    solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12)
    kf_augm = DAEUnscentedKalmanFilter{false,false,false,true}(
        t1_dynamics, t1_measurement, t1_residual, get_x_z_scalar, build_xz_scalar,
        SA[Q;;], SA[R;;], d0; xz0, nu = 1, ny = 1, Ts = DT1, constraint_solver = solver)
    @test kf_augm isa DAEUnscentedKalmanFilter{false,false,false,true}

    e_c = _catch_err(() -> correct!(kf_augm, SA[0.0], SA[0.0]))
    @test e_c isa ErrorException
    @test occursin("AUGM=true is not yet supported", sprint(showerror, e_c))

    e_s = _catch_err(() -> sample_measurement(kf_augm, xz0, SA[0.0]))
    @test e_s isa ErrorException
    @test occursin("AUGM=true is not yet supported", sprint(showerror, e_s))
end


@testset "AUGD=true is rejected by the constructor" begin
    # AUGD=true (augmented process noise) is not implemented; predict! always
    # runs the additive-noise path, so the constructor must refuse the flag
    # rather than silently ignore it.
    Q, R, P0 = 0.05, 0.02, 0.5
    x0  = SA[0.3]
    xz0 = build_xz_scalar(x0, SA[C1] - x0)
    d0  = SimpleMvNormal(x0, SA[P0;;])
    solver = LowLevelParticleFilters.scimlbase_solver(SimpleNewtonRaphson(); reltol = 1e-12)
    e = _catch_err(() -> DAEUnscentedKalmanFilter{false,false,true,false}(
        t1_dynamics, t1_measurement, t1_residual, get_x_z_scalar, build_xz_scalar,
        SA[Q;;], SA[R;;], d0; xz0, nu = 1, ny = 1, Ts = DT1, constraint_solver = solver))
    @test e isa ErrorException
    @test occursin("AUGD=true is not yet supported", sprint(showerror, e))
end


@testset "non-positive-definite innovation covariance raises a clear error" begin
    # The Cholesky guard in correct! (src/ukf.jl:1372-1373) catches a broken-down
    # innovation covariance. A non-informative (constant) measurement makes
    # cov(ys) = 0, so S = cov(ys) + R2 = R2; overriding R2 with an indefinite
    # value drives S non-PD and the factorization must fail loudly.
    const_meas = (xz, u, p, t) -> SA[0.0]
    kf = _mk_t1(measurement = const_meas)
    predict!(kf, SA[0.0])
    e = _catch_err(() -> correct!(kf, SA[0.0], SA[0.0]; R2 = SA[-1.0;;]))
    @test e isa ErrorException
    @test occursin("innovation covariance failed", sprint(showerror, e))
end
