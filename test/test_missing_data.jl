using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal, ismissing_measurement
using Test, Random, LinearAlgebra, StaticArrays, Distributions, Plots
import ForwardDiff
Random.seed!(0)

nx, nu, ny = 2, 1, 1
A = SA[0.99 0.1; 0 0.95]
B = SA[0.0; 1.0;;]
C = SA[1.0 0.0]
R1 = SA[0.01 0; 0 0.02]
R2 = SA[0.1;;]
d0 = SimpleMvNormal(SA[1.0, 0.0], SMatrix{2,2}(1.0I(2)))
dynamics(x, u, p, t) = A*x + B*u
measurement(x, u, p, t) = C*x

T = 60
kf = KalmanFilter(A, B, C, 0, R1, R2, d0)
x, u, y = simulate(kf, [SA[randn()] for _ in 1:T])
miss_inds = [1, 2, 20, 21, 22, 40, T]
ym = Vector{Union{Missing, eltype(y)}}(y)
ym[miss_inds] .= missing
avail = setdiff(1:T, miss_inds)

"Reference implementation that skips the correction step manually"
function manual_filter(kf, u, y)
    reset!(kf)
    xs, xts, ll = [], [], 0.0
    for k in eachindex(y)
        push!(xs, state(kf))
        if y[k] !== missing
            ll += correct!(kf, u[k], y[k])[1]
        end
        push!(xts, state(kf))
        predict!(kf, u[k])
    end
    xs, xts, ll
end

@testset "ismissing_measurement" begin
    @test ismissing_measurement(missing)
    @test !ismissing_measurement(SA[1.0])
    @test !ismissing_measurement([1.0, 2.0])
    @test ismissing_measurement(SVector{2,Missing}(missing, missing))
    @test ismissing_measurement([missing, missing])
    @test_throws ArgumentError ismissing_measurement([1.0, missing])
    @test !ismissing_measurement(Union{Missing,Float64}[1.0, 2.0])
end

@testset "KalmanFilter with missing measurements" begin
    sol = forward_trajectory(kf, u, ym)
    xs, xts, llref = manual_filter(kf, u, ym)
    @test sol.x ≈ xs
    @test sol.xt ≈ xts
    @test sol.ll ≈ llref
    for k in miss_inds
        @test sol.xt[k] == sol.x[k]
        @test sol.Rt[k] == sol.R[k]
        @test sol.e[k] === missing
        @test sol.S[k] === missing
        @test sol.K[k] === missing
    end
    @test all(sol.S[k] !== missing for k in avail)

    # Equivalence with an inflated measurement covariance
    yfill = [yk === missing ? SA[0.0] : yk for yk in ym]
    pre_correct_cb = (kf, u, y, p, t) -> t/kf.Ts+1 ∈ miss_inds ? 1e12*R2 : nothing
    sol_inf = forward_trajectory(kf, u, yfill; pre_correct_cb)
    @test sol_inf.xt ≈ sol.xt rtol=1e-6
    @test sol_inf.x ≈ sol.x rtol=1e-6

    # Complete data gives unchanged element types
    sol_full = forward_trajectory(kf, u, y)
    @test eltype(sol_full.S) <: Cholesky
    @test sol_full.ll ≈ forward_trajectory(kf, u, Vector{Union{Missing, eltype(y)}}(y)).ll

    @test loglik(kf, u, ym) ≈ sol.ll
    sse_ref = sum(sol.e[k]'sol.e[k] for k in avail)
    @test LowLevelParticleFilters.sse(kf, u, ym) ≈ sse_ref

    res = zeros(T*ny)
    LowLevelParticleFilters.prediction_errors!(res, kf, u, ym)
    @test res'res ≈ sse_ref
    @test all(iszero, res[miss_inds])

    offset = 1.0
    resl = zeros(T*(ny+1))
    LowLevelParticleFilters.prediction_errors!(resl, kf, u, ym; loglik=true, offset)
    @test resl'resl ≈ -sol.ll + T*offset
    @test_throws ErrorException LowLevelParticleFilters.prediction_errors!(resl, kf, u, ym; loglik=true, offset=-1.0)

    # All entries missing as a static vector
    ysv = [yk === missing ? SVector{1,Missing}(missing) : yk for yk in ym]
    @test forward_trajectory(kf, u, ysv).ll ≈ sol.ll
    @test loglik(kf, u, ysv) ≈ sol.ll

    # Partially missing measurements are not supported
    yp = [Union{Missing,Float64}[yk[1], 1.0] for yk in y]
    yp[3] = [missing, 1.0]
    kf2 = KalmanFilter(A, B, [C; C], 0, R1, SMatrix{2,2}(0.1I(2)), d0)
    @test_throws ArgumentError forward_trajectory(kf2, u, yp)
    @test_throws ArgumentError loglik(kf2, u, yp)

    # All measurements missing
    yall = fill(missing, T)
    sol_all = forward_trajectory(kf, u, yall)
    @test sol_all.ll == 0
    @test sol_all.xt == sol_all.x
    @test all(ismissing, sol_all.S)

    # Smoothing
    ssol = smooth(sol)
    @test all(isfinite, reduce(hcat, ssol.xT))
    ssol_mbf = LowLevelParticleFilters.smooth_mbf(sol)[1]
    @test reduce(hcat, ssol_mbf.xT) ≈ reduce(hcat, ssol.xT) rtol=1e-6
    @test reduce(hcat, ssol_mbf.RT) ≈ reduce(hcat, ssol.RT) rtol=1e-6
    # Smoothed estimates over the gap are better than the filter estimates
    gap = 20:22
    @test sum(abs2, reduce(hcat, ssol.xT[gap]) - reduce(hcat, x[gap])) < sum(abs2, reduce(hcat, sol.xt[gap]) - reduce(hcat, x[gap]))

    # Plotting
    plot(sol; plote=true, plotS=true, plotR=true, plotRt=true)
    plot(ssol)
    validationplot(sol)
    # time-varying A
    A3 = cat([A + 0.001k*I for k in 1:T]..., dims=3)
    kftv = KalmanFilter(A3, B, C, 0, R1, R2, d0)
    soltv = forward_trajectory(kftv, u, ym)
    xs, xts, llref = manual_filter(kftv, u, ym)
    @test soltv.xt ≈ xts
    @test soltv.ll ≈ llref
end

@testset "Gradient with missing measurements" begin
    function cost(θ)
        T_ = eltype(θ)
        kfθ = KalmanFilter(A, B, C, 0, exp(θ[1])*R1, exp(θ[2])*R2, SimpleMvNormal(T_.(d0.μ), T_.(d0.Σ)); check=false)
        -loglik(kfθ, u, ym)
    end
    g = ForwardDiff.gradient(cost, [0.1, -0.2])
    @test all(isfinite, g)
    h = 1e-6
    gfd = [(cost([0.1+h, -0.2]) - cost([0.1-h, -0.2]))/(2h), (cost([0.1, -0.2+h]) - cost([0.1, -0.2-h]))/(2h)]
    @test g ≈ gfd rtol=1e-4
end

@testset "Other Kalman-type filters with missing measurements" begin
    solkf = forward_trajectory(kf, u, ym)
    filters = [
        SqKalmanFilter(A, B, C, 0, R1, R2, d0),
        ExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; nu),
        IteratedExtendedKalmanFilter(dynamics, measurement, R1, R2, d0; nu),
        UnscentedKalmanFilter(dynamics, measurement, R1, R2, d0; ny, nu),
    ]
    for f in filters
        sol = forward_trajectory(f, u, ym)
        @test sol.ll ≈ solkf.ll rtol=1e-6
        @test sol.xt ≈ solkf.xt rtol=1e-6
        @test loglik(f, u, ym) ≈ solkf.ll rtol=1e-6
        @test all(sol.xt[k] == sol.x[k] for k in miss_inds)
        @test LowLevelParticleFilters.sse(f, u, ym) ≈ LowLevelParticleFilters.sse(kf, u, ym) rtol=1e-6
        f isa SqKalmanFilter || smooth(sol) # No smoother is implemented for SqKalmanFilter
    end

    # Augmented UKF with noise entering through a disturbance input
    Bw = SA[0.0; 1.0;;]
    dynamics_w(x, u, p, t, w) = A*x + B*u + Bw*w
    Rw = SA[0.02;;]
    ukfw = UnscentedKalmanFilter{false,false,true,false}(dynamics_w, measurement, Rw, R2, d0; ny, nu)
    kfw = KalmanFilter(A, B, C, 0, Bw*Rw*Bw', R2, d0)
    solw = forward_trajectory(ukfw, u, ym)
    @test solw.ll ≈ forward_trajectory(kfw, u, ym).ll rtol=1e-6
    @test all(solw.xt[k] == solw.x[k] for k in miss_inds)

    # Ensemble Kalman filter
    enkf = EnsembleKalmanFilter(dynamics, measurement, R1, R2, d0, 200; nu, ny)
    sole = forward_trajectory(enkf, u, ym)
    @test isfinite(sole.ll)
    @test all(sole.e[k] === missing for k in miss_inds)
    @test isfinite(loglik(enkf, u, ym))

    # IMM, the mode probabilities follow the transition matrix during missing measurements
    kf1 = KalmanFilter(A, B, C, 0, R1, R2, d0)
    kf2 = KalmanFilter(A, B, C, 0, 10R1, R2, d0)
    P = [0.9 0.1; 0.2 0.8]
    imm = IMM([kf1, kf2], P, [0.5, 0.5])
    soli = forward_trajectory(imm, u, ym)
    @test isfinite(soli.ll)
    μ = soli.extra
    for k in miss_inds[2:end]
        k == 1 && continue
        @test μ[:, k] ≈ P'μ[:, k-1]
    end
    @test loglik(imm, u, ym) ≈ soli.ll

    # The unknown-input Kalman filter requires all measurements
    uikf = UIKalmanFilter(A, B, C, zeros(ny, nu), B, R1, R2; nu, ny)
    @test_throws ArgumentError forward_trajectory(uikf, u, ym)
    @test_throws ArgumentError loglik(uikf, u, ym)
end

@testset "Particle filters with missing measurements" begin
    pf = ParticleFilter(500, dynamics, measurement, MvNormal(zeros(2), Matrix(R1)), MvNormal(zeros(1), Matrix(R2)), MvNormal(Vector(d0.μ), Matrix(d0.Σ)); nu, ny)
    solpf = forward_trajectory(pf, u, ym)
    @test isfinite(solpf.ll)
    @test isfinite(loglik(pf, u, ym))
    apf = AuxiliaryParticleFilter(pf)
    @test isfinite(loglik(apf, u, ym))
end

@testset "Unsupported filters in sse and prediction_errors!" begin
    pf = ParticleFilter(100, dynamics, measurement, MvNormal(zeros(2), Matrix(R1)), MvNormal(zeros(1), Matrix(R2)), MvNormal(Vector(d0.μ), Matrix(d0.Σ)); nu, ny)
    @test_throws ArgumentError LowLevelParticleFilters.sse(pf, u, y)
    @test_throws ArgumentError LowLevelParticleFilters.prediction_errors!(zeros(T*ny), pf, u, y)
    imm = IMM([kf, kf], [0.5 0.5; 0.5 0.5], [0.5, 0.5])
    @test_throws ArgumentError LowLevelParticleFilters.sse(imm, u, y)
    @test_throws ArgumentError LowLevelParticleFilters.prediction_errors!(zeros(T*ny), imm, u, y)
end
