using LowLevelParticleFilters
import LowLevelParticleFilters as LLPF
import LowLevelParticleFilters: SimpleMvNormal
using Test, Random, LinearAlgebra, Statistics, StaticArrays

Random.seed!(0)

## Linear jump Markov system with two modes ===================================
const A1s = SA[1.0 0.1; 0 1.0]
const A2s = SA[1.0 0.1; 0 0.6]
const Bs  = SA[0.0; 0.1;;]
const Cs  = SA[1.0 0.0]
const R11s = SA[1e-4 0; 0 1e-3]
const R12s = SA[1e-4 0; 0 0.3]
const R2s  = SA[0.05;;]
const Ptr  = [0.95 0.05; 0.1 0.9]
const μ0   = [0.5, 0.5]
const d0s  = SimpleMvNormal(@SVector(zeros(2)), SMatrix{2,2}(1.0I))

make_kfs() = (KalmanFilter(A1s, Bs, Cs, 0, R11s, R2s, d0s), KalmanFilter(A2s, Bs, Cs, 0, R12s, R2s, d0s))

function simulate_jmls(T, rng)
    s = zeros(Int, T)
    x = Vector{Vector{Float64}}(undef, T)
    y = similar(x)
    u = [randn(rng, 1) for _ in 1:T]
    s[1] = rand(rng) < (Ptr'μ0)[1] ? 1 : 2
    x[1] = Vector(d0s.μ) .+ randn(rng, 2)
    for k in 1:T
        if k > 1
            s[k] = rand(rng) < Ptr[s[k-1], 1] ? 1 : 2
            A, R1 = s[k] == 1 ? (A1s, R11s) : (A2s, R12s)
            x[k] = A*x[k-1] + Bs*u[k-1] + cholesky(Matrix(R1)).L*randn(rng, 2)
        end
        y[k] = Cs*x[k] + sqrt(R2s[1])*randn(rng, 1)
    end
    x, u, y, s
end

# Exact posterior means and mode probabilities by enumerating all 2^T mode sequences
function exact_smoother(u, y)
    T = length(y)
    lws = Float64[]
    means = Vector{Vector{Vector{Float64}}}()
    mprobs = Vector{Matrix{Float64}}()
    for bits in 0:(2^T-1)
        s = [((bits >> (k-1)) & 1) + 1 for k in 1:T]
        lw = log((Ptr'μ0)[s[1]]) + sum(log(Ptr[s[k-1], s[k]]) for k in 2:T; init=0.0)
        xp = [zeros(2) for _ in 1:T]; Pp = [zeros(2,2) for _ in 1:T]
        xt = similar(xp); Pt = similar(Pp)
        xp[1] = Vector(d0s.μ); Pp[1] = Matrix(d0s.Σ)
        for k in 1:T
            if k > 1
                A, R1 = s[k] == 1 ? (A1s, R11s) : (A2s, R12s)
                xp[k] = A*xt[k-1] + Bs*u[k-1]; Pp[k] = A*Pt[k-1]*A' + R1
            end
            S = Cs*Pp[k]*Cs' + R2s; e = y[k] - Cs*xp[k]; K = Pp[k]*Cs'/S
            lw += -0.5*(dot(e, S\e) + logdet(S))
            xt[k] = xp[k] + K*e; Pt[k] = (I - K*Cs)*Pp[k]
        end
        xs = copy(xt)
        for k in T-1:-1:1
            A = s[k+1] == 1 ? A1s : A2s
            G = Pt[k]*A'/Pp[k+1]
            xs[k] = xt[k] + G*(xs[k+1] - xp[k+1])
        end
        mp = zeros(2, T)
        for k in 1:T
            mp[s[k], k] = 1
        end
        push!(lws, lw); push!(means, xs); push!(mprobs, mp)
    end
    w = exp.(lws .- maximum(lws)); w ./= sum(w)
    sum(w .* means), sum(w .* mprobs)
end

kf1, kf2 = make_kfs()
T = 100
x, u, y, s = simulate_jmls(T, MersenneTwister(1))

@testset "Single model reduces to RTS smoother" begin
    kf = deepcopy(kf1)
    rts = smooth(kf, u, y)
    imm = IMM([deepcopy(kf1)], ones(1,1), [1.0])
    ssol = smooth(imm, u, y)
    @test ssol isa KalmanSmoothingSolution
    @test all(isapprox.(ssol.xT, rts.xT, atol=1e-7))
    @test all(isapprox.(ssol.RT, rts.RT, atol=1e-7))
    @test all(ssol.extraT .≈ 1)

    # Two identical models also reduce to the RTS smoother, the mode probabilities are uninformed
    imm = IMM([deepcopy(kf1), deepcopy(kf1)], Ptr, μ0)
    ssol = smooth(imm, u, y)
    @test all(isapprox.(ssol.xT, rts.xT, atol=1e-7))
    @test all(isapprox.(ssol.RT, rts.RT, atol=1e-7))
    @test ssol.extraT ≈ ssol.extra # Filtered mode probabilities equal the prior here
    prior = Ptr'μ0
    @test ssol.extraT[:, 1] ≈ prior

    # The initialization using the forward filter double counts information at the end of the interval
    ssolf = smooth(imm, u, y; backward_init = :forward)
    @test tr(ssolf.RT[end-1]) < tr(rts.RT[end-1])
    @test_throws ArgumentError smooth(imm, u, y; backward_init = :foo)
end

@testset "Close to exact smoother on short sequences" begin
    rng = MersenneTwister(3)
    e_filt = 0.0; e_smooth = 0.0; e_exact = 0.0; d_smooth = 0.0; d_filt = 0.0; m_smooth = 0.0; m_filt = 0.0
    N = 30
    for rep in 1:N
        Ts = 10
        x, u, y, s = simulate_jmls(Ts, rng)
        imm = IMM(collect(make_kfs()), Ptr, μ0)
        ssol = smooth(imm, u, y)
        xe, me = exact_smoother(u, y)
        @test all(sum(ssol.extraT, dims=1) .≈ 1)
        e_filt   += mean(sum(abs2, ssol.xt[k] - x[k]) for k in 1:Ts)
        e_smooth += mean(sum(abs2, ssol.xT[k] - x[k]) for k in 1:Ts)
        e_exact  += mean(sum(abs2, xe[k] - x[k]) for k in 1:Ts)
        d_filt   += mean(sum(abs2, ssol.xt[k] - xe[k]) for k in 1:Ts)
        d_smooth += mean(sum(abs2, ssol.xT[k] - xe[k]) for k in 1:Ts)
        m_filt   += mean(abs, ssol.extra .- me)
        m_smooth += mean(abs, ssol.extraT .- me)
    end
    @test e_smooth < 0.6e_filt
    @test e_smooth < 1.1e_exact
    @test d_smooth < 0.1d_filt   # The smoothed estimate is much closer to the exact posterior mean than the filtered estimate
    # The smoothed mode probabilities of the HBH smoother are not closer to the exact posterior mode
    # probabilities than the filtered ones in this example. The approximation of Helmick et al. replaces
    # the likelihood of future data under mode i by a Gaussian density with the covariance of the backward
    # prediction under mode i, which favours modes with small process noise (mode 1 here).
    @test_broken m_smooth < m_filt
end

@testset "Mode estimation on a longer sequence" begin
    imm = IMM(collect(make_kfs()), Ptr, μ0)
    ssol = smooth(imm, u, y)
    acc_filt   = mean((ssol.extra[1, :]  .> 0.5) .== (s .== 1))
    acc_smooth = mean((ssol.extraT[1, :] .> 0.5) .== (s .== 1))
    @test acc_smooth >= acc_filt
    rmse(xh) = sqrt(mean(sum(abs2, xh[k] - x[k]) for k in 1:T))
    @test rmse(ssol.xT) < rmse(ssol.xt)
    @test ssol.xT[end] ≈ ssol.xt[end]
    @test ssol.extraT[:, end] ≈ ssol.extra[:, end]

    # Calling smooth on a filtering solution
    sol = forward_trajectory(imm, u, y)
    ssol2 = smooth(sol)
    @test ssol2.xT ≈ ssol.xT
    @test ssol2.extraT ≈ ssol.extraT
    @test ssol2.xt ≈ sol.xt
    @test ssol2.ll ≈ sol.ll
    @test ssol2.extra ≈ sol.extra
end

@testset "Missing measurements" begin
    ym = Vector{Union{Missing, Vector{Float64}}}(copy(y))
    ym[20:25] .= missing
    imm = IMM(collect(make_kfs()), Ptr, μ0)
    ssol = smooth(imm, u, ym)
    @test all(all(isfinite, xi) for xi in ssol.xT)
    @test all(isfinite, ssol.extraT)
    # The smoother bridges the gap, so the uncertainty in the gap is smaller than that of the filter
    @test tr(ssol.RT[23]) < tr(ssol.Rt[23])
end

@testset "Nonlinear model types" begin
    imm_kf = IMM(collect(make_kfs()), Ptr, μ0)
    ref = smooth(imm_kf, u, y)

    # EKF with linear dynamics should give the same result as the KF
    dyn1(x, u, p, t) = A1s*x + Bs*u
    dyn2(x, u, p, t) = A2s*x + Bs*u
    meas(x, u, p, t) = Cs*x
    ekf1 = ExtendedKalmanFilter(dyn1, meas, R11s, R2s, d0s; nu = 1)
    ekf2 = ExtendedKalmanFilter(dyn2, meas, R12s, R2s, d0s; nu = 1)
    ssol = smooth(IMM([ekf1, ekf2], Ptr, μ0), u, y)
    @test all(isapprox.(ssol.xT, ref.xT, atol=1e-6))
    @test all(isapprox.(ssol.RT, ref.RT, atol=1e-6))
    @test ssol.extraT ≈ ref.extraT atol=1e-6

    # EKF with in-place dynamics
    dyn1!(xp, x, u, p, t) = (xp .= A1s*x + Bs*u)
    dyn2!(xp, x, u, p, t) = (xp .= A2s*x + Bs*u)
    measv(x, u, p, t) = Matrix(Cs)*x
    d0v = SimpleMvNormal(zeros(2), Matrix(1.0I(2)))
    ekf1 = ExtendedKalmanFilter(dyn1!, measv, Matrix(R11s), Matrix(R2s), d0v; nu = 1, ny = 1)
    ekf2 = ExtendedKalmanFilter(dyn2!, measv, Matrix(R12s), Matrix(R2s), d0v; nu = 1, ny = 1)
    @test ekf1 isa ExtendedKalmanFilter{true}
    ssol = smooth(IMM([ekf1, ekf2], Ptr, μ0), u, y)
    @test all(isapprox.(ssol.xT, ref.xT, atol=1e-6))
    @test all(isapprox.(ssol.RT, ref.RT, atol=1e-6))

    # UKF is exact for linear systems
    ukf1 = UnscentedKalmanFilter(dyn1, meas, R11s, R2s, d0s; ny = 1, nu = 1)
    ukf2 = UnscentedKalmanFilter(dyn2, meas, R12s, R2s, d0s; ny = 1, nu = 1)
    ssol = smooth(IMM([ukf1, ukf2], Ptr, μ0), u, y)
    @test all(isapprox.(ssol.xT, ref.xT, atol=1e-6))
    @test all(isapprox.(ssol.RT, ref.RT, atol=1e-6))

    # Nonlinear (invertible) dynamics
    fnl1(x, u, p, t) = SA[x[1] + 0.1*x[2], x[2] - 0.05*sin(x[2])] + Bs*u
    fnl2(x, u, p, t) = SA[x[1] + 0.1*x[2], 0.6*x[2] + 0.1*x[2]^3/(1 + x[2]^2)] + Bs*u
    ekf1 = ExtendedKalmanFilter(fnl1, meas, R11s, R2s, d0s; nu = 1)
    ekf2 = ExtendedKalmanFilter(fnl2, meas, R12s, R2s, d0s; nu = 1)
    ukf1 = UnscentedKalmanFilter(fnl1, meas, R11s, R2s, d0s; ny = 1, nu = 1)
    ukf2 = UnscentedKalmanFilter(fnl2, meas, R12s, R2s, d0s; ny = 1, nu = 1)
    for imm in (IMM([ekf1, ekf2], Ptr, μ0), IMM([ukf1, ukf2], Ptr, μ0))
        ssol = smooth(imm, u, y)
        @test all(all(isfinite, xi) for xi in ssol.xT)
        @test all(isfinite, ssol.extraT)
        @test all(sum(ssol.extraT, dims=1) .≈ 1)
        @test mean(sum(abs2, ssol.xT[k] - x[k]) for k in 1:T) < mean(sum(abs2, ssol.xt[k] - x[k]) for k in 1:T)
    end
end

@testset "Unsupported model types" begin
    sqkf = SqKalmanFilter(A1s, Bs, Cs, 0, R11s, R2s, d0s)
    imm = IMM([sqkf, sqkf], Ptr, μ0)
    @test_throws ArgumentError smooth(imm, u, y)
end
