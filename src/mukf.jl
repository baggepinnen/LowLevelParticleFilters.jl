
module MUKF

using LinearAlgebra
using LowLevelParticleFilters

export UTParams, MUKFState, init_mukf, reset!, predict!, correct!, step!, filter!, xl_mean

############################
# Unscented Transform (UT) #
############################

Base.@kwdef struct UTParams
    α::Float64 = 1e-3
    β::Float64 = 2.0
    κ::Float64 = 0.0
end

struct SigmaPack
    X::Matrix{Float64}   # n × (2n+1)
    Wm::Vector{Float64}
    Wc::Vector{Float64}
end

function sigma_points(μ::AbstractVector, P::AbstractMatrix, ut::UTParams)
    n = length(μ)
    λ = ut.α^2*(n + ut.κ) - n
    S = cholesky(Symmetric((n+λ)*P), check=false).L
    X = Matrix{Float64}(undef, n, 2n+1)
    X[:,1] = μ
    @inbounds for i in 1:n
        X[:,1+i]   = μ .+ view(S,:,i)
        X[:,1+n+i] = μ .- view(S,:,i)
    end
    Wm = zeros(Float64, 2n+1)
    Wc = zeros(Float64, 2n+1)
    Wm[1] = λ/(n+λ)
    Wc[1] = λ/(n+λ) + (1 - ut.α^2 + ut.β)
    @inbounds for j in 2:(2n+1)
        Wm[j] = 1/(2(n+λ))
        Wc[j] = 1/(2(n+λ))
    end
    return SigmaPack(X, Wm, Wc)
end

wmean(X::AbstractMatrix, W::AbstractVector) = mapreduce(identity, +, (W[j] .* view(X, :, j) for j in 1:size(X,2)))

function wcov(X::AbstractMatrix, μ::AbstractVector, Wc::AbstractVector)
    n = length(μ)
    P = zeros(Float64, n, n)
    @inbounds for j in 1:size(X,2)
        d = view(X, :, j) .- μ
        P .+= Wc[j] .* (d * d')
    end
    return P
end

########################
# MUKF filter state    #
########################

Base.@kwdef mutable struct MUKFState
    # model dimensions
    nxn::Int
    nxl::Int
    ny::Int

    # Nonlinear/linear model pieces
    fn::Function              # xⁿ_{k+1} = fₙ(xⁿ_k) + Aₙ xˡ_k + wⁿ_k
    g::Function               # y_k = g(xⁿ_k) + C xˡ_k + e_k
    An::Matrix{Float64}
    Al::Matrix{Float64}
    Cl::Matrix{Float64}

    # Noise covariances
    R1n::Matrix{Float64}
    R1l::Matrix{Float64}
    R2::Matrix{Float64}

    # Current estimates
    μn::Vector{Float64}
    Σn::Matrix{Float64}
    μl::Vector{Vector{Float64}}   # per-sigma means of xˡ
    Σl::Vector{Matrix{Float64}}   # per-sigma covariances of xˡ

    # UT params
    ut::UTParams
end

function init_mukf(; fn, g, An, Al, Cl, R1n, R1l, R2, d0n::LowLevelParticleFilters.SimpleMvNormal,
    d0l::LowLevelParticleFilters.SimpleMvNormal, ut::UTParams=UTParams())

    nxn = length(d0n.μ)
    nxl = length(d0l.μ)
    ny  = size(R2,1)

    # initialize sigma-conditioned linear KFs around prior of xˡ
    sp = sigma_points(d0n.μ, d0n.Σ, ut)
    μl = [copy(d0l.μ) for _ in 1:size(sp.X,2)]
    Σl = [copy(d0l.Σ) for _ in 1:size(sp.X,2)]

    return MUKFState(nxn, nxl, ny, fn, g, Matrix(An), Matrix(Al), Matrix(Cl), Matrix(R1n), Matrix(R1l), Matrix(R2),
                     copy(d0n.μ), copy(d0n.Σ), μl, Σl, ut)
end

reset!(f::MUKFState, d0n::LowLevelParticleFilters.SimpleMvNormal, d0l::LowLevelParticleFilters.SimpleMvNormal) = begin
    f.μn .= d0n.μ
    f.Σn .= d0n.Σ
    sp = sigma_points(f.μn, f.Σn, f.ut)
    for i in 1:length(f.μl)
        f.μl[i] .= d0l.μ
        f.Σl[i] .= d0l.Σ
    end
    f
end

function predict!(f::MUKFState)
    sp = sigma_points(f.μn, f.Σn, f.ut)

    # linear substate time update per sigma point: xˡ⁺ = Al xˡ + wˡ
    for i in 1:length(f.μl)
        f.μl[i] = f.Al * f.μl[i]
        f.Σl[i] = f.Al * f.Σl[i] * f.Al' .+ f.R1l
    end

    # propagate xⁿ using per-sigma xˡ means
    Xn_pred = similar(sp.X)
    @inbounds for i in 1:size(sp.X,2)
        Xn_pred[:,i] = f.fn(view(sp.X,:,i), nothing, nothing, 0) .+ f.An * f.μl[i]
    end

    μn_pred = wmean(Xn_pred, sp.Wm)
    Σn_pred = wcov(Xn_pred, μn_pred, sp.Wc) .+ f.R1n

    f.μn .= μn_pred
    f.Σn .= Σn_pred
    return f
end

function correct!(f::MUKFState, y::AbstractVector)
    sp = sigma_points(f.μn, f.Σn, f.ut)

    # predicted measurement per sigma: ŷᵢ = g(xⁿᵢ) + C xˡᵢ
    Y = zeros(f.ny, size(sp.X,2))
    Σy = zeros(f.ny, f.ny)
    @inbounds for i in 1:size(sp.X,2)
        Y[:,i] = f.g(view(sp.X,:,i), 0, 0, 0) .+ f.Cl * f.μl[i]
        Σy .+= sp.Wc[i] .* (f.Cl * f.Σl[i] * f.Cl')  # from linear substate uncertainty
    end
    yhat = wmean(Y, sp.Wm)

    # innovation covariance and cross-covariance
    S = wcov(Y, yhat, sp.Wc) .+ Σy .+ f.R2

    Σny = zeros(f.nxn, f.ny)
    @inbounds for i in 1:size(sp.X,2)
        dx = view(sp.X,:,i) .- f.μn
        dy = view(Y,:,i)   .- yhat
        Σny .+= sp.Wc[i] .* (dx * dy')
    end

    # UKF-style update for xⁿ
    Kn = Σny * inv(S)
    f.μn .= f.μn .+ Kn * (y .- yhat)
    f.Σn .= f.Σn .- Kn * S * Kn'

    # per-sigma linear KF measurement updates with residuals
    @inbounds for i in 1:size(sp.X,2)
        r_i = y .- f.g(view(sp.X,:,i), 0, 0, 0) .- f.Cl * f.μl[i]
        Si  = f.Cl * f.Σl[i] * f.Cl' .+ f.R2
        Ki  = f.Σl[i] * f.Cl' * inv(Si)
        f.μl[i] .= f.μl[i] .+ Ki * r_i
        f.Σl[i]  = (I - Ki * f.Cl) * f.Σl[i]
    end
    return f
end

step!(f::MUKFState, y) = (predict!(f); correct!(f, y))

function filter!(f::MUKFState, yseq)
    out = Vector{Vector{Float64}}(undef, length(yseq))
    for t in eachindex(yseq)
        step!(f, yseq[t])
        out[t] = copy(f.μn)
    end
    return out
end

# collapse xˡ mixture mean across sigma-conditioned linear KFs
function xl_mean(f::MUKFState)
    sp = sigma_points(f.μn, f.Σn, f.ut)
    μ = zeros(f.nxl)
    @inbounds for i in 1:length(f.μl)
        μ .+= sp.Wm[i] .* f.μl[i]
    end
    return μ
end

end # module

using Test
using LinearAlgebra
using LowLevelParticleFilters
using .MUKF
filter! = MUKF.filter!

# --- Recreate the RBPF tutorial system exactly ---
nxn, nxl, ny, nu = 1, 3, 2, 0
fn(xn, u, p, t) = atan.(xn)
g(xn, u, p, t)  = [0.1 * xn[]^2 * sign(xn[]), 0.0]
An = [1.0 0.0 0.0]
Al = [ 1.0  0.3   0.0;
       0.0  0.92 -0.3;
       0.0  0.3   0.92 ]
Cl = [0.0 0.0 0.0;
      1.0 -1.0 1.0]
R1n = [0.01;;]
R1l = 0.01I(nxl)
R2  = 0.1I(ny)
x0n = zeros(nxn); R0n = [1.0;;]
x0l = zeros(nxl); R0l = 0.01I(nxl)

d0n = LowLevelParticleFilters.SimpleMvNormal(x0n, R0n)
d0l = LowLevelParticleFilters.SimpleMvNormal(x0l, R0l)

# Use package RBPF to generate consistent data
kf_lin = KalmanFilter(Al, zeros(nxl,nu), Cl, 0, R1l, R2, d0l; ny, nu)
mm     = RBMeasurementModel(g, R2, ny)

names = SignalNames(; x=["xnl", "xl1", "xl2", "xl3"], u=[], y=["y1", "y2"], name="RBPF_tutorial")
rbpf   = RBPF(300, kf_lin, fn, mm, R1n, d0n; nu, An, Ts=1.0, names)

T = 150
u_data = [zeros(nu) for _ in 1:T]
x_true, _, y_meas = simulate(rbpf, u_data)

# --- Run MUKF ---
state = init_mukf(fn=fn, g=g, An=An, Al=Al, Cl=Cl, R1n=R1n, R1l=R1l, R2=R2, d0n=d0n, d0l=d0l)
μn_hist = filter!(state, y_meas)

# Helpers
xn_true = [x_true[t][1] for t in 1:T]
xn_est  = [μ[1] for μ in μn_hist]
rmse(v1, v2) = sqrt(sum((v1 .- v2).^2) / length(v1))

@testset "MUKF sanity" begin
    @test length(μn_hist) == T
    @test isposdef(state.Σn + state.Σn') # symmetric positive semi-def check (approx)
    @test rmse(xn_true, xn_est) < 0.6    # loose upper bound; adjust as needed
end

## `examples/mukf_tutorial.jl`


using LinearAlgebra
using LowLevelParticleFilters
using Statistics
using Printf
# Optional plotting (comment out if you avoid deps in CI)
try
    using Σlots
catch err
    @warn "Σlots.jl not found; example will run without figures" err
end

# --- Model (same as RBPF tutorial) ---
fn(xn, u, p, t) = atan.(xn)
g(xn, u, p, t)  = [0.1 * xn[]^2 * sign(xn[]), 0.0]
An = [1.0 0.0 0.0]
Al = [ 1.0  0.3   0.0;
       0.0  0.92 -0.3;
       0.0  0.3   0.92 ]
Cl = [0.0 0.0 0.0;
      1.0 -1.0 1.0]
R1n = [0.01;;]
R1l = 0.01I(3)
R2  = 0.1I(2)

x0n = zeros(1); R0n = [1.0;;]
x0l = zeros(3); R0l = 0.01I(3)

d0n = LowLevelParticleFilters.SimpleMvNormal(x0n, R0n)
d0l = LowLevelParticleFilters.SimpleMvNormal(x0l, R0l)

# Use RBPF simulate for data parity
kf_lin = KalmanFilter(Al, zeros(3,0), Cl, 0, R1l, R2, d0l; ny=2, nu=0)
mm     = RBMeasurementModel(g, R2, 2)
rbpf   = RBPF(300, kf_lin, fn, mm, R1n, d0n; nu=0, An=An, Ts=1.0, names=SignalNames(; x=["xnl", "xl1", "xl2", "xl3"], u=[], y=["y1", "y2"], name="RBPF_tutorial"))

T = 150
u_data = [zeros(0) for _ in 1:T]
x_true, _, y_meas = simulate(rbpf, u_data)

# --- Run MUKF ---
state = init_mukf(fn=fn, g=g, An=An, Al=Al, Cl=Cl, R1n=R1n, R1l=R1l, R2=R2, d0n=d0n, d0l=d0l)
μn_hist = filter!(state, y_meas)

# Collect arrays for plotting
xn_true = [x_true[t][1] for t in 1:T]
xn_est  = [μ[1] for μ in μn_hist]
RMSE = sqrt(mean((xn_true .- xn_est).^2))
@printf("MUKF RMSE on xⁿ: %.4f\n", RMSE)

using Plots
t = 1:T
plt = plot(t, xn_true, label="true xⁿ", lw=2)
plot!(plt, t, xn_est, label="MUKF xⁿ est", lw=2, ls=:dash)
xlabel!(plt, "time step")
ylabel!(plt, "xⁿ")
title!(plt, @sprintf("MUKF on RBPF tutorial system (RMSE=%.3f)", RMSE))
display(plt)
