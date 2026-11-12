using LowLevelParticleFilters, StaticArrays, LinearAlgebra, Test
import LowLevelParticleFilters as LLPF

# Filters constructed from Float32 arrays perform their computations in Float32
d0 = LLPF.SimpleMvNormal(SA[0f0, 0f0], SMatrix{2,2,Float32}(I))
A = SA[1f0 0.1f0; 0 1f0]; B = SA[0f0; 0.1f0;;]; C = SA[1f0 0f0]
R1 = SMatrix{2,2,Float32}(0.01I); R2 = SA[0.1f0;;]
dyn_f32(x, u, p, t) = A*x + B*u
meas_f32(x, u, p, t) = C*x
T = 10
u = [Float32[sin(k)] for k in 1:T]
y = [Float32[cos(k)] for k in 1:T]

filters = [
    "KalmanFilter"                 => () -> KalmanFilter(A, B, C, 0, R1, R2, d0),
    "KalmanFilter with α"          => () -> KalmanFilter(A, B, C, 0, R1, R2, d0; α = 1.01f0),
    "ExtendedKalmanFilter"         => () -> ExtendedKalmanFilter(dyn_f32, meas_f32, R1, R2, d0; nu = 1),
    "IteratedExtendedKalmanFilter" => () -> IteratedExtendedKalmanFilter(dyn_f32, meas_f32, R1, R2, d0; nu = 1),
    "UnscentedKalmanFilter"        => () -> UnscentedKalmanFilter(dyn_f32, meas_f32, R1, R2, d0; nu = 1, ny = 1),
    "SqKalmanFilter"               => () -> SqKalmanFilter(A, B, C, 0, R1, R2, d0),
    "SqExtendedKalmanFilter"       => () -> SqExtendedKalmanFilter(dyn_f32, meas_f32, R1, R2, d0; nu = 1),
]
for (name, mk) in filters
    @testset "$name" begin
        f = mk()
        predict!(f, u[1])
        r = correct!(f, u[1], y[1])
        @test r.ll isa Float32
        @test eltype(r.K) == Float32
        hasproperty(r, :S) && @test eltype(r.S) == Float32
        @test eltype(f.x) == Float32
        @test eltype(f.R) == Float32
        sol = forward_trajectory(mk(), u, y)
        @test sol.ll isa Float32
    end
end

@testset "IMM" begin
    imm = IMM([KalmanFilter(A, B, C, 0, R1, R2, d0), KalmanFilter(A, B, C, 0, 10R1, R2, d0)], Float32[0.9 0.1; 0.1 0.9], Float32[0.5, 0.5])
    sol = forward_trajectory(imm, u, y)
    @test sol.ll isa Float32
end

@testset "Helpers" begin
    @test eltype(LLPF.symmetrize(SA[1f0 2f0; 3f0 4f0])) == Float32
    @test eltype(LLPF.SimpleMvNormal(SMatrix{2,2,Float32}(I)).μ) == Float32
    @test eltype(LLPF.SimpleMvNormal(Matrix{Float32}(I, 2, 2)).μ) == Float32
    W = LLPF.UKFWeights(LLPF.MerweParams(), 2, Float32)
    @test W isa LLPF.UKFWeights{Float32}
    @test W.wm ≈ LLPF.UKFWeights(LLPF.MerweParams(), 2).wm
end
