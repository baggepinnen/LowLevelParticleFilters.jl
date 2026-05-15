# Regression test: get_mat for a 3-D time-varying matrix should
# (a) work for integer-valued time
# (b) error with a clear message for non-integer time, instead of
#     bubbling up a confusing index error from getindex.

using LowLevelParticleFilters, LinearAlgebra, Test
import LowLevelParticleFilters: SimpleMvNormal, get_mat

@testset "get_mat 3-D array time arg" begin
    T = 10
    A = randn(2, 2, T)

    # Integer time works.
    @test get_mat(A, nothing, nothing, nothing, 0) == A[:, :, 1]
    @test get_mat(A, nothing, nothing, nothing, 3) == A[:, :, 4]
    # Real but integer-valued time works too (most call sites pass Float64).
    @test get_mat(A, nothing, nothing, nothing, 0.0) == A[:, :, 1]
    @test get_mat(A, nothing, nothing, nothing, 3.0) == A[:, :, 4]

    # Non-integer time must produce a clear ArgumentError rather than
    # whatever obscure failure `A[:,:,0.1+1]` would have raised.
    err = try
        get_mat(A, nothing, nothing, nothing, 0.1)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("3-D", err.msg)
    @test occursin("function", err.msg)
end

@testset "predict! with 3-D A and non-unit Ts errors clearly" begin
    nx, nu, ny = 2, 1, 1
    Tlen = 8
    A_seq = randn(nx, nx, Tlen)
    B = reshape([0.0, 1.0], nx, nu)
    C = [1.0 0.0]
    R1 = 0.01 * I(nx)
    R2 = 0.1 * I(ny)
    d0 = SimpleMvNormal(zeros(nx), Matrix(1.0I, nx, nx))

    # Ts=1.0 with an integer-step index works (regression: previously
    # also worked, but now the new method dispatch must not regress it).
    kf_int = KalmanFilter(A_seq, B, C, 0, R1, R2, d0; Ts = 1.0)
    @test_nowarn predict!(kf_int, [0.0])

    # Ts=0.1 makes the second predict!'s time argument 0.1, which is
    # not integer-valued and so must error.
    kf_frac = KalmanFilter(A_seq, B, C, 0, R1, R2, d0; Ts = 0.1)
    @test_nowarn predict!(kf_frac, [0.0])  # first call has t = 0.0, ok
    @test_throws ArgumentError predict!(kf_frac, [0.0])  # t = 0.1, not ok
end
