# Regression test for SqKalmanFilter's @kwdef-generated keyword
# constructor. The previous default for `names` had a stray `names=`
# in the middle and called `default_names(...; name="SqKF")` as a
# keyword — but `default_names` takes `name` as a positional argument,
# so the default value errored as soon as anyone constructed via the
# kwdef constructor without supplying `names`.

using LowLevelParticleFilters, LinearAlgebra, StaticArrays, Test
import LowLevelParticleFilters: SimpleMvNormal

@testset "SqKalmanFilter @kwdef defaults" begin
    nx, nu, ny = 2, 1, 1
    A = [0.9 0.0; 0.1 0.95]
    B = reshape([1.0, 0.0], nx, nu)
    C = [1.0 0.0]
    D = zeros(ny, nu)
    R1 = UpperTriangular(cholesky(0.01I(nx)).U)
    R2 = UpperTriangular(cholesky(0.1I(ny)).U)
    d0 = SimpleMvNormal(zeros(nx), Matrix(1.0I, nx, nx))
    x0 = zeros(nx)
    R0 = UpperTriangular(cholesky(Matrix(1.0I, nx, nx)).U)

    # Construct via the @kwdef-generated keyword constructor without
    # passing `names`. The default for `names` must not error.
    sqkf = @test_nowarn SqKalmanFilter(
        A = A, B = B, C = C, D = D,
        R1 = R1, R2 = R2, d0 = d0,
        x = x0, R = R0,
    )

    # Sanity: the produced filter has the expected default names.
    @test sqkf.names.name == "SqKF"
    @test length(sqkf.names.x) == nx
end
