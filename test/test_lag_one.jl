using Test, LinearAlgebra, LowLevelParticleFilters

@testset "Lag-one posterior covariance against joint Gaussian conditioning" begin
    A = [0.81 0.35; -0.16 0.72]
    B = zeros(2,1)
    C = [1. -0.4]
    Q = [0.20 0.02; 0.02 0.13]
    V = fill(0.3,1,1)
    P = [1.3 0.1; 0.1 0.6]
    prior = LowLevelParticleFilters.SimpleMvNormal(zeros(2), P)
    n = 4
    # Map independent initial and process disturbances to the stacked trajectory.
    M = zeros(2n,2n)
    blocks = [2i-1:2i for i in 1:n]
    for i in 1:n, j in 1:i
        M[blocks[i],blocks[j]] = A^(i-j)
    end
    U = M * cat(P,Q,Q,Q; dims=(1,2)) * M'
    H = kron(Matrix(I,n,n), C)
    post = U - U*H' * ((H*U*H' + V[1]*I) \ (H*U))
    f(x,u,p,t) = A*x
    h(x,u,p,t) = C*x
    inputs = fill([0.0], n)
    outputs = [[0.1i] for i in 1:n]
    for kf in (KalmanFilter(A,B,C,0,Q,V,prior),
               ExtendedKalmanFilter(f,h,Q,V,prior; nu=1),
               SqExtendedKalmanFilter(f,h,Q,V,prior; nu=1),
               UnscentedKalmanFilter(f,h,Q,V,prior; nu=1,ny=1))
        sol = forward_trajectory(kf, inputs, outputs)
        base = smooth(sol)
        joint = smooth(sol; lag_one=true)
        @test isnothing(base.extraT)
        @test base.xT == joint.xT && base.RT == joint.RT
        @test length(joint.extraT) == n-1
        for i in 1:n-1
            @test joint.extraT[i] ≈ post[blocks[i],blocks[i+1]]
        end
    end
    kf = KalmanFilter(A,B,C,0,Q,V,prior)
    @test isempty(smooth(kf, inputs[1:1], outputs[1:1]; lag_one=true).extraT)
end
