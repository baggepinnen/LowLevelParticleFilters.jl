using Test, LowLevelParticleFilters, ForwardDiff, StaticArrays, LinearAlgebra

@testset "Interval output discretization" begin
    # The physical derivative is u+t and both output channels observe the state.
    for vec in (collect, SVector{3})
        Ts = 0.4
        f(x,u,p,t) = p .* x .+ (u+t)
        g(x,u,p,t) = vcat(x, x/Ts)
        step = interval_output_rk4(f,g,1,Ts; supersample=3)
        z = vec([1.2, NaN, -19.0])
        t, u = 2.0, -0.3
        expected(x,t) = x*Ts + (u+t)*Ts^2/2 + Ts^3/6
        next = step(z,u,0.0,t)
        @test next ≈ [z[1]+(u+t)*Ts+Ts^2/2, expected(z[1],t), expected(z[1],t)/Ts]
        @test step(next,u,0.0,t+Ts)[2] ≈ expected(next[1],t+Ts)
        J = ForwardDiff.jacobian(v -> step(v,u,0.0,t), z)
        @test J ≈ [1 0 0; Ts 0 0; 1 0 0]
        vec === SVector{3} && @test next isa SVector{3}
    end
    # Under constant velocity, the interval average is position plus half the displacement.
    step = interval_output_rk4((x,u,p,t) -> [x[2],0.0], (x,u,p,t) -> [x[1]/2], 2, 2.0)
    prior = LowLevelParticleFilters.SimpleMvNormal([1.,0.5,0.], Matrix(1.0I,3,3))
    ekf = ExtendedKalmanFilter(step, (z,u,p,t) -> [z[end]], zeros(3,3), [0.2;;], prior; nu=0)
    predict!(ekf, Float64[])
    @test ekf.R[1:2,3] ≈ [3.,1.]
    ret = correct!(ekf, Float64[], [2.])
    @test ekf.x[1:2] ≈ [2.,0.5] + (0.5/2.2)*[3.,1.]
    @test ret.S ≈ [2.2;;]
    @test_throws ArgumentError interval_output_rk4(identity, identity, 1, 0)
    @test_throws DimensionMismatch interval_output_rk4((x,u,p,t)->x,(x,u,p,t)->x,1,1.0)(ones(3),0,0,0)
end
