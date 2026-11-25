using Test, LinearAlgebra, LowLevelParticleFilters

@testset "Exact correction for observed channels" begin
    A = [0.8 0.1; 0.0 0.9]
    C = [1. 2.; 0.3 1.; -0.2 0.5]
    D = [0.5; 0.1; 0.2;;]
    R2 = [0.8 0.1 0.2; 0.1 1.0 0.15; 0.2 0.15 0.6]
    R12 = [0.1 0.0 -0.05; 0.0 0.08 0.04]
    kf = KalmanFilter(A, zeros(2,1), C, D, [1. 0.; 0. 1.], R2)
    rows = [3,1]
    y = Union{Missing,Float64}[2., missing, -0.3]
    u = [0.5]
    model = LinearMeasurementModel((x,u,p,t)->C, D, R2; ny=3, R12)
    reduced = LinearMeasurementModel(C[rows,:], D[rows,:], R2[rows,rows]; R12=R12[:,rows])
    ref = deepcopy(kf)
    a = correct!(ref, reduced, u, Float64.(y[rows]))
    b = correct_observed!(kf, u, y; model, observed=rows)
    for s in (:e,:K,:S,:ll)
        @test getproperty(a,s) ≈ getproperty(b,s)
    end
    @test kf.x ≈ ref.x && kf.R ≈ ref.R
    @test correct_observed!(kf, u, fill(missing,3)).ll == 0
    @test kf.x == ref.x && kf.R == ref.R
    @test_throws ArgumentError correct_observed!(kf,u,y; observed=[1,1])
    @test_throws ArgumentError correct_observed!(kf,u,y; observed=[2])
    @test_throws DimensionMismatch correct_observed!(kf,u,y; observed=[true])

    h(x,u,p,t) = [x[1]^2+u[1],sin(x[2]),x[1]+x[2]]
    f(x,u,p,t) = A*x
    d0 = LowLevelParticleFilters.SimpleMvNormal([0.5,0.8], Matrix(1.0I,2,2))
    ekf = ExtendedKalmanFilter(f,h,Matrix(1.0I,2,2),R2,d0; nu=1,ny=3)
    k2 = deepcopy(ekf)
    hsub(x,u,p,t) = h(x,u,p,t)[[1,3]]
    small = EKFMeasurementModel{Float64,false}(hsub,R2[[1,3],[1,3]]; nx=2,ny=2)
    expected = correct!(k2,small,u,[2.,-0.3])
    actual = correct_observed!(ekf,u,y)
    hip!(out,x,u,p,t) = (out .= h(x,u,p,t))
    mip = EKFMeasurementModel{Float64,true}(hip!,R2; nx=2,ny=3)
    ipfilter = ExtendedKalmanFilter(f,h,Matrix(1.0I,2,2),R2,d0; nu=1,ny=3)
    ip = correct_observed!(ipfilter,u,[2.,NaN,-0.3]; model=mip, observed=[true,false,true])
    @test ip.ll ≈ expected.ll
    @test ipfilter.x ≈ k2.x && ipfilter.R ≈ k2.R
    full = deepcopy(kf)
    @test correct_observed!(full,u,ones(3)).ll ≈ correct!(deepcopy(kf),u,ones(3)).ll
    @test actual.ll ≈ expected.ll
    @test ekf.x ≈ k2.x && ekf.R ≈ k2.R
end
