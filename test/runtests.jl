using LowLevelParticleFilters
import LowLevelParticleFilters.resample
using Test, Random
Random.seed!(0)

@testset "logsumexp" begin
    w = randn(10); we = similar(w)
    logsumexp!(w,we)
    @test sum(we) ≈ 1
    @test sum(exp.(w)) ≈ 1
    wc = copy(w)
    I = ones(10)
    logsumexp!(I,we)
    @test w ≈ wc.-log(sum(exp, wc))
    @test I ≈ fill(log(1/10),10)
end

@testset "weigthed_mean" begin
    x = [randn(3) for i = 1:10000]
    w = ones(10000); we = similar(w)
    logsumexp!(w,we)
    @test sum(abs, weigthed_mean(x,we)) < 0.06
end

# @inline logsumexp!(w) = w .-= log(sum(exp, w))




@testset "resample" begin
    s = PFstate([zeros(10)],[zeros(10)],ones(10),ones(10),zeros(Int,10),zeros(10), Ref(1))
    logsumexp!(s.w, s.we)
    @test resample(s.we) ≈ 1:10

    w = [1.,1,1,2,2,2,3,3,3]; we=similar(w)
    logsumexp!(w,we)
    @test we |> resample |> sum >= 56
    @test length(resample(we)) == length(we)
    for i = 1:10000
        w = randn(100); we = randn(100)
        logsumexp!(w,we)
        j = resample(we)
        @test maximum(j) <= 100
        @test minimum(j) >= 1
    end
end
