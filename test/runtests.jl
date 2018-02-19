using LowLevelParticleFilters
import LowLevelParticleFilters.resample
using Base.Test


@testset "weigthed_mean" begin
x = [randn(3) for i = 1:10000]
w = ones(10000)
logsumexp!(w)
@test sum(abs, weigthed_mean(x,w)) < 0.05
end

# @inline logsumexp!(w) = w .-= log(sum(exp, w))

@testset "logsumexp" begin
w = randn(10)
logsumexp!(w)
wc = copy(w)
I = ones(10)
logsumexp!(I)
@test w ≈ wc.-log(sum(exp, wc))
@test I ≈ fill(log(1/10),10)
end



@testset "resample" begin
s = PFstate([zeros(10)],[zeros(10)],ones(10),zeros(Int,10),zeros(10), Ref(1))
logsumexp!(s.w)
@test resample(s.w) ≈ 1:10

w = [1.,1,1,2,2,2,3,3,3]
logsumexp!(w)
@test w |> resample |> sum >= 56
@test length(resample(w)) == length(w)
for i = 1:10000
    w = randn(100)
    logsumexp!(w)
    j = resample(w)
    @test maximum(j) <= 100
    @test minimum(j) >= 1
end
end


# testpf = ParticleFilter(100, p0, linear_gaussian_f, linear_gaussian_g)
# @btime resample(testpf.state)
# w = randn(1000)
# @btime LowLevelParticleFilters.logsumexp!($w) # ≈ 15 μs
# @code_warntype LowLevelParticleFilters.logsumexp!(w)
# @code_warntype LowLevelParticleFilters.resample(w, zeros(Int,length(w)), zeros(length(w)))
