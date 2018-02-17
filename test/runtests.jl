using LowLevelParticleFilters
using Base.Test


@testset "weigthed_mean" begin
x = [randn(3) for i = 1:10000]
w = ones(10000) |> logsumexp!
@test sum(abs, weigthed_mean(x,w)) < 0.05
end

# @inline logsumexp!(w) = w .-= log(sum(exp, w))

@testset "logsumexp" begin
w = randn(10)
wc = copy(w)
@test logsumexp!(w) ≈ wc.-log(sum(exp, wc))
@test logsumexp!(ones(10)) ≈ fill(log(1/10),10)
end



@testset "resample" begin
s = PFstate([zeros(10)],[zeros(10)],ones(10),zeros(Int,10),zeros(10))
w = logsumexp!(s.w)
@test resample(s) ≈ 1:10
@test [1.,1,1,2,2,2,3,3,3] |> logsumexp! |> resample |> sum >= 56
@test length(resample(w)) == length(w)
for i = 1:10000
    j = randn(100) |> logsumexp! |> resample
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
