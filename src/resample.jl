function shouldresample(pf::AbstractParticleFilter)
    we      = expweights(pf)
    N       = num_particles(pf)
    th      = 1/(N*pf.resample_threshold)
    initial = round(Int, 1/th)
    s       = zero(eltype(we))
    @inbounds @simd for i = 1:initial
        s += we[i]^2
    end
    for i = initial+1:N
        s += we[i]^2
        if s > th
            return true
        end
    end
    return false
end

resample(pf::AbstractParticleFilter, M::Int=num_particles(pf)) = resample(pf.resampling_strategy, pf.state.we, pf.state.j, pf.state.bins, M)
resample(T::Type{<:ResamplingStrategy}, s::PFstate, M::Int=num_particles(s)) = resample(T, s.we, s.j, s.bins, M)
resample(T::Type{<:ResamplingStrategy}, we::AbstractVector, M::Int=length(we)) = resample(T, we, zeros(Int,M), zeros(length(we)), M)
resample(we::AbstractArray) = resample(ResampleSystematic,we)

function resample(::Type{ResampleSystematic}, we, j, bins, M = length(we))
    N = length(we)
    bins[1] = we[1]
    for i = 2:N
        bins[i] = bins[i-1] + we[i]
    end
    r = rand()/N
    s = r:(1/M):(bins[N]+r) # Added r in the end to ensure correct length (r < 1/N)
    bo = 1
    for i = 1:M
        @inbounds for b = bo:N
            if s[i] < bins[b]
                j[i] = b
                bo = b
                break
            end
        end
    end
    return j
end


function resample(::Type{ResampleSystematicExp}, w, j, bins, M = length(w))
    N = length(exp(w))
    bins[1] = exp(w[1])
    for i = 2:N
        bins[i] = bins[i-1] + exp(w[i])
    end
    r = rand()/N
    s = r:(1/M):(bins[N]+r) # Added r in the end to ensure correct length (r < 1/N)
    bo = 1
    for i = 1:M
        @inbounds for b = bo:N
            if s[i] < bins[b]
                j[i] = b
                bo = b
                break
            end
        end
    end
    return j
end



# """
# There is probably lots of room for improvement here. All bins need not be formed in the beginning.
# One only has to keep 1 values, the current upper limit, no array needed.
# """
"""
    draw_one_categorical(pf,w)
Obs! This function expects log-weights
"""
function draw_one_categorical(pf,w)
    bins = pf.state.bins
    Yeppp.exp!(bins,w)
    for i = 2:length(w)
        bins[i] += bins[i-1]
    end
    s = rand()*bins[end]
    midpoint = length(bins)÷2
    if s < bins[midpoint]
        for b = 1:midpoint
            if s <= bins[b]
                return b
            end
        end
    else
        for b = midpoint:length(bins)
            if s <= bins[b]
                return b
            end
        end
    end
    length(bins)
end
