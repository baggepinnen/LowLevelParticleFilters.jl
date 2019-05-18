effective_particles(pf) = effective_particles(expweights(pf))
effective_particles(we::AbstractVector) = 1/sum(abs2, we)


function shouldresample(pf::AbstractParticleFilter)
    resample_threshold(pf) == 1 && (return true)
    th      = num_particles(pf)*resample_threshold(pf)
    ne = effective_particles(pf)
    return ne < th
end

resample(pf::AbstractParticleFilter, M::Int=num_particles(pf)) = resample(resampling_strategy(pf), expweights(pf), state(pf).j, state(pf).bins, M)
resample(T::Type{<:ResamplingStrategy}, s::PFstate, M::Int=num_particles(s)) = resample(T, s.we, s.j, s.bins, M)
resample(T::Type{<:ResamplingStrategy}, we::AbstractVector, M::Int=length(we)) = resample(T, we, zeros(Int,M), zeros(length(we)), M)
resample(we::AbstractArray) = resample(ResampleSystematic,we)

function resample(::Type{ResampleSystematic}, we, j, bins, M = length(we))
    N = length(we)
    bins[1] = we[1]
    for i = 2:N
        bins[i] = bins[i-1] + we[i]
    end
    r = rand()*bins[end]/N
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
    N = length(w)
    bins[1] = exp(w[1])
    for i = 2:N
        bins[i] = bins[i-1] + exp(w[i])
    end
    bins ./= bins[end]
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
    bins = state(pf).bins
    logsumexp!(w,bins)
    for i = 2:length(w)
        bins[i] += bins[i-1]
    end
    @assert bins[end] ≈ 1 "All expweigths 0"
    s = rand()*bins[end]
    # ind = findfirst(x->bins[x]>=s, 1:length(bins))
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
