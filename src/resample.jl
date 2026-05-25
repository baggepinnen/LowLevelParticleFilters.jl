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

function resample(::Type{ResampleStratified}, we, j, bins, M = length(we))
    N = length(we)
    bins[1] = we[1]
    for i = 2:N
        bins[i] = bins[i-1] + we[i]
    end

    # u_i = (i - 1 + rand()) / M
    bo = 1

    for i = 1:M
        u = (i - 1 + rand()) / M * bins[N]  # scale to sum(weights)

        @inbounds for b = bo:N
            if u < bins[b]
                j[i] = b
                bo = b
                break
            end
        end
    end

    return j
end

function resample(::Type{ResampleResidual}, we, j, bins, M = length(we))
    N = length(we)

    wsum = zero(eltype(we))
    @inbounds for i = 1:N
        wsum += we[i]
    end

    inv_wsum = 1 / wsum

    num = 0

    @inbounds for i = 1:N
        nw = we[i] * inv_wsum * M
        cnt = floor(Int, nw)
        bins[i] = nw - cnt   # store residual weight

        for k = 1:cnt
            num += 1
            j[num] = i
        end
    end

    if num == M
        return j
    end

    rsum = zero(eltype(we))
    @inbounds for i = 1:N
        rsum += bins[i]
    end

    inv_rsum = 1 / rsum
    @inbounds for i = 1:N
        bins[i] *= inv_rsum
    end

    bins[1] = bins[1]
    @inbounds for i = 2:N
        bins[i] += bins[i-1]
    end

    @inbounds for m = (num + 1):M
        u = rand()

        for i = 1:N
            if u < bins[i]
                j[m] = i
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
