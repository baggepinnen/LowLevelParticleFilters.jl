"""
    IndexingMatrix{N, VT} <: AbstractMatrix{Bool}

A sparse representation of a matrix containing only 0s and 1s, where each row contains exactly one 1.
This is stored efficiently as a vector of column indices.

# Type Parameters
- `N::Int`: Number of columns in the matrix
- `VT`: Type of the indices vector

# Fields
- `indices::VT`: Column indices where the 1s appear for each row

# Example
```julia
# Represents the matrix:
# [0 1 0
#  0 0 1
#  1 0 0]
I = IndexingMatrix([2, 3, 1], 3)
```
"""
struct IndexingMatrix{N, VT <: AbstractVector{Int}} <: AbstractMatrix{Bool}
    indices::VT
    
    function IndexingMatrix{N}(indices::AbstractVector{Int}) where {N}
        if any(i -> i < 1 || i > N, indices)
            throw(ArgumentError("Indices must be between 1 and $N"))
        end
        new{N, typeof(indices)}(indices)
    end
end

# Convenience constructor that infers N
IndexingMatrix(indices::AbstractVector{Int}, n::Int) = IndexingMatrix{n}(indices)

# AbstractArray interface
ncols(::IndexingMatrix{N}) where {N} = N
Base.size(I::IndexingMatrix{N}) where {N} = (length(I.indices), N)
Base.getindex(I::IndexingMatrix, i::Int, j::Int) = I.indices[i] == j

# Pretty printing
Base.show(io::IO, ::MIME"text/plain", I::IndexingMatrix) = 
    print(io, "$(size(I, 1))×$(ncols(I)) IndexingMatrix with indices: $(I.indices)")

"""
    *(I::IndexingMatrix, v::AbstractVector)

Multiply an IndexingMatrix by a vector, which simply selects elements from v.
"""
Base.@propagate_inbounds function Base.:*(I::IndexingMatrix, v::AbstractVector)
    @boundscheck ncols(I) == length(v) || throw(DimensionMismatch("Matrix columns $(ncols(I)) != vector length $(length(v))"))
    return v[I.indices]
end

"""
    *(I::IndexingMatrix, M::AbstractMatrix)

Multiply an IndexingMatrix by a matrix from the left, selecting rows from M.
"""
Base.@propagate_inbounds function Base.:*(I::IndexingMatrix, M::AbstractMatrix)
    @boundscheck ncols(I) == size(M, 1) || throw(DimensionMismatch("Matrix dimensions incompatible"))
    return M[I.indices, :]
end

"""
    *(M::AbstractMatrix, I::IndexingMatrix)

Multiply a matrix by an IndexingMatrix from the right, creating a permuted/duplicated column matrix.
"""
Base.@propagate_inbounds function Base.:*(M::AbstractMatrix, I::IndexingMatrix)
    @boundscheck size(M, 2) == size(I, 1) || throw(DimensionMismatch("Matrix dimensions incompatible"))
    result = zeros(eltype(M), size(M, 1), ncols(I))
    for (i, idx) in enumerate(I.indices)
        result[:, idx] .+= @view M[:, i]
    end
    return result
end

@generated function Base.:*(M::SMatrix{R, C1, MT}, I::IndexingMatrix{C, <:SVector{C1}}) where {R, C1, MT, C}
    L = R*C
    quote
        result = zero(MMatrix{$R, $C, $MT, $L})
        # for (i, idx) in enumerate(I.indices)
        #     result[:, idx] .+= M[:, i]
        # end
        Base.Cartesian.@nexprs $R i -> result[:, I.indices[i]] .+= M[:, i]
        return SMatrix{$R, $C, $MT, $L}(result)
    end
end

Base.@propagate_inbounds function Base.:*(A::AbstractMatrix, It::Adjoint{<:Any, <:IndexingMatrix})
    @boundscheck size(A, 2) == ncols(It.parent) || throw(DimensionMismatch("Matrix columns $(size(A, 2)) != width of indexing matrix $(ncols(It.parent))"))
    A[:, It.parent.indices]
end

Base.@propagate_inbounds function Base.:*(I::IndexingMatrix, A::AbstractMatrix, It::Adjoint{<:Any, <:IndexingMatrix})
    @boundscheck size(A, 2) == ncols(It.parent) || throw(DimensionMismatch("Matrix columns $(size(A, 2)) != vector length $(ncols(It.parent))"))
    @boundscheck ncols(I) == size(A, 1) || throw(DimensionMismatch("LHS matrix cols $(ncols(I)) != RHS matrix rows $(size(A, 1))"))
    A[I.indices, It.parent.indices]
end

# """
#     *(v::AbstractVector', I::IndexingMatrix)

# Multiply a row vector by an IndexingMatrix from the right.
# """
# function Base.:*(v::Adjoint{<:Any, <:AbstractVector}, I::IndexingMatrix)
#     vec = parent(v)
#     size(vec, 1) == size(I, 1) || throw(DimensionMismatch("Vector length $(length(vec)) != matrix rows $(size(I, 1))"))
#     result = zeros(eltype(vec), ncols(I))
#     for (i, idx) in enumerate(I.indices)
#         result[idx] += vec[i]
#     end
#     return result'
# end

"""
    is_indexing_matrix(M::AbstractMatrix)

Check if a matrix is a valid indexing matrix (contains only 0s and 1s with exactly one 1 per row).
"""
function is_indexing_matrix(M::AbstractMatrix)
    for i in 1:size(M, 1)
        found = false
        for j in 1:size(M, 2)
            if M[i, j] == true
                if found
                    return false
                else
                    found = true
                end
            end
        end
        found || return false
    end
    return true
end

"""
    IndexingMatrix(M::AbstractMatrix)

Convert a valid indexing matrix to an IndexingMatrix type.
Throws an error if M is not a valid indexing matrix.
"""
function IndexingMatrix(M::AbstractMatrix)
    if !is_indexing_matrix(M)
        throw(ArgumentError("Matrix is not a valid indexing matrix"))
    end
    
    m, n = size(M)
    indices = Vector{Int}(undef, m)
    
    for i in 1:m
        for j in 1:n
            if M[i, j] != 0
                indices[i] = j
                break
            end
        end
    end
    
    return IndexingMatrix{n}(indices)
end



# Export the main functions
export IndexingMatrix, is_indexing_matrix