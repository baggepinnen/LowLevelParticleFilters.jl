using Test
using LinearAlgebra
using LowLevelParticleFilters, StaticArrays
using LowLevelParticleFilters: ncols

@testset "IndexingMatrix Tests" begin
    
    @testset "Construction and Basic Properties" begin
        # Create a simple indexing matrix
        I = IndexingMatrix([2, 3, 1], 3)
        
        @test size(I) == (3, 3)
        @test I[1, 2] == true
        @test I[1, 1] == false
        @test I[2, 3] == true
        @test I[3, 1] == true
        
        # Test invalid construction
        @test_throws ArgumentError IndexingMatrix([1, 4, 2], 3)  # index out of bounds
        @test_throws ArgumentError IndexingMatrix([0, 1, 2], 3)  # index out of bounds
    end
    
    @testset "Vector Multiplication" begin
        I = IndexingMatrix([2, 3, 1], 3)
        v = [10, 20, 30]
        
        result = I * v
        @test result == [20, 30, 10]
        
        # Test dimension mismatch
        @test_throws DimensionMismatch I * [1, 2]
    end
    
    @testset "Matrix Multiplication (Left)" begin
        I = IndexingMatrix([2, 1], 3)  # 2×3 matrix selecting rows 2 and 1
        M = [1 2 3; 4 5 6; 7 8 9]
        
        result = I * M
        @test result == [4 5 6; 1 2 3]
    end
    
    @testset "Matrix Multiplication (Right)" begin
        I = IndexingMatrix([3, 1, 2], 3)  # 3×3 permutation matrix
        M = [1 2 3; 4 5 6]
        
        result = M * I
        @test result == M*Matrix(I)
        
        # Test with duplication (non-permutation case)
        I2 = IndexingMatrix([1, 1, 2], 3)  # Maps columns 1,2,3 to columns 1,1,2
        result2 = M * I2
        @test result2[:, 1] == M[:, 1] + M[:, 2]  # Columns 1 and 2 both map to column 1
        @test result2[:, 2] == M[:, 3]
    end
    
    # @testset "Row Vector Multiplication" begin
    #     I = IndexingMatrix([2, 3, 1], 3)
    #     v = [10, 20, 30]
        
    #     result = v' * I
    #     @test result == [30, 10, 20]'
    # end
    
    @testset "Validation Functions" begin
        # Valid indexing matrix
        M1 = [0 1 0; 0 0 1; 1 0 0]
        @test is_indexing_matrix(M1) == true
        
        # Invalid: multiple 1s in a row
        M2 = [1 1 0; 0 0 1; 1 0 0]
        @test is_indexing_matrix(M2) == false
        
        # Invalid: no 1 in a row
        M3 = [0 0 0; 0 0 1; 1 0 0]
        @test is_indexing_matrix(M3) == false
        
        # Invalid: contains values other than 0 and 1
        M4 = [0 2 0; 0 0 1; 1 0 0]
        @test is_indexing_matrix(M4) == false
        
        # Valid with Bool type
        M5 = Bool[0 1 0; 0 0 1; 1 0 0]
        @test is_indexing_matrix(M5) == true
    end
    
    @testset "Conversion Functions" begin
        M = [0 1 0; 0 0 1; 1 0 0]
        I = IndexingMatrix(M)
        
        @test I.indices == [2, 3, 1]
        @test ncols(I) == 3
        @test I isa IndexingMatrix{3}
        
        # Test invalid conversion
        M_invalid = [1 1 0; 0 0 1; 1 0 0]
        @test_throws ArgumentError IndexingMatrix(M_invalid)
    end


    @testset "static indexing matrices" begin
        inds = SA[1,2]
        I = IndexingMatrix(inds, 3)
        A = @SMatrix randn(3,3)
        yA = I*A
        @test yA isa SMatrix{2,3,Float64,6}
        @test yA == Matrix(I)*A
        yA = A*I'
        @test yA isa SMatrix{3,2,Float64,6}
        @test yA == A*Matrix(I)'
        yA = I*A*I'
        @test yA == Matrix(I)*A*Matrix(I)'
        @test yA isa SMatrix{2,2,Float64,4}


        inds = SA[3,2,1]
        I = IndexingMatrix(inds, 3)
        yA = A*I
        @test yA == A*Matrix(I)
        @test yA isa SMatrix{3,3,Float64,9}
    end
    
end
