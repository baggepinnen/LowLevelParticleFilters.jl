using LowLevelParticleFilters
using LowLevelParticleFilters: SimpleMvNormal
using Test
using LinearAlgebra
using Random
using Statistics
using Distributions
using Plots


@testset "Constraint Handling" begin
    
    @testset "project_bound" begin
        @testset "Basic functionality" begin
            # Test mean projection to lower bound
            μ = [1.0, -2.0, 3.0]
            Σ = [2.0 0.5 0.1;
                 0.5 1.0 0.2;
                 0.1 0.2 0.5]
            
            # Project x[2] to lower bound of 0
            μ_proj, Σ_proj = LowLevelParticleFilters.project_bound(μ, Σ, 2; lower=0.0)
            
            @test μ_proj[2] ≈ 0.0  # Mean should be at bound
            @test issymmetric(Σ_proj)  # Covariance should remain symmetric
            @test isposdef(Σ_proj)  # Covariance should remain positive definite
            @test Σ_proj[2,2] < Σ[2,2]  # Variance should decrease
            @test Σ_proj[1,2] != Σ[1,2]  # Correlation should change due to projection
        end
        
        @testset "Upper bound projection" begin
            μ = [1.0, 5.0, 3.0]
            Σ = [2.0 0.5 0.1;
                 0.5 1.0 0.2;
                 0.1 0.2 0.5]
            
            # Project x[2] to upper bound of 2
            μ_proj, Σ_proj = LowLevelParticleFilters.project_bound(μ, Σ, 2; upper=2.0)
            
            @test μ_proj[2] ≈ 2.0  # Mean should be at bound
            @test issymmetric(Σ_proj)
            @test isposdef(Σ_proj)
        end
        
        @testset "Two-sided bounds" begin
            μ = [1.0, -2.0, 3.0]
            Σ = [2.0 0.5 0.1;
                 0.5 1.0 0.2;
                 0.1 0.2 0.5]
            
            # Project x[2] to interval [0, 1]
            μ_proj, Σ_proj = LowLevelParticleFilters.project_bound(μ, Σ, 2; lower=0.0, upper=1.0)
            
            @test μ_proj[2] ≈ 0.0  # Should project to nearest bound (lower)
            @test issymmetric(Σ_proj)
            @test isposdef(Σ_proj)
        end
        
        @testset "Already feasible" begin
            μ = [1.0, 0.5, 3.0]
            Σ = [2.0 0.5 0.1;
                 0.5 1.0 0.2;
                 0.1 0.2 0.5]
            
            # x[2] is already in [0, 1]
            μ_proj, Σ_proj = LowLevelParticleFilters.project_bound(μ, Σ, 2; lower=0.0, upper=1.0)
            
            @test μ_proj ≈ μ  # Should remain unchanged
            @test Σ_proj ≈ Σ  # Should remain unchanged
        end
        
        @testset "Degenerate covariance" begin
            μ = [1.0, -2.0, 3.0]
            Σ = [2.0 0.5 0.1;
                 0.5 0.0 0.2;  # Zero variance for x[2]
                 0.1 0.2 0.5]
            
            # Should handle zero variance gracefully
            μ_proj, Σ_proj = LowLevelParticleFilters.project_bound(μ, Σ, 2; lower=0.0)
            
            @test μ_proj[2] ≈ 0.0  # Mean should be clamped
            @test Σ_proj ≈ Σ  # Covariance unchanged when variance is zero
        end
        
        @testset "Correlation preservation" begin
            # Test that conditional distributions are preserved
            μ = [0.0, -3.0]
            Σ = [1.0 0.8;
                 0.8 1.0]  # High correlation
            
            μ_proj, Σ_proj = LowLevelParticleFilters.project_bound(μ, Σ, 2; lower=0.0)
            
            # After projection, x[1] should be updated due to correlation
            @test μ_proj[1] > μ[1]  # x[1] should increase due to positive correlation
            @test μ_proj[2] ≈ 0.0
            @test Σ_proj[1,2] / sqrt(Σ_proj[1,1] * Σ_proj[2,2]) < 0.8  # Correlation should decrease

            covplot(μ, Σ, lab="Original")
            covplot!(μ_proj, Σ_proj, lab="Projected")
        end
    end
    
    @testset "truncated_moment_match" begin
        @testset "Basic functionality" begin
            μ = [1.0, -2.0, 3.0]
            Σ = [2.0 0.5 0.1;
                 0.5 1.0 0.2;
                 0.1 0.2 0.5]
            
            # Truncate x[2] to [0, Inf)
            μ_tmm, Σ_tmm = LowLevelParticleFilters.truncated_moment_match(μ, Σ, 2; lower=0.0)
            
            @test μ_tmm[2] > 0.0  # Mean should be positive
            @test μ_tmm[2] > μ[2]  # Mean should increase
            @test issymmetric(Σ_tmm)
            @test isposdef(Σ_tmm)
            @test Σ_tmm[2,2] < Σ[2,2]  # Variance should decrease due to truncation

            covplot(μ, Σ, mean=true, label="Original")
            covplot!(μ_tmm, Σ_tmm, mean=true, label="Projected")
        end
        
        @testset "Monte Carlo verification - univariate" begin
            # Test that the moments match empirical moments from sampling
            μ = 2.0
            σ = 1.5
            lower = 0.0
            upper = 5.0
            
            # Get truncated moments using the function
            μ_vec = [μ]
            Σ_mat = reshape([σ^2], 1, 1)
            μ_tmm, Σ_tmm = LowLevelParticleFilters.truncated_moment_match(μ_vec, Σ_mat, 1; lower, upper)
            
            # Generate samples from truncated normal
            d = Truncated(Normal(μ, σ), lower, upper)
            samples = rand(d, 100000)
            
            # Compare moments
            @test μ_tmm[1] ≈ mean(samples) rtol=0.01
            @test Σ_tmm[1,1] ≈ var(samples) rtol=0.02
        end
        
        @testset "Monte Carlo verification - multivariate" begin
            # Test preservation of conditional distributions
            μ = [1.0, -1.0]
            Σ = [2.0 1.0;
                 1.0 1.5]
            
            # Truncate x[2] to [0, Inf)
            μ_tmm, Σ_tmm = LowLevelParticleFilters.truncated_moment_match(μ, Σ, 2; lower=0.0)
            
            # Generate samples to verify
            # We sample from the joint distribution and then condition
            n_samples = 100000
            d = MvNormal(μ, Σ)
            samples = rand(d, n_samples)
            
            # Keep only samples where x[2] >= 0
            valid_samples = samples[:, samples[2,:] .>= 0]
            
            # Compare moments
            emp_mean = mean(valid_samples, dims=2)
            emp_cov = cov(valid_samples')
            
            @test μ_tmm ≈ vec(emp_mean) rtol=0.02
            @test Σ_tmm ≈ emp_cov rtol=0.05
        end
        
        @testset "One-sided lower truncation" begin
            μ = [0.0, -2.0, 1.0]
            Σ = Diagonal([1.0, 2.0, 0.5])
            
            # Truncate x[2] to [0, Inf)
            μ_tmm, Σ_tmm = LowLevelParticleFilters.truncated_moment_match(μ, Σ, 2; lower=0.0)
            
            @test μ_tmm[2] > 0.0
            @test μ_tmm[2] > μ[2]
            @test Σ_tmm[2,2] < Σ[2,2]
            # Other components should be unchanged (no correlation)
            @test μ_tmm[1] ≈ μ[1]
            @test μ_tmm[3] ≈ μ[3]
            @test Σ_tmm[1,1] ≈ Σ[1,1]
            @test Σ_tmm[3,3] ≈ Σ[3,3]

            covplot(μ, Σ, mean=true, label="Original")
            covplot!(μ_tmm, Σ_tmm, mean=true, label="Projected")
        end
        
        @testset "One-sided upper truncation" begin
            μ = [0.0, 3.0, 1.0]
            Σ = Diagonal([1.0, 2.0, 0.5])
            
            # Truncate x[2] to (-Inf, 1]
            μ_tmm, Σ_tmm = LowLevelParticleFilters.truncated_moment_match(μ, Σ, 2; upper=1.0)
            
            @test μ_tmm[2] < 1.0
            @test μ_tmm[2] < μ[2]
            @test Σ_tmm[2,2] < Σ[2,2]
        end
        
        @testset "Two-sided truncation" begin
            μ = [0.0, 0.0, 1.0]
            Σ = Diagonal([1.0, 4.0, 0.5])
            
            # Truncate x[2] to [-1, 1]
            μ_tmm, Σ_tmm = LowLevelParticleFilters.truncated_moment_match(μ, Σ, 2; lower=-1.0, upper=1.0)
            
            @test -1.0 <= μ_tmm[2] <= 1.0
            @test μ_tmm[2] ≈ 0.0 atol=0.1  # Should stay near 0 due to symmetry
            @test Σ_tmm[2,2] < Σ[2,2]  # Variance should decrease

            covplot(μ, Σ, mean=true, label="Original")
            covplot!(μ_tmm, Σ_tmm, mean=true, label="Projected")
        end
        
        @testset "Extreme truncation" begin
            # Test when truncation removes most of the probability mass
            μ = [0.0, 0.0]
            Σ = [1.0 0.0;
                 0.0 1.0]
            
            # Truncate far in the tail
            μ_tmm, Σ_tmm = LowLevelParticleFilters.truncated_moment_match(μ, Σ, 2; lower=50.0)
            
            # Should fall back to projection when mass is too small
            @test μ_tmm[2] ≈ 50.0  # Should project to bound
            @test Σ_tmm[2,2] ≈ 0.0 atol=1e-6  # Variance should collapse
        end
        
        @testset "Correlation handling" begin
            # Test that correlations are properly updated
            μ = [0.0, 0.0]
            Σ = [1.0 0.7;
                 0.7 1.0]
            
            # Truncate x[2] to [1, Inf)
            μ_tmm, Σ_tmm = LowLevelParticleFilters.truncated_moment_match(μ, Σ, 2; lower=1.0)
            
            # x[1] should increase due to positive correlation
            @test μ_tmm[1] > μ[1]
            @test μ_tmm[2] > 1.0
            # Correlation should still exist but might change
            @test Σ_tmm[1,2] > 0
        end
    end
    
    @testset "Integration with UKF" begin
        # Test using the functions within an actual filter
        nx = 2
        ny = 1
        nu = 1
        
        # Simple linear system with one state that should stay positive
        A = [0.9 0.1; 0.0 0.8]
        B = [1.0; 0.5;;]
        C = [1.0 0.0]
        
        dynamics(x,u,p,t) = A*x + B*u
        measurement(x,u,p,t) = C*x
        
        R1 = Diagonal([0.1, 0.1])
        R2 = Diagonal([0.01])
        
        x0 = [1.0, 0.5]
        P0 = Diagonal([0.1, 0.1])
        
        ukf = UnscentedKalmanFilter(dynamics, measurement, R1, R2, SimpleMvNormal(x0, P0); ny, nu)
        
        # Test with project_bound callback
        function project_callback(kf, u, y, p, ll, e)
            if kf.x[2] < 0
                kf.x, kf.R = LowLevelParticleFilters.project_bound(kf.x, kf.R, 2; lower=0.0)
            end
            nothing
        end
        
        # Test with truncated_moment_match callback
        function tmm_callback(kf, u, y, p, ll, e)
            if kf.x[2] < 0
                kf.x, kf.R = LowLevelParticleFilters.truncated_moment_match(kf.x, kf.R, 2; lower=0.0)
            end
            nothing
        end
        
        # Generate some test data
        T = 50
        u_test = [randn(nu) for _ in 1:T]
        y_test = [measurement(x0, u_test[1], 0, 0) + 0.1*randn(ny) for _ in 1:T]
        
        # Run filter with project_bound
        ukf_proj = deepcopy(ukf)
        for t in 1:T
            predict!(ukf_proj, u_test[t])
            correct!(ukf_proj, u_test[t], y_test[t])
            project_callback(ukf_proj, u_test[t], y_test[t], 0, 0, 0)
        end
        
        # Run filter with truncated_moment_match
        ukf_tmm = deepcopy(ukf)
        for t in 1:T
            predict!(ukf_tmm, u_test[t])
            correct!(ukf_tmm, u_test[t], y_test[t])
            tmm_callback(ukf_tmm, u_test[t], y_test[t], 0, 0, 0)
        end
        
        # Both should maintain positive second state
        @test ukf_proj.x[2] >= 0
        @test ukf_tmm.x[2] >= 0
        
        # Both should maintain valid covariances
        @test isposdef(ukf_proj.R)
        @test isposdef(ukf_tmm.R)
    end
    
    @testset "Numerical stability" begin
        # Test with poorly conditioned matrices
        μ = [1.0, -1.0, 0.0]
        Σ = [1e-8  0    0;
             0     1.0  0.99;
             0     0.99 1.0]  # Nearly singular
        
        # Should handle without errors
        μ_proj, Σ_proj = LowLevelParticleFilters.project_bound(μ, Σ, 2; lower=0.0)
        @test !any(isnan, μ_proj)
        @test !any(isnan, Σ_proj)
        
        μ_tmm, Σ_tmm = LowLevelParticleFilters.truncated_moment_match(μ, Σ, 2; lower=0.0)
        @test !any(isnan, μ_tmm)
        @test !any(isnan, Σ_tmm)
    end
end