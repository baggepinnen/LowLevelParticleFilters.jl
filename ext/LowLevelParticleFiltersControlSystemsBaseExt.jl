module LowLevelParticleFiltersControlSystemsBaseExt
import LowLevelParticleFilters.KalmanFilter
using ControlSystemsBase: AbstractStateSpace, Discrete, ssdata
using Distributions

"""
    KalmanFilter(sys::StateSpace{Discrete}, R1, R2, d0 = MvNormal(Matrix(R1)); kwargs...)

Construct a `KalmanFilter` from a predefined `StateSpace` system from ControlSystems.jl
"""
function KalmanFilter(sys::AbstractStateSpace{<:Discrete}, R1, R2, d0=MvNormal(Matrix(R1)); kwargs...)
    A, B, C, D = ssdata(sys)
    kf = KalmanFilter(A, B, C, D, Matrix(R1), Matrix(R2), d0; kwargs...)
end

end