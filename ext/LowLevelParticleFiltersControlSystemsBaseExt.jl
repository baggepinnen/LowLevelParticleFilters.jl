module LowLevelParticleFiltersControlSystemsBaseExt
import LowLevelParticleFilters.KalmanFilter
using LowLevelParticleFilters: AbstractExtendedKalmanFilter, AbstractUnscentedKalmanFilter
using ControlSystemsBase: AbstractStateSpace, ssdata
import ControlSystemsBase
using Distributions

"""
    KalmanFilter(sys::StateSpace{Discrete}, R1, R2, d0 = MvNormal(Matrix(R1)); kwargs...)

Construct a `KalmanFilter` from a predefined `StateSpace` system from ControlSystems.jl
"""
function KalmanFilter(sys::AbstractStateSpace{<:ControlSystemsBase.Discrete}, R1, R2, d0=MvNormal(Matrix(R1)); kwargs...)
    A, B, C, D = ssdata(sys)
    KalmanFilter(A, B, C, D, Matrix(R1), Matrix(R2), d0; kwargs...)
end

function ControlSystemsBase.linearize(kf::Union{AbstractExtendedKalmanFilter, AbstractUnscentedKalmanFilter}, x::AbstractVector, u::AbstractVector, p, t)
    A,B = ControlSystemsBase.linearize(kf.dynamics, x, u, p, t)
    C,D = ControlSystemsBase.linearize(kf.measurement, x, u, p, t)
    (; A, B, C, D)
end

end