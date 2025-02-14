module LowLevelParticleFiltersControlSystemsBaseExt
import LowLevelParticleFilters.KalmanFilter
using LowLevelParticleFilters: AbstractFilter, AbstractKalmanFilter, AbstractExtendedKalmanFilter, AbstractUnscentedKalmanFilter, SimpleMvNormal, SignalNames
using ControlSystemsBase: AbstractStateSpace, ssdata
import ControlSystemsBase

"""
    KalmanFilter(sys::StateSpace{Discrete}, R1, R2, d0 = MvNormal(Matrix(R1)); kwargs...)

Construct a `KalmanFilter` from a predefined `StateSpace` system from ControlSystems.jl
"""
function KalmanFilter(sys::AbstractStateSpace{<:ControlSystemsBase.Discrete}, R1, R2, d0=SimpleMvNormal(Matrix(R1)); kwargs...)
    A, B, C, D = ssdata(sys)
    name = ControlSystemsBase.system_name(sys)
    x = ControlSystemsBase.state_names(sys)
    u = ControlSystemsBase.input_names(sys)
    y = ControlSystemsBase.output_names(sys)
    names = SignalNames(x, u, y, name)
    KalmanFilter(A, B, C, D, R1, R2, d0; names, kwargs...)
end

function ControlSystemsBase.linearize(kf::Union{AbstractExtendedKalmanFilter, AbstractUnscentedKalmanFilter}, x::AbstractVector, u::AbstractVector, p, t)
    A,B = ControlSystemsBase.linearize(kf.dynamics, x, u, p, t)
    C,D = ControlSystemsBase.linearize(kf.measurement, x, u, p, t)
    (; A, B, C, D)
end

ControlSystemsBase.state_names(f::AbstractKalmanFilter)  = f.names.x
ControlSystemsBase.input_names(f::AbstractKalmanFilter)  = f.names.u
ControlSystemsBase.output_names(f::AbstractKalmanFilter) = f.names.y
ControlSystemsBase.system_name(f::AbstractKalmanFilter)  = f.names.name

end