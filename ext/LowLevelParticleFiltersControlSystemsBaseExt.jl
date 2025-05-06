module LowLevelParticleFiltersControlSystemsBaseExt
import LowLevelParticleFilters.KalmanFilter
using LowLevelParticleFilters: AbstractFilter, AbstractKalmanFilter, AbstractExtendedKalmanFilter, AbstractUnscentedKalmanFilter, AbstractParticleFilter, SimpleMvNormal, SignalNames, get_mat
using ControlSystemsBase: AbstractStateSpace, ssdata, ss, observability, linearize
import ControlSystemsBase

"""
    KalmanFilter(sys::StateSpace{Discrete}, R1, R2, d0 = MvNormal(Matrix(R1)); kwargs...)

Construct a `KalmanFilter` from a predefined `StateSpace` system from ControlSystems.jl
"""
function KalmanFilter(sys::AbstractStateSpace{<:ControlSystemsBase.Discrete}, R1, R2, d0=SimpleMvNormal(R1); kwargs...)
    A, B, C, D = ssdata(sys)
    name = ControlSystemsBase.system_name(sys)
    x = ControlSystemsBase.state_names(sys)
    u = ControlSystemsBase.input_names(sys)
    y = ControlSystemsBase.output_names(sys)
    names = SignalNames(x, u, y, name)
    Ts = sys.Ts
    KalmanFilter(A, B, C, D, R1, R2, d0; names, Ts, kwargs...)
end

"""
    A, B, C, D = ControlSystemsBase.linearize(kf::AbstractKalmanFilter, x::AbstractVector, u::AbstractVector, p=kf.p, t = 0.0)

Linearize a nonlinear Kalman filter at the given state `x`, input `u`, and parameter `p` at time `t`. Returns the linearized system matrices `A`, `B`, `C`, and `D`. Call `ss(A, B, C, D, kf.Ts)` to get a `StateSpace` object.
"""
function ControlSystemsBase.linearize(kf::Union{AbstractParticleFilter,AbstractExtendedKalmanFilter, AbstractUnscentedKalmanFilter}, x::AbstractVector, u::AbstractVector, p=kf.p, t=0.0)
    A,B = linearize(kf.dynamics, x, u, p, t)
    C,D = linearize(kf.measurement, x, u, p, t)
    (; A, B, C, D)
end

function ControlSystemsBase.linearize(kf::AbstractKalmanFilter, x::AbstractVector, u::AbstractVector, p=kf.p, t=0.0)
    A = get_mat(kf.A, x, u, p, t) 
    B = get_mat(kf.B, x, u, p, t)
    C = get_mat(kf.C, x, u, p, t)
    D = get_mat(kf.D, x, u, p, t)
    (; A, B, C, D)
end

ControlSystemsBase.state_names(f::AbstractKalmanFilter)  = f.names.x
ControlSystemsBase.input_names(f::AbstractKalmanFilter)  = f.names.u
ControlSystemsBase.output_names(f::AbstractKalmanFilter) = f.names.y
ControlSystemsBase.system_name(f::AbstractKalmanFilter)  = f.names.name

"""
    ControlSystemsBase.observability(f::AbstractKalmanFilter, x, u, p, t=0.0)

Linearize (if needed) the filter and call `ControlSystemsBase.observability` on the resulting system.

# Arguments:
- `x`: The state in which to linearize the system
- `u`: The input in which to linearize the system
- `p`: The parameter vector
- `t`: The time at which to linearize the system
"""
ControlSystemsBase.observability(f::AbstractFilter, x, u, p, t=0.0) = observability(ss(linearize(f, x, u, p, t)..., f.Ts))


end