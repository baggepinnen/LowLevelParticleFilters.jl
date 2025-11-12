module LowLevelParticleFiltersControlSystemsBaseExt
import LowLevelParticleFilters.KalmanFilter
using LowLevelParticleFilters: AbstractFilter, AbstractKalmanFilter, AbstractExtendedKalmanFilter, AbstractUnscentedKalmanFilter, AbstractParticleFilter, UnscentedKalmanFilter, SimpleMvNormal, SignalNames, get_mat
using ControlSystemsBase: AbstractStateSpace, ssdata, ss, observability, controllability, linearize, obsv, kalman, covar, innovation_form
import ControlSystemsBase
using LinearAlgebra: I

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
function ControlSystemsBase.linearize(kf::Union{AbstractParticleFilter,AbstractExtendedKalmanFilter, AbstractUnscentedKalmanFilter}, x::AbstractVector, u::AbstractVector, p=kf.p, t=0.0, args...)
    A,B = linearize(kf.dynamics, x, u, p, t, args...)
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

"""
    ControlSystemsBase.controllability(f::UnscentedKalmanFilter, x, u, p, t=0.0)

Analyze the controllability _from the noise input_ (assuming `R1 = I`). Note: This does not analyze the controllability from the control input `u`, which is a more common analysis to perform. Controllability from the noise input is related to the filter's ability to handle model error and disturbances.

# Arguments:
- `x`: The state in which to linearize the system
- `u`: The input in which to linearize the system
- `p`: The parameter vector
- `t`: The time at which to linearize the system
"""
function ControlSystemsBase.controllability(f::UnscentedKalmanFilter{<:Any, <:Any, true}, x, u, p=f.p, t=0.0)
    A,B = linearize_noise_input(f, x, u, p, t)
    ControlSystemsBase.controllability(ss(A, B, I, 0, f.Ts))
end

function linearize_noise_input(f, x, u, p=f.p, t=0)
    linearize((x,w,p,t)->f.dynamics(x, u, p, t, w), x, zeros(f.nw), p, t)
end

"""
    ControlSystemsBase.obsv(f::AbstractKalmanFilter, x, u, p, t=0.0)

Linearize (if needed) the filter and call `ControlSystemsBase.obsv` on the resulting system.
# Arguments:
- `x`: The state in which to linearize the system
- `u`: The input in which to linearize the system
- `p`: The parameter vector
- `t`: The time at which to linearize the system
"""
ControlSystemsBase.obsv(f::AbstractFilter, x, u, p, t=0.0) = obsv(ss(linearize(f, x, u, p, t)..., f.Ts))


ControlSystemsBase.kalman(f::AbstractFilter, x, u, p, t=0.0) = kalman(ss(linearize(f, x, u, p, t)..., f.Ts), get_mat(f.R1, x, u, p, t), get_mat(f.R2, x, u, p, t), direct=true)

function ControlSystemsBase.kalman(f::UnscentedKalmanFilter{<:Any, <:Any, true}, x, u, p, t=0.0)
    A,Bw = linearize_noise_input(f, x, u, p, t)
    kalman(ss(linearize(f, x, u, p, t, zeros(f.nw))..., f.Ts), Bw*get_mat(f.R1, x, u, p, t)*Bw', get_mat(f.R2, x, u, p, t), direct=true)
end


function ControlSystemsBase.covar(f::AbstractFilter, x, u, p, t=0.0)
    R1 = get_mat(f.R1, x, u, p, t)
    A,B,C,D = linearize(f, x, u, p, t)
    covar(ss(A,I(f.nx),I,0,f.Ts), R1)
end

function ControlSystemsBase.covar(f::UnscentedKalmanFilter{<:Any, <:Any, true}, x, u, p, t=0.0)
    A,Bw = linearize_noise_input(f, x, u, p, t)
    R1 = Bw*get_mat(f.R1, x, u, p, t)*Bw'
    A,B,C,D = linearize(f, x, u, p, t, zeros(f.nw))
    covar(ss(A,I(f.nx),I,0,f.Ts), R1)
end



# function ControlSystemsBase.innovation_form(f::AbstractFilter, x, u, p, t=0.0)
#     R1 = get_mat(f.R1, x, u, p, t)
#     R2 = get_mat(f.R2, x, u, p, t)
#     G = ss(linearize(f, x, u, p, t)..., f.Ts)
#     innovation_form(G; R1, R2)
# end

# function ControlSystemsBase.innovation_form(f::UnscentedKalmanFilter{<:Any, <:Any, true}, x, u, p, t=0.0)
#     A,Bw = linearize_noise_input(f, x, u, p, t)
#     R1 = Bw*get_mat(f.R1, x, u, p, t)*Bw'
#     R2 = get_mat(f.R2, x, u, p, t)
#     G = ss(linearize(f, x, u, p, t, zeros(f.nw))..., f.Ts)
#     (; C) = G
#     innovation_form(G; R1, R2)
# end

# """
#     Ge = bias2innovation(f::UnscentedKalmanFilter{<:Any, <:Any, true}, x, u, p, t=0.0)

# Returns two linear systems assuming
# ```
# xp = Ax + Bu + d + w
# y = Cx + Du + b + v
# ```

# Ged is the system from dynamics disturbance (I as input matrix) to innovation. The input ressolvent of this is the system from dynamics disturbance to state prediction error
# Geb is the system from measurement disturbance to innovation. The input ressolvent of this is the system from measurement disturbance to state prediction error.

# This function is an experiement in trying to identify which disturbances can cause a static error in the difference between predicted and filtered state. We output systems from disturbances to innovations, but the input resolvent of those are the systems from disturbances to state error, that is, the following DC gain matrix may be worth investigating

# ```
# n2(x) = x ./ maximum(abs, x, dims=2)
# Plots.heatmap(n2(dcgain(input_resolvent(Ge.Ged))), yflip=true)
# ```

# I am not 100% convinced of the utility of this function, if sol.e[t][i] is large, one could compute the gradient of this w.r.t., parameters to see which parameter is likely to affect this innovation. This might be easier for nonlinear systems rather than trying to linearize around a reasonable point and interpret these DC gains.
# """
# function bias2innovation(f::UnscentedKalmanFilter{<:Any, <:Any, true}, x, u, p, t=0.0)
#     A,Bw = linearize_noise_input(f, x, u, p, t)
#     R1 = Bw*get_mat(f.R1, x, u, p, t)*Bw'
#     R2 = get_mat(f.R2, x, u, p, t)
#     G = ss(linearize(f, x, u, p, t)..., f.Ts)
#     (; C) = G
#     innovation_form(G; R1, R2)
#     K = kalman(f, x, u, p, t)
#     A2 = A*(I-K*C) # The extra A comes from the fact that we are working with the one-step prediction error rather than the filter error. This A also appears in AK as the input matrix for the innovation system below
#     Geb = ss(A2, A*K, -C, I(f.ny), f.Ts)
#     Ged = ss(A2, I(f.nx), -C, 0, f.Ts)
#     (; Ged, Geb)
# end

end