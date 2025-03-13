struct RBParticle{nxn,nxl,T,NT,LT,PT} <: AbstractVector{T}
    xn::NT  # Nonlinear state
    xl::LT  # Linear state mean
    R::PT   # Linear state covariance
end

"""
    RBParticle(xn, xl, R) <: AbstractVector

A struct that represents the state of a Rao-Blackwellized particle filter. The struct is an abstract vector, and when indexed like a vector it behaves as `[xn; xl]`. To access nonlinear or linear substate individually, axxess the fields `xn` and `xl`.

# Arguments:
- `xn`: The nonlinear state vector
- `xl`: The linear state vector
- `R`: The covariance matrix for the linear state
"""
function RBParticle(xn, xl, R) 
    T = promote_type(eltype(xn), eltype(xl), eltype(R))
    NT = typeof(xn)
    LT = typeof(xl)
    RBParticle{length(xn), length(xl), T, NT, LT, typeof(R)}(xn, xl, R)
end

function Base.getindex(x::RBParticle{nxn,nxl}, i::Int) where {nxn,nxl}
    if i > nxn
        return x.xl[i-nxn]
    else
        return x.xn[i]
    end
end

Base.size(::RBParticle{nxl, nxn}) where {nxl, nxn} = (nxl + nxn,)
Base.size(::RBParticle{nxl, nxn}, i) where {nxl, nxn} = i == 1 ? nxl + nxn : 1
Base.length(::RBParticle{nxl, nxn}) where {nxl, nxn} = nxl + nxn

"""
    RBMeasurementModel{IPM}(measurement, R2, ny)

A measurement model for the Rao-Blackwellized particle filter.

# Fields:
- `measurement`: The contribution from the nonlinar state to the output, ``g`` in ``y = g(x^n, u, p, t) + C x^l + e``
- `R2`: The probability distribution of the measurement noise. If `C == 0`, this may be any distribution, otherwise it must be an instance of `MvNormal` or `SimpleMvNormal`.
- `ny`: The number of outputs
"""
struct RBMeasurementModel{IPM,MT,R2T} <: AbstractMeasurementModel
    measurement::MT
    R2::R2T                 # Measurement noise distribution
    ny::Int
end

function RBMeasurementModel{IPM}(measurement, R2, ny) where IPM
    d_R2 = to_mv_normal(R2)
    RBMeasurementModel{IPM,typeof(measurement),typeof(d_R2)}(measurement, d_R2, ny)
end

isinplace(::RBMeasurementModel{IPM}) where IPM = IPM
has_oop(::RBMeasurementModel{IPM}) where IPM = !IPM


struct RBPF{IPD, IPM, ST, KFT <: AbstractKalmanFilter, FT, MT, ANT, R1NT, D0NT, TS, P, RNG} <: AbstractParticleFilter # QUESTION: AbstractKalmanFilter?
    state::ST
    kf::KFT
    dynamics::FT          # Nonlinear dynamics
    nl_measurement_model::MT # Nonlinear Measurement function
    An::ANT               # A_n matrix (nonlinear state)
    R1n::R1NT             # Nonlinear state noise distribution
    d0n::D0NT
    Ts::TS
    p::P
    rng::RNG
    resample_threshold::Float64
    names::SignalNames
end

to_mv_normal(d::AbstractMatrix) = SimpleMvNormal(d)
to_mv_normal(d) = d

"""
    RBPF{IPD,IPM}(N::Int, kf, dynamics, nl_measurement_model::AbstractMeasurementModel, R1n, d0n; An, nu::Int, Ts=1.0, p=NullParameters(), names, rng = Xoshiro(), resample_threshold=0.1)

Rao-Blackwellized particle filter, also called "Marginalized particle filter".

The filter assumes that the dynamics follow "model 2" in the reference below, i.e., the dynamics is described by
```math
 \\begin{align}
     x_{t+1}^n &= f_n(x_t^n, u, p, t) + A_n(x_t^n, u, p, t) x_t^l + w_t^n, \\quad w_t^n \\sim \\mathcal{N}(0, R_1^n) \\\\
     x_{t+1}^l &= A x_t^l + Bu w_t^l, \\quad w_t^l \\sim \\mathcal{N}(0, R_1^l) \\\\
     y_t &= g(x_t^n, u, p, t) + C x_t^l + e_t, \\quad e_t \\sim \\mathcal{N}(0, R_2)
 \\end{align}
```
where ``x^n`` is a subset of the state that has nonlinear dynamics, and ``x^l`` is the linear part of the state. The entire state vector is reprsented by a special type [`RBParticle`](@ref) that behaves like the vector `[xn; xl]`, but stores `xn, xl` and the covariance `R` or `xl` separately. The filter is effectively a particle filter where each particle is a Kalman filter that is responsible for the estimation of the linear sub structure.

- `N`: Number of particles
- `kf`: The internal Kalman filter that will be used for the linear part. This encodes the dynamics of the linear subspace. The matrices ``A, B, C, D, R_1^l`` of the Kalman filter may be functions of `x, u, p, t` that return a matrix as usual.
- `dynamics`: The nonlinear part of the dynamics of the nonlinear substate `f(xn, u, p, t)`
- `nl_measurement_model`: An instance of [`RBMeasurementModel`](@ref)
- `R1n`: The noise distribution of the nonlinear state dynamics, this may be a covariance matrix or a distribution. If `An = nothing`, this may be any distribution, otherwise it must be an instance of `MvNormal` or `SimpleMvNormal`.
- `d0n`: The initial distribution of the nonlinear state
- `An`: The matrix that describes the linear effect on the nonlinear state, i.e., `An*xl`. This may be a matrix or a function of `x^n, u, p, t` that returns a matrix. Pass `An = nothing` if there is no linear effect on the nonlinear state.
- `nu`: The number of control inputs
- `Ts`: The sampling time
- `p`: Parameters
- `names`: Signal names, an instance of [`SignalNames`](@ref)
- `rng`: Random number generator
- `resample_threshold`: The threshold for resampling


Based on the article ["Marginalized Particle Filters for Mixed Linear/Nonlinear State-space Models" by Thomas Schön, Fredrik Gustafsson, and Per-Johan Nordlund](https://people.isy.liu.se/rt/schon/Publications/SchonGN2004.pdf)

## Extended help
The paper contains an even more general model, where the linear part is linearly affected by the nonlinear state. It further details a number of special cases in which possible simplifications arise. 

- If `C == 0` and `D == 0`, the measurement is not used by the Kalman filter and we may thus have an arbitrary probability distribution for the measurement noise.
- If `An == 0`, the nonlinear state is not affected by the linear state and we may have an arbitrary probability distribution for the nonlinear state noise `R1n`. Otherwise `R1n` must be Gaussian.

"""
function RBPF{IPD,IPM}(N::Int, kf, dynamics, nl_measurement_model::AbstractMeasurementModel, R1n, d0n; An, nu::Int, Ts=1.0, p=NullParameters(), names, rng = Xoshiro(), resample_threshold=0.1) where {IPD, IPM}

    nxn = length(d0n)
    nxl = length(kf.d0.μ)
    T = promote_type(eltype(d0n), eltype(kf.d0))
    xn = SVector(zeros(T, nxn)...)
    xl = kf.x
    RT = typeof(kf.R)
    NT = typeof(xn)
    LT = typeof(xl)
    x = [RBParticle{nxn, nxl, T, NT, LT, RT}(NT(rand(rng, d0n)), copy(xl), copy(kf.R)) for i = 1:N]
    xprev = deepcopy(x)
    w = fill(log(1/N), N)
    we = fill(1/N, N)
    s = PFstate(x,xprev,w,we,Ref(0.), collect(1:N), zeros(N),Ref(0))

    d_R1n = to_mv_normal(R1n)


    RBPF{IPD,IPM,typeof(s),typeof(kf),typeof(dynamics),typeof(nl_measurement_model),typeof(An),typeof(d_R1n),typeof(d0n),typeof(Ts),typeof(p), typeof(rng)}(s, kf, dynamics, nl_measurement_model, An, d_R1n, d0n, Ts, p, rng, resample_threshold, names)
end

function reset!(pf::RBPF)
    s = state(pf)
    PT = eltype(s.x)
    for i = eachindex(s.xprev)
        xn = rand(pf.rng, pf.d0n)
        xl = copy(pf.kf.d0.μ)
        R = copy(pf.kf.d0.Σ)
        x = PT(xn, xl, R)
        s.x[i] = x
        s.xprev[i] = x
    end
    fill!(s.w, -log(num_particles(s)))
    fill!(s.we, 1/num_particles(s))
    s.t[] = 1
end


function predict!(pf::RBPF, u, p = parameters(pf), t = index(pf)*pf.Ts)
    s = pf.state
    N = num_particles(s)
    f = dynamics(pf)
    if shouldresample(pf)
        j = resample(pf)
        reset_weights!(s)
        # s.x .= s.x[j]
    else # Resample not needed
        s.j .= 1:N
        j = s.j
    end

    zeroAn = pf.An === nothing || iszero(get_mat(pf.An, s.x[1].xn, u, p, t))
    
    for i = 1:N
        xi = s.xprev[j[i]]
        # xi = s.x[i]
        Al = get_mat(pf.kf.A, xi, u, p, t)
        Bl = get_mat(pf.kf.B, xi, u, p, t)
        R1l = get_mat(pf.kf.R1, xi, u, pf.kf.p, t)
        R = xi.R


        # Propagate particles
        fi = f(xi.xn, u, p, t)
        
        if zeroAn
            xl1 = Al*xi.xl + Bl*u
            R1 = Al*R*Al' + R1l
            xn1 = fi + rand(pf.rng, pf.R1n) # This assumes additive noise, if An = 0, we could pass w into f instead
        else
            An = get_mat(pf.An, xi.xn, u, p, t)
            Axl = An*xi.xl
            z = Axl + rand(pf.rng, pf.R1n) 
            xn1 = fi + z
            Nt = An*R*An' + pf.R1n.Σ # Nonlinear state noise used in linear update if An != 0, must then be Gaussian
            L = Al*R*An' / Nt
            
            # NOTE: this is not general, it requires kf to be a fully linear KF to have an A matrix
            xl1 = Al*xi.xl + Bl*u + L*(z - Axl)
            R1 = Al*R*Al' + R1l - L*Nt*L'
        end

        s.x[i] = RBParticle(xn1, xl1, R1)
    end

    copyto!(s.xprev, s.x)
    pf.state.t[] += 1
    nothing
end


function correct!(pf::RBPF, u, y, p = parameters(pf), t = index(pf)*pf.Ts, args...; kwargs...)

    g = pf.nl_measurement_model.measurement
    w = pf.state.w
    s = state(pf)
    x = particles(pf)
    kf = pf.kf

    C = get_mat(kf.C, x[1], u, p, t)
    zeroC = C === nothing || iszero(C)

    # PF correct
    for i = 1:num_particles(pf)
        yn = g(x[i].xn, u, p, t)
        yl = measurement(kf)(x[i].xl, u, p, t)
        yh = yn + yl

        # KF correct adjusted with yn
        # We mutate the inner KF state to use the current particle
        if !zeroC
            kf.x = x[i].xl # Not thread safe
            kf.R = x[i].R
            (; ll) = correct!(kf, u, y-yn, p, t; kwargs...)
            # we should ll the matrix S = CRC' + R2 (eq 13a) which is exactly what the KF does
            w[i] += ll
        else
            # In this case CRC' = 0 and we have only R2 left
            d = to_mv_normal(pf.nl_measurement_model.R2)
            w[i] += extended_logpdf(d, y-yh)
        end

        x[i] = RBParticle(x[i].xn, kf.x, kf.R)
    end

    copyto!(s.xprev, s.x)
    logsumexp!(state(pf)), 0
end






@forward RBPF.state num_particles, weights, particles, particletype

function measurement(pf::RBPF)
    h = measurement(pf.nl_measurement_model)
    function (x,u,p,t) 
        h(x.xn, u, p, t) + measurement(pf.kf)(x.xl, u, p, t)
    end
end
# @inline measurement_likelihood(pf::RBPF) = measurement_likelihood(pf.nl_measurement_model)
function dynamics_density(pf::RBPF)
    dn = pf.R1n
    dl = SimpleMvNormal(pf.kf.R1)
    SimpleMvNormal([dn.μ; dl.μ], cat(dn.Σ, dl.Σ, dims=(1,2)))
end
@inline measurement_density(pf::RBPF) = pf.nl_measurement_model.R2
@inline initial_density(pf::RBPF) = error("Not yet supported. The initial density of the RBPF is complicated, it's a combination of the initial density of nl and the initial density of the linear part, including its covariance matrix.")
@inline resampling_strategy(pf::RBPF) = ResampleSystematic

